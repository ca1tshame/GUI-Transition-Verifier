"""Conv3d-free patch embedding for Qwen-VL style vision towers.

Motivation
----------
The Qwen2-VL / Qwen3-VL vision tower projects already-patchified pixels with a
``nn.Conv3d`` whose kernel, stride and patch geometry are identical:

    kernel = stride = (temporal_patch_size, patch_size, patch_size)
    padding = 0, dilation = 1, groups = 1

Under those constraints every output element is the dot product of one
flattened patch with one flattened filter, so the convolution is a single
matrix multiplication. Replacing it with ``F.linear`` removes the im2col /
``conv3d`` dispatch path without changing the function being computed.

Equivalence contract
--------------------
* Algebraic equivalence holds for the **already patchified** tensor
  ``[N, C * T * P * P]``. This module does not perform image patchification.
* Under BF16 the replacement is **not bitwise identical**: reduction order in
  the GEMM differs from cuDNN's convolution kernel. Observed drift on real
  weights is small but not zero, and a small fraction of argmax decisions can
  flip. Callers must audit parity (:func:`audit_parity`) and keep the original
  path available for high-risk rows.
* Parameters are shared, not copied. ``mode="original"`` restores the exact
  upstream module at any time, so the fast path is always reversible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "FastPatchEmbed",
    "ParityBudget",
    "ParityReport",
    "apply_fastpe",
    "audit_parity",
    "patch_embed_geometry",
]

_REQUIRED_ATTRS = (
    "in_channels",
    "temporal_patch_size",
    "patch_size",
    "embed_dim",
    "proj",
)

_MODES = ("gemm", "original")


def patch_embed_geometry(module: nn.Module) -> dict[str, int]:
    """Return the validated geometry of an upstream patch-embed module.

    Raises ``TypeError`` when required attributes are missing and ``ValueError``
    when the convolution cannot be expressed as a per-patch GEMM.
    """
    for name in _REQUIRED_ATTRS:
        if not hasattr(module, name):
            raise TypeError(f"unsupported patch embedding: missing attribute {name!r}")

    in_channels = int(module.in_channels)
    temporal = int(module.temporal_patch_size)
    patch = int(module.patch_size)
    embed_dim = int(module.embed_dim)
    proj = module.proj

    kernel = (temporal, patch, patch)
    expected_weight = (embed_dim, in_channels, *kernel)

    if not isinstance(proj, nn.Conv3d):
        raise ValueError("patch embedding projection must be nn.Conv3d")
    if tuple(proj.weight.shape) != expected_weight:
        raise ValueError(
            f"unexpected Conv3d weight shape {tuple(proj.weight.shape)}, "
            f"expected {expected_weight}"
        )
    if tuple(proj.kernel_size) != kernel:
        raise ValueError("kernel size must equal the patch geometry")
    if tuple(proj.stride) != kernel:
        raise ValueError("stride must equal the patch geometry (non-overlapping patches)")
    if tuple(proj.padding) != (0, 0, 0):
        raise ValueError("padding must be zero")
    if tuple(proj.dilation) != (1, 1, 1):
        raise ValueError("dilation must be one")
    if proj.groups != 1:
        raise ValueError("grouped convolution is not supported")

    return {
        "in_channels": in_channels,
        "temporal_patch_size": temporal,
        "patch_size": patch,
        "embed_dim": embed_dim,
        "flat_dim": in_channels * temporal * patch * patch,
    }


class FastPatchEmbed(nn.Module):
    """Drop-in wrapper that can run patch embedding as a GEMM.

    Args:
        original: upstream patch-embed module (parameters are shared).
        mode: ``"gemm"`` for the fast path, ``"original"`` for the exact
            upstream path. The mode is mutable through :meth:`set_mode` so a
            service can fall back per request without rebuilding the model.
    """

    def __init__(self, original: nn.Module, mode: str = "gemm") -> None:
        super().__init__()
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        geometry = patch_embed_geometry(original)
        self.original = original
        self.mode = mode
        self.in_channels = geometry["in_channels"]
        self.temporal_patch_size = geometry["temporal_patch_size"]
        self.patch_size = geometry["patch_size"]
        self.embed_dim = geometry["embed_dim"]
        self.flat_dim = geometry["flat_dim"]

    # -- runtime control ---------------------------------------------------
    def set_mode(self, mode: str) -> None:
        """Switch between the fast and the reference implementation."""
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        self.mode = mode

    @property
    def is_fast(self) -> bool:
        return self.mode == "gemm"

    # -- computation -------------------------------------------------------
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.mode == "original":
            return self.original(hidden_states)

        if hidden_states.shape[-1] != self.flat_dim:
            raise ValueError(
                f"expected patchified input with last dimension {self.flat_dim}, "
                f"got shape {tuple(hidden_states.shape)}"
            )

        proj = self.original.proj
        weight = proj.weight
        bias = proj.bias

        # ``view`` mirrors upstream semantics. ``reshape`` would silently copy
        # non-contiguous inputs and hide a real layout bug.
        patches = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        flat = patches.to(dtype=weight.dtype).view(-1, weight[0].numel())
        return F.linear(flat, weight.view(self.embed_dim, -1), bias)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, temporal_patch_size={self.temporal_patch_size}, "
            f"patch_size={self.patch_size}, embed_dim={self.embed_dim}, mode={self.mode}"
        )


def _patch_embed_candidates(model: nn.Module) -> Iterable[tuple[str, nn.Module]]:
    """Yield ``(name, module)`` for unwrapped patch-embed modules.

    Descendants of an existing :class:`FastPatchEmbed` are skipped by prefix,
    not only by type: the wrapper keeps the original module as a child, and
    wrapping that child again would nest two fast paths.
    """
    wrapped_prefixes: list[str] = []
    for name, module in model.named_modules():
        if any(
            prefix == "" or name == prefix or name.startswith(prefix + ".")
            for prefix in wrapped_prefixes
        ):
            continue
        if isinstance(module, FastPatchEmbed):
            wrapped_prefixes.append(name)
            continue
        if module.__class__.__name__.endswith("PatchEmbed") and hasattr(module, "proj"):
            yield name, module


def apply_fastpe(model: nn.Module, mode: str = "gemm", strict: bool = True) -> int:
    """Replace every patch-embed module in ``model`` with :class:`FastPatchEmbed`.

    Args:
        model: vision tower or full multimodal model.
        mode: initial mode for the replacements.
        strict: when ``True`` a candidate that fails geometry validation raises;
            when ``False`` it is left untouched and counted as skipped.

    Returns:
        Number of modules replaced. Idempotent: already-wrapped modules are
        skipped, so calling twice is safe.
    """
    replaced = 0
    for name, module in list(_patch_embed_candidates(model)):
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        try:
            wrapper = FastPatchEmbed(module, mode=mode)
        except (TypeError, ValueError):
            if strict:
                raise
            continue
        setattr(parent, attr, wrapper)
        replaced += 1
    return replaced


# ---------------------------------------------------------------------------
# Parity auditing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParityBudget:
    """Acceptance thresholds for a fast-path rollout.

    Defaults are deliberately conservative and describe what the project
    observed on BF16 weights, not a universal guarantee.

    Attributes:
        max_abs: maximum tolerated absolute logit difference.
        max_rel_l2: maximum tolerated relative L2 difference of the logits.
        max_flip_rate: maximum tolerated argmax flip rate over the audited set.
        max_risk_flip_rate: maximum tolerated flip rate among rows whose
            reference risk score is at or above ``risk_threshold``. Zero by
            default: a fast path must never silently drop a high-risk alert.
    """

    max_abs: float = 2e-2
    max_rel_l2: float = 5e-3
    max_flip_rate: float = 2e-2
    max_risk_flip_rate: float = 0.0


@dataclass
class ParityReport:
    """Numerical and decision parity between a reference and a fast path."""

    n: int
    max_abs: float
    mean_abs: float
    rel_l2: float
    flips: int
    flip_rate: float
    risk_flips: int
    risk_flip_rate: float
    budget: ParityBudget
    reasons: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        """``accept`` when every budget is met, else ``review_full_precision``."""
        return "review_full_precision" if self.reasons else "accept"

    @property
    def passed(self) -> bool:
        return not self.reasons

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "max_abs": self.max_abs,
            "mean_abs": self.mean_abs,
            "rel_l2": self.rel_l2,
            "flips": self.flips,
            "flip_rate": self.flip_rate,
            "risk_flips": self.risk_flips,
            "risk_flip_rate": self.risk_flip_rate,
            "verdict": self.verdict,
            "reasons": list(self.reasons),
        }


def audit_parity(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    *,
    reference_risk: Sequence[float] | None = None,
    risk_threshold: float | None = None,
    budget: ParityBudget = ParityBudget(),
) -> ParityReport:
    """Compare a fast path against the reference path row by row.

    Args:
        reference_logits: ``[N, K]`` logits from the original implementation.
        candidate_logits: ``[N, K]`` logits from the fast implementation.
        reference_risk: optional per-row risk score (for example the hijack
            probability). When provided together with ``risk_threshold``, flips
            inside the high-risk band are counted separately and must stay at
            or below ``budget.max_risk_flip_rate``.
        risk_threshold: operating threshold that defines the high-risk band.
        budget: acceptance thresholds.

    Returns:
        A :class:`ParityReport`; ``report.verdict`` is the rollout decision.
    """
    ref = torch.as_tensor(reference_logits)
    cand = torch.as_tensor(candidate_logits)
    if ref.shape != cand.shape:
        raise ValueError(f"logit shape mismatch: {tuple(ref.shape)} vs {tuple(cand.shape)}")
    if ref.ndim != 2:
        raise ValueError("expected [N, K] logits")

    ref_f = ref.detach().float().cpu()
    cand_f = cand.detach().float().cpu()
    diff = (cand_f - ref_f).abs()

    n = int(ref_f.shape[0])
    max_abs = float(diff.max()) if n else 0.0
    mean_abs = float(diff.mean()) if n else 0.0
    denom = float(torch.linalg.vector_norm(ref_f)) or 1.0
    rel_l2 = float(torch.linalg.vector_norm(diff) / denom)

    ref_arg = ref_f.argmax(dim=1)
    cand_arg = cand_f.argmax(dim=1)
    flip_mask = ref_arg != cand_arg
    flips = int(flip_mask.sum())
    flip_rate = flips / n if n else 0.0

    risk_flips = 0
    risk_flip_rate = 0.0
    if reference_risk is not None and risk_threshold is not None:
        if len(reference_risk) != n:
            raise ValueError("reference_risk length must match the number of rows")
        high_risk = torch.tensor([float(v) >= risk_threshold for v in reference_risk])
        risk_flips = int((flip_mask & high_risk).sum())
        n_high = int(high_risk.sum())
        risk_flip_rate = risk_flips / n_high if n_high else 0.0

    reasons: list[str] = []
    if max_abs > budget.max_abs:
        reasons.append(f"max_abs {max_abs:.6g} exceeds {budget.max_abs:g}")
    if rel_l2 > budget.max_rel_l2:
        reasons.append(f"rel_l2 {rel_l2:.6g} exceeds {budget.max_rel_l2:g}")
    if flip_rate > budget.max_flip_rate:
        reasons.append(f"flip_rate {flip_rate:.4%} exceeds {budget.max_flip_rate:.2%}")
    if risk_flip_rate > budget.max_risk_flip_rate:
        reasons.append(
            f"risk_flip_rate {risk_flip_rate:.4%} exceeds {budget.max_risk_flip_rate:.2%}"
        )

    return ParityReport(
        n=n,
        max_abs=max_abs,
        mean_abs=mean_abs,
        rel_l2=rel_l2,
        flips=flips,
        flip_rate=flip_rate,
        risk_flips=risk_flips,
        risk_flip_rate=risk_flip_rate,
        budget=budget,
        reasons=reasons,
    )


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile; ``q`` in ``[0, 1]``."""
    if not values:
        raise ValueError("empty sample")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must lie in [0, 1]")
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac
