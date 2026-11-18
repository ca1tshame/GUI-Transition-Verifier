"""Cross-term-free Exact Adapter Soup in LoRA delta space.

Motivation
----------
Averaging two LoRA adapters factor-by-factor::

    A_avg = w_a * A_1 + w_b * A_2
    B_avg = w_a * B_1 + w_b * B_2

does **not** average their weight updates: expanding ``B_avg @ A_avg``
introduces spurious cross terms ``w_a * w_b * (B_1 @ A_2 + B_2 @ A_1)`` that
neither adapter was trained to produce.

Exact Adapter Soup instead constructs a block-concatenated rank-``2r`` adapter::

    A_merged = concat([A_1, A_2], dim=0)              # [2r, in_features]
    B_merged = concat([w_a * B_1, w_b * B_2], dim=1)  # [out_features, 2r]

Doubling both rank (``r -> 2r``, e.g. ``16 -> 32``) and ``lora_alpha``
(``alpha -> 2 * alpha``, e.g. ``32 -> 64``) keeps the PEFT scaling factor
``lora_alpha / r`` invariant (``64 / 32 == 32 / 16 == 2.0``), so a single
forward pass computes the exact linear combination of LoRA deltas::

    (alpha_merged / r_merged) * (B_merged @ A_merged)
        == (alpha / r) * (w_a * (B_1 @ A_1) + w_b * (B_2 @ A_2))

with zero cross terms and single-pass inference latency.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch

__all__ = [
    "apply_lora_delta",
    "merge_lora_adapters",
    "merged_lora_config",
    "verify_adapter_soup_equivalence",
]


def _partner_lora_b_key(key: str) -> str | None:
    """Return the corresponding ``lora_B`` key for a ``lora_A`` key, else ``None``."""
    if ".lora_A." in key:
        return key.replace(".lora_A.", ".lora_B.", 1)
    if key.endswith(".lora_A"):
        return key[:-7] + ".lora_B"
    if key == "lora_A":
        return "lora_B"
    return None


def _collect_lora_pairs(state: Mapping[str, torch.Tensor]) -> list[tuple[str, str]]:
    if not state:
        raise ValueError("adapter state dictionary must not be empty")
    pairs: list[tuple[str, str]] = []
    handled: set[str] = set()
    for key in sorted(state.keys()):
        partner = _partner_lora_b_key(key)
        if partner is None:
            continue
        if partner not in state:
            raise ValueError(f"missing matching lora_B tensor {partner!r} for {key!r}")
        pairs.append((key, partner))
        handled.add(key)
        handled.add(partner)

    unhandled = sorted(set(state.keys()) - handled)
    if unhandled:
        raise ValueError(f"unhandled or non-LoRA tensor keys in adapter state: {unhandled}")
    return pairs


def _validate_weights(weights: tuple[float, float]) -> tuple[float, float]:
    if len(weights) != 2:
        raise ValueError("weights must be a 2-tuple (w_a, w_b)")
    w_a, w_b = weights
    if (
        isinstance(w_a, bool)
        or isinstance(w_b, bool)
        or not math.isfinite(w_a)
        or not math.isfinite(w_b)
    ):
        raise ValueError("adapter weights must be finite floats")
    return float(w_a), float(w_b)


def merged_lora_config(
    rank: int = 16,
    alpha: float = 32.0,
    num_adapters: int = 2,
) -> dict[str, float | int]:
    """Return the rank-doubled LoRA configuration preserving ``lora_alpha / r``.

    For ``rank=16, alpha=32.0, num_adapters=2``, returns ``r=32`` and
    ``lora_alpha=64`` with ``alpha_over_r=2.0``.
    """
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("rank must be a positive integer")
    if isinstance(num_adapters, bool) or not isinstance(num_adapters, int) or num_adapters < 1:
        raise ValueError("num_adapters must be a positive integer")
    if isinstance(alpha, bool) or not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be a positive finite float")

    merged_r = rank * num_adapters
    raw_alpha = float(alpha) * num_adapters
    merged_alpha: float | int = int(raw_alpha) if raw_alpha.is_integer() else raw_alpha
    return {
        "r": merged_r,
        "lora_alpha": merged_alpha,
        "alpha_over_r": float(alpha) / float(rank),
    }


def merge_lora_adapters(
    state_a: Mapping[str, torch.Tensor],
    state_b: Mapping[str, torch.Tensor],
    weights: tuple[float, float] = (0.5, 0.5),
) -> dict[str, torch.Tensor]:
    """Merge two compatible LoRA state dicts by exact delta concatenation.

    Args:
        state_a: first LoRA state mapping ``{name: tensor}`` where ``lora_A``
            has shape ``[r, in_features]`` and ``lora_B`` has shape
            ``[out_features, r]``.
        state_b: second LoRA state mapping with identical keys and shapes.
        weights: linear combination weights ``(w_a, w_b)``, defaulting to
            equal-weight ``(0.5, 0.5)``.

    Returns:
        New ``dict[str, torch.Tensor]`` with ``lora_A`` of shape
        ``[2r, in_features]`` and ``lora_B`` of shape ``[out_features, 2r]``.
    """
    w_a, w_b = _validate_weights(weights)
    if set(state_a.keys()) != set(state_b.keys()):
        raise ValueError("state_a and state_b must contain identical parameter keys")

    pairs = _collect_lora_pairs(state_a)
    merged: dict[str, torch.Tensor] = {}

    for key_a, key_b in pairs:
        a1, b1 = state_a[key_a], state_a[key_b]
        a2, b2 = state_b[key_a], state_b[key_b]
        if a1.ndim != 2 or b1.ndim != 2 or a2.ndim != 2 or b2.ndim != 2:
            raise ValueError(f"LoRA factors must be 2D matrices for {key_a!r}")
        if a1.shape != a2.shape or b1.shape != b2.shape:
            raise ValueError(f"shape mismatch between adapters at {key_a!r}")
        if a1.shape[0] != b1.shape[1] or a1.shape[0] == 0:
            raise ValueError(
                f"incompatible LoRA rank dimensions for {key_a!r}: "
                f"A={tuple(a1.shape)} vs B={tuple(b1.shape)}"
            )
        if not (
            torch.isfinite(a1).all()
            and torch.isfinite(b1).all()
            and torch.isfinite(a2).all()
            and torch.isfinite(b2).all()
        ):
            raise ValueError(f"non-finite LoRA tensor encountered in {key_a!r}")

        merged_a = torch.cat([a1, a2], dim=0).contiguous()
        merged_b = torch.cat([b1 * w_a, b2 * w_b], dim=1).contiguous()
        if not (torch.isfinite(merged_a).all() and torch.isfinite(merged_b).all()):
            raise FloatingPointError(f"non-finite merged LoRA tensor in {key_a!r}")

        merged[key_a] = merged_a
        merged[key_b] = merged_b

    return merged


def apply_lora_delta(
    inputs: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    *,
    alpha_over_r: float = 2.0,
) -> torch.Tensor:
    """Compute ``(alpha_over_r) * ((inputs @ lora_a.T) @ lora_b.T)``."""
    if isinstance(alpha_over_r, bool) or not math.isfinite(alpha_over_r) or alpha_over_r <= 0.0:
        raise ValueError("alpha_over_r must be a positive finite float")
    if lora_a.ndim != 2 or lora_b.ndim != 2 or lora_a.shape[0] != lora_b.shape[1]:
        raise ValueError("invalid lora_a / lora_b shapes")
    if inputs.ndim < 1 or inputs.shape[-1] != lora_a.shape[1]:
        raise ValueError("inputs last dimension must match lora_a in_features")
    compute_dtype = (
        torch.float64
        if (
            inputs.dtype == torch.float64
            or lora_a.dtype == torch.float64
            or lora_b.dtype == torch.float64
        )
        else torch.float32
    )
    x = inputs.to(dtype=compute_dtype)
    a = lora_a.to(device=x.device, dtype=compute_dtype)
    b = lora_b.to(device=x.device, dtype=compute_dtype)
    return float(alpha_over_r) * ((x @ a.T) @ b.T)


def verify_adapter_soup_equivalence(
    state_a: Mapping[str, torch.Tensor],
    state_b: Mapping[str, torch.Tensor],
    merged_state: Mapping[str, torch.Tensor] | None = None,
    weights: tuple[float, float] = (0.5, 0.5),
    alpha_over_r: float = 2.0,
    num_probes: int = 8,
    seed: int = 0,
) -> dict[str, float | int]:
    """Audit CPU forward equivalence of Exact Adapter Soup vs two separate passes.

    Compares the single-pass merged adapter output against the exact weighted
    sum of two independent LoRA deltas across ``num_probes`` unit-norm random
    probe vectors per module, and also measures ``naive_avg_max_abs_diff`` from
    factor-wise parameter averaging ``(w_a * B_1 + w_b * B_2) @ (w_a * A_1 + w_b * A_2)``
    to quantify the eliminated cross-term error.
    """
    w_a, w_b = _validate_weights(weights)
    if isinstance(alpha_over_r, bool) or not math.isfinite(alpha_over_r) or alpha_over_r <= 0.0:
        raise ValueError("alpha_over_r must be a positive finite float")
    if isinstance(num_probes, bool) or not isinstance(num_probes, int) or num_probes < 1:
        raise ValueError("num_probes must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    if merged_state is None:
        merged_state = merge_lora_adapters(state_a, state_b, weights=weights)
    else:
        if set(merged_state.keys()) != set(state_a.keys()):
            raise ValueError("merged_state keys must match state_a and state_b")

    pairs = _collect_lora_pairs(state_a)
    rng = torch.Generator(device="cpu").manual_seed(seed)

    max_abs_diff = 0.0
    max_rel_l2 = 0.0
    naive_avg_max_abs_diff = 0.0

    for key_a, key_b in pairs:
        a1 = state_a[key_a].detach().cpu().to(dtype=torch.float64)
        b1 = state_a[key_b].detach().cpu().to(dtype=torch.float64)
        a2 = state_b[key_a].detach().cpu().to(dtype=torch.float64)
        b2 = state_b[key_b].detach().cpu().to(dtype=torch.float64)
        am = merged_state[key_a].detach().cpu().to(dtype=torch.float64)
        bm = merged_state[key_b].detach().cpu().to(dtype=torch.float64)

        expected_a_shape = (a1.shape[0] + a2.shape[0], a1.shape[1])
        expected_b_shape = (b1.shape[0], b1.shape[1] + b2.shape[1])
        if am.shape != expected_a_shape or bm.shape != expected_b_shape:
            raise ValueError(f"merged tensor shape mismatch for {key_a!r}")

        raw_probes = torch.randn((num_probes, a1.shape[1]), generator=rng, dtype=torch.float64)
        norms = torch.linalg.vector_norm(raw_probes, dim=-1, keepdim=True).clamp(min=1e-12)
        probes = raw_probes / norms

        delta_1 = apply_lora_delta(probes, a1, b1, alpha_over_r=alpha_over_r)
        delta_2 = apply_lora_delta(probes, a2, b2, alpha_over_r=alpha_over_r)
        reference = w_a * delta_1 + w_b * delta_2

        merged_out = apply_lora_delta(probes, am, bm, alpha_over_r=alpha_over_r)
        diff = (merged_out - reference).abs()
        max_abs_diff = max(max_abs_diff, float(diff.max()))
        denom = float(torch.linalg.vector_norm(reference)) or 1.0
        rel_l2 = float(torch.linalg.vector_norm(diff) / denom)
        max_rel_l2 = max(max_rel_l2, rel_l2)

        a_naive = w_a * a1 + w_b * a2
        b_naive = w_a * b1 + w_b * b2
        naive_out = apply_lora_delta(probes, a_naive, b_naive, alpha_over_r=alpha_over_r)
        naive_diff = (naive_out - reference).abs()
        naive_avg_max_abs_diff = max(naive_avg_max_abs_diff, float(naive_diff.max()))

    return {
        "num_modules": len(pairs),
        "num_probes": int(num_probes),
        "max_abs_diff": max_abs_diff,
        "max_rel_l2": max_rel_l2,
        "naive_avg_max_abs_diff": naive_avg_max_abs_diff,
    }
