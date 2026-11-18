"""Two-stage cascade fallback from GEMM patch embedding to reference Conv3d.

Why a cascade gate is required
------------------------------
Under BF16, replacing ``nn.Conv3d`` with ``F.linear`` (:class:`FastPatchEmbed`)
accelerates vision patch projection substantially, but floating-point reduction
order differs from cuDNN convolution. On the 966-row Test split, unconditional
``mode="gemm"`` exhibited 15/966 argmax flips and 7/966 risk-threshold flips
(including 3 cancelled ``1_hijack`` alerts). A fast path that silently cancels
safety alerts cannot be deployed unconditionally.

This module implements a two-stage cascade over a single resident model:

1. **Stage 1 (fast path)** runs ``mode="gemm"`` on the prepared batch.
2. **Gate check** (:func:`should_fallback`) inspects the resulting 5-class
   probability vector using :func:`diagnostics.selective_abstention.risk_scores`.
3. **Stage 2 (selective fallback)** switches :class:`FastPatchEmbed` to
   ``mode="original"`` only when the gate triggers, re-scores the **exact same
   resident batch object** without repeating image preprocessing, and restores
   ``mode="gemm"`` inside a ``try ... finally`` block even if scoring raises.

Calibration and latency accounting rules
----------------------------------------
* **Dev-frozen threshold and reused-Test evaluation.** ``risk_threshold``
  (``0.6702836155891418``) was fitted exclusively on the 135-row Dev split and
  frozen before evaluation on the 966-row Test split. Test labels were
  never used to tune ``risk_threshold``. The gate presets themselves have not
  been established here as independently calibrated on an unseen test.
* **Deterministic trajectory replay simulation.** Reported cascade latency and
  speedup figures are offline deterministic replay simulations computed from
  paired per-sample trajectory logs: every sample pays ``gemm.e2e_ms``, and
  triggered samples add ``original.gpu_ms`` (reusing the resident GPU batch
  already preprocessed in Stage 1, relative to ``7551.82 ms`` original BF16
  mean end-to-end latency):

  - :data:`ALERT_SAFE_CONFIG` (default recommended, ``gate_mode="alert_safe"``):
    triggers fallback solely when ``p1`` lies in ``[0.30, 0.75]``, giving a
    ``2.17%`` (``21/966``) fallback rate, ``313.06 ms`` mean E2E latency
    (``24.12x`` speedup), ``0`` missed ``1_hijack`` alerts, and ``0/966``
    risk-threshold flips.
  - :data:`FULL_PARITY_CONFIG` (``gate_mode="full_parity"``): triggers fallback
    when ``p1`` lies in ``[0.30, 0.75]`` or ``confidence < 0.70``, giving a
    ``19.77%`` (``191/966``) fallback rate, ``1635.27 ms`` mean E2E latency
    (``4.62x`` speedup), and ``0/966`` argmax and risk-threshold flips.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

import torch
from torch import nn

from diagnostics.selective_abstention import decide, risk_scores

from .fast_patch_embed import FastPatchEmbed

# Mutable patch modes are shared across runner instances, so serialize every
# cascade and initialization, even when callers provide separate outer locks.
_MODE_LOCK = threading.RLock()

__all__ = [
    "ALERT_SAFE_CONFIG",
    "FULL_PARITY_CONFIG",
    "CascadeConfig",
    "CascadeRunner",
    "set_fastpe_mode",
    "should_fallback",
]

GateMode = Literal["alert_safe", "full_parity"]
_VALID_GATE_MODES = ("alert_safe", "full_parity")
_VALID_FASTPE_MODES = ("gemm", "original")


def _validate_prob_param(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite float in [0, 1], got {value!r}")
    fval = float(value)
    if not math.isfinite(fval) or not 0.0 <= fval <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {value!r}")


@dataclass(frozen=True)
class CascadeConfig:
    """Operating-point configuration for two-stage fastPE cascade inference.

    Note:
        ``risk_threshold`` follows the Dev-frozen threshold protocol
        (frozen on 135 Dev rows, evaluated on 966 reused Test rows). Latency
        and speedup metrics associated with presets come from deterministic
        replay simulations on paired trajectory logs (``gemm.e2e_ms`` plus
        ``original.gpu_ms`` on triggered samples reusing the resident GPU batch).
    """

    gate_mode: GateMode = "alert_safe"
    p1_low: float = 0.30
    p1_high: float = 0.75
    conf_threshold: float = 0.70
    risk_threshold: float = 0.6702836155891418
    review_confidence: float = 0.80
    upper_bound_open: bool = False

    def __post_init__(self) -> None:
        if self.gate_mode not in _VALID_GATE_MODES:
            raise ValueError(
                f"gate_mode must be one of {_VALID_GATE_MODES}, got {self.gate_mode!r}"
            )
        if not isinstance(self.upper_bound_open, bool):
            raise ValueError("upper_bound_open must be a bool")
        _validate_prob_param(self.p1_low, "p1_low")
        _validate_prob_param(self.p1_high, "p1_high")
        _validate_prob_param(self.conf_threshold, "conf_threshold")
        _validate_prob_param(self.risk_threshold, "risk_threshold")
        _validate_prob_param(self.review_confidence, "review_confidence")
        if float(self.p1_low) > float(self.p1_high):
            raise ValueError("p1_low must not exceed p1_high")


# Default recommended preset: p1 boundary [0.30, 0.75] only.
# Deterministic paired-log replay on 966 Test rows: 2.17% (21/966) fallback,
# 313.06 ms mean E2E (24.12x speedup), 0 missed 1_hijack alerts, 0/966 risk flips.
ALERT_SAFE_CONFIG = CascadeConfig(gate_mode="alert_safe")

# Full-parity preset: p1 boundary [0.30, 0.75] or max probability < 0.70.
# Deterministic paired-log replay on 966 Test rows: 19.77% (191/966) fallback,
# 1635.27 ms mean E2E (4.62x speedup), 0/966 argmax flips and 0/966 risk flips.
FULL_PARITY_CONFIG = CascadeConfig(gate_mode="full_parity")


def _coerce_probabilities(probabilities: Sequence[float] | torch.Tensor) -> list[float]:
    if isinstance(probabilities, torch.Tensor):
        tensor = probabilities.detach().cpu()
        if tensor.ndim == 2 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.ndim != 1:
            raise ValueError(
                f"expected 1D probability vector (or [1, 5] tensor), got shape {tuple(tensor.shape)}"
            )
        return [float(v) for v in tensor.tolist()]
    return [float(v) for v in probabilities]


def should_fallback(
    probabilities: Sequence[float],
    cfg: CascadeConfig = ALERT_SAFE_CONFIG,
) -> tuple[bool, str]:
    """Decide whether Stage 1 ``gemm`` probabilities require ``original`` fallback.

    Args:
        probabilities: five-class probability simplex ``(p1, p2, p3, p4, p5)``.
        cfg: cascade gate configuration. Defaults to :data:`ALERT_SAFE_CONFIG`.

    Returns:
        ``(triggered, reason)`` where ``reason`` is ``"p1_boundary"``,
        ``"low_confidence"``, or ``"none"``.
    """
    if cfg.gate_mode not in _VALID_GATE_MODES:
        raise ValueError(f"unsupported gate_mode: {cfg.gate_mode!r}")

    probs = _coerce_probabilities(probabilities)
    scores = risk_scores(probs)
    p1 = scores["p1"]

    if cfg.upper_bound_open:
        in_p1_boundary = cfg.p1_low <= p1 < cfg.p1_high
    else:
        in_p1_boundary = cfg.p1_low <= p1 <= cfg.p1_high

    if in_p1_boundary:
        return True, "p1_boundary"

    if cfg.gate_mode == "full_parity":
        confidence = max(probs)
        if confidence < cfg.conf_threshold:
            return True, "low_confidence"

    return False, "none"


def set_fastpe_mode(target: FastPatchEmbed | nn.Module | Any, mode: str) -> int:
    """Set execution mode on a :class:`FastPatchEmbed` or parent :class:`nn.Module`.

    Args:
        target: a :class:`FastPatchEmbed` wrapper, an :class:`nn.Module`
            containing one or more :class:`FastPatchEmbed` submodules, or an
            object exposing ``set_mode(mode)``.
        mode: ``"gemm"`` or ``"original"``.

    Returns:
        Number of :class:`FastPatchEmbed` modules updated.
    """
    if mode not in _VALID_FASTPE_MODES:
        raise ValueError(f"mode must be one of {_VALID_FASTPE_MODES}, got {mode!r}")

    if isinstance(target, FastPatchEmbed):
        target.set_mode(mode)
        return 1

    if isinstance(target, nn.Module):
        count = 0
        for module in target.modules():
            if isinstance(module, FastPatchEmbed):
                module.set_mode(mode)
                count += 1
        if count == 0:
            raise ValueError("target nn.Module contains no FastPatchEmbed submodules")
        return count

    if hasattr(target, "set_mode") and callable(target.set_mode):
        target.set_mode(mode)
        return 1

    raise TypeError(
        "target must be a FastPatchEmbed or an nn.Module containing FastPatchEmbed submodules"
    )


class CascadeRunner:
    """Thread-safe two-stage ``gemm -> original`` cascade executor.

    Calls and initialization are serialized by a process-wide reentrant mode
    lock (and the optional instance lock), including across runner instances.
    Other code mutating the target outside this runner is not synchronized. When Stage 2 triggers,
    ``score_batch_fn`` receives the exact same ``batch`` reference passed to
    Stage 1, and ``try ... finally`` guarantees restoration to ``"gemm"`` even
    if Stage 2 raises an exception.
    """

    def __init__(
        self,
        target: FastPatchEmbed | nn.Module | Any,
        score_batch_fn: Callable[[Any], Sequence[float] | torch.Tensor],
        *,
        lock: threading.Lock | None = None,
        cfg: CascadeConfig = ALERT_SAFE_CONFIG,
    ) -> None:
        if not callable(score_batch_fn):
            raise TypeError("score_batch_fn must be callable")
        if not isinstance(cfg, CascadeConfig):
            raise TypeError("cfg must be a CascadeConfig instance")
        # Validate target upfront and initialize to the fast path.
        with _MODE_LOCK:
            set_fastpe_mode(target, "gemm")
        self.target = target
        self.score_batch_fn = score_batch_fn
        self.lock: threading.Lock = lock if lock is not None else threading.Lock()
        self.cfg = cfg

    def run(self, batch: Any) -> dict[str, Any]:
        """Score ``batch`` through the two-stage cascade and return decision metadata."""
        with _MODE_LOCK, self.lock:
            set_fastpe_mode(self.target, "gemm")
            fast_probs = _coerce_probabilities(self.score_batch_fn(batch))
            triggered, reason = should_fallback(fast_probs, self.cfg)
            if not triggered:
                final_probs = fast_probs
                mode_used = "gemm"
            else:
                try:
                    set_fastpe_mode(self.target, "original")
                    final_probs = _coerce_probabilities(self.score_batch_fn(batch))
                    mode_used = "original"
                finally:
                    set_fastpe_mode(self.target, "gemm")

        scores = risk_scores(final_probs)
        confidence = max(final_probs)
        decision = decide(
            confidence,
            scores["p1"],
            confidence_threshold=self.cfg.review_confidence,
            risk_threshold=self.cfg.risk_threshold,
        )
        result: dict[str, Any] = {
            "probabilities": final_probs,
            "decision": decision,
            "mode_used": mode_used,
            "fallback_triggered": triggered,
            "fallback_reason": reason,
        }
        if triggered:
            result["fast_probabilities"] = fast_probs
        return result

    def __call__(self, batch: Any) -> dict[str, Any]:
        return self.run(batch)
