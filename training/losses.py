"""Loss functions for five-class GUI state-transition verification.

Overview
--------
The verifier predicts one of five transition classes over candidate single-token
class IDs ``1..5``:

* ``0`` -> ``1_hijack``
* ``1`` -> ``2_paired_safe``
* ``2`` -> ``3_legit_cred``
* ``3`` -> ``4_surprising``
* ``4`` -> ``5_benign_misplace``

Three design choices from the M1-A and E5-A1 experiments are implemented here:

1. **Effective-number class weights** (:func:`class_weights`): computes
   ``(1 - beta) / (1 - beta ** n_c)``, normalizes to unit mean, caps extreme
   rare-class multipliers at ``cap``, and renormalizes to unit mean.
2. **Per-sample full-vocabulary weighting without batch=1 cancellation**
   (:func:`full_vocab_loss`): PyTorch's built-in
   ``F.cross_entropy(logits, target, weight=w, reduction="mean")`` divides by
   ``sum_i w[target_i]``. At ``batch_size == 1`` that denominator equals the
   numerator weight ``w[target_0]``, cancelling class weights completely in
   both loss value and gradient. Computing ``reduction="none"`` and multiplying
   by ``w[labels]`` before taking ``.mean()`` preserves exact class-weight
   scaling at ``batch_size == 1``.
3. **Unilateral positive margin** (:func:`pos_only_margin_loss` and
   ``margin_mode="pos_only"`` in :func:`m1_loss`): in bilateral margin training
   (``margin_mode="both"``), negative samples from ``2_paired_safe`` and
   ``5_benign_misplace`` outnumber ``4_surprising`` by roughly 9.8:1 and apply
   2.86x the cumulative reverse gradient force, eroding the Class 4 boundary.
   ``margin_mode="pos_only"`` applies ``softplus(margin + max(z_2, z_5) - z_4)``
   exclusively on ``4_surprising`` examples and applies zero reverse margin
   penalty on negative classes.
"""

from __future__ import annotations

import math
from typing import Literal, Sequence

import torch
import torch.nn.functional as F

__all__ = [
    "MarginMode",
    "VALID_MARGIN_MODES",
    "class_weights",
    "full_vocab_loss",
    "m1_loss",
    "pos_only_margin_loss",
]

MarginMode = Literal["pos_only", "both", "none"]
VALID_MARGIN_MODES: tuple[str, str, str] = ("pos_only", "both", "none")


def _validate_labels(labels: torch.Tensor | Sequence[int], num_classes: int) -> torch.Tensor:
    if isinstance(labels, torch.Tensor):
        if labels.dtype == torch.bool:
            raise ValueError("labels must be integer class indices, not bool")
        if labels.is_floating_point() or labels.is_complex():
            raise ValueError("labels must be integer class indices")
        labels_t = labels.to(dtype=torch.long)
    else:
        raw = list(labels)
        if any(isinstance(v, bool) or not isinstance(v, int) for v in raw):
            raise ValueError("labels must be integer class indices")
        labels_t = torch.tensor(raw, dtype=torch.long)

    if labels_t.ndim != 1 or labels_t.numel() == 0:
        raise ValueError("labels must be a non-empty 1D tensor of class indices")
    if (labels_t < 0).any() or (labels_t >= num_classes).any():
        raise ValueError(f"labels must lie in [0, {num_classes - 1}]")
    return labels_t


def _validate_class_weight_tensor(weight: torch.Tensor, num_classes: int) -> None:
    if weight.ndim != 1 or weight.numel() != num_classes:
        raise ValueError(f"expected {num_classes} class weights, got shape {tuple(weight.shape)}")
    if weight.dtype == torch.bool or not torch.isfinite(weight).all() or (weight <= 0).any():
        raise ValueError("class weights must be finite and strictly positive")


def class_weights(
    labels: torch.Tensor | Sequence[int],
    beta: float = 0.999,
    cap: float = 3.0,
    num_classes: int = 5,
) -> torch.Tensor:
    """Compute effective-number class weights normalized to mean 1.0.

    Matches the M1-A formulation:

    1. ``w_c = (1 - beta) / (1 - beta ** count_c)``
    2. normalize ``w = w / mean(w)``
    3. cap ``w = clamp(w, max=cap)``
    4. renormalize ``w = w / mean(w)``

    The final renormalization can increase a capped value above ``cap``.
    Thus ``cap`` is an intermediate clipping parameter, not a hard upper
    bound on the returned weights; this preserves the documented formula.

    Args:
        labels: 1D integer class indices in ``[0, num_classes - 1]``. Every
            class must appear at least once.
        beta: effective-number hyperparameter in ``[0, 1)``. ``beta == 0``
            yields uniform weights.
        cap: upper bound applied after initial mean normalization (``>= 1.0``).
        num_classes: number of classes (default ``5``).

    Returns:
        1D ``float32`` tensor of shape ``[num_classes]`` with mean ``1.0``.
    """
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError("num_classes must be a positive integer")
    if isinstance(beta, bool) or not math.isfinite(beta) or not 0.0 <= beta < 1.0:
        raise ValueError("beta must be finite and satisfy 0 <= beta < 1")
    if isinstance(cap, bool) or not math.isfinite(cap) or cap < 1.0:
        raise ValueError("cap must be finite and >= 1.0")

    labels_t = _validate_labels(labels, num_classes)
    counts = torch.bincount(labels_t, minlength=num_classes).to(dtype=torch.float32)
    if (counts == 0).any():
        raise ValueError(f"all {num_classes} training classes are required")

    if beta == 0.0:
        return torch.ones(num_classes, dtype=torch.float32, device=counts.device)

    beta_t = torch.tensor(float(beta), dtype=torch.float32, device=counts.device)
    weights = (1.0 - beta_t) / (1.0 - torch.pow(beta_t, counts))
    weights = weights / weights.mean()
    weights = weights.clamp(max=float(cap))
    return weights / weights.mean()


def pos_only_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin: float = 0.0,
    positive_class: int = 3,
    negative_classes: Sequence[int] = (1, 4),
) -> torch.Tensor:
    """Unilateral positive margin loss for rare ``4_surprising`` transitions.

    Only samples with ``labels == positive_class`` (default index ``3``,
    ``4_surprising``) contribute::

        mean(softplus(margin + max_{c in negative_classes}(z_c) - z_pos))

    Negative-class samples receive zero reverse margin force, preventing
    majority safe classes from pushing down the rare-class logit boundary.
    When a batch contains zero ``positive_class`` samples, returns a
    differentiable zero scalar connected to ``logits``.
    """
    if logits.ndim != 2 or logits.shape[0] == 0:
        raise ValueError("logits must be a non-empty 2D tensor [N, K]")
    num_classes = int(logits.shape[1])
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")
    if isinstance(margin, bool) or not math.isfinite(margin):
        raise ValueError("margin must be a finite float")
    if labels.dtype != torch.long or labels.shape != (logits.shape[0],):
        raise ValueError("labels must be int64 with shape [N] matching logits")
    if (labels < 0).any() or (labels >= num_classes).any():
        raise ValueError(f"labels must lie in [0, {num_classes - 1}]")
    if not 0 <= positive_class < num_classes:
        raise ValueError("positive_class out of range")
    neg_idx = [int(c) for c in negative_classes]
    if not neg_idx or any(c < 0 or c >= num_classes or c == positive_class for c in neg_idx):
        raise ValueError("negative_classes must be valid class indices distinct from positive_class")

    logits_f = logits.float()
    zero = logits_f.sum() * 0.0
    positive_mask = labels == positive_class
    if not positive_mask.any():
        return zero

    pos_rows = logits_f[positive_mask]
    hardest_neg = pos_rows[:, neg_idx].max(dim=1).values
    pos_logit = pos_rows[:, positive_class]
    return F.softplus(float(margin) + hardest_neg - pos_logit).mean()


def m1_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor | None = None,
    margin: float = 0.0,
    margin_weight: float = 0.0,
    teacher_logits: torch.Tensor | None = None,
    kd_weight: float = 0.0,
    temperature: float = 2.0,
    margin_mode: MarginMode = "pos_only",
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Five-class candidate-logit loss combining weighted CE, margin, and KD.

    Args:
        logits: ``[N, 5]`` finite candidate logits.
        labels: ``[N]`` ``torch.long`` class indices in ``0..4``.
        weight: optional ``[5]`` positive class weights applied per sample
            prior to ``.mean()`` (preserving weight scaling at ``batch_size=1``).
        margin: softplus margin offset.
        margin_weight: non-negative multiplier on the margin term.
        teacher_logits: optional ``[N, 5]`` reference logits for protected-class
            distillation on classes ``{0, 1, 2, 4}`` (excluding class ``3``).
        kd_weight: non-negative multiplier on the KL distillation term.
        temperature: positive distillation softmax temperature.
        margin_mode: ``"pos_only"`` (default, E5-A1 unilateral Class 4 margin),
            ``"both"`` (historical M1-C bilateral group-balanced margin), or
            ``"none"`` (disables margin calculation).

    Returns:
        ``(total_loss, parts_dict)`` where ``parts_dict`` holds detached scalar
        diagnostics.
    """
    if logits.ndim != 2 or logits.shape[1] != 5 or logits.shape[0] == 0:
        raise ValueError("expected non-empty [N, 5] logits")
    if labels.dtype != torch.long or labels.shape != (logits.shape[0],):
        raise ValueError("expected int64 labels of shape [N]")
    if (labels < 0).any() or (labels > 4).any():
        raise ValueError("labels must lie in [0, 4]")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")
    if (
        isinstance(margin, bool)
        or isinstance(margin_weight, bool)
        or isinstance(kd_weight, bool)
        or isinstance(temperature, bool)
        or not math.isfinite(margin)
        or not math.isfinite(margin_weight)
        or not math.isfinite(kd_weight)
        or not math.isfinite(temperature)
        or temperature <= 0.0
        or margin_weight < 0.0
        or kd_weight < 0.0
    ):
        raise ValueError("invalid margin, margin_weight, kd_weight, or temperature")
    if margin_mode not in VALID_MARGIN_MODES:
        raise ValueError(f"margin_mode must be one of {VALID_MARGIN_MODES}, got {margin_mode!r}")

    logits_f = logits.float()
    ce_per_sample = F.cross_entropy(logits_f, labels, reduction="none")
    if weight is not None:
        _validate_class_weight_tensor(weight, 5)
        w = weight.to(device=logits.device, dtype=ce_per_sample.dtype)
        ce = (ce_per_sample * w[labels]).mean()
    else:
        ce = ce_per_sample.mean()

    zero = logits_f.sum() * 0.0
    positive = labels == 3
    negative = (labels == 1) | (labels == 4)

    if margin_mode == "none":
        lm = zero
    elif margin_mode == "pos_only":
        lm = pos_only_margin_loss(logits_f, labels, margin=margin)
    else:
        lp = (
            F.softplus(
                float(margin)
                + torch.maximum(logits_f[positive, 1], logits_f[positive, 4])
                - logits_f[positive, 3]
            ).mean()
            if positive.any()
            else zero
        )
        ln = (
            F.softplus(
                float(margin)
                + logits_f[negative, 3]
                - logits_f[negative].gather(1, labels[negative, None]).squeeze(1)
            ).mean()
            if negative.any()
            else zero
        )
        denom = (1 if positive.any() else 0) + (1 if negative.any() else 0) or 1
        lm = (lp + ln) / denom

    keep = labels != 3
    if kd_weight > 0.0:
        if (
            teacher_logits is None
            or teacher_logits.shape != logits.shape
            or not torch.isfinite(teacher_logits).all()
        ):
            raise ValueError("finite aligned teacher_logits required when kd_weight > 0")
        protected_idx = [0, 1, 2, 4]
        if keep.any():
            student_scaled = logits_f[keep][:, protected_idx] / float(temperature)
            teacher_scaled = teacher_logits.detach().float()[keep][:, protected_idx] / float(
                temperature
            )
            kd = F.kl_div(
                F.log_softmax(student_scaled, dim=-1),
                F.softmax(teacher_scaled, dim=-1),
                reduction="batchmean",
            ) * (float(temperature) ** 2)
        else:
            kd = zero
    else:
        kd = zero

    total = ce + float(margin_weight) * lm + float(kd_weight) * kd
    if not torch.isfinite(total):
        raise FloatingPointError("non-finite m1_loss")

    parts: dict[str, float | int] = {
        "ce": float(ce.detach()),
        "margin": float(lm.detach()),
        "kd": float(kd.detach()),
        "margin_pos_n": int(positive.sum()),
        "margin_neg_n": int(negative.sum()),
        "kd_n": int(keep.sum()),
        "margin_mode_pos_only": 1.0 if margin_mode == "pos_only" else 0.0,
    }
    return total, parts


def full_vocab_loss(
    full_logits: torch.Tensor,
    labels: torch.Tensor,
    token_ids: Sequence[int],
    weight: torch.Tensor | None = None,
    margin: float = 0.0,
    margin_weight: float = 0.0,
    teacher_logits: torch.Tensor | None = None,
    kd_weight: float = 0.0,
    temperature: float = 2.0,
    margin_mode: MarginMode = "pos_only",
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Full-vocabulary supervised next-token CE plus candidate-logit auxiliaries.

    Per-sample full-vocabulary CE is multiplied directly by ``weight[labels]``
    before ``.mean()``, preventing PyTorch's ``F.cross_entropy(..., weight=...)``
    self-normalization from cancelling sample weights when ``batch_size == 1``.
    Auxiliary terms from :func:`m1_loss` on the five candidate token logits
    ``z = full_logits[:, token_ids]`` are combined via::

        total = ce + aux_total - F.cross_entropy(z, labels)

    so that when ``margin_weight == 0`` and ``kd_weight == 0``, ``total``
    reduces identically to weighted full-vocabulary next-token CE.
    """
    if full_logits.ndim != 2 or full_logits.shape[0] == 0:
        raise ValueError("expected non-empty [N, V] full_logits")
    if labels.dtype != torch.long or labels.ndim != 1 or len(full_logits) != len(labels):
        raise ValueError("expected [N, V] full_logits and int64 [N] labels")
    if (labels < 0).any() or (labels > 4).any() or not torch.isfinite(full_logits).all():
        raise ValueError("invalid labels or non-finite full_logits")

    candidate_ids = list(token_ids)
    if (
        len(candidate_ids) != 5
        or any(isinstance(t, bool) or not isinstance(t, int) for t in candidate_ids)
        or len(set(candidate_ids)) != 5
        or min(candidate_ids) < 0
        or max(candidate_ids) >= full_logits.shape[1]
    ):
        raise ValueError("token_ids must be 5 distinct valid vocabulary indices")

    ids = torch.tensor(candidate_ids, device=full_logits.device, dtype=torch.long)
    logits_f = full_logits.float()
    token_targets = ids[labels]

    ce_per_sample = F.cross_entropy(logits_f, token_targets, reduction="none")
    if weight is not None:
        _validate_class_weight_tensor(weight, 5)
        w = weight.to(device=full_logits.device, dtype=ce_per_sample.dtype)
        ce_per_sample = ce_per_sample * w[labels]
    ce = ce_per_sample.mean()

    z = logits_f[:, ids]
    aux_total, parts = m1_loss(
        z,
        labels,
        weight=None,
        margin=margin,
        margin_weight=margin_weight,
        teacher_logits=teacher_logits,
        kd_weight=kd_weight,
        temperature=temperature,
        margin_mode=margin_mode,
    )
    total = ce + aux_total - F.cross_entropy(z, labels)
    if not torch.isfinite(total):
        raise FloatingPointError("non-finite full_vocab_loss")

    parts["candidate_ce"] = parts["ce"]
    parts["ce"] = float(ce.detach())
    return total, parts
