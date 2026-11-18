"""Single-token ``ABCD`` scoring and training objectives for PRM and Action Judge V3.

Architectural contract
----------------------
Both **Action Judge V3** and **Learned Process Reward Model (PRM)** reuse the
M1-A safety-aligned Qwen3-VL language-tower LoRA backbone (with the vision tower
frozen) and score transitions directly over **single-token ``ABCD`` last-position
next-token logits** (``DEFAULT_ABCD_TOKEN_IDS = (32, 33, 34, 35)`` in the Qwen
vocabulary). Neither model attaches a separate MLP classification head, a scalar
regression head, or an explicit VisCritic-style visual feature-difference head.

Task formulations
-----------------
* **Learned PRM (4-way over ``ABCD``)**:
  ``A`` -> ``progress`` (``+1.0``),
  ``B`` -> ``no_progress`` (``0.0``),
  ``C`` -> ``regression`` (``-1.0``),
  ``D`` -> ``recovery`` (``+1.0``).
* **Action Judge V3 (dual binary over ``AB``)**:
  two binary next-token classification prompts sharing token IDs ``(32, 33)``
  for ``("yes", "no")``:

  - ``change``: whether the relevant GUI state actually changed between the
    ``before`` and ``after`` frames;
  - ``goal``: whether the stated local target is achieved in the ``after``
    frame (which is distinct from step-level progress when a target was already
    satisfied prior to a redundant action).

All functions in this module are pure PyTorch tensor operations with zero
``transformers`` or ``peft`` runtime dependency.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Sequence

import torch
import torch.nn.functional as F

__all__ = [
    "DEFAULT_ABCD_TOKEN_IDS",
    "DEFAULT_AB_TOKEN_IDS",
    "InputLayout",
    "JUDGE_BINARY_CLASSES",
    "JUDGE_TASKS",
    "PRM_CLASSES",
    "PRM_CLASS_REWARDS",
    "extract_ab_logits",
    "extract_abcd_logits",
    "extract_candidate_logits",
    "judge_loss",
    "prm_loss",
    "score_action_judge",
    "score_binary_judge",
    "score_prm",
]

InputLayout = Literal["auto", "btv", "bv", "tv"]

DEFAULT_ABCD_TOKEN_IDS: tuple[int, int, int, int] = (32, 33, 34, 35)
DEFAULT_AB_TOKEN_IDS: tuple[int, int] = (32, 33)

PRM_CLASSES: tuple[str, str, str, str] = (
    "progress",
    "no_progress",
    "regression",
    "recovery",
)
PRM_CLASS_REWARDS: tuple[float, float, float, float] = (1.0, 0.0, -1.0, 1.0)

JUDGE_BINARY_CLASSES: tuple[str, str] = ("yes", "no")
JUDGE_TASKS: tuple[str, str] = ("change", "goal")


def _validate_confidence(minimum_confidence: float) -> float:
    if (
        isinstance(minimum_confidence, bool)
        or not math.isfinite(minimum_confidence)
        or not 0.0 <= minimum_confidence <= 1.0
    ):
        raise ValueError("minimum_confidence must be a finite float in [0, 1]")
    return float(minimum_confidence)


def _validate_token_ids(token_ids: Sequence[int], vocab_size: int, expected_len: int | None = None) -> list[int]:
    ids = list(token_ids)
    if not ids:
        raise ValueError("token_ids must not be empty")
    if expected_len is not None and len(ids) != expected_len:
        raise ValueError(f"expected {expected_len} token_ids, got {len(ids)}")
    if any(isinstance(t, bool) or not isinstance(t, int) for t in ids):
        raise ValueError("token_ids must be integers, not bool")
    if len(set(ids)) != len(ids):
        raise ValueError("token_ids must be distinct")
    if min(ids) < 0 or max(ids) >= vocab_size:
        raise ValueError(f"token_ids must lie in [0, {vocab_size - 1}]")
    return ids


def extract_candidate_logits(
    logits: torch.Tensor,
    token_ids: Sequence[int] = DEFAULT_ABCD_TOKEN_IDS,
    *,
    input_layout: InputLayout = "auto",
    expected_num_tokens: int | None = None,
) -> torch.Tensor:
    """Extract last-position candidate token logits as a ``[B, K]`` float32 tensor.

    Supported ``logits`` shapes:

    * ``[B, T, V]`` (``input_layout`` in ``{"auto", "btv"}``): selects the
      final sequence position ``logits[:, -1, token_ids]`` -> ``[B, K]``.
    * ``[B, V]`` (``input_layout`` in ``{"auto", "bv"}``): treats ``logits`` as
      already-sliced last-position vocabulary logits ``logits[:, token_ids]``
      -> ``[B, K]``.
    * ``[T, V]`` (``input_layout == "tv"``): unbatched sequence output; selects
      the final position ``logits[-1:, token_ids]`` -> ``[1, K]``.
    * ``[V]`` (1D unbatched last-position vector, ``input_layout == "auto"``):
      selects ``logits[token_ids].unsqueeze(0)`` -> ``[1, K]``.
    """
    if input_layout not in ("auto", "btv", "bv", "tv"):
        raise ValueError(f"unsupported input_layout {input_layout!r}")
    if logits.dtype == torch.bool or not torch.is_floating_point(logits):
        raise ValueError("logits must be a floating-point tensor")
    if logits.numel() == 0 or not torch.isfinite(logits).all():
        raise ValueError("logits must be non-empty and finite")

    if logits.ndim == 3:
        if input_layout not in ("auto", "btv"):
            raise ValueError(f"3D logits require input_layout='btv' or 'auto', got {input_layout!r}")
        last_pos = logits[:, -1, :]
    elif logits.ndim == 2:
        if input_layout == "btv":
            raise ValueError("input_layout='btv' requires 3D [B, T, V] logits")
        if input_layout == "tv":
            last_pos = logits[-1:, :]
        else:
            last_pos = logits
    elif logits.ndim == 1:
        if input_layout != "auto":
            raise ValueError(f"1D logits only support input_layout='auto', got {input_layout!r}")
        last_pos = logits.unsqueeze(0)
    else:
        raise ValueError(f"expected 1D, 2D, or 3D logits, got shape {tuple(logits.shape)}")

    ids = _validate_token_ids(token_ids, int(last_pos.shape[-1]), expected_len=expected_num_tokens)
    idx_tensor = torch.tensor(ids, device=last_pos.device, dtype=torch.long)
    return last_pos.index_select(dim=-1, index=idx_tensor).float()


def extract_abcd_logits(
    logits: torch.Tensor,
    token_ids: Sequence[int] = DEFAULT_ABCD_TOKEN_IDS,
    *,
    input_layout: InputLayout = "auto",
) -> torch.Tensor:
    """Extract 4-way ``ABCD`` last-position logits of shape ``[B, 4]``."""
    return extract_candidate_logits(
        logits,
        token_ids=token_ids,
        input_layout=input_layout,
        expected_num_tokens=4,
    )


def extract_ab_logits(
    logits: torch.Tensor,
    token_ids: Sequence[int] = DEFAULT_AB_TOKEN_IDS,
    *,
    input_layout: InputLayout = "auto",
) -> torch.Tensor:
    """Extract 2-way ``AB`` last-position logits of shape ``[B, 2]``."""
    return extract_candidate_logits(
        logits,
        token_ids=token_ids,
        input_layout=input_layout,
        expected_num_tokens=2,
    )


def _resolve_candidate_matrix(
    logits: torch.Tensor,
    *,
    num_classes: int,
    token_ids: Sequence[int] | None,
    input_layout: InputLayout,
    expected_batch_size: int | None = None,
) -> torch.Tensor:
    """Resolve either pre-extracted ``[B, K]`` logits or vocabulary logits."""
    if token_ids is None:
        if logits.dtype == torch.bool or not torch.is_floating_point(logits):
            raise ValueError("candidate logits must be floating-point")
        if logits.numel() == 0 or not torch.isfinite(logits).all():
            raise ValueError("candidate logits must be non-empty and finite")
        if logits.ndim == 1 and logits.shape[0] == num_classes:
            cand = logits.unsqueeze(0).float()
        elif logits.ndim == 2 and logits.shape[1] == num_classes:
            cand = logits.float()
        else:
            raise ValueError(
                f"when token_ids=None, expected [{num_classes}] or [B, {num_classes}] logits, "
                f"got {tuple(logits.shape)}"
            )
    else:
        effective_layout = input_layout
        if (
            effective_layout == "auto"
            and logits.ndim == 2
            and expected_batch_size == 1
            and logits.shape[0] > 1
        ):
            effective_layout = "tv"
        cand = extract_candidate_logits(
            logits,
            token_ids=token_ids,
            input_layout=effective_layout,
            expected_num_tokens=num_classes,
        )

    if expected_batch_size is not None and cand.shape[0] != expected_batch_size:
        raise ValueError(
            f"batch size mismatch: logits resolved to {cand.shape[0]} rows, "
            f"labels have {expected_batch_size}"
        )
    return cand


def _validate_target_labels(labels: torch.Tensor | Sequence[int], num_classes: int) -> torch.Tensor:
    if isinstance(labels, torch.Tensor):
        if labels.dtype == torch.bool or labels.is_floating_point() or labels.is_complex():
            raise ValueError("labels must be integer class indices")
        target = labels.to(dtype=torch.long)
    else:
        raw = list(labels)
        if any(isinstance(v, bool) or not isinstance(v, int) for v in raw):
            raise ValueError("labels must be integer class indices")
        target = torch.tensor(raw, dtype=torch.long)

    if target.ndim != 1 or target.numel() == 0:
        raise ValueError("labels must be a non-empty 1D sequence of class indices")
    if (target < 0).any() or (target >= num_classes).any():
        raise ValueError(f"labels must lie in [0, {num_classes - 1}]")
    return target


def score_prm(
    logits: torch.Tensor,
    *,
    token_ids: Sequence[int] | None = DEFAULT_ABCD_TOKEN_IDS,
    input_layout: InputLayout = "auto",
    minimum_confidence: float = 0.8,
) -> dict[str, Any]:
    """Score 4-class Process Reward Model predictions over ``ABCD`` tokens.

    Args:
        logits: full-vocabulary ``[B, T, V]`` / ``[B, V]`` / ``[T, V]`` logits,
            or pre-extracted ``[B, 4]`` candidate logits when ``token_ids=None``.
        token_ids: 4 vocabulary indices for ``('A', 'B', 'C', 'D')``, or
            ``None`` when ``logits`` is already ``[B, 4]``.
        input_layout: layout disambiguator (``"auto"``, ``"btv"``, ``"bv"``,
            or ``"tv"``).
        minimum_confidence: abstention threshold in ``[0, 1]``. Rows below
            ``minimum_confidence`` emit ``label=None``, ``model_reward=None``,
            and ``needs_review=True``.
    """
    min_conf = _validate_confidence(minimum_confidence)
    cand = _resolve_candidate_matrix(
        logits,
        num_classes=4,
        token_ids=token_ids,
        input_layout=input_layout,
    )
    probs = F.softmax(cand, dim=-1)
    confs, pred_idx = probs.max(dim=-1)
    needs_review = confs < min_conf

    raw_labels = [PRM_CLASSES[int(i)] for i in pred_idx.tolist()]
    labels: list[str | None] = []
    model_rewards: list[float | None] = []
    for idx_val, review_flag in zip(pred_idx.tolist(), needs_review.tolist()):
        if review_flag:
            labels.append(None)
            model_rewards.append(None)
        else:
            labels.append(PRM_CLASSES[int(idx_val)])
            model_rewards.append(PRM_CLASS_REWARDS[int(idx_val)])

    return {
        "candidate_logits": cand,
        "probabilities": probs,
        "predicted_indices": pred_idx,
        "confidences": confs,
        "needs_review": needs_review,
        "raw_labels": raw_labels,
        "labels": labels,
        "model_rewards": model_rewards,
    }


def prm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor | Sequence[int],
    *,
    token_ids: Sequence[int] | None = DEFAULT_ABCD_TOKEN_IDS,
    class_weight: torch.Tensor | None = None,
    input_layout: InputLayout = "auto",
) -> torch.Tensor:
    """Compute 4-class ``ABCD`` cross-entropy loss for the Learned PRM."""
    target = _validate_target_labels(labels, num_classes=4)
    cand = _resolve_candidate_matrix(
        logits,
        num_classes=4,
        token_ids=token_ids,
        input_layout=input_layout,
        expected_batch_size=int(target.shape[0]),
    )
    target = target.to(device=cand.device)
    ce_per_sample = F.cross_entropy(cand, target, reduction="none")
    if class_weight is not None:
        if (
            class_weight.ndim != 1
            or class_weight.numel() != 4
            or class_weight.dtype == torch.bool
            or not torch.isfinite(class_weight).all()
            or (class_weight <= 0).any()
        ):
            raise ValueError("class_weight must be 4 finite positive weights")
        w = class_weight.to(device=cand.device, dtype=ce_per_sample.dtype)
        ce_per_sample = ce_per_sample * w[target]

    loss = ce_per_sample.mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite prm_loss")
    return loss


def score_binary_judge(
    logits: torch.Tensor,
    *,
    token_ids: Sequence[int] | None = DEFAULT_AB_TOKEN_IDS,
    input_layout: InputLayout = "auto",
    minimum_confidence: float = 0.8,
) -> dict[str, Any]:
    """Score a binary Action Judge head (``change`` or ``goal``) over ``AB``.

    Token ``A`` (index ``0``) corresponds to ``"yes"`` (``True``) and token
    ``B`` (index ``1``) corresponds to ``"no"`` (``False``).
    """
    min_conf = _validate_confidence(minimum_confidence)
    cand = _resolve_candidate_matrix(
        logits,
        num_classes=2,
        token_ids=token_ids,
        input_layout=input_layout,
    )
    probs = F.softmax(cand, dim=-1)
    confs, pred_idx = probs.max(dim=-1)
    needs_review = confs < min_conf
    yes_prob = probs[:, 0]

    raw_labels = [JUDGE_BINARY_CLASSES[int(i)] for i in pred_idx.tolist()]
    labels: list[str | None] = []
    verdicts: list[bool | None] = []
    for idx_val, review_flag in zip(pred_idx.tolist(), needs_review.tolist()):
        if review_flag:
            labels.append(None)
            verdicts.append(None)
        else:
            labels.append(JUDGE_BINARY_CLASSES[int(idx_val)])
            verdicts.append(int(idx_val) == 0)

    return {
        "candidate_logits": cand,
        "probabilities": probs,
        "yes_probabilities": yes_prob,
        "predicted_indices": pred_idx,
        "confidences": confs,
        "needs_review": needs_review,
        "raw_labels": raw_labels,
        "labels": labels,
        "verdicts": verdicts,
    }


def score_action_judge(
    change_logits: torch.Tensor,
    goal_logits: torch.Tensor | None = None,
    *,
    token_ids: Sequence[int] | None = DEFAULT_AB_TOKEN_IDS,
    input_layout: InputLayout = "auto",
    minimum_confidence: float = 0.8,
) -> dict[str, Any]:
    """Score Action Judge V3 dual binary heads (``change`` and optional ``goal``)."""
    change_result = score_binary_judge(
        change_logits,
        token_ids=token_ids,
        input_layout=input_layout,
        minimum_confidence=minimum_confidence,
    )
    goal_result = (
        score_binary_judge(
            goal_logits,
            token_ids=token_ids,
            input_layout=input_layout,
            minimum_confidence=minimum_confidence,
        )
        if goal_logits is not None
        else None
    )
    return {
        "change": change_result,
        "goal": goal_result,
    }


def judge_loss(
    change_logits: torch.Tensor | None = None,
    change_labels: torch.Tensor | Sequence[int] | None = None,
    *,
    goal_logits: torch.Tensor | None = None,
    goal_labels: torch.Tensor | Sequence[int] | None = None,
    token_ids: Sequence[int] | None = DEFAULT_AB_TOKEN_IDS,
    goal_weight: float = 1.0,
    input_layout: InputLayout = "auto",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Multi-task binary ``AB`` cross-entropy loss for Action Judge V3.

    Supports single-task updates (``change`` only or ``goal`` only, as in
    interleaved multi-task schedules) as well as joint ``change + goal_weight * goal``
    updates when both heads are supplied.
    """
    if isinstance(goal_weight, bool) or not math.isfinite(goal_weight) or goal_weight < 0.0:
        raise ValueError("goal_weight must be a non-negative finite float")
    if (change_logits is None) != (change_labels is None):
        raise ValueError("change_logits and change_labels must be provided together")
    if (goal_logits is None) != (goal_labels is None):
        raise ValueError("goal_logits and goal_labels must be provided together")
    if change_logits is None and goal_logits is None:
        raise ValueError("at least one of (change_logits, change_labels) or (goal_logits, goal_labels) is required")

    change_ce: torch.Tensor | None = None
    goal_ce: torch.Tensor | None = None

    if change_logits is not None and change_labels is not None:
        c_target = _validate_target_labels(change_labels, num_classes=2)
        c_cand = _resolve_candidate_matrix(
            change_logits,
            num_classes=2,
            token_ids=token_ids,
            input_layout=input_layout,
            expected_batch_size=int(c_target.shape[0]),
        )
        change_ce = F.cross_entropy(c_cand, c_target.to(device=c_cand.device))

    if goal_logits is not None and goal_labels is not None:
        g_target = _validate_target_labels(goal_labels, num_classes=2)
        g_cand = _resolve_candidate_matrix(
            goal_logits,
            num_classes=2,
            token_ids=token_ids,
            input_layout=input_layout,
            expected_batch_size=int(g_target.shape[0]),
        )
        goal_ce = F.cross_entropy(g_cand, g_target.to(device=g_cand.device))

    if change_ce is not None and goal_ce is not None:
        total = change_ce + float(goal_weight) * goal_ce
    elif change_ce is not None:
        total = change_ce
    else:
        assert goal_ce is not None
        total = float(goal_weight) * goal_ce

    if not torch.isfinite(total):
        raise FloatingPointError("non-finite judge_loss")

    parts = {
        "change_ce": float(change_ce.detach()) if change_ce is not None else 0.0,
        "goal_ce": float(goal_ce.detach()) if goal_ce is not None else 0.0,
        "total": float(total.detach()),
    }
    return total, parts
