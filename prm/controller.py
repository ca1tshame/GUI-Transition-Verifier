"""Fail-closed 7-state post-action controller and pure-stdlib Feedback contract.

Design invariants
-----------------
1. **No privileged DOM Oracle in runtime model feedback**: :class:`Feedback`
   carries only model-derived step predictions, uncertainty, and candidate risk
   scores. Privileged evaluation-only oracle rewards are excluded from the
   runtime controller contract.
2. **Model signals are advisory, never environment truth**: a VLM prediction
   alone cannot declare terminal task ``stop``, confirm a safety ``block``, or
   mint positive ``conservative_reward``.
3. **Idempotent & stale-failure guard**: non-idempotent actions (such as
   ``checkbox``, ``toggle``, ``submit``, ``pay``, or ``delete``, or any action
   with ``idempotent=False``) are never automatically retried. Stale failure or
   unchanged-state signals on non-idempotent controls escalate to ``continue``
   (when verified local goal state is already achieved), ``observe``, or
   ``review`` instead of blind re-execution that would double-toggle or
   double-submit.
4. **Bounded budgets and cycle guard**: ``retry``, ``recover``, and ``observe``
   each enforce explicit budgets, and repeated ``(state, action)`` cycles
   escalate immediately to ``review``.
5. **Original-precision BF16 safety escalation**: high-risk actions, safety
   review events, or elevated ``safety_p1`` scores require re-verification
   through a pinned ``backend == "original_bf16"`` service via
   :func:`require_original_bf16_safety_review` (fail-closed on fastPE, missing
   service, exception, or identity mismatch).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

__all__ = [
    "ACTIONS",
    "ControllerAction",
    "ControllerContext",
    "ControllerDecision",
    "Decision",
    "DEFAULT_RISK_THRESHOLD",
    "Evidence",
    "Feedback",
    "ModelSignal",
    "NON_IDEMPOTENT_ACTION_TYPES",
    "PRM_LABELS",
    "PRM_REWARDS",
    "Result",
    "StateEvidence",
    "Context",
    "conservative_reward",
    "decide",
    "decide_action",
    "evaluate_progress_feedback",
    "has_state_action_loop",
    "is_non_idempotent_action",
    "require_original_bf16_safety_review",
]

ACTIONS: tuple[str, str, str, str, str, str, str] = (
    "continue",
    "observe",
    "retry",
    "recover",
    "review",
    "stop",
    "block",
)

PRM_LABELS: tuple[str, str, str, str] = (
    "progress",
    "no_progress",
    "regression",
    "recovery",
)
PRM_REWARDS: tuple[float, float, float, float] = (1.0, 0.0, -1.0, 1.0)

# Dev-frozen hijack alert threshold used across cascade and safety review gates.
DEFAULT_RISK_THRESHOLD: float = 0.6702836155891418

NON_IDEMPOTENT_ACTION_TYPES: frozenset[str] = frozenset(
    {
        "checkbox",
        "click_checkbox",
        "click_checkboxes",
        "toggle",
        "submit",
        "pay",
        "purchase",
        "delete",
        "remove",
        "send",
        "transfer",
    }
)


class ControllerAction(str, Enum):
    """Seven post-action controller states."""

    CONTINUE = "continue"
    OBSERVE = "observe"
    RETRY = "retry"
    RECOVER = "recover"
    REVIEW = "review"
    STOP = "stop"
    BLOCK = "block"


Decision = ControllerAction


def _check_unit_prob(value: float, name: str, *, upper_slack: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a float in [0, 1], got {value!r}")
    fval = float(value)
    if not math.isfinite(fval) or fval < 0.0 or fval > 1.0 + upper_slack:
        raise ValueError(f"{name} must be a finite probability in [0, 1], got {value!r}")
    return min(1.0, fval)


def _check_optional_bool(value: Any, name: str) -> None:
    if value is not None and type(value) is not bool:
        raise ValueError(f"{name} must be bool or None, got {value!r}")


def _check_simplex(probabilities: Sequence[float], size: int, name: str) -> tuple[float, ...]:
    vals = tuple(_check_unit_prob(p, name) for p in probabilities)
    if len(vals) != size:
        raise ValueError(f"{name} must have length {size}, got {len(vals)}")
    if abs(sum(vals) - 1.0) > 1e-4:
        raise ValueError(f"{name} must sum to 1.0")
    return vals


@dataclass(frozen=True, slots=True)
class Feedback:
    """Runtime model feedback contract (strictly free of privileged DOM Oracle fields).

    Attributes:
        label: abstention-gated PRM class in ``PRM_LABELS`` or ``None`` when
            ``needs_review`` is ``True``.
        model_reward: advisory step score in ``{-1.0, 0.0, 1.0}`` or ``None``
            when abstaining. Never treated as verified outcome reward.
        confidence: maximum predicted class probability in ``[0, 1]``.
        needs_review: ``True`` when model confidence fell below threshold.
        changed: optional Action Judge binary state-change verdict.
        goal_achieved: optional Action Judge local-target verdict.
        safety_p1: optional 5-class verifier ``1_hijack`` probability ``p1``.
        safety_p1p5: optional 5-class inconsistency score ``p1 + p5``.
        stale_signal: ``True`` when the observation/prediction may lag behind
            an already-applied side effect.
    """

    label: str | None
    model_reward: float | None
    confidence: float
    needs_review: bool
    changed: bool | None = None
    goal_achieved: bool | None = None
    safety_p1: float | None = None
    safety_p1p5: float | None = None
    stale_signal: bool = False

    def __post_init__(self) -> None:
        _check_unit_prob(self.confidence, "confidence")
        if type(self.needs_review) is not bool:
            raise ValueError("needs_review must be a bool")
        if type(self.stale_signal) is not bool:
            raise ValueError("stale_signal must be a bool")
        if self.label is not None and (
            not isinstance(self.label, str) or self.label not in PRM_LABELS
        ):
            raise ValueError(f"label must be one of {PRM_LABELS} or None, got {self.label!r}")
        if self.model_reward is not None:
            if isinstance(self.model_reward, bool) or not math.isfinite(self.model_reward):
                raise ValueError("model_reward must be a finite float or None")
        if self.needs_review and (self.label is not None or self.model_reward is not None):
            raise ValueError("abstained Feedback (needs_review=True) must have label=None and model_reward=None")
        _check_optional_bool(self.changed, "changed")
        _check_optional_bool(self.goal_achieved, "goal_achieved")
        if self.safety_p1 is not None:
            _check_unit_prob(self.safety_p1, "safety_p1")
        if self.safety_p1p5 is not None:
            _check_unit_prob(self.safety_p1p5, "safety_p1p5", upper_slack=1e-4)


def evaluate_progress_feedback(
    probabilities: Sequence[float],
    *,
    safety_probabilities: Sequence[float] | None = None,
    change_probabilities: Sequence[float] | None = None,
    goal_probabilities: Sequence[float] | None = None,
    minimum_confidence: float = 0.8,
    stale_signal: bool = False,
) -> Feedback:
    """Build a :class:`Feedback` object from normalized model probability vectors."""
    min_conf = _check_unit_prob(minimum_confidence, "minimum_confidence")
    prm_probs = _check_simplex(probabilities, 4, "PRM probabilities")
    best_idx = max(range(4), key=prm_probs.__getitem__)
    conf = prm_probs[best_idx]
    needs_review = conf < min_conf

    p1: float | None = None
    p1p5: float | None = None
    if safety_probabilities is not None:
        s_probs = _check_simplex(safety_probabilities, 5, "safety probabilities")
        p1 = s_probs[0]
        p1p5 = min(1.0, s_probs[0] + s_probs[4])

    changed: bool | None = None
    if change_probabilities is not None:
        c_probs = _check_simplex(change_probabilities, 2, "change probabilities")
        if max(c_probs) >= min_conf:
            changed = c_probs[0] > c_probs[1]

    goal_hit: bool | None = None
    if goal_probabilities is not None:
        g_probs = _check_simplex(goal_probabilities, 2, "goal probabilities")
        if max(g_probs) >= min_conf:
            goal_hit = g_probs[0] > g_probs[1]

    return Feedback(
        label=None if needs_review else PRM_LABELS[best_idx],
        model_reward=None if needs_review else PRM_REWARDS[best_idx],
        confidence=conf,
        needs_review=needs_review,
        changed=changed,
        goal_achieved=goal_hit,
        safety_p1=p1,
        safety_p1p5=p1p5,
        stale_signal=stale_signal,
    )


@dataclass(frozen=True, slots=True)
class ModelSignal:
    """Compact advisory model prediction ``(label, confidence)``."""

    label: str
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("ModelSignal.label must be a non-empty string")
        _check_unit_prob(self.confidence, "ModelSignal.confidence")


@dataclass(frozen=True, slots=True)
class StateEvidence:
    """Runtime environment / task-adapter state evidence (untrusted unless verified).

    All boolean status fields are ignored for ``stop`` / ``continue`` / ``retry``
    authorization unless ``trusted=True`` and a non-empty ``evidence_id`` is
    attached by a runtime state adapter.
    """

    trusted: bool = False
    goal_achieved: bool | None = None
    action_failed: bool | None = None
    task_state_changed: bool | None = None
    progress: bool | None = None
    regression: bool | None = None
    recovered: bool | None = None
    safe_to_retry: bool | None = None
    safe_recovery_available: bool | None = None
    stale_signal: bool = False
    terminal: str | None = None
    safety_event: str | None = None
    evidence_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.trusted) is not bool:
            raise ValueError("trusted must be a bool")
        if type(self.stale_signal) is not bool:
            raise ValueError("stale_signal must be a bool")
        for field_name in (
            "goal_achieved",
            "action_failed",
            "task_state_changed",
            "progress",
            "regression",
            "recovered",
            "safe_to_retry",
            "safe_recovery_available",
        ):
            _check_optional_bool(getattr(self, field_name), field_name)
        if self.terminal not in (None, "success", "failure", "truncated"):
            raise ValueError(f"invalid terminal status {self.terminal!r}")
        if self.safety_event not in (None, "confirmed_block", "needs_original_bf16_review"):
            raise ValueError(f"invalid safety_event {self.safety_event!r}")
        if self.trusted and (not isinstance(self.evidence_id, str) or not self.evidence_id.strip()):
            raise ValueError("trusted evidence requires a non-empty traceable evidence_id")


Evidence = StateEvidence


def _validate_sha256(digest: str | None, name: str) -> None:
    if digest is None:
        return
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(ch not in "0123456789abcdef" for ch in digest.lower())
    ):
        raise ValueError(f"invalid {name}: expected 64-character hex SHA-256")


@dataclass(frozen=True, slots=True)
class ControllerContext:
    """Execution context, budgets, and action metadata for one controller step."""

    task: str
    action_type: str
    idempotent: bool = False
    before_sha256: str | None = None
    after_sha256: str | None = None
    retries_used: int = 0
    retry_budget: int = 1
    recoveries_used: int = 0
    recovery_budget: int = 1
    same_action_attempts: int = 1
    max_same_action_attempts: int = 2
    observations_used: int = 0
    observation_budget: int = 1
    recent_state_action: tuple[str, ...] = field(default_factory=tuple)
    high_risk_action: bool = False
    risk_threshold: float = DEFAULT_RISK_THRESHOLD

    def __post_init__(self) -> None:
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("task must be a non-empty string")
        if not isinstance(self.action_type, str) or not self.action_type.strip():
            raise ValueError("action_type must be a non-empty string")
        if type(self.idempotent) is not bool:
            raise ValueError("idempotent must be a bool")
        if type(self.high_risk_action) is not bool:
            raise ValueError("high_risk_action must be a bool")
        _validate_sha256(self.before_sha256, "before_sha256")
        _validate_sha256(self.after_sha256, "after_sha256")
        _check_unit_prob(self.risk_threshold, "risk_threshold")
        for budget_name in (
            "retries_used",
            "retry_budget",
            "recoveries_used",
            "recovery_budget",
            "same_action_attempts",
            "max_same_action_attempts",
            "observations_used",
            "observation_budget",
        ):
            val = getattr(self, budget_name)
            if isinstance(val, bool) or not isinstance(val, int) or val < 0:
                raise ValueError(f"{budget_name} must be a non-negative integer")
        if not isinstance(self.recent_state_action, tuple) or any(
            not isinstance(item, str) for item in self.recent_state_action
        ):
            raise ValueError("recent_state_action must be a tuple[str, ...]")


Context = ControllerContext


@dataclass(frozen=True, slots=True)
class ControllerDecision:
    """Decision emitted by :func:`decide_action`."""

    decision: ControllerAction
    reason: str
    evidence_id: str | None
    model_advisory: tuple[str, ...] = ()
    requires_original_bf16: bool = False

    @property
    def action(self) -> str:
        return self.decision.value


Result = ControllerDecision


def is_non_idempotent_action(action_type: str, *, declared_idempotent: bool = False) -> bool:
    """Return ``True`` when re-executing ``action_type`` risks side-effect duplication."""
    if not isinstance(action_type, str) or not action_type.strip():
        raise ValueError("action_type must be a non-empty string")
    if type(declared_idempotent) is not bool:
        raise ValueError("declared_idempotent must be a bool")
    normalized = action_type.strip().lower()
    if normalized in NON_IDEMPOTENT_ACTION_TYPES:
        return True
    return not declared_idempotent


def has_state_action_loop(recent_state_action: Sequence[str]) -> bool:
    """Return ``True`` if any ``(state, action)`` fingerprint repeated."""
    items = list(recent_state_action)
    return len(items) != len(set(items))


def _collect_advisory(
    feedback: Feedback | None,
    judge: ModelSignal | None,
    reward: ModelSignal | None,
) -> tuple[str, ...]:
    notes: list[str] = []
    if feedback is not None:
        tag = feedback.label if feedback.label is not None else "abstain"
        notes.append(f"feedback:{tag}:{feedback.confidence:.4f}")
        if feedback.changed is not None:
            notes.append(f"feedback_changed:{feedback.changed}")
        if feedback.goal_achieved is not None:
            notes.append(f"feedback_goal:{feedback.goal_achieved}")
        if feedback.safety_p1 is not None:
            notes.append(f"safety_p1:{feedback.safety_p1:.4f}")
    if judge is not None:
        notes.append(f"judge:{judge.label}:{judge.confidence:.4f}")
    if reward is not None:
        notes.append(f"reward:{reward.label}:{reward.confidence:.4f}")
    return tuple(notes)


def decide_action(
    context: ControllerContext,
    evidence: StateEvidence | None = None,
    feedback: Feedback | None = None,
    *,
    judge: ModelSignal | None = None,
    reward: ModelSignal | None = None,
) -> ControllerDecision:
    """Fail-closed 7-state post-action transition controller.

    Outputs one of ``("continue", "observe", "retry", "recover", "review",
    "stop", "block")`` wrapped in :class:`ControllerDecision`.
    """
    e = evidence if evidence is not None else StateEvidence()
    advisory = _collect_advisory(feedback, judge, reward)
    trusted_id = e.evidence_id if e.trusted else None
    non_idempotent = is_non_idempotent_action(
        context.action_type,
        declared_idempotent=context.idempotent,
    )
    stale = e.stale_signal or (feedback is not None and feedback.stale_signal)

    def emit(
        act: ControllerAction,
        reason: str,
        *,
        raw_bf16: bool = False,
    ) -> ControllerDecision:
        return ControllerDecision(
            decision=act,
            reason=reason,
            evidence_id=trusted_id,
            model_advisory=advisory,
            requires_original_bf16=raw_bf16,
        )

    # 1. Safety block & original-precision BF16 escalation take strict priority.
    if e.trusted and e.safety_event == "confirmed_block":
        return emit(
            ControllerAction.BLOCK,
            "confirmed safety block",
            raw_bf16=context.high_risk_action,
        )

    elevated_model_risk = (
        feedback is not None
        and feedback.safety_p1 is not None
        and feedback.safety_p1 >= context.risk_threshold
    )
    if (
        context.high_risk_action
        or (e.trusted and e.safety_event == "needs_original_bf16_review")
        or elevated_model_risk
    ):
        return emit(
            ControllerAction.REVIEW,
            "high-risk action or safety score requires original BF16 safety review",
            raw_bf16=True,
        )

    # 2. Verified terminal states.
    if e.trusted and e.terminal == "success":
        return emit(ControllerAction.STOP, "verified environment terminal success")
    if e.trusted and e.terminal in ("failure", "truncated"):
        return emit(ControllerAction.REVIEW, "terminal failure or truncation requires review")

    # 3. Verified goal / recovery prevents unnecessary or harmful re-execution
    #    (including when a stale failure/regression signal arrives on checkboxes/text).
    if e.trusted and e.goal_achieved is True:
        return emit(
            ControllerAction.CONTINUE,
            "verified local goal already achieved; do not retry",
        )
    if e.trusted and e.recovered is True:
        return emit(ControllerAction.CONTINUE, "verified recovery; do not repeat")

    # 4. Verified regression: bounded safe recovery or human review.
    if e.trusted and e.regression is True:
        if e.safe_recovery_available is True and context.recoveries_used < context.recovery_budget:
            return emit(ControllerAction.RECOVER, "verified regression and safe recovery available")
        return emit(ControllerAction.REVIEW, "verified regression without bounded safe recovery")

    # 5. Verified action failure: enforce recovery, non-idempotent stale guard,
    #    state-change check, retry/attempt budgets, and cycle detection.
    if e.trusted and e.action_failed is True:
        if e.safe_recovery_available is True and context.recoveries_used < context.recovery_budget:
            return emit(
                ControllerAction.RECOVER,
                "verified failure with explicitly safe bounded recovery",
            )
        if non_idempotent:
            return emit(
                ControllerAction.REVIEW,
                "non-idempotent action blocked from automatic retry",
            )
        if stale:
            if context.observations_used < context.observation_budget and not has_state_action_loop(
                context.recent_state_action
            ):
                return emit(
                    ControllerAction.OBSERVE,
                    "stale failure signal; observe current GUI state before retrying",
                )
            return emit(
                ControllerAction.REVIEW,
                "stale failure signal with exhausted observation budget or cycle",
            )
        if e.safe_to_retry is not True:
            return emit(ControllerAction.REVIEW, "retry safety not established")
        if e.task_state_changed is not False:
            if context.observations_used < context.observation_budget and not has_state_action_loop(
                context.recent_state_action
            ):
                return emit(ControllerAction.OBSERVE, "verify state before retry")
            return emit(
                ControllerAction.REVIEW,
                "state changed after failed action and observation budget exhausted",
            )
        if (
            context.retries_used >= context.retry_budget
            or context.same_action_attempts >= context.max_same_action_attempts
        ):
            return emit(ControllerAction.REVIEW, "retry or repetition budget exhausted")
        if has_state_action_loop(context.recent_state_action):
            return emit(ControllerAction.REVIEW, "repeated state-action cycle detected")
        return emit(
            ControllerAction.RETRY,
            "verified failure, unchanged task state, bounded safe idempotent retry",
        )

    # 6. Verified benign state progress without failure.
    if e.trusted and e.task_state_changed is True and not stale:
        return emit(ControllerAction.CONTINUE, "verified change without established failure")

    # 7. Untrusted, ambiguous, or stale signals: guard non-idempotent actions,
    #    reject state-action cycles, and fall back to bounded observe -> review.
    if has_state_action_loop(context.recent_state_action):
        return emit(ControllerAction.REVIEW, "repeated state-action cycle detected")

    if non_idempotent and stale:
        if context.observations_used < context.observation_budget:
            return emit(
                ControllerAction.OBSERVE,
                "stale signal on non-idempotent action; observe state without re-executing",
            )
        return emit(
            ControllerAction.REVIEW,
            "stale signal on non-idempotent action requires review",
        )

    if context.observations_used < context.observation_budget:
        return emit(
            ControllerAction.OBSERVE,
            "insufficient trusted evidence; observe instead of retrying",
        )
    return emit(
        ControllerAction.REVIEW,
        "insufficient trusted evidence and observation budget exhausted",
    )


decide = decide_action


def conservative_reward(
    evidence: StateEvidence,
    reward_signal: ModelSignal | Feedback | None = None,
) -> float | None:
    """Return conservative verified step reward; unverified signals yield ``None``.

    Model predictions in ``reward_signal`` are intentionally advisory and can
    never mint positive reward without traceable ``trusted`` state evidence.
    """
    _ = reward_signal
    if not evidence.trusted or not evidence.evidence_id:
        return None
    if evidence.regression is True:
        return -1.0
    if evidence.recovered is True:
        return 1.0
    if evidence.progress is True:
        return 1.0
    if evidence.task_state_changed is False:
        return 0.0
    return None


def require_original_bf16_safety_review(
    review_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None,
    request: Mapping[str, Any],
    expected_identity: str,
) -> dict[str, Any]:
    """Execute fail-closed safety escalation against a pinned ``original_bf16`` service.

    Returns ``allow=False`` with ``status="review"`` whenever ``review_fn`` is
    missing, raises an exception, reports a non-``"original_bf16"`` backend
    (such as ``"fastpe"`` or ``"gemm"``), mismatches ``expected_identity``, or
    omits a strict ``bool`` ``allow`` field.
    """
    if (
        review_fn is None
        or not isinstance(expected_identity, str)
        or not expected_identity.strip()
    ):
        return {
            "status": "review",
            "allow": False,
            "backend": None,
            "reason": "original BF16 safety verifier unavailable",
        }
    try:
        result = review_fn(request)
    except Exception:
        return {
            "status": "review",
            "allow": False,
            "backend": None,
            "reason": "original BF16 safety verifier failed",
        }
    if (
        not isinstance(result, Mapping)
        or result.get("backend") != "original_bf16"
        or result.get("identity") != expected_identity
        or type(result.get("allow")) is not bool
    ):
        return {
            "status": "review",
            "allow": False,
            "backend": result.get("backend") if isinstance(result, Mapping) else None,
            "reason": "safety verifier identity or response mismatch",
        }
    return {
        "status": "verified",
        "allow": result["allow"],
        "backend": "original_bf16",
        "identity": expected_identity,
        "risk_alert": result.get("risk_alert"),
        "p1": result.get("p1"),
        "p1_plus_p5": result.get("p1_plus_p5"),
        "reason": "caller-provided pinned original BF16 safety result",
    }
