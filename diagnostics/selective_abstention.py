"""Risk-aware selective verification with three decision states.

A verifier that is forced to answer every transition will answer some of them
wrongly. A verifier that can say "I am not sure" turns a fraction of those
errors into review work. This module implements that second behaviour and, just
as importantly, scores it honestly.

Decision states
---------------
``accept``
    Confident and low risk: the automatic decision stands.
``flag``
    The risk score crossed the alert threshold: raise it even if the model is
    confident. Alerts are scored separately, by alert precision and by missed
    hijacks that were accepted.
``review``
    Evidence is insufficient: no automatic final decision. Review rows are
    **never** counted as correct. Accuracy is reported on accepted rows only,
    with coverage alongside it, because accuracy without coverage is free.

Priority is ``flag > review > accept``. A high-risk row is never downgraded to
``review`` just because the model is uncertain.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Sequence

__all__ = [
    "Decision",
    "PROGRESS_LABELS",
    "apply_temperature",
    "decide",
    "expected_calibration_error",
    "evaluate_decisions",
    "fit_temperature",
    "negative_log_likelihood",
    "oracle_review_curve",
    "progress_feedback",
    "reliability_bins",
    "risk_coverage_curve",
    "risk_scores",
    "selective_accuracy",
]

PROGRESS_LABELS = ("progress", "no_progress", "regression", "recovery")
PROGRESS_REWARDS = (1.0, 0.0, -1.0, 1.0)


class Decision(str, Enum):
    ACCEPT = "accept"
    FLAG = "flag"
    REVIEW = "review"


def _check_probability(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite probability, got {value!r}")
    return value


def decide(
    confidence: float,
    risk: float,
    *,
    confidence_threshold: float = 0.8,
    risk_threshold: float = 0.5,
) -> Decision:
    """Assign one decision state to a single prediction.

    Args:
        confidence: probability of the selected class, in ``[0, 1]``.
        risk: hazard score, in ``[0, 1]``. Higher means more dangerous.
        confidence_threshold: below this, the row goes to review.
        risk_threshold: at or above this, the row is flagged regardless of
            confidence.
    """
    confidence = _check_probability(confidence, "confidence")
    risk = _check_probability(risk, "risk")
    if risk >= risk_threshold:
        return Decision.FLAG
    if confidence < confidence_threshold:
        return Decision.REVIEW
    return Decision.ACCEPT


def _simplex(probabilities: Sequence[float], size: int, name: str) -> tuple[float, ...]:
    values = tuple(float(p) for p in probabilities)
    if len(values) != size:
        raise ValueError(f"{name} must contain {size} probabilities")
    if any(not math.isfinite(p) or p < 0 or p > 1 for p in values):
        raise ValueError(f"{name} must be finite probabilities")
    if abs(sum(values) - 1.0) > 1e-4:
        raise ValueError(f"{name} must sum to one")
    return values


def progress_feedback(
    probabilities: Sequence[float],
    *,
    minimum_confidence: float = 0.8,
) -> dict[str, object]:
    """Map a four-way progress distribution to a label or a review request.

    The four labels are ``progress``, ``no_progress``, ``regression`` and
    ``recovery``. Below ``minimum_confidence`` no label is emitted: the caller
    receives ``needs_review=True`` instead of the argmax. A process reward that
    always answers is indistinguishable from one that guesses.
    """
    values = _simplex(probabilities, 4, "progress probabilities")
    _check_probability(minimum_confidence, "minimum_confidence")
    index = max(range(4), key=values.__getitem__)
    confidence = values[index]
    if confidence < minimum_confidence:
        return {
            "label": None,
            "reward": None,
            "confidence": confidence,
            "needs_review": True,
        }
    return {
        "label": PROGRESS_LABELS[index],
        "reward": PROGRESS_REWARDS[index],
        "confidence": confidence,
        "needs_review": False,
    }


def risk_scores(safety_probabilities: Sequence[float]) -> dict[str, float]:
    """Collapse a five-way risk distribution into ``p1`` and ``p1+p5``.

    ``p1`` is the hijack mass alone; ``p1+p5`` adds the benign-misplacement
    mass and is the regime used for inconsistency-risk AUCs. Scores are not
    safety truth - they are model outputs with a calibrated operating point.
    """
    values = _simplex(safety_probabilities, 5, "safety probabilities")
    return {"p1": values[0], "p1p5": values[0] + values[4]}


def evaluate_decisions(
    labels: Sequence[int],
    predictions: Sequence[int],
    decisions: Sequence[Decision | str],
    *,
    hijack_label: int = 0,
) -> dict[str, object]:
    """Score a decision assignment against ground truth.

    Args:
        labels: gold class indices.
        predictions: argmax class indices.
        decisions: one decision per row.
        hijack_label: index of the high-risk class, used for missed-alert
            accounting.

    Returns:
        Coverage, accepted accuracy, and three *separate* error denominators
        that must not be conflated:

        * ``overall_errors`` - every argmax error in the set.
        * ``errors_in_flagged`` - errors a human reviewer would see because the
          row was flagged.
        * ``errors_in_reviewed`` - errors resolved only if review is perfect.
        * ``oracle_remaining_errors`` - errors left after both flagged and
          reviewed rows are perfectly resolved.
    """
    if not (len(labels) == len(predictions) == len(decisions)):
        raise ValueError("labels, predictions and decisions must have equal length")
    if not labels:
        raise ValueError("empty evaluation set")

    normalised = [
        d.value if isinstance(d, Decision) else str(d) for d in decisions
    ]
    invalid = set(normalised) - {Decision.ACCEPT.value, Decision.FLAG.value, Decision.REVIEW.value}
    if invalid:
        raise ValueError(f"unknown decision value(s): {sorted(invalid)}")

    accepted = [i for i, d in enumerate(normalised) if d == Decision.ACCEPT.value]
    flagged = [i for i, d in enumerate(normalised) if d == Decision.FLAG.value]
    reviewed = [i for i, d in enumerate(normalised) if d == Decision.REVIEW.value]

    correct = sum(1 for i in accepted if predictions[i] == labels[i])
    overall_errors = [i for i in range(len(labels)) if predictions[i] != labels[i]]
    errors_in_flagged = [i for i in flagged if predictions[i] != labels[i]]
    errors_in_reviewed = [i for i in reviewed if predictions[i] != labels[i]]
    missed = [i for i in accepted if labels[i] == hijack_label and predictions[i] != hijack_label]
    hijack_rows = [i for i in range(len(labels)) if labels[i] == hijack_label]
    false_alarms = [i for i in flagged if labels[i] != hijack_label]

    n = len(labels)
    return {
        "n": n,
        "accept": len(accepted),
        "flag": len(flagged),
        "review": len(reviewed),
        "coverage": len(accepted) / n,
        "accept_accuracy": correct / len(accepted) if accepted else None,
        "accept_errors": len(accepted) - correct,
        "overall_errors": len(overall_errors),
        "errors_in_flagged": len(errors_in_flagged),
        "errors_in_reviewed": len(errors_in_reviewed),
        "oracle_remaining_errors": len(overall_errors)
        - len(errors_in_flagged)
        - len(errors_in_reviewed),
        "hijack_total": len(hijack_rows),
        "hijack_flagged": sum(1 for i in flagged if labels[i] == hijack_label),
        "hijack_reviewed": sum(1 for i in reviewed if labels[i] == hijack_label),
        "missed_hijack_accepted": len(missed),
        "false_alarms": len(false_alarms),
        "flag_precision": (
            (len(flagged) - len(false_alarms)) / len(flagged) if flagged else None
        ),
    }


def risk_coverage_curve(
    errors: Sequence[int],
    reject_scores: Sequence[float],
) -> tuple[list[tuple[float, float]], float]:
    """Selective risk as a function of coverage.

    Rows are accepted in ascending ``reject_scores`` order (least suspicious
    first). At coverage ``k / n`` the accepted set is the first ``k`` rows and
    the selective risk is their error rate. At coverage 1 the risk equals the
    full error rate, so no-rejection behaviour is the right endpoint.

    Returns:
        ``(points, aurc)`` where each point is ``(coverage, risk)`` and ``aurc``
        is the area under the risk-coverage curve (lower is better).
    """
    if len(errors) != len(reject_scores):
        raise ValueError("errors and reject_scores must have equal length")
    if not errors:
        raise ValueError("empty curve")
    order = sorted(range(len(errors)), key=lambda i: reject_scores[i])
    points: list[tuple[float, float]] = []
    cumulative = 0
    for k, index in enumerate(order, start=1):
        cumulative += int(bool(errors[index]))
        points.append((k / len(errors), cumulative / k))
    aurc = sum(risk for _, risk in points) / len(points)
    return points, aurc


def selective_accuracy(
    errors: Sequence[int],
    reject_scores: Sequence[float],
    coverage: float,
) -> dict[str, float]:
    """Accuracy obtained by abstaining on the most suspicious rows."""
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must lie in (0, 1]")
    keep = max(1, int(round(coverage * len(errors))))
    order = sorted(range(len(errors)), key=lambda i: reject_scores[i])[:keep]
    accepted_errors = sum(int(bool(errors[i])) for i in order)
    return {
        "coverage": keep / len(errors),
        "selective_accuracy": 1.0 - accepted_errors / keep,
        "accepted": keep,
    }


def oracle_review_curve(
    errors: Sequence[int],
    reject_scores: Sequence[float],
) -> tuple[list[tuple[float, float, int]], float]:
    """Ideal-review upper bound: the reviewed prefix is resolved perfectly.

    This is an oracle simulation, not a deployable number. It answers "how much
    error would perfect review remove at this review budget", which bounds the
    value of any real reviewer.

    Returns:
        ``(points, mean_residual_risk)`` with points
        ``(review_rate, residual_risk, residual_errors)``.
    """
    if len(errors) != len(reject_scores):
        raise ValueError("errors and reject_scores must have equal length")
    if not errors:
        raise ValueError("empty curve")
    order = sorted(range(len(errors)), key=lambda i: -reject_scores[i])
    total = sum(int(bool(e)) for e in errors)
    points: list[tuple[float, float, int]] = []
    seen = 0
    for k, index in enumerate(order, start=1):
        seen += int(bool(errors[index]))
        remaining = total - seen
        accepted = len(errors) - k
        risk = remaining / accepted if accepted else 0.0
        points.append((k / len(errors), risk, remaining))
    mean_risk = sum(p[1] for p in points) / len(points)
    return points, mean_risk


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def _softmax(logits: Sequence[float]) -> list[float]:
    if not logits:
        raise ValueError("empty logits")
    if any(not math.isfinite(v) for v in logits):
        raise ValueError("non-finite logits")
    peak = max(logits)
    exps = [math.exp(v - peak) for v in logits]
    total = sum(exps)
    return [v / total for v in exps]


def apply_temperature(logits: Sequence[float], temperature: float) -> list[float]:
    """Temperature-scale a logit vector. ``temperature == 1`` is identity."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive")
    return _softmax([v / temperature for v in logits])


def negative_log_likelihood(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    temperature: float = 1.0,
) -> float:
    """Mean NLL of the labelled class under temperature-scaled softmax."""
    if len(logits) != len(labels):
        raise ValueError("logits and labels must have equal length")
    if not logits:
        raise ValueError("empty set")
    total = 0.0
    for row, label in zip(logits, labels):
        probabilities = apply_temperature(row, temperature)
        if not 0 <= label < len(probabilities):
            raise ValueError("label out of range")
        total -= math.log(max(probabilities[label], 1e-12))
    return total / len(logits)


def fit_temperature(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    lower: float = 0.05,
    upper: float = 20.0,
    iterations: int = 80,
) -> dict[str, float]:
    """Fit a single temperature by golden-section search on NLL.

    One scalar cannot fix a systematically overconfident model, but it is a
    cheap, monotone calibration baseline and a useful diagnostic. Report the
    NLL before and after; never report accuracy as if calibration changed it -
    temperature scaling does not change argmax.
    """
    if len(logits) != len(labels) or not logits:
        raise ValueError("logits and labels must be non-empty and aligned")
    if not 0 < lower < upper:
        raise ValueError("invalid search interval")

    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = math.log(lower), math.log(upper)
    c = b - ratio * (b - a)
    d = a + ratio * (b - a)
    f_c = negative_log_likelihood(logits, labels, math.exp(c))
    f_d = negative_log_likelihood(logits, labels, math.exp(d))
    for _ in range(iterations):
        if f_c < f_d:
            b, d, f_d = d, c, f_c
            c = b - ratio * (b - a)
            f_c = negative_log_likelihood(logits, labels, math.exp(c))
        else:
            a, c, f_c = c, d, f_d
            d = a + ratio * (b - a)
            f_d = negative_log_likelihood(logits, labels, math.exp(d))
        if b - a < 1e-6:
            break

    best = math.exp((a + b) / 2.0)
    return {
        "temperature": best,
        "nll_before": negative_log_likelihood(logits, labels, 1.0),
        "nll_after": negative_log_likelihood(logits, labels, best),
    }


def reliability_bins(
    confidences: Sequence[float],
    correct: Sequence[int | bool],
    *,
    n_bins: int = 15,
) -> list[dict[str, float]]:
    """Equal-width reliability bins over predicted confidence."""
    if len(confidences) != len(correct):
        raise ValueError("confidences and correctness must have equal length")
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    bins: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]
    for confidence, hit in zip(confidences, correct):
        confidence = _check_probability(confidence, "confidence")
        index = min(int(confidence * n_bins), n_bins - 1)
        bins[index].append((confidence, float(bool(hit))))
    out = []
    for index, bucket in enumerate(bins):
        if not bucket:
            continue
        out.append(
            {
                "bin_lower": index / n_bins,
                "bin_upper": (index + 1) / n_bins,
                "n": len(bucket),
                "mean_confidence": sum(c for c, _ in bucket) / len(bucket),
                "accuracy": sum(h for _, h in bucket) / len(bucket),
            }
        )
    return out


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[int | bool],
    *,
    n_bins: int = 15,
) -> float:
    """Weighted mean gap between confidence and accuracy (ECE)."""
    if not confidences:
        raise ValueError("empty calibration set")
    bins = reliability_bins(confidences, correct, n_bins=n_bins)
    n = len(confidences)
    return sum(b["n"] / n * abs(b["mean_confidence"] - b["accuracy"]) for b in bins)
