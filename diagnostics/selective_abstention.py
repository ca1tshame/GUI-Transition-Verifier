"""Risk-aware selective verification, calibration, and ranking invariance audits.

A verifier that is forced to answer every transition will answer some of them
wrongly. A verifier that can say "I am not sure" turns a fraction of those
errors into review work. This module implements that second behaviour and, just
as importantly, scores it honestly using only the Python standard library.

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

Calibration vs cross-sample ranking
-----------------------------------
Multiclass Temperature Scaling divides all class logits in a row by a shared
scalar ``T > 0``. That preserves single-sample ``argmax`` ordering within each
row, yet **can alter cross-sample ``p1`` ROC-AUC ranking** because the softmax
denominator ``sum_k exp(z_k / T)`` couples ``p1`` nonlinearly to how competitor
mass is distributed across classes ``2..K``. By contrast, binary Platt scaling
``sigmoid(a * logit(p1) + b)`` on ``p1`` alone is a strictly increasing map
whenever ``a > 0``, strictly preserving cross-sample ``p1`` ROC-AUC ranking.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Sequence

__all__ = [
    "Decision",
    "PROGRESS_LABELS",
    "apply_binary_platt",
    "apply_temperature",
    "audit_temperature_vs_platt_auc",
    "binary_platt",
    "brier_binary",
    "brier_multiclass",
    "decide",
    "evaluate_decisions",
    "expected_calibration_error",
    "fit_temperature",
    "negative_log_likelihood",
    "oracle_review_curve",
    "progress_feedback",
    "reliability_bins",
    "risk_coverage_curve",
    "risk_scores",
    "roc_auc",
    "selective_accuracy",
]

PROGRESS_LABELS = ("progress", "no_progress", "regression", "recovery")
PROGRESS_REWARDS = (1.0, 0.0, -1.0, 1.0)
_LOG_EPS = 1e-12


class Decision(str, Enum):
    ACCEPT = "accept"
    FLAG = "flag"
    REVIEW = "review"


def _check_probability(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite probability, got {value!r}")
    fval = float(value)
    if not math.isfinite(fval) or not 0.0 <= fval <= 1.0:
        raise ValueError(f"{name} must be a finite probability, got {value!r}")
    return fval


def _check_binary_label(value: int | bool, name: str = "label") -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return value
    raise ValueError(f"{name} must be 0 or 1, got {value!r}")


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
    _check_probability(confidence_threshold, "confidence_threshold")
    _check_probability(risk_threshold, "risk_threshold")
    if risk >= risk_threshold:
        return Decision.FLAG
    if confidence < confidence_threshold:
        return Decision.REVIEW
    return Decision.ACCEPT


def _simplex(probabilities: Sequence[float], size: int, name: str) -> tuple[float, ...]:
    if len(probabilities) != size:
        raise ValueError(f"{name} must contain {size} probabilities")
    values: list[float] = []
    for p in probabilities:
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            raise ValueError(f"{name} must be finite probabilities")
        fp = float(p)
        if not math.isfinite(fp) or fp < 0.0 or fp > 1.0:
            raise ValueError(f"{name} must be finite probabilities")
        values.append(fp)
    if abs(sum(values) - 1.0) > 1e-4:
        raise ValueError(f"{name} must sum to one")
    return tuple(values)


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
    invalid = set(normalised) - {
        Decision.ACCEPT.value,
        Decision.FLAG.value,
        Decision.REVIEW.value,
    }
    if invalid:
        raise ValueError(f"unknown decision value(s): {sorted(invalid)}")

    accepted = [i for i, d in enumerate(normalised) if d == Decision.ACCEPT.value]
    flagged = [i for i, d in enumerate(normalised) if d == Decision.FLAG.value]
    reviewed = [i for i, d in enumerate(normalised) if d == Decision.REVIEW.value]

    correct = sum(1 for i in accepted if predictions[i] == labels[i])
    accept_errors = len(accepted) - correct
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
        "accept_errors": accept_errors,
        "accepted_error_rate": accept_errors / len(accepted) if accepted else None,
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
# Calibration and Brier / ROC-AUC diagnostics
# ---------------------------------------------------------------------------


def _softmax(logits: Sequence[float]) -> list[float]:
    if not logits:
        raise ValueError("empty logits")
    clean: list[float] = []
    for v in logits:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise ValueError("non-finite logits")
        clean.append(float(v))
    peak = max(clean)
    exps = [math.exp(v - peak) for v in clean]
    total = sum(exps)
    return [v / total for v in exps]


def _sigmoid(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _logit_prob(prob: float) -> float:
    p = _check_probability(prob, "score")
    return math.log(max(p, _LOG_EPS)) - math.log(max(1.0 - p, _LOG_EPS))


def apply_temperature(logits: Sequence[float], temperature: float) -> list[float]:
    """Temperature-scale a logit vector. ``temperature == 1`` is identity."""
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("temperature must be positive")
    temp = float(temperature)
    if not math.isfinite(temp) or temp <= 0.0:
        raise ValueError("temperature must be positive")
    return _softmax([float(v) / temp for v in logits])


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
        if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < len(probabilities):
            raise ValueError("label out of range")
        total -= math.log(max(probabilities[label], _LOG_EPS))
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
    cheap, monotone within-row calibration baseline and a useful diagnostic.
    Report the NLL before and after; never report accuracy as if calibration
    changed it - temperature scaling preserves single-sample argmax.
    """
    if len(logits) != len(labels) or not logits:
        raise ValueError("logits and labels must be non-empty and aligned")
    if not 0.0 < lower < upper:
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


def brier_multiclass(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
) -> float:
    """Mean multiclass Brier score ``(1/N) sum_i sum_k (p_{i,k} - y_{i,k})^2``."""
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have equal length")
    if not probabilities:
        raise ValueError("empty evaluation set")
    num_classes = len(probabilities[0])
    if num_classes < 2:
        raise ValueError("multiclass Brier requires at least two classes")

    total = 0.0
    for row, label in zip(probabilities, labels):
        vals = _simplex(row, num_classes, "probabilities")
        if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < num_classes:
            raise ValueError(f"label out of range: {label!r}")
        total += sum((pi - (1.0 if idx == label else 0.0)) ** 2 for idx, pi in enumerate(vals))
    return total / len(labels)


def brier_binary(
    probabilities: Sequence[float],
    labels: Sequence[int | bool],
) -> float:
    """Mean binary Brier score ``(1/N) sum_i (p_i - y_i)^2``."""
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have equal length")
    if not probabilities:
        raise ValueError("empty evaluation set")

    total = 0.0
    for prob, label in zip(probabilities, labels):
        p = _check_probability(prob, "probability")
        y = float(_check_binary_label(label))
        total += (p - y) ** 2
    return total / len(labels)


def binary_platt(
    scores: Sequence[float],
    labels: Sequence[int | bool],
    iters: int = 400,
    lr: float = 0.5,
    l2: float = 1e-3,
) -> tuple[float, float]:
    """Fit 1D binary Platt scaling ``P(y=1) = sigmoid(a * logit(s) + b)``.

    Uses deterministic pure-stdlib batch gradient descent with weak L2
    regularisation on the slope ``a``. Whenever ``a > 0``, ``apply_binary_platt``
    is strictly monotone in ``s`` on ``(0, 1)`` and preserves ROC-AUC ranking.

    Returns:
        ``(a, b)`` slope and intercept in logit space.
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have equal length")
    if not scores:
        raise ValueError("empty calibration set")
    if isinstance(iters, bool) or not isinstance(iters, int) or iters < 1:
        raise ValueError("iters must be a positive integer")
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0.0:
        raise ValueError("lr must be positive")
    if isinstance(l2, bool) or not isinstance(l2, (int, float)) or not math.isfinite(l2) or l2 < 0.0:
        raise ValueError("l2 must be non-negative")

    x = [_logit_prob(s) for s in scores]
    y = [float(_check_binary_label(lbl)) for lbl in labels]
    a, b = 1.0, 0.0
    step = float(lr)
    reg = float(l2)
    n = len(y)
    for _ in range(iters):
        ga = 0.0
        gb = 0.0
        for xi, yi in zip(x, y):
            p = _sigmoid(a * xi + b)
            diff = p - yi
            ga += diff * xi
            gb += diff
        ga = ga / n + reg * a
        gb = gb / n
        a -= step * ga
        b -= step * gb
        step *= 0.995
    return a, b


def apply_binary_platt(
    scores: Sequence[float],
    a: float,
    b: float,
) -> list[float]:
    """Apply fitted Platt parameters ``(a, b)`` to binary probability scores."""
    if (
        isinstance(a, bool)
        or isinstance(b, bool)
        or not isinstance(a, (int, float))
        or not isinstance(b, (int, float))
    ):
        raise ValueError("Platt parameters a and b must be finite numbers")
    fa, fb = float(a), float(b)
    if not (math.isfinite(fa) and math.isfinite(fb)):
        raise ValueError("Platt parameters a and b must be finite numbers")
    return [_sigmoid(fa * _logit_prob(s) + fb) for s in scores]


def roc_auc(
    scores: Sequence[float],
    binary_labels: Sequence[int | bool],
) -> float:
    """Exact tie-aware Mann-Whitney midrank ROC-AUC for binary targets."""
    if len(scores) != len(binary_labels):
        raise ValueError("scores and binary_labels must have equal length")
    if not scores:
        raise ValueError("empty evaluation set")

    pos: list[float] = []
    neg: list[float] = []
    for s, lbl in zip(scores, binary_labels):
        if isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(float(s)):
            raise ValueError(f"non-finite score: {s!r}")
        y = _check_binary_label(lbl, "binary_label")
        if y == 1:
            pos.append(float(s))
        else:
            neg.append(float(s))

    if not pos or not neg:
        raise ValueError("roc_auc requires at least one positive and one negative sample")

    paired = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank_sum = 0.0
    i = 0
    while i < len(paired):
        j = i + 1
        while j < len(paired) and paired[j][0] == paired[i][0]:
            j += 1
        mid_rank = (i + 1 + j) / 2.0
        rank_sum += sum(flag for _, flag in paired[i:j]) * mid_rank
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def audit_temperature_vs_platt_auc(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int] | None = None,
    *,
    binary_labels: Sequence[int | bool] | None = None,
    temperature: float | None = None,
    platt_params: tuple[float, float] | None = None,
    risk_index: int = 0,
    fit_logits: Sequence[Sequence[float]] | None = None,
    fit_labels: Sequence[int] | None = None,
    tol: float = 1e-12,
) -> dict[str, object]:
    """Audit how multiclass Temperature Scaling vs binary Platt affect ``p1`` ROC-AUC.

    Within each sample, dividing logits by ``T > 0`` is strictly monotone, so
    ``argmax`` is invariant. Across samples, however,
    ``p1(T) = 1 / (1 + sum_{k != 0} exp((z_k - z_0) / T))`` depends on how
    non-target logit mass is concentrated vs spread across classes ``2..K``.
    Two samples can therefore reverse their ``p1`` rank order when ``T != 1``,
    shifting cross-sample ``p1`` ROC-AUC. Positive-slope (``a > 0``) binary
    Platt scaling ``sigmoid(a * logit(p1) + b)`` transforms ``p1`` alone by a
    strictly increasing function and preserves cross-sample ``p1`` ranking.

    Args:
        logits: ``[N, K]`` raw multiclass logits evaluated by the audit.
        labels: optional ``[N]`` multiclass gold indices ``0..K-1`` (class
            ``risk_index`` is positive when ``binary_labels`` is omitted).
        binary_labels: optional explicit ``{0, 1}`` target indicators for class
            ``risk_index``. Overrides ``labels == risk_index`` for binary metrics.
        temperature: explicit positive scalar ``T``. When ``None``, fitted via
            :func:`fit_temperature` on ``(fit_logits or logits, fit_labels or labels)``.
        platt_params: optional ``(a, b)`` tuple. When ``None``, fitted via
            :func:`binary_platt` on ``fit_logits``/``fit_labels`` if supplied,
            otherwise on ``(raw_p1, binary_targets)``.
        risk_index: class index corresponding to ``p1`` (default ``0``).
        fit_logits: optional separate fit-split logits for leak-free fitting.
        fit_labels: optional separate fit-split multiclass labels.
        tol: numerical tolerance for ROC-AUC equality checks.
    """
    if not logits:
        raise ValueError("empty logits")
    num_classes = len(logits[0])
    if num_classes < 2:
        raise ValueError("logits must have at least two classes")
    if isinstance(risk_index, bool) or not isinstance(risk_index, int) or not 0 <= risk_index < num_classes:
        raise ValueError("risk_index out of range")

    clean_logits: list[list[float]] = []
    for row in logits:
        if len(row) != num_classes:
            raise ValueError("all logit rows must have identical class count")
        clean_row: list[float] = []
        for v in row:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                raise ValueError("non-finite logits")
            clean_row.append(float(v))
        clean_logits.append(clean_row)

    if labels is not None:
        if len(labels) != len(clean_logits):
            raise ValueError("logits and labels must have equal length")
        for lbl in labels:
            if isinstance(lbl, bool) or not isinstance(lbl, int) or not 0 <= lbl < num_classes:
                raise ValueError("label out of range")

    if binary_labels is not None:
        if len(binary_labels) != len(clean_logits):
            raise ValueError("logits and binary_labels must have equal length")
        eval_binary = [_check_binary_label(lbl, "binary_label") for lbl in binary_labels]
    elif labels is not None:
        eval_binary = [1 if lbl == risk_index else 0 for lbl in labels]
    else:
        raise ValueError("either labels or binary_labels must be provided")

    if temperature is not None:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ValueError("temperature must be positive")
        temp_used = float(temperature)
        if not math.isfinite(temp_used) or temp_used <= 0.0:
            raise ValueError("temperature must be positive")
    else:
        src_logits = fit_logits if fit_logits is not None else clean_logits
        src_labels = fit_labels if fit_labels is not None else labels
        if src_labels is None:
            raise ValueError("temperature must be provided when multiclass labels are omitted")
        temp_used = fit_temperature(src_logits, src_labels)["temperature"]

    raw_probs = [apply_temperature(row, 1.0) for row in clean_logits]
    ts_probs = [apply_temperature(row, temp_used) for row in clean_logits]

    raw_argmax = [max(range(num_classes), key=row.__getitem__) for row in clean_logits]
    ts_argmax = [max(range(num_classes), key=probs.__getitem__) for probs in ts_probs]
    argmax_preserved = raw_argmax == ts_argmax

    raw_p1 = [probs[risk_index] for probs in raw_probs]
    ts_p1 = [probs[risk_index] for probs in ts_probs]

    if platt_params is not None:
        platt_a, platt_b = float(platt_params[0]), float(platt_params[1])
    elif fit_logits is not None and fit_labels is not None:
        fit_raw_p1 = [apply_temperature(row, 1.0)[risk_index] for row in fit_logits]
        fit_bin = [1 if lbl == risk_index else 0 for lbl in fit_labels]
        platt_a, platt_b = binary_platt(fit_raw_p1, fit_bin)
    else:
        platt_a, platt_b = binary_platt(raw_p1, eval_binary)

    platt_p1 = apply_binary_platt(raw_p1, platt_a, platt_b)

    raw_auc_val = roc_auc(raw_p1, eval_binary)
    ts_auc_val = roc_auc(ts_p1, eval_binary)
    platt_auc_val = roc_auc(platt_p1, eval_binary)

    pair_inversions = 0
    for i in range(len(raw_p1)):
        for j in range(i + 1, len(raw_p1)):
            if (raw_p1[i] - raw_p1[j]) * (ts_p1[i] - ts_p1[j]) < 0.0:
                pair_inversions += 1

    return {
        "n": len(clean_logits),
        "risk_index": risk_index,
        "temperature": temp_used,
        "platt_a": platt_a,
        "platt_b": platt_b,
        "raw_p1": raw_p1,
        "ts_p1": ts_p1,
        "platt_p1": platt_p1,
        "raw_roc_auc": raw_auc_val,
        "ts_roc_auc": ts_auc_val,
        "platt_roc_auc": platt_auc_val,
        "ts_auc_delta": ts_auc_val - raw_auc_val,
        "platt_auc_delta": platt_auc_val - raw_auc_val,
        "argmax_preserved": argmax_preserved,
        "ts_auc_changed": abs(ts_auc_val - raw_auc_val) > tol,
        "ts_pair_inversions": pair_inversions,
        "platt_order_preserving": (
            platt_a > 0.0 and abs(platt_auc_val - raw_auc_val) <= tol
        ),
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
