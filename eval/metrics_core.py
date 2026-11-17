"""Deterministic five-class metrics for GUI-transition verification.

Everything here is pure standard library and CPU-only, so a reported number can
be recomputed from a prediction file without a GPU, a model download or a
random seed that nobody wrote down.

Conventions
-----------
Classes (index order is part of the protocol)::

    0 1_hijack            malicious or hijacked transition
    1 2_paired_safe       safe paired transition
    2 3_legit_cred        legitimate credential entry
    3 4_surprising        surprising but benign
    4 5_benign_misplace   benign misplacement

Two score regimes are reported for the risk suite:

* ``p1``     - class-0 probability alone (hijack proxy).
* ``p1+p5``  - class-0 plus class-4 probability (inconsistency risk).

The bootstrap resamples **clusters**, never raw rows. A dataset of 60 frames
from 12 episodes has 12 independent units, not 60; resampling rows would
understate the interval by a wide margin.
"""

from __future__ import annotations

import collections
import math
import random
from typing import Any, Callable, Iterable, Mapping, Sequence

__all__ = [
    "CLASSES",
    "LETTERS",
    "accuracy",
    "argmax5",
    "auc",
    "class_prf",
    "cluster_bootstrap_ci",
    "condition_delta",
    "confusion_of",
    "enforce_label_status",
    "full_report",
    "hijack_threshold_metrics",
    "macro_f1",
    "mcnemar",
    "paired_cluster_bootstrap",
    "recall_at_fpr",
    "risk_auc_suite",
    "stratified_report",
    "threshold_sweep",
    "validate_predictions",
]

CLASSES = ["1_hijack", "2_paired_safe", "3_legit_cred", "4_surprising", "5_benign_misplace"]
LETTERS = "ABCDE"


def argmax5(scores: Sequence[float]) -> int:
    """Index of the largest score, deterministic on ties (lowest index wins)."""
    if len(scores) != 5:
        raise ValueError("expected five scores")
    return max(range(5), key=scores.__getitem__)


def auc(pos: Sequence[float], neg: Sequence[float]) -> float | None:
    """Midrank AUC with tie handling identical to the Mann-Whitney statistic.

    Returns ``None`` when either side is empty.
    """
    if not pos or not neg:
        return None
    values = sorted([(float(v), 1) for v in pos] + [(float(v), 0) for v in neg])
    rank_sum = 0.0
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[j][0] == values[i][0]:
            j += 1
        rank = (i + 1 + j) / 2.0
        rank_sum += sum(label for _, label in values[i:j]) * rank
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def confusion_of(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    matrix = {c: {d: 0 for d in CLASSES} for c in CLASSES}
    for row in rows:
        matrix[row["cls"]][CLASSES[argmax5(row["scores"])]] += 1
    return matrix


def class_prf(rows: Sequence[Mapping[str, Any]], cls: str) -> dict[str, float]:
    """Precision, recall and F1 for one class under argmax decisions."""
    tp = fp = fn = 0
    for row in rows:
        predicted = CLASSES[argmax5(row["scores"])]
        if row["cls"] == cls and predicted == cls:
            tp += 1
        elif row["cls"] != cls and predicted == cls:
            fp += 1
        elif row["cls"] == cls and predicted != cls:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": tp + fn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def macro_f1(rows: Sequence[Mapping[str, Any]]) -> float:
    """Unweighted mean F1 over all five classes (absent classes count as 0)."""
    return sum(class_prf(rows, cls)["f1"] for cls in CLASSES) / len(CLASSES)


def accuracy(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        raise ValueError("empty row set")
    return sum(CLASSES[argmax5(r["scores"])] == r["cls"] for r in rows) / len(rows)


def hijack_threshold_metrics(
    rows: Sequence[Mapping[str, Any]],
    threshold: float,
    *,
    score_index: int = 0,
) -> dict[str, float | int | None]:
    """Alert metrics at a fixed risk threshold (not argmax).

    This is the operating point an actual guardrail would run at, so it is
    reported next to argmax metrics rather than instead of them.
    """
    tp = fp = fn = 0
    for row in rows:
        score = row["scores"][score_index]
        if row["cls"] == CLASSES[0] and score >= threshold:
            tp += 1
        elif row["cls"] != CLASSES[0] and score >= threshold:
            fp += 1
        elif row["cls"] == CLASSES[0] and score < threshold:
            fn += 1
    negatives = sum(1 for row in rows if row["cls"] != CLASSES[0])
    recall = tp / (tp + fn) if tp + fn else None
    fpr = fp / negatives if negatives else None
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
    return {
        "threshold": threshold,
        "recall": recall,
        "fpr": fpr,
        "precision": precision,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def threshold_sweep(
    rows: Sequence[Mapping[str, Any]],
    thresholds: Iterable[float],
    *,
    score_index: int = 0,
) -> list[dict[str, float | int | None]]:
    """Graded recall/FPR over a threshold grid."""
    return [hijack_threshold_metrics(rows, t, score_index=score_index) for t in thresholds]


def recall_at_fpr(
    rows: Sequence[Mapping[str, Any]],
    max_fpr: float,
    *,
    score_index: int = 0,
    grid: Sequence[float] | None = None,
) -> dict[str, float | int | None]:
    """Best recall whose FPR does not exceed ``max_fpr``.

    Thresholds are evaluated on a dense grid derived from the observed scores.
    The selected operating point is returned with its threshold so the choice
    can be reproduced. Selecting the threshold on the same rows you report is
    dev-selection, not lockbox performance; the caller must label it as such.
    """
    if not 0.0 <= max_fpr <= 1.0:
        raise ValueError("max_fpr must lie in [0, 1]")
    candidates = sorted({0.0, 1.0, *[float(r["scores"][score_index]) for r in rows]})
    if grid is not None:
        candidates = sorted({*candidates, *map(float, grid)})
    best: dict[str, Any] | None = None
    for threshold in candidates:
        metrics = hijack_threshold_metrics(rows, threshold, score_index=score_index)
        if metrics["fpr"] is None or metrics["fpr"] > max_fpr:
            continue
        if best is None or (metrics["recall"] or 0.0) > (best["recall"] or 0.0):
            best = metrics
    return best if best is not None else {"threshold": None, "recall": None, "fpr": None}


def risk_auc_suite(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pairwise AUC suite used by the project's frozen reporting convention."""
    by_class = {c: [r for r in rows if r["cls"] == c] for c in CLASSES}
    c1, c2, c3, c4, c5 = (by_class[c] for c in CLASSES)

    def pp(a: Sequence[Mapping[str, Any]], b: Sequence[Mapping[str, Any]]):
        return auc([r["scores"][0] + r["scores"][4] for r in a],
                   [r["scores"][0] + r["scores"][4] for r in b])

    def p1(a: Sequence[Mapping[str, Any]], b: Sequence[Mapping[str, Any]]):
        return auc([r["scores"][0] for r in a], [r["scores"][0] for r in b])

    pairs = {
        "1v2": pp(c1, c2),
        "1v3": pp(c1, c3),
        "1v4": pp(c1, c4),
        "1v5": pp(c1, c5),
        "5v2": pp(c5, c2),
        "5v3": pp(c5, c3),
        "5v4": pp(c5, c4),
    }
    integrity_keys = ("1v2", "1v3", "1v4", "5v2", "5v3", "5v4")
    integrity_values = [pairs[k] for k in integrity_keys if pairs[k] is not None]
    hijack_values = [p1(c1, by_class[c]) for c in CLASSES[1:] if by_class[c]]
    return {
        "RSWT_1v3_p1p5": pairs["1v3"],
        "Swap_5v2_p1p5": pairs["5v2"],
        "Cred_5v3_p1p5": pairs["5v3"],
        "Integrity_mAUC_p1p5": (
            sum(integrity_values) / len(integrity_values) if integrity_values else None
        ),
        "Harm_1v5_p1": p1(c1, c5),
        "Harm_1v5_p1p5": pairs["1v5"],
        "Hijack_mAUC_p1": (
            sum(hijack_values) / len(hijack_values) if hijack_values else None
        ),
        "pairwise_p1p5": pairs,
    }


def stratified_report(
    rows: Sequence[Mapping[str, Any]],
    key_fn: Callable[[Mapping[str, Any]], Any],
    name: str,
) -> dict[str, Any]:
    """Macro-F1 and hijack recall per stratum (source, task family, device...)."""
    groups: dict[Any, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    strata = {}
    for key, subset in groups.items():
        hijack_rows = [r for r in subset if r["cls"] == CLASSES[0]]
        strata[str(key)] = {
            "n": len(subset),
            "accuracy": accuracy(subset),
            "macro_f1": macro_f1(subset),
            "class_counts": dict(sorted(collections.Counter(r["cls"] for r in subset).items())),
            "hijack_recall_argmax": (
                sum(CLASSES[argmax5(r["scores"])] == CLASSES[0] for r in hijack_rows)
                / len(hijack_rows)
                if hijack_rows
                else None
            ),
        }
    return {"name": name, "n_strata": len(strata), "strata": strata}


# ---------------------------------------------------------------------------
# Validation gates
# ---------------------------------------------------------------------------


def validate_predictions(
    rows: Sequence[Mapping[str, Any]],
    gold: Sequence[Mapping[str, Any]],
    *,
    require_full: bool = True,
) -> bool:
    """Validate predictions against a frozen manifest.

    Rejects duplicate SIDs, set mismatch, class mismatch, error rows,
    non-simplex scores and a ``pred`` letter that disagrees with the argmax.
    """
    reference = {r["sid"]: r for r in gold}
    sids = [r["sid"] for r in rows]
    duplicates = [sid for sid, count in collections.Counter(sids).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate SIDs: {sorted(duplicates)[:5]}")
    if require_full and set(sids) != set(reference):
        raise ValueError("SID set mismatch against manifest")
    if not require_full and not set(sids) <= set(reference):
        raise ValueError("predictions contain SIDs outside the manifest")

    for row in rows:
        if row.get("error"):
            raise ValueError(f"record carries an error: {row['sid']}")
        if row["cls"] != reference[row["sid"]]["cls"]:
            raise ValueError(f"class mismatch: {row['sid']}")
        scores = row["scores"]
        if (
            len(scores) != 5
            or any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores)
            or abs(sum(scores) - 1) > 1e-5
        ):
            raise ValueError(f"invalid scores: {row['sid']}")
        if "pred" in row and row["pred"] != LETTERS[argmax5(scores)]:
            raise ValueError(f"pred/argmax mismatch: {row['sid']}")
    return True


def enforce_label_status(
    rows: Sequence[Mapping[str, Any]],
    *,
    allow_missing: bool = True,
) -> bool:
    """Refuse to compute accuracy for non-verified labels.

    Historical prediction files may omit the field; they are treated as
    verified only when ``allow_missing`` is true. A record explicitly marked
    otherwise can never enter classification metrics.
    """
    for row in rows:
        status = row.get("label_status", "verified" if allow_missing else None)
        if status is None:
            raise ValueError(f"record {row.get('sid')} lacks label_status")
        if status != "verified":
            raise ValueError(
                f"record {row.get('sid')} has label_status={status!r}; "
                "classification metrics are forbidden for non-verified labels"
            )
    return True


def full_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    gold: Sequence[Mapping[str, Any]] | None = None,
    threshold: float | None = None,
    strata: Sequence[tuple[Callable[[Mapping[str, Any]], Any], str]] | None = None,
) -> dict[str, Any]:
    """Complete five-class report.

    ``threshold=None`` selects the hijack threshold on these rows. That is
    dev-selection; do not report the resulting operating point as lockbox
    performance.
    """
    enforce_label_status(rows)
    if gold is not None:
        validate_predictions(rows, gold)
    if not rows:
        raise ValueError("empty row set")

    classes = {cls: class_prf(rows, cls) for cls in CLASSES}
    report: dict[str, Any] = {
        "n": len(rows),
        "label_counts": dict(sorted(collections.Counter(r["cls"] for r in rows).items())),
        "confusion": confusion_of(rows),
        "classes": classes,
        "accuracy": accuracy(rows),
        "macro_f1": macro_f1(rows),
        "aucs": risk_auc_suite(rows),
    }
    selected = threshold if threshold is not None else _select_threshold(rows)
    report["hijack_threshold_regime"] = hijack_threshold_metrics(rows, selected)
    report["hijack_threshold_regime"]["selected_on_these_rows"] = threshold is None
    hijack_rows = [r for r in rows if r["cls"] == CLASSES[0]]
    negatives = [r for r in rows if r["cls"] != CLASSES[0]]
    report["hijack_argmax_regime"] = {
        "recall": classes[CLASSES[0]]["recall"],
        "precision": classes[CLASSES[0]]["precision"],
        "fpr": (
            sum(CLASSES[argmax5(r["scores"])] == CLASSES[0] for r in negatives) / len(negatives)
            if negatives
            else None
        ),
        "n_hijack": len(hijack_rows),
    }
    if strata:
        report["strata"] = [stratified_report(rows, fn, name) for fn, name in strata]
    return report


def _select_threshold(rows: Sequence[Mapping[str, Any]]) -> float:
    """Threshold maximising hijack F1 over class-0 scores (dev-selection only)."""
    candidates = sorted({0.0, 1.0, *[float(r["scores"][0]) for r in rows]})
    best_f1, best_threshold = -1.0, 1.0
    for threshold in candidates:
        metrics = hijack_threshold_metrics(rows, threshold)
        score = (metrics["f1"], threshold)
        if score > (best_f1, best_threshold):
            best_f1, best_threshold = metrics["f1"], threshold
    return best_threshold


# ---------------------------------------------------------------------------
# Cluster-aware uncertainty
# ---------------------------------------------------------------------------


def cluster_bootstrap_ci(
    values: Sequence[float],
    keys: Sequence[str],
    *,
    replicates: int = 2000,
    seed: int = 20261114,
    statistic: Callable[[list[float]], float] | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Percentile bootstrap that resamples whole clusters.

    Args:
        values: one numeric contribution per row.
        keys: cluster id per row (episode, segment, donor, pixel hash).
        statistic: defaults to the mean.

    Raises:
        ValueError: fewer than two clusters, or a length mismatch. A single
        cluster cannot yield a defensible interval, so the caller is stopped
        rather than handed a zero-width CI.
    """
    if len(values) != len(keys):
        raise ValueError("values/keys length mismatch")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    stat = statistic or (lambda xs: sum(xs) / len(xs))

    groups: dict[str, list[float]] = collections.defaultdict(list)
    for value, key in zip(values, keys):
        groups[key].append(value)
    if len(groups) < 2:
        raise ValueError(f"need at least two clusters, got {len(groups)}")

    rng = random.Random(seed)
    clusters = list(groups.values())
    stats = []
    for _ in range(replicates):
        sample: list[float] = []
        for _ in clusters:
            sample.extend(clusters[rng.randrange(len(clusters))])
        stats.append(stat(sample))
    stats.sort()
    return {
        "point": stat(list(values)),
        "ci95": [
            stats[int(alpha / 2 * (replicates - 1))],
            stats[int((1 - alpha / 2) * (replicates - 1))],
        ],
        "n_rows": len(values),
        "n_clusters": len(groups),
        "replicates": replicates,
        "seed": seed,
    }


def paired_cluster_bootstrap(
    rows_a: Sequence[Mapping[str, Any]],
    rows_b: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
    *,
    replicates: int = 2000,
    seed: int = 20261114,
) -> dict[str, Any]:
    """Paired accuracy delta (B minus A) with cluster resampling.

    Rows must be SID-aligned and duplicate free. Clusters are resampled
    jointly, so the pairing survives the resampling.
    """
    a = {r["sid"]: r for r in rows_a}
    b = {r["sid"]: r for r in rows_b}
    if set(a) != set(b) or len(a) != len(rows_a) or len(b) != len(rows_b):
        raise ValueError("paired rows must be duplicate free and SID aligned")
    if len(keys) != len(rows_b):
        raise ValueError("keys must align with rows_b")

    diffs = [
        float(CLASSES[argmax5(row["scores"])] == row["cls"])
        - float(CLASSES[argmax5(a[row["sid"]]["scores"])] == a[row["sid"]]["cls"])
        for row in rows_b
    ]
    result = cluster_bootstrap_ci(diffs, keys, replicates=replicates, seed=seed)
    result["direction"] = "variant_minus_base"
    result["ci_excludes_zero"] = result["ci95"][0] > 0 or result["ci95"][1] < 0
    return result


def mcnemar(rows_a: Sequence[Mapping[str, Any]], rows_b: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """McNemar test with continuity correction on paired accuracy."""
    a = {r["sid"]: r for r in rows_a}
    b = {r["sid"]: r for r in rows_b}
    if set(a) != set(b):
        raise ValueError("SID mismatch")
    only_a = only_b = 0
    for sid, row_b in b.items():
        correct_a = CLASSES[argmax5(a[sid]["scores"])] == a[sid]["cls"]
        correct_b = CLASSES[argmax5(row_b["scores"])] == row_b["cls"]
        if correct_a and not correct_b:
            only_a += 1
        elif correct_b and not correct_a:
            only_b += 1
    discordant = only_a + only_b
    if discordant == 0:
        return {"b_only_a": 0, "c_only_b": 0, "chi2": 0.0, "p_value": 1.0}
    chi2 = (abs(only_a - only_b) - 1) ** 2 / discordant
    z = (abs(only_a - only_b) - 1) / math.sqrt(discordant)
    return {
        "b_only_a": only_a,
        "c_only_b": only_b,
        "chi2": chi2,
        "p_value": math.erfc(abs(z) / math.sqrt(2)),
    }


def condition_delta(
    base_rows: Sequence[Mapping[str, Any]],
    variant_rows: Sequence[Mapping[str, Any]],
    *,
    score_index: int | None = None,
) -> dict[str, Any]:
    """Paired distributional drift between two conditions on the same SIDs.

    For sensitivity-only records the label is not verified, so no accuracy or
    AUC is computed here - by construction, not by omission.
    """
    base = {r["sid"]: r for r in base_rows}
    variant = {r["sid"]: r for r in variant_rows}
    if set(base) != set(variant) or len(base) != len(base_rows) or len(variant) != len(variant_rows):
        raise ValueError("condition rows must be duplicate free and SID aligned")

    deltas = []
    flips = 0
    for sid in sorted(base):
        before, after = base[sid]["scores"], variant[sid]["scores"]
        record = {
            "sid": sid,
            "delta_p1": after[0] - before[0],
            "delta_p2": after[1] - before[1],
            "delta_p4": after[3] - before[3],
            "margin_shift": (after[3] - after[1]) - (before[3] - before[1]),
            "argmax_base": LETTERS[argmax5(before)],
            "argmax_variant": LETTERS[argmax5(after)],
        }
        if score_index is not None:
            record["delta_score"] = after[score_index] - before[score_index]
        if record["argmax_base"] != record["argmax_variant"]:
            flips += 1
        deltas.append(record)

    n = len(deltas)
    return {
        "per_sample": deltas,
        "summary": {
            "n_pairs": n,
            "argmax_flip_rate": flips / n if n else None,
            "mean_abs_delta_p1": sum(abs(d["delta_p1"]) for d in deltas) / n if n else None,
            "mean_abs_delta_p4": sum(abs(d["delta_p4"]) for d in deltas) / n if n else None,
            "mean_margin_shift": sum(d["margin_shift"] for d in deltas) / n if n else None,
            "metric_class": "sensitivity_only",
        },
    }
