"""Per-layer LoRA gradient-conflict probe.

Training a verifier on several objectives at once (classification, margin,
distillation) can produce LoRA updates that point in opposite directions. The
probe answers one narrow question per layer:

    For two loss terms, what is the cosine similarity between their gradients
    with respect to the same LoRA tensor?

A consistently negative cosine in a layer means the two objectives are fighting
over the same subspace; that layer is a candidate for freezing, re-weighting or
gradient projection. The probe does not decide the remedy and does not claim
causality - it only measures.

Design split
------------
* :func:`lora_grad_vectors` collects detached gradients from a live model
  (torch required) and reduces them to plain Python lists.
* Everything else in this module is pure standard library, so the aggregation
  and statistics can be unit-tested on synthetic vectors without a GPU.

All functions reject non-finite values. Zero-norm gradients return ``None``
rather than a fabricated cosine.
"""

from __future__ import annotations

import math
import random
import re
import statistics
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "LAYER_RE",
    "MODULES",
    "bootstrap_ci",
    "combine",
    "cosine",
    "decomposition_consistency",
    "dot",
    "group_by",
    "layer_conflict_table",
    "lora_grad_vectors",
    "norm",
    "norm_table",
    "pair_records",
    "pair_records_with_skips",
    "parse_lora_param",
    "restore_requires_grad",
    "snapshot_requires_grad",
    "summarize_cosines",
]

LAYER_RE = re.compile(r"layers\.(\d+)\..*?\.([qkvo]_proj)\.(lora_[AB])")
MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
FACTORS = ("lora_A", "lora_B")


# ---------------------------------------------------------------------------
# Vector primitives
# ---------------------------------------------------------------------------


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b))


def norm(a: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def cosine(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Cosine similarity in ``[-1, 1]``, or ``None`` when a vector is zero.

    Raises:
        ValueError: length mismatch or any non-finite component.
    """
    if len(a) != len(b):
        raise ValueError("length mismatch")
    if any(not math.isfinite(x) for x in a) or any(not math.isfinite(x) for x in b):
        raise ValueError("non-finite gradient component")
    na, nb = norm(a), norm(b)
    if na == 0.0 or nb == 0.0:
        return None
    value = dot(a, b) / (na * nb)
    # Guard against 1 + 1e-16 rounding.
    return max(-1.0, min(1.0, value))


def combine(vectors: Sequence[Sequence[float]], weights: Sequence[float] | None = None):
    """Weighted elementwise sum of equally sized vectors."""
    if not vectors:
        raise ValueError("empty vector list")
    weights = list(weights) if weights is not None else [1.0] * len(vectors)
    if len(weights) != len(vectors):
        raise ValueError("weights length mismatch")
    out = [0.0] * len(vectors[0])
    for vector, weight in zip(vectors, weights):
        if len(vector) != len(out):
            raise ValueError("vector length mismatch")
        for i, value in enumerate(vector):
            out[i] += weight * value
    return out


# ---------------------------------------------------------------------------
# Parameter-name parsing
# ---------------------------------------------------------------------------


def parse_lora_param(name: str) -> tuple[int | None, str | None, str | None]:
    """Split a LoRA parameter name into ``(layer, module, factor)``.

    ``...layers.12.self_attn.q_proj.lora_A.weight`` ->
    ``(12, "q_proj", "lora_A")``. Non-matching names yield ``(None, None, None)``
    so that unrelated parameters can be kept in the same table.
    """
    match = LAYER_RE.search(name)
    if not match:
        return None, None, None
    return int(match.group(1)), match.group(2), match.group(3)


def group_by(param_names: Iterable[str], key: str = "layer") -> dict[Any, list[str]]:
    """Group parameter names by ``layer``, ``module`` or ``factor``."""
    index = {"layer": 0, "module": 1, "factor": 2}[key]
    grouped: dict[Any, list[str]] = {}
    for name in param_names:
        parts = parse_lora_param(name)
        grouped.setdefault(parts[index], []).append(name)
    return grouped


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def summarize_cosines(pairs: Sequence[Mapping[str, Any]], group: str | None = None):
    """Aggregate ``{"cos": float | None, ...}`` records into conflict statistics.

    ``conflict_ratio`` is the share of negative cosines; ``strong_conflict_ratio``
    the share below -0.1; ``aligned_ratio`` the share above +0.1. Zero-norm
    pairs are counted in ``n_skipped`` and excluded from every ratio.
    """
    values = [p["cos"] for p in pairs if p.get("cos") is not None]
    skipped = [p for p in pairs if p.get("cos") is None]
    if not values:
        return {
            "group": group,
            "n": 0,
            "n_skipped": len(skipped),
            "mean": None,
            "median": None,
            "conflict_ratio": None,
            "strong_conflict_ratio": None,
            "aligned_ratio": None,
        }
    return {
        "group": group,
        "n": len(values),
        "n_skipped": len(skipped),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "conflict_ratio": sum(1 for v in values if v < 0) / len(values),
        "strong_conflict_ratio": sum(1 for v in values if v < -0.1) / len(values),
        "aligned_ratio": sum(1 for v in values if v > 0.1) / len(values),
    }


def layer_conflict_table(pairs: Sequence[Mapping[str, Any]]):
    """Group pair records by layer and by ``(layer, module)``.

    Records must carry ``a`` and ``b`` parameter names together with ``cos``.
    """
    by_layer: dict[int, list[Mapping[str, Any]]] = {}
    by_unit: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    unparsed: list[Mapping[str, Any]] = []

    for record in pairs:
        layer_a, module_a, _ = parse_lora_param(str(record.get("a", "")))
        layer_b, module_b, _ = parse_lora_param(str(record.get("b", "")))
        if layer_a is None or layer_b is None or layer_a != layer_b or module_a != module_b:
            unparsed.append(record)
            continue
        by_layer.setdefault(layer_a, []).append(record)
        by_unit.setdefault((layer_a, str(module_a)), []).append(record)

    return {
        "by_layer": {
            str(layer): summarize_cosines(records, group=str(layer))
            for layer, records in sorted(by_layer.items())
        },
        "by_layer_module": {
            f"{layer}:{module}": summarize_cosines(records, group=f"{layer}:{module}")
            for (layer, module), records in sorted(by_unit.items())
        },
        "n_unparsed": len(unparsed),
    }


def bootstrap_ci(
    values: Sequence[float],
    *,
    replicates: int = 2000,
    seed: int = 20261113,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Deterministic percentile bootstrap CI for the mean."""
    if not values:
        return {"n": 0, "mean": None, "ci95": None}
    if len(values) < 2:
        return {"n": 1, "mean": float(values[0]), "ci95": None}
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(replicates):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(alpha / 2 * (replicates - 1))]
    hi = means[int((1 - alpha / 2) * (replicates - 1))]
    return {
        "n": n,
        "mean": sum(values) / n,
        "median": statistics.median(values),
        "ci95": [lo, hi],
        "seed": seed,
        "replicates": replicates,
    }


def norm_table(norms_by_param: Mapping[str, float]):
    """Aggregate per-parameter gradient norms into layer/module/factor tables.

    Squared norms are summed within a group and the square root is taken once,
    which is the norm of the concatenated group gradient.
    """
    table: dict[str, Any] = {
        "total": 0.0,
        "by_layer": {},
        "by_module": {},
        "by_factor": {},
        "n_params": 0,
    }
    for name, value in norms_by_param.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid gradient norm for {name}")
        layer, module, factor = parse_lora_param(name)
        table["n_params"] += 1
        table["total"] += value * value
        if layer is not None:
            table["by_layer"][str(layer)] = table["by_layer"].get(str(layer), 0.0) + value * value
        if module:
            table["by_module"][module] = table["by_module"].get(module, 0.0) + value * value
        if factor:
            table["by_factor"][factor] = table["by_factor"].get(factor, 0.0) + value * value

    table["total"] = math.sqrt(table["total"])
    for key in ("by_layer", "by_module", "by_factor"):
        table[key] = {k: math.sqrt(v) for k, v in sorted(table[key].items())}
    return table


def decomposition_consistency(
    grad_ce: Sequence[float],
    grad_aux: Sequence[float],
    grad_total: Sequence[float],
    weight: float,
    *,
    tol: float = 1e-9,
):
    """Check ``grad_total == grad_ce + weight * grad_aux`` elementwise."""
    if not (len(grad_ce) == len(grad_aux) == len(grad_total)):
        raise ValueError("length mismatch")
    worst = max(
        (abs(total - (ce + weight * aux)) for ce, aux, total in zip(grad_ce, grad_aux, grad_total)),
        default=0.0,
    )
    return {"max_abs_error": worst, "ok": worst <= tol, "weight": weight}


def pair_records(
    vectors_a: Mapping[str, Sequence[float]],
    vectors_b: Mapping[str, Sequence[float]],
    *,
    min_length: int = 1,
) -> list[dict[str, Any]]:
    """Compute one cosine record per shared parameter name.

    Unpaired or length-mismatched names are dropped silently. Use
    :func:`pair_records_with_skips` when the caller must report them.
    """
    records, _ = pair_records_with_skips(vectors_a, vectors_b, min_length=min_length)
    return records


def pair_records_with_skips(
    vectors_a: Mapping[str, Sequence[float]],
    vectors_b: Mapping[str, Sequence[float]],
    *,
    min_length: int = 1,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Like :func:`pair_records` but also returns names that could not be paired."""
    skipped: list[str] = []
    records: list[dict[str, Any]] = []
    for name in sorted(set(vectors_a) & set(vectors_b)):
        a, b = vectors_a[name], vectors_b[name]
        if len(a) != len(b) or len(a) < min_length:
            skipped.append(name)
            continue
        records.append({"a": name, "b": name, "cos": cosine(a, b)})
    skipped.extend(sorted(set(vectors_a) ^ set(vectors_b)))
    return records, skipped


# ---------------------------------------------------------------------------
# Live-model collection (torch required)
# ---------------------------------------------------------------------------


def lora_grad_vectors(model, *, factors: Sequence[str] = ("lora_A", "lora_B")):
    """Collect detached LoRA gradients from ``model`` after ``backward()``.

    Returns:
        ``{parameter_name: [float, ...]}`` on CPU. Parameters without a
        gradient are omitted; callers can detect that by set difference.
    """
    vectors: dict[str, list[float]] = {}
    for name, parameter in model.named_parameters():
        if not any(factor in name for factor in factors):
            continue
        if parameter.grad is None:
            continue
        flat = parameter.grad.detach().reshape(-1).float().cpu().tolist()
        if any(not math.isfinite(v) for v in flat):
            raise ValueError(f"non-finite gradient in {name}")
        vectors[name] = flat
    return vectors


def snapshot_requires_grad(model) -> dict[str, bool]:
    """Record ``requires_grad`` flags so a probe can restore them exactly."""
    return {name: bool(p.requires_grad) for name, p in model.named_parameters()}


def restore_requires_grad(model, state: Mapping[str, bool]) -> list[str]:
    """Restore flags captured by :func:`snapshot_requires_grad`.

    Returns the names whose flag actually changed back.
    """
    changed = []
    for name, parameter in model.named_parameters():
        if name in state and bool(parameter.requires_grad) != state[name]:
            parameter.requires_grad = state[name]
            changed.append(name)
    return changed
