"""Recompute V5 OpenApps Policy, Verifier, and Heldout headline metrics from anonymized per-task records."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Any, Mapping, Sequence

LABELS_4CLS: tuple[str, str, str, str] = ("A", "B", "C", "D")


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    """Compute exact two-sided 95% Wilson binomial confidence interval."""
    if n <= 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return [round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)]


def exact_mcnemar_from_counts(
    wins: int, losses: int, both_pass: int = 0, both_fail: int = 0
) -> dict[str, Any]:
    """Compute exact two-sided McNemar test directly from discordant/concordant counts.

    Args:
        wins: Count where ``arm_b`` passed and ``arm_a`` failed.
        losses: Count where ``arm_a`` passed and ``arm_b`` failed.
        both_pass: Count where both ``arm_a`` and ``arm_b`` passed.
        both_fail: Count where both ``arm_a`` and ``arm_b`` failed.
    """
    if min(wins, losses, both_pass, both_fail) < 0:
        raise ValueError("Contingency counts must be non-negative")
    n_disc = wins + losses
    if n_disc == 0:
        p_val = 1.0
    else:
        m = min(wins, losses)
        p_val = min(1.0, 2.0 * sum(math.comb(n_disc, i) * (0.5**n_disc) for i in range(m + 1)))
    return {
        "n_wins": wins,
        "n_losses": losses,
        "n_both_pass": both_pass,
        "n_both_fail": both_fail,
        "exact_mcnemar_two_sided_p": round(p_val, 6),
    }


def exact_mcnemar_from_booleans(
    a_bools: Sequence[bool], b_bools: Sequence[bool]
) -> dict[str, Any]:
    """Compute exact two-sided McNemar test over paired per-task success booleans."""
    if len(a_bools) != len(b_bools):
        raise ValueError(f"Mismatched paired lengths: {len(a_bools)} vs {len(b_bools)}")
    wins = sum(1 for a, b in zip(a_bools, b_bools) if bool(b) and not bool(a))
    losses = sum(1 for a, b in zip(a_bools, b_bools) if bool(a) and not bool(b))
    both_pass = sum(1 for a, b in zip(a_bools, b_bools) if bool(a) and bool(b))
    both_fail = sum(1 for a, b in zip(a_bools, b_bools) if not bool(a) and not bool(b))
    return exact_mcnemar_from_counts(wins, losses, both_pass, both_fail)


def metrics_from_confusion_matrix(
    conf: Mapping[str, Mapping[str, int]],
    labels: tuple[str, ...] = LABELS_4CLS,
) -> dict[str, Any]:
    """Recompute accuracy, per-class P/R/F1, Macro-F1, and false-intervention rate from raw counts."""
    total = sum(sum(int(conf[g][p]) for p in labels) for g in labels)
    correct = sum(int(conf[c][c]) for c in labels)
    acc = correct / max(1, total)
    per_class: dict[str, dict[str, Any]] = {}
    for c in labels:
        tp = int(conf[c][c])
        fp = sum(int(conf[g][c]) for g in labels if g != c)
        fn = sum(int(conf[c][p]) for p in labels if p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[c] = {
            "support": sum(int(conf[c][p]) for p in labels),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }
    sup_a = per_class["A"]["support"]
    fa_on_a = (int(conf["A"]["B"]) + int(conf["A"]["C"])) / max(1, sup_a)
    sup_bc = per_class["B"]["support"] + per_class["C"]["support"]
    bc_detect = (
        int(conf["B"]["B"]) + int(conf["B"]["C"]) + int(conf["C"]["B"]) + int(conf["C"]["C"])
    ) / max(1, sup_bc)
    return {
        "n": total,
        "accuracy": round(acc, 4),
        "macro_f1_4cls": round(sum(per_class[c]["f1"] for c in labels) / len(labels), 4),
        "class_c_semantic_err_recall": per_class["C"]["recall"],
        "class_c_precision": per_class["C"]["precision"],
        "bc_combined_failure_recall": round(bc_detect, 4),
        "false_intervention_rate_on_true_a": round(fa_on_a, 4),
        "per_class": per_class,
    }


def _verify_split_table(doc: Mapping[str, Any], expected_n: int, expected_sub_total: int) -> dict[str, Any]:
    checked_runs: dict[str, Any] = {}
    has_per_task = all("anonymized_tasks" in rec for rec in doc["runs"].values())
    for arm, rec in doc["runs"].items():
        rep = rec["reported"]
        if "anonymized_tasks" in rec:
            rows = rec["anonymized_tasks"]
            if len(rows) != expected_n:
                raise AssertionError(f"{arm}: expected {expected_n} tasks, got {len(rows)}")
            n_succ = sum(1 for r in rows if r["oracle_terminal_success"])
            sub_passed = sum(int(r["subtasks_passed"]) for r in rows)
            sub_total = sum(int(r["subtasks_total"]) for r in rows)
            if sub_total != expected_sub_total:
                raise AssertionError(f"{arm}: subtasks_total={sub_total} != {expected_sub_total}")
            macro_rate = round(
                sum(round(r["subtasks_passed"] / r["subtasks_total"], 4) for r in rows) / expected_n, 4
            )
            if "macro_subtask_rate" in rep:
                assert rep["macro_subtask_rate"] == macro_rate, f"{arm}: macro_subtask_rate mismatch"
        else:
            assert rep["n_tasks"] == expected_n, f"{arm}: n_tasks={rep['n_tasks']} != {expected_n}"
            assert rep["subtasks_total"] == expected_sub_total, (
                f"{arm}: subtasks_total={rep['subtasks_total']} != {expected_sub_total}"
            )
            n_succ = int(rep["n_success"])
            sub_passed = int(rep["subtasks_passed"])
            sub_total = int(rep["subtasks_total"])

        s_rate = round(n_succ / expected_n, 4)
        micro_rate = round(sub_passed / sub_total, 4)
        ci = wilson_ci(n_succ, expected_n)

        assert rep["n_success"] == n_succ, f"{arm}: n_success mismatch"
        assert rep["success_rate"] == s_rate, f"{arm}: success_rate mismatch"
        assert rep["success_wilson_95_ci"] == ci, f"{arm}: wilson CI mismatch"
        if "subtasks_passed" in rep:
            assert rep["subtasks_passed"] == sub_passed, f"{arm}: subtasks_passed mismatch"
            assert rep["micro_subtask_rate"] == micro_rate, f"{arm}: micro_subtask_rate mismatch"
        checked_runs[arm] = {
            "n_success": n_succ,
            "n_tasks": expected_n,
            "success_rate": s_rate,
            "success_wilson_95_ci": ci,
            "subtasks_passed": sub_passed,
            "subtasks_total": sub_total,
        }

    checked_pairs: dict[str, Any] = {}
    for pair_name, p_rec in doc.get("paired_comparisons", {}).items():
        if has_per_task:
            a_bools = [r["oracle_terminal_success"] for r in doc["runs"][p_rec["arm_a"]]["anonymized_tasks"]]
            b_bools = [r["oracle_terminal_success"] for r in doc["runs"][p_rec["arm_b"]]["anonymized_tasks"]]
            recomp = exact_mcnemar_from_booleans(a_bools, b_bools)
        else:
            wins = int(p_rec["n_wins"])
            losses = int(p_rec["n_losses"])
            both_pass = int(p_rec["n_both_pass"])
            both_fail = int(p_rec["n_both_fail"])
            arm_a_succ = checked_runs[p_rec["arm_a"]]["n_success"]
            arm_b_succ = checked_runs[p_rec["arm_b"]]["n_success"]
            assert both_pass + losses == arm_a_succ, f"{pair_name}: arm_a marginal mismatch"
            assert both_pass + wins == arm_b_succ, f"{pair_name}: arm_b marginal mismatch"
            assert both_pass + both_fail + wins + losses == expected_n, f"{pair_name}: total marginal mismatch"
            recomp = exact_mcnemar_from_counts(wins, losses, both_pass, both_fail)
        for k, v in recomp.items():
            assert p_rec[k] == v, f"{pair_name}.{k}: expected {v}, got {p_rec[k]}"
        checked_pairs[pair_name] = recomp
    return {"runs": checked_runs, "paired": checked_pairs}


def verify_all_results(results_dir: pathlib.Path | str | None = None) -> dict[str, Any]:
    """Verify all published V5 anonymized JSON summaries via direct recomputation."""
    rdir = pathlib.Path(results_dir) if results_dir else pathlib.Path(__file__).resolve().parents[1] / "results"
    dev_doc = json.loads((rdir / "policy_dev_summary.json").read_text(encoding="utf-8"))
    held_doc = json.loads((rdir / "policy_heldout_summary.json").read_text(encoding="utf-8"))
    ver_doc = json.loads((rdir / "verifier_metrics.json").read_text(encoding="utf-8"))

    dev_chk = _verify_split_table(dev_doc, expected_n=19, expected_sub_total=55)
    held_chk = _verify_split_table(held_doc, expected_n=12, expected_sub_total=36)

    ver_checked: dict[str, Any] = {}
    for split_key in ("dev936_zeroshot_vs_adapted", "v_val_epochs"):
        for arm_name, rec in ver_doc[split_key].items():
            recomp = metrics_from_confusion_matrix(rec["confusion_matrix"])
            rep = rec["reported"]
            for k in (
                "n",
                "accuracy",
                "macro_f1_4cls",
                "class_c_semantic_err_recall",
                "class_c_precision",
                "bc_combined_failure_recall",
                "false_intervention_rate_on_true_a",
            ):
                assert rep[k] == recomp[k], f"{split_key}.{arm_name}.{k}: {rep[k]} != {recomp[k]}"
            ver_checked[f"{split_key}.{arm_name}"] = recomp

    return {
        "ok": True,
        "dev_19": dev_chk,
        "heldout_12": held_chk,
        "verifier_checked": ver_checked,
    }


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Recompute and verify anonymized V5 experiment results.")
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args(argv)
    report = verify_all_results(args.results_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
