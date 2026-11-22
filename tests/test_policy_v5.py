"""Pure-CPU unit tests for V5 policy/verifier training, step labeling, composite evaluation, and result recomputation."""

from __future__ import annotations

import copy
import pathlib
import subprocess
import sys

import pytest

from eval.v5_recompute import (
    exact_mcnemar_from_booleans,
    exact_mcnemar_from_counts,
    metrics_from_confusion_matrix,
    verify_all_results,
    wilson_ci,
)
from policy.collator import VERIFIER_LABELS, build_verifier_messages
from policy.eval_openapps import evaluate_composite_subtasks, is_non_idempotent_action
from policy.step_labeler import extract_labeled_steps_from_episode
from policy.train_verifier_sft import compute_class_weights, tune_thresholds_on_vval

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class _FakeSubtask:
    def __init__(self, app: str, key: str, expected_val: object) -> None:
        self.app = app
        self.key = key
        self.expected_val = expected_val

    def get_target_state(self, state: dict) -> dict:
        nxt = copy.deepcopy(state)
        nxt[self.app] = dict(nxt.get(self.app, {}))
        nxt[self.app][self.key] = self.expected_val
        return nxt


class _FakeCompositeTask:
    def __init__(self, goal: str = "", subtasks=None) -> None:
        self.goal = goal
        self.subtasks = list(subtasks or [])

    def get_target_state(self, initial_state: dict) -> dict:
        st = initial_state
        for sub in self.subtasks:
            st = sub.get_target_state(st)
        return st

    def check_if_task_is_complete(self, initial_state: dict, current_state: dict, current_url=None) -> bool:
        return self.get_target_state(initial_state) == current_state


def test_step_labeler_objective_classes_without_episode_outcome_leakage() -> None:
    def mk_ep(oracle_success: bool) -> dict:
        return {
            "task_id": "t_demo",
            "component": "c_demo",
            "goal": "Create item then send notice",
            "oracle_terminal_success": oracle_success,
            "steps": [
                {
                    "step_idx": 0,
                    "parse_ok": True,
                    "before_image": "000_before.png",
                    "after_image": "000_after.png",
                    "before_sha256": "sha0",
                    "after_sha256": "sha1",
                    "visual_delta_96x54": 1.2,
                    "parsed_action": {"action": "left_click", "coordinate": [200, 900]},
                    "audit_before": {"state": {"url": "http://app/", "active": {"tag": "body"}, "context": {"tag": "a", "href": "/calendar/create_event"}}},
                    "audit_after": {"state": {"url": "http://app/calendar/create_event", "active": {"tag": "body"}}},
                },
                {
                    "step_idx": 1,
                    "parse_ok": True,
                    "before_image": "001_before.png",
                    "after_image": "001_after.png",
                    "before_sha256": "sha1",
                    "after_sha256": "sha2",
                    "visual_delta_96x54": 0.8,
                    "parsed_action": {"action": "left_click", "coordinate": [100, 900]},
                    "audit_before": {"state": {"url": "http://app/calendar/create_event", "active": {"tag": "body"}, "context": {"tag": "a", "href": "/todo", "text": "ToDo"}}},
                    "audit_after": {"state": {"url": "http://app/todo", "active": {"tag": "body"}}},
                },
                {
                    "step_idx": 2,
                    "parse_ok": True,
                    "before_image": "002_before.png",
                    "after_image": "002_after.png",
                    "before_sha256": "sha2",
                    "after_sha256": "sha3",
                    "visual_delta_96x54": 1.1,
                    "parsed_action": {"action": "left_click", "coordinate": [200, 900]},
                    "audit_before": {"state": {"url": "http://app/todo", "active": {"tag": "body"}, "context": {"tag": "a", "href": "/calendar/create_event"}}},
                    "audit_after": {"state": {"url": "http://app/calendar/create_event", "active": {"tag": "body"}}},
                },
                {
                    "step_idx": 3,
                    "parse_ok": True,
                    "before_image": "003_before.png",
                    "after_image": "003_after.png",
                    "before_sha256": "sha3",
                    "after_sha256": "sha3b",
                    "visual_delta_96x54": 0.01,
                    "parsed_action": {"action": "type", "text": "Buy milk"},
                    "audit_before": {"state": {"url": "http://app/todo", "active": {"tag": "body"}}},
                    "audit_after": {"state": {"url": "http://app/todo", "active": {"tag": "body"}}},
                },
                {
                    "step_idx": 4,
                    "parse_ok": True,
                    "before_image": "004_before.png",
                    "after_image": "004_after.png",
                    "before_sha256": "sha3b",
                    "after_sha256": "sha3b",
                    "visual_delta_96x54": 0.0,
                    "parsed_action": {"action": "scroll", "coordinate": [500, 500], "pixels": -400},
                    "audit_before": {"state": {"url": "http://app/todo", "active": {"tag": "body"}}},
                    "audit_after": {"state": {"url": "http://app/todo", "active": {"tag": "body"}}},
                },
            ],
        }

    rows_win = extract_labeled_steps_from_episode(mk_ep(True), "train")
    rows_lose = extract_labeled_steps_from_episode(mk_ep(False), "train")
    assert [r["label"] for r in rows_win] == [r["label"] for r in rows_lose]
    assert rows_win[0]["label"] == "A"
    assert rows_win[1]["label_reason"] == "premature_form_abandon_before_submit" and rows_win[1]["label"] == "C"
    assert rows_win[2]["label_reason"] == "empty_app_ping_pong" and rows_win[2]["label"] == "C"
    assert rows_win[3]["label_reason"] == "unfocused_typing_non_input" and rows_win[3]["label"] == "C"
    assert rows_win[4]["label_reason"] == "scroll_no_movement" and rows_win[4]["label"] == "B"


def test_evaluate_composite_subtasks_per_app_group() -> None:
    init = {"calendar": {"ev1": False}, "todo": {"t1": False, "t2": False}, "messages": {"m1": False}}
    s_ev1 = _FakeSubtask("calendar", "ev1", True)
    s_todo1 = _FakeSubtask("todo", "t1", True)
    s_todo2 = _FakeSubtask("todo", "t2", True)
    s_msg = _FakeSubtask("messages", "m1", True)
    task = _FakeCompositeTask(subtasks=[s_ev1, s_todo1, s_todo2, s_msg])

    fin_all = task.get_target_state(init)
    res_all = evaluate_composite_subtasks(task, init, fin_all, composite_cls=_FakeCompositeTask)
    assert (res_all["subtasks_passed"], res_all["subtasks_total"]) == (4, 4)

    # Corrupt calendar field while todo (2 subtasks) and messages (1 subtask) are satisfied -> 3/4
    fin_partial = copy.deepcopy(fin_all)
    fin_partial["calendar"]["ev1"] = "wrong_desc"
    res_partial = evaluate_composite_subtasks(task, init, fin_partial, composite_cls=_FakeCompositeTask)
    assert (res_partial["subtasks_passed"], res_partial["subtasks_total"]) == (3, 4)
    assert [x["passed"] for x in res_partial["subtasks"]] == [False, True, True, True]

    # Same-app partial pass (only s_todo1 done in todo) + messages -> 2/4
    fin_same_app = s_msg.get_target_state(s_todo1.get_target_state(init))
    res_same = evaluate_composite_subtasks(task, init, fin_same_app, composite_cls=_FakeCompositeTask)
    assert (res_same["subtasks_passed"], res_same["subtasks_total"]) == (2, 4)
    assert [x["passed"] for x in res_same["subtasks"]] == [False, True, False, True]


def test_non_idempotent_guard_and_verifier_utilities() -> None:
    assert is_non_idempotent_action({"action": "left_click"}, {"state": {"context": {"text": "Delete Event"}}})
    assert is_non_idempotent_action({"action": "key", "keys": ["Enter"]}, None)
    assert not is_non_idempotent_action({"action": "scroll", "coordinate": [500, 500], "pixels": -400}, None)

    w = compute_class_weights(["A"] * 441 + ["B"] * 183 + ["C"] * 78 + ["D"] * 11, cap=3.5)
    assert len(w) == 4 and w[0] < w[1] < w[2] <= w[3] <= 3.5

    preds = [
        {"gt": "A", "probs": {"A": 0.9, "B": 0.05, "C": 0.05, "D": 0.0}},
        {"gt": "B", "probs": {"A": 0.1, "B": 0.8, "C": 0.1, "D": 0.0}},
        {"gt": "C", "probs": {"A": 0.2, "B": 0.1, "C": 0.7, "D": 0.0}},
    ]
    th = tune_thresholds_on_vval(preds, max_false_a_rate=0.10)
    assert th["false_a_rate"] <= 0.10 and th["c_rec"] == 1.0 and th["b_rec"] == 1.0

    msg = build_verifier_messages("b.png", "a.png", "Do task", {"action": "left_click", "coordinate": [10, 20]})
    assert len(msg[0]["content"]) == 5 and len(VERIFIER_LABELS) == 4


def test_verifier_batch1_class_weight_cancellation_regression() -> None:
    import torch
    import torch.nn.functional as F

    w = torch.tensor([0.5, 1.2, 2.1, 3.5], dtype=torch.float32)
    tgt = torch.tensor([2], dtype=torch.long)

    logits_weighted = torch.tensor([[1.2, -0.3, 0.4, -1.1]], dtype=torch.float32, requires_grad=True)
    logits_unweighted = logits_weighted.detach().clone().requires_grad_(True)
    logits_fixed = logits_weighted.detach().clone().requires_grad_(True)

    loss_w = F.cross_entropy(logits_weighted, tgt, weight=w)
    loss_u = F.cross_entropy(logits_unweighted, tgt)
    assert torch.allclose(loss_w, loss_u, atol=1e-6)
    loss_w.backward()
    loss_u.backward()
    assert torch.allclose(logits_weighted.grad, logits_unweighted.grad, atol=1e-6)

    loss_fixed = (F.cross_entropy(logits_fixed, tgt, reduction="none") * w[tgt]).mean()
    loss_fixed.backward()
    assert torch.allclose(loss_fixed, loss_u * w[tgt.item()], atol=1e-6)
    assert torch.allclose(logits_fixed.grad, logits_unweighted.grad * w[tgt.item()], atol=1e-6)


def test_v5_recompute_against_published_results() -> None:
    assert wilson_ci(0, 12) == [0.0, 0.2425]
    assert exact_mcnemar_from_booleans([False] * 12, [True] * 7 + [False] * 5)["exact_mcnemar_two_sided_p"] == 0.015625
    assert exact_mcnemar_from_counts(wins=7, losses=0, both_pass=0, both_fail=5)["exact_mcnemar_two_sided_p"] == 0.015625
    assert exact_mcnemar_from_counts(wins=0, losses=1, both_pass=6, both_fail=5)["exact_mcnemar_two_sided_p"] == 1.0
    report = verify_all_results(REPO_ROOT / "results")
    assert report["ok"] is True
    assert report["heldout_12"]["runs"]["heldout_arm_a"]["n_success"] == 7
    assert report["heldout_12"]["runs"]["heldout_arm_c"]["n_success"] == 6
    assert report["verifier_checked"]["dev936_zeroshot_vs_adapted.adapted_v5"]["macro_f1_4cls"] == 0.451


@pytest.mark.parametrize(
    "module",
    [
        "policy.train_policy_sft",
        "policy.train_verifier_sft",
        "policy.dual_lora_router",
        "policy.eval_openapps",
        "eval.v5_recompute",
    ],
)
def test_cli_entrypoints_help_on_cpu(module: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-B", "-m", module, "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "usage:" in proc.stdout.lower()
