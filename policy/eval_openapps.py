"""Closed-loop 3-arm evaluator (Policy / Policy+Rules / Policy+Rules+Verifier) for OpenApps.

Requires an external OpenApps + BrowserGym installation only at runtime when
executing live Playwright episodes; subtask composition logic, failure
categorization, and non-idempotent guards are pure Python and CPU-testable.
"""

from __future__ import annotations

import argparse
import base64
import collections
import copy
import hashlib
import itertools
import json
import pathlib
import time
import urllib.request
from typing import Any, Callable, Mapping, Sequence

from prm.controller import Context, Evidence, ModelSignal, decide
from protocols.qwen_computer_use_v2 import Geometry, parse_tool_call

VIEWPORT_W, VIEWPORT_H = 1024, 576
SPLIT_SEEDS: dict[str, int] = {"dev": 911, "heldout": 915, "train_sample": 281}
NON_IDEMPOTENT_SUBSTRINGS: tuple[str, ...] = (
    "delete",
    "remove",
    "send",
    "create",
    "save",
    "add",
    "submit",
    "pay",
    "buy",
    "transfer",
)


def is_non_idempotent_action(action: Mapping[str, Any] | None, audit_before: Mapping[str, Any] | None) -> bool:
    """Identify controls whose duplicate execution could mutate state twice."""
    act_name = (action or {}).get("action")
    if act_name in ("type", "scroll"):
        return False
    if act_name == "key":
        keys = [str(k).lower() for k in ((action or {}).get("keys") or [])]
        return "enter" in keys
    if act_name == "left_click":
        state = (audit_before or {}).get("state") or {}
        ctx = state.get("context") or {}
        hit = state.get("hit") or {}
        text = (str(ctx.get("text") or "") + " " + str(hit.get("text") or "") + " " + str(ctx.get("tag") or "")).lower()
        return any(tok in text for tok in NON_IDEMPOTENT_SUBSTRINGS)
    return False


def evaluate_composite_subtasks(
    task_obj: Any,
    initial_state: dict[str, Any],
    final_state: dict[str, Any],
    final_url: str | None = None,
    *,
    composite_cls: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Evaluate per-subtask completion via per-app-group official CompositeTask checks.

    Calling a single child ``sub.check_if_task_is_complete`` against full
    multi-app ``final_state`` rejects sibling subtasks' legitimate edits. Grouping
    subtasks by touched app keys and evaluating masked app-group subsets through
    ``CompositeTask`` eliminates sibling interference while preserving zero-drift
    official field checks.
    """
    if composite_cls is None:
        from open_apps.tasks.tasks import CompositeTask

        composite_cls = CompositeTask

    subs = list(getattr(task_obj, "subtasks", None) or [task_obj])
    n_sub = len(subs)
    try:
        whole_ok = bool(task_obj.check_if_task_is_complete(initial_state, final_state, current_url=final_url))
    except TypeError:
        whole_ok = bool(task_obj.check_if_task_is_complete(initial_state, final_state))

    if whole_ok or not getattr(task_obj, "subtasks", None):
        results = [{"kind": type(s).__name__, "passed": whole_ok, "detail": "official_whole_task"} for s in subs]
        passed = sum(1 for r in results if r["passed"])
        return {
            "subtasks_total": n_sub,
            "subtasks_passed": passed,
            "subtask_completion_rate": round(passed / n_sub, 4) if n_sub else 0.0,
            "subtasks": results,
        }

    app_keys = [k for k in initial_state.keys() if not k.startswith("_") and k not in ("codeeditor", "ui_library")]
    sub_apps: list[set[str]] = []
    cur = initial_state
    for s in subs:
        try:
            nxt = s.get_target_state(cur)
            touched = {k for k in app_keys if nxt.get(k) != cur.get(k)}
            cur = nxt
        except Exception:
            touched = set(app_keys)
        if not touched:
            touched = set(app_keys)
        sub_apps.append(touched)

    parent = list(range(n_sub))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n_sub):
        for j in range(i + 1, n_sub):
            if sub_apps[i] & sub_apps[j]:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    groups: dict[int, list[int]] = {}
    for i in range(n_sub):
        groups.setdefault(find(i), []).append(i)

    passed_idx: set[int] = set()
    for idxs in groups.values():
        group_apps = set().union(*(sub_apps[i] for i in idxs))
        masked_final = copy.deepcopy(initial_state)
        for k in group_apps:
            if k in final_state:
                masked_final[k] = copy.deepcopy(final_state[k])
        best_subset: tuple[int, ...] = ()
        for r in range(len(idxs), 0, -1):
            found = False
            for comb in itertools.combinations(idxs, r):
                ct = composite_cls(goal="", subtasks=[subs[i] for i in comb])
                try:
                    ok = ct.check_if_task_is_complete(initial_state, masked_final, current_url=final_url)
                except Exception:
                    ok = False
                if ok:
                    best_subset = comb
                    found = True
                    break
            if found:
                break
        passed_idx.update(best_subset)

    results = [
        {"kind": type(s).__name__, "passed": (i in passed_idx), "detail": "official_app_group_composite"}
        for i, s in enumerate(subs)
    ]
    passed = sum(1 for r in results if r["passed"])
    return {
        "subtasks_total": n_sub,
        "subtasks_passed": passed,
        "subtask_completion_rate": round(passed / n_sub, 4) if n_sub else 0.0,
        "subtasks": results,
    }


def classify_failure(episode: Mapping[str, Any], subtask_eval: Mapping[str, Any]) -> str:
    """Categorize closed-loop episode outcome or dominant failure mode."""
    if episode.get("oracle_terminal_success"):
        return "success"
    status = episode.get("status")
    steps = episode.get("steps") or []
    if status == "repeated_invalid_action":
        return "repeated_invalid_protocol"
    if status == "deterministic_cycle_fuse":
        return "state_action_cycle_stuck"
    if status == "teacher_stop":
        return "premature_terminate_partial" if subtask_eval.get("subtasks_passed", 0) > 0 else "premature_terminate_zero"
    unfocused = sum(
        1
        for s in steps
        if (s.get("parsed_action") or {}).get("action") == "type"
        and ((s.get("audit_before") or {}).get("state") or {}).get("active", {}).get("tag") not in ("input", "textarea")
    )
    if unfocused >= 2:
        return "unfocused_typing"
    if status == "repeated_action_fuse":
        last_act = (steps[-1].get("parsed_action") or {}).get("action") if steps else ""
        return "scroll_loop_stuck" if last_act == "scroll" else "repeated_click_or_action_stuck"
    if subtask_eval.get("subtasks_passed", 0) > 0:
        return "partial_progress_budget_exhausted"
    return "zero_progress_budget_exhausted"


def safe_screenshot(page: Any, path: pathlib.Path) -> None:
    """Capture viewport PNG with Chromium CDP fallback when Playwright waits on webfonts."""
    try:
        page.screenshot(path=str(path), timeout=20000)
    except Exception:
        cdp = page.context.new_cdp_session(page)
        try:
            b64 = cdp.send("Page.captureScreenshot", {"format": "png"})["data"]
            path.write_bytes(base64.b64decode(b64))
        finally:
            try:
                cdp.detach()
            except Exception:
                pass


def _post_json(api_url: str, endpoint: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(
        api_url.rstrip("/") + endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.load(resp)


def run_openapps_eval(args: argparse.Namespace) -> dict[str, Any]:
    """Run controlled 3-arm closed-loop evaluation against a live OpenApps AppServer."""
    if args.split == "heldout" and not args.confirm_one_shot_heldout:
        raise RuntimeError("Refusing heldout evaluation without --confirm-one-shot-heldout")

    out_dir = pathlib.Path(args.output_dir) / args.run_tag
    if args.split == "heldout" and (out_dir / "results.json").exists():
        raise RuntimeError(f"Heldout output already exists at {out_dir / 'results.json'}")

    from PIL import Image, ImageChops, ImageStat
    from browsergym.core.action.highlevel import HighLevelActionSet
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from open_apps.mcp.appserver import AppServer
    from open_apps.tasks.add_tasks_to_browsergym import register_tasks_with_browsergym
    import gymnasium as gym

    ep_dir = out_dir / "episodes"
    logs_dir = out_dir / "server_logs"
    ep_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed if args.seed > 0 else SPLIT_SEEDS[args.split]
    freeze = json.loads(pathlib.Path(args.task_freeze_json).read_text(encoding="utf-8"))
    lh = OmegaConf.to_container(OmegaConf.load(args.tasks_yaml), resolve=False)
    rows = sorted([t for t in freeze["tasks"] if t["split"] == args.split], key=lambda x: x["official_task_id"])
    if args.max_tasks > 0:
        rows = rows[: args.max_tasks]

    tasks = [(r["official_task_id"], r["component"], instantiate(OmegaConf.create(lh[r["official_task_id"]]))) for r in rows]
    register_tasks_with_browsergym([t[2] for t in tasks])
    aset = HighLevelActionSet(subsets=["coord"], multiaction=True)
    server = AppServer(
        "todo",
        extra_overrides=[f"logs_dir={logs_dir}", "headless=True", f"seed={seed}", "apps/theme=dark", "apps/start_page/layout=desktop"],
    )

    all_summaries: list[dict[str, Any]] = []
    t_all = time.time()
    try:
        for t_idx, (tid, comp, tobj) in enumerate(tasks):
            frame_dir = out_dir / "frames" / tid
            frame_dir.mkdir(parents=True, exist_ok=True)
            env = gym.make(
                "browsergym/" + tobj.task_id,
                task_kwargs={"base_url": server.base_url, "screen_resolution": (VIEWPORT_W, VIEWPORT_H)},
                action_mapping=aset.to_python_code,
                headless=True,
            )
            record: dict[str, Any] = {
                "run_tag": args.run_tag,
                "split": args.split,
                "mode": args.mode,
                "task_id": tid,
                "component": comp,
                "steps": [],
                "status": "running",
                "invalid_actions": 0,
                "repeat_events": 0,
                "rule_interventions": 0,
                "verifier_calls": 0,
                "verifier_interventions": 0,
            }
            t_ep = time.time()
            try:
                server.reset()
                env.reset(seed=seed)
                page = env.unwrapped.page
                page.wait_for_timeout(150)
                initial_state = server.get_state()
                history: list[dict[str, Any]] = []
                seen_keys: dict[tuple[Any, ...], int] = {}
                pending_feedback = ""
                next_temp = 0.0
                recoveries_used = 0
                same_no_prog = 0
                prev_sig = None
                recent_sa: list[str] = []

                for step_i in range(args.max_steps):
                    b_path = frame_dir / f"{step_i:03d}_before.png"
                    safe_screenshot(page, b_path)
                    b_sha = hashlib.sha256(b_path.read_bytes()).hexdigest()
                    key = (b_sha, json.dumps(history[-4:], sort_keys=True)) if args.mode == "policy_only" else (b_sha, json.dumps(history[-4:], sort_keys=True), pending_feedback)
                    if key in seen_keys:
                        record["status"] = "deterministic_cycle_fuse"
                        record["repeat_events"] += 1
                        break
                    seen_keys[key] = step_i

                    res = _post_json(
                        args.api_url,
                        "/predict",
                        {
                            "image_path": str(b_path),
                            "instruction": tobj.goal,
                            "visible_history": history[-4:],
                            "visible_feedback": pending_feedback,
                            "temperature": next_temp,
                            "max_new_tokens": 128,
                        },
                    )
                    pending_feedback, next_temp = "", 0.0
                    parsed = parse_tool_call(res["raw"], Geometry(VIEWPORT_W, VIEWPORT_H), space="normalized", viewport=Geometry(VIEWPORT_W, VIEWPORT_H))
                    act = parsed.arguments
                    if act["action"] == "terminate":
                        record["status"] = "teacher_stop"
                        break
                    _, rew, term, trunc, _ = env.step(parsed.action_text)
                    page = env.unwrapped.page
                    page.wait_for_timeout(150)
                    a_path = frame_dir / f"{step_i:03d}_after.png"
                    safe_screenshot(page, a_path)
                    a_sha = hashlib.sha256(a_path.read_bytes()).hexdigest()
                    with Image.open(b_path) as ib, Image.open(a_path) as ia:
                        delta = ImageStat.Stat(ImageChops.difference(ib.convert("L").resize((96, 54)), ia.convert("L").resize((96, 54)))).mean[0]

                    sig = json.dumps(act, sort_keys=True)
                    same_no_prog = same_no_prog + 1 if (sig == prev_sig and delta < 0.05) else 0
                    prev_sig = sig
                    recent_sa.append(f"{b_sha[:12]}:{sig}")
                    non_idemp = is_non_idempotent_action(act, None)

                    if args.mode in ("rules", "verifier") and not (term or trunc):
                        rule_triggered = False
                        if same_no_prog >= 1 or (len(recent_sa) >= 4 and len(set(recent_sa[-4:])) <= 2):
                            if not non_idemp and recoveries_used < 4:
                                pending_feedback = "Previous action produced no meaningful visual change or is repeating a cycle."
                                next_temp = 0.5
                                rule_triggered = True
                                recoveries_used += 1
                                record["rule_interventions"] += 1
                        if args.mode == "verifier":
                            vout = _post_json(
                                args.api_url,
                                "/verify",
                                {"before_image": str(b_path), "after_image": str(a_path), "instruction": tobj.goal, "action": act, "visible_history": history[-4:]},
                            )
                            record["verifier_calls"] += 1
                            v_lbl = vout["pred_label"]
                            v_probs = vout["probs"]
                            decide(
                                Context(
                                    task=tobj.goal,
                                    action_type=act["action"],
                                    idempotent=not non_idemp,
                                    before_sha256=b_sha,
                                    after_sha256=a_sha,
                                    retries_used=record["verifier_interventions"],
                                    recoveries_used=recoveries_used,
                                    recent_state_action=tuple(recent_sa[-4:]),
                                    high_risk_action=non_idemp and v_lbl == "C",
                                ),
                                Evidence(trusted=False, terminal="success" if term else None),
                                reward=ModelSignal(label={"A": "progress", "B": "no_progress", "C": "regression", "D": "observe"}[v_lbl], confidence=float(v_probs[v_lbl])),
                            )
                            if not rule_triggered and recoveries_used < 4:
                                if v_lbl == "C" and v_probs["C"] >= args.tau_c:
                                    pending_feedback = "Semantic check flagged a potential form/target mistake or premature app switch."
                                    next_temp = 0.35
                                    recoveries_used += 1
                                    record["verifier_interventions"] += 1
                                elif v_lbl == "B" and v_probs["B"] >= args.tau_b and not non_idemp:
                                    pending_feedback = "Verifier detected stalled goal progress."
                                    next_temp = 0.4
                                    recoveries_used += 1
                                    record["verifier_interventions"] += 1

                    record["steps"].append({"step_idx": step_i, "parsed_action": act, "visual_delta_96x54": round(delta, 4)})
                    history.append(act)
                    if term or trunc:
                        record["status"] = "env_terminated" if term else "env_truncated"
                        break
                else:
                    record["status"] = "step_budget"

                val_r, val_done, _, _ = env.unwrapped._task_validate()
                final_state = server.get_state()
                subtask_eval = evaluate_composite_subtasks(tobj, initial_state, final_state, str(env.unwrapped.page.url))
                record["oracle_terminal_success"] = bool(val_done and float(val_r) >= 1.0)
                record["subtask_eval"] = subtask_eval
                record["failure_mode"] = classify_failure(record, subtask_eval)
                all_summaries.append(
                    {
                        "idx": t_idx + 1,
                        "task_id": tid,
                        "oracle_terminal_success": record["oracle_terminal_success"],
                        "subtasks_passed": subtask_eval["subtasks_passed"],
                        "subtasks_total": subtask_eval["subtasks_total"],
                        "elapsed_s": round(time.time() - t_ep, 2),
                    }
                )
            finally:
                env.close()
            (ep_dir / f"{tid}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    finally:
        server.close()

    n_tasks = len(all_summaries)
    n_succ = sum(1 for r in all_summaries if r["oracle_terminal_success"])
    summary = {
        "run_tag": args.run_tag,
        "split": args.split,
        "mode": args.mode,
        "elapsed_s": round(time.time() - t_all, 2),
        "n_tasks": n_tasks,
        "n_success": n_succ,
        "success_rate": round(n_succ / n_tasks, 4) if n_tasks else 0.0,
        "tasks": all_summaries,
    }
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Evaluate Qwen3-VL Policy / Rules / Verifier on OpenApps.")
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--split", choices=("dev", "heldout", "train_sample"), required=True)
    ap.add_argument("--mode", choices=("policy_only", "rules", "verifier"), default="policy_only")
    ap.add_argument("--api-url", required=True, help="Dual-LoRA server base URL.")
    ap.add_argument("--task-freeze-json", required=True)
    ap.add_argument("--tasks-yaml", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-steps", type=int, default=25)
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau-c", type=float, default=0.65)
    ap.add_argument("--tau-b", type=float, default=0.65)
    ap.add_argument("--confirm-one-shot-heldout", action="store_true")
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(json.dumps(run_openapps_eval(args), indent=2))


if __name__ == "__main__":
    main()
