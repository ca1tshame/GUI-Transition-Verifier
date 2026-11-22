"""Objective step-level 4-class GUI transition labeler with zero episode-outcome copying.

Each step ``(before, action, after)`` is assigned one of ``A/B/C/D`` strictly
from step-level observable DOM/focus/URL/scroll/visual-delta evidence and causal
intra-episode form history — never from terminal episode reward or success.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

LABEL_NAMES: dict[str, str] = {
    "A": "effective_progress",
    "B": "no_progress_or_stagnation",
    "C": "semantic_error_or_regression",
    "D": "insufficient_evidence_or_observe",
}

INTERACTIVE_TAGS: frozenset[str] = frozenset(
    {"a", "button", "input", "textarea", "select", "label", "option", "summary"}
)
INTERACTIVE_ROLES: frozenset[str] = frozenset(
    {"button", "link", "checkbox", "menuitem", "tab", "radio", "switch", "textbox", "combobox"}
)
SUBMIT_WORDS: tuple[str, ...] = (
    "create",
    "save",
    "add",
    "submit",
    "send",
    "confirm",
    "delete",
    "remove",
    "update",
    "done",
)


def _scroller_positions(audit: Mapping[str, Any] | None) -> list[tuple[Any, Any, float]]:
    positions: list[tuple[Any, Any, float]] = []
    for sc in ((audit or {}).get("state") or {}).get("scrollers") or []:
        positions.append((sc.get("tag"), sc.get("id"), round(float(sc.get("scrollTop") or 0.0), 1)))
    return positions


def label_step(
    step: Mapping[str, Any],
    history_sigs: set[tuple[str, str]],
    prev_sig: str | None,
    typed_on_create: list[str],
    submitted_noncal_texts: set[str],
    last_typed_text_on_url: tuple[str, str] | None,
    seen_checkbox_sigs: set[str],
    visited_urls: dict[str, int],
    actions_inside_current_url: int,
) -> tuple[str, str]:
    """Assign ``A/B/C/D`` (or ``SKIP`` for ``terminate``) from step-level evidence."""
    parsed = step.get("parsed_action") or {}
    act = parsed.get("action")
    if not step.get("parse_ok") or not act:
        return "C", "invalid_parse_action"
    if act == "terminate":
        return "SKIP", "terminate_no_after"

    err = str(step.get("env_error") or "").strip()
    if err:
        return "C", "env_action_error"

    ab = (step.get("audit_before") or {}).get("state") or {}
    aa = (step.get("audit_after") or {}).get("state") or {}
    url_b = str(ab.get("url") or "")
    url_a = str(aa.get("url") or "")
    actv_b = ab.get("active") or {}
    actv_a = aa.get("active") or {}
    tag_b = str(actv_b.get("tag") or "").lower()
    tag_a = str(actv_a.get("tag") or "").lower()
    val_b = actv_b.get("value")
    val_a = actv_a.get("value")
    ctx_b = ab.get("context") or {}
    hit_b = ab.get("hit") or {}
    ctx_tag = str(ctx_b.get("tag") or "").lower()
    hit_tag = str(hit_b.get("tag") or "").lower()
    ctx_role = str(ctx_b.get("role") or "").lower()
    ctx_href = str(ctx_b.get("href") or hit_b.get("href") or "")
    ctx_txt = (str(ctx_b.get("text") or "") + " " + str(hit_b.get("text") or "")).strip().lower()
    delta = float(step.get("visual_delta_96x54") or 0.0)
    bsha = str(step.get("before_sha256") or "")
    sig = json.dumps(parsed, sort_keys=True)
    term = bool(step.get("env_terminated"))
    rew = float(step.get("env_reward") or 0.0)

    if term or rew > 0.0:
        return "A", "env_positive_reward_or_terminal"

    if act == "scroll":
        sc_b = _scroller_positions(step.get("audit_before"))
        sc_a = _scroller_positions(step.get("audit_after"))
        moved = any(
            len(sc_a) > i and len(sc_b) > i and abs(sc_a[i][2] - sc_b[i][2]) >= 5.0
            for i in range(min(len(sc_a), len(sc_b)))
        )
        if (bsha, sig) in history_sigs or (sig == prev_sig and delta < 0.05):
            return "B", "repeated_noop_scroll"
        if moved or delta >= 0.35:
            return "A", "scroll_moved_viewport"
        if delta < 0.05:
            return "B", "scroll_no_movement"
        return "D", "scroll_ambiguous_microchange"

    if act == "type":
        txt = str(parsed.get("text") or "")
        if tag_b not in ("input", "textarea") and str(actv_b.get("contenteditable") or "").lower() != "true":
            return "C", "unfocused_typing_non_input"
        if "create_event" in url_b and txt in typed_on_create:
            return "C", "duplicate_or_wrong_field_create_event"
        if "create_event" not in url_b and txt and (f"{url_b}::{txt}" in submitted_noncal_texts):
            return "C", "duplicate_redundant_submission_type"
        if val_a is not None and txt and txt.strip() in str(val_a):
            if val_b is None or str(val_a) != str(val_b) or delta >= 0.03:
                return "A", "type_value_updated"
        if val_a is not None and val_b is not None and str(val_a) != str(val_b):
            return "A", "type_active_value_changed"
        if delta < 0.04 and val_a == val_b:
            return "B", "type_no_observable_change"
        return "D", "type_microchange_unconfirmed"

    if act == "key":
        if url_a != url_b or delta >= 0.40:
            if (
                last_typed_text_on_url
                and last_typed_text_on_url[0] == url_b
                and f"{url_b}::{last_typed_text_on_url[1]}" in submitted_noncal_texts
            ):
                return "C", "duplicate_redundant_submission_submit"
            return "A", "key_triggered_transition"
        if delta < 0.05:
            return "B", "key_no_effect"
        return "D", "key_ambiguous_microchange"

    if act == "left_click":
        coord = parsed.get("coordinate") or [0, 0]
        coord_y = float(coord[1]) if len(coord) >= 2 else 0.0
        if "create_event" in url_b and url_a and ("create_event" not in url_a):
            is_submit = any(w in ctx_txt for w in SUBMIT_WORDS) and coord_y < 850
            if not is_submit and (coord_y >= 850 or (ctx_href and "/calendar" not in ctx_href)):
                return "C", "premature_form_abandon_before_submit"

        if (
            any(w in ctx_txt for w in ("add", "send", "submit", "create", "save"))
            and last_typed_text_on_url is not None
            and last_typed_text_on_url[0] == url_b
            and f"{url_b}::{last_typed_text_on_url[1]}" in submitted_noncal_texts
        ):
            return "C", "duplicate_redundant_submission_click"

        is_dock_nav = (ctx_href != "" and url_a != url_b) or (coord_y >= 880 and url_a != url_b)
        if is_dock_nav and url_a in visited_urls and actions_inside_current_url == 0:
            return "C", "empty_app_ping_pong"

        if (bsha, sig) in history_sigs:
            if is_dock_nav or coord_y >= 850 or ctx_href:
                return "C", "repeated_dock_or_nav_cycle"
            return "B", "repeated_same_state_action"

        if ctx_b.get("type") == "checkbox" or hit_b.get("type") == "checkbox" or ctx_role == "checkbox":
            cb_sig = f"{url_b}:{round(coord[0], -1)}:{round(coord_y, -1)}:{ctx_txt[:30]}"
            if cb_sig in seen_checkbox_sigs:
                return "C", "redundant_checkbox_toggle"

        focus_gained_input = (tag_a in ("input", "textarea")) and (
            tag_b != tag_a or actv_a.get("id") != actv_b.get("id")
        )
        if delta < 0.05 and url_a == url_b and not focus_gained_input and actv_a == actv_b:
            return "B", "click_no_observable_change"

        interactive = (
            ctx_tag in INTERACTIVE_TAGS
            or hit_tag in INTERACTIVE_TAGS
            or ctx_role in INTERACTIVE_ROLES
            or bool(ctx_href)
            or bool(ab.get("contextContainsTarget"))
        )
        if url_a != url_b or focus_gained_input:
            return "A", "click_nav_or_input_focus"
        if interactive and (delta >= 0.25 or (ctx_b.get("type") == "checkbox" and delta >= 0.08)):
            return "A", "click_interactive_state_change"
        if not interactive and delta < 0.15:
            return "B", "click_noninteractive_minor_drift"
        return "D", "click_ambiguous_visual_change"

    return "D", "other_action_ambiguous"


def extract_labeled_steps_from_episode(
    episode: Mapping[str, Any], split_tag: str
) -> list[dict[str, Any]]:
    """Extract objective step-level ``A/B/C/D`` records from an episode trajectory."""
    steps = episode.get("steps") or []
    history: list[dict[str, Any]] = []
    history_sigs: set[tuple[str, str]] = set()
    typed_on_create: list[str] = []
    submitted_noncal_texts: set[str] = set()
    last_typed_text_on_url: tuple[str, str] | None = None
    seen_checkbox_sigs: set[str] = set()
    visited_urls: dict[str, int] = {}
    actions_inside_current_url = 0
    prev_sig: str | None = None
    out: list[dict[str, Any]] = []

    for step in steps:
        if not step.get("before_image") or not step.get("after_image"):
            pa = step.get("parsed_action")
            if pa:
                history.append(pa)
            continue

        lbl, reason = label_step(
            step,
            history_sigs,
            prev_sig,
            typed_on_create,
            submitted_noncal_texts,
            last_typed_text_on_url,
            seen_checkbox_sigs,
            visited_urls,
            actions_inside_current_url,
        )
        pa = step.get("parsed_action") or {}
        ab = (step.get("audit_before") or {}).get("state") or {}
        aa = (step.get("audit_after") or {}).get("state") or {}
        url_b = str(ab.get("url") or "")
        url_a = str(aa.get("url") or "")

        if lbl != "SKIP":
            source_run = episode.get("arm") or episode.get("run_tag") or split_tag
            out.append(
                {
                    "sample_id": f"{split_tag}:{source_run}:{episode['task_id']}:{step['step_idx']}",
                    "source_run": source_run,
                    "split_tag": split_tag,
                    "task_id": episode["task_id"],
                    "parent_task_id": episode.get("parent_task_id")
                    or episode.get("component")
                    or episode["task_id"],
                    "component": episode.get("component") or episode["task_id"],
                    "step_idx": int(step["step_idx"]),
                    "instruction": episode["goal"],
                    "visible_history": list(history[-4:]),
                    "before_image": step["before_image"],
                    "after_image": step["after_image"],
                    "before_sha256": step.get("before_sha256"),
                    "after_sha256": step.get("after_sha256"),
                    "action": pa,
                    "visual_delta_96x54": step.get("visual_delta_96x54"),
                    "url_before": url_b,
                    "label": lbl,
                    "label_name": LABEL_NAMES[lbl],
                    "label_reason": reason,
                }
            )

        sig = json.dumps(pa, sort_keys=True)
        bsha = str(step.get("before_sha256") or "")
        history_sigs.add((bsha, sig))
        prev_sig = sig
        if url_b:
            visited_urls[url_b] = visited_urls.get(url_b, 0) + 1
        if url_a != url_b:
            actions_inside_current_url = 0
        elif pa.get("action") in ("type", "scroll", "key") or (
            pa.get("action") == "left_click"
            and float((pa.get("coordinate") or [0, 0])[1]) < 850
        ):
            actions_inside_current_url += 1

        if pa.get("action") == "type":
            txt = str(pa.get("text") or "")
            if "create_event" in url_b:
                typed_on_create.append(txt)
            else:
                last_typed_text_on_url = (url_b, txt)
        elif pa.get("action") in ("left_click", "key"):
            ctx_b = ab.get("context") or {}
            hit_b = ab.get("hit") or {}
            ctx_txt = (str(ctx_b.get("text") or "") + " " + str(hit_b.get("text") or "")).strip().lower()
            is_sub_action = (pa.get("action") == "key") or (
                float((pa.get("coordinate") or [0, 0])[1]) < 850
                and (
                    any(w in ctx_txt for w in ("add", "send", "submit", "save", "create"))
                    or float(step.get("visual_delta_96x54") or 0.0) >= 0.4
                )
            )
            if is_sub_action and last_typed_text_on_url and last_typed_text_on_url[0] == url_b:
                submitted_noncal_texts.add(f"{url_b}::{last_typed_text_on_url[1]}")
            if pa.get("action") == "left_click" and (
                ctx_b.get("type") == "checkbox" or hit_b.get("type") == "checkbox"
            ):
                coord = pa.get("coordinate") or [0, 0]
                seen_checkbox_sigs.add(f"{url_b}:{round(coord[0], -1)}:{round(coord[1], -1)}:{ctx_txt[:30]}")

        if pa:
            history.append(pa)

    return out
