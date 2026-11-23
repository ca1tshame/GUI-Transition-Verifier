"""CPU unit tests for tools/record_openapps_gif.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.record_openapps_gif import _format_action, build_openapps_gif


def test_format_action_variants():
    assert _format_action({"action": "left_click", "x": 10, "y": 20}) == "left_click(10, 20)"
    assert _format_action({"action": "scroll", "dx": 0, "dy": -300}) == "scroll(dx=0, dy=-300)"
    assert _format_action({"action": "type", "text": "x" * 50}).endswith('...")')


def test_record_openapps_gif_builds_and_guards_heldout(tmp_path: Path):
    PIL_Image = pytest.importorskip("PIL.Image")

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for i in range(2):
        PIL_Image.new("RGB", (320, 180), color=(20 * (i + 1), 40, 80)).save(frames_dir / f"{i:03d}_before.png")
        PIL_Image.new("RGB", (320, 180), color=(40, 90, 50)).save(frames_dir / f"{i:03d}_after.png")

    ep_dev = {
        "task_id": "demo_task_dev",
        "split": "dev",
        "mode": "policy_only",
        "oracle_terminal_success": True,
        "subtask_eval": {"subtasks_passed": 2, "subtasks_total": 2},
        "steps": [
            {"step_idx": 0, "parsed_action": {"action": "left_click", "x": 100, "y": 80}},
            {"step_idx": 1, "parsed_action": {"action": "type", "text": "hello"}},
        ],
    }
    ep_path = tmp_path / "ep_dev.json"
    ep_path.write_text(json.dumps(ep_dev), encoding="utf-8")

    out_gif = tmp_path / "out.gif"
    meta = build_openapps_gif(ep_path, frames_dir, out_gif, width=320, duration_ms=200)
    assert out_gif.is_file()
    assert out_gif.with_suffix(".json").is_file()
    assert meta["steps"] == 2
    assert meta["oracle_terminal_success"] is True

    ep_heldout = dict(ep_dev, split="heldout")
    held_path = tmp_path / "ep_heldout.json"
    held_path.write_text(json.dumps(ep_heldout), encoding="utf-8")
    with pytest.raises(ValueError, match="heldout"):
        build_openapps_gif(held_path, frames_dir, tmp_path / "held.gif")
