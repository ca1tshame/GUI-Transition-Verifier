"""Build a verifiable, sanitized rollout GIF from an existing OpenApps episode trace.

Reads the real screenshots and episode JSON saved by ``policy/eval_openapps.py``
(without modifying model behavior or re-running frozen Heldout tasks) and writes
an annotated GIF plus a SHA-256 provenance manifest.

Usage::

    python -B tools/record_openapps_gif.py \
        --episode runs/eval_demo/dev_demo/episodes/<TASK_ID>.json \
        --frames-dir runs/eval_demo/dev_demo/frames/<TASK_ID> \
        --out assets/openapps_dev_rollout.gif
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _format_action(act: dict[str, Any]) -> str:
    name = str(act.get("action") or "?")
    if name in ("left_click", "right_click", "double_click", "middle_click"):
        return f"{name}({act.get('x')}, {act.get('y')})"
    if name == "type":
        txt = str(act.get("text") or "")
        if len(txt) > 36:
            txt = txt[:33] + "..."
        return f'type("{txt}")'
    if name == "scroll":
        return f"scroll(dx={act.get('dx', 0)}, dy={act.get('dy', 0)})"
    if name == "key":
        return f"key({act.get('keys')})"
    return name


def build_openapps_gif(
    episode_path: Path,
    frames_dir: Path,
    out_path: Path,
    *,
    width: int = 960,
    duration_ms: int = 700,
    allow_heldout: bool = False,
    require_success: bool = True,
) -> dict[str, Any]:
    """Assemble real ``eval_openapps.py`` screenshots into a traced GIF."""
    from PIL import Image, ImageDraw, ImageFont

    ep = json.loads(episode_path.read_text(encoding="utf-8"))
    split = str(ep.get("split") or "")
    if split == "heldout" and not allow_heldout:
        raise ValueError("Refusing to export frozen heldout episode without --allow-heldout")
    if require_success and not ep.get("oracle_terminal_success"):
        raise ValueError("Refusing non-successful episode when require_success=True")

    steps = ep.get("steps") or []
    if not steps:
        raise ValueError("Episode has zero recorded steps")

    font = ImageFont.load_default()
    frames: list[Image.Image] = []
    final_after: Path | None = None

    for i, st in enumerate(steps):
        img_path = frames_dir / f"{i:03d}_before.png"
        if not img_path.is_file():
            raise FileNotFoundError(f"Missing real screenshot: {img_path}")
        im = Image.open(img_path).convert("RGB")
        orig_w, orig_h = im.size
        scale = width / float(orig_w)
        im = im.resize((width, int(round(orig_h * scale))), Image.BILINEAR)
        draw = ImageDraw.Draw(im)

        act = st.get("parsed_action") or {}
        if act.get("x") is not None and act.get("y") is not None:
            cx = int(round(float(act["x"]) * scale))
            cy = int(round(float(act["y"]) * scale))
            r = 10
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(255, 75, 75), width=3)
            draw.line((cx - 14, cy, cx + 14, cy), fill=(255, 75, 75), width=2)
            draw.line((cx, cy - 14, cx, cy + 14), fill=(255, 75, 75), width=2)

        sub = ep.get("subtask_eval") or {}
        banner = (
            f"Step {i + 1}/{len(steps)} | {_format_action(act)} | "
            f"Split: {split or 'dev'} | Subtasks: {sub.get('subtasks_passed', '?')}/{sub.get('subtasks_total', '?')}"
        )
        draw.rectangle((0, 0, im.width, 28), fill=(15, 23, 42))
        draw.text((10, 8), banner, fill=(248, 250, 252), font=font)
        frames.append(im.convert("P", palette=Image.ADAPTIVE))
        cand_after = frames_dir / f"{i:03d}_after.png"
        if cand_after.is_file():
            final_after = cand_after

    if final_after is not None:
        im = Image.open(final_after).convert("RGB")
        scale = width / float(im.width)
        im = im.resize((width, int(round(im.height * scale))), Image.BILINEAR)
        draw = ImageDraw.Draw(im)
        sub = ep.get("subtask_eval") or {}
        status = "PASS (Oracle Verified)" if ep.get("oracle_terminal_success") else str(ep.get("status"))
        banner = f"Final State | {status} | Subtasks: {sub.get('subtasks_passed', '?')}/{sub.get('subtasks_total', '?')}"
        draw.rectangle((0, 0, im.width, 28), fill=(20, 83, 45))
        draw.text((10, 8), banner, fill=(248, 250, 252), font=font)
        frames.append(im.convert("P", palette=Image.ADAPTIVE))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    durations = [duration_ms] * (len(frames) - 1) + [max(duration_ms * 2, 1400)]
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )

    meta = {
        "task_id": ep.get("task_id"),
        "split": split,
        "mode": ep.get("mode"),
        "steps": len(steps),
        "oracle_terminal_success": bool(ep.get("oracle_terminal_success")),
        "episode_sha256": _sha256(episode_path),
        "gif_sha256": _sha256(out_path),
        "gif_bytes": out_path.stat().st_size,
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--duration-ms", type=int, default=700)
    ap.add_argument("--allow-heldout", action="store_true")
    ap.add_argument("--allow-fail", action="store_true")
    args = ap.parse_args(argv)
    meta = build_openapps_gif(
        Path(args.episode),
        Path(args.frames_dir),
        Path(args.out),
        width=args.width,
        duration_ms=args.duration_ms,
        allow_heldout=args.allow_heldout,
        require_success=not args.allow_fail,
    )
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
