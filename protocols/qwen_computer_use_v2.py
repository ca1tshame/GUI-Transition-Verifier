"""Versioned Qwen ComputerUse tool protocol with explicit coordinate spaces.

Why a versioned adapter
-----------------------
Qwen3-VL emits tool calls as a strict JSON object wrapped in
``<tool_call>...</tool_call>``. Two details decide whether the action is
executable at all:

1. **The tool must actually be declared** in ``apply_chat_template(tools=...)``.
   Without the declaration the model falls back to prose and no call parses.
2. **The coordinate space must be declared.** Qwen's official ComputerUse tool
   uses absolute screenshot pixels; a 0-999 normalized variant is a deliberate
   project extension. Mixing the two silently moves every click.

The second trap is double scaling. BrowserGym's ``map_coordinates`` already
converts screenshot pixels to CSS pixels. A normalized coordinate must be
converted to *screenshot* pixels exactly once and then handed to BrowserGym
untouched.

This module therefore:
* declares the schema (:data:`TOOL_SCHEMA`),
* parses exactly one complete tool call and rejects everything else,
* converts coordinates once, with bounds checks that fail closed,
* validates action payloads before an environment ever sees them.

Protocol identifier: ``qwen-computer-use-browsergym-v2.1``.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

__all__ = [
    "ACTION_NAMES",
    "ALLOWED_KEYS",
    "COORDINATE_SPACES",
    "MAX_SCROLL_PIXELS",
    "MAX_TEXT_LENGTH",
    "PROTOCOL_ID",
    "SCHEMA_VERSION",
    "TOOL_SCHEMA",
    "Geometry",
    "ParsedAction",
    "ToolCallError",
    "parse_tool_call",
    "render_tool_call",
    "to_screenshot_coords",
]

SCHEMA_VERSION = "qwen-computer-use-browsergym-v2.1"
PROTOCOL_ID = SCHEMA_VERSION

COORDINATE_SPACES = ("pixel", "normalized")
ACTION_NAMES = ("left_click", "type", "scroll", "key", "terminate")

MAX_TEXT_LENGTH = 200
MAX_SCROLL_PIXELS = 600

ALLOWED_KEYS = frozenset(
    {
        "Enter",
        "Tab",
        "ArrowDown",
        "ArrowUp",
        "ArrowLeft",
        "ArrowRight",
        "Backspace",
        "Delete",
        "Escape",
        "Space",
        "ControlOrMeta+a",
    }
)

TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "computer_use",
        "description": (
            "Use the mouse and keyboard on the current screenshot. "
            "Coordinates must use the screenshot coordinate space declared by the harness."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": list(ACTION_NAMES)},
                "coordinate": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "text": {"type": "string"},
                "keys": {"type": "array", "items": {"type": "string"}},
                "pixels": {"type": "integer"},
                "status": {"type": "string", "enum": ["success", "failure"]},
            },
            "required": ["action"],
        },
    },
}

_TOOL_CALL_RE = re.compile(
    r"\s*<tool_call>\s*(\{.*?\})\s*</tool_call>\s*(?:<\|im_end\|>)?\s*",
    re.DOTALL,
)


class ToolCallError(ValueError):
    """Raised when a raw model output is not a valid, executable tool call."""


@dataclass(frozen=True)
class Geometry:
    """Screenshot and viewport size in pixels.

    ``viewport`` is validated but never used for coordinate math: BrowserGym
    performs the screenshot-to-CSS mapping internally.
    """

    width: int
    height: int

    def __post_init__(self) -> None:
        if min(int(self.width), int(self.height)) <= 0:
            raise ValueError("geometry must be positive")


@dataclass(frozen=True)
class ParsedAction:
    """A validated action and its BrowserGym command string."""

    action: str
    command: str | None
    arguments: Mapping[str, Any] = field(default_factory=dict)
    coordinate: tuple[float, float] | None = None
    text: str | None = None
    keys: tuple[str, ...] | None = None
    pixels: int | None = None
    status: str | None = None

    @property
    def terminates(self) -> bool:
        return self.action == "terminate"


def _finite_pair(value: Any) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ToolCallError("coordinate must be a two-element list")
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ToolCallError("coordinate must contain numbers")
        if not math.isfinite(item):
            raise ToolCallError("coordinate must be finite")
    return float(value[0]), float(value[1])


def to_screenshot_coords(
    xy: Sequence[float] | None,
    screenshot: Geometry,
    space: str,
    *,
    viewport: Geometry | None = None,
) -> tuple[float, float]:
    """Convert a declared coordinate into screenshot pixels.

    Args:
        xy: two-element coordinate as emitted by the model.
        screenshot: size of the image the model actually saw.
        space: ``"pixel"`` (absolute screenshot pixels) or ``"normalized"``
            (integers in ``[0, 999]``, mapped linearly onto the screenshot).
        viewport: optional browser viewport. Accepted for call-site symmetry
            and validated, deliberately **not** applied here.

    Returns:
        ``(x, y)`` in screenshot pixel space.

    Raises:
        ToolCallError: unknown space, non-finite value, or out-of-bounds
            coordinate. Conversion never clamps.
    """
    if space not in COORDINATE_SPACES:
        raise ToolCallError(f"unknown coordinate space: {space!r}")
    if viewport is not None:
        Geometry(int(viewport.width), int(viewport.height))
    if xy is None:
        raise ToolCallError("missing coordinate")
    x, y = _finite_pair(list(xy))

    width, height = screenshot.width, screenshot.height
    if space == "pixel":
        if not (0 <= x < width and 0 <= y < height):
            raise ToolCallError("coordinate out of screenshot bounds")
        return x, y

    # normalized 0-999: half-open interval, 999 is the last valid cell index.
    if not (0 <= x < 1000 and 0 <= y < 1000):
        raise ToolCallError("coordinate out of normalized [0, 999] bounds")
    return x * width / 1000.0, y * height / 1000.0


def parse_tool_call(
    raw: str,
    screenshot: Geometry,
    *,
    space: str = "normalized",
    viewport: Geometry | None = None,
) -> ParsedAction:
    """Parse one complete ``<tool_call>`` block into an executable action.

    The parser is intentionally strict: partial calls, multiple calls, trailing
    prose, unknown tool names and malformed payloads all raise
    :class:`ToolCallError` instead of being repaired. A verifier that guesses
    what the model meant cannot measure the model.

    Returns:
        A :class:`ParsedAction` carrying both structured fields and the exact
        BrowserGym command string.
    """
    if not isinstance(raw, str):
        raise ToolCallError("raw model output must be a string")
    match = _TOOL_CALL_RE.fullmatch(raw)
    if not match:
        raise ToolCallError("expected a single complete <tool_call> block")

    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ToolCallError("invalid tool-call JSON") from exc

    if not isinstance(payload, dict):
        raise ToolCallError("tool call must be a JSON object")
    if payload.get("name") != "computer_use":
        raise ToolCallError("unexpected tool name")
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        raise ToolCallError("arguments must be a JSON object")

    action = arguments.get("action")
    if action not in ACTION_NAMES:
        raise ToolCallError(f"unsupported action: {action!r}")

    if action == "left_click":
        x, y = to_screenshot_coords(
            arguments.get("coordinate"), screenshot, space, viewport=viewport
        )
        return ParsedAction(
            action=action,
            command=f"mouse_click({x!r}, {y!r})",
            arguments=arguments,
            coordinate=(x, y),
        )

    if action == "scroll":
        x, y = to_screenshot_coords(
            arguments.get("coordinate"), screenshot, space, viewport=viewport
        )
        pixels = arguments.get("pixels")
        if isinstance(pixels, bool) or not isinstance(pixels, int):
            raise ToolCallError("scroll pixels must be an integer")
        if not 0 < abs(pixels) <= MAX_SCROLL_PIXELS:
            raise ToolCallError("scroll pixels out of range")
        # Official ComputerUse: positive means "up". BrowserGym: positive means
        # "down". The sign is flipped exactly here, and nowhere else.
        return ParsedAction(
            action=action,
            command=f"scroll_at({x!r}, {y!r}, 0, {-pixels!r})",
            arguments=arguments,
            coordinate=(x, y),
            pixels=pixels,
        )

    if action == "type":
        text = arguments.get("text")
        if not isinstance(text, str) or not text:
            raise ToolCallError("type requires non-empty text")
        if len(text) > MAX_TEXT_LENGTH:
            raise ToolCallError("type text exceeds length budget")
        return ParsedAction(
            action=action,
            command=f"keyboard_type({text!r})",
            arguments=arguments,
            text=text,
        )

    if action == "key":
        keys = arguments.get("keys")
        if not isinstance(keys, list) or len(keys) != 1:
            raise ToolCallError("key requires exactly one key")
        key = keys[0]
        if key not in ALLOWED_KEYS:
            raise ToolCallError(f"key not in the executable whitelist: {key!r}")
        return ParsedAction(
            action=action,
            command=f"keyboard_press({key!r})",
            arguments=arguments,
            keys=(key,),
        )

    status = arguments.get("status")
    if status not in ("success", "failure"):
        raise ToolCallError("terminate requires status success or failure")
    return ParsedAction(
        action=action,
        command=None,
        arguments=arguments,
        status=status,
    )


def render_tool_call(arguments: Mapping[str, Any]) -> str:
    """Render a canonical single tool call, used by tests and fixtures."""
    if not isinstance(arguments, Mapping):
        raise ToolCallError("arguments must be a mapping")
    payload = {"name": "computer_use", "arguments": dict(arguments)}
    return "<tool_call>" + json.dumps(payload, separators=(",", ":")) + "</tool_call>"
