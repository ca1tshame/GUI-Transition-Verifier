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
* rejects booleans masquerading as integers across coordinates, ``pixels``,
  ``time``, and :class:`Geometry` dimensions,
* rejects undeclared extra keys in ``arguments`` per action,
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
    "ALLOWED_ARGUMENT_KEYS",
    "ALLOWED_KEYS",
    "COORDINATE_SPACES",
    "MAX_SCROLL_PIXELS",
    "MAX_TEXT_LENGTH",
    "MAX_WAIT_TIME",
    "PROTOCOL_ID",
    "SCHEMA_VERSION",
    "TOOL_SCHEMA",
    "Geometry",
    "ParsedAction",
    "ProtocolError",
    "ToolCallError",
    "parse_tool_call",
    "render_tool_call",
    "to_screenshot_coords",
]

SCHEMA_VERSION = "qwen-computer-use-browsergym-v2.1"
PROTOCOL_ID = SCHEMA_VERSION

COORDINATE_SPACES = ("pixel", "normalized")
ACTION_NAMES = ("left_click", "type", "scroll", "key", "wait", "terminate")

MAX_TEXT_LENGTH = 200
MAX_SCROLL_PIXELS = 600
MAX_WAIT_TIME = 30.0

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

ALLOWED_ARGUMENT_KEYS: dict[str, frozenset[str]] = {
    "left_click": frozenset({"action", "coordinate"}),
    "type": frozenset({"action", "text"}),
    "scroll": frozenset({"action", "coordinate", "pixels"}),
    "key": frozenset({"action", "keys"}),
    "wait": frozenset({"action", "time"}),
    "terminate": frozenset({"action", "status"}),
}

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
                "time": {"type": "number"},
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


class ProtocolError(ValueError):
    """Raised when a raw model output or geometry violates the tool protocol."""


# Backward-compatible alias used across existing callers and tests.
ToolCallError = ProtocolError


def _require_strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"{name} must be an integer, not {type(value).__name__}")
    return value


def _require_finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{name} must be a number, not {type(value).__name__}")
    fval = float(value)
    if not math.isfinite(fval):
        raise ProtocolError(f"{name} must be finite")
    return fval


@dataclass(frozen=True)
class Geometry:
    """Screenshot and viewport size in pixels.

    ``viewport`` is validated but never used for coordinate math: BrowserGym
    performs the screenshot-to-CSS mapping internally.
    """

    width: int
    height: int

    def __post_init__(self) -> None:
        w = _require_strict_int(self.width, "geometry width")
        h = _require_strict_int(self.height, "geometry height")
        if min(w, h) <= 0:
            raise ProtocolError("geometry must be positive")


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
    time: float | None = None

    @property
    def terminates(self) -> bool:
        return self.action == "terminate"


def _finite_pair(value: Any) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ProtocolError("coordinate must be a two-element list")
    x = _require_finite_number(value[0], "coordinate[0]")
    y = _require_finite_number(value[1], "coordinate[1]")
    return x, y


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
        ProtocolError: unknown space, boolean or non-finite value, invalid
            geometry, or out-of-bounds coordinate. Conversion never clamps.
    """
    if space not in COORDINATE_SPACES:
        raise ProtocolError(f"unknown coordinate space: {space!r}")
    if not isinstance(screenshot, Geometry):
        raise ProtocolError("screenshot must be a Geometry instance")
    if viewport is not None:
        if not isinstance(viewport, Geometry):
            raise ProtocolError("viewport must be a Geometry instance")
    if xy is None or isinstance(xy, (str, bytes)) or not isinstance(xy, Sequence):
        raise ProtocolError("missing or invalid coordinate sequence")
    x, y = _finite_pair(list(xy))

    width, height = screenshot.width, screenshot.height
    if space == "pixel":
        if not (0 <= x < width and 0 <= y < height):
            raise ProtocolError("coordinate out of screenshot bounds")
        return x, y

    # normalized 0-999: half-open interval, 999 is the last valid cell index.
    if not (0 <= x < 1000 and 0 <= y < 1000):
        raise ProtocolError("coordinate out of normalized [0, 999] bounds")
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
    prose, unknown tool names, extra argument keys, boolean-for-integer values,
    and malformed payloads all raise :class:`ProtocolError` instead of being
    repaired. A verifier that guesses what the model meant cannot measure the
    model.

    Returns:
        A :class:`ParsedAction` carrying both structured fields and the exact
        BrowserGym command string.
    """
    if not isinstance(raw, str):
        raise ProtocolError("raw model output must be a string")
    if not isinstance(screenshot, Geometry):
        raise ProtocolError("screenshot must be a Geometry instance")
    if viewport is not None and not isinstance(viewport, Geometry):
        raise ProtocolError("viewport must be a Geometry instance")

    match = _TOOL_CALL_RE.fullmatch(raw)
    if not match:
        raise ProtocolError("expected a single complete <tool_call> block")

    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ProtocolError("invalid tool-call JSON") from exc

    if not isinstance(payload, dict):
        raise ProtocolError("tool call must be a JSON object")
    if payload.get("name") != "computer_use":
        raise ProtocolError("unexpected tool name")
    extra_payload_keys = sorted(set(payload.keys()) - {"name", "arguments"})
    if extra_payload_keys:
        raise ProtocolError(f"unexpected top-level tool-call key(s): {extra_payload_keys}")

    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        raise ProtocolError("arguments must be a JSON object")

    action = arguments.get("action")
    if not isinstance(action, str) or action not in ACTION_NAMES:
        raise ProtocolError(f"unsupported action: {action!r}")

    allowed_arg_keys = ALLOWED_ARGUMENT_KEYS[action]
    extra_keys = sorted(set(arguments.keys()) - allowed_arg_keys)
    if extra_keys:
        raise ProtocolError(
            f"unexpected argument key(s) for action {action!r}: {extra_keys}"
        )

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
        pixels = _require_strict_int(arguments.get("pixels"), "scroll pixels")
        if not 0 < abs(pixels) <= MAX_SCROLL_PIXELS:
            raise ProtocolError("scroll pixels out of range")
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
            raise ProtocolError("type requires non-empty text")
        if len(text) > MAX_TEXT_LENGTH:
            raise ProtocolError("type text exceeds length budget")
        return ParsedAction(
            action=action,
            command=f"keyboard_type({text!r})",
            arguments=arguments,
            text=text,
        )

    if action == "key":
        keys = arguments.get("keys")
        if not isinstance(keys, list) or len(keys) != 1:
            raise ProtocolError("key requires exactly one key")
        key = keys[0]
        if not isinstance(key, str) or key not in ALLOWED_KEYS:
            raise ProtocolError(f"key not in the executable whitelist: {key!r}")
        return ParsedAction(
            action=action,
            command=f"keyboard_press({key!r})",
            arguments=arguments,
            keys=(key,),
        )

    if action == "wait":
        wait_time = _require_finite_number(arguments.get("time"), "wait time")
        if not 0.0 < wait_time <= MAX_WAIT_TIME:
            raise ProtocolError("wait time out of range")
        wait_ms = int(round(wait_time * 1000.0))
        return ParsedAction(
            action=action,
            command=f"noop({wait_ms})",
            arguments=arguments,
            time=wait_time,
        )

    status = arguments.get("status")
    if not isinstance(status, str) or status not in ("success", "failure"):
        raise ProtocolError("terminate requires status success or failure")
    return ParsedAction(
        action=action,
        command=None,
        arguments=arguments,
        status=status,
    )


def render_tool_call(arguments: Mapping[str, Any]) -> str:
    """Render a canonical single tool call, used by tests and fixtures."""
    if not isinstance(arguments, Mapping):
        raise ProtocolError("arguments must be a mapping")
    payload = {"name": "computer_use", "arguments": dict(arguments)}
    return "<tool_call>" + json.dumps(payload, separators=(",", ":")) + "</tool_call>"
