"""Tool-call protocol adapters for GUI agents."""

from .qwen_computer_use_v2 import (
    COORDINATE_SPACES,
    PROTOCOL_ID,
    TOOL_SCHEMA,
    Geometry,
    ParsedAction,
    ToolCallError,
    parse_tool_call,
    render_tool_call,
    to_screenshot_coords,
)

__all__ = [
    "COORDINATE_SPACES",
    "PROTOCOL_ID",
    "TOOL_SCHEMA",
    "Geometry",
    "ParsedAction",
    "ToolCallError",
    "parse_tool_call",
    "render_tool_call",
    "to_screenshot_coords",
]
