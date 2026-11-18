"""Operator-level acceleration and cascade fallback for vision patch embedding."""

from .cascade import (
    ALERT_SAFE_CONFIG,
    FULL_PARITY_CONFIG,
    CascadeConfig,
    CascadeRunner,
    set_fastpe_mode,
    should_fallback,
)
from .fast_patch_embed import (
    FastPatchEmbed,
    ParityBudget,
    ParityReport,
    apply_fastpe,
    audit_parity,
    patch_embed_geometry,
)

__all__ = [
    "ALERT_SAFE_CONFIG",
    "FULL_PARITY_CONFIG",
    "CascadeConfig",
    "CascadeRunner",
    "FastPatchEmbed",
    "ParityBudget",
    "ParityReport",
    "apply_fastpe",
    "audit_parity",
    "patch_embed_geometry",
    "set_fastpe_mode",
    "should_fallback",
]
