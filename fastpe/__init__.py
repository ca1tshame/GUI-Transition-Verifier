"""Operator-level acceleration for vision patch embedding."""

from .fast_patch_embed import (
    FastPatchEmbed,
    ParityBudget,
    ParityReport,
    apply_fastpe,
    audit_parity,
    patch_embed_geometry,
)

__all__ = [
    "FastPatchEmbed",
    "ParityBudget",
    "ParityReport",
    "apply_fastpe",
    "audit_parity",
    "patch_embed_geometry",
]
