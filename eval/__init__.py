"""Deterministic evaluation and leakage protection."""

from .leak_guard import (
    FrameFingerprint,
    Identity,
    ProtectedIndex,
    dhash64,
    identity_overlap,
    is_near_duplicate,
    load_protected_manifest,
    propagate_group_rejection,
    screen_episode,
)
from .metrics_core import (
    CLASSES,
    LETTERS,
    auc,
    cluster_bootstrap_ci,
    full_report,
    macro_f1,
    risk_auc_suite,
)

__all__ = [
    "CLASSES",
    "LETTERS",
    "FrameFingerprint",
    "Identity",
    "ProtectedIndex",
    "auc",
    "cluster_bootstrap_ci",
    "dhash64",
    "full_report",
    "identity_overlap",
    "is_near_duplicate",
    "load_protected_manifest",
    "macro_f1",
    "propagate_group_rejection",
    "risk_auc_suite",
    "screen_episode",
]
