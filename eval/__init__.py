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
from .v5_recompute import (
    exact_mcnemar_from_booleans,
    exact_mcnemar_from_counts,
    metrics_from_confusion_matrix,
    verify_all_results,
    wilson_ci,
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
    "exact_mcnemar_from_booleans",
    "exact_mcnemar_from_counts",
    "full_report",
    "identity_overlap",
    "is_near_duplicate",
    "load_protected_manifest",
    "macro_f1",
    "metrics_from_confusion_matrix",
    "propagate_group_rejection",
    "risk_auc_suite",
    "screen_episode",
    "verify_all_results",
    "wilson_ci",
]
