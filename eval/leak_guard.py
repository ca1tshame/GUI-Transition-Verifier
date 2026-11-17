"""Leakage guard for protected evaluation sets.

A protected test set is only protected if three things hold:

1. **Identity isolation** - no training or collection record may reuse a
   protected ``(task, seed)`` instance.
2. **Pixel isolation** - no frame may be byte-identical to a protected frame.
3. **Near-duplicate isolation** - no frame may be visually indistinguishable
   from a protected frame (same screen, re-encoded or slightly shifted).

The third check is a conservative rejection rule, not a proof of semantic
difference. Two different tasks can share an identical-looking screen; that is
exactly why the rule rejects the whole task/seed group once any frame trips it,
instead of dropping single episodes.

This module contains no data and no manifest. It is the reusable logic only:
plug in your own fingerprints and identities.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "FrameFingerprint",
    "Identity",
    "ProtectedIndex",
    "ProtectedManifest",
    "ScreenResult",
    "dhash64",
    "file_sha256",
    "gray32_mean_abs",
    "hamming",
    "identity_overlap",
    "image_fingerprint",
    "is_near_duplicate",
    "load_protected_manifest",
    "propagate_group_rejection",
    "screen_episode",
]

NEAR_DUPLICATE_HAMMING = 3
NEAR_DUPLICATE_MEAN_ABS = 2.0
PROTECTED_STATUS = "protected_for_test_no_training"


@dataclass(frozen=True, order=True)
class Identity:
    """A protected instance: task name plus seed."""

    task: str
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.task, str) or not self.task:
            raise ValueError("task must be a non-empty string")


@dataclass(frozen=True)
class FrameFingerprint:
    """Three views of one frame, cheapest first.

    ``sha256`` catches exact copies, ``dhash`` catches re-encodes, and
    ``gray32`` (a 32x32 grayscale vector) catches small shifts that dHash
    tolerates.
    """

    sha256: str
    dhash: int
    gray32: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.sha256) != 64:
            raise ValueError("sha256 must be a 64-character hex digest")
        if len(self.gray32) != 32 * 32:
            raise ValueError("gray32 must contain 1024 values")


@dataclass
class ScreenResult:
    episode_id: str
    identity: Identity
    reasons: set[str]
    frames_checked: int = 0

    @property
    def rejected(self) -> bool:
        return bool(self.reasons)


def file_sha256(path: str | pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _grid(gray: Sequence[Any]) -> list[list[float]]:
    """Accept a flat 72-value sequence or an 8x9 matrix and return 8x9."""
    if len(gray) == 72:
        flat = [float(v) for v in gray]
        return [flat[row * 9 : (row + 1) * 9] for row in range(8)]
    if len(gray) == 8 and all(len(row) == 9 for row in gray):
        return [[float(v) for v in row] for row in gray]
    raise ValueError("expected 8x9 grayscale matrix or 72 flat values")


def dhash64(gray: Sequence[Any]) -> int:
    """64-bit difference hash over a 9x8 grayscale image.

    Each bit compares a pixel with its right neighbour, row-major, MSB first.
    """
    rows = _grid(gray)
    bits = 0
    for row in rows:
        for x in range(8):
            bits = (bits << 1) | int(row[x] > row[x + 1])
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def gray32_mean_abs(a: Sequence[float], b: Sequence[float]) -> float:
    """Mean absolute difference between two 32x32 grayscale vectors."""
    if len(a) != len(b):
        raise ValueError("gray32 length mismatch")
    if not a:
        raise ValueError("empty vector")
    return sum(abs(float(x) - float(y)) for x, y in zip(a, b)) / len(a)


def is_near_duplicate(
    left: FrameFingerprint,
    right: FrameFingerprint,
    *,
    max_hamming: int = NEAR_DUPLICATE_HAMMING,
    max_mean_abs: float = NEAR_DUPLICATE_MEAN_ABS,
) -> bool:
    """Conservative visual near-duplicate rule.

    Two frames match when their dHash distance is at most ``max_hamming``
    **and** their mean absolute 32x32 grayscale difference is at most
    ``max_mean_abs``. Requiring both keeps a flipped or inverted screen from
    matching on hash alone.
    """
    if hamming(left.dhash, right.dhash) > max_hamming:
        return False
    return gray32_mean_abs(left.gray32, right.gray32) <= max_mean_abs


class ProtectedIndex:
    """Exact and near-duplicate index over protected frames."""

    def __init__(self) -> None:
        self._by_sha: dict[str, list[FrameFingerprint]] = {}
        self._all: list[FrameFingerprint] = []

    def __len__(self) -> int:
        return len(self._all)

    @property
    def frames(self) -> tuple[FrameFingerprint, ...]:
        return tuple(self._all)

    def add(self, fingerprint: FrameFingerprint) -> None:
        self._by_sha.setdefault(fingerprint.sha256, []).append(fingerprint)
        self._all.append(fingerprint)

    def match(self, fingerprint: FrameFingerprint) -> str | None:
        """Return ``exact_protected_image``, ``near_protected_image`` or ``None``."""
        if fingerprint.sha256 in self._by_sha:
            return "exact_protected_image"
        for protected in self._all:
            if is_near_duplicate(fingerprint, protected):
                return "near_protected_image"
        return None


@dataclass(frozen=True)
class ProtectedManifest:
    """Validated protected manifest and its identity set."""

    path: str
    sha256: str
    identities: frozenset[Identity]
    episode_ids: frozenset[str]

    @property
    def n_instances(self) -> int:
        return len(self.identities)


def load_protected_manifest(path: str | pathlib.Path) -> ProtectedManifest:
    """Load and validate a protected manifest.

    Required shape::

        {"status": "protected_for_test_no_training",
         "episodes": [{"episode_id": "...", "task": "...", "seed": 0}, ...]}

    Duplicate episode ids, duplicate ``(task, seed)`` pairs, an empty episode
    list or a different status all raise. A guard that silently accepts a
    malformed manifest is worse than no guard.
    """
    manifest_path = pathlib.Path(path)
    raw = manifest_path.read_bytes()
    document = json.loads(raw.decode("utf-8"))

    if document.get("status") != PROTECTED_STATUS:
        raise ValueError(f"manifest status must be {PROTECTED_STATUS!r}")
    episodes = document.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("manifest must contain a non-empty episodes list")

    identities = set()
    episode_ids = set()
    for episode in episodes:
        identity = Identity(str(episode["task"]), int(episode["seed"]))
        if identity in identities:
            raise ValueError(f"duplicate protected identity: {identity}")
        episode_id = str(episode["episode_id"])
        if episode_id in episode_ids:
            raise ValueError(f"duplicate episode id: {episode_id}")
        identities.add(identity)
        episode_ids.add(episode_id)

    return ProtectedManifest(
        path=str(manifest_path),
        sha256=hashlib.sha256(raw).hexdigest(),
        identities=frozenset(identities),
        episode_ids=frozenset(episode_ids),
    )


def identity_overlap(
    records: Iterable[Mapping[str, Any]],
    protected: Iterable[Identity],
) -> list[dict[str, Any]]:
    """Return records whose direct ``(task, seed)`` fields hit the protected set.

    This is a direct-field screen only. It deliberately does not scan nested
    structures or images; those need a separate, stricter pass.
    """
    protected_set = set(protected)
    hits = []
    for record in records:
        task = record.get("task", record.get("task_id"))
        seed = record.get("seed")
        if task is None or seed is None:
            continue
        identity = Identity(str(task), int(seed))
        if identity in protected_set:
            hits.append({"task": identity.task, "seed": identity.seed})
    return hits


def screen_episode(
    episode_id: str,
    identity: Identity,
    frames: Sequence[FrameFingerprint],
    index: ProtectedIndex,
    protected_identities: Iterable[Identity],
) -> ScreenResult:
    """Screen one episode against identity and pixel protection."""
    reasons: set[str] = set()
    if identity in set(protected_identities):
        reasons.add("protected_task_seed")
    for frame in frames:
        match = index.match(frame)
        if match:
            reasons.add(match)
    return ScreenResult(
        episode_id=episode_id,
        identity=identity,
        reasons=reasons,
        frames_checked=len(frames),
    )


def propagate_group_rejection(
    results: Sequence[ScreenResult],
    episode_groups: Mapping[Identity, Sequence[str]],
) -> list[dict[str, Any]]:
    """Reject every episode sharing a protected task/seed group.

    Args:
        results: per-episode screening results.
        episode_groups: ``identity -> episode ids`` for all collected episodes,
            including clean ones.

    Returns:
        Sorted rejection records ``{"episode_id", "reasons"}``. An episode that
        shares a ``(task, seed)`` group with a rejected episode gains
        ``related_task_seed_group`` even when its own frames looked clean. The
        group is the unit of rejection because a near-duplicate screen proves
        nothing about the other frames of the same instance.
    """
    rejected: dict[str, set[str]] = {
        result.episode_id: set(result.reasons) for result in results if result.reasons
    }
    flagged_groups = {result.identity for result in results if result.reasons}
    for identity in flagged_groups:
        for episode_id in episode_groups.get(identity, ()):
            if episode_id in rejected:
                continue
            rejected[episode_id] = {"related_task_seed_group"}
    return [
        {"episode_id": episode_id, "reasons": sorted(reasons)}
        for episode_id, reasons in sorted(rejected.items())
    ]


def image_fingerprint(path: str | pathlib.Path) -> FrameFingerprint:
    """Compute a fingerprint from an image file.

    Requires Pillow at call time only; the rest of the module is dependency
    free. Images are converted to 8-bit grayscale before hashing.
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("image_fingerprint requires Pillow: pip install pillow") from exc

    image_path = pathlib.Path(path)
    with Image.open(image_path) as image:
        gray = image.convert("L")
        tiny = list(gray.resize((9, 8), Image.BILINEAR).getdata())
        coarse = list(gray.resize((32, 32), Image.BILINEAR).getdata())
    return FrameFingerprint(
        sha256=file_sha256(image_path),
        dhash=dhash64(tiny),
        gray32=tuple(float(v) for v in coarse),
    )
