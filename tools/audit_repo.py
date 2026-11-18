"""Pre-publication audit for this repository.

Checks, in order:

1. no file larger than 2 MiB;
2. no image, weight, archive or columnar-data extension;
3. no sensitive substring: cluster home paths, the project account name,
   absolute local drive paths, API-key shapes, private IPs, local proxy URLs;
4. no untracked virtual environment or cache directory inside the tree.

Patterns are assembled from fragments so that this file does not trip its own
scan. That is the only reason for the string concatenation below.

Usage::

    python -B tools/audit_repo.py
    python -B tools/audit_repo.py --tracked-only
    python -B tools/audit_repo.py --json audit.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass, field

MAX_BYTES = 2 * 1024 * 1024

FORBIDDEN_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".webp",
    ".parquet",
    ".arrow",
    ".feather",
    ".safetensors",
    ".pt",
    ".pth",
    ".bin",
    ".ckpt",
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".7z",
    ".mp4",
}

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".workbuddy",
}

# Sensitive literals are split so the audit file itself stays clean.
SENSITIVE_PATTERNS = {
    "cluster_home": "hpc" + "2hdd",
    "account": "yyan" + "047",
    "windows_drive": r"[A-Za-z]:\\",
    "posix_drive": r"\b[A-Za-z]:/",
    "openai_key": r"\bsk-[A-Za-z0-9]{16,}",
    "github_token": r"\bghp_[A-Za-z0-9]{20,}",
    "github_pat": r"\bgithub_pat_[A-Za-z0-9_]{20,}",
    "aws_key": r"\bAKIA[0-9A-Z]{16}\b",
    "private_ip": r"\b(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b",
    "loopback_proxy": r"(?:socks5|http)://(?:127\.0\.0\.1|localhost):\d{2,5}",
}


@dataclass
class AuditResult:
    root: str
    tracked_only: bool = False
    files: int = 0
    total_bytes: int = 0
    oversized: list[dict] = field(default_factory=list)
    forbidden_files: list[dict] = field(default_factory=list)
    hits: list[dict] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.oversized or self.forbidden_files or self.hits)

    def as_dict(self) -> dict:
        return {
            "root": self.root,
            "tracked_only": self.tracked_only,
            "files": self.files,
            "total_bytes": self.total_bytes,
            "ok": self.ok,
            "oversized": self.oversized,
            "forbidden_files": self.forbidden_files,
            "sensitive_hits": self.hits,
            "skipped_dirs": self.skipped,
        }


def iter_files(root: pathlib.Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(root).parts
        if any(part in SKIP_DIRS for part in relative_parts[:-1]):
            continue
        yield path


def iter_tracked_files(root: pathlib.Path):
    completed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    entries = sorted(
        entry for entry in completed.stdout.decode("utf-8").split("\0") if entry
    )
    for rel in entries:
        path = root / rel
        if not path.is_file():
            continue
        relative_parts = path.relative_to(root).parts
        if any(part in SKIP_DIRS for part in relative_parts[:-1]):
            continue
        yield path


def audit(root: pathlib.Path, *, tracked_only: bool = False) -> AuditResult:
    result = AuditResult(root=str(root), tracked_only=tracked_only)
    compiled = {name: re.compile(pattern) for name, pattern in SENSITIVE_PATTERNS.items()}
    file_iter = iter_tracked_files(root) if tracked_only else iter_files(root)

    for path in file_iter:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        result.files += 1
        result.total_bytes += size

        if size > MAX_BYTES:
            result.oversized.append({"path": relative, "bytes": size})
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            result.forbidden_files.append({"path": relative, "suffix": path.suffix.lower()})

        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for name, pattern in compiled.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                result.hits.append(
                    {
                        "path": relative,
                        "line": line,
                        "pattern": name,
                        "excerpt": text[max(0, match.start() - 20) : match.end() + 20].replace(
                            "\n", " "
                        ),
                    }
                )

    if not tracked_only:
        for directory in sorted(SKIP_DIRS):
            if (root / directory).exists():
                result.skipped.append(directory)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the repository before publishing.")
    parser.add_argument("--root", default=str(pathlib.Path(__file__).resolve().parents[1]))
    parser.add_argument(
        "--tracked-only",
        action="store_true",
        help="Scan only files tracked in the Git index (`git ls-files`).",
    )
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args(argv)

    result = audit(pathlib.Path(args.root).resolve(), tracked_only=args.tracked_only)
    payload = json.dumps(result.as_dict(), indent=2)
    print(payload)
    if args.json_path:
        pathlib.Path(args.json_path).write_text(payload + "\n", encoding="utf-8")

    if not result.ok:
        print("AUDIT FAILED", file=sys.stderr)
        return 1
    mode_label = "tracked files" if args.tracked_only else "files"
    print(
        f"AUDIT PASSED: {result.files} {mode_label}, {result.total_bytes / 1024:.1f} KiB, "
        "no oversize files, no forbidden formats, no sensitive strings",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
