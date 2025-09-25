"""Content-addressed hashing helpers.

Includes helpers for computing full and short SHA256 digests of files.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: Path) -> str:
    """Return the SHA256 hex digest of *path*."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def short_hash(path: Path, length: int = 8) -> str:
    """Return a shortened SHA256 hex digest for *path*.

    The file's SHA256 digest is computed and truncated to ``length`` characters,
    which is suitable for logging identifiers without leaking filenames.
    """
    return file_sha256(path)[:length]


__all__ = ["file_sha256", "short_hash"]
