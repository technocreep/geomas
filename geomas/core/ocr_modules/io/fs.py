"""Filesystem helpers with atomic operations."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def ensure_dir(path: Path) -> None:
    """Create *path* directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def atomic_write(path: Path, data: bytes) -> None:
    """Write *data* to *path* atomically."""
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile(delete=False, dir=path.parent) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp.name, path)


__all__ = ["ensure_dir", "atomic_write"]
