from __future__ import annotations

from pathlib import Path

from .fs import ensure_dir
from .hashing import file_sha256


def content_addressed_output(output_dir: Path, src: Path, ext: str) -> Path:
    """Return output path under ``output_dir`` based on SHA256 of ``src``."""
    digest = file_sha256(src)
    out = output_dir / f"{digest}{ext}"
    ensure_dir(out.parent)
    return out


def content_addressed_work_path(work_dir: Path, src: Path, ext: str) -> Path:
    """Return work path under ``work_dir`` based on SHA256 of ``src``."""
    digest = file_sha256(src)
    out = work_dir / f"{digest}{ext}"
    ensure_dir(out.parent)
    return out


def mirror_output_path(input_dir: Path, output_dir: Path, src: Path, ext: str) -> Path:
    """Return path under ``output_dir`` mirroring ``input_dir`` structure."""
    rel = src.relative_to(input_dir)
    out = (output_dir / rel).with_suffix(ext)
    ensure_dir(out.parent)
    return out


__all__ = [
    "content_addressed_output",
    "content_addressed_work_path",
    "mirror_output_path",
]
