from __future__ import annotations

from io import BytesIO
from pathlib import Path
import logging
from typing import Callable

from PIL import Image

from ..io.hashing import short_hash
from ..io.paths import content_addressed_work_path
from ..io.fs import atomic_write
from .base import Converter


class ImageToPNG(Converter):
    """Convert any image format Pillow understands to PNG."""

    def __init__(self, work_dir: Path, *, redact: bool = False) -> None:
        self.work_dir = work_dir
        self.redact = redact

    def convert(
        self,
        path: Path,
        *,
        get_logger: Callable[[str, str | None], logging.LoggerAdapter],
        request_id: str | None = None,
    ) -> Path:
        logger = get_logger(__name__, request_id=request_id)
        logger.debug("Converting %s to PNG", short_hash(path))
        out = content_addressed_work_path(self.work_dir, path, ".png")
        with Image.open(path) as img:
            exif = None if self.redact else img.info.get("exif")
            buffer = BytesIO()
            img.save(buffer, format="PNG", exif=exif)
        atomic_write(out, buffer.getvalue())
        logger.debug("Wrote PNG to %s", short_hash(out))
        return out


__all__ = ["ImageToPNG"]
