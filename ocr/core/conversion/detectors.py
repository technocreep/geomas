import logging
from pathlib import Path
from typing import Callable

from ..io.fs import atomic_write
from ..io.hashing import short_hash
from ..io.paths import content_addressed_work_path
from .base import Converter
from .image_to_png import ImageToPNG
from .pptx_splitter import PPTXSplitter
from .text_table_to_pdf import TextTableToPDF

TEXT_EXTS = {
    ".txt",
    ".csv",
    ".tsv",
    ".html",
    ".htm",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
}

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}

ConverterFactory = Callable[[Path], Converter]


class _IdentityConverter(Converter):
    """Copy PDF inputs to the work directory without modification."""

    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir

    def convert(
        self,
        path: Path,
        *,
        get_logger: Callable[[str, str | None], logging.LoggerAdapter],
        request_id: str | None = None,
    ) -> Path:
        logger = get_logger(__name__, request_id=request_id)
        logger.debug("Copying %s", short_hash(path))
        out = content_addressed_work_path(self.work_dir, path, path.suffix)
        atomic_write(out, path.read_bytes())
        logger.debug("Copied to %s", short_hash(out))
        return out


CONVERTERS: dict[str, ConverterFactory] = {
    ".pdf": _IdentityConverter,
    ".pptx": PPTXSplitter,
    **{ext: TextTableToPDF for ext in TEXT_EXTS},
    **{ext: ImageToPNG for ext in IMG_EXTS},
}


def detect_converter(path: Path, work_dir: Path) -> Converter:
    ext = path.suffix.lower()
    factory = CONVERTERS.get(ext)
    if factory is None:
        raise ValueError(f"Unsupported input type: {ext}")
    return factory(work_dir)


__all__ = ["detect_converter"]
