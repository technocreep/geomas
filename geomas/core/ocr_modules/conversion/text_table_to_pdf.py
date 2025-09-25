from __future__ import annotations

from io import BytesIO
from pathlib import Path
import logging
from typing import Callable

from fpdf import FPDF

from ..io.hashing import short_hash
from ..io.paths import content_addressed_work_path
from ..io.fs import atomic_write
from .base import Converter

FONT_PATHS = (
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
)


def _read_text_with_fallback(path: Path) -> str:
    encodings = ["utf-8"]
    try:
        from charset_normalizer import from_path

        result = from_path(str(path)).best()
        if result and result.encoding:
            encodings.append(result.encoding)
    except Exception:
        pass
    encodings.extend(["iso-8859-1", "cp1252"])
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="ignore")


class TextTableToPDF(Converter):
    """Naive text-to-PDF converter used for testing."""

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
        logger.debug("Converting %s to PDF", short_hash(path))
        out = content_addressed_work_path(self.work_dir, path, ".pdf")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        font_path = next((p for p in FONT_PATHS if p.exists()), None)
        if font_path is None:
            logger.warning("No Unicode font found; falling back to core Helvetica")
            pdf.set_font("Helvetica", size=12)
        else:
            font_name = font_path.stem
            pdf.add_font(font_name, "", str(font_path), uni=True)
            pdf.set_font(font_name, size=12)
        text = _read_text_with_fallback(path)
        tuple(pdf.multi_cell(pdf.epw, 10, line) for line in text.splitlines())
        buffer = BytesIO()
        pdf.output(buffer)
        atomic_write(out, buffer.getvalue())
        logger.debug("Wrote PDF to %s", short_hash(out))
        return out


__all__ = ["TextTableToPDF"]
