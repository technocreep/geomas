"""Rasterise PDF pages to avoid missing-font issues."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import List, Callable
import logging
import time

import fitz  # PyMuPDF
from PIL import Image

from ...io.fs import atomic_write
from ...io.hashing import short_hash
from ...logging.metrics import observe_latency


def fix_fonts(
    path: Path,
    *,
    get_logger: Callable[[str, str | None], logging.LoggerAdapter],
    request_id: str | None = None,
) -> Path:
    """Return a copy of *path* with all pages rasterised.

    Rasterising removes any dependency on embedded fonts and ensures consistent
    rendering across environments.
    """

    logger = get_logger(__name__, request_id=request_id)
    rid = logger.extra.get("request_id")
    logger.info("fix_fonts start: %s", short_hash(path), extra={"request_id": rid})
    doc = fitz.open(path)
    images: List[Image.Image] = []
    for page in doc:
        start = time.perf_counter()
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        images.append(img)
        observe_latency("fix_fonts.page", time.perf_counter() - start)

    out_path = path.with_name(f"{path.stem}_fontfix.pdf")
    buf = BytesIO()
    images[0].save(buf, format="PDF", save_all=True, append_images=images[1:])
    atomic_write(out_path, buf.getvalue())
    logger.info(
        "fix_fonts finish: %s -> %s",
        short_hash(path),
        short_hash(out_path),
        extra={"request_id": rid},
    )
    return out_path


__all__ = ["fix_fonts"]
