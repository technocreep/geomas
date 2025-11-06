"""Crop uniform margins from PDF pages."""

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


def crop(
    path: Path,
    *,
    margin: int = 0,
    get_logger: Callable[[str, str | None], logging.LoggerAdapter],
    request_id: str | None = None,
) -> Path:
    """Return a copy of *path* cropped by ``margin`` pixels on all sides."""

    logger = get_logger(__name__, request_id=request_id)
    rid = logger.extra.get("request_id")
    logger.info("crop start: %s", short_hash(path), extra={"request_id": rid})
    doc = fitz.open(path)
    images: List[Image.Image] = []
    for page in doc:
        start = time.perf_counter()
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        if margin > 0:
            w, h = img.size
            left = min(margin, w // 2)
            upper = min(margin, h // 2)
            right = w - left
            lower = h - upper
            img = img.crop((left, upper, right, lower))
        images.append(img)
        observe_latency("crop.page", time.perf_counter() - start)

    out_path = path.with_name(f"{path.stem}_crop.pdf")
    buf = BytesIO()
    images[0].save(buf, format="PDF", save_all=True, append_images=images[1:])
    atomic_write(out_path, buf.getvalue())
    logger.info(
        "crop finish: %s -> %s",
        short_hash(path),
        short_hash(out_path),
        extra={"request_id": rid},
    )
    return out_path


__all__ = ["crop"]
