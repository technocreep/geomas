"""Scale PDF pages to a target DPI using PyMuPDF and Pillow."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Callable
import logging
import time

import fitz  # PyMuPDF
from PIL import Image

from ...io.fs import atomic_write
from ...io.hashing import short_hash
from ...logging.metrics import observe_latency


def scale(
    path: Path,
    *,
    target_dpi: int = 300,
    get_logger: Callable[[str, str | None], logging.LoggerAdapter],
    request_id: str | None = None,
) -> Path:
    """Return a copy of *path* scaled to ``target_dpi``."""

    logger = get_logger(__name__, request_id=request_id)
    rid = logger.extra.get("request_id")
    logger.info("scale start: %s", short_hash(path), extra={"request_id": rid})
    try:
        images: list[Image.Image] = []
        with fitz.open(path) as doc:
            for page in doc:
                start = time.perf_counter()
                pix = page.get_pixmap(dpi=target_dpi)
                if pix.width * pix.height > 10_000_000:
                    pix.shrink(1)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                images.append(img)
                pix = None
                observe_latency("scale.page", time.perf_counter() - start)

        out_path = path.with_name(f"{path.stem}_sc{target_dpi}.pdf")
        buf = BytesIO()
        images[0].save(
            buf,
            format="PDF",
            save_all=True,
            append_images=images[1:],
            resolution=target_dpi,
        )
        atomic_write(out_path, buf.getvalue())
    except MemoryError as exc:
        logger.exception(
            "scale memory error: %s", short_hash(path), extra={"request_id": rid}
        )
        raise RuntimeError("scale failed: insufficient memory") from exc

    logger.info(
        "scale finish: %s -> %s",
        short_hash(path),
        short_hash(out_path),
        extra={"request_id": rid},
    )
    return out_path


__all__ = ["scale"]
