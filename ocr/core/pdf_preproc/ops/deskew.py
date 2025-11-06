"""Deskew PDF pages using a simple variance-based approach.

This implementation renders each page to a bitmap via PyMuPDF, estimates the
skew angle by maximising the vertical projection variance and writes a new PDF
with corrected orientation.  The output path uses a ``_deskew`` suffix and the
write is performed atomically.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Callable
import logging
import time

import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageFilter

from ...io.fs import atomic_write
from ...io.hashing import short_hash
from ...logging.metrics import observe_latency


def _estimate_skew(
    img: Image.Image,
    max_angle_deg: int,
    logger: logging.LoggerAdapter | None = None,
) -> float:
    """Estimate skew angle of *img* using variance of projections.

    If a :class:`MemoryError` occurs, the page is skipped and ``0.0`` is
    returned.
    """

    if logger is None:
        logger = logging.getLogger(__name__)
    try:
        gray = img.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        gray.close()
        edge_arr = np.asarray(edges, dtype=float)
        edges.close()

        angles = np.arange(-max_angle_deg, max_angle_deg + 1)
        edge_img = Image.fromarray(edge_arr)
        rotated_arrays: list[np.ndarray] = []
        for ang in angles:
            rot = edge_img.rotate(int(ang), resample=Image.BICUBIC)
            rotated_arrays.append(np.asarray(rot, dtype=float))
            rot.close()
        edge_img.close()

        stack = np.stack(rotated_arrays, axis=0)
        variances = stack.sum(axis=2).var(axis=1)
        best_angle = angles[int(np.argmax(variances))]
        return -float(best_angle)
    except MemoryError:
        logger.warning("memory error during skew estimation; skipping page")
        return 0.0


def deskew(
    path: Path,
    *,
    max_angle_deg: int = 5,
    get_logger: Callable[[str, str | None], logging.LoggerAdapter],
    request_id: str | None = None,
) -> Path:
    """Return a deskewed copy of *path*.

    Parameters
    ----------
    path:
        Source PDF.
    max_angle_deg:
        Maximum angle in degrees to search when estimating skew.
    """

    logger = get_logger(__name__, request_id=request_id)
    rid = logger.extra.get("request_id")
    logger.info("deskew start: %s", short_hash(path), extra={"request_id": rid})
    images: list[Image.Image] = []
    with fitz.open(path) as doc:
        for page in doc:
            start = time.perf_counter()
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            del pix
            # Downscale to limit memory usage during skew estimation
            img_skew = img
            max_dim = max(img.width, img.height)
            if max_dim > 2048:
                scale = 2048 / max_dim
                new_size = (int(img.width * scale), int(img.height * scale))
                img_skew = img.resize(new_size, Image.LANCZOS)
            angle = _estimate_skew(img_skew, max_angle_deg, logger)
            if img_skew is not img:
                img_skew.close()
            if abs(angle) > 0.01:
                rotated = img.rotate(angle, expand=True, fillcolor="white")
                img.close()
                img = rotated
            images.append(img)
            observe_latency("deskew.page", time.perf_counter() - start)

    out_path = path.with_name(f"{path.stem}_deskew.pdf")
    buf = BytesIO()
    try:
        images[0].save(buf, format="PDF", save_all=True, append_images=images[1:])
        atomic_write(out_path, buf.getvalue())
    finally:
        for img in images:
            img.close()
        buf.close()
    logger.info(
        "deskew finish: %s -> %s",
        short_hash(path),
        short_hash(out_path),
        extra={"request_id": rid},
    )
    return out_path


__all__ = ["deskew", "_estimate_skew"]
