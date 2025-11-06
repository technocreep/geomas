"""Binarize PDF pages using Otsu or Sauvola thresholding.

Pages are rasterised via PyMuPDF, converted to grayscale and thresholded using
either Otsu's global method or Sauvola's local adaptive method.  The result is
saved to a new PDF with a ``_bin`` suffix using an atomic write.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Callable, List
import logging
import time

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from ...io.fs import atomic_write
from ...io.hashing import short_hash
from ...logging.metrics import observe_latency


def _otsu_threshold(arr: np.ndarray) -> int:
    """Compute Otsu's threshold for an array of grayscale pixels."""

    hist, _ = np.histogram(arr.flatten(), bins=256, range=(0, 256))
    total = arr.size
    sum_total = np.dot(np.arange(256), hist)
    sumB = 0.0
    wB = 0
    max_var = 0.0
    threshold = 0
    for i in range(256):
        wB += hist[i]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += i * hist[i]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF
        var_between = wB * wF * (mB - mF) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = i
    return int(threshold)


def _sauvola_threshold(
    arr: np.ndarray, *, window_size: int = 25, k: float = 0.2, r: float = 128
) -> np.ndarray:
    """Return Sauvola's local threshold for ``arr``.

    Parameters are chosen to match typical defaults used by olmOCR-bench.  The
    implementation follows the integral image approach for efficiency and
    returns an array of per-pixel thresholds.
    """

    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    pad = window_size // 2
    padded = np.pad(arr.astype(np.float64), pad, mode="reflect")
    integ = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    integ = np.pad(integ, ((1, 0), (1, 0)), mode="constant")
    sq = np.cumsum(np.cumsum(padded**2, axis=0), axis=1)
    sq = np.pad(sq, ((1, 0), (1, 0)), mode="constant")
    area = float(window_size**2)
    i = window_size
    sums = integ[i:, i:] - integ[:-i, i:] - integ[i:, :-i] + integ[:-i, :-i]
    sq_sums = sq[i:, i:] - sq[:-i, i:] - sq[i:, :-i] + sq[:-i, :-i]
    mean = sums / area
    variance = sq_sums / area - mean**2
    std = np.sqrt(np.clip(variance, 0, None))
    return mean * (1 + k * ((std / r) - 1))


_THRESHOLDERS: dict[str, Callable[[np.ndarray], np.ndarray | int]] = {
    "otsu": _otsu_threshold,
    "sauvola": _sauvola_threshold,
}


def binarize(
    path: Path,
    *,
    method: str = "otsu",
    get_logger: Callable[[str, str | None], logging.LoggerAdapter],
    request_id: str | None = None,
) -> Path:
    """Return a binarized copy of *path*."""

    logger = get_logger(__name__, request_id=request_id)
    rid = logger.extra.get("request_id")
    logger.info("binarize start: %s", short_hash(path), extra={"request_id": rid})

    images: List[Image.Image] = []
    arr: np.ndarray | None = None
    bw_arr: np.ndarray | None = None

    with fitz.open(path) as doc:
        for page in doc:
            start = time.perf_counter()
            pix = page.get_pixmap()
            h, w = pix.height, pix.width
            if arr is None or arr.shape != (h, w):
                arr = np.empty((h, w), dtype=np.uint8)
                bw_arr = np.empty((h, w), dtype=np.uint8)
            samples = np.frombuffer(pix.samples, dtype=np.uint8).reshape(h, w, pix.n)
            if pix.n == 1:
                arr[:, :] = samples[:, :, 0]
            else:
                arr[:, :] = (
                    samples[:, :, 0].astype(np.uint16) * 299
                    + samples[:, :, 1].astype(np.uint16) * 587
                    + samples[:, :, 2].astype(np.uint16) * 114
                ) // 1000
            thresh_fn = _THRESHOLDERS.get(method)
            if thresh_fn is None:
                raise ValueError(f"unknown binarize method: {method}")
            thr = thresh_fn(arr)
            bw_arr.fill(0)
            bw_arr[arr > thr] = 255

            images.append(Image.fromarray(bw_arr.copy(), mode="L"))
            observe_latency("binarize.page", time.perf_counter() - start)
            pix = None

    out_path = path.with_name(f"{path.stem}_bin.pdf")
    buf = BytesIO()
    images[0].save(buf, format="PDF", save_all=True, append_images=images[1:])
    atomic_write(out_path, buf.getvalue())
    for img in images:
        img.close()
    logger.info(
        "binarize finish: %s -> %s",
        short_hash(path),
        short_hash(out_path),
        extra={"request_id": rid},
    )
    return out_path


__all__ = ["binarize"]
