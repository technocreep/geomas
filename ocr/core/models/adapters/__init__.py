"""Built-in OCR adapter implementations."""

from __future__ import annotations

from .marker import Marker
from .mineru import MinerU
from .olmocr import OlmOCR
from .qwen_vl import QwenVL

__all__ = [
    "Marker",
    "MinerU",
    "OlmOCR",
    "QwenVL",
]
