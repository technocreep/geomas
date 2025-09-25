"""Utilities for offline model downloads."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def offline_mode(allow_network: bool) -> Iterator[None]:
    """Temporarily enable Hugging Face offline mode."""
    prev = os.environ.get("HF_HUB_OFFLINE")
    if allow_network:
        yield
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev


__all__ = ["offline_mode"]
