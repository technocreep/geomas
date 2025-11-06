"""Top level package for OCR.

Exposes the high level :func:`process_path` API lazily to avoid importing
heavy dependencies during module initialization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from .api import process_path as process_path


def __getattr__(name: str):
    if name == "process_path":  # pragma: no cover - simple lazy import
        from .api import process_path

        return process_path
    raise AttributeError(name)


__all__ = ["process_path"]
