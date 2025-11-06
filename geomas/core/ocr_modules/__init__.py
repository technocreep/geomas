"""OCR module integrated into geomas core architecture.

Provides document OCR processing functionality with multiple adapter support.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from .api import process_path as process_path
    from .api import process_paths as process_paths


def __getattr__(name: str):
    if name == "process_path":  # pragma: no cover - simple lazy import
        from .api import process_path
        return process_path
    if name == "process_paths":  # pragma: no cover - simple lazy import  
        from .api import process_paths
        return process_paths
    raise AttributeError(name)


__all__ = ["process_path", "process_paths"]
