"""Base classes for document converters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import logging
from typing import List, Callable


class Converter(ABC):
    """Convert input files to a normalized representation."""

    @abstractmethod
    def convert(
        self,
        path: Path,
        *,
        get_logger: Callable[[str, str | None], logging.LoggerAdapter],
        request_id: str | None = None,
    ) -> Path | tuple[Path, List[Path]]:
        """Convert *path* and return path(s) to converted file(s).

        Parameters
        ----------
        path:
            Input file path.
        get_logger:
            Callable returning configured loggers.
        request_id:
            Identifier for correlating logs across the pipeline. A UUID4 is
            generated when omitted.
        """
        raise NotImplementedError


__all__ = ["Converter"]
