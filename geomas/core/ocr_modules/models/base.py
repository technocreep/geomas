"""OCR base interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import logging
from typing import Any, Callable, Dict, List, Mapping


@dataclass(slots=True, init=False)
class OCRItem:
    """Unit of work passed to an OCR adapter."""

    pdf_path: Path
    page_index: int
    image_path: Path | None
    hints: dict[str, Any]

    def __init__(
        self,
        pdf_path: Path | None = None,
        *,
        page_index: int = 0,
        image_path: Path | None = None,
        hints: Mapping[str, Any] | None = None,
        path: Path | None = None,
    ) -> None:
        actual_path = pdf_path or path
        if actual_path is None:
            msg = "pdf_path must be provided"
            raise ValueError(msg)
        actual_path = Path(actual_path)
        self.pdf_path = actual_path
        self.page_index = page_index
        self.image_path = Path(image_path) if image_path is not None else None
        self.hints = dict(hints) if hints is not None else {}

    @property
    def path(self) -> Path:
        """Backwards compatible alias for ``pdf_path``."""

        return self.pdf_path


@dataclass(slots=True)
class OCRResult:
    """Normalized OCR output ready for Markdown writing."""

    markdown: str
    warnings: List[str] = field(default_factory=list)
    time_ms: float | None = None
    tokens_used: int | None = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    images: List[Path] = field(default_factory=list)
    cleanup_paths: List[Path] = field(default_factory=list)


class BaseOCR(ABC):
    """Common OCR adapter interface."""

    #: Human-friendly adapter name, e.g. ``"marker"``.
    name: str

    #: Upstream library or model version. ``"unknown"`` when not available.
    version: str

    #: Whether the adapter can process more than one item per call.
    supports_batch: bool

    #: Whether the adapter performs substantial work asynchronously.
    supports_async: bool

    @abstractmethod
    async def recognize_many(
        self,
        items: List[OCRItem],
        *,
        batch_size: int,
        get_logger: Callable[..., logging.LoggerAdapter],
        request_id: str | None = None,
    ) -> List[OCRResult]:
        """Run OCR on ``items`` in batches of ``batch_size``.

        Each :class:`OCRItem` exposes the normalized PDF page to process via
        :attr:`OCRItem.pdf_path`, a zero-based :attr:`OCRItem.page_index`, an
        optional :attr:`OCRItem.image_path` for page-rendered imagery, and free
        form :attr:`OCRItem.hints` provided by the orchestrator (language,
        layout, etc.).  Implementations should preserve input order and return a
        matching list of :class:`OCRResult` objects whose ``markdown`` content is
        ready for downstream normalization.  ``OCRResult`` instances may include
        adapter specific ``warnings``, execution ``time_ms`` and ``tokens_used``
        metrics, plus provenance metadata for Markdown generation.
        """
        raise NotImplementedError


__all__ = ["BaseOCR", "OCRItem", "OCRResult"]
