"""Simplified MinerU adapter that mirrors the official runner usage."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import ExitStack
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

from ..base import BaseOCR, OCRItem, OCRResult
from ...io.hashing import short_hash
from ...io.offline import offline_mode
from ...logging.metrics import observe_latency


@lru_cache(maxsize=1)
def _load_mineru_components() -> tuple[Any, ...]:
    """Return the heavy ``magic_pdf`` objects used by the official pipeline."""

    from magic_pdf.config.enums import SupportedPdfParseMethod  # type: ignore[import-not-found]
    from magic_pdf.config.make_content_config import (  # type: ignore[import-not-found]
        DropMode,
        MakeMode,
    )
    from magic_pdf.data.data_reader_writer import (  # type: ignore[import-not-found]
        FileBasedDataReader,
        FileBasedDataWriter,
    )
    from magic_pdf.data.dataset import PymuDocDataset  # type: ignore[import-not-found]
    from magic_pdf.model.doc_analyze_by_custom_model import (  # type: ignore[import-not-found]
        doc_analyze,
    )

    return (
        PymuDocDataset,
        doc_analyze,
        FileBasedDataReader,
        FileBasedDataWriter,
        DropMode,
        MakeMode,
        SupportedPdfParseMethod,
    )


def _language_hint(item: OCRItem, adapter_language: str | None) -> str | None:
    if adapter_language:
        return adapter_language
    language = item.hints.get("language")
    if language is None:
        return None
    return str(language)


def _use_ocr(dataset: Any, supported_enum: Any) -> bool:
    """Mirror MinerU's automatic classification to decide OCR vs. text mode."""

    classification = dataset.classify()
    value = getattr(classification, "value", classification)
    txt_value = str(getattr(supported_enum.TXT, "value", "txt")).lower()
    return str(value).lower() != txt_value


class MinerU(BaseOCR):
    """Run the MinerU parsing pipeline exactly as the official runner does."""

    name = "mineru"
    try:
        version = metadata.version("magic-pdf")
    except metadata.PackageNotFoundError:  # pragma: no cover - optional dependency
        version = "unknown"
    supports_batch = True
    supports_async = True

    def __init__(self, *, language: str | None = None, allow_network: bool = False) -> None:
        self.language = language
        self.allow_network = bool(allow_network)

    async def recognize_many(
        self,
        items: list[OCRItem],
        *,
        batch_size: int,
        get_logger: Callable[[str, str | None], logging.LoggerAdapter],
        request_id: str | None = None,
    ) -> list[OCRResult]:
        logger = get_logger(__name__, request_id=request_id)
        results: list[OCRResult] = []

        for item in items:
            result = await asyncio.to_thread(self._process_single, item, logger)
            results.append(result)

        return results

    def _process_single(self, item: OCRItem, logger: logging.LoggerAdapter) -> OCRResult:
        start = time.perf_counter()
        provenance = {
            "model": self.name,
            "version": self.version,
            "source_hash": short_hash(item.pdf_path),
            "page_index": item.page_index,
        }

        try:
            markdown, images = self._run_pipeline(item)
        except Exception as exc:  # pragma: no cover - exercised in tests
            elapsed = time.perf_counter() - start
            observe_latency("mineru.page", elapsed)
            logger.exception("MinerU failed for item at %s", item.pdf_path)
            warning = f"MinerU pipeline error: {exc}"
            return OCRResult(
                markdown="# OCR Failed\n\n> MinerU pipeline error.",
                warnings=[warning],
                time_ms=elapsed * 1000.0,
                provenance=provenance,
            )

        elapsed = time.perf_counter() - start
        observe_latency("mineru.page", elapsed)
        return OCRResult(
            markdown=markdown,
            time_ms=elapsed * 1000.0,
            provenance=provenance,
            images=images,
        )

    def _run_pipeline(self, item: OCRItem) -> tuple[str, list[Path]]:
        (
            PymuDocDataset,
            doc_analyze,
            FileBasedDataReader,
            FileBasedDataWriter,
            DropMode,
            MakeMode,
            SupportedPdfParseMethod,
        ) = _load_mineru_components()

        with offline_mode(self.allow_network):
            reader = FileBasedDataReader()
            pdf_bytes = reader.read(str(item.pdf_path))
            language = _language_hint(item, self.language)

            with TemporaryDirectory(prefix="ocr-mineru-") as temp_dir:
                work_dir = Path(temp_dir)
                markdown_path = work_dir / "page.md"
                images_dir = work_dir / "images"
                images_dir.mkdir(parents=True, exist_ok=True)

                dataset = PymuDocDataset(pdf_bytes, lang=language)
                with ExitStack() as stack:
                    stack.callback(getattr(dataset, "close", lambda: None))
                    use_ocr = _use_ocr(dataset, SupportedPdfParseMethod)
                    inference = dataset.apply(
                        doc_analyze,
                        ocr=use_ocr,
                        lang=language,
                    )
                    closer = getattr(inference, "close", None)
                    if callable(closer):
                        stack.callback(closer)

                    pipe = (
                        getattr(inference, "pipe_ocr_mode")
                        if use_ocr
                        else getattr(inference, "pipe_txt_mode")
                    )
                    pipe_result = pipe(  # type: ignore[call-arg]
                        FileBasedDataWriter(str(images_dir)),
                        start_page_id=item.page_index,
                        end_page_id=item.page_index + 1,
                        debug_mode=False,
                        lang=language,
                    )

                    pipe_result.dump_md(  # type: ignore[attr-defined]
                        FileBasedDataWriter(str(work_dir)),
                        markdown_path.name,
                        "images",
                        drop_mode=DropMode.NONE,
                        md_make_mode=MakeMode.MM_MD,
                    )

                markdown = markdown_path.read_text(encoding="utf-8")
                images = sorted(child for child in images_dir.iterdir() if child.is_file())

        return markdown, images


__all__ = ["MinerU"]

