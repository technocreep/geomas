"""Minimal OlmOCR adapter that mirrors the upstream pipeline bootstrap."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

from ..base import BaseOCR, OCRItem, OCRResult
from ...io.hashing import short_hash
from ...io.offline import offline_mode
from ...logging.metrics import observe_latency

try:  # pragma: no cover - optional dependency at runtime
    import olmocr.pipeline as _pipeline
except (ImportError, ModuleNotFoundError):  # pragma: no cover - exercised in tests
    _pipeline = None

try:  # pragma: no cover - optional dependency at runtime
    from olmocr.bench.runners.run_olmocr_pipeline import Args as _BenchmarkArgs
except (ImportError, ModuleNotFoundError):  # pragma: no cover - exercised in tests
    _BenchmarkArgs = None  # type: ignore[assignment]


@dataclass(slots=True)
class _ServerState:
    ready: bool = False
    starting: bool = False
    model: str | None = None


_SERVER_STATE = _ServerState()
_SERVER_LOCK = asyncio.Lock()


class OlmOCR(BaseOCR):
    """Asynchronous wrapper around ``olmocr.pipeline.process_page``."""

    name = "olmocr"
    supports_batch = True
    supports_async = True

    try:  # pragma: no cover - dependency optional in CI
        import olmocr as _olmocr

        version = getattr(_olmocr, "VERSION", getattr(_olmocr, "__version__", "unknown"))
    except ModuleNotFoundError:  # pragma: no cover - exercised in tests
        version = "unknown"

    def __init__(self, *, model: str | None = None, allow_network: bool = False) -> None:
        if _pipeline is None:  # pragma: no cover - exercised in tests
            msg = "olmocr.pipeline is not available"
            raise RuntimeError(msg)

        self.model = model
        self.allow_network = bool(allow_network)
        self._args: Any | None = None

    async def recognize_many(
        self,
        items: list[OCRItem],
        *,
        batch_size: int,
        get_logger: Callable[[str, str | None], logging.LoggerAdapter],
        request_id: str | None = None,
    ) -> list[OCRResult]:
        logger = get_logger(__name__, request_id=request_id)
        args = self._build_args()
        await self._ensure_server(args, logger)

        results: list[OCRResult] = []
        for index, item in enumerate(items, start=1):
            result = await self._process_page(item, index, args, logger)
            results.append(result)
        return results

    def _build_args(self) -> Any:
        if self._args is not None:
            return self._args

        if _BenchmarkArgs is not None:
            args: Any = _BenchmarkArgs()
        else:  # pragma: no cover - exercised in tests
            args = type("Args", (), {})()

        if self.model is not None:
            setattr(args, "model", self.model)
        self._args = args
        return args

    async def _ensure_server(self, args: Any, logger: logging.LoggerAdapter) -> None:
        ready_fn = getattr(_pipeline, "sglang_server_ready", None)
        host_fn = getattr(_pipeline, "sglang_server_host", None)
        if not callable(ready_fn) or not callable(host_fn):  # pragma: no cover - defensive
            return

        async with _SERVER_LOCK:
            if _SERVER_STATE.ready and (_SERVER_STATE.model in {None, getattr(args, "model", None)}):
                return

            if not _SERVER_STATE.starting:
                _SERVER_STATE.starting = True
                logger.debug("Starting OlmOCR server for model %s", getattr(args, "model", None))
                await host_fn(getattr(args, "model", None), args)

            try:
                await ready_fn()
            except Exception as exc:  # pragma: no cover - exercised in tests
                _SERVER_STATE.ready = False
                _SERVER_STATE.starting = False
                logger.error("OlmOCR server failed to start: %s", exc)
                raise RuntimeError("OlmOCR server failed to start") from exc

            _SERVER_STATE.ready = True
            _SERVER_STATE.starting = False
            _SERVER_STATE.model = getattr(args, "model", None)

    async def _process_page(
        self,
        item: OCRItem,
        worker_id: int,
        args: Any,
        logger: logging.LoggerAdapter,
    ) -> OCRResult:
        process_page = getattr(_pipeline, "process_page", None)
        if not callable(process_page):  # pragma: no cover - exercised in tests
            msg = "olmocr.pipeline.process_page is not available"
            raise RuntimeError(msg)

        start = time.perf_counter()
        provenance = {
            "model": self.name,
            "version": self.version,
            "source_hash": short_hash(item.pdf_path),
            "page_index": item.page_index,
        }

        with offline_mode(self.allow_network):
            try:
                page_result = await process_page(
                    args=args,
                    worker_id=worker_id,
                    pdf_orig_path=str(item.pdf_path),
                    pdf_local_path=str(item.pdf_path),
                    page_num=item.page_index + 1,
                )
            except Exception as exc:  # pragma: no cover - exercised in tests
                elapsed = time.perf_counter() - start
                observe_latency("olmocr.page", elapsed)
                logger.exception("OlmOCR failed for %s", item.pdf_path)
                warning = f"OlmOCR pipeline error: {exc}"
                return OCRResult(
                    markdown="# OCR Failed\n\n> OlmOCR pipeline error.",
                    warnings=[warning],
                    time_ms=elapsed * 1000.0,
                    provenance=provenance,
                )

        elapsed = time.perf_counter() - start
        observe_latency("olmocr.page", elapsed)

        response = getattr(page_result, "response", None)
        body_text = getattr(response, "natural_text", "") if response is not None else ""
        markdown = body_text or ""
        warnings: list[str] = []
        if getattr(page_result, "is_fallback", False):
            warnings.append("OlmOCR used fallback output")

        provenance.update(
            {
                "input_tokens": getattr(page_result, "input_tokens", None),
                "output_tokens": getattr(page_result, "output_tokens", None),
            }
        )

        return OCRResult(
            markdown=markdown,
            warnings=warnings,
            time_ms=elapsed * 1000.0,
            provenance=provenance,
        )


__all__ = ["OlmOCR"]

