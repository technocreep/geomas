"""Adapter for the Marker pipeline."""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterator

from ..base import BaseOCR, OCRItem, OCRResult
from ...io import offline_mode
from ...io.hashing import short_hash
from ...logging.metrics import observe_latency

try:  # pragma: no cover - compatibility shim
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover - Python <3.8 fallback
    import importlib_metadata  # type: ignore[no-redef]

try:
    from marker.config.parser import ConfigParser as _MarkerConfigParser
    from marker.converters.pdf import PdfConverter as _MarkerPdfConverter
    from marker.models import create_model_dict as _marker_create_model_dict
    from marker.output import text_from_rendered as _marker_text_from_rendered
except ModuleNotFoundError as exc:
    _IMPORT_ERROR: ModuleNotFoundError | None = exc
    _MarkerConfigParser = None  # type: ignore[assignment]
    _MarkerPdfConverter = None  # type: ignore[assignment]
    _marker_create_model_dict = None  # type: ignore[assignment]
    _marker_text_from_rendered = None  # type: ignore[assignment]
else:
    _IMPORT_ERROR = None


DEFAULT_LANGUAGE = "auto"
DEFAULT_FORCE_OCR = True
DEFAULT_DISABLE_TQDM = True
DEFAULT_ALLOW_NETWORK = False


RUN_MARKER_BATCH_DEFAULTS: dict[str, int] = {
    "layout_batch_size": 12,
    "detection_batch_size": 8,
    "table_rec_batch_size": 12,
    "ocr_error_batch_size": 12,
    "recognition_batch_size": 64,
    "equation_batch_size": 16,
    "detector_postprocessing_cpu_workers": 2,
}


def _detect_marker_version() -> str:
    """Best-effort discovery of the installed Marker version."""

    for dist_name in ("marker-pdf", "marker"):
        try:
            return importlib_metadata.version(dist_name)
        except Exception:  # pragma: no cover - best effort only
            continue
    for module_name in ("marker", "marker_pdf"):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:  # pragma: no cover - optional dependency
            continue
        version = getattr(module, "__version__", None)
        if version:
            return str(version)
    return "unknown"


def _coerce_warnings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [str(value)]
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value]
    return [str(value)]


def _metadata_from_rendered(rendered: Any) -> dict[str, Any]:
    metadata: Any = getattr(rendered, "metadata", None)
    if isinstance(metadata, dict):
        return dict(metadata)
    if isinstance(rendered, dict):
        meta = rendered.get("metadata")
        if isinstance(meta, dict):
            return dict(meta)
    return {}


class Marker(BaseOCR):
    """Run OCR using the upstream Marker pipeline."""

    name = "marker"
    version = _detect_marker_version()
    supports_batch = True
    supports_async = True

    def __init__(
        self,
        *,
        language: str = DEFAULT_LANGUAGE,
        force_ocr: bool = DEFAULT_FORCE_OCR,
        disable_tqdm: bool = DEFAULT_DISABLE_TQDM,
        allow_network: bool = DEFAULT_ALLOW_NETWORK,
    ) -> None:
        self.language = language
        self.force_ocr = bool(force_ocr)
        self.disable_tqdm = bool(disable_tqdm)
        self.allow_network = bool(allow_network)

    async def recognize_many(
        self,
        items: list[OCRItem],
        *,
        batch_size: int,
        get_logger: Callable[..., logging.LoggerAdapter[logging.Logger]],
        request_id: str | None = None,
    ) -> list[OCRResult]:
        """Run Marker OCR on *items* sequentially."""

        if batch_size < 1:
            msg = "batch_size must be >= 1"
            raise ValueError(msg)

        logger = get_logger(__name__, request_id=request_id)
        rid = getattr(logger, "extra", {}).get("request_id", request_id)
        extra = {"request_id": rid}

        if not items:
            logger.info("Marker received no items", extra=extra)
            return []

        hashes = [short_hash(item.pdf_path) for item in items]
        logger.info("Marker start: %s", hashes, extra=extra)

        results: list[OCRResult] = []
        for item in items:
            result = await self._process_item(item, logger, extra)
            results.append(result)

        logger.info("Marker finish: %s", hashes, extra=extra)
        return results

    async def _process_item(
        self,
        item: OCRItem,
        logger: logging.LoggerAdapter[logging.Logger],
        extra: dict[str, Any],
    ) -> OCRResult:
        hashed = short_hash(item.pdf_path)
        start = time.perf_counter()
        page_hash: str | None = None

        self._ensure_dependencies()

        try:
            with self._single_page_pdf(item.pdf_path, item.page_index) as (
                single_page_pdf,
                page_hash,
            ):
                converter = self._build_converter()
                with offline_mode(self.allow_network):
                    rendered = converter(str(single_page_pdf))
                text, _, _ = _marker_text_from_rendered(rendered)  # type: ignore[misc]
                metadata = _metadata_from_rendered(rendered)
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise self._dependency_error(exc) from exc
        except Exception as exc:  # pragma: no cover - defensive fallback
            if isinstance(exc, RuntimeError):
                raise
            elapsed = time.perf_counter() - start
            logger.exception(
                "Marker conversion failed for %s", hashed, extra=extra
            )
            return self._failure_result(
                hashed=hashed,
                item=item,
                page_hash=page_hash,
                error=exc,
                elapsed=elapsed,
            )

        sanitized_obj = metadata.pop("sanitized_pdf", None)
        sanitized = f"hash:{hashed}" if sanitized_obj is None else str(sanitized_obj)
        warnings = _coerce_warnings(metadata.pop("warnings", None))

        provenance: dict[str, Any] = {
            "model": self.name,
            "model_version": self.version,
            "page_index": item.page_index,
            "sanitized_pdf": sanitized,
        }
        if page_hash is not None:
            provenance["page_hash"] = page_hash

        elapsed = time.perf_counter() - start
        observe_latency("marker.page", elapsed)
        logger.info("Marker converted %s in %.3f s", hashed, elapsed, extra=extra)

        return OCRResult(
            markdown=text,
            warnings=warnings,
            time_ms=elapsed * 1000.0,
            provenance=provenance,
        )

    def _build_converter(self) -> Any:
        cli_options = self._cli_options()
        parser = _MarkerConfigParser(cli_options)  # type: ignore[operator]
        config = parser.generate_config_dict()
        config["force_ocr"] = True
        config["disable_tqdm"] = bool(self.disable_tqdm)
        config["use_llm"] = False
        for key, value in RUN_MARKER_BATCH_DEFAULTS.items():
            config.setdefault(key, value)

        artifact_dict = _marker_create_model_dict()  # type: ignore[operator]
        return _MarkerPdfConverter(  # type: ignore[operator]
            config=config,
            artifact_dict=artifact_dict,
            processor_list=parser.get_processors(),
            renderer=parser.get_renderer(),
            llm_service=parser.get_llm_service(),
        )

    def _cli_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {
            "output_format": "markdown",
            "force_ocr": True,
            "disable_tqdm": bool(self.disable_tqdm),
            "use_llm": False,
        }
        if self.language != DEFAULT_LANGUAGE:
            options["language"] = self.language
        return options

    @contextlib.contextmanager
    def _single_page_pdf(
        self, pdf_path: Path, page_index: int
    ) -> Iterator[tuple[Path, str | None]]:
        try:
            from pypdf import PdfReader, PdfWriter
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise self._dependency_error(exc) from exc

        reader = PdfReader(str(pdf_path))
        page_count = len(reader.pages)
        if page_count <= 0:
            msg = "PDF file contains no pages"
            raise ValueError(msg)

        if page_count == 1:
            page_index = 0
        elif page_index < 0 or page_index >= page_count:
            msg = f"page_index {page_index} out of range"
            raise ValueError(msg)

        if page_count == 1:
            data = pdf_path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            yield pdf_path, digest
            return

        writer = PdfWriter()
        writer.add_page(reader.pages[page_index])
        buffer = io.BytesIO()
        writer.write(buffer)
        data = buffer.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        with self._temporary_pdf(data) as single_page:
            yield single_page, digest

    def _failure_result(
        self,
        *,
        hashed: str,
        item: OCRItem,
        page_hash: str | None,
        error: Exception,
        elapsed: float,
    ) -> OCRResult:
        message = f"Marker OCR failed for {hashed}: {error}".strip()
        markdown = f"# OCR Failed\n\n> {message}"

        provenance: dict[str, Any] = {
            "model": self.name,
            "model_version": self.version,
            "page_index": item.page_index,
            "sanitized_pdf": f"hash:{hashed}",
        }
        if page_hash is not None:
            provenance["page_hash"] = page_hash

        return OCRResult(
            markdown=markdown,
            warnings=[message],
            time_ms=elapsed * 1000.0,
            provenance=provenance,
        )

    @contextlib.contextmanager
    def _temporary_pdf(self, data: bytes) -> Iterator[Path]:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
            handle.write(data)
            temp_path = Path(handle.name)
        try:
            yield temp_path
        finally:
            with contextlib.suppress(FileNotFoundError):
                temp_path.unlink()

    def _ensure_dependencies(self) -> None:
        if (
            _MarkerConfigParser is None
            or _MarkerPdfConverter is None
            or _marker_text_from_rendered is None
            or _marker_create_model_dict is None
        ):
            raise self._dependency_error(_IMPORT_ERROR)

    @staticmethod
    def _dependency_error(exc: ModuleNotFoundError | None) -> RuntimeError:
        missing = getattr(exc, "name", None) if exc is not None else None
        detail = f" (missing module: {missing})" if missing else ""
        message = (
            "Marker dependencies are unavailable"
            f"{detail}. Install the optional 'marker-pdf' extras to enable this adapter."
        )
        return RuntimeError(message)


__all__ = [
    "Marker",
    "DEFAULT_LANGUAGE",
    "DEFAULT_FORCE_OCR",
    "DEFAULT_DISABLE_TQDM",
    "DEFAULT_ALLOW_NETWORK",
]
