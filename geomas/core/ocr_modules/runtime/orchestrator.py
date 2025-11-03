"""Asynchronous pipeline orchestrator with staged queues.

Paths are logged as short hashes to avoid leaking user filenames.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import mimetypes
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from itertools import cycle
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Callable,
    Coroutine,
    AsyncIterator,
    Iterable,
    Mapping,
)
from ..conversion.detectors import detect_converter
from ..io.hashing import short_hash
from ..io.paths import mirror_output_path
from ..io.fs import atomic_write, ensure_dir
from ..io.offline import offline_mode
from ..models import BaseOCR, OCRItem, OCRResult
from ..pdf_preproc import run_pipeline
from ..standardize import write_markdown
from .queue import WorkQueue
from .resources import Resources
from ..logging.logger import LoggerFactory
from ..logging.metrics import increment, observe_latency


DEFAULT_BATCH_SIZE = 8
DEFAULT_LANGUAGE = "auto"
DEFAULT_ALLOW_NETWORK = False
DEFAULT_PREPROCESSING_OPS: tuple[dict[str, dict[str, Any]], ...] = (
    {"split_pages": {}},
    {"deskew": {"max_angle_deg": 5}},
    {"binarize": {"method": "sauvola"}},
    {"scale": {"target_dpi": 300}},
)
DEFAULT_ADAPTER_PREPROC_OVERRIDES: dict[str, dict[str, bool]] = {
    "marker": {"binarize": False}
}
DEFAULT_QUEUE_SIZES: dict[str, int] = {
    "conversion": 32,
    "preproc": 32,
    "models": 32,
}
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF_MS = (250, 2000)
DEFAULT_HINT_FLAGS: dict[str, bool] = {
    "include_language": True,
    "include_images": True,
}
DEFAULT_METADATA_FLAGS: dict[str, bool] = {
    "capture_timings": True,
    "capture_tokens": True,
}


def _copy_preproc_ops(
    ops: Iterable[Mapping[str, Mapping[str, Any]]]
) -> list[dict[str, dict[str, Any]]]:
    """Return a deep-ish copy of *ops* suitable for mutation at runtime."""

    copied: list[dict[str, dict[str, Any]]] = []
    for op in ops:
        copied.append({name: dict(params) for name, params in op.items()})
    return copied


def _copy_adapter_overrides(
    overrides: Mapping[str, Mapping[str, bool]]
) -> dict[str, dict[str, bool]]:
    """Return a shallow copy of adapter-specific preprocessing overrides."""

    return {name: dict(flags) for name, flags in overrides.items()}


@dataclass
class _PreprocItem:
    src: Path
    normalized: Path
    images: List[Path]


@dataclass
class _OCRItem:
    src: Path
    preproc: Path
    images: List[Path]
    page: int


@dataclass
class _ConvertStage:
    work_dir: Path
    prep_q: WorkQueue[_PreprocItem]
    get_logger: LoggerFactory
    request_id: str

    async def __call__(self, path: Path) -> None:
        logger = self.get_logger(__name__, request_id=self.request_id)
        logger.info("Converting %s", short_hash(path))
        converter = detect_converter(path, self.work_dir)
        normalized = await asyncio.to_thread(
            converter.convert,
            path,
            get_logger=self.get_logger,
            request_id=self.request_id,
        )
        norm_path, images = (
            normalized if isinstance(normalized, tuple) else (normalized, [])
        )
        await self.prep_q.put(_PreprocItem(path, norm_path, images))


@dataclass
class _PreprocessStage:
    preproc_ops: List[dict[str, dict[str, Any]]]
    adapter_name: str
    adapter_overrides: dict[str, dict[str, bool]]
    ocr_q: WorkQueue[_OCRItem]
    get_logger: LoggerFactory
    request_id: str

    async def __call__(self, item: _PreprocItem) -> None:
        logger = self.get_logger(__name__, request_id=self.request_id)
        if not item.normalized.exists():
            logger.warning(
                "Missing normalized file for %s; conversion may have failed upstream",
                short_hash(item.src),
            )
            return
        logger.info("Preprocessing %s", short_hash(item.normalized))
        pages = await asyncio.to_thread(
            run_pipeline,
            item.normalized,
            self.preproc_ops,
            get_logger=self.get_logger,
            format_hint=item.normalized.suffix.lstrip("."),
            adapter=self.adapter_name,
            adapter_ops=self.adapter_overrides,
            request_id=self.request_id,
        )
        for idx, page in enumerate(pages, start=1):
            await self.ocr_q.put(_OCRItem(item.src, page, item.images, idx))
        keep_normalized = any(page == item.normalized for page in pages)
        if item.normalized != item.src and not keep_normalized:
            item.normalized.unlink(missing_ok=True)


@dataclass
class _OCRPageStage:
    input_dir: Path
    output_dir: Path
    batch_size: int
    get_logger: LoggerFactory
    request_id: str
    create_adapter: Callable[[int | None], BaseOCR]
    adapters: Dict[int | None, BaseOCR]
    gpu_semaphores: Dict[int, asyncio.Semaphore]
    device_cycle: Iterator[int | None]
    write_markdown_cb: Callable[["_OCRItem", OCRResult, BaseOCR, str], Path]
    results: List[Path]
    pending: List[_OCRItem] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    include_language_hint: bool = True
    include_image_hint: bool = True
    capture_timings: bool = True
    capture_tokens: bool = True
    language: str | None = None

    async def __call__(self, item: _OCRItem) -> None:
        batch: list[_OCRItem] | None = None
        async with self.lock:
            self.pending.append(item)
            if len(self.pending) >= self._effective_batch_size:
                batch = self.pending[: self._effective_batch_size]
                del self.pending[: self._effective_batch_size]
        if batch:
            await self._process_with_logging(batch)

    async def flush(self) -> None:
        while True:
            async with self.lock:
                if not self.pending:
                    break
                batch = self.pending[: self._effective_batch_size]
                del self.pending[: self._effective_batch_size]
            if batch:
                await self._process_with_logging(batch)

    @property
    def _effective_batch_size(self) -> int:
        return self.batch_size if self.batch_size > 0 else 1

    async def _process_with_logging(self, batch: list[_OCRItem]) -> None:
        if not batch:
            return
        logger = self.get_logger(__name__, request_id=self.request_id)
        hashes = [short_hash(item.preproc) for item in batch]
        logger.info("OCR batch start: %d item(s) %s", len(batch), hashes)
        try:
            await self._process_batch(batch, logger)
        finally:
            logger.info("OCR batch finish: %d item(s) %s", len(batch), hashes)

    async def _process_batch(
        self, batch: list[_OCRItem], logger: logging.LoggerAdapter
    ) -> None:
        assignments = [
            (
                index,
                item,
                next(self.device_cycle),
                self._build_ocr_item(item),
            )
            for index, item in enumerate(batch)
        ]
        ordered_paths: list[Path | None] = [None] * len(batch)
        adapter_for_index: Dict[int, BaseOCR] = {}

        def store_result(
            index: int,
            ocr_item: _OCRItem,
            adapter: BaseOCR,
            result: OCRResult,
            per_item: float,
        ) -> None:
            if ordered_paths[index] is not None:
                return
            if self.capture_timings:
                result.time_ms = result.time_ms or per_item * 1000.0
            else:
                result.time_ms = None
            if not self.capture_tokens:
                result.tokens_used = None
            md_path = self.write_markdown_cb(
                ocr_item, result, adapter, request_id=self.request_id
            )
            copied_images = _copy_result_images(result, md_path.parent)
            if copied_images:
                result.images = copied_images
            _cleanup_result_paths(result.cleanup_paths)
            observe_latency("orchestrator.page", per_item)
            increment("orchestrated_pages")
            _cleanup_temp_files([p for p in ocr_item.images + [ocr_item.preproc]])
            ordered_paths[index] = md_path

        def failure_result(
            ocr_item: _OCRItem, message: str, per_item: float
        ) -> OCRResult:
            markdown = f"# OCR Failed\n\n> {message}"
            return OCRResult(
                markdown=markdown,
                warnings=[message],
                time_ms=per_item * 1000.0 if self.capture_timings else None,
                provenance={"stage": "models", "status": "failed"},
            )

        def record_failure(
            index: int,
            ocr_item: _OCRItem,
            adapter: BaseOCR,
            exc: Exception | None,
            per_item: float,
        ) -> None:
            hashed = short_hash(ocr_item.preproc)
            if exc is not None:
                logger.exception("OCR adapter failed for %s", hashed)
                hint = f" ({exc.__class__.__name__})"
            else:
                logger.error("OCR adapter failed for %s", hashed)
                hint = ""
            message = f"OCR adapter failure for {hashed}{hint}"
            store_result(
                index,
                ocr_item,
                adapter,
                failure_result(ocr_item, message, per_item),
                per_item,
            )

        def record_success(
            index: int,
            ocr_item: _OCRItem,
            adapter: BaseOCR,
            result: OCRResult,
            per_item: float,
        ) -> None:
            store_result(index, ocr_item, adapter, result, per_item)

        try:
            grouped = self._group_by_device(assignments)
            for dev, group in grouped.items():
                await self._process_device_group(
                    dev,
                    group,
                    adapter_for_index,
                    record_failure,
                    record_success,
                    logger,
                )
        finally:
            self._finalize_batch(
                batch, ordered_paths, adapter_for_index, record_failure
            )

    async def _process_device_group(
        self,
        device: int | None,
        group: list[tuple[int, _OCRItem, int | None, OCRItem]],
        adapter_for_index: Dict[int, BaseOCR],
        record_failure: Callable[
            [int, _OCRItem, BaseOCR, Exception | None, float], None
        ],
        record_success: Callable[[int, _OCRItem, BaseOCR, OCRResult, float], None],
        logger: logging.LoggerAdapter,
    ) -> None:
        if not group:
            return
        group_size = len(group)
        try:
            adapter = self.adapters.setdefault(device, self.create_adapter(device))
        except Exception as exc:
            logger.exception("Failed to initialize adapter for device %s", device)
            base_adapter = self.adapters.get(None) or self.create_adapter(None)
            self.adapters.setdefault(None, base_adapter)
            for index, ocr_item, _device, _ in group:
                record_failure(index, ocr_item, base_adapter, exc, 0.0)
            return
        for index, *_ in group:
            adapter_for_index[index] = adapter
        ocr_items = [entry[3] for entry in group]
        async with _gpu_lock(device, self.gpu_semaphores):
            start = time.perf_counter()
            try:
                results = await adapter.recognize_many(
                    ocr_items,
                    batch_size=self.batch_size,
                    get_logger=self.get_logger,
                    request_id=self.request_id,
                )
            except Exception as exc:
                elapsed = time.perf_counter() - start
                per_item = elapsed / group_size
                for index, ocr_item, _device, _ in group:
                    record_failure(index, ocr_item, adapter, exc, per_item)
                return
        elapsed = time.perf_counter() - start
        per_item = elapsed / group_size
        for (index, ocr_item, _device, _), result in zip(group, results):
            record_success(index, ocr_item, adapter, result, per_item)

    def _finalize_batch(
        self,
        batch: list[_OCRItem],
        ordered_paths: list[Path | None],
        adapter_for_index: Dict[int, BaseOCR],
        record_failure: Callable[
            [int, _OCRItem, BaseOCR, Exception | None, float], None
        ],
    ) -> None:
        base_adapter = self.adapters.get(None)
        for idx, path in enumerate(ordered_paths):
            if path is not None:
                continue
            adapter = adapter_for_index.get(idx)
            if adapter is None:
                if base_adapter is None:
                    base_adapter = self.create_adapter(None)
                    self.adapters.setdefault(None, base_adapter)
                adapter = base_adapter
            record_failure(idx, batch[idx], adapter, None, 0.0)
        for path in ordered_paths:
            assert path is not None
            self.results.append(path)

    @staticmethod
    def _group_by_device(
        assignments: list[tuple[int, _OCRItem, int | None, OCRItem]],
    ) -> Dict[int | None, list[tuple[int, _OCRItem, int | None, OCRItem]]]:
        grouped: Dict[int | None, list[tuple[int, _OCRItem, int | None, OCRItem]]] = {}
        for assignment in assignments:
            grouped.setdefault(assignment[2], []).append(assignment)
        return grouped

    def _build_ocr_item(self, item: _OCRItem) -> OCRItem:
        hints: dict[str, Any] = {}
        if self.include_language_hint and self.language:
            hints["language"] = self.language
        if self.include_image_hint and item.images:
            hints["images"] = [str(image) for image in item.images]
        image_path = item.images[0] if item.images else None
        return OCRItem(
            pdf_path=item.preproc,
            page_index=item.page - 1,
            image_path=image_path,
            hints=hints,
        )


@asynccontextmanager
async def _gpu_lock(
    device: int | None, semaphores: Dict[int, asyncio.Semaphore]
) -> AsyncIterator[None]:
    sem = semaphores.get(device) if device is not None else None
    if sem is None:
        yield
    else:
        async with sem:
            yield


def _cleanup_temp_files(paths: List[Path]) -> None:
    logger = logging.getLogger(__name__)
    max_attempts = 3
    base_delay = 0.05
    for path in paths:
        for attempt in range(1, max_attempts + 1):
            try:
                path.unlink(missing_ok=True)
                break
            except PermissionError as exc:
                if attempt == max_attempts:
                    identifier = hashlib.sha256(
                        path.as_posix().encode("utf-8", "surrogatepass")
                    ).hexdigest()[:8]
                    logger.warning(
                        "Failed to remove temp artifact %s after %d attempt(s) due to"
                        " permissions (errno=%s); leaving it in place.",
                        identifier,
                        attempt,
                        getattr(exc, "errno", "?"),
                    )
                    break
                time.sleep(base_delay * attempt)


def _src_from_item(item: Path | _PreprocItem | _OCRItem) -> Path:
    return item if isinstance(item, Path) else item.src


def _relative_image_subpath(path: Path) -> Path:
    """Return subpath beginning with the first ``images`` component."""

    for index, part in enumerate(path.parts):
        if part == "images":
            return Path(*path.parts[index:])
    return Path(path.name)


def _copy_result_images(result: OCRResult, destination_root: Path) -> list[Path]:
    """Copy adapter produced images into the Markdown output folder."""

    logger = logging.getLogger(__name__)
    copied: list[Path] = []
    for raw in result.images:
        source = Path(raw)
        try:
            if not source.exists():
                logger.warning("Skipping missing OCR image artifact at %s", source)
                continue
            relative = _relative_image_subpath(source)
            target = destination_root / relative
            ensure_dir(target.parent)
            try:
                if source.resolve() == target.resolve():
                    copied.append(target)
                    continue
            except OSError:
                # ``Path.resolve`` may fail on certain virtual filesystems.
                pass
            shutil.copy2(source, target)
            copied.append(target)
        except FileNotFoundError:
            continue
        except OSError as exc:
            logger.warning("Failed to copy OCR image artifact %s: %s", source, exc)
    return copied


def _cleanup_result_paths(paths: Iterable[Path]) -> None:
    """Remove adapter-provided cleanup directories without raising."""

    logger = logging.getLogger(__name__)
    for raw in paths:
        target = Path(raw)
        try:
            shutil.rmtree(target)
        except FileNotFoundError:
            continue
        except NotADirectoryError:
            try:
                target.unlink(missing_ok=True)
            except FileNotFoundError:
                continue
            except OSError as exc:  # pragma: no cover - highly platform specific
                logger.warning("Failed to remove cleanup file %s: %s", target, exc)
        except OSError as exc:  # pragma: no cover - highly platform specific
            logger.warning("Failed to remove cleanup directory %s: %s", target, exc)


class Orchestrator:
    """Coordinate conversion → preprocessing → OCR pipeline."""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        work_dir: Path,
        preproc_ops: Iterable[Mapping[str, Mapping[str, Any]]] | None = None,
        adapter_preproc_overrides: Mapping[str, Mapping[str, bool]] | None = None,
        queues: Mapping[str, int] | None = None,
        batch_size: int | None = None,
        allow_network: bool | None = None,
        adapter_factory: Callable[[int | None, str], BaseOCR] | None = None,
        resources: Resources | None = None,
        *,
        get_logger: LoggerFactory,
        language: str | None = None,
        retries: int | None = None,
        backoff_ms: Iterable[int] | None = None,
        include_language_hint: bool | None = None,
        include_image_hint: bool | None = None,
        capture_timings: bool | None = None,
        capture_tokens: bool | None = None,
    ) -> None:
        if adapter_factory is None:
            msg = "adapter_factory must be provided"
            raise ValueError(msg)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.work_dir = work_dir
        ops_source = preproc_ops if preproc_ops is not None else DEFAULT_PREPROCESSING_OPS
        self.preproc_ops = _copy_preproc_ops(ops_source)
        overrides_source = (
            adapter_preproc_overrides
            if adapter_preproc_overrides is not None
            else DEFAULT_ADAPTER_PREPROC_OVERRIDES
        )
        self.adapter_preproc_overrides = _copy_adapter_overrides(overrides_source)
        queue_defaults = dict(DEFAULT_QUEUE_SIZES)
        if queues:
            queue_defaults.update({key: int(value) for key, value in queues.items()})
        self.queues = queue_defaults
        self.batch_size = batch_size if batch_size is not None else DEFAULT_BATCH_SIZE
        self.allow_network = (
            allow_network if allow_network is not None else DEFAULT_ALLOW_NETWORK
        )
        self._adapter_factory = adapter_factory
        self.language = language or DEFAULT_LANGUAGE
        self.resources = resources or Resources()
        self._get_logger = get_logger
        self._include_language_hint = (
            include_language_hint
            if include_language_hint is not None
            else DEFAULT_HINT_FLAGS["include_language"]
        )
        self._include_image_hint = (
            include_image_hint
            if include_image_hint is not None
            else DEFAULT_HINT_FLAGS["include_images"]
        )
        self._capture_timings = (
            capture_timings
            if capture_timings is not None
            else DEFAULT_METADATA_FLAGS["capture_timings"]
        )
        self._capture_tokens = (
            capture_tokens
            if capture_tokens is not None
            else DEFAULT_METADATA_FLAGS["capture_tokens"]
        )
        self._retries = retries if retries is not None else DEFAULT_RETRIES
        backoff_values = backoff_ms if backoff_ms is not None else DEFAULT_BACKOFF_MS
        backoff_tuple = tuple(int(v) for v in backoff_values)
        if len(backoff_tuple) != 2:
            msg = "backoff_ms must contain exactly two values"
            raise ValueError(msg)
        self._backoff_ms = backoff_tuple
        self._conv_fail_q: asyncio.Queue[tuple[Path, Exception]] = asyncio.Queue()
        self._prep_fail_q: asyncio.Queue[tuple[_PreprocItem, Exception]] = (
            asyncio.Queue()
        )
        self._ocr_fail_q: asyncio.Queue[tuple[_OCRItem, Exception]] = asyncio.Queue()
        self._conv_q: WorkQueue[Path] = WorkQueue(
            maxsize=self.queues["conversion"], failure_q=self._conv_fail_q
        )
        self._prep_q: WorkQueue[_PreprocItem] = WorkQueue(
            maxsize=self.queues["preproc"], failure_q=self._prep_fail_q
        )
        self._ocr_q: WorkQueue[_OCRItem] = WorkQueue(
            maxsize=self.queues["models"], failure_q=self._ocr_fail_q
        )
        self._results: List[Path] = []


    def _write_failures(self) -> None:
        failure_dir = self.output_dir.parent / "failures"
        failures: list[tuple[Path, str, Exception]] = []
        while not self._conv_fail_q.empty():
            item, err = self._conv_fail_q.get_nowait()
            failures.append((_src_from_item(item), "conversion", err))
        while not self._prep_fail_q.empty():
            item, err = self._prep_fail_q.get_nowait()
            failures.append((_src_from_item(item), "preprocess", err))
        while not self._ocr_fail_q.empty():
            item, err = self._ocr_fail_q.get_nowait()
            failures.append((_src_from_item(item), "models", err))
        for src, stage, err in failures:
            rel = src.relative_to(self.input_dir)
            out = (failure_dir / rel).with_suffix(".json")
            payload = json.dumps(
                {"source": str(src), "stage": stage, "error": str(err)}
            ).encode("utf-8")
            atomic_write(out, payload)

    def _create_adapter(self, device: int | None) -> BaseOCR:
        return self._adapter_factory(device, self.language)

    def _write_markdown(
        self,
        item: _OCRItem,
        result: OCRResult,
        adapter: BaseOCR,
        *,
        request_id: str,
    ) -> Path:
        md_path = mirror_output_path(self.input_dir, self.output_dir, item.src, ".md")
        page = item.page
        suffix = f"_p{page}" if page > 1 else ""
        md_path = md_path.with_name(f"{md_path.stem}{suffix}{md_path.suffix}")
        page_index = page - 1
        enriched = dict(result.provenance)
        enriched.update(
            {
                "source_file": str(item.src),
                "source_mime": mimetypes.guess_type(item.src)[0]
                or "application/octet-stream",
                "converted": str(item.preproc),
                "model": adapter.name,
                "model_version": adapter.version,
                "pipeline": "models",
                "page_index": page_index,
            }
        )
        if item.images:
            enriched["images"] = [str(i) for i in item.images]
        if result.time_ms is not None:
            enriched["time_ms"] = result.time_ms
        else:
            enriched.pop("time_ms", None)
        if result.tokens_used is not None:
            enriched["tokens_used"] = result.tokens_used
        else:
            enriched.pop("tokens_used", None)
        result.provenance = enriched
        write_markdown(
            result,
            md_path,
            get_logger=self._get_logger,
            adapter=adapter,
            request_id=request_id,
        )
        return md_path

    async def _ocr_page(
        self,
        item: _OCRItem,
        *,
        request_id: str,
        adapters: Dict[int | None, BaseOCR],
        gpu_semaphores: Dict[int, asyncio.Semaphore],
        device_cycle: Iterator[int | None],
    ) -> None:
        worker = _OCRPageStage(
            self.input_dir,
            self.output_dir,
            self.batch_size,
            self._get_logger,
            request_id,
            self._create_adapter,
            adapters,
            gpu_semaphores,
            device_cycle,
            self._write_markdown,
            self._results,
            include_language_hint=self._include_language_hint,
            include_image_hint=self._include_image_hint,
            capture_timings=self._capture_timings,
            capture_tokens=self._capture_tokens,
            language=self.language,
        )
        await worker(item)

    def _init_devices(
        self, base_adapter: BaseOCR
    ) -> tuple[
        Iterator[int | None], Dict[int, asyncio.Semaphore], Dict[int | None, BaseOCR]
    ]:
        devices = self.resources.devices or []
        device_cycle: Iterator[int | None] = cycle(devices or [None])
        limit = max(
            1,
            int(
                self.resources.max_workers
                * self.resources.max_gpu_utilization
                / (len(devices) or 1)
            ),
        )
        semaphores = {d: asyncio.Semaphore(limit) for d in devices}
        adapters: Dict[int | None, BaseOCR] = {None: base_adapter}
        return device_cycle, semaphores, adapters

    async def _drain_queues(
        self,
        pairs: list[tuple[WorkQueue[Any], list[asyncio.Task[None]]]],
    ) -> None:
        for queue, workers in pairs:
            await queue.join()
            for _ in workers:
                await queue.put(None)
            await asyncio.gather(*workers)

    def _start_workers(
        self,
        qcfg: Dict[str, int],
        retries: int,
        backoff: tuple[int, int],
        create_task: Callable[[Coroutine[Any, Any, None]], asyncio.Task[None]],
        conv_worker: Callable[[Path], Coroutine[Any, Any, None]],
        prep_worker: Callable[[_PreprocItem], Coroutine[Any, Any, None]],
        ocr_worker: Callable[[_OCRItem], Coroutine[Any, Any, None]],
    ) -> tuple[
        list[asyncio.Task[None]],
        list[asyncio.Task[None]],
        list[asyncio.Task[None]],
    ]:
        conv_workers = self._conv_q.start_workers(
            conv_worker,
            concurrency=qcfg["conversion"],
            retries=retries,
            backoff_ms=backoff,
            create_task=create_task,
        )
        prep_workers = self._prep_q.start_workers(
            prep_worker,
            concurrency=qcfg["preproc"],
            retries=retries,
            backoff_ms=backoff,
            create_task=create_task,
        )
        ocr_workers = self._ocr_q.start_workers(
            ocr_worker,
            concurrency=qcfg["models"],
            retries=retries,
            backoff_ms=backoff,
            create_task=create_task,
        )
        return conv_workers, prep_workers, ocr_workers

    async def run(
        self, paths: List[Path], *, request_id: str | None = None
    ) -> List[Path]:
        request_id = request_id or str(uuid.uuid4())

        base_adapter = self._create_adapter(None)
        adapter_name = base_adapter.name

        device_cycle, gpu_semaphores, adapters = self._init_devices(base_adapter)

        conv_worker = _ConvertStage(
            self.work_dir, self._prep_q, self._get_logger, request_id
        )
        prep_worker = _PreprocessStage(
            self.preproc_ops,
            adapter_name,
            self.adapter_preproc_overrides,
            self._ocr_q,
            self._get_logger,
            request_id,
        )
        ocr_worker = _OCRPageStage(
            self.input_dir,
            self.output_dir,
            self.batch_size,
            self._get_logger,
            request_id,
            self._create_adapter,
            adapters,
            gpu_semaphores,
            device_cycle,
            self._write_markdown,
            self._results,
            include_language_hint=self._include_language_hint,
            include_image_hint=self._include_image_hint,
            capture_timings=self._capture_timings,
            capture_tokens=self._capture_tokens,
            language=self.language,
        )

        with offline_mode(self.allow_network):
            async with asyncio.TaskGroup() as tg:
                conv_workers, prep_workers, ocr_workers = self._start_workers(
                    self.queues,
                    self._retries,
                    self._backoff_ms,
                    tg.create_task,
                    conv_worker,
                    prep_worker,
                    ocr_worker,
                )

                for p in paths:
                    await self._conv_q.put(p)

                await self._drain_queues(
                    [
                        (self._conv_q, conv_workers),
                        (self._prep_q, prep_workers),
                        (self._ocr_q, ocr_workers),
                    ]
                )
            await ocr_worker.flush()

        self._write_failures()
        return self._results


__all__ = ["Orchestrator"]
