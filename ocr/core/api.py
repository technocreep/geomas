"""High-level Python API helpers for running the OCR pipeline.

These helpers provide convenient defaults for input, output and working
directories while ensuring the required folder structure exists before the OCR
pipeline starts. When a working directory is not supplied, a temporary
directory is created for the duration of the call and cleaned up afterwards.
"""

from __future__ import annotations

import asyncio
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Callable, Iterable, List

from .logging import LoggerRegistry
from .models import BaseOCR
from .runtime.orchestrator import Orchestrator
from .runtime.resources import Resources


DEFAULT_INPUT_DIR = Path("input/raw")
DEFAULT_WORK_DIR = Path("work")
DEFAULT_OUTPUT_DIR = Path("output/markdown")
DEFAULT_BATCH_SIZE = 8
DEFAULT_LANGUAGE = "auto"
DEFAULT_ALLOW_NETWORK = False


def _run_paths(
    paths: Iterable[Path | str],
    adapter_factory: Callable[[int | None, str], BaseOCR],
    *,
    input_dir: Path,
    output_dir: Path,
    work_dir: Path | None,
    allow_network: bool,
    batch_size: int,
    language: str,
    resources: Resources | None,
    devices: List[int] | None,
    request_id: str | None,
) -> List[Path]:
    """Execute OCR for a collection of paths using the provided adapter factory."""

    res = resources or Resources(devices=devices)
    registry = LoggerRegistry()

    with ExitStack() as stack:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        if work_dir is None:
            tmp_dir = stack.enter_context(tempfile.TemporaryDirectory())
            work_dir_path = Path(tmp_dir)
        else:
            work_dir_path = Path(work_dir)
            work_dir_path.mkdir(parents=True, exist_ok=True)

        orch = Orchestrator(
            input_dir,
            output_dir,
            work_dir_path,
            batch_size=batch_size,
            allow_network=allow_network,
            adapter_factory=adapter_factory,
            resources=res,
            get_logger=registry.get_logger,
            language=language,
        )
        result = asyncio.run(
            orch.run([Path(p) for p in paths], request_id=request_id)
        )

    return result


def process_path(
    path: Path | str,
    adapter: BaseOCR,
    *,
    input_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    work_dir: Path | str | None = None,
    allow_network: bool | None = None,
    batch_size: int | None = None,
    language: str | None = None,
    resources: Resources | None = None,
    devices: List[int] | None = None,
    request_id: str | None = None,
) -> List[Path]:
    """Process a single filesystem path through the OCR pipeline."""

    return process_paths(
        [path],
        adapter,
        input_dir=input_dir,
        output_dir=output_dir,
        work_dir=work_dir,
        allow_network=allow_network,
        batch_size=batch_size,
        language=language,
        resources=resources,
        devices=devices,
        request_id=request_id,
    )


def process_paths(
    paths: Iterable[Path | str],
    adapter: BaseOCR,
    *,
    input_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    work_dir: Path | str | None = None,
    allow_network: bool | None = None,
    batch_size: int | None = None,
    language: str | None = None,
    resources: Resources | None = None,
    devices: List[int] | None = None,
    request_id: str | None = None,
) -> List[Path]:
    """Process multiple filesystem paths through the OCR pipeline."""

    resolved_input_dir = (
        Path(input_dir) if input_dir is not None else Path(DEFAULT_INPUT_DIR)
    )
    resolved_output_dir = (
        Path(output_dir) if output_dir is not None else Path(DEFAULT_OUTPUT_DIR)
    )
    resolved_work_dir = (
        Path(work_dir) if work_dir is not None else Path(DEFAULT_WORK_DIR)
    )
    resolved_allow_network = (
        allow_network if allow_network is not None else DEFAULT_ALLOW_NETWORK
    )
    resolved_batch_size = batch_size if batch_size is not None else DEFAULT_BATCH_SIZE
    resolved_language = language or DEFAULT_LANGUAGE

    def factory(dev: int | None, lang: str = resolved_language) -> BaseOCR:
        del dev, lang
        return adapter

    return _run_paths(
        paths,
        factory,
        input_dir=resolved_input_dir,
        output_dir=resolved_output_dir,
        work_dir=resolved_work_dir,
        allow_network=resolved_allow_network,
        batch_size=resolved_batch_size,
        language=resolved_language,
        resources=resources,
        devices=devices,
        request_id=request_id,
    )


__all__ = ["process_path", "process_paths"]
