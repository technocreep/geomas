"""Minimal helper for running the OCR pipeline on a folder of documents."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Mapping, Sequence

from core.api import process_paths
from core.models.base import BaseOCR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration variables (edit as needed)
INPUT_DIR = Path("input/raw")
OUTPUT_DIR = Path("output/markdown")
WORK_DIR = Path("work")
ADAPTER_NAME = "marker"


def _marker_factory() -> BaseOCR:
    from core.models.adapters.marker import Marker

    return Marker()


def _mineru_factory() -> BaseOCR:
    from core.models.adapters.mineru import MinerU

    return MinerU()


def _olmocr_factory() -> BaseOCR:
    from core.models.adapters.olmocr import OlmOCR

    return OlmOCR()


def _qwen_vl_factory() -> BaseOCR:
    from core.models.adapters.qwen_vl import QwenVL

    return QwenVL()


ADAPTER_FACTORIES: Mapping[str, Callable[[], BaseOCR]] = {
    "marker": _marker_factory,
    "mineru": _mineru_factory,
    "olmocr": _olmocr_factory,
    "qwen_vl": _qwen_vl_factory,
}


def build_adapter() -> tuple[str, BaseOCR] | None:
    """Instantiate the OCR adapter requested by ``ADAPTER_NAME``."""

    key = ADAPTER_NAME.lower()
    factory = ADAPTER_FACTORIES.get(key)
    if factory is None:
        available = ", ".join(sorted(ADAPTER_FACTORIES))
        logger.warning("adapter %s is not available (choose from %s)", ADAPTER_NAME, available)
        return None

    try:
        adapter = factory()
    except Exception as exc:  # pragma: no cover - defensive logging for optional deps
        logger.warning("adapter %s unavailable: %s", ADAPTER_NAME, exc)
        return None

    adapter_name = getattr(adapter, "name", None) or key
    return adapter_name, adapter


def _page_sort_key(path: Path) -> tuple[int, str]:
    """Return a key sorting Markdown paths by page number then lexicographically."""

    stem, sep, page = path.stem.rpartition("_p")
    number = int(page) if sep and page.isdigit() else 1
    return (number, str(path))


TEXT_EXTS = {
    ".txt",
    ".csv",
    ".tsv",
    ".html",
    ".htm",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
}

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}

SUPPORTED_EXTS = {".pdf", ".pptx", *TEXT_EXTS, *IMG_EXTS}


def collect_supported_paths(root: Path) -> list[Path]:
    """Gather supported file paths under ``root``."""

    files = (p for p in root.rglob("*") if p.is_file())
    return sorted(
        {
            p
            for p in files
            if not p.name.startswith("~$") and p.suffix.lower() in SUPPORTED_EXTS
        }
    )


def _markdown_has_body(path: Path) -> bool:
    """Return ``True`` when the Markdown file contains non-empty body content."""

    text = path.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith("---"):
        parts = stripped.split("---", 2)
        if len(parts) < 3:
            return False
        body = parts[2].strip()
        return bool(body)
    return True


def _validate_outputs(paths: Sequence[Path]) -> None:
    """Ensure produced Markdown files include body content."""

    empty = [path for path in paths if not _markdown_has_body(path)]
    if empty:
        joined = ", ".join(str(path) for path in empty)
        raise RuntimeError(
            "Markdown output empty after stripping front matter: %s" % joined
        )


def _preload_if_available(adapter: BaseOCR) -> None:
    preload = getattr(adapter, "preload_models", None)
    if callable(preload):
        preload()


def main() -> None:
    """Convert input documents to Markdown using the selected adapter."""

    built = build_adapter()
    if built is None:
        return

    adapter_name, adapter = built
    input_dir = INPUT_DIR.resolve()
    output_dir = OUTPUT_DIR.resolve()
    work_dir = WORK_DIR.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    input_paths = collect_supported_paths(input_dir)
    if not input_paths:
        logger.info("no supported inputs under %s", input_dir)
        return

    _preload_if_available(adapter)

    adapter_output_dir = (output_dir / adapter_name).resolve()
    adapter_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "processing %d file(s) with adapter %s", len(input_paths), adapter_name
    )

    md_paths = process_paths(
        input_paths,
        adapter=adapter,
        input_dir=input_dir,
        output_dir=adapter_output_dir,
        work_dir=work_dir / "runs" / adapter_name,
    )

    _validate_outputs(md_paths)

    for md_path in sorted(md_paths, key=_page_sort_key):
        logger.info("wrote %s via %s", md_path, adapter_name)


if __name__ == "__main__":
    main()
