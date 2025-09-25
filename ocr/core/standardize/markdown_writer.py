"""Write standardized Markdown files."""

from __future__ import annotations

from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Callable

import yaml

from ..io.fs import atomic_write
from ..io.hashing import file_sha256, short_hash
from ..logging.metrics import increment
from ..models.base import OCRResult, BaseOCR
from .md_spec import REQUIRED_FIELDS, validate_markdown


def write_markdown(
    result: OCRResult,
    path: Path,
    *,
    get_logger: Callable[[str, str | None], logging.LoggerAdapter],
    adapter: BaseOCR | None = None,
    request_id: str | None = None,
) -> Path:
    """Write ``result`` to ``path`` as standardized Markdown with provenance."""

    logger = get_logger(__name__, request_id=request_id)
    if not result.provenance:
        msg = "OCRResult.provenance must contain metadata for Markdown output"
        raise ValueError(msg)
    source = str(result.provenance.get("source_file", "").strip())
    if not source:
        msg = "provenance requires a 'source_file' entry"
        raise ValueError(msg)
    hash_target = Path(result.provenance.get("converted", source))
    meta: Dict[str, object] = {
        **result.provenance,
        "source_file": source,
        "sha256": file_sha256(hash_target),
        "created_utc": datetime.utcnow().isoformat(),
        "model": getattr(adapter, "name", result.provenance.get("model")),
        "model_version": getattr(
            adapter, "version", result.provenance.get("model_version")
        ),
        "pipeline": result.provenance.get("pipeline", "ocr"),
    }
    if "time_ms" not in meta and result.time_ms is not None:
        meta["time_ms"] = result.time_ms
    if "tokens_used" not in meta and result.tokens_used is not None:
        meta["tokens_used"] = result.tokens_used
    missing = {k for k in REQUIRED_FIELDS if meta.get(k) is None}
    assert not missing, f"Missing metadata field(s): {', '.join(sorted(missing))}"
    front = yaml.safe_dump(meta, sort_keys=True)
    text = f"---\n{front}---\n{result.markdown}\n"
    if not validate_markdown(text):
        raise ValueError("Markdown does not conform to spec")
    logger.debug("Writing Markdown for %s", short_hash(hash_target))
    atomic_write(path, text.encode("utf-8"))
    increment("markdown_files", 1)
    return path


__all__ = ["write_markdown"]
