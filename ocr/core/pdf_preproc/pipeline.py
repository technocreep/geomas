"""PDF preprocessing pipeline with basic logging."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, DefaultDict, Iterable, Tuple

import fitz

from ..io.hashing import short_hash
from ..logging.metrics import increment
from .ops.binarize import binarize
from .ops.crop import crop
from .ops.deskew import deskew
from .ops.fix_fonts import fix_fonts
from .ops.scale import scale
from .ops.split_pages import split_pages

OP_MAP: dict[str, Callable[..., Path | list[Path]]] = {
    "split_pages": split_pages,
    "deskew": deskew,
    "binarize": binarize,
    "scale": scale,
    "crop": crop,
    "fix_fonts": fix_fonts,
}

SAFE_OPS_BY_FORMAT: dict[str, frozenset[str]] = {
    "pdf": frozenset(OP_MAP),
}

SUFFIX_MAP: dict[str, Callable[[dict[str, Any]], str]] = {
    "binarize": lambda _: "_bin",
    "scale": lambda p: f"_sc{p.get('target_dpi', 300)}",
    "crop": lambda _: "_crop",
    "deskew": lambda _: "_deskew",
    "fix_fonts": lambda _: "_fontfix",
}


def _already_processed(
    cache: DefaultDict[str, set[str]], name: str, p: Path, params: dict[str, Any]
) -> bool:
    stem = p.stem
    if stem in cache[name]:
        return True
    suffix_fn = SUFFIX_MAP.get(name)
    if suffix_fn is not None and stem.endswith(suffix_fn(params)):
        cache[name].add(stem)
        return True
    return False


def _has_extractable_text(path: Path) -> bool:
    """Return ``True`` if any page in *path* yields text via PyMuPDF."""

    with fitz.open(path) as doc:
        return any(page.get_text() for page in doc)


def _skip_op(
    name: str,
    p: Path,
    fmt: str,
    forced: Iterable[str],
    logger: logging.LoggerAdapter,
) -> bool:
    if name in set(forced):
        return False
    if name not in {"binarize", "deskew", "fix_fonts"}:
        return False
    if fmt != "pdf":
        logger.debug("Skipping %s for %s: format %s", name, short_hash(p), fmt)
        return True
    if _has_extractable_text(p):
        logger.debug("Skipping %s for %s: text detected", name, short_hash(p))
        return True
    return False


def _register_results(
    results: list[Path],
    paths: list[Path],
    processed_stems: set[str],
    logger: logging.LoggerAdapter,
) -> bool:
    for r in results:
        if r.stem in processed_stems:
            logger.warning(
                "Stem %s reappeared; aborting pipeline to avoid loop", r.stem
            )
            if r not in paths:
                try:
                    r.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Failed to delete %s", r, exc_info=True)
            return False
        processed_stems.add(r.stem)
    return True


def _prepare_ops(
    ops_cfg: list[dict[str, dict[str, Any]]],
    split_pages: bool,
    auto_binarize: bool,
    path: Path,
    fmt: str,
    has_text: bool,
    adapter: str | None,
    fmt_overrides: dict[str, dict[str, bool]] | None,
    adapter_overrides: dict[str, dict[str, bool]] | None,
    logger: logging.LoggerAdapter,
) -> Tuple[list[dict[str, dict[str, Any]]], set[str]]:
    """Return final operation list and forced operations.

    ``fmt_overrides`` and ``adapter_overrides`` map operation names to booleans
    indicating whether the operation should be enabled for the given input
    format or active adapter respectively.
    """

    ops: list[dict[str, dict[str, Any]]] = list(ops_cfg)
    forced: set[str] = set()

    if fmt == "pdf" and split_pages and not any("split_pages" in op for op in ops):
        ops.insert(0, {"split_pages": {}})

    if auto_binarize and not any("binarize" in op for op in ops):
        if fmt != "pdf" or has_text:
            logger.debug(
                "Skipping binarize for %s: format %s or text detected",
                short_hash(path),
                fmt,
            )
        else:
            ops.append({"binarize": {"method": "sauvola"}})

    fmt_cfg = (fmt_overrides or {}).get(fmt, {})
    adapter_cfg = (
        (adapter_overrides or {}).get(adapter, {}) if adapter is not None else {}
    )

    def _apply(cfg: dict[str, bool]) -> None:
        nonlocal ops, forced
        for name, enabled in cfg.items():
            if enabled:
                forced.add(name)
                if not any(name in op for op in ops):
                    ops.append({name: {}})
            else:
                ops = [op for op in ops if name not in op]

    _apply(fmt_cfg)
    _apply(adapter_cfg)

    if fmt == "pdf":
        if has_text:
            ops = [
                op
                for op in ops
                if (name := next(iter(op)))
                not in {"binarize", "deskew", "fix_fonts"}
                or name in forced
            ]
    else:
        allowed_ops = SAFE_OPS_BY_FORMAT.get(fmt, frozenset())
        ops = [
            op
            for op in ops
            if (name := next(iter(op))) in allowed_ops or name in forced
        ]

    return ops, forced


def run_pipeline(
    path: Path,
    ops_cfg: list[dict[str, dict[str, Any]]],
    *,
    get_logger: Callable[[str, str | None], logging.LoggerAdapter[Any]],
    split_pages: bool = False,
    auto_binarize: bool = False,
    format_hint: str | None = None,
    adapter: str | None = None,
    format_ops: dict[str, dict[str, bool]] | None = None,
    adapter_ops: dict[str, dict[str, bool]] | None = None,
    request_id: str | None = None,
) -> list[Path]:
    """Apply configured preprocessing operations to a PDF.

    Parameters
    ----------
    path:
        Path to the input PDF.
    ops_cfg:
        Sequence of operation configurations.
    get_logger:
        Callable returning configured loggers.
    split_pages:
        When ``True``, prepend a ``split_pages`` operation unless one is
        already present.  Typically sourced from ``PDFPreprocConfig``.
    auto_binarize:
        Append a ``binarize`` operation for scanned PDFs when ``True`` and
        no such operation is already configured.
    format_hint:
        Optional input format hint (e.g., "pdf", "png") to enable or disable
        certain operations.
    request_id:
        Identifier for correlating logs. Generated automatically when omitted.

    Returns
    -------
    list[Path]
        Paths to processed PDFs produced by the pipeline.
    """

    logger = get_logger(__name__, request_id=request_id)
    rid = logger.extra.get("request_id")
    fmt = (format_hint or path.suffix.lstrip(".")).lower()
    has_text = _has_extractable_text(path) if fmt == "pdf" else False
    ops, forced = _prepare_ops(
        ops_cfg,
        split_pages,
        auto_binarize,
        path,
        fmt,
        has_text,
        adapter,
        format_ops,
        adapter_ops,
        logger,
    )
    original_hash = short_hash(path)
    if not ops:
        logger.info(
            "No preprocessing ops for %s; returning original file",
            original_hash,
            extra={"request_id": rid},
        )
        increment("processed_pages", 1)
        return [path]

    logger.info(
        "Starting PDF preprocessing for %s with %d ops",
        original_hash,
        len(ops),
        extra={"request_id": rid},
    )
    paths = [path]
    processed: DefaultDict[str, set[str]] = defaultdict(set)
    processed_stems: set[str] = {path.stem}
    for op_cfg in ops:
        name, params = next(iter(op_cfg.items()))
        func = OP_MAP[name]
        logger.debug("Applying %s with params %s", name, params)
        new_paths: list[Path] = []
        for p in paths:
            if _already_processed(processed, name, p, params) or _skip_op(
                name, p, fmt, forced, logger
            ):
                new_paths.append(p)
                continue
            try:
                res = func(p, get_logger=get_logger, request_id=request_id, **params)
            except Exception:
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Failed to delete %s", p, exc_info=True)
                raise
            results = res if isinstance(res, list) else [res]
            processed[name].update(r.stem for r in results)
            if not _register_results(results, paths, processed_stems, logger):
                return paths
            new_paths.extend(results)
            if p not in results:
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Failed to delete %s", p, exc_info=True)
        paths = new_paths
    logger.info(
        "Finished preprocessing %s; produced %d file(s)",
        original_hash,
        len(paths),
        extra={"request_id": rid},
    )
    increment("processed_pages", len(paths))
    return paths


__all__ = ["run_pipeline"]
