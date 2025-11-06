"""Split PDF into single-page documents using pypdfium2."""

from __future__ import annotations

import inspect
import logging
import os
import time
from contextlib import contextmanager, closing
from decimal import Decimal, InvalidOperation
from pathlib import Path
from tempfile import mkstemp
from typing import Callable, Iterator, List

import fitz
import pypdfium2

from core.io.hashing import short_hash
from core.logging.metrics import observe_latency


def _open_pdf_document(
    path: Path,
    *,
    logger: logging.LoggerAdapter,
    request_id: str | None,
) -> tuple[pypdfium2.PdfDocument, Path | None]:
    """Return an open :class:`PdfDocument` for *path* with data-format recovery."""

    try:
        return pypdfium2.PdfDocument(path), None
    except FileNotFoundError as exc:
        logger.error(
            "split_pages read error: %s",
            short_hash(path),
            extra={"request_id": request_id},
            exc_info=exc,
        )
        msg = f"Unable to read PDF {short_hash(path)}"
        raise ValueError(msg) from exc
    except pypdfium2.PdfiumError as exc:
        if "Data format error" not in str(exc):
            logger.error(
                "split_pages read error: %s",
                short_hash(path),
                extra={"request_id": request_id},
                exc_info=exc,
            )
            msg = f"Unable to read PDF {short_hash(path)}"
            raise ValueError(msg) from exc

        logger.warning(
            "split_pages data format error: %s",
            short_hash(path),
            extra={"request_id": request_id},
            exc_info=exc,
        )
        try:
            sanitized_path = _rewrite_pdf_with_pymupdf(path)
        except Exception as rewrite_exc:  # pragma: no cover - defensive
            logger.error(
                "split_pages rewrite failed: %s",
                short_hash(path),
                extra={"request_id": request_id},
                exc_info=rewrite_exc,
            )
            msg = f"Unable to read PDF {short_hash(path)}"
            raise ValueError(msg) from rewrite_exc

        logger.warning(
            "split_pages retry sanitized copy: %s -> %s",
            short_hash(path),
            short_hash(sanitized_path),
            extra={"request_id": request_id},
        )

        try:
            return pypdfium2.PdfDocument(sanitized_path), sanitized_path
        except (FileNotFoundError, pypdfium2.PdfiumError) as retry_exc:
            logger.error(
                "split_pages read error after rewrite: %s -> %s",
                short_hash(path),
                short_hash(sanitized_path),
                extra={"request_id": request_id},
                exc_info=retry_exc,
            )
            sanitized_path.unlink(missing_ok=True)
            msg = f"Unable to read PDF {short_hash(path)}"
            raise ValueError(msg) from retry_exc


def split_pages(
    path: Path,
    *,
    get_logger: Callable[[str, str | None], logging.LoggerAdapter],
    request_id: str | None = None,
    split: bool = True,
) -> List[Path]:
    """Return a list of single-page PDFs extracted from *path*.

    Parameters
    ----------
    path:
        Source PDF to split.
    get_logger:
        Callable returning configured loggers.
    request_id:
        Identifier for correlating logs.
    split:
        When ``False`` the function short-circuits and returns ``[path]``.

    Returns
    -------
    list[Path]
        Paths to the generated single-page PDFs or ``[path]`` when disabled.
    """

    logger = get_logger(__name__, request_id=request_id)
    rid = logger.extra.get("request_id")
    if not split:
        logger.info("split_pages skip: %s", short_hash(path), extra={"request_id": rid})
        return [path]

    logger.info("split_pages start: %s", short_hash(path), extra={"request_id": rid})
    save_signature = inspect.signature(pypdfium2.PdfDocument.save)
    supports_relinearize = "relinearize" in save_signature.parameters
    supports_version = "version" in save_signature.parameters
    version_value: int | None = None
    if supports_version:
        try:
            version_value = int(Decimal("1.7") * 10)
        except (InvalidOperation, ValueError):
            version_value = None
    cleanup_path: Path | None = None
    source, cleanup_path = _open_pdf_document(path, logger=logger, request_id=rid)

    def finalize(paths: List[Path]) -> List[Path]:
        logger.info(
            "split_pages finish: %s -> %s",
            short_hash(path),
            [short_hash(p) for p in paths],
            extra={"request_id": rid},
        )
        return paths

    try:
        with closing(source):
            total_pages = len(source)
            if total_pages == 0:
                return finalize([])

            out_paths: List[Path] = []
            for index in range(total_pages):
                page_number = index + 1
                out_path = path.with_name(f"{path.stem}_p{page_number}.pdf")
                start = time.perf_counter()
                with tempfile_path(path.parent, out_path.name) as tmp_path:
                    save_kwargs: dict[str, object] = {}
                    if supports_version and version_value is not None:
                        save_kwargs["version"] = version_value
                    if supports_relinearize:
                        save_kwargs["relinearize"] = True
                    try:
                        with closing(pypdfium2.PdfDocument.new()) as writer:
                            writer.import_pages(source, [index])
                            writer.save(str(tmp_path), **save_kwargs)
                    except (OSError, pypdfium2.PdfiumError, TypeError) as exc:
                        logger.error(
                            "split_pages write failed: %s page %d",
                            short_hash(path),
                            page_number,
                            extra={"request_id": rid},
                            exc_info=exc,
                        )
                        msg = (
                            "Unable to write split PDF page"
                            f" {page_number} from {short_hash(path)}"
                        )
                        raise ValueError(msg) from exc
                    try:
                        _validate_with_pdfium(tmp_path)
                    except (FileNotFoundError, OSError) as exc:
                        logger.error(
                            "split_pages validation failed: %s page %d",
                            short_hash(path),
                            page_number,
                            extra={"request_id": rid},
                            exc_info=exc,
                        )
                        msg = (
                            "Unable to validate split PDF page"
                            f" {page_number} from {short_hash(path)}"
                        )
                        raise ValueError(msg) from exc
                    except (pypdfium2.PdfiumError, ValueError) as exc:
                        if "Data format error" not in str(exc):
                            logger.error(
                                "split_pages validation failed: %s page %d",
                                short_hash(path),
                                page_number,
                                extra={"request_id": rid},
                                exc_info=exc,
                            )
                            msg = (
                                "Unable to validate split PDF page"
                                f" {page_number} from {short_hash(path)}"
                            )
                            raise ValueError(msg) from exc

                        logger.warning(
                            "split_pages validation retry via pymupdf: %s page %d (pdfium error: %s)",
                            short_hash(path),
                            page_number,
                            str(exc),
                            extra={"request_id": rid},
                        )
                        try:
                            _rewrite_page_with_pymupdf(path, page_number - 1, tmp_path)
                        except Exception as fallback_exc:  # pragma: no cover - defensive
                            logger.error(
                                "split_pages fallback write failed: %s page %d",
                                short_hash(path),
                                page_number,
                                extra={"request_id": rid},
                                exc_info=fallback_exc,
                            )
                            msg = (
                                "Unable to write split PDF page"
                                f" {page_number} from {short_hash(path)}"
                            )
                            raise ValueError(msg) from fallback_exc

                        try:
                            _validate_with_pymupdf(tmp_path, expected_pages=1)
                        except (FileNotFoundError, OSError) as retry_exc:
                            logger.error(
                                "split_pages fallback validation failed: %s page %d",
                                short_hash(path),
                                page_number,
                                extra={"request_id": rid},
                                exc_info=retry_exc,
                            )
                            msg = (
                                "Unable to validate split PDF page"
                                f" {page_number} from {short_hash(path)}"
                            )
                            raise ValueError(msg) from retry_exc
                        except (RuntimeError, ValueError) as retry_exc:
                            logger.error(
                                "split_pages fallback validation failed: %s page %d",
                                short_hash(path),
                                page_number,
                                extra={"request_id": rid},
                                exc_info=retry_exc,
                            )
                            msg = (
                                "Unable to validate split PDF page"
                                f" {page_number} from {short_hash(path)}"
                            )
                            raise ValueError(msg) from retry_exc
                elapsed = time.perf_counter() - start
                observe_latency("split_pages.page", elapsed)
                logger.debug(
                    "split_pages page %d: %s in %.3f s",
                    page_number,
                    short_hash(out_path),
                    elapsed,
                )
                out_paths.append(out_path)

            return finalize(out_paths)
    finally:
        if cleanup_path is not None:
            cleanup_path.unlink(missing_ok=True)


def _validate_with_pdfium(pdf_path: Path) -> None:
    with closing(pypdfium2.PdfDocument(str(pdf_path))):
        pass


def _validate_with_pymupdf(pdf_path: Path, *, expected_pages: int | None = None) -> None:
    with fitz.open(pdf_path) as document:
        page_count = document.page_count
        if page_count == 0:
            msg = f"Split PDF {short_hash(pdf_path)} produced zero pages"
            raise ValueError(msg)
        if expected_pages is not None and page_count != expected_pages:
            msg = (
                f"Split PDF {short_hash(pdf_path)} expected {expected_pages} pages"
                f" but found {page_count}"
            )
            raise ValueError(msg)
        for page_index in range(page_count):
            document.load_page(page_index)


def _rewrite_pdf_with_pymupdf(source: Path) -> Path:
    fd, tmp_name = mkstemp(dir=source.parent, suffix=".pdf")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with fitz.open(source) as document:
            document.save(tmp_path, garbage=4, deflate=True, clean=True)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return tmp_path


def _rewrite_page_with_pymupdf(source: Path, page_index: int, destination: Path) -> None:
    with fitz.open(source) as document:
        if page_index < 0 or page_index >= document.page_count:
            msg = (
                "Unable to access page"
                f" {page_index + 1} from {short_hash(source)}"
            )
            raise ValueError(msg)

        with fitz.open() as single_page:
            single_page.insert_pdf(
                document,
                from_page=page_index,
                to_page=page_index,
            )
            single_page.save(destination, garbage=4, deflate=True, clean=True)


@contextmanager
def tempfile_path(dir: Path, name: str) -> Iterator[Path]:
    """Yield a temp :class:`Path` then atomically replace it with *name*.

    The temporary file is created using :func:`mkstemp` to ensure the file
    descriptor is closed immediately.  This mirrors the semantics needed on
    Windows where open files cannot be replaced.
    """

    fd, tmp_name = mkstemp(dir=dir, suffix=".pdf")
    os.close(fd)
    tmp_path = Path(tmp_name)
    target = dir / name
    try:
        yield tmp_path
        os.replace(tmp_path, target)
    finally:
        tmp_path.unlink(missing_ok=True)


__all__ = ["split_pages"]
