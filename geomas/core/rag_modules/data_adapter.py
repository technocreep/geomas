from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Mapping, Sequence

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_community.document_loaders import JSONLoader

from geomas.core.data.custom_dataloaders import LangChainDocumentLoader
from geomas.core.rag_modules.parser.rag_parser import DocumentParser
from geomas.core.rag_modules.steps.chunking import TextChunker

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class AdapterResult:
    """Container describing the outcome of a loader invocation."""
    documents: list[Document]
    cleanup_paths: tuple[Path, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.documents)


class DataLoaderAdapter:
    """Load raw artefacts and adapt them for the LangChain ecosystem."""
    HTML_SUFFIXES = {".html", ".htm"}
    MARKDOWN_SUFFIXES = {".md", ".markdown", ".mmd"}
    TEXT_SUFFIXES = {".txt"}
    JSON_SUFFIXES = {".json", ".jsonl"}
    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff", ".tif"}
    SUPPORTED_SUFFIXES = HTML_SUFFIXES | MARKDOWN_SUFFIXES | TEXT_SUFFIXES | JSON_SUFFIXES | IMAGE_SUFFIXES

    def __init__(
        self,
        loader_type: str | None = None,
        *,
        parser: DocumentParser | None = None,
        loader_params: Mapping[str, object] | None = None,
        transformation_config: Mapping[str, object] | None = None,
        allowed_suffixes: Iterable[str] | None = None,
        chunking_params: Mapping[str, Any] | None = None,
    ) -> None:
        self.loader_type = loader_type or "auto"
        self.parser = parser
        self.loader_params = dict(loader_params or {})
        self.transformation_config = dict(transformation_config or {})
        self.chunking_params: Mapping[str, Any] | None
        if chunking_params is None:
            self.chunking_params = None
        else:
            self.chunking_params = dict(chunking_params)
        if allowed_suffixes is None:
            self.allowed_suffixes = set(self.SUPPORTED_SUFFIXES)
        else:
            self.allowed_suffixes = {suffix.lower() for suffix in allowed_suffixes}
        self._chunker = TextChunker(chunking_params=self.chunking_params)

    def load_and_transform(
        self,
        source: str | Path,
        document_name: str | None = None,
        *,
        loader_overrides: Mapping[str, object] | None = None,
    ) -> AdapterResult:
        """Load data from ``source`` and return parsed ``Document`` objects."""
        del loader_overrides  # Loader overrides are not used in the streamlined adapter.

        path = Path(source).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Provided source does not exist: {source}")

        if path.is_dir():
            documents, cleanup_candidates = self._load_directory(path, document_name)
        else:
            documents, cleanup_candidates = self._load_file(path, document_name)

        unique_cleanup_paths = tuple(dict.fromkeys(cleanup_candidates))

        return AdapterResult(documents=documents, cleanup_paths=unique_cleanup_paths)

    def _load_directory(
        self, path: Path, document_name: str | None
    ) -> tuple[list[Document], list[Path]]:
        documents: list[Document] = []
        cleanup_paths: list[Path] = []
        for file_path in self._iter_supported_files(path):
            file_documents, file_cleanup = self._load_file(file_path, document_name)
            documents.extend(file_documents)
            cleanup_paths.extend(file_cleanup)
        return documents, cleanup_paths

    def _iter_supported_files(self, root: Path) -> Iterator[Path]:
        for current_root, dirs, files in os.walk(root, followlinks=True):
            dirs.sort()
            files.sort()
            base_path = Path(current_root)
            for filename in files:
                candidate = base_path / filename
                if candidate.is_file() and self._is_supported(candidate):
                    yield candidate

    def _is_supported(self, path: Path) -> bool:
        return path.suffix.lower() in self.allowed_suffixes

    def _load_file(self, path: Path, document_name: str | None) -> tuple[list[Document], list[Path]]:
        suffix = path.suffix.lower()
        resolved_name = document_name or path.stem + path.suffix
        cleanup_paths: list[Path] = []

        fingerprint, last_modified, size_bytes = self._collect_file_metadata(path)

        if suffix in self.JSON_SUFFIXES:
            entries = self._entries_from_json(path)
        elif suffix in self.HTML_SUFFIXES | self.MARKDOWN_SUFFIXES | self.TEXT_SUFFIXES:
            entries, cleanup_paths = self._entries_from_textual_file(
                path, resolved_name, suffix
            )
        else:
            logger.info("Skipping unsupported file '%s'", path)
            return [], []

        documents = self._chunk_entries(entries, resolved_name)

        enriched = self._enrich_metadata(
            path,
            resolved_name,
            documents,
            fingerprint=fingerprint,
            last_modified=last_modified,
            size_bytes=size_bytes,
        )
        return enriched, cleanup_paths

    @staticmethod
    def _collect_file_metadata(path: Path) -> tuple[str | None, str | None, int | None]:
        fingerprint: str | None = None
        last_modified: str | None = None
        size_bytes: int | None = None

        try:
            stat_result = path.stat()
        except OSError as exc:
            logger.warning("Failed to stat '%s' while computing metadata: %s", path, exc)
        else:
            size_bytes = int(stat_result.st_size)
            try:
                last_modified = datetime.fromtimestamp(stat_result.st_mtime, tz=timezone.utc).isoformat()
            except (OverflowError, OSError, ValueError) as exc:
                logger.warning("Failed to normalise mtime for '%s': %s", path, exc)

        try:
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(65536), b""):
                    hasher.update(chunk)
            fingerprint = hasher.hexdigest()
        except OSError as exc:
            logger.warning("Failed to fingerprint '%s': %s", path, exc)

        return fingerprint, last_modified, size_bytes

    def _entries_from_json(self, path: Path) -> list[tuple[str, Mapping[str, object]]]:
        try:
            loader = JSONLoader(
                file_path=path,
                jq_schema='{page_content: .page_content, metadata: .metadata}',
                text_content=True,
                content_key="page_content",
                metadata_func=lambda record, index: record["metadata"]
            )
            raw_documents = list(loader.lazy_load())
        except Exception as exc:
            logger.error("Failed to load JSON document '%s': %s", path, exc)
            return []

        entries: list[tuple[str, Mapping[str, object]]] = []
        for document in raw_documents:
            metadata = dict(document.metadata or {})
            metadata.setdefault("source", metadata.get("source_path") or str(path))
            metadata.setdefault("source_path", str(path))
            entries.append((str(document.page_content), metadata))
        return entries

    def _entries_from_textual_file(
        self, path: Path, document_name: str, suffix: str
    ) -> tuple[list[tuple[str, Mapping[str, object]]], list[Path]]:
        cleanup_paths: list[Path] = []
        raw_text = self._read_text(path)
        if raw_text is None:
            return [], cleanup_paths

        processed_text = raw_text
        if self.parser is not None:
            try:
                processed_text, _ = self.parser.preprocessing(
                    document_name, path.parent, raw_text
                )
            except Exception as exc:
                logger.warning("Failed to preprocess '%s': %s", path, exc)
            else:
                processed_path = path.parent / f"{document_name}_processed.html"
                if processed_path.exists():
                    cleanup_paths.append(processed_path)

        normalized_text = self._normalize_text_content(processed_text, suffix, path)

        entries = [(normalized_text, {"source_path": str(path)})]
        return entries, cleanup_paths

    @staticmethod
    def _clean_text_value(content: str) -> str:
        normalised = content.replace("\r\n", "\n").replace("\r", "\n")
        try:
            from geomas.core.rag_modules.convertation import pdf_to_json as converters

            if hasattr(converters, "clean_text"):
                placeholder = "<GEOMAS_NL>"
                prepared = normalised.replace("\n", f" {placeholder} ")
                cleaned = converters.clean_text(prepared)
                return cleaned.replace(placeholder, "\n").strip()
        except ModuleNotFoundError:
            return normalised.strip()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("Failed to apply converter clean_text: %s", exc)
        return normalised.strip()

    @staticmethod
    def _html_to_text_content(source_path: Path) -> str | None:
        try:
            from geomas.core.rag_modules.convertation import pdf_to_json as converters
        except ModuleNotFoundError:
            return None

        try:
            return converters.html_to_text(str(source_path))
        except Exception as exc:  # pragma: no cover - converter is optional
            logger.debug("html_to_text failed for '%s': %s", source_path, exc)
            return None

    def _normalize_text_content(
        self, content: str, suffix: str, source_path: Path
    ) -> str:
        try:
            if suffix in self.HTML_SUFFIXES:
                converter_text = self._html_to_text_content(source_path)
                if converter_text:
                    return converter_text
                try:
                    soup = BeautifulSoup(content, "html.parser")
                    return soup.get_text(separator=" ")
                except Exception:
                    return content
            return content
        except Exception as exc:
            logger.warning("Failed to normalise content for '%s': %s", source_path, exc)
            return content

    def _chunk_entries(
        self, entries: Sequence[tuple[str, Mapping[str, object]]], document_name: str
    ) -> list[Document]:
        documents: list[Document] = []
        for text, base_metadata in entries:
            try:
                chunks = self._chunker.apply_chunking(text, document_name, "markdown")
            except Exception as exc:
                logger.error("Failed to chunk '%s': %s", document_name, exc)
                continue
            for chunk in chunks:
                cleaned = self._clean_text_value(chunk.page_content)
                if not cleaned:
                    continue
                metadata = dict(chunk.metadata or {})
                metadata.update(base_metadata or {})
                documents.append(Document(page_content=cleaned, metadata=metadata))
        return documents

    @staticmethod
    def _read_text(path: Path) -> str | None:
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            logger.error("Failed to read '%s': %s", path, exc)
            return None

    def _enrich_metadata(
        self,
        source_path: Path,
        document_name: str,
        documents: Sequence[Document],
        *,
        fingerprint: str | None = None,
        last_modified: str | None = None,
        size_bytes: int | None = None,
    ) -> list[Document]:
        valid_documents = [doc for doc in documents if isinstance(doc, Document)]
        chunk_count = len(valid_documents)
        enriched: list[Document] = []
        for index, document in enumerate(valid_documents):
            metadata = dict(document.metadata or {})
            existing_source = metadata.get("source")
            if existing_source in {
                None,
                "",
                document_name,
                f"{document_name}.pdf",
            }:
                metadata["source"] = str(source_path)
            else:
                metadata["source"] = existing_source
            metadata.setdefault("source_path", str(source_path))
            metadata["document_name"] = document_name
            metadata["chunk_index"] = index
            metadata["chunk_count"] = chunk_count
            if fingerprint:
                metadata["source_fingerprint"] = fingerprint
            if last_modified:
                metadata["source_last_modified"] = last_modified
            if size_bytes is not None:
                metadata["source_size_bytes"] = size_bytes
            enriched.append(Document(page_content=document.page_content, metadata=metadata))
        return enriched


def format_text_context(
    text_context: Iterable[Sequence[object]],
    limit: int | None = None,
) -> list[dict[str, object]]:
    """Summarise raw ``text_context`` entries for presentation layers.

    Args:
        text_context: Iterable of scored context tuples returned by
            :class:`DatabaseRagPipeline`.
        limit: Optional cap on the number of items to include in the formatted
            output. When omitted, all entries are preserved.
    """
    formatted: list[dict[str, object]] = []

    for index, entry in enumerate(text_context):
        if limit is not None and len(formatted) >= limit:
            break
        if not isinstance(entry, Sequence) or len(entry) < 5:
            continue
        doc_id, chunk_index, doc_text, metadata, score = entry[:5]
        metadata_map: Mapping[str, object] = metadata if isinstance(metadata, Mapping) else {}
        formatted.append(
            {
                "id": doc_id,
                "chunk_index": chunk_index,
                "document": metadata_map.get("document_name")
                or metadata_map.get("source")
                or str(doc_id),
                "score": metadata_map.get("normalized_score"),
                "preview": str(doc_text).strip().replace("\n", " "),
                "type": metadata_map.get("type", "text"),
                "database_scope": metadata_map.get("scope"),
                "source_path": metadata_map.get("source_path"),
            }
        )
    return formatted

