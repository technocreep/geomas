from __future__ import annotations

import logging
from dataclasses import dataclass
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Mapping, Sequence

from langchain_core.documents import Document

from geomas.core.data.custom_dataloaders import LangChainDocumentLoader
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
    SUPPORTED_SUFFIXES = HTML_SUFFIXES | MARKDOWN_SUFFIXES | TEXT_SUFFIXES | JSON_SUFFIXES

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
        self._markdown_chunker: TextChunker | None = None

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
        for entry in sorted(root.rglob("*")):
            if not entry.is_file():
                continue
            if self._is_supported(entry):
                yield entry

    def _is_supported(self, path: Path) -> bool:
        return path.suffix.lower() in self.allowed_suffixes

    def _load_file(self, path: Path, document_name: str | None) -> tuple[list[Document], list[Path]]:
        suffix = path.suffix.lower()
        resolved_name = document_name or path.stem
        cleanup_paths: list[Path] = []

        if suffix in self.JSON_SUFFIXES:
            documents = self._load_json_documents(path)
        elif suffix in self.HTML_SUFFIXES:
            documents, cleanup_paths = self._parse_textual_file(path, resolved_name, "html")
        elif suffix in self.MARKDOWN_SUFFIXES | self.TEXT_SUFFIXES:
            documents, cleanup_paths = self._parse_markdown_file(path, resolved_name)
        else:
            logger.info("Skipping unsupported file '%s'", path)
            return [], []

        enriched = self._enrich_metadata(path, resolved_name, documents)
        return enriched, cleanup_paths

    def _load_json_documents(self, path: Path) -> list[Document]:
        try:
            loader = LangChainDocumentLoader(path)
            documents = list(loader.lazy_load())
        except Exception as exc:
            logger.error("Failed to load JSON document '%s': %s", path, exc)
            return []
        return documents

    def _parse_textual_file(
        self,
        path: Path,
        document_name: str,
        document_type: str,
    ) -> tuple[list[Document], list[Path]]:
        parser = self.parser
        cleanup_paths: list[Path] = []
        if parser is None:
            logger.warning("Parser is not available; skipping '%s'", path)
            return [], cleanup_paths

        raw_text = self._read_text(path)
        if raw_text is None:
            return [], cleanup_paths

        try:
            preprocessed_text, _ = parser.preprocessing(document_name, path.parent, raw_text)
        except Exception as exc:
            logger.error("Failed to preprocess '%s': %s", path, exc)
            return [], cleanup_paths

        processed_path = path.parent / f"{document_name}_processed.html"
        if processed_path.exists():
            cleanup_paths.append(processed_path)

        documents = self._invoke_parser(
            parser,
            preprocessed_text,
            document_name,
            document_type,
            path,
        )

        return documents, cleanup_paths

    def _parse_markdown_file(self, path: Path, document_name: str) -> tuple[list[Document], list[Path]]:
        parser = self.parser
        raw_text = self._read_text(path)
        if raw_text is None:
            return [], []

        if parser is None:
            documents = self._chunk_markdown(raw_text, document_name, path)
            return documents, []

        try:
            preprocessed_text, _ = parser.preprocessing(document_name, path.parent, raw_text)
        except Exception:
            preprocessed_text = raw_text
            cleanup_paths: list[Path] = []
        else:
            processed_path = path.parent / f"{document_name}_processed.html"
            cleanup_paths = [processed_path] if processed_path.exists() else []

        documents = self._invoke_parser(parser, preprocessed_text, document_name, "markdown", path)
        return documents, cleanup_paths

    def _invoke_parser(
        self,
        parser: DocumentParser,
        raw_text: str,
        document_name: str,
        document_type: str,
        path: Path,
    ) -> list[Document]:
        try:
            documents = parser.parse(
                raw_text,
                document_name,
                document_type,
                source_path=str(path),
            )
        except TypeError:
            documents = parser.parse(raw_text, document_name, document_type)
        except Exception as exc:
            logger.error("Parser failed for '%s': %s", path, exc)
            return []
        return list(documents)

    def _chunk_markdown(self, raw_text: str, document_name: str, path: Path) -> list[Document]:
        logger.info(
            "Parser unavailable; chunking Markdown document '%s' using TextChunker fallback",
            path,
        )
        try:
            chunker = self._get_markdown_chunker()
            documents = chunker.apply_chunking(raw_text, document_name, "markdown")
        except Exception as exc:
            logger.error("Failed to chunk Markdown document '%s': %s", path, exc)
            return []
        return list(documents)

    def _get_markdown_chunker(self) -> TextChunker:
        if self._markdown_chunker is None:
            self._markdown_chunker = TextChunker(chunking_params=self.chunking_params)
        return self._markdown_chunker

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
    ) -> list[Document]:
        valid_documents = [doc for doc in documents if isinstance(doc, Document)]
        chunk_count = len(valid_documents)
        enriched: list[Document] = []
        for index, document in enumerate(valid_documents):
            metadata = dict(document.metadata or {})
            existing_source = metadata.get("source")
            if existing_source in {None, "", document_name}:
                metadata["source"] = str(source_path)
            else:
                metadata["source"] = existing_source
            metadata["document_name"] = document_name
            metadata["chunk_index"] = index
            metadata["chunk_count"] = chunk_count
            enriched.append(Document(page_content=document.page_content, metadata=metadata))
        return enriched


def format_text_context(
    text_context: Iterable[Sequence[object]],
    *,
    limit: int = 3,
) -> list[dict[str, object]]:
    """Summarise raw ``text_context`` entries for presentation layers."""

    formatted: list[dict[str, object]] = []
    for index, entry in enumerate(text_context):
        if index >= limit:
            break
        if not isinstance(entry, Sequence) or len(entry) < 4:
            continue
        doc_id, raw_text, metadata, score = entry[:4]
        metadata_map: Mapping[str, object] = metadata if isinstance(metadata, Mapping) else {}
        formatted.append(
            {
                "id": doc_id,
                "document": metadata_map.get("document_name")
                or metadata_map.get("source")
                or str(doc_id),
                "score": score,
                "preview": str(raw_text).strip().replace("\n", " "),
            }
        )
    return formatted
