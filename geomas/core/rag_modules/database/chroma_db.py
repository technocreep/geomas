from __future__ import annotations

import logging
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from langchain_core.documents.base import Document
from langchain_core.embeddings import Embeddings

from geomas.core.rag_modules.data_adapter import AdapterResult, DataLoaderAdapter
from geomas.core.rag_modules.database.database_utils import ChromaDatabaseClient

logger = logging.getLogger(__name__)

class ChromaDatabaseStore:
    """Persist and query document artefacts stored in ChromaDB."""

    _MAX_INSERT_BATCH_SIZE = 5_461

    def __init__(
        self,
        client: ChromaDatabaseClient | None = None,
        *,
        collection_name: str,
        embedding: Embeddings | None = None,
    ) -> None:
        if not collection_name:
            raise ValueError("collection_name must be provided")

        self.client = client or ChromaDatabaseClient()
        self.collection_name = collection_name
        self.collection = self.client.ensure_collection(collection_name)
        self.embedding = embedding

        self._closed = False

    def close(self) -> None:
        """Close the underlying client and release cached connectors."""

        if getattr(self, "_closed", False):
            return

        self._closed = True

        client = getattr(self, "client", None)
        if client is not None:
            close_method = getattr(client, "close", None)
            if callable(close_method):
                try:
                    close_method()
                except Exception as exc:
                    logger.debug("Failed to close ChromaDatabaseClient: %s", exc)
            self.client = None

        self.collection = None

    def add_documents(self, documents: Sequence[Document], *, batch_size: int = 32) -> bool:
        """Persist text documents in ChromaDB."""

        if not documents:
            logger.info("No documents received for ingestion into ChromaDB")
            return False

        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        if self.collection is None:
            raise RuntimeError("Text collection is not initialised")

        if self.embedding is None:
            raise RuntimeError("ChromaDatabaseStore requires an embedding to add documents")

        for slice_start in range(0, len(documents), self._MAX_INSERT_BATCH_SIZE):
            document_slice = documents[slice_start : slice_start + self._MAX_INSERT_BATCH_SIZE]
            if not document_slice:
                continue

            chunk_texts = [text_chunk.page_content for text_chunk in document_slice]
            embeddings: list[list[float]] = []
            for batch_start in range(0, len(chunk_texts), batch_size):
                window = chunk_texts[batch_start : batch_start + batch_size]
                if not window:
                    continue
                vectors = self.embedding.embed_documents(window)
                embeddings.extend(
                    [
                        vector.tolist() if isinstance(vector, np.ndarray) else list(vector)
                        for vector in vectors
                    ]
                )

            if len(embeddings) != len(document_slice):
                raise RuntimeError(
                    "Embedding function returned a different number of vectors than input documents",
                )

            metadata_payload = [
                dict(text_chunk.metadata) if isinstance(text_chunk.metadata, Mapping) else {}
                for text_chunk in document_slice
            ]

            self.collection.add(
                ids=[str(uuid.uuid4()) for _ in range(len(document_slice))],
                documents=chunk_texts,
                embeddings=embeddings,
                metadatas=metadata_payload,
            )

        return True

    def search(
        self,
        query: str,
        *,
        collection_type: str = "text",
        top_k: int = 5,
        filters: Mapping[str, object] | None = None,
    ) -> dict:
        """Query a Chroma collection for the given text."""

        if collection_type.lower() != "text":
            raise ValueError(f"Unknown collection type: {collection_type}")

        if self.collection is None:
            raise RuntimeError("Text collection is not initialised")

        if self.embedding is None:
            raise RuntimeError("ChromaDatabaseStore requires an embedding to search")

        query_embedding = self.embedding.embed_query(query)
        prepared_embedding = (
            query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else list(query_embedding)
        )

        return self.client.query_chromadb(
            self.collection,
            query_embeddings=[prepared_embedding],
            metadata_filter=dict(filters) if isinstance(filters, Mapping) and filters else None,
            chunk_num=top_k,
        )


@dataclass(slots=True, frozen=True)
class ProcessingResult:
    """Outcome of a database ingestion attempt."""

    success: bool
    documents_ingested: int = 0
    summaries_created: int = 0


class DatabaseRagPipeline:
    """Ingest artefacts into the Chroma store using the adapter pipeline."""

    def __init__(
        self,
        *,
        store: ChromaDatabaseStore | None = None,
        parser: object | None = None,
        data_loader: DataLoaderAdapter | None = None,
        default_text_top_k: int | None = None,
    ) -> None:
        if store is None:
            raise ValueError("ChromaDatabaseStore instance is required")

        self.store = store
        self.parser = parser
        if data_loader is None:
            self.data_loader = DataLoaderAdapter(parser=parser)
        else:
            self.data_loader = data_loader
            if parser is not None and hasattr(self.data_loader, "parser"):
                self.data_loader.parser = parser

        self.default_text_top_k = default_text_top_k

    def process(
        self,
        folder_path: Path | str,
        *,
        loader_overrides: Mapping[str, object] | None = None,
        document_name: str | None = None,
    ) -> ProcessingResult:
        """Process ``folder_path`` and persist extracted documents."""

        path = Path(folder_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Provided folder does not exist: {path}")

        try:
            adapter_result: AdapterResult = self.data_loader.load_and_transform(
                path,
                document_name=document_name,
                loader_overrides=loader_overrides,
            )
        except Exception as exc:
            logger.exception("Failed to load artefact '%s': %s", path, exc)
            return ProcessingResult(success=False)

        documents = adapter_result.documents
        if not documents:
            logger.info("No documents produced for '%s'", path)
            return ProcessingResult(success=False)

        try:
            stored = self.store.add_documents(documents)
        except Exception as exc:
            logger.exception("Failed to store documents in ChromaDB: %s", exc)
            return ProcessingResult(success=False)

        if not stored:
            return ProcessingResult(success=False)

        self._cleanup_ingest_artifacts(adapter_result.cleanup_paths)

        return ProcessingResult(
            success=True,
            documents_ingested=len(documents),
            summaries_created=0,
        )

    def _cleanup_ingest_artifacts(self, cleanup_paths: Iterable[Path]) -> None:
        for candidate in cleanup_paths:
            path = Path(candidate)
            try:
                if not path.exists():
                    continue
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            except FileNotFoundError:
                continue
            except Exception as exc:
                logger.warning("Failed to remove ingest artefact '%s': %s", path, exc)

    def search_for_papers(
        self,
        query: str,
        *,
        top_k: int | None = None,
        final_top_k: int | None = None,
        filters: Mapping[str, object] | None = None,
    ) -> dict:
        """Return a list of candidate sources for ``query`` based on text chunks."""

        limit = self._resolve_limit(top_k)
        raw_docs = self.store.search(
            query,
            collection_type="text",
            top_k=limit,
            filters=dict(filters) if isinstance(filters, Mapping) and filters else None,
        )

        metadatas = raw_docs.get("metadatas", [[]])
        candidates = metadatas[0] if metadatas else []

        sources: list[str] = []
        seen: set[str] = set()
        max_results = final_top_k if final_top_k is not None else limit
        for metadata in candidates:
            if not isinstance(metadata, Mapping):
                continue
            source = metadata.get("source")
            if not isinstance(source, str):
                continue
            if source in seen:
                continue
            seen.add(source)
            sources.append(source)
            if len(sources) >= max_results:
                break

        return {"answer": sources}

    def retrieve_context(
        self,
        query: str,
        relevant_papers: Mapping[str, object] | None = None,
        *,
        filters: Mapping[str, object] | None = None,
        text_top_k: int | None = None,
    ) -> tuple[list[tuple[str, str, dict, float]], dict]:
        """Retrieve text context for ``query`` using previously ingested chunks."""

        candidate_sources = []
        if relevant_papers:
            candidate_sources = list(
                value for value in relevant_papers.get("answer", []) if isinstance(value, str)
            )

        filter_spec: dict[str, object] = {}
        if candidate_sources:
            filter_spec["source"] = {"$in": candidate_sources}

        if isinstance(filters, Mapping) and filters:
            filter_spec.update(dict(filters))

        active_filter = filter_spec or None
        text_limit = self._resolve_limit(text_top_k)

        raw_text_context = self.store.search(
            query,
            collection_type="text",
            top_k=text_limit,
            filters=active_filter,
        )

        scored_docs = self._build_scored_context(raw_text_context, text_limit)
        return scored_docs, {"answer": candidate_sources}

    def _resolve_limit(self, explicit: int | None) -> int:
        if explicit is not None:
            return explicit
        if self.default_text_top_k is not None:
            return self.default_text_top_k
        return 5

    @staticmethod
    def _build_scored_context(raw_results: Mapping[str, object], top_k: int) -> list[tuple[str, str, dict, float]]:
        documents = raw_results.get("documents", [[]])
        metadatas = raw_results.get("metadatas", [[]])
        ids = raw_results.get("ids", [[]])

        candidate_docs = documents[0] if isinstance(documents, Sequence) and documents else []
        candidate_metas = metadatas[0] if isinstance(metadatas, Sequence) and metadatas else []
        candidate_ids = ids[0] if isinstance(ids, Sequence) and ids else []

        limit = min(top_k, len(candidate_docs), len(candidate_metas), len(candidate_ids))
        if limit <= 0:
            return []

        scores = DatabaseRagPipeline._extract_scores(raw_results, len(candidate_docs))

        scored_docs: list[tuple[str, str, dict, float]] = []
        for index in range(limit):
            doc_id = str(candidate_ids[index])
            doc_text = candidate_docs[index]
            metadata = candidate_metas[index] if isinstance(candidate_metas[index], Mapping) else {}
            score = scores[index] if index < len(scores) else float("nan")
            scored_docs.append((doc_id, doc_text, dict(metadata), float(score)))
        return scored_docs

    @staticmethod
    def _extract_scores(raw_results: Mapping[str, object], expected: int) -> list[float]:
        for key in ("distances", "similarities"):
            values = raw_results.get(key)
            if not isinstance(values, Sequence) or not values:
                continue
            primary = values[0] if isinstance(values[0], Sequence) else values
            try:
                scores = [float(item) for item in primary][:expected]
            except (TypeError, ValueError):
                continue
            if len(scores) == expected:
                return scores
        return [float("nan")] * expected

