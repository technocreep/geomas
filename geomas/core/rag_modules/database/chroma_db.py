from __future__ import annotations

import logging
import math
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


def _distance_to_similarity(distance: float) -> float:
    """Convert a distance score into a similarity where higher is better."""
    if not math.isfinite(distance):
        return 0.0
    safe_distance = max(distance, 0.0)
    return 1.0 / (1.0 + safe_distance)


def _extract_scores_from_payload(
    raw_results: Mapping[str, object], expected: int
) -> list[float]:
    """Return similarity scores derived from a Chroma payload."""
    for key in ("similarities", "distances"):
        values = raw_results.get(key)
        if not isinstance(values, Sequence) or not values:
            continue

        primary = values[0] if isinstance(values[0], Sequence) else values
        try:
            scores = [float(item) for item in primary][:expected]
        except (TypeError, ValueError):
            continue

        if len(scores) != expected:
            continue

        if key == "distances":
            return [_distance_to_similarity(score) for score in scores]
        return scores

    return [float("nan")] * expected


def _normalise_scope_values(value: object) -> tuple[list[str], list[str]]:
    """Parse scope filter values into recognised and unknown candidates."""
    recognised: list[str] = []
    unknown: list[str] = []

    def _register(candidate: object) -> None:
        if not isinstance(candidate, str):
            unknown.append(repr(candidate))
            return
        scope_label = candidate.strip().lower()
        if not scope_label:
            unknown.append(repr(candidate))
            return
        if scope_label in {"global", "local"}:
            if scope_label not in recognised:
                recognised.append(scope_label)
        else:
            unknown.append(repr(candidate))

    if isinstance(value, Mapping):
        in_clause = value.get("$in")
        if isinstance(in_clause, Sequence):
            for entry in in_clause:
                _register(entry)
        elif value:
            for entry in value.values():
                _register(entry)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for entry in value:
            _register(entry)
    else:
        _register(value)

    return recognised, unknown

class ChromaDatabaseStore:
    """Persist and query document artefacts stored in ChromaDB."""
    _MAX_INSERT_BATCH_SIZE = 5_461

    def __init__(
        self,
        client: ChromaDatabaseClient | None = None,
        *,
        collection_name: str = "abc",
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

    def add_documents(
        self,
        documents: Sequence[Document],
        *,
        batch_size: int = 32,
        namespace: str | None = None,
    ) -> bool:
        """Persist text documents in ChromaDB."""

        _ = namespace  # Namespace is ignored for single-store deployments.

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

    def store_visual_documents(
        self,
        documents: list[Document],
        collection_name: str,
        embeddings: np.ndarray = None,
    ) -> None:
        """
        Store visual document descriptions in a ChromaDB collection.

        This method adds Document objects containing visual descriptions (e.g., descriptions of geological maps,
        charts, or figures) to a specified ChromaDB collection. It either uses precomputed embeddings (if provided)
        or generates them in batches using the internal embedding model. Each document is stored with its text content,
        a unique ID, and enriched metadata indicating its type as 'visual'.

        Args:
            documents (list[Document]): A list of LangChain Document objects, each containing page_content (text description)
                                        and optional metadata.
            collection_name (str): The name of the ChromaDB collection where documents will be stored.
            embeddings (np.ndarray, optional): Precomputed embeddings for the document texts.
            window_size (int, optional): The batch size used when generating embeddings internally. Defaults to 15.

        Returns:
            None

        Notes:
            - The method enriches each document's metadata with {"type": "visual"}.
            - Duplicate storage is not checked; new UUIDs are assigned on every call.
        """
        if not documents:
            logger.warning("No documents to store")
            return
        logger.info(f"Storing descriptions in ChromaDB collection: {collection_name}")
        # Get or create collection for visual documents
        visual_collection = self.client.get_or_create_chroma_collection(
            collection=collection_name, embedding_function=None
        )
        # Prepare documents and metadata
        doc_texts = [doc.page_content for doc in documents]
        doc_metadatas = [{"type": "visual", **doc.metadata} for doc in documents]

        # Store in ChromaDB
        try:
            visual_collection.add(
                ids=[str(uuid.uuid4()) for _ in range(len(documents))],
                documents=doc_texts,
                embeddings=embeddings,
                metadatas=doc_metadatas,
            )
            logger.info(
                f"Stored {len(documents)} documents in collection '{collection_name}'"
            )

        except Exception as e:
            logger.error(
                f"Failed to store documents in ChromaDB collection '{collection_name}': {e}"
            )
            return

    def _ensure_collections_ready(self) -> None:
        """Ensure the backing collection exists."""
        client = getattr(self, "client", None)
        name = getattr(self, "collection_name", None)
        if client is None or name is None:
            return

        try:
            self.collection = client.ensure_collection(name)
        except Exception as exc:
            logger.warning("Failed to ensure Chroma collection '%s': %s", name, exc)


@dataclass(slots=True)
class _SearchEntry:
    """Internal helper representing a scored document."""
    scope: str
    identifier: str
    document: str
    metadata: dict[str, object]
    similarity: float
    distance: float | None = None


class PartitionedChromaDatabaseStore:
    """Aggregate store coordinating global and optional local Chroma instances."""
    def __init__(
        self,
        *,
        global_store: ChromaDatabaseStore,
        local_store: ChromaDatabaseStore | None = None,
    ) -> None:
        if global_store is None:
            raise ValueError("global_store must be provided")

        self.global_store = global_store
        self.local_store = local_store
        self.client = getattr(global_store, "client", None)
        self.local_client = getattr(local_store, "client", None) if local_store else None
        self.collection_name = getattr(global_store, "collection_name", None)
        self.collection = self.client.ensure_collection(self.collection_name)
        self.embedding = getattr(global_store, "embedding", None)

        self._closed = False

        self._ensure_collections_ready()

    def close(self) -> None:
        """Close both backing stores and release their resources."""
        if getattr(self, "_closed", False):
            return

        self._closed = True

        for store in (self.global_store, self.local_store):
            if store is None:
                continue
            try:
                store.close()
            except Exception as exc:
                logger.debug("Failed to close %s store: %s", store.collection_name, exc)

        self.global_store = None
        self.local_store = None
        self.client = None
        self.local_client = None
        self.collection_name = None
        self.embedding = None

    def ensure_collections(self) -> None:
        """Ensure all managed collections exist."""
        for store in (self.global_store, self.local_store):
            if store is None:
                continue
            store._ensure_collections_ready()

    def _ensure_collections_ready(self) -> None:
        self.ensure_collections()

    def add_documents(
        self,
        documents: Sequence[Document],
        *,
        batch_size: int = 32,
        namespace: str | None = None,
    ) -> bool:
        """Delegate ingestion to the appropriate backing store."""
        store = self._select_store(namespace)
        return store.add_documents(documents, batch_size=batch_size)

    def search(
        self,
        query: str,
        *,
        collection_type: str = "text",
        top_k: int = 5,
        filters: Mapping[str, object] | None = None,
    ) -> dict:
        """Query all relevant stores and merge their responses."""
        if collection_type.lower() != "text":
            raise ValueError(f"Unknown collection type: {collection_type}")

        scopes, base_filters = self._prepare_filters(filters)
        targets = self._resolve_targets(scopes)

        aggregate: list[_SearchEntry] = []
        for scope_label, store in targets:
            payload = store.search(
                query,
                collection_type=collection_type,
                top_k=top_k,
                filters=dict(base_filters) if base_filters is not None else None,
            )
            aggregate.extend(self._expand_payload(payload, scope_label))

        if not aggregate:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "similarities": [[]],
                "distances": [[]],
            }

        try:
            limit = max(0, int(top_k))
        except (TypeError, ValueError):
            limit = 5

        aggregate.sort(
            key=lambda entry: (
                float("-inf") if math.isnan(entry.similarity) else entry.similarity
            ),
            reverse=True,
        )
        trimmed = aggregate[:limit] if limit else []

        similarities = [entry.similarity for entry in trimmed]
        distances = [entry.distance if entry.distance is not None else float("nan") for entry in trimmed]

        return {
            "ids": [[entry.identifier for entry in trimmed]],
            "documents": [[entry.document for entry in trimmed]],
            "metadatas": [[entry.metadata for entry in trimmed]],
            "similarities": [similarities],
            "distances": [distances],
        }

    def _prepare_filters(
        self, filters: Mapping[str, object] | None
    ) -> tuple[list[str], Mapping[str, object] | None]:
        if not isinstance(filters, Mapping):
            return [], None

        base_filters = dict(filters)
        raw_scope = base_filters.pop("scope", None)
        if raw_scope is None:
            return [], base_filters

        scopes, unknown = _normalise_scope_values(raw_scope)
        if unknown:
            logger.warning("Ignoring unsupported scope filter values: %s", ", ".join(unknown))

        return scopes, base_filters

    def _resolve_targets(
        self, scopes: Sequence[str]
    ) -> list[tuple[str, ChromaDatabaseStore]]:
        global_store = getattr(self, "global_store", None)
        local_store = getattr(self, "local_store", None)

        targets: list[tuple[str, ChromaDatabaseStore]] = []
        request_global = not scopes or "global" in scopes
        request_local = ("local" in scopes) if scopes else bool(local_store)

        if "local" in scopes and local_store is None:
            logger.info(
                "Local scope requested but no local store is configured; falling back to global scope",
            )
            request_global = True
            request_local = False

        if request_global and global_store is not None:
            targets.append(("global", global_store))

        if request_local and local_store is not None:
            targets.append(("local", local_store))

        if not targets:
            if global_store is None and local_store is None:
                raise RuntimeError("No Chroma stores are available for querying")
            if global_store is not None:
                targets.append(("global", global_store))

        return targets

    def _select_store(self, namespace: str | None) -> ChromaDatabaseStore:
        label = (namespace or "global").strip().lower()
        local_store = getattr(self, "local_store", None)
        global_store = getattr(self, "global_store", None)

        if global_store is None:
            raise RuntimeError("Global store is not available for ingestion")

        if label == "local":
            if local_store is not None:
                return local_store
            logger.info(
                "Local namespace requested but unavailable; defaulting ingestion to the global store",
            )
            return global_store

        if label not in {"", "global"}:
            logger.warning("Unknown namespace '%s'; defaulting to global store", namespace)

        return global_store

    def _expand_payload(self, payload: Mapping[str, object], scope: str) -> list[_SearchEntry]:
        if not isinstance(payload, Mapping):
            return []

        documents_row = self._first_row(payload.get("documents"))
        metadata_row = self._first_row(payload.get("metadatas"))
        ids_row = self._first_row(payload.get("ids"))
        distances_row = self._first_row(payload.get("distances"))

        lengths = [
            len(documents_row),
            len(metadata_row) if metadata_row is not None else len(documents_row),
            len(ids_row),
        ]
        expected = min(lengths) if lengths else 0
        if expected <= 0:
            return []

        scores = _extract_scores_from_payload(payload, expected)

        distances: list[float | None] = []
        if distances_row is not None:
            for index in range(expected):
                try:
                    distances.append(float(distances_row[index]))
                except (TypeError, ValueError, IndexError):
                    distances.append(float("nan"))
        else:
            distances = [float("nan")] * expected

        entries: list[_SearchEntry] = []
        for index in range(expected):
            metadata_value = metadata_row[index] if metadata_row and index < len(metadata_row) else {}
            if isinstance(metadata_value, Mapping):
                metadata = dict(metadata_value)
            else:
                metadata = {}
            metadata["scope"] = scope

            document_value = documents_row[index] if index < len(documents_row) else ""
            identifier_value = ids_row[index] if index < len(ids_row) else ""
            similarity = scores[index] if index < len(scores) else float("nan")

            entries.append(
                _SearchEntry(
                    scope=scope,
                    identifier=str(identifier_value),
                    document=str(document_value),
                    metadata=metadata,
                    similarity=float(similarity),
                    distance=float(distances[index]) if index < len(distances) else float("nan"),
                )
            )

        return entries

    @staticmethod
    def _first_row(value: object) -> Sequence[object]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            return []
        if not value:
            return []
        first = value[0]
        if isinstance(first, Sequence) and not isinstance(first, (str, bytes, bytearray)):
            return first
        return value

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
        namespace: str = "global",
    ) -> ProcessingResult:
        """Process ``folder_path`` and persist extracted documents.

        Args:
            folder_path: Root directory containing artefacts to ingest.
            loader_overrides: Optional adapter overrides applied to the data loader.
            document_name: Optional explicit document label for metadata enrichment.
            namespace: Target store namespace (``"global"`` or ``"local"``).
        """
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
            stored = self.store.add_documents(documents, namespace=namespace)
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

        scope_filter = filter_spec.get("scope")
        if scope_filter is not None:
            _, unknown_scopes = _normalise_scope_values(scope_filter)
            if unknown_scopes:
                logger.warning(
                    "retrieve_context received unsupported scope filters: %s",
                    ", ".join(unknown_scopes),
                )

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
    def _distance_to_similarity(distance: float) -> float:
        return _distance_to_similarity(distance)

    @classmethod
    def _extract_scores(cls, raw_results: Mapping[str, object], expected: int) -> list[float]:
        return _extract_scores_from_payload(raw_results, expected)

