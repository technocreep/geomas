from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever


def _coerce_scope_filter(scopes: Sequence[str] | str | None) -> Dict[str, object] | None:
    """Return a Chroma metadata filter matching one or more ``scopes``."""
    if scopes is None:
        return None
    if isinstance(scopes, str):
        cleaned = scopes.strip()
        return {"scope": cleaned} if cleaned else None

    values = [scope for scope in scopes if isinstance(scope, str) and scope.strip()]
    if not values:
        return None
    if len(values) == 1:
        return {"scope": values[0]}
    return {"scope": {"$in": values}}


def _normalise_relevance(score: float | None) -> float | None:
    if score is None:
        return None
    try:
        distance = float(score)
    except (TypeError, ValueError):
        return None
    if distance != distance:  # NaN guard
        return None
    return 1.0 / (1.0 + max(distance, 0.0))


def _document_with_score(
    document: Document, score: float | None, *, scope: str | None = None
) -> Document:
    metadata = dict(document.metadata or {})

    raw_distance = metadata.get("distances")
    existing_score: float | None = None
    for key in ("relevance_score", "normalized_score"):
        if key in metadata:
            try:
                existing_score = float(metadata[key])
                break
            except (TypeError, ValueError):
                continue

    normalised = existing_score
    if normalised is None:
        normalised = _normalise_relevance(score if score is not None else raw_distance)
    if normalised is not None:
        metadata.setdefault("relevance_score", normalised)
        metadata.setdefault("normalized_score", normalised)
    if raw_distance is None and score is not None:
        metadata.setdefault("distances", float(score))

    if scope:
        metadata.setdefault("scope", scope)

    return Document(
        page_content=document.page_content,
        metadata=metadata,
        id=document.id,
    )


@dataclass
class DocsSearcherModels:
    embedding_model: Embeddings | None = None
    vector_store: Chroma | None = None


class Retriever:
    def __init__(
        self,
        top_k: int,
        docs_searcher_models: DocsSearcherModels,
        preprocess_query: Optional[Callable[[str], str]] = None,
        *,
        search_type: str = "similarity_score_threshold",
        search_kwargs: Mapping[str, Any] | None = None,
        scoped_vector_stores: Mapping[str, Chroma] | None = None,
    ) -> None:
        """Chroma-backed retriever using LangChain's native wrapper."""
        self.top_k = top_k
        self.embedding_function = docs_searcher_models.embedding_model
        self.vector_store = docs_searcher_models.vector_store
        self.scoped_vector_stores = {
            str(key): store
            for key, store in (scoped_vector_stores or {}).items()
            if isinstance(key, str) and hasattr(store, "as_retriever")
        }
        self.preprocess_query = preprocess_query
        self.search_type = search_type
        self.search_kwargs = dict(search_kwargs or {})

    def _build_retriever(
        self,
        k: int,
        filter: Mapping[str, Any] | None,
        *,
        vector_store: Chroma | None = None,
        allow_default_store: bool = True,
    ) -> BaseRetriever:
        store = vector_store or (self.vector_store if allow_default_store else None)
        if store is None:
            raise ValueError("Chroma vector store is required for retrieval")

        search_kwargs: Dict[str, Any] = {"k": k, **self.search_kwargs}
        if filter:
            search_kwargs["filter"] = filter

        return store.as_retriever(
            search_type=self.search_type,
            search_kwargs=search_kwargs,
        )

    @staticmethod
    def _prepare_filters(
        filters: Optional[Mapping[str, Any]], scopes: Sequence[str] | str | None
    ) -> Dict[str, Any]:
        filter_payload: Dict[str, Any] = dict(filters or {})
        scope_filter = _coerce_scope_filter(scopes)
        if scope_filter and "scope" not in filter_payload:
            filter_payload.update(scope_filter)
        return filter_payload

    @classmethod
    def compose_filters(
        cls, filters: Optional[Mapping[str, Any]], scopes: Sequence[str] | str | None
    ) -> Dict[str, Any]:
        """Public wrapper for composing Chroma filters with optional scopes."""
        return cls._prepare_filters(filters, scopes)

    def _prepare_query(self, query: str | None) -> str | None:
        if self.preprocess_query is not None and query is not None:
            return self.preprocess_query(query)
        return query

    def _resolve_vector_store(self, scope: str | None) -> Chroma | None:
        if scope is None:
            return self.vector_store
        if scope in self.scoped_vector_stores:
            return self.scoped_vector_stores.get(scope)
        if self.scoped_vector_stores:
            return None
        return self.vector_store

    def _resolve_limit(self, top_k: int | None) -> int:
        limit = int(top_k) if top_k is not None else int(self.top_k)
        return max(limit, 0)

    def _search_by_images(
        self,
        images: Sequence[str | bytes | bytearray],
        *,
        k: int,
        filter: Mapping[str, Any] | None,
    ) -> list[Document]:
        if self.vector_store is None:
            warnings.warn("Chroma vector store is required for image retrieval")
            return []

        embedding_model = self.embedding_function
        if embedding_model is None or not hasattr(embedding_model, "embed_image"):
            warnings.warn("Configured embedding function does not support images")
            return []

        embeddings = embedding_model.embed_image(list(images))
        if not embeddings:
            return []

        query_embedding = embeddings[0]
        results = self.vector_store.similarity_search_by_vector_with_relevance_scores(
            embedding=query_embedding,
            k=k,
            filter=dict(filter) if filter else None,
        )
        return [_document_with_score(doc, score) for doc, score in results]

    def _execute_search(
        self,
        query: str | None,
        *,
        limit: int,
        filters: Mapping[str, Any] | None,
        query_images: Sequence[str | bytes | bytearray] | None,
        vector_store: Chroma | None = None,
        scope: str | None = None,
        allow_default_store: bool = True,
    ) -> list[Document]:
        filter_payload = dict(filters or {})
        if query_images:
            return self._search_by_images(
                query_images,
                k=limit,
                filter=filter_payload or None,
            )
        documents = vector_store.similarity_search_with_score(
            query=query, 
            k=limit, 
            filter=filter_payload or None,
        )
        return [
            _document_with_score(doc[0], doc[1], scope=scope)
            for doc in documents
        ]

    @staticmethod
    def _uuid_filter_from_docs(documents: Sequence[Document]) -> dict[str, object] | None:
        uuids = [
            doc.metadata.get("uuid")
            for doc in documents
            if isinstance(doc.metadata, Mapping) and isinstance(doc.metadata.get("uuid"), str)
        ]
        uuids = [uuid for uuid in uuids if uuid]
        if not uuids:
            return None
        return {"uuid": {"$in": uuids}}

    def _run_single_search(
        self,
        query: str | None,
        *,
        top_k: int | None,
        filters: Mapping[str, Any] | None,
        query_images: Sequence[str | bytes | bytearray] | None,
        vector_store: Chroma | None = None,
        scope: str | None = None,
        allow_default_store: bool = True,
    ) -> list[Document]:
        limit = self._resolve_limit(top_k)
        prepared_query = self._prepare_query(query)
        return self._execute_search(
            prepared_query,
            limit=limit,
            filters=dict(filters or {}),
            query_images=query_images if query_images else None,
            vector_store=vector_store,
            scope=scope,
            allow_default_store=allow_default_store,
        )

    def search(
        self,
        query: str | None,
        *,
        top_k: int | None = None,
        filters: Optional[Dict[str, Any]] = None,
        query_images: Sequence[str | bytes | bytearray] | None = None,
        scopes: Sequence[str] | str | None = None,
        collection_overrides: Mapping[str, Chroma] | None = None,
    ) -> list[Document]:
        """Perform a retrieval against the configured Chroma instance."""
        scope_list: list[str]
        if isinstance(scopes, str):
            scope_list = [scopes]
        elif isinstance(scopes, Sequence):
            scope_list = [scope for scope in scopes if isinstance(scope, str)]
        else:
            scope_list = []

        scoped_stores = {
            **self.scoped_vector_stores,
            **{str(key): store for key, store in (collection_overrides or {}).items()},
        }

        if len(scope_list) > 1:
            documents: list[Document] = []
            for scope in scope_list:
                store = scoped_stores.get(scope)
                if store is None:
                    continue
                scope_filters = self._prepare_filters(filters, scope)
                documents.extend(
                    self._run_single_search(
                        query,
                        top_k=top_k,
                        filters=scope_filters,
                        query_images=query_images,
                        vector_store=store,
                        scope=scope,
                        allow_default_store=False,
                    )
                )
            return documents

        filter_payload = self._prepare_filters(filters, scopes)
        scoped_store = scoped_stores.get(scope_list[0]) if scope_list else None
        allow_default_store = not bool(scoped_stores)
        return self._run_single_search(
            query,
            top_k=top_k,
            filters=filter_payload,
            query_images=query_images,
            vector_store=scoped_store
            or self._resolve_vector_store(scope_list[0] if scope_list else None),
            scope=scope_list[0] if scope_list else None,
            allow_default_store=allow_default_store,
        )

    def search_chained(
        self,
        query: str | None,
        *,
        retrievers: Sequence["Retriever"] | None = None,
        top_k: int | None = None,
        filters: Optional[Dict[str, Any]] = None,
        query_images: Sequence[str | bytes | bytearray] | None = None,
        scopes: Sequence[str] | str | None = None,
        collection_overrides: Mapping[str, Chroma] | None = None,
    ) -> list[Document]:
        """Perform chained retrieval across multiple retrievers.

        Each subsequent retriever receives a metadata filter containing the UUIDs
        returned by the previous step, mirroring the legacy ``RetrievingPipeline``
        behaviour.
        """
        chain = [self, *(retrievers or [])]
        if not chain:
            return []

        scoped_stores = {
            **self.scoped_vector_stores,
            **{str(key): store for key, store in (collection_overrides or {}).items()},
        }

        scope_list: list[str | None]
        if isinstance(scopes, str):
            scope_list = [scopes]
        elif isinstance(scopes, Sequence):
            scope_list = [scope for scope in scopes if isinstance(scope, str)]
        else:
            scope_list = []

        if scope_list:
            if len(chain) > len(scope_list):
                chain = chain[: len(scope_list)]
            elif len(chain) < len(scope_list):
                scope_list = scope_list[: len(chain)]
        if not scope_list:
            scope_list = [None] * len(chain)

        base_filters = dict(filters or {})
        documents: list[Document] = []
        for index, (retriever, current_scope) in enumerate(zip(chain, scope_list)):
            step_filters = dict(base_filters)
            store_override = scoped_stores.get(current_scope) if current_scope else None
            step_filters = self._prepare_filters(step_filters, current_scope)
            if index > 0 and documents:
                uuid_filter = self._uuid_filter_from_docs(documents)
                if uuid_filter and "uuid" not in step_filters:
                    step_filters.update(uuid_filter)
            allow_default_store = current_scope is None and not scoped_stores
            documents.extend(
                retriever._run_single_search(
                    query,
                    top_k=top_k,
                    filters=step_filters,
                    query_images=query_images if index == 0 else None,
                    vector_store=store_override or retriever._resolve_vector_store(current_scope),
                    scope=current_scope,
                    allow_default_store=allow_default_store,
                )
            )
            del retriever
        return documents