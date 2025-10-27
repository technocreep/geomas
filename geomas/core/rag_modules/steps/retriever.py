import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from chromadb import ClientAPI
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_core.documents import Document

from geomas.core.rag_modules.database.chroma_db import ChromaDatabaseStore


def _is_nonstring_sequence(value: object) -> bool:
    """Return ``True`` when ``value`` behaves like a sequence of results."""

    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _extract_ids_row(payload: Mapping[str, object]) -> Sequence[object] | None:
    """Return the first ids row when the response payload matches expectations."""

    ids_field = payload.get("ids")
    if not _is_nonstring_sequence(ids_field):
        return None
    if len(ids_field) != 1:
        return None

    first_row = ids_field[0]
    if not _is_nonstring_sequence(first_row):
        return None
    return first_row


def _unique_indices(ids_row: Sequence[object], limit: int) -> list[int]:
    """Return indices representing the first unique ids up to ``limit``."""

    if limit <= 0:
        return []

    seen: set[str] = set()
    indices: list[int] = []
    for position, chunk_id in enumerate(ids_row):
        key = str(chunk_id)
        if key in seen:
            continue
        seen.add(key)
        indices.append(position)
        if len(indices) >= limit:
            break
    return indices


def _coerce_row(row: Sequence[object], indices: Sequence[int]) -> Sequence[object]:
    """Project ``row`` onto ``indices`` while preserving its container type."""

    selected: list[object] = []
    for index in indices:
        if index < len(row):
            selected.append(row[index])

    if isinstance(row, list):
        return selected
    if isinstance(row, tuple):
        return tuple(selected)

    try:
        return type(row)(selected)  # type: ignore[call-arg]
    except Exception:
        return selected


def _project_payload(
    payload: Mapping[str, object],
    *,
    indices: Sequence[int],
) -> Mapping[str, object]:
    """Return a payload projected to ``indices`` when deduplication is required."""

    # Preserve the original container type when the payload is a mutable mapping.
    try:
        projected_payload: Dict[str, object] = dict(payload)
    except Exception:
        projected_payload = {key: value for key, value in payload.items()}

    for key, value in payload.items():
        if not _is_nonstring_sequence(value) or not value:
            continue

        rebuilt_rows: list[Sequence[object]] = []
        aborted = False
        for row in value:
            if not _is_nonstring_sequence(row):
                aborted = True
                break
            rebuilt_rows.append(_coerce_row(row, indices))

        if aborted:
            continue

        try:
            projected_payload[key] = type(value)(rebuilt_rows)  # type: ignore[call-arg]
        except Exception:
            projected_payload[key] = rebuilt_rows

    return projected_payload


def _deduplicate_payload(payload: Mapping[str, object], top_k: int) -> Mapping[str, object]:
    """Remove duplicate ids from ``payload`` while respecting ``top_k``."""

    ids_row = _extract_ids_row(payload)
    if ids_row is None:
        return payload

    limit = max(0, int(top_k))
    indices = _unique_indices(ids_row, limit)

    if not indices and (limit != 0 or not ids_row):
        return payload

    needs_update = bool(ids_row) if limit == 0 else len(indices) != len(ids_row)
    if not needs_update:
        return payload

    return _project_payload(payload, indices=indices)


@dataclass
class DocsSearcherModels:
    embedding_model: SentenceTransformerEmbeddings = None
    chroma_client: ClientAPI = None


class Retriever:
    def __init__(
        self,
        top_k: int,
        docs_searcher_models: DocsSearcherModels,
        preprocess_query: Optional[Callable[[str], str]] = None,
    ):
        """
        :param preprocess_query: for example, get keywords from query
        """
        self.top_k = top_k
        self.embedding_function = docs_searcher_models.embedding_model
        self.client = docs_searcher_models.chroma_client
        self.preprocess_query = preprocess_query

    def retrieve_top(
        self,
        collection_name: str,
        query: str,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Optional[list[Document]]:
        """
        Retrieve top K documents from Vector DB (e.x., Chroma).
        """
        if filter is None:
            _filter: Dict[str, Any] = {}
        else:
            _filter = filter.copy()
        if collection_name is None:
            warnings.warn("Collection name is None")
            return None
        if self.preprocess_query is not None:
            query = self.preprocess_query(query)
        if collection_name not in [col.name for col in self.client.list_collections()]:
            warnings.warn("There is no collection named {} in Chroma DB".format(collection_name))
            return None

        store = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embedding_function,
        )
        return store.as_retriever(search_kwargs={"k": self.top_k, "filter": _filter}).invoke(query)


DocRetriever = Retriever


class BasicRetriever:
    """Perform similarity search over the configured :class:`ChromaDatabaseStore`."""

    def __init__(self, store: ChromaDatabaseStore) -> None:
        self.store = store

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        """Proxy ``query`` to the underlying store and deduplicate the results."""

        payload = self.store.search(query, top_k=top_k, filters=filters)
        return _deduplicate_payload(payload, top_k)


class RetrievingPipeline:
    def __init__(self):
        self._retrievers: Optional[list[DocRetriever]] = None
        self._collection_names: Optional[list[str]] = None

    def set_retrievers(self, retrievers: list[DocRetriever]) -> "RetrievingPipeline":
        self._retrievers = retrievers
        return self

    def set_collection_names(self, collection_names: list[str]) -> "RetrievingPipeline":
        self._collection_names = collection_names
        return self

    def get_retrieved_docs(self, query: str) -> list[Document]:
        if any([self._retrievers is None, self._collection_names is None]):
            raise ValueError("Either retrievers or collection_names must not be None")

        if len(self._retrievers) == len(self._collection_names):
            _query = query
            docs = self._retrievers[0].retrieve_top(self._collection_names[0], _query)
            for i in range(1, len(self._retrievers)):
                filter = {"uuid": {"$in": [doc.metadata["uuid"] for doc in docs]}}
                docs_next = self._retrievers[i].retrieve_top(self._collection_names[i], _query, filter)
                docs = docs_next
        else:
            raise Exception("The length of retrievers and collection_names must match")

        return docs
