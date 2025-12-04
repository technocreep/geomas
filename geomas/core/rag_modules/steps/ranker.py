from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Sequence, Tuple

from langchain_chroma import Chroma
from langchain_chroma.vectorstores import cosine_similarity
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import LLM
from langchain_core.prompts import PromptTemplate

from geomas.core.rag_modules.steps.retriever import _normalise_relevance

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LengthReranker:
    """Reranks documents by descending character length."""
    def rerank(self, documents: Sequence[Document]) -> list[Document]:
        if not documents:
            return []

        enumerated = list(enumerate(documents))
        enumerated.sort(
            key=lambda item: (-len(item[1].page_content or ""), item[0])
        )
        return [document for _, document in enumerated]


class ChromaReranker:
    """Rerank documents using the shared multimodal embedding function."""
    def __init__(
        self,
        *,
        ranking_config: Mapping[str, Any] | None = None,
        embedding_function: Embeddings | None = None,
        vector_store: Chroma | None = None,
        fallback_reranker: LengthReranker | None = None,
        collection_vector_stores: Mapping[str, Chroma] | None = None,
        collection_embeddings: Mapping[str, Embeddings] | None = None,
    ) -> None:
        self._embedding_function = embedding_function or (
            vector_store.embeddings if vector_store is not None else None
        )
        self._fallback = fallback_reranker or LengthReranker()
        self._vector_store = vector_store
        self._collection_embeddings = {
            str(key): value
            for key, value in (collection_embeddings or {}).items()
            if isinstance(key, str)
        }
        self._collection_vector_stores = {
            str(key): store
            for key, store in (collection_vector_stores or {}).items()
            if isinstance(key, str)
        }

    def rerank(
        self, 
        query: str, 
        documents: Sequence[Document],
        score_threshold: float
    ) -> list[Document]:
        if not documents:
            return []

        unique_documents = self._deduplicate(documents)
        if not unique_documents:
            return []

        embedding_map = self._build_embedding_map(unique_documents)
        if not embedding_map:
            logger.info(
                "Embedding function unavailable; using length-based reranking"
            )
            return self._fallback.rerank(unique_documents)

        scored_documents: list[tuple[int, Document, float]] = []
        for embedding_fn, grouped in embedding_map.values():
            try:
                query_vector = self._embed_query_with_fn(embedding_fn, query)
                if not query_vector:
                    scored_documents.extend(
                        (index, document, float("-inf"))
                        for index, document in grouped
                    )
                    continue

                vectorised_documents: list[tuple[int, Document, list[float]]] = []
                for index, document in grouped:
                    vector = self._embed_document_with_fn(document, embedding_fn)
                    if vector:
                        vectorised_documents.append((index, document, vector))
                    else:
                        scored_documents.append((index, document, float("-inf")))
                if not vectorised_documents:
                    continue

                similarity = cosine_similarity(
                    [vector for _, _, vector in vectorised_documents],
                    [query_vector],
                )
                if hasattr(similarity, "__getitem__") and not isinstance(
                    similarity, list
                ):
                    scores = similarity[:, 0].tolist()
                else:
                    try:
                        first_row = similarity[0]
                        scores = list(first_row)
                    except Exception:  # pragma: no cover - defensive fallback
                        scores = list(similarity) if similarity is not None else []
            except Exception as exc:
                logger.warning(
                    "Failed to compute embedding similarities; using fallback reranker",
                    exc_info=exc,
                )
                return self._fallback.rerank(unique_documents)

            for position, (index, document, _) in enumerate(vectorised_documents):
                score = scores[position] if position < len(scores) else float("-inf")
                if score >= score_threshold:
                    scored_documents.append((index, document, float(score)))

        if not scored_documents:
            return self._fallback.rerank(unique_documents)

        scored_documents.sort(key=lambda item: (-item[2], item[0]))
        return [document for _, document, _ in scored_documents]

    def _deduplicate(self, documents: Sequence[Document]) -> list[Document]:
        seen: set[str] = set()
        unique_documents: list[Document] = []

        for document in documents:
            key = self._document_key(document)
            if key in seen:
                continue
            seen.add(key)
            unique_documents.append(document)

        return unique_documents

    def _document_key(self, document: Document) -> str:
        content = document.page_content or ""
        metadata_repr = self._stringify_metadata(document.metadata)
        return f"{content}\n{metadata_repr}"

    def _embed_query(self, query: str) -> list[float]:
        return self._embed_query_with_fn(self._embedding_function, query)

    def _embed_query_with_fn(
        self, embedding_function: Embeddings | None, query: str
    ) -> list[float]:
        if embedding_function is None:
            return []

        if hasattr(embedding_function, "embed_query"):
            return list(embedding_function.embed_query(query))

        if hasattr(embedding_function, "embed_documents"):
            embeddings = embedding_function.embed_documents([query])
            return list(embeddings[0]) if embeddings else []

        embeddings = embedding_function([query])
        return list(embeddings[0]) if embeddings else []

    def _embed_document(self, document: Document) -> list[float]:
        return self._embed_document_with_fn(document, self._embedding_function)

    def _embed_document_with_fn(
        self, document: Document, embedding_function: Embeddings | None
    ) -> list[float]:
        if embedding_function is None:
            return []

        metadata = document.metadata if isinstance(document.metadata, Mapping) else {}
        is_image = metadata.get("type") == "image"
        source_path = metadata.get("source_path") or metadata.get("source")

        if is_image and hasattr(embedding_function, "embed_image") and isinstance(
            source_path, str
        ):
            embeddings = embedding_function.embed_image([source_path])
            return list(embeddings[0]) if embeddings else []

        if hasattr(embedding_function, "embed_documents"):
            embeddings = embedding_function.embed_documents([document.page_content])
            return list(embeddings[0]) if embeddings else []

        if hasattr(embedding_function, "embed_query"):
            return list(embedding_function.embed_query(document.page_content))

        embeddings = embedding_function([document.page_content])
        return list(embeddings[0]) if embeddings else []

    def _embedding_for_document(self, document: Document) -> Embeddings | None:
        metadata = document.metadata if isinstance(document.metadata, Mapping) else {}
        scope = metadata.get("scope") or metadata.get("namespace")
        if isinstance(scope, str):
            if scope in self._collection_embeddings:
                return self._collection_embeddings[scope]
            scoped_store = self._collection_vector_stores.get(scope)
            if scoped_store is not None and scoped_store.embeddings is not None:
                return scoped_store.embeddings
        return self._embedding_function

    def _build_embedding_map(
        self, documents: Sequence[Document]
    ) -> dict[int, tuple[Embeddings, list[tuple[int, Document]]]]:
        embedding_map: dict[int, tuple[Embeddings, list[tuple[int, Document]]]] = {}
        for index, document in enumerate(documents):
            embedding_fn = self._embedding_for_document(document)
            if embedding_fn is None:
                continue
            key = id(embedding_fn)
            if key not in embedding_map:
                embedding_map[key] = (embedding_fn, [])
            embedding_map[key][1].append((index, document))
        return embedding_map

    @classmethod
    def _stringify_metadata(cls, metadata: Any) -> str:
        if metadata is None:
            return ""

        try:
            normalised = cls._normalise_metadata(metadata)
        except Exception:  # pragma: no cover - defensive guard
            return repr(metadata)
        return repr(normalised)

    @classmethod
    def _normalise_metadata(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): cls._normalise_metadata(item)
                for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
            }
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [cls._normalise_metadata(item) for item in value]
        return value


def _score_from_metadata(metadata: Mapping[str, Any]) -> float:
    for key in ("relevance_score", "normalized_score", "score", "similarity"):
        if key in metadata:
            try:
                return float(metadata[key])
            except (TypeError, ValueError):
                continue

    distance = metadata.get("distance")
    normalised = _normalise_relevance(distance) if distance is not None else None
    if normalised is not None:
        return normalised

    return float("nan")


def _extract_chunk_index(metadata: Mapping[str, Any] | None, doc_id: str | None) -> int | None:
    candidates: list[object] = []
    if isinstance(metadata, Mapping):
        for key in ("chunk_index", "chunkId", "chunkNumber"):
            if key in metadata:
                candidates.append(metadata[key])
    if doc_id:
        matches = re.findall(r"chunk_(\d+)", str(doc_id))
        if matches:
            candidates.append(matches[-1])

    for candidate in candidates:
        try:
            return int(candidate)
        except (TypeError, ValueError):
            continue
    return None


def build_scored_context(
    raw_results: Sequence[Document] | Sequence[tuple[Document, float]] | Mapping[str, Any],
    top_k: int,
) -> list[tuple[str, int | None, str, dict, float]]:
    """Normalise retrieval results into a scored context list."""

    if isinstance(raw_results, Mapping):
        documents = raw_results.get("documents", [[]])
        metadatas = raw_results.get("metadatas", [[]])
        ids = raw_results.get("ids", [[]])

        candidate_docs = documents[0] if isinstance(documents, Sequence) and documents else []
        candidate_metas = metadatas[0] if isinstance(metadatas, Sequence) and metadatas else []
        candidate_ids = ids[0] if isinstance(ids, Sequence) and ids else []

        limit = min(top_k, len(candidate_docs), len(candidate_metas), len(candidate_ids))
        if limit <= 0:
            return []

        scores_row = raw_results.get("similarities") or raw_results.get("distances") or []
        score_candidates = scores_row[0] if isinstance(scores_row, Sequence) and scores_row else []

        scored_docs: list[tuple[str, int | None, str, dict, float]] = []
        for index in range(limit):
            doc_id = str(candidate_ids[index])
            doc_text = candidate_docs[index]
            metadata = (
                candidate_metas[index] if isinstance(candidate_metas[index], Mapping) else {}
            )
            score_candidate = score_candidates[index] if index < len(score_candidates) else None
            normalised = (
                _normalise_relevance(score_candidate) if score_candidate is not None else None
            )
            if normalised is not None:
                metadata.setdefault("relevance_score", normalised)
                metadata.setdefault("normalized_score", normalised)
            score = normalised if normalised is not None else float("nan")
            scored_docs.append(
                (
                    doc_id,
                    _extract_chunk_index(metadata, doc_id),
                    doc_text,
                    dict(metadata),
                    score,
                )
            )
        return scored_docs

    scored_docs: list[tuple[str, int | None, str, dict, float]] = []
    if not isinstance(raw_results, Sequence):
        return scored_docs

    if raw_results and isinstance(raw_results[0], tuple):
        for document, score in list(raw_results)[:top_k]:
            metadata = dict(document.metadata or {})
            normalised = _normalise_relevance(score)
            if normalised is not None:
                metadata.setdefault("relevance_score", normalised)
                metadata.setdefault("normalized_score", normalised)
            scored_docs.append(
                (
                    document.id or "",
                    _extract_chunk_index(document.metadata, document.id),
                    document.page_content,
                    metadata,
                    normalised if normalised is not None else float("nan"),
                )
            )
        return scored_docs

    for document in list(raw_results)[:top_k]:
        metadata = dict(document.metadata or {})
        score = _score_from_metadata(metadata)
        scored_docs.append(
            (
                document.id or "",
                _extract_chunk_index(metadata, document.id),
                document.page_content,
                metadata,
                score,
            )
        )
    return scored_docs


def _ranking_flag(
    ranking_config: "RankingConfigTemplate" | Mapping[str, Any] | None,
    key: str,
) -> bool:
    if ranking_config is None:
        return False
    if hasattr(ranking_config, key):
        return bool(getattr(ranking_config, key))
    if isinstance(ranking_config, Mapping):
        return bool(ranking_config.get(key))
    return False


def _ranking_value(
    ranking_config: "RankingConfigTemplate" | Mapping[str, Any] | None,
    key: str,
) -> Any:
    if ranking_config is None:
        return None
    if hasattr(ranking_config, key):
        return getattr(ranking_config, key)
    if isinstance(ranking_config, Mapping):
        return ranking_config.get(key)
    return None


def _ranking_mapping(
    ranking_config: "RankingConfigTemplate" | Mapping[str, Any] | None,
    key: str,
) -> Mapping[str, Any]:
    candidate = _ranking_value(ranking_config, key)
    if isinstance(candidate, Mapping):
        return dict(candidate)
    return {}


def _extract_chroma_settings(
    ranking_config: "RankingConfigTemplate" | Mapping[str, Any] | None,
) -> bool:
    if ranking_config is None:
        return False

    if hasattr(ranking_config, "chroma"):
        chroma_template = getattr(ranking_config, "chroma")
        return bool(getattr(chroma_template, "enabled", False))

    if isinstance(ranking_config, Mapping):
        chroma_section = ranking_config.get("chroma")
        enabled = _ranking_flag(ranking_config, "use_chroma_reranking")

        if isinstance(chroma_section, Mapping):
            if "enabled" in chroma_section:
                enabled = bool(chroma_section.get("enabled"))

        return enabled
    return False


def build_chroma_reranker(
    ranking_config: "RankingConfigTemplate" | Mapping[str, Any] | None,
    *,
    embedding_function: Embeddings | None = None,
    vector_store: Chroma | None = None,
    collection_name: str | None = None,
    collection_vector_stores: Mapping[str, Chroma] | None = None,
    collection_embeddings: Mapping[str, Embeddings] | None = None,
    logger: logging.Logger | None = None,
) -> ChromaReranker | None:
    enabled = _extract_chroma_settings(ranking_config)
    if not enabled:
        return None
    try:
        return ChromaReranker(
            ranking_config=ranking_config,
            embedding_function=embedding_function,
            vector_store=vector_store,
            collection_vector_stores=collection_vector_stores,
            collection_embeddings=collection_embeddings,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        if logger is not None:
            logger.warning("Failed to initialise Chroma reranker: %s", exc)
        return None


def build_llm_reranker(
    ranking_config: "RankingConfigTemplate" | Mapping[str, Any] | None,
    *,
    connector_factory: Callable[[str, Mapping[str, Any]], Any],
    reranker_factory: Callable[[Any, Any], Any],
    prompt_template: Any,
    logger: logging.Logger | None = None,
) -> Any | None:
    if not _ranking_flag(ranking_config, "use_llm_reranking"):
        return None

    llm_url = _ranking_value(ranking_config, "llm_url")
    if not llm_url:
        if logger is not None:
            logger.warning(
                "LLM reranking requested but no URL provided; skipping reranker"
            )
        return None

    inference_config = _ranking_mapping(ranking_config, "inference_config")

    try:
        connector = connector_factory(str(llm_url), dict(inference_config))
        return reranker_factory(connector, prompt_template)
    except Exception as exc:  # pragma: no cover - defensive logging
        if logger is not None:
            logger.warning("Failed to initialise LLM reranker: %s", exc)
        return None


class LLMReranker:
    def __init__(self, llm: LLM, prompt_template: PromptTemplate):
        """
        Reranker to change the order of documents using LLM.

        :param prompt_template: prompt template for reranking. It should contain the 'question' and 'context' fields
        """
        self._prompt_template = prompt_template
        # The retries number if the first model response was obtained in wrong format or have another anomalies
        # that are not characteristic for correct operation of the model in accordance with the prompt template
        self.num_retries = 3
        # The lower boundary of context LLM estimation
        self.qual_threshold = 2
        self._llm = llm

    def rerank_context(self, context: list[Document], user_query: str, top_k: int = 3) -> list[Document]:
        ranking_prompts = [self._prompt_template.format(question=user_query,
                                                        context=context_i.page_content +
                                                                " Имя файла, откуда взят параграф " +
                                                                context_i.metadata.get('source',
                                                                                       '/None').split('/')[-1])
                           for context_i in context]
        answers_ranking, bad_query = self._get_ranking_answer(ranking_prompts)
        if bad_query:
            fixed_answers = self._regenerate_answer(bad_query)
            answers_ranking += fixed_answers
        ext_context = self._extract_top_context(answers_ranking, top_k)
        if not ext_context:
            warnings.warn('Reranker does not support retrieved context')
        res_context = [context[ranking_prompts.index(i)] for i in ext_context]
        return res_context

    def _extract_top_context(self, pairs_to_rank: List[Tuple[str, int]], top_k: int) -> list[str]:
        if not pairs_to_rank:
            return []
        pairs_to_rank.sort(key=lambda x: x[1], reverse=True)
        context = [x for x, y in pairs_to_rank if y >= self.qual_threshold]
        context = context[:top_k]
        return context

    def _get_ranking_answer(self, ranking_prompts: list[str]) -> Tuple[list[Tuple[str, int]], list[str]]:
        answer = [self._llm.invoke(prompt) for prompt in ranking_prompts]
        answers_ranking = []
        bad_queries = []
        for i, ans_i in enumerate(answer):
            try:
                score = int(ans_i.split('ОЦЕНКА: ')[-1].strip())
                answers_ranking.append((ranking_prompts[i], score))
            except:
                bad_queries.append(ranking_prompts[i])
        return answers_ranking, bad_queries

    def _regenerate_answer(self, queries: list[str]) -> list[str]:
        fixed_queries = []
        for i in range(self.num_retries):
            good_res, bad_res = self._get_ranking_answer(queries)
            fixed_queries += good_res
            queries = bad_res
            if not bad_res:
                return fixed_queries
        return fixed_queries

    def merge_docs(self, query: str, contexts: list[list[Document]], top_k: int = 3) -> list[Document]:
        ctx = []
        for context in zip(*contexts):
            ctx.extend(self.rerank_context(context, query, 1))

        if len(ctx) > top_k:
            return self.rerank_context(ctx, query, top_k)

        return ctx
