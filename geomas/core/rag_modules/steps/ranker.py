from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Sequence, Tuple, TYPE_CHECKING

from langchain_core.documents import Document
from langchain_core.language_models import LLM
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)


def _load_embedding_function(
    embedding_function_name: str,
    *,
    embedding_model_name: str | None = None,
    embedding_function_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Resolve a Chroma embedding function by name."""

    try:
        from chromadb.utils import embedding_functions as chroma_embeddings
    except ImportError as exc:  # pragma: no cover - depends on optional chromadb
        message = "Chroma dependencies are not installed"
        raise ImportError(message) from exc

    try:
        embedding_cls = getattr(chroma_embeddings, embedding_function_name)
    except AttributeError as exc:
        message = (
            f"Embedding function '{embedding_function_name}' is not available in"
            " chromadb.utils.embedding_functions"
        )
        raise ImportError(message) from exc

    initialisation_kwargs: dict[str, Any] = dict(embedding_function_kwargs or {})
    if embedding_model_name is not None and "model_name" not in initialisation_kwargs:
        initialisation_kwargs["model_name"] = embedding_model_name

    return embedding_cls(**initialisation_kwargs)


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
    """Rerank documents using Chroma-compatible embedding functions."""

    _DEFAULT_FUNCTION = "SentenceTransformerEmbeddingFunction"

    def __init__(
        self,
        *,
        ranking_config: Mapping[str, Any] | None = None,
        embedding_function: Any | None = None,
        fallback_reranker: LengthReranker | None = None,
    ) -> None:
        self._overrides: Mapping[str, Any] = dict(ranking_config or {})
        self._embedding_function_name = self._resolve_embedding_function_name()
        self._embedding_model_name = self._resolve_override("embedding_model_name")
        self._embedding_function_kwargs = self._resolve_embedding_kwargs()
        self._fallback = fallback_reranker or LengthReranker()
        self._embedding_function = embedding_function

        if self._embedding_function is None:
            try:
                self._embedding_function = _load_embedding_function(
                    self._embedding_function_name,
                    embedding_model_name=self._embedding_model_name,
                    embedding_function_kwargs=self._embedding_function_kwargs,
                )
            except Exception as exc:  # pragma: no cover - exercised in tests
                logger.warning(
                    "Unable to initialise embedding function '%s'; falling back to"
                    " length-based reranking",
                    self._embedding_function_name,
                    exc_info=exc,
                )
                self._embedding_function = None

    def rerank(self, query: str, documents: Sequence[Document]) -> list[Document]:
        if not documents:
            return []

        unique_documents = self._deduplicate(documents)
        if not unique_documents:
            return []

        if not self._embedding_function:
            logger.info(
                "Embedding function unavailable; using length-based reranking"
            )
            return self._fallback.rerank(unique_documents)

        try:
            query_vector = self._embed_single(query)
            document_vectors = self._embed_documents(unique_documents)
            scores = [
                self._cosine_similarity(query_vector, vector)
                for vector in document_vectors
            ]
        except Exception as exc:
            logger.warning(
                "Failed to compute embedding similarities; using fallback reranker",
                exc_info=exc,
            )
            return self._fallback.rerank(unique_documents)

        indexed_documents = list(enumerate(unique_documents))
        scored_documents = [
            (index, document, score)
            for (index, document), score in zip(indexed_documents, scores)
        ]
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

    def _resolve_embedding_function_name(self) -> str:
        name = self._resolve_override(
            "embedding_function_name",
            fallback=self._resolve_override("embedding_function", fallback=None),
        )
        if not name:
            return self._DEFAULT_FUNCTION
        return str(name)

    def _resolve_embedding_kwargs(self) -> Mapping[str, Any]:
        overrides = self._resolve_override("embedding_function_kwargs", fallback={})
        if isinstance(overrides, Mapping):
            return dict(overrides)
        logger.warning(
            "Ignoring embedding_function_kwargs override with unexpected type %s",
            type(overrides).__name__,
        )
        return {}

    def _resolve_override(self, key: str, *, fallback: Any | None = None) -> Any:
        alias_keys = {
            "embedding_function_name": "function",
            "embedding_model_name": "model_name",
            "embedding_function_kwargs": "kwargs",
        }
        if key in self._overrides:
            return self._overrides[key]
        alias = alias_keys.get(key)
        if alias and alias in self._overrides:
            return self._overrides[alias]
        return fallback

    def _embed_single(self, text: str) -> list[float]:
        embeddings = self._call_embedding_function([text], single=True)
        return embeddings[0]

    def _embed_documents(self, documents: Sequence[Document]) -> list[list[float]]:
        inputs = [document.page_content for document in documents]
        return self._call_embedding_function(inputs, single=False)

    def _call_embedding_function(
        self, texts: Sequence[str], *, single: bool
    ) -> list[list[float]]:
        if not texts:
            return []

        embedding_function = self._embedding_function
        raw_embeddings: Any

        if single and hasattr(embedding_function, "embed_query") and callable(
            getattr(embedding_function, "embed_query")
        ):
            raw_embeddings = [embedding_function.embed_query(texts[0])]
        elif (
            not single
            and hasattr(embedding_function, "embed_documents")
            and callable(getattr(embedding_function, "embed_documents"))
        ):
            raw_embeddings = embedding_function.embed_documents(list(texts))
        else:
            raw_embeddings = embedding_function(list(texts))

        if not isinstance(raw_embeddings, Sequence):
            raise TypeError("Embedding function returned a non-sequence result")

        embeddings_list = list(raw_embeddings)
        if len(embeddings_list) != len(texts):
            raise ValueError("Embedding output length mismatch")

        processed: list[list[float]] = []
        for vector in embeddings_list:
            if not isinstance(vector, Sequence):
                raise TypeError("Embedding vector is not a sequence")
            processed_vector = []
            for value in vector:
                try:
                    processed_vector.append(float(value))
                except (TypeError, ValueError) as exc:
                    raise TypeError("Embedding value is not a number") from exc
            processed.append(processed_vector)

        return processed

    @staticmethod
    def _cosine_similarity(lhs: Sequence[float], rhs: Sequence[float]) -> float:
        if len(lhs) != len(rhs):
            raise ValueError("Embedding vectors must be of identical dimensions")

        dot = sum(l * r for l, r in zip(lhs, rhs))
        lhs_norm = math.sqrt(sum(value * value for value in lhs))
        rhs_norm = math.sqrt(sum(value * value for value in rhs))
        if lhs_norm == 0.0 or rhs_norm == 0.0:
            return 0.0

        similarity = dot / (lhs_norm * rhs_norm)
        if not math.isfinite(similarity):
            raise ValueError("Cosine similarity computation produced a NaN value")
        return similarity


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
) -> tuple[bool, dict[str, Any]]:
    if ranking_config is None:
        return False, {}

    if hasattr(ranking_config, "chroma"):
        chroma_template = getattr(ranking_config, "chroma")
        enabled = bool(getattr(chroma_template, "enabled", False))
        overrides_method = getattr(chroma_template, "to_overrides", None)
        overrides = (
            overrides_method() if callable(overrides_method) else {}
        )
        return enabled, dict(overrides)

    if isinstance(ranking_config, Mapping):
        overrides: dict[str, Any] = {}
        chroma_section = ranking_config.get("chroma")
        enabled = _ranking_flag(ranking_config, "use_chroma_reranking")

        if isinstance(chroma_section, Mapping):
            if "enabled" in chroma_section:
                enabled = bool(chroma_section.get("enabled"))

            function = (
                chroma_section.get("function")
                or chroma_section.get("embedding_function_name")
                or chroma_section.get("embedding_function")
            )
            if function:
                overrides["embedding_function_name"] = str(function)

            model_name = (
                chroma_section.get("model_name")
                if "model_name" in chroma_section
                else chroma_section.get("embedding_model_name")
            )
            if model_name is not None:
                overrides["embedding_model_name"] = (
                    None if model_name is None else str(model_name)
                )

            kwargs_value = (
                chroma_section.get("kwargs")
                if "kwargs" in chroma_section
                else chroma_section.get("embedding_function_kwargs")
            )
            if isinstance(kwargs_value, Mapping):
                overrides["embedding_function_kwargs"] = dict(kwargs_value)

        return enabled, overrides

    return False, {}


def build_chroma_reranker(
    ranking_config: "RankingConfigTemplate" | Mapping[str, Any] | None,
    *,
    embedding_function: Any | None = None,
    collection_name: str | None = None,
    logger: logging.Logger | None = None,
) -> ChromaReranker | None:
    enabled, overrides = _extract_chroma_settings(ranking_config)
    if not enabled:
        return None

    overrides = dict(overrides)
    if collection_name and "collection_name" not in overrides:
        overrides["collection_name"] = collection_name

    try:
        return ChromaReranker(
            ranking_config=overrides,
            embedding_function=embedding_function,
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
