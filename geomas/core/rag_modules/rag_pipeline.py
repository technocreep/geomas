from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import requests

from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_core.documents import Document

from geomas.core.inference.interface import LlmConnector
from geomas.core.rag_modules.data_adapter import DataLoaderAdapter
from geomas.core.rag_modules.database.chroma_db import (
    ChromaDatabaseClient,
    ChromaDatabaseStore,
    DatabaseRagPipeline,
)
from geomas.core.rag_modules.database.chroma_db import Embeddings, ProcessingResult
from geomas.core.rag_modules.parser.rag_parser import DocumentParser
from geomas.core.rag_modules.steps.ranker import LLMReranker
from geomas.core.repository.promts_repository import PROMPT_RANK
from geomas.core.repository.rag_repository import (
    RAGConfig,
    RAGConfigTemplate,
    InferenceConfigTemplate,
    RankingConfigTemplate,
    RetrievalConfigTemplate,
)

logger = logging.getLogger(__name__)


_SUPPORTED_VECTOR_STORE_TYPE = "chroma"
_SUPPORTED_CLIENT_OPTIONS: frozenset[str] = frozenset(
    {"host", "port", "allow_reset", "mode", "persistent_path"}
)


class BasicRetriever:
    """Perform similarity search over the configured Chroma store."""

    def __init__(self, store: ChromaDatabaseStore) -> None:
        self.store = store

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        """Proxy ``query`` to the underlying :class:`ChromaDatabaseStore`."""

        return self.store.search(query, top_k=top_k, filters=filters)


class LmStudioClient:
    """Thin HTTP client targeting an LM Studio OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("LM Studio base_url must be provided")
        if not model:
            raise ValueError("LM Studio model must be provided")

        self._endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self._model = model
        self._headers = {str(key): str(value) for key, value in dict(headers or {}).items()}
        self._timeout = timeout

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float,
    ) -> str:
        """Send ``messages`` to the LM Studio endpoint and return the response."""

        payload = {
            "model": self._model,
            "messages": [dict(message) for message in messages],
            "temperature": temperature,
        }

        try:
            response = requests.post(
                self._endpoint,
                json=payload,
                headers=self._headers or None,
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to reach LM Studio endpoint: {exc}") from exc

        if response.status_code >= 400:
            snippet = response.text.strip()
            raise RuntimeError(
                "LM Studio request failed with status "
                f"{response.status_code}: {snippet or response.reason}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError("LM Studio response was not valid JSON") from exc

        choices = data.get("choices")
        if not isinstance(choices, Sequence) or not choices:
            raise RuntimeError("LM Studio response did not include any choices")

        first_choice = choices[0]
        if isinstance(first_choice, Mapping):
            message_payload = first_choice.get("message")
            if isinstance(message_payload, Mapping):
                content = message_payload.get("content")
            else:
                content = first_choice.get("text")
        else:
            content = None

        if not content:
            raise RuntimeError("LM Studio response was missing completion content")

        return str(content)


class LengthReranker:
    """Select the longest documents within a retrieval context."""

    def rerank_context(
        self,
        documents: Sequence[Document],
        question: str,
        *,
        top_k: int,
    ) -> list[Document]:
        """Order ``documents`` by length and return the ``top_k`` entries."""

        del question
        ordered = sorted(documents, key=lambda doc: len(doc.page_content), reverse=True)
        return list(ordered[:top_k])


class BaseRAGPipeline(ABC):
    """Common interface for all GeoMAS retrieval pipelines."""

    @abstractmethod
    def ingest_documents(self, documents_path: str, **kwargs: Any) -> bool:
        """Ingest documents into the vector store."""

    @abstractmethod
    def query(self, question: str, **kwargs: Any) -> Dict[str, Any]:
        """Query the pipeline and return structured results."""


class StandardRAGPipeline(BaseRAGPipeline):
    """Reference implementation that wires the legacy database pipeline."""

    def __init__(
        self,
        config: Optional[Mapping[str, Any] | RAGConfig | RAGConfigTemplate] = None,
    ) -> None:
        self.config = RAGConfig.ensure(config)
        self.config_template = self.config.as_template()

        parsing_config = self.config_template.parsing
        chunking_params = parsing_config.chunking_parameters()
        parser_enabled = getattr(parsing_config, "enable_parser", True)
        self.parser: DocumentParser | None
        if parser_enabled:
            self.parser = DocumentParser(
                chunking_params=chunking_params, use_llm=parsing_config.use_llm)
        else:
            self.parser = None

        data_config = self.config_template.data
        self.data_loader = DataLoaderAdapter(
            loader_type=data_config.loader_type,
            parser=self.parser,
            loader_params=dict(data_config.loader_params) or None,
            chunking_params=chunking_params,
        )

        database_config = self.config_template.database
        collection_name = database_config.collection_name or "geomas_text_documents"
        store_config = self.config_template.vector_store
        shared_embedding = self._initialise_embedding(self.config_template.retrieval)
        self.store: ChromaDatabaseStore = self._initialise_store(
            store_config.to_dict(),
            shared_embedding,
            collection_name=collection_name,
        )
        self.embedding_function: Embeddings | None = (
            shared_embedding or getattr(self.store, "embedding", None)
        )

        self.database_pipeline = DatabaseRagPipeline(
            store=self.store,
            parser=self.parser,
            data_loader=self.data_loader,
            default_text_top_k=(
                self.config_template.retrieval.text_top_k
                or self.config_template.retrieval.top_k
            ),
        )
        self.data_loader = self.database_pipeline.data_loader
        self.last_ingest_result: ProcessingResult | None = None

        retrieval_config = self.config_template.retrieval
        self.retriever = BasicRetriever(self.store)
        self.reranker = self._initialise_reranker(self.config_template.ranking)

        (
            self._lm_client,
            self._lm_temperature,
            self._lm_system_prompt,
        ) = self._initialise_inference(self.config_template.inference)

        self._closed = False

    @staticmethod
    def _call_shutdown(target: object | None) -> bool:
        """Attempt to call a shutdown hook on ``target`` when present."""

        if target is None:
            return False

        for attr in ("close", "shutdown", "teardown", "stop", "dispose", "release"):
            method = getattr(target, attr, None)
            if callable(method):
                try:
                    method()
                except Exception as exc:
                    logger.debug("Failed to call %s on %s: %s", attr, type(target).__name__, exc)
                    continue
                return True
        return False

    def _shutdown_embedding(self, embedding: object | None) -> None:
        """Release resources held by ``embedding`` when possible."""

        if embedding is None:
            return

        if self._call_shutdown(embedding):
            return

        for attr in ("client", "model", "_model", "_embeddings_model"):
            candidate = getattr(embedding, attr, None)
            if self._call_shutdown(candidate):
                return

    def _initialise_embedding(
        self, retrieval_config: RetrievalConfigTemplate
    ) -> Embeddings | None:
        model_name = retrieval_config.embedding_model_name
        if not model_name:
            logger.info("Shared embedding initialisation skipped: no model configured")
            return None

        try:
            return SentenceTransformerEmbeddings(model_name=model_name)
        except Exception as exc:
            logger.warning(
                "Failed to load embedding model %s: %s", model_name, exc
            )
            return None

    def _initialise_store(
        self,
        config: Mapping[str, Any] | None,
        embedding: Embeddings | None,
        *,
        collection_name: str,
    ) -> ChromaDatabaseStore:
        """Create a Chroma-backed vector store for ``config``."""

        store_config = dict(config or {})
        store_type = str(store_config.get("type", _SUPPORTED_VECTOR_STORE_TYPE)).lower()
        if store_type != _SUPPORTED_VECTOR_STORE_TYPE:
            raise ValueError(
                f"Unsupported vector store type '{store_type}'. Only "
                f"'{_SUPPORTED_VECTOR_STORE_TYPE}' is supported."
            )

        client_config = store_config.get("client")
        client_kwargs: dict[str, Any] = {}
        if isinstance(client_config, Mapping):
            for key, value in client_config.items():
                if key in _SUPPORTED_CLIENT_OPTIONS:
                    client_kwargs[key] = value

        client = ChromaDatabaseClient(**client_kwargs)
        return ChromaDatabaseStore(
            client=client,
            collection_name=collection_name,
            embedding=embedding,
        )

    def _initialise_reranker(
        self, ranking_config: RankingConfigTemplate
    ) -> Optional[LLMReranker]:
        if not ranking_config.use_llm_reranking:
            return None

        llm_url = ranking_config.llm_url
        if not llm_url:
            logger.warning("LLM reranking requested but no URL provided; skipping reranker")
            return None

        try:
            connector = LlmConnector(llm_url, ranking_config.inference_config)
            return LLMReranker(connector, PROMPT_RANK)
        except Exception as exc:
            logger.warning("Failed to initialise LLM reranker: %s", exc)
            return None

    def _initialise_inference(
        self, inference_config: InferenceConfigTemplate
    ) -> tuple[LmStudioClient | None, float, str | None]:
        """Initialise LM Studio inference when configuration is available."""

        enabled = getattr(inference_config, "enable_remote_services", True)
        params = getattr(inference_config, "params", {})
        if not enabled or not isinstance(params, Mapping):
            return None, 0.0, None

        base_url = params.get("base_url")
        model = params.get("model")
        if not base_url or not model:
            logger.info("LM Studio inference skipped: base_url or model missing")
            return None, 0.0, None

        headers = params.get("headers") if isinstance(params.get("headers"), Mapping) else None
        timeout_value: float | None = None
        raw_timeout = params.get("timeout")
        if raw_timeout is not None:
            try:
                timeout_value = float(raw_timeout)
            except (TypeError, ValueError):
                logger.warning("Invalid LM Studio timeout value '%s'; ignoring", raw_timeout)

        raw_temperature = params.get("temperature", 0.0)
        try:
            temperature = float(raw_temperature)
        except (TypeError, ValueError):
            logger.warning("Invalid LM Studio temperature '%s'; defaulting to 0.0", raw_temperature)
            temperature = 0.0

        system_prompt = params.get("system_prompt")
        if system_prompt is not None:
            system_prompt = str(system_prompt)

        try:
            client = LmStudioClient(
                base_url=str(base_url),
                model=str(model),
                headers={str(k): str(v) for k, v in dict(headers or {}).items()},
                timeout=timeout_value,
            )
        except Exception as exc:
            logger.warning("Failed to initialise LM Studio client: %s", exc)
            return None, temperature, system_prompt

        return client, temperature, system_prompt

    def ingest_documents(self, documents_path: str, **kwargs: Any) -> bool:
        try:
            result = self.database_pipeline.process(
                documents_path,
                document_name=kwargs.get("document_name"),
            )
        except FileNotFoundError:
            raise
        except Exception as exc:
            logger.exception("Database ingestion failed for '%s': %s", documents_path, exc)
            result = ProcessingResult(success=False)

        self.last_ingest_result = result
        if result.success:
            logger.info("Ingested %s documents", result.documents_ingested)
        return result.success

    def query(self, question: str, **kwargs: Any) -> Dict[str, Any]:
        retrieval_config = self.config_template.retrieval

        def _to_positive_int(value: Any, default: int) -> int:
            try:
                candidate = int(value)
            except (TypeError, ValueError):
                return max(1, default)
            return max(1, candidate)

        base_filters: dict[str, object] = {}
        if isinstance(retrieval_config.filters, Mapping):
            base_filters.update(retrieval_config.filters)
        runtime_filters = kwargs.get("filters")
        if isinstance(runtime_filters, Mapping):
            base_filters.update(runtime_filters)
        filters = base_filters or None

        default_top_k = max(1, int(getattr(retrieval_config, "top_k", 5) or 5))
        candidate_top_k = _to_positive_int(kwargs.get("top_k"), default_top_k)
        configured_text_top_k = getattr(retrieval_config, "text_top_k", None)
        text_top_k = _to_positive_int(
            kwargs.get("text_top_k"),
            configured_text_top_k if configured_text_top_k else candidate_top_k,
        )
        configured_final_top_k = getattr(retrieval_config, "final_top_k", None)
        rerank_limit = _to_positive_int(
            kwargs.get("rerank_top_k") or kwargs.get("final_top_k"),
            configured_final_top_k if configured_final_top_k else text_top_k,
        )

        search_limit = max(candidate_top_k, text_top_k, rerank_limit)
        raw_results = self.retriever.search(
            question,
            top_k=search_limit,
            filters=filters,
        )

        text_context = DatabaseRagPipeline._build_scored_context(raw_results, text_top_k)
        if self.reranker and text_context:
            documents = [
                Document(page_content=doc_text, metadata=metadata)
                for _, doc_text, metadata, _ in text_context
            ]
            reranked_documents = self.reranker.rerank_context(
                documents,
                question,
                top_k=min(rerank_limit, len(documents)),
            )
            ordered_context: list[tuple[str, str, dict, float]] = []
            for reranked_doc in reranked_documents:
                for doc_id, original_text, original_metadata, score in text_context:
                    if reranked_doc.page_content == original_text:
                        ordered_context.append((doc_id, original_text, original_metadata, score))
                        break
            if ordered_context:
                text_context = ordered_context

        relevant_sources = self._collect_sources(text_context, candidate_top_k)
        answer = self._generate_answer(question, text_context)

        return {
            "question": question,
            "relevant_papers": relevant_sources,
            "text_context": text_context,
            "answer": answer,
        }

    @staticmethod
    def _collect_sources(
        text_context: Sequence[tuple[str, str, Mapping[str, object] | dict, float]],
        limit: int,
    ) -> list[str]:
        sources: list[str] = []
        seen: set[str] = set()
        for _, _, metadata, _ in text_context:
            source = None
            if isinstance(metadata, Mapping):
                source = metadata.get("source")
            if isinstance(source, str) and source not in seen:
                seen.add(source)
                sources.append(source)
                if len(sources) >= limit:
                    break
        return sources

    def _generate_answer(
        self, question: str, text_context: list[tuple[str, str, dict, float]]
    ) -> str | None:
        client = getattr(self, "_lm_client", None)
        if client is None:
            return None

        context_block = self._format_context(text_context)
        if not context_block:
            logger.info("LM Studio generation skipped: no retrieval context available")
            return None

        messages: list[dict[str, str]] = []
        if self._lm_system_prompt:
            messages.append({"role": "system", "content": self._lm_system_prompt})

        prompt = (
            "Use the provided context to answer the user's question. "
            "If the context lacks the answer, reply that the information is unavailable.\n\n"
            f"Context:\n{context_block}\n\nQuestion: {question}\nAnswer:"
        )
        messages.append({"role": "user", "content": prompt})

        try:
            return client.generate(messages, temperature=self._lm_temperature)
        except Exception as exc:
            logger.warning("LM Studio generation failed: %s", exc)
            return None

    @staticmethod
    def _format_context(text_context: list[tuple[str, str, dict, float]]) -> str:
        formatted: list[str] = []
        for index, (_, content, metadata, score) in enumerate(text_context, start=1):
            title = (
                metadata.get("document_name")
                if isinstance(metadata, Mapping)
                else None
            )
            if isinstance(metadata, Mapping):
                source = metadata.get("source")
            else:
                source = None
            header_parts = [f"[{index}]"]
            if title:
                header_parts.append(str(title))
            elif source:
                header_parts.append(str(source))
            if not (score != score):  # NaN check
                header_parts.append(f"(score={score:.4f})")
            header = " ".join(part for part in header_parts if part)
            formatted.append(f"{header}\n{str(content).strip()}")
        return "\n\n".join(formatted)

    def close(self) -> None:
        """Release resources held by the pipeline components."""

        if getattr(self, "_closed", False):
            return

        self._closed = True

        if hasattr(self, "store") and self.store is not None:
            try:
                self.store.close()
            except Exception as exc:
                logger.debug("Failed to close database store: %s", exc)
            finally:
                self.store = None

        if getattr(self, "retriever", None) is not None:
            self.retriever = None

        if getattr(self, "embedding_function", None) is not None:
            self._shutdown_embedding(self.embedding_function)
            self.embedding_function = None

        if getattr(self, "reranker", None) is not None:
            reranker = self.reranker
            self._call_shutdown(getattr(reranker, "_llm", None))
            self._call_shutdown(reranker)
            self.reranker = None

        if getattr(self, "database_pipeline", None) is not None:
            self.database_pipeline = None

        self.data_loader = None
        self._lm_client = None


def create_standard_pipeline(
    config: Mapping[str, Any] | RAGConfig | RAGConfigTemplate | None,
    *,
    attach_reranker: bool = True,
) -> StandardRAGPipeline:
    """Build a :class:`StandardRAGPipeline` with normalised configuration."""

    resolved_config = RAGConfig.ensure(config)
    pipeline = StandardRAGPipeline(resolved_config)
    if attach_reranker:
        pipeline.reranker = LengthReranker()
    return pipeline


def ingest_documents(
    pipeline: StandardRAGPipeline,
    documents_dir: Path | str,
    *,
    document_name: str | None = None,
) -> ProcessingResult:
    """Ingest ``documents_dir`` and surface failures via ``RuntimeError``."""

    success = pipeline.ingest_documents(
        str(documents_dir),
        document_name=document_name,
    )
    result = pipeline.last_ingest_result or ProcessingResult(success=bool(success))
    return result

