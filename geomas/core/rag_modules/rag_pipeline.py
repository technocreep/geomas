from __future__ import annotations

import importlib.util
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_core.documents import Document

from geomas.core.inference.lmstudio_client import LmStudioClient
from geomas.core.inference.ollama_client import OllamaClient
from geomas.core.rag_modules.data_adapter import DataLoaderAdapter
from geomas.core.rag_modules.database.chroma_db import (
    ChromaDatabaseClient,
    ChromaDatabaseStore,
    DatabaseRagPipeline,
)
from geomas.core.rag_modules.database.chroma_db import Embeddings, ProcessingResult
from geomas.core.rag_modules.parser.rag_parser import DocumentParser
from geomas.core.rag_modules.steps.ranker import (
    LLMReranker,
    build_chroma_reranker,
    build_llm_reranker,
)
from geomas.core.rag_modules.steps.retriever import BasicRetriever
from geomas.core.repository.promts_repository import PROMPT_RANK
from geomas.core.repository.rag_repository import (
    RAGConfig,
    RAGConfigTemplate,
    InferenceConfigTemplate,
    IntegrationsConfigTemplate,
    RankingConfigTemplate,
    RetrievalConfigTemplate,
)

logger = logging.getLogger(__name__)


_SUPPORTED_VECTOR_STORE_TYPE = "chroma"
_SUPPORTED_CLIENT_OPTIONS: frozenset[str] = frozenset(
    {"host", "port", "allow_reset", "mode", "persistent_path"}
)


def _create_llm_connector(
    url: str, model_params: Mapping[str, Any] | None
) -> "_LlmConnector":
    """Create the LLM connector for reranking once dependencies are present."""

    if importlib.util.find_spec("unsloth") is None:
        raise RuntimeError(
            "LLM reranker requires the optional 'unsloth' dependency to be installed"
        )

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("LLM reranker requires the 'torch' package") from exc

    cuda = getattr(torch, "cuda", None)
    try:
        is_available_callable = getattr(cuda, "is_available", None)
        cuda_available = bool(
            cuda and callable(is_available_callable) and is_available_callable()
        )
    except Exception as exc:
        raise RuntimeError("Failed to determine CUDA availability for LLM reranker") from exc

    if not cuda_available:
        raise RuntimeError(
            "LLM reranker requires a CUDA-enabled torch installation"
        )

    from geomas.core.inference.interface import LlmConnector as RuntimeLlmConnector

    params = dict(model_params or {})
    return RuntimeLlmConnector(url, params)


class BaseRAGPipeline(ABC):
    """Common interface for all GeoMAS retrieval pipelines."""

    @abstractmethod
    def ingest_documents(self, documents_path: str, **kwargs: Any) -> bool:
        """Ingest documents into the vector store."""

    @abstractmethod
    def query(self, question: str, **kwargs: Any) -> Dict[str, Any]:
        """Query the pipeline and return structured results."""


class StandardRAGPipeline(BaseRAGPipeline):
    """Reference implementation that wires the legacy database pipeline.

    The pipeline optionally attaches an :class:`LLMReranker` when reranking is
    enabled in the configuration; no alternative rerankers are constructed here.
    """

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
        self.chroma_reranker = build_chroma_reranker(
            self.config_template.ranking,
            embedding_function=self.embedding_function,
            collection_name=getattr(self.store, "collection_name", None),
            logger=logger,
        )
        self.reranker: LLMReranker | None = self._initialise_reranker(
            self.config_template.ranking
        )

        (
            self._lm_client,
            self._lm_temperature,
            self._lm_system_prompt,
            self._lm_provider_label,
        ) = self._initialise_inference(
            self.config_template.inference,
            self.config_template.integrations,
        )

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
        """Initialise the optional LLM reranker when dependencies are available.

        The connector factory defers importing the Unsloth-backed implementation
        until after lightweight dependency checks succeed. This keeps
        ``StandardRAGPipeline`` usable in environments where the optional stack
        is not installed while still surfacing a clear warning when reranking
        cannot be attached.
        """

        return build_llm_reranker(
            ranking_config,
            connector_factory=_create_llm_connector,
            reranker_factory=lambda connector, prompt: LLMReranker(connector, prompt),
            prompt_template=PROMPT_RANK,
            logger=logger,
        )

    def _initialise_inference(
        self,
        inference_config: InferenceConfigTemplate,
        integrations_config: IntegrationsConfigTemplate,
    ) -> tuple[LmStudioClient | OllamaClient | None, float, str | None, str | None]:
        """Initialise chat inference according to the configured provider."""

        enabled = getattr(inference_config, "enable_remote_services", True)
        if not enabled:
            return None, 0.0, None, None

        params = getattr(inference_config, "params", {})
        params_map: Mapping[str, Any]
        if isinstance(params, Mapping):
            params_map = params
        else:
            params_map = {}

        def _normalise_provider(value: object | None) -> str | None:
            if isinstance(value, str):
                candidate = value.strip().lower()
                return candidate or None
            return None

        provider = _normalise_provider(getattr(inference_config, "provider", None))
        provider = provider or _normalise_provider(getattr(inference_config, "service", None))
        provider = (
            _normalise_provider(params_map.get("provider"))
            or _normalise_provider(params_map.get("service"))
            or provider
        )

        if provider is None and getattr(integrations_config, "enable_ollama", False):
            provider = "ollama"

        if provider not in {"ollama"}:
            provider = "lm_studio"

        provider_label = "Ollama" if provider == "ollama" else "LM Studio"

        def _parse_float(value: object, default: float, *, warning: str) -> float:
            if value is None:
                return default
            try:
                return float(value)
            except (TypeError, ValueError):
                logger.warning("Invalid %s '%s'; defaulting to %.1f", warning, value, default)
                return default

        raw_temperature = params_map.get("temperature", 0.0)
        temperature = _parse_float(raw_temperature, 0.0, warning=f"{provider_label} temperature")

        raw_system_prompt = params_map.get("system_prompt")
        system_prompt = str(raw_system_prompt) if raw_system_prompt is not None else None

        raw_timeout = params_map.get("timeout")
        timeout_value: float | None
        if raw_timeout is None:
            timeout_value = None
        else:
            timeout_value = _parse_float(
                raw_timeout, 0.0, warning=f"{provider_label} timeout"
            )
            if timeout_value == 0.0:
                timeout_value = 0.0 if isinstance(raw_timeout, (int, float)) else None

        if provider == "ollama":
            model = params_map.get("model")
            if not model:
                logger.info("Ollama inference skipped: model missing")
                return None, temperature, system_prompt, provider_label

            host = params_map.get("host") or params_map.get("base_url")
            if not host:
                host = getattr(integrations_config, "ollama_endpoint", None)

            try:
                client = OllamaClient(
                    model=str(model),
                    host=str(host) if host else None,
                    timeout=timeout_value,
                )
            except Exception as exc:
                logger.warning("Failed to initialise Ollama client: %s", exc)
                return None, temperature, system_prompt, provider_label

            return client, temperature, system_prompt, provider_label

        if not isinstance(params, Mapping):
            logger.info("LM Studio inference skipped: configuration missing")
            return None, 0.0, None, provider_label

        base_url = params_map.get("base_url")
        model = params_map.get("model")
        if not base_url or not model:
            logger.info("LM Studio inference skipped: base_url or model missing")
            return None, temperature, system_prompt, provider_label

        headers_param = params_map.get("headers")
        headers = headers_param if isinstance(headers_param, Mapping) else None

        try:
            client = LmStudioClient(
                base_url=str(base_url),
                model=str(model),
                headers={str(k): str(v) for k, v in dict(headers or {}).items()},
                timeout=timeout_value if timeout_value is not None else None,
            )
        except Exception as exc:
            logger.warning("Failed to initialise LM Studio client: %s", exc)
            return None, temperature, system_prompt, provider_label

        return client, temperature, system_prompt, provider_label

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
        documents_for_context = self._documents_from_context(text_context)

        if self.chroma_reranker and text_context:
            chroma_documents = list(documents_for_context)
            reranked_documents = self.chroma_reranker.rerank(question, chroma_documents)
            ordered_context = self._map_documents_to_context(
                reranked_documents,
                chroma_documents,
                text_context,
            )
            if ordered_context:
                text_context = ordered_context
                documents_for_context = self._documents_from_context(text_context)

        if self.reranker and text_context:
            rerank_documents = list(documents_for_context)
            reranked_documents = self.reranker.rerank_context(
                rerank_documents,
                question,
                top_k=min(rerank_limit, len(rerank_documents)),
            )
            ordered_context = self._map_documents_to_context(
                reranked_documents,
                rerank_documents,
                text_context,
            )
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
            provider_label = getattr(self, "_lm_provider_label", None) or "LLM"
            logger.info("%s generation skipped: no retrieval context available", provider_label)
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
            provider_label = getattr(self, "_lm_provider_label", None) or "LLM"
            message = str(exc).splitlines()[0]
            logger.warning("%s generation failed: %s", provider_label, message)
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

        if getattr(self, "chroma_reranker", None) is not None:
            self._call_shutdown(self.chroma_reranker)
            self.chroma_reranker = None

        if getattr(self, "database_pipeline", None) is not None:
            self.database_pipeline = None

        self.data_loader = None
        self._lm_client = None

    @staticmethod
    def _documents_from_context(
        text_context: Sequence[tuple[str, str, Mapping[str, object] | dict, float]]
    ) -> list[Document]:
        return [
            Document(page_content=doc_text, metadata=dict(metadata))
            for _, doc_text, metadata, _ in text_context
        ]

    @staticmethod
    def _map_documents_to_context(
        reranked_documents: Sequence[Document],
        original_documents: Sequence[Document],
        base_context: Sequence[tuple[str, str, Mapping[str, object] | dict, float]],
    ) -> list[tuple[str, str, dict, float]]:
        if not reranked_documents or not original_documents or not base_context:
            return []

        identity_map = {id(doc): index for index, doc in enumerate(original_documents)}
        text_buckets: dict[str, list[int]] = {}
        for index, (_, doc_text, _, _) in enumerate(base_context):
            text_buckets.setdefault(str(doc_text), []).append(index)

        used_indices: set[int] = set()
        ordered_context: list[tuple[str, str, dict, float]] = []

        for document in reranked_documents:
            matched_index = None
            doc_identity = identity_map.get(id(document))
            if doc_identity is not None and doc_identity not in used_indices:
                matched_index = doc_identity
            else:
                text_value = getattr(document, "page_content", None)
                if text_value is not None:
                    bucket = text_buckets.get(str(text_value), [])
                    while bucket:
                        candidate_index = bucket.pop(0)
                        if candidate_index in used_indices:
                            continue
                        matched_index = candidate_index
                        break

            if matched_index is None:
                continue

            used_indices.add(matched_index)
            original_text = base_context[matched_index]
            ordered_context.append(
                (
                    original_text[0],
                    original_text[1],
                    dict(original_text[2]),
                    original_text[3],
                )
            )

            # Ensure future lookups respect the consumed entry.
            text_value = base_context[matched_index][1]
            bucket = text_buckets.get(str(text_value))
            if bucket and matched_index in bucket:
                bucket.remove(matched_index)

        return ordered_context


def create_standard_pipeline(
    config: Mapping[str, Any] | RAGConfig | RAGConfigTemplate | None,
    *,
    attach_reranker: bool = True,
) -> StandardRAGPipeline:
    """Build a :class:`StandardRAGPipeline` with normalised configuration."""

    resolved_config = RAGConfig.ensure(config)
    pipeline = StandardRAGPipeline(resolved_config)
    if not attach_reranker:
        pipeline.reranker = None
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

