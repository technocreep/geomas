from __future__ import annotations

import copy
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional, Sequence

from langchain_chroma import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from geomas.core.inference.interface import LlmConnector as RuntimeLlmConnector
from geomas.core.inference.lmstudio_client import LmStudioClient
from geomas.core.inference.ollama_client import OllamaClient
from geomas.core.rag_modules.data_adapter import DataLoaderAdapter
from geomas.core.rag_modules.database.chroma_db import (
    DatabaseRagPipeline,
    ProcessingResult,
)
from geomas.core.rag_modules.parser.rag_parser import DocumentParser
from geomas.core.rag_modules.steps.ranker import (
    LLMReranker,
    build_chroma_reranker,
    build_llm_reranker,
    build_scored_context,
)
from geomas.core.rag_modules.steps.retriever import (
    DocsSearcherModels,
    Retriever,
)
from geomas.core.repository.promts_repository import PROMPT_RANK
from geomas.core.repository.rag_repository import (
    RAGConfig,
    RAGConfigTemplate,
    InferenceConfigTemplate,
    IntegrationsConfigTemplate,
    RankingConfigTemplate,
    DataConfigTemplate,
    RetrievalConfigTemplate,
)


logger = logging.getLogger(__name__)

def _create_llm_connector(
    url: str, model_params: Mapping[str, Any] | None
) -> "_LlmConnector":
    """Create the LLM connector for reranking once dependencies are present."""
    params = dict(model_params or {})
    return RuntimeLlmConnector(url, params)


class BaseRAGPipeline(ABC):
    @abstractmethod
    def ingest_documents(self, documents_path: str, **kwargs: Any) -> bool:
        """Ingest documents into the vector store."""

    @abstractmethod
    def query(self, question: str, **kwargs: Any) -> Dict[str, Any]:
        """Query the pipeline and return structured results."""


class StandardRAGPipeline(BaseRAGPipeline):
    """Reference implementation that wires the legacy database pipeline.

    The pipeline routes all ingestion through :class:`DataLoaderAdapter`, which
    applies unified cleaning for supported formats before chunking and
    enrichment. Retrieval flows through the enhanced :class:`Retriever` helper
    (replacing the old ``RetrievingPipeline``), preserving chained collection
    semantics while keeping vector-store access confined to the LangChain
    ``Chroma`` wrapper. The pipeline optionally attaches an :class:`LLMReranker`
    when reranking is enabled in the configuration; no alternative rerankers are
    constructed here.
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
        self.data_loader = self._initialise_data_loader(
            self.config_template.data, chunking_params
        )
        database_config = self.config_template.database
        collection_name = database_config.collection_name
        store_config = self.config_template.vector_store
        self.embedding_function = self._initialise_embedding(self.config_template.retrieval)
        self.vector_store_config = store_config.to_dict()
        self.collection_name = collection_name
        self.vector_store = self._initialise_store(
            self.vector_store_config,
            self.embedding_function,
            collection_name=collection_name,
        )
        self.database_pipeline = DatabaseRagPipeline(
            vector_store=self.vector_store,
            embedding_function=self.embedding_function,
            parser=self.parser,
            data_loader=self.data_loader,
        )
        self.data_loader = self.database_pipeline.data_loader
        self.last_ingest_result: ProcessingResult | None = None
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

    def _initialise_data_loader(
        self,
        data_config: "DataConfigTemplate",
        chunking_params: Mapping[str, Any] | None,
    ) -> DataLoaderAdapter:
        """Build the shared :class:`DataLoaderAdapter` with unified cleaning."""
        transformation_config = (
            dict(data_config.transformations)
            if getattr(data_config, "transformations", None)
            else None
        )
        loader_params = dict(data_config.loader_params) or None
        return DataLoaderAdapter(
            loader_type=data_config.loader_type,
            parser=self.parser,
            loader_params=loader_params,
            transformation_config=transformation_config,
            chunking_params=chunking_params,
        )

    @staticmethod
    def _build_search_kwargs(retrieval_config: RetrievalConfigTemplate) -> dict[str, Any]:
        """Prepare retriever kwargs without shadowing runtime ``k`` values."""
        raw_kwargs = dict(getattr(retrieval_config, "search_kwargs", {}) or {})
        for reserved in ("k", "filter"):
            raw_kwargs.pop(reserved, None)
        return raw_kwargs

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

    def _shutdown_embedding(self, embedding_function: object | None) -> None:
        """Release resources held by ``embedding_function`` when possible."""
        if embedding_function is None:
            return
        if self._call_shutdown(embedding_function):
            return
        for attr in ("client", "model", "_model", "_embeddings_model"):
            candidate = getattr(embedding_function, attr, None)
            if self._call_shutdown(candidate):
                return

    @staticmethod
    def _initialise_embedding(
        retrieval_config: RetrievalConfigTemplate
    ) -> Embeddings:
        model_name = retrieval_config.embedding_model_name
        checkpoint = retrieval_config.checkpoint
        return OpenCLIPEmbeddings(model_name=model_name, checkpoint=checkpoint)

    @staticmethod
    def _initialise_store(
        store_config: dict[str, Any],
        embedding_function: Embeddings,
        *,
        collection_name: str,
    ) -> Chroma:
        """Create a Chroma-backed vector store for ``config``."""
        client_config = dict(store_config.get("client", {}))
        chroma_kwargs: dict[str, Any] = {
            "collection_name": collection_name,
            "embedding_function": embedding_function,
            "persist_directory": client_config.get("persist_directory"),
            "host": client_config.get("host"),
            "port": client_config.get("port"),
            "ssl": client_config.get("ssl"),
            "headers": client_config.get("headers"),
            "chroma_cloud_api_key": client_config.get("chroma_cloud_api_key"),
            "tenant": client_config.get("tenant"),
            "database": client_config.get("database"),
            "client_settings": client_config.get("client_settings"),
        }

        filtered_kwargs = {
            key: value for key, value in chroma_kwargs.items() if value is not None
        }

        return Chroma(**filtered_kwargs)

    def _scoped_vector_store(
        self, collection_name: str, *, scope_config: Mapping[str, Any] | None = None
    ) -> Chroma:
        if collection_name == self.collection_name:
            return self.vector_store

        merged_config = copy.deepcopy(self.vector_store_config)
        base_client = dict(merged_config.get("client", {}))

        client_overrides: dict[str, Any] = {}
        if isinstance(scope_config, Mapping):
            scope_client = scope_config.get("client")
            if isinstance(scope_client, Mapping):
                client_overrides.update(dict(scope_client))

            for key in (
                "persist_directory",
                "host",
                "port",
                "ssl",
                "headers",
                "chroma_cloud_api_key",
                "tenant",
                "database",
                "client_settings",
            ):
                if key in scope_config:
                    client_overrides.setdefault(key, scope_config[key])

            collection_override = scope_config.get("collection_name")
            if isinstance(collection_override, str) and collection_override.strip():
                collection_name = collection_override

        merged_client = {**base_client, **client_overrides}
        if merged_client:
            merged_config["client"] = merged_client
            return self._initialise_store(
                merged_config,
                self.embedding_function,
                collection_name=collection_name,
            )
        runtime_client = getattr(self.vector_store, "_client", None)
        return Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_function,
            client=runtime_client,
        )

    @staticmethod
    def _initialise_reranker(
            ranking_config: RankingConfigTemplate
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

    @staticmethod
    def _initialise_inference(
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

        base_url = params_map.get("base_url")
        model = params_map.get("model")
        if not base_url or not model:
            logger.info("LM Studio inference skipped: base_url or model missing")
            return None, temperature, system_prompt, provider_label

        headers_param = params_map.get("headers")
        headers = headers_param if isinstance(headers_param, Mapping) else None

        client = LmStudioClient(
            base_url=str(base_url),
            model=str(model),
            headers={str(k): str(v) for k, v in dict(headers or {}).items()},
            timeout=timeout_value if timeout_value is not None else None,
        )
        return client, temperature, system_prompt, provider_label

    def ingest_documents(
            self,
            documents_path: str,
            *,
            describe_images: bool = False,
            **kwargs: Any
    ) -> bool:
        """Ingest documents into the configured vector store.

        Args:
            documents_path: Directory containing artefacts to ingest.
            describe_images: When ``True``, generate textual descriptions for
                discovered images and store them as LangChain documents alongside
                the base chunks.
            **kwargs: Additional parameters forwarded to the database pipeline.
                Supported keys include ``namespace`` to isolate uploads.
        """
        namespace_value = kwargs.pop("namespace", "global")
        namespace = str(namespace_value) if namespace_value is not None else "global"
        try:
            result = self.database_pipeline.process(
                documents_path,
                document_name=kwargs.get("document_name"),
                namespace=namespace,
                generate_image_descriptions=describe_images,
            )
        except FileNotFoundError:
            raise
        except Exception as exc:
            logger.exception("Database ingestion failed for '%s': %s", documents_path, exc)
            result = ProcessingResult(success=False)

        self.last_ingest_result = result
        if result.success:
            if result.documents_ingested > 0:
                logger.info("Ingested %s documents", result.documents_ingested)
            elif result.documents_skipped > 0:
                logger.info(
                    "Skipped ingestion for %s; documents unchanged",
                    documents_path,
                )
        return result.success

    def query(self, question: str, **kwargs: Any) -> Dict[str, Any]:
        search_kwargs = self._build_search_kwargs(self.config_template.retrieval)
        score_threshold = getattr(self.config_template.retrieval, "score_threshold", 0.5)
        if score_threshold is not None:
            search_kwargs.setdefault("score_threshold", score_threshold)
        base_top_k = (
            self.config_template.retrieval.text_top_k
            or self.config_template.retrieval.chunk_limit
            or self.config_template.retrieval.top_k
            or 5
        )
        scopes_argument = kwargs.get("scopes") or kwargs.get("namespaces")
        scopes: list[str] = []
        scope_store_configs: dict[str, Mapping[str, Any]] = {}
        for scope, config in scopes_argument.items():
            scopes.append(scope)
            scope_store_configs[scope] = {
                "client": {"persist_directory": str(config)}
            }
        scoped_vector_stores: dict[str, Chroma] = {}
        scoped_embeddings: dict[str, Embeddings] = {}
        retriever_chain: list[Retriever] = []
        if scopes:
            for scope in scopes:
                scope_config = scope_store_configs.get(scope)
                if scope_config is None:
                    logger.warning(
                        "Scope '%s' is not configured; using base vector store settings",
                        scope,
                    )

                store = self._scoped_vector_store(scope, scope_config=scope_config)
                embedding = scoped_embeddings.get(scope)
                if embedding is None:
                    embedding = (
                        getattr(store, "_embedding_function", None)
                        or getattr(store, "embeddings", None)
                        or self.embedding_function
                    )
                scoped_vector_stores[scope] = store
                if embedding is not None:
                    scoped_embeddings.setdefault(scope, embedding)
                retriever_chain.append(
                    Retriever(
                        top_k=base_top_k,
                        docs_searcher_models=DocsSearcherModels(
                            embedding_model=embedding,
                            vector_store=store,
                        ),
                        search_type=(
                            getattr(self.config_template.retrieval, "search_type", None)
                            or "similarity_score_threshold"
                        ),
                        search_kwargs=search_kwargs,
                        scoped_vector_stores=scoped_vector_stores,
                    )
                )
            retriever = retriever_chain[0]
            chained_retrievers = retriever_chain[1:]
        else:
            retriever = Retriever(
                top_k=base_top_k,
                docs_searcher_models=DocsSearcherModels(
                    embedding_model=self.embedding_function,
                    vector_store=self.vector_store,
                ),
                search_type=(
                    getattr(self.config_template.retrieval, "search_type", None)
                    or "similarity_score_threshold"
                ),
                search_kwargs=search_kwargs,
            )
            chained_retrievers = [retriever]

        reranker = build_chroma_reranker(
            self.config_template.ranking,
            embedding_function=self.embedding_function,
            vector_store=self.vector_store,
            collection_vector_stores=scoped_vector_stores,
            collection_embeddings=scoped_embeddings,
            logger=logger,
        )

        # reranker: LLMReranker | None = self._initialise_reranker(
        #     self.config_template.ranking
        # )

        base_filters: dict[str, object] = {}
        runtime_filters = kwargs.get("filters")
        if isinstance(runtime_filters, Mapping):
            base_filters.update(runtime_filters)

        if scopes:
            filters = dict(base_filters or {}) or None
        else:
            filters = Retriever.compose_filters(base_filters or None, scopes_argument) or None

        query_images = kwargs.get("query_images")

        raw_results = (
            retriever.search_chained(
                question,
                retrievers=chained_retrievers,
                top_k=kwargs.get("top_k"),
                filters=filters,
                query_images=query_images if isinstance(query_images, Sequence) else None,
                scopes=scopes,
                collection_overrides=scoped_vector_stores,
            )
        )
        unique_results = self._deduplicate_documents(raw_results)
        text_context = build_scored_context(unique_results, kwargs.get("top_k"))
        documents_for_context = self._documents_from_context(text_context)

        if reranker and text_context:
            chroma_documents = list(documents_for_context)
            reranked_documents = reranker.rerank(question, chroma_documents)
            ordered_context = self._map_documents_to_context(
                reranked_documents,
                chroma_documents,
                text_context,
            )
            if ordered_context:
                text_context = ordered_context
                documents_for_context = self._documents_from_context(text_context)

        # if reranker and text_context:
        #     rerank_documents = list(documents_for_context)
        #     reranked_documents = reranker.rerank_context(
        #         rerank_documents,
        #         question,
        #         top_k=min(kwargs.get("top_k"), len(rerank_documents)),
        #     )
        #     ordered_context = self._map_documents_to_context(
        #         reranked_documents,
        #         rerank_documents,
        #         text_context,
        #     )
        #     if ordered_context:
        #         text_context = ordered_context

        relevant_sources = self._collect_sources(text_context, kwargs.get("top_k"))
        answer = self._generate_answer(question, text_context)

        return {
            "question": question,
            "relevant_papers": relevant_sources,
            "text_context": text_context,
            "answer": answer,
        }

    @staticmethod
    def _collect_sources(
        text_context: Sequence[tuple[str, int | None, str, Mapping[str, object] | dict, float]],
        limit: int,
    ) -> list[Mapping[str, object]]:
        sources: list[Mapping[str, object]] = []
        seen: set[tuple[str, int | None]] = set()
        for _, chunk_index, _, metadata, _ in text_context:
            source = None
            if isinstance(metadata, Mapping):
                source = metadata.get("source")
            if isinstance(source, str) and (source, chunk_index) not in seen:
                seen.add((source, chunk_index))
                sources.append({"source": source, "chunk_index": chunk_index})
                if len(sources) >= limit:
                    break
        return sources

    def _generate_answer(
        self, question: str, text_context: list[tuple[str, int | None, str, dict, float]]
    ) -> str | None:
        client = getattr(self, "_lm_client", None)
        if client is None:
            return None

        context_block = self._format_context(text_context)

        messages: list[dict[str, str]] = []
        if self._lm_system_prompt:
            messages.append({"role": "system", "content": self._lm_system_prompt})

        instruction = (
            "Ты — геологический ассистент. Используй данные из раздела Sources, чтобы дать максимально точный и развернутый ответ."
        )
        prompt = (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{question}\n\n"
            "### Sources:\n"
            f"{context_block}\n\n"
            "### Response:\n"
        )
        messages.append({"role": "user", "content": prompt})
        return client.generate(messages, temperature=self._lm_temperature)

    @staticmethod
    def _format_context(text_context: list[tuple[str, int | None, str, dict, float]]) -> str:
        formatted: list[str] = []
        for index, (_, chunk_index, content, metadata, score) in enumerate(
            text_context, start=1
        ):
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
            if chunk_index is not None:
                header_parts.append(f"chunk {chunk_index}")
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
        if getattr(self, "vector_store", None) is not None:
            self._call_shutdown(self.vector_store)
            self.vector_store = None

        if getattr(self, "embedding_function", None) is not None:
            self._shutdown_embedding(self.embedding_function)
            self.embedding_function = None

        if getattr(self, "database_pipeline", None) is not None:
            self.database_pipeline = None

        self.data_loader = None
        self._lm_client = None

    @staticmethod
    def _documents_from_context(
        text_context: Sequence[tuple[str, int | None, str, Mapping[str, object] | dict, float]]
    ) -> list[Document]:
        return [
            Document(
                page_content=doc_text,
                metadata={
                    **dict(metadata),
                    **(
                        {"chunk_index": chunk_index}
                        if chunk_index is not None and "chunk_index" not in metadata
                        else {}
                    ),
                },
            )
            for _, chunk_index, doc_text, metadata, _ in text_context
        ]

    @staticmethod
    def _map_documents_to_context(
        reranked_documents: Sequence[Document],
        original_documents: Sequence[Document],
        base_context: Sequence[tuple[str, int | None, str, Mapping[str, object] | dict, float]],
    ) -> list[tuple[str, int | None, str, dict, float]]:
        if not reranked_documents or not original_documents or not base_context:
            return []

        identity_map = {id(doc): index for index, doc in enumerate(original_documents)}
        text_buckets: dict[str, list[int]] = {}
        for index, (_, _, doc_text, _, _) in enumerate(base_context):
            text_buckets.setdefault(str(doc_text), []).append(index)

        used_indices: set[int] = set()
        ordered_context: list[tuple[str, int | None, str, dict, float]] = []
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
                    original_text[2],
                    dict(original_text[3]),
                    original_text[4],
                )
            )
            # Ensure future lookups respect the consumed entry.
            text_value = base_context[matched_index][2]
            bucket = text_buckets.get(str(text_value))
            if bucket and matched_index in bucket:
                bucket.remove(matched_index)
        return ordered_context

    @staticmethod
    def _document_chunk_index(document: Document) -> int | None:
        metadata = document.metadata if isinstance(document.metadata, Mapping) else {}
        chunk_value = None
        for key in ("chunk_index", "chunkId", "chunkNumber"):
            if key in metadata:
                chunk_value = metadata[key]
                break
        if chunk_value is None and document.id:
            match = re.findall(r"chunk_(\d+)", str(document.id))
            if match:
                chunk_value = match[-1]
        try:
            return int(chunk_value) if chunk_value is not None else None
        except (TypeError, ValueError):
            return None

    def _deduplicate_documents(self, documents: Sequence[Document]) -> list[Document]:
        seen: set[tuple[str, int | None]] = set()
        unique: list[Document] = []
        for document in documents:
            metadata = document.metadata if isinstance(document.metadata, Mapping) else {}
            chunk_index = self._document_chunk_index(document)
            scope = metadata.get("scope")
            identifier = document.id or metadata.get("uuid") or metadata.get("document_name")
            if identifier:
                key = (str(identifier), chunk_index)
            elif scope:
                key = (str(scope), chunk_index, document.page_content)
            else:
                key = (document.page_content, chunk_index)
            if key in seen:
                continue
            seen.add(key)
            unique.append(document)
        return unique
