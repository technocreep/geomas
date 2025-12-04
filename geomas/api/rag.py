from __future__ import annotations

import logging
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Mapping, Optional
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from geomas.core.rag_modules.rag_pipeline import StandardRAGPipeline
from geomas.core.inference import ollama_client
from geomas.core.repository.rag_repository import (
    RAGConfig,
    RAGConfigTemplate,
    _deep_update,
)


logger = logging.getLogger(__name__)


_OLLAMA_EXPORTS = {
    "OllamaSettings",
    "load_ollama_settings",
}


def build_ollama_rag_config(
    documents_dir: Path | str | None = None,
    *,
    cache_dir: Path | str | None = None,
    local_cache_dir: Path | str | None = None,
    settings: Mapping[str, object] | object | None = None,
    chat_id: str | None = None,
    global_rag_dir: Path | str | None = None,
    local_rag_dir: Path | str | None = None,
):
    """Backward-compatible wrapper around the core Ollama config builder."""
    base_settings = settings
    if isinstance(base_settings, Mapping):
        base_settings = ollama_client.load_ollama_settings().with_overrides(base_settings)
    elif base_settings is None:
        base_settings = ollama_client.load_ollama_settings()

    documents_path: Path | None = None
    if documents_dir is not None:
        documents_path = Path(documents_dir).expanduser().resolve()

    global_source = global_rag_dir or cache_dir or documents_path
    if global_source is None:
        raise ValueError(
            "A documents_dir, cache_dir, or global_rag_dir must be provided to build the config",
        )
    resolved_global = Path(global_source).expanduser().resolve()

    local_source = local_rag_dir or local_cache_dir
    resolved_local = (
        Path(local_source).expanduser().resolve() if local_source is not None else None
    )
    resolved_chat = chat_id.strip() if isinstance(chat_id, str) and chat_id.strip() else None

    build_core = ollama_client.build_ollama_rag_config

    cache_value = cache_dir
    local_cache_value = local_cache_dir

    if documents_path is None:
        documents_path = resolved_global

    try:
        config = build_core(
            documents_path,
            cache_dir=Path(cache_value).expanduser().resolve() if cache_value is not None else None,
            local_cache_dir=(
                Path(local_cache_value).expanduser().resolve()
                if local_cache_value is not None
                else None
            ),
            settings=base_settings,
        )
    except TypeError:
        config = build_core(
            chat_id=resolved_chat,
            global_rag_dir=resolved_global,
            local_rag_dir=resolved_local,
            settings=base_settings,
        )
    if isinstance(config, RAGConfig):
        overrides = config.to_dict()
        database_overrides = overrides.setdefault("database", {})
        database_overrides["persistent_path"] = str(resolved_global)
        collection_name = database_overrides.get("collection_name", "geomas")
        vector_overrides = overrides.setdefault("vector_store", {})
        if resolved_local is not None:
            database_overrides.setdefault("local_collection_name", f"{collection_name}_local")
            vector_overrides["local_client"] = {"persistent_path": str(resolved_local)}
        elif "local_client" in vector_overrides:
            vector_overrides.pop("local_client", None)
            database_overrides.pop("local_collection_name", None)
        config = RAGConfig.from_mapping(overrides)
    return config


class RagApi:
    """High-level façade that orchestrates the standard GeoMAS RAG pipeline."""
    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        config_path: Optional[str] = None,
    ) -> None:
        """Initialise the API façade.

        Args:
            config: Optional mapping of configuration overrides applied on top of
                :func:`RAGConfig.default`.
            config_path: Optional filesystem path to a YAML/JSON configuration
                file. When provided, its values become the baseline before
                ``config`` overrides are applied.

        The constructor eagerly builds :class:`StandardRAGPipeline` so callers
        can immediately configure or initialise the system without additional
        setup calls. All public interactions are guarded by a re-entrant lock to
        avoid state races while reconfiguring the pipeline.
        """
        self._state_lock = RLock()
        self._chroma_client: "chromadb.ClientAPI | None" = None
        self.is_initialized = False
        self.config = self._build_config(overrides=config, config_path=config_path)
        self.pipeline = StandardRAGPipeline(self.config.to_dict())

    def _build_config(
        self,
        overrides: Optional[Mapping[str, Any] | RAGConfig | RAGConfigTemplate] = None,
        config_path: Optional[str] = None,
    ) -> RAGConfig:
        """Return a :class:`RAGConfig` built from defaults and overrides."""
        if isinstance(overrides, RAGConfig):
            return overrides.copy()
        if isinstance(overrides, RAGConfigTemplate):
            return RAGConfig.from_template(overrides)

        if config_path:
            try:
                base_config = RAGConfig.from_path(config_path)
            except (OSError, ValueError) as exc:
                logger.error("Failed to load config from %s: %s", config_path, exc)
                base_config = RAGConfig.default()
        else:
            base_config = RAGConfig.default()

        if overrides is None:
            return base_config

        mapping: Optional[Mapping[str, Any]] = None
        if isinstance(overrides, Mapping):
            mapping = overrides
        elif hasattr(overrides, "to_dict") and callable(getattr(overrides, "to_dict")):
            mapping = overrides.to_dict()

        if mapping is None:
            raise TypeError(f"Unsupported overrides type: {type(overrides)!r}")

        logger.info("Applying configuration overrides to base config")
        merged = base_config.to_dict()
        _deep_update(merged, dict(mapping))
        return RAGConfig.from_mapping(merged)

    def apply_config(
        self,
        overrides: Optional[Mapping[str, Any] | RAGConfig | RAGConfigTemplate] = None,
        config_path: Optional[str] = None,
    ) -> None:
        """Rebuild the pipeline using a fresh configuration.

        Args:
            overrides: Mapping containing configuration overrides.
            config_path: Optional path to a YAML/JSON configuration file.

        The method rebuilds ``self.config`` and ``self.pipeline`` using the
        provided inputs, resets ``is_initialized`` to ``False`` and ensures the
        update is atomic with respect to other public operations. When both
        arguments are omitted the pipeline falls back to
        :func:`RAGConfig.default`.
        """
        with self._state_lock:
            new_config = self._build_config(overrides=overrides, config_path=config_path)
            new_pipeline = StandardRAGPipeline(new_config.to_dict())
            old_pipeline = self.pipeline
            self.config = new_config
            self.pipeline = new_pipeline
            self.is_initialized = False
            logger.info("RAG configuration reapplied; pipeline reset and awaiting initialisation")

        if old_pipeline is not None:
            try:
                old_pipeline.close()
            except Exception as exc:
                logger.debug("Failed to close previous pipeline during reconfiguration: %s", exc)

    def _require_pipeline(self) -> StandardRAGPipeline:
        pipeline = self.pipeline
        if pipeline is None:
            logger.error("Attempted to use RAG pipeline after it was closed")
            raise RuntimeError("RAG pipeline has been closed")
        return pipeline

    def initialize_pipeline(
            self,
            documents_dir: Optional[str | Path],
            namespace: str,
            include_images: bool = False,
            describe_images: bool = False,
    ) -> bool:
        """Initialise the pipeline and optionally ingest documents.

        Args:
            documents_dir: Optional path pointing to documents that should be
                ingested as part of the initialisation sequence.
            namespace: Target namespace ("global" or f"{chat_id}_local").
            include_images: Optional flag enabling multimodal ingestion when
                supported by the configured vector store.
            describe_images: When ``True``, generate captions for detected
                images and store them as text documents within the namespace.

        Returns:
            ``True`` when the pipeline is ready for queries. ``False`` when the
            optional ingestion step fails.
        """
        with self._state_lock:
            self._require_pipeline()
            self.is_initialized = True
            result = self.pipeline.ingest_documents(
                documents_dir,
                namespace=namespace,
                include_images=include_images,
                describe_images=describe_images,
            )
            return result

    def ask_question(self, question: str, history: str, **kwargs: Any) -> Dict[str, Any]:
        """Query the pipeline and return the structured response.

        Args:
            question: Natural-language query issued to the pipeline.
            **kwargs: Additional parameters forwarded to
                :meth:`StandardRAGPipeline.query`.

        Returns:
            A dictionary matching the schema provided by
            :class:`StandardRAGPipeline`.

        Raises:
            ValueError: If ``question`` is empty.
            RuntimeError: When the pipeline has not been initialised or if the
                underlying pipeline reports a failure.
        """
        if not question:
            raise ValueError("Question must be a non-empty string")

        with self._state_lock:
            if not self.is_initialized:
                logger.error("Attempted to query RAG pipeline before initialisation")
                raise RuntimeError("RAG pipeline is not initialised")

            pipeline = self._require_pipeline()
            return pipeline.query(question, history, **kwargs)

    def close(self) -> None:
        """Close the underlying pipeline and release its resources."""
        with self._state_lock:
            pipeline = self.pipeline
            if pipeline is None:
                return
            self.pipeline = None
            self.is_initialized = False
        try:
            pipeline.close()
        except Exception as exc:
            logger.debug("Failed to close pipeline: %s", exc)

    def __enter__(self) -> "RagApi":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _get_chroma_client(self, chroma_client: "chromadb.ClientAPI | None" = None):
        """Return a cached Chroma client or store a provided one."""
        import chromadb

        if chroma_client is not None:
            self._chroma_client = chroma_client
        if self._chroma_client is None:
            self._chroma_client = chromadb.Client()
        return self._chroma_client

    def _build_store(
        self,
        *,
        collection_name: str,
        embedding_function: "Embeddings",
        chroma_client: "chromadb.ClientAPI | None" = None,
    ) -> "Chroma":
        return Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            client=self._get_chroma_client(chroma_client),
        )

    def delete_collection(
        self, collection: str, *, chroma_client: "chromadb.ClientAPI | None" = None
    ) -> None:
        """Delete a Chroma collection."""
        client = self._get_chroma_client(chroma_client)
        client.delete_collection(collection)

    def list_collections(
        self, *, chroma_client: "chromadb.ClientAPI | None" = None
    ) -> list:
        """List all Chroma collections."""
        client = self._get_chroma_client(chroma_client)
        return client.list_collections()

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Return metadata describing the current pipeline state.

        Returns:
            A dictionary containing the serialisable configuration snapshot,
            component availability flags, and the initialisation state. The
            ``components`` entry includes whether reranking is configured and
            active.
        """
        with self._state_lock:
            config_state = self.config.to_dict()
            config_template = self.config.as_template()
            ranking_template = config_template.ranking
            llm_requested = bool(getattr(ranking_template, "use_llm_reranking", False))
            chroma_requested = bool(getattr(ranking_template, "use_chroma_reranking", False))
            llm_reranker_enabled = bool(getattr(self.pipeline, "reranker", None))
            chroma_reranker_enabled = bool(getattr(self.pipeline, "chroma_reranker", None))
            pipeline_details = {
                "retriever_enabled": bool(getattr(self.pipeline, "retriever", None)),
                "reranker_enabled": llm_reranker_enabled,
                "reranking_active": llm_requested and llm_reranker_enabled,
                "llm_reranker_enabled": llm_reranker_enabled,
                "llm_reranking_active": llm_requested and llm_reranker_enabled,
                "chroma_reranker_enabled": chroma_reranker_enabled,
                "chroma_reranking_active": chroma_requested and chroma_reranker_enabled,
                "store_ready": hasattr(self.pipeline, "store"),
                "inference_configured": bool(getattr(self.pipeline, "_lm_client", None)),
                "monitoring_configured": bool(getattr(self.pipeline, "monitoring", None)),
            }
            last_result = getattr(self.pipeline, "last_ingest_result", None)
            ingestion_snapshot = None
            if last_result is not None:
                ingestion_snapshot = {
                    "success": bool(getattr(last_result, "success", False)),
                    "documents": int(getattr(last_result, "documents_ingested", 0)),
                    "skipped": int(getattr(last_result, "documents_skipped", 0)),
                    "summaries": int(getattr(last_result, "summaries_created", 0)),
                }
            return {
                "is_initialized": self.is_initialized,
                "config": config_state,
                "components": pipeline_details,
                "last_ingestion": ingestion_snapshot,
            }