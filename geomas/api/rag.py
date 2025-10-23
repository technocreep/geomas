from __future__ import annotations

import logging
from threading import RLock
from typing import Any, Dict, Mapping, Optional

from geomas.core.rag_modules.rag_pipeline import StandardRAGPipeline
from geomas.core.repository.rag_repository import (
    RAGConfig,
    RAGConfigTemplate,
    _deep_update,
)

logger = logging.getLogger(__name__)


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
            self.config = self._build_config(overrides=overrides, config_path=config_path)
            self.pipeline = StandardRAGPipeline(self.config.to_dict())
            self.is_initialized = False
            logger.info("RAG configuration reapplied; pipeline reset and awaiting initialisation")

    def initialize_pipeline(self, documents_path: Optional[str] = None) -> bool:
        """Initialise the pipeline and optionally ingest documents.

        Args:
            documents_path: Optional path pointing to documents that should be
                ingested as part of the initialisation sequence.

        Returns:
            ``True`` when the pipeline is ready for queries. ``False`` when the
            optional ingestion step fails.
        """

        with self._state_lock:
            if not documents_path and self.is_initialized:
                logger.info("RAG pipeline already initialised; skipping ingestion")
                return True

            if not documents_path:
                logger.info("Initialising RAG pipeline without ingestion")
                self.is_initialized = True
                return True

            logger.info("Starting ingestion from %s", documents_path)
            success = self.pipeline.ingest_documents(documents_path)
            if not success:
                logger.error("Failed to ingest documents from %s", documents_path)
                return False

            self.is_initialized = True
            logger.info("RAG pipeline initialised successfully from %s", documents_path)
            return True

    def ask_question(self, question: str, **kwargs: Any) -> Dict[str, Any]:
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

            try:
                return self.pipeline.query(question, **kwargs)
            except Exception as exc:
                logger.exception("Pipeline query failed: %s", exc)
                raise RuntimeError("Failed to process the question") from exc

    def add_documents(self, path: str) -> bool:
        """Ingest additional documents into the pipeline.

        Args:
            path: Filesystem path referencing documents that should be added to
                the active vector store.

        Returns:
            ``True`` if ingestion succeeds, ``False`` otherwise.

        Raises:
            ValueError: If ``path`` is an empty string.
        """

        if not path:
            raise ValueError("A valid path must be provided for ingestion")

        with self._state_lock:
            logger.info("Ingesting documents from %s", path)
            success = self.pipeline.ingest_documents(path)
            if not success:
                logger.error("Ingestion pipeline reported failure for %s", path)
                return False

            self.is_initialized = True
            logger.info("Successfully ingested documents from %s", path)
            return True

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
            reranking_active = bool(config_template.ranking.use_llm_reranking)
            reranker_enabled = bool(getattr(self.pipeline, "reranker", None))
            pipeline_details = {
                "retriever_enabled": bool(getattr(self.pipeline, "retriever", None)),
                "reranker_enabled": reranker_enabled,
                "reranking_active": reranking_active and reranker_enabled,
                "store_ready": hasattr(self.pipeline, "store"),
                "inference_configured": bool(getattr(self.pipeline, "inference", None)),
                "monitoring_configured": bool(getattr(self.pipeline, "monitoring", None)),
            }

            last_result = getattr(self.pipeline, "last_ingest_result", None)
            ingestion_snapshot = None
            if last_result is not None:
                ingestion_snapshot = {
                    "success": bool(getattr(last_result, "success", False)),
                    "documents": int(getattr(last_result, "documents_ingested", 0)),
                    "summaries": int(getattr(last_result, "summaries_created", 0)),
                }

            return {
                "is_initialized": self.is_initialized,
                "config": config_state,
                "components": pipeline_details,
                "last_ingestion": ingestion_snapshot,
            }


