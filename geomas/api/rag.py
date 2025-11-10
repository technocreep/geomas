from __future__ import annotations

import logging
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Mapping, Optional, Sequence

from geomas.core.rag_modules import rag_pipeline
from geomas.core.rag_modules.database.chroma_db import ProcessingResult
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
        self.pipeline: StandardRAGPipeline | None = StandardRAGPipeline(self.config.to_dict())

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
            mapping = overrides.to_dict()  # type: ignore[arg-type]

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

    def _ingest_path(
        self,
        documents_path: Path | str,
        *,
        document_name: str | None = None,
    ) -> ProcessingResult:
        """Ingest ``documents_path`` through the standard pipeline helper."""
        pipeline = self._require_pipeline()
        result = rag_pipeline.ingest_documents(
            pipeline,
            Path(documents_path),
            document_name=document_name,
        )
        if result.success:
            self.is_initialized = True
        return result

    def initialize_pipeline(self, documents_path: Optional[str | Path] = None) -> bool:
        """Initialise the pipeline and optionally ingest documents.

        Args:
            documents_path: Optional path pointing to documents that should be
                ingested as part of the initialisation sequence.

        Returns:
            ``True`` when the pipeline is ready for queries. ``False`` when the
            optional ingestion step fails.
        """
        with self._state_lock:
            self._require_pipeline()

            if documents_path:
                logger.info("Starting ingestion from %s", documents_path)
                result = self._ingest_path(documents_path)
                if not result.success:
                    logger.error("Failed to ingest documents from %s", documents_path)
                    return False

                logger.info("RAG pipeline initialised successfully from %s", documents_path)
                return True

            if self.is_initialized:
                logger.info("RAG pipeline already initialised; skipping reinitialisation")
                return True

            self.is_initialized = True
            logger.info("Initialising RAG pipeline without ingestion")
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
                pipeline = self._require_pipeline()
                return pipeline.query(question, **kwargs)
            except Exception as exc:
                logger.exception("Pipeline query failed: %s", exc)
                raise RuntimeError("Failed to process the question") from exc

    def add_documents(self, path: Path | str) -> bool:
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
            result = self._ingest_path(path)
            if not result.success:
                logger.error("Ingestion pipeline reported failure for %s", path)
                return False

            logger.info("Successfully ingested documents from %s", path)
            return True

    def run_workflow(
        self,
        question: str,
        *,
        documents_path: Path | str | None = None,
        uploaded_documents: Sequence[Path | str] | None = None,
        query_kwargs: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Ingest supplied artefacts and execute a single query workflow.

        Args:
            question: Natural-language prompt executed against the pipeline.
            documents_path: Optional base path ingested before the query.
            uploaded_documents: Optional sequence of additional artefacts to ingest.
            query_kwargs: Optional mapping forwarded to :meth:`StandardRAGPipeline.query`.

        Returns:
            A dictionary containing the ``question``, the ingestion result for
            ``documents_path`` (``base_ingestion``), a list of ingestion results for
            ``uploaded_documents`` (``uploaded_ingestions``), and the pipeline
            ``response``.

        Raises:
            ValueError: If ``question`` is blank.
            RuntimeError: When the pipeline has been closed, fails to initialise, or the
                query raises an error.
        """
        if not question:
            raise ValueError("Question must be a non-empty string")

        uploads: Sequence[Path | str] = uploaded_documents or ()
        kwargs: Dict[str, Any] = dict(query_kwargs or {})

        with self._state_lock:
            pipeline = self._require_pipeline()

            base_result: ProcessingResult | None = None
            if documents_path is not None:
                logger.info("Running workflow ingestion from %s", documents_path)
                base_result = self._ingest_path(documents_path)

            uploaded_results: list[ProcessingResult] = []
            for candidate in uploads:
                logger.info("Running workflow ingestion for upload %s", candidate)
                uploaded_results.append(self._ingest_path(candidate))

            if not self.is_initialized:
                logger.error("Attempted to query RAG pipeline before initialisation")
                raise RuntimeError("RAG pipeline is not initialised")

            try:
                response = pipeline.query(question, **kwargs)
            except Exception as exc:
                logger.exception("Pipeline query failed: %s", exc)
                raise RuntimeError("Failed to process the question") from exc

            return {
                "question": question,
                "base_ingestion": base_result,
                "uploaded_ingestions": uploaded_results,
                "response": response,
            }

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
                    "summaries": int(getattr(last_result, "summaries_created", 0)),
                }
            return {
                "is_initialized": self.is_initialized,
                "config": config_state,
                "components": pipeline_details,
                "last_ingestion": ingestion_snapshot,
            }


