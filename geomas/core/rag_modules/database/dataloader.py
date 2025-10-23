from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from geomas.core.rag_modules.data_adapter import DataLoaderAdapter
from geomas.core.rag_modules.database.chroma_db import ChromaDatabaseStore, DatabaseRagPipeline
from geomas.core.rag_modules.parser import PARSER_DEPENDENCY_ERROR
from geomas.core.repository.database_repository import ChromaSettings, chroma_default_settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_documents_to_chroma_db(
    settings: Optional[ChromaSettings] = chroma_default_settings,
    *,
    parser: DocumentParser | None = None,
    store: ChromaDatabaseStore | None = None,
) -> None:
    """Ingest documents from ``settings.docs_collection_path`` into ChromaDB."""

    if settings is None:
        raise ValueError("Chroma settings must be provided for document ingestion")

    if parser is None:
        try:
            from geomas.core.rag_modules.parser.rag_parser import DocumentParser
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(PARSER_DEPENDENCY_ERROR) from exc

        try:
            parser = DocumentParser()
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(PARSER_DEPENDENCY_ERROR) from exc

    if store is None:
        raise ValueError(
            "ChromaDatabaseStore instance must be provided to ingest documents",
        )

    adapter = DataLoaderAdapter(parser=parser)
    pipeline = DatabaseRagPipeline(store=store, parser=parser, data_loader=adapter)

    target_path = Path(settings.docs_collection_path).expanduser()
    result = pipeline.process(target_path)

    if not result.success:
        logger.info("No documents were ingested from '%s'", target_path)
    else:
        logger.info(
            "Ingested %s documents from '%s'", result.documents_ingested, target_path
        )
