import logging
from typing import Optional

import chromadb
from langchain_community.embeddings.huggingface_hub import HuggingFaceHubEmbeddings
from langchain_community.vectorstores.chroma import Chroma

from geomas.core.data.data_transformation import DataExtraction
from geomas.core.repository.database_repository import (
    ChromaSettings,
    chroma_default_settings,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_documents_to_chroma_db(settings: Optional[ChromaSettings] = chroma_default_settings,
                                processing_batch_size: int = 100,
                                loading_batch_size: int = 32,
                                **kwargs) -> None:

    logger.info(
        f'Initializing batch generator with processing_batch_size: {processing_batch_size},'
        f' loading_batch_size: {loading_batch_size}'
    )

    pipeline_settings = PipelineSettings.config_from_file(settings.docs_processing_config)

    store = "./"
    # Documents loading and processing
    DataExtraction(pipeline_settings) \
        .go_to_next_step(docs_collection_path=settings.docs_collection_path) \
        .update_docs_transformers(**kwargs) \
        .go_to_next_step(batch_size=processing_batch_size) \
        .load(store, loading_batch_size=loading_batch_size)
