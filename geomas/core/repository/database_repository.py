import os

from pydantic_settings import BaseSettings


class ChromaSettings(BaseSettings):
    """
    Manages settings for Chroma database and related components.

        This class encapsulates configuration details for connecting to and interacting with
        Chroma, an embedding database, as well as related embedding and reranking services.

        Class Attributes:
        - chroma_host
        - chroma_port
        - allow_reset
        - embedding_host
        - embedding_port
        - embedding_endpoint
        - reranker_host
        - reranker_port
        - reranker_endpoint
    """

    # Chroma DB settings
    chroma_host: str = os.getenv("CHROMA_HOST", "localhost")
    chroma_port: int = int(os.getenv("CHROMA_PORT", "8000"))
    allow_reset: bool = False

    # Documents collection's settings
    embedding_host: str = os.getenv("EMBEDDING_HOST", "localhost")
    embedding_port: int = int(os.getenv("EMBEDDING_PORT", "8001"))
    embedding_endpoint: str = "/embed"

    # Reranker settings
    reranker_host: str = os.getenv("RERANKER_HOST", "localhost")
    reranker_port: int = int(os.getenv("RERANKER_PORT", "8002"))
    reranker_endpoint: str = "/rerank"


chroma_default_settings = ChromaSettings()
DATABASE_HOST = chroma_default_settings .chroma_host
DATABASE_PORT = chroma_default_settings .chroma_port
RESET_DATABASE = chroma_default_settings .allow_reset