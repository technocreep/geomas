from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from geomas.core.repository.constant_repository import load_repository_env


load_repository_env()


class ChromaSettings(BaseSettings):
    """Manages settings for Chroma database and related services."""

    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False, extra="ignore")

    chroma_host: str | None = Field(
        default=None,
        validation_alias=AliasChoices("CHROMA_HOST"),
    )
    chroma_port: int | None = Field(
        default=None,
        validation_alias=AliasChoices("CHROMA_PORT"),
    )
    allow_reset: bool = Field(
        default=False,
        validation_alias=AliasChoices("CHROMA_ALLOW_RESET"),
    )

    embedding_host: str | None = Field(
        default=None,
        validation_alias=AliasChoices("EMBEDDING_HOST"),
    )
    embedding_port: int | None = Field(
        default=None,
        validation_alias=AliasChoices("EMBEDDING_PORT"),
    )
    embedding_endpoint: str = "/embed"

    reranker_host: str | None = Field(
        default=None,
        validation_alias=AliasChoices("RERANKER_HOST"),
    )
    reranker_port: int | None = Field(
        default=None,
        validation_alias=AliasChoices("RERANKER_PORT"),
    )
    reranker_endpoint: str = "/rerank"


chroma_default_settings = ChromaSettings()
DATABASE_HOST = chroma_default_settings.chroma_host
DATABASE_PORT = chroma_default_settings.chroma_port
RESET_DATABASE = chroma_default_settings.allow_reset
