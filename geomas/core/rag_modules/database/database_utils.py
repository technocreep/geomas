"""Utility helpers for the Chroma database integration."""

from __future__ import annotations

from typing import Optional, Sequence

import logging
import os
from pathlib import Path

import chromadb
from chromadb import EmbeddingFunction
from chromadb.api.models import Collection
from pydantic import BaseModel, Field

from geomas.core.repository.constant_repository import load_repository_env
from geomas.core.repository.database_repository import (
    DATABASE_HOST,
    DATABASE_PORT,
    RESET_DATABASE,
)


logger = logging.getLogger(__name__)


load_repository_env()


class ExpandedSummary(BaseModel):
    """Structured representation of a generated paper summary."""

    paper_summary: str = Field(description="Summary of the paper.")
    paper_title: str = Field(
        description=(
            "Title of the paper. If the title is not explicitly specified, "
            "use the default value - 'NO TITLE'"
        )
    )
    publication_year: int = Field(
        description=(
            "Year of publication of the paper. If the publication year is "
            "not explicitly specified, use the default value - 9999."
        )
    )


class ChromaDatabaseClient:
    """Light-weight façade over the Chroma client implementations."""

    def __init__(
        self,
        *,
        host: str | None = DATABASE_HOST,
        port: int | str | None = DATABASE_PORT,
        allow_reset: bool | None = RESET_DATABASE,
        mode: str | None = None,
        persistent_path: str | os.PathLike[str] | None = None,
    ) -> None:
        """Initialise the Chroma client with HTTP or persistent storage.

        Parameters
        ----------
        host:
            Explicit host name for the Chroma HTTP client. Defaults to
            :data:`DATABASE_HOST` or the ``CHROMA_HOST`` environment variable.
        port:
            Port number used by the Chroma HTTP client. Accepts integers or
            numeric strings and defaults to :data:`DATABASE_PORT` or
            ``CHROMA_PORT``.
        allow_reset:
            When ``True`` the Chroma client may accept destructive reset
            operations. Defaults to :data:`RESET_DATABASE` or the
            ``CHROMA_ALLOW_RESET`` environment variable.
        mode:
            Optional override for selecting ``"http"`` or ``"persistent"``
            client implementations. When omitted the mode is inferred from the
            provided host and port values.
        persistent_path:
            Filesystem path to use for persistent Chroma deployments. Falls
            back to ``CHROMA_PERSIST_PATH`` or a cache directory under the
            user's home when not supplied.
        """

        env_host = host if host is not None else os.getenv("CHROMA_HOST")
        raw_port: int | str | None = port if port is not None else os.getenv("CHROMA_PORT")
        env_port = self._normalise_port(raw_port)

        explicit_mode = mode or os.getenv("CHROMA_CLIENT_MODE")
        if explicit_mode:
            resolved_mode = explicit_mode.strip().lower()
        else:
            resolved_mode = "http" if env_host and env_port is not None else "persistent"

        if resolved_mode not in {"http", "persistent"}:
            raise ValueError(f"Unsupported CHROMA_CLIENT_MODE '{resolved_mode}'")

        reset_flag = allow_reset
        if reset_flag is None:
            reset_env = os.getenv("CHROMA_ALLOW_RESET")
            reset_flag = reset_env.lower() == "true" if reset_env else False

        self._mode = resolved_mode
        self._closed = False
        self._allow_reset = bool(reset_flag)

        if resolved_mode == "http":
            if not env_host or env_port is None:
                raise RuntimeError(
                    "CHROMA_CLIENT_MODE=http requires both CHROMA_HOST and CHROMA_PORT to be defined"
                )
            self.client = chromadb.HttpClient(
                host=env_host,
                port=env_port,
                settings=chromadb.Settings(allow_reset=self._allow_reset),
            )
            return

        storage_root = self._resolve_persistent_path(persistent_path)
        persistent_cls = getattr(chromadb, "PersistentClient", None)
        if persistent_cls is not None:
            self.client = persistent_cls(path=str(storage_root))
            return

        client_cls = getattr(chromadb, "Client", None)
        if client_cls is None:
            raise RuntimeError("Chroma persistent client is unavailable in the current installation")

        settings = None
        if hasattr(chromadb, "Settings"):
            settings = chromadb.Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=str(storage_root),
            )
            self.client = client_cls(settings=settings)
        else:
            self.client = client_cls(persist_directory=str(storage_root))

    @staticmethod
    def _call_shutdown(target: object | None) -> bool:
        """Invoke a supported shutdown hook on ``target`` when available."""

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

    @staticmethod
    def _duckdb_connection(candidate: object) -> object | None:
        """Return a DuckDB connection object from ``candidate`` when available."""

        for attr in (
            "duckdb_connection",
            "_duckdb_connection",
            "connection",
            "_connection",
            "conn",
            "_conn",
        ):
            connection = getattr(candidate, attr, None)
            if connection is not None:
                return connection

        backend = getattr(candidate, "_db", None) or getattr(candidate, "db", None)
        if backend is not None:
            for attr in ("connection", "_connection", "conn", "_conn"):
                connection = getattr(backend, attr, None)
                if connection is not None:
                    return connection
        return None

    @staticmethod
    def _normalise_port(raw_port: int | str | None) -> int | None:
        """Convert ``raw_port`` into an integer when possible."""
        if raw_port is None:
            return None
        if isinstance(raw_port, int):
            return raw_port
        candidate = str(raw_port).strip()
        if not candidate:
            return None
        if not candidate.isdigit():
            raise ValueError(f"Invalid port value '{raw_port}'")
        return int(candidate)

    @staticmethod
    def _resolve_persistent_path(
        path_hint: str | os.PathLike[str] | None,
    ) -> Path:
        """Return an absolute path suitable for initialising persistent Chroma."""
        raw_path = path_hint if path_hint is not None else os.getenv("CHROMA_PERSIST_PATH")
        base_path = Path(raw_path) if raw_path else Path.home() / ".cache" / "geomas" / "chroma"
        base_path.mkdir(parents=True, exist_ok=True)
        return base_path

    def list_collections(self) -> list[Collection]:
        """Return all collections available on the remote instance."""

        return self.client.list_collections()

    def delete_collection(self, name: str) -> None:
        """Remove the collection identified by ``name`` if it exists."""

        self.client.delete_collection(name)

    def ensure_collection(
        self,
        name: str,
        embedding_function: Optional[EmbeddingFunction] = None,
    ) -> Collection:
        """Return an existing collection or create it when missing."""

        return self.client.get_or_create_collection(
            name=name,
            embedding_function=embedding_function,
        )

    @staticmethod
    def query_chromadb(
        collection: Collection,
        *,
        query_text: str | None = None,
        query_embeddings: Optional[Sequence[Sequence[float]]] = None,
        metadata_filter: Optional[dict] = None,
        chunk_num: int = 3,
    ) -> dict:
        """Execute a semantic query against ``collection``."""

        if query_text is None and not query_embeddings:
            raise ValueError("Either query_text or query_embeddings must be provided")

        payload: dict[str, object] = {
            "n_results": chunk_num,
            "where": metadata_filter,
            "include": ["documents", "metadatas", "distances"],
        }

        if query_embeddings is not None:
            payload["query_embeddings"] = list(query_embeddings)
        else:
            payload["query_texts"] = [query_text]

        return collection.query(**payload)

    def close(self) -> None:
        """Release resources held by the underlying Chroma client."""

        if getattr(self, "_closed", False):
            return

        client = getattr(self, "client", None)
        if client is None:
            self._closed = True
            return

        closed = self._call_shutdown(client)

        if not closed and self._mode == "persistent":
            duckdb_connection = self._duckdb_connection(client)
            if duckdb_connection is not None:
                closed = self._call_shutdown(duckdb_connection)

        if not closed and getattr(self, "_allow_reset", False):
            reset_method = getattr(client, "reset", None)
            if callable(reset_method):
                try:
                    reset_method()
                except Exception as exc:
                    logger.debug("Failed to reset Chroma client %s: %s", type(client).__name__, exc)

        self.client = None
        self._closed = True

