from __future__ import annotations

import importlib.util
import logging
import os
import ollama
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence

from dotenv import load_dotenv

from geomas.core.repository.rag_repository import RAGConfig, _deep_update

logger = logging.getLogger(__name__)


class OllamaClient:
    """Chat-oriented client compatible with :class:`StandardRAGPipeline`."""
    _SUPPORTED_ROLES = {"system", "user", "assistant"}

    def __init__(
        self,
        *,
        model: str,
        host: str | None = None,
        timeout: float | None = None,
    ) -> None:
        if not model:
            raise ValueError("Ollama model must be provided")

        spec = importlib.util.find_spec("ollama")
        if spec is None:
            raise RuntimeError(
                "Ollama inference requires the optional 'ollama' package to be installed"
            )

        client_kwargs: dict[str, object] = {}
        if host:
            client_kwargs["host"] = host
        if timeout is not None:
            client_kwargs["timeout"] = timeout

        try:
            self._client = ollama.Client(**client_kwargs)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialise Ollama client: {exc}") from exc
        self._model = model

    @classmethod
    def _normalise_messages(
        cls, messages: Sequence[Mapping[str, object]]
    ) -> list[MutableMapping[str, object]]:
        normalised: list[MutableMapping[str, object]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise TypeError(
                    "Ollama messages must be mappings with 'role' and 'content' keys"
                )

            role_raw = message.get("role")
            if not isinstance(role_raw, str):
                raise ValueError(
                    f"Message at position {index} is missing a textual role"
                )

            role = role_raw.strip().lower()
            if role not in cls._SUPPORTED_ROLES:
                raise ValueError(
                    "Ollama messages must contain a role from 'system', 'user', or 'assistant'"
                )

            content = message.get("content")
            if content is None:
                raise ValueError(
                    f"Message at position {index} is missing completion content"
                )

            normalised.append(
                {
                    "role": role,
                    "content": str(content),
                }
            )
        return normalised

    def generate(
        self,
        messages: Sequence[Mapping[str, object]],
        *,
        temperature: float,
    ) -> str:
        """Send ``messages`` to an Ollama chat endpoint and return the response."""
        payload = self._normalise_messages(messages)
        options = {"temperature": float(temperature)}

        try:
            response = self._client.chat(
                model=self._model,
                messages=payload,
                options=options,
            )
        except Exception as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        message = response.get("message")
        content = message.get("content")

        if not content:
            raise RuntimeError("Ollama response was missing completion content")

        return str(content)


@dataclass(slots=True, frozen=True)
class OllamaSettings:
    """Configuration payload describing how to reach and tune an Ollama model."""
    base_url: str | None
    model: str
    temperature: float = 0.0
    timeout: float | None = None
    system_prompt: str | None = None

    def with_overrides(self, overrides: Mapping[str, object]) -> "OllamaSettings":
        """Return a new instance with ``overrides`` applied to supported fields."""
        if not overrides:
            return self

        valid_fields = {
            "base_url",
            "model",
            "temperature",
            "timeout",
            "system_prompt",
        }

        unknown = sorted(set(overrides) - valid_fields)
        if unknown:
            raise ValueError(f"Unsupported Ollama settings: {', '.join(unknown)}")

        data = {
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "system_prompt": self.system_prompt,
        }

        for key, value in overrides.items():
            if key == "temperature":
                data[key] = float(value)
            elif key == "timeout":
                data[key] = None if value is None else float(value)
            elif key == "base_url":
                data[key] = None if value in {None, ""} else str(value)
            elif key == "model":
                if not value:
                    raise ValueError("Ollama model must be a non-empty string")
                data[key] = str(value)
            elif key == "system_prompt":
                data[key] = None if value in {None, ""} else str(value)

        return OllamaSettings(**data)

    def to_inference_params(self) -> Dict[str, object]:
        """Materialise the settings as parameters for ``RAGConfig.inference``."""
        params: Dict[str, object] = {
            "provider": "ollama",
            "model": self.model,
            "temperature": float(self.temperature),
        }
        if self.base_url:
            params["host"] = self.base_url
        if self.timeout is not None:
            params["timeout"] = float(self.timeout)
        if self.system_prompt:
            params["system_prompt"] = self.system_prompt
        return params


def _read_float_env(name: str, *, default: float, env: Mapping[str, str]) -> float:
    value = env.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be numeric") from exc


def _read_optional_float_env(name: str, *, env: Mapping[str, str]) -> float | None:
    value = env.get(name)
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be numeric") from exc


def load_ollama_settings(
    *,
    use_dotenv: bool = True,
    environ: Mapping[str, str] | None = None,
) -> OllamaSettings:
    """Load :class:`OllamaSettings` from environment variables."""
    if use_dotenv:
        try:
            load_dotenv()
        except Exception as exc:
            logger.debug("Failed to load .env file: %s", exc)

    env = dict(os.environ)
    if environ is not None:
        env.update(environ)

    base_url = env.get("OLLAMA_URL") or env.get("OLLAMA_BASE_URL")
    if not base_url:
        host = env.get("OLLAMA_HOST")
        port = env.get("OLLAMA_PORT")
        if host and port:
            base_url = f"http://{host}:{port}"

    model = env.get("OLLAMA_MODEL", "granite3.2")
    if not model:
        raise RuntimeError("OLLAMA_MODEL must be set to a non-empty string")

    temperature = _read_float_env("OLLAMA_TEMPERATURE", default=0.0, env=env)
    timeout = _read_optional_float_env("OLLAMA_TIMEOUT", env=env)
    system_prompt = env.get("OLLAMA_SYSTEM_PROMPT") or "Ответ должен быть на русском"

    return OllamaSettings(
        base_url=base_url,
        model=model,
        temperature=temperature,
        timeout=timeout,
        system_prompt=system_prompt,
    )


def build_ollama_rag_config(
    documents_dir: Path | str,
    *,
    cache_dir: Path | str | None = None,
    local_cache_dir: Path | str | None = None,
    settings: OllamaSettings | None = None,
) -> RAGConfig:
    """Create a :class:`RAGConfig` tuned for the Ollama integration.

    Args:
        documents_dir: Directory containing the base document collection.
        cache_dir: Optional path controlling the persistent Chroma cache used by
            the primary database client. When omitted the cache resides within
            ``documents_dir``.
        local_cache_dir: Optional path enabling a secondary local cache. When
            provided additional configuration keys are emitted to describe the
            local collection name and vector-store client path while preserving
            backwards compatible defaults when the argument is ``None``.
        settings: Optional :class:`OllamaSettings` instance describing the model
            connection parameters. When omitted the settings are loaded from the
            environment.

    Returns:
        A fully initialised :class:`RAGConfig` instance ready for use with the
        Ollama workflow helpers.
    """
    resolved_settings = settings or load_ollama_settings()
    documents_path = Path(documents_dir)
    persistent_path = (
        Path(cache_dir) if cache_dir is not None else documents_path / ".vector-store"
    )
    local_persistent_path = Path(local_cache_dir) if local_cache_dir is not None else None

    overrides: Dict[str, Any] = {
        "parsing": {"enable_parser": False},
        "database": {
            "client_mode": "persistent",
            "persistent_path": str(persistent_path),
            "collection_name": "geomas",
        },
        "retrieval": {
            "top_k": 5,
            "text_top_k": 5,
            "embedding_model_name": "labse",
        },
        "ranking": {
            "use_llm_reranking": False,
            "chroma": {"enabled": True},
        },
        "inference": {
            "enable_remote_services": True,
            "provider": "ollama",
            "params": resolved_settings.to_inference_params(),
        },
        "integrations": {
            "enable_ollama": True,
            "ollama_endpoint": resolved_settings.base_url,
        },
    }

    if local_persistent_path is not None:
        database_overrides = overrides.setdefault("database", {})
        collection_name = str(database_overrides.get("collection_name", "geomas"))
        database_overrides["local_collection_name"] = f"{collection_name}_local"

        vector_store_overrides = overrides.setdefault("vector_store", {})
        vector_store_overrides["local_client"] = {
            "persistent_path": str(local_persistent_path),
        }

    base_config = RAGConfig.default().to_dict()
    _deep_update(base_config, overrides)
    return RAGConfig.from_mapping(base_config)


def run_ollama_workflow(
    question: str,
    *,
    documents_dir: Path | str,
    uploaded_documents: Sequence[Path | str] | None = None,
    cache_dir: Path | str | None = None,
    local_cache_dir: Path | str | None = None,
    settings: Mapping[str, object] | OllamaSettings | None = None,
) -> Dict[str, Any]:
    """Execute an Ollama-backed workflow using :class:`RagApi`.

    Args:
        question: Natural language query forwarded to the workflow.
        documents_dir: Base directory containing reference documents.
        uploaded_documents: Optional collection of additional artefacts to
            ingest before answering the question.
        cache_dir: Optional remote/persistent cache directory used for the
            shared vector store.
        local_cache_dir: Optional path configuring a local vector-store cache.
            When omitted the workflow matches previous behaviour and only the
            persistent cache is used.
        settings: Optional mapping or :class:`OllamaSettings` describing the
            target Ollama instance.
    """
    if not question:
        raise ValueError("Question must be a non-empty string")

    uploads = [Path(candidate) for candidate in (uploaded_documents or [])]
    base_settings = load_ollama_settings()

    if isinstance(settings, OllamaSettings):
        resolved_settings = settings
    elif isinstance(settings, Mapping):
        resolved_settings = base_settings.with_overrides(settings)
    elif settings is None:
        resolved_settings = base_settings
    else:
        raise TypeError("settings must be a mapping or OllamaSettings instance")

    documents_path = Path(documents_dir)
    cache_path: Path | str | None = cache_dir
    config = build_ollama_rag_config(
        documents_path,
        cache_dir=cache_path,
        local_cache_dir=local_cache_dir,
        settings=resolved_settings,
    )

    from geomas.core.rag_modules import rag_pipeline

    override_pipeline = rag_pipeline.create_standard_pipeline(config)

    from geomas.api.rag import RagApi

    with RagApi(config=config) as api:
        previous_pipeline = api.pipeline
        api.pipeline = override_pipeline
        api.is_initialized = False

        query_kwargs = {"text_top_k": 4, "rerank_top_k": 3}

        try:
            workflow = api.run_workflow(
                question,
                documents_path=documents_path,
                uploaded_documents=uploads or None,
                query_kwargs=query_kwargs,
            )
        finally:
            if previous_pipeline is not None and previous_pipeline is not override_pipeline:
                try:
                    previous_pipeline.close()
                except Exception:
                    pass

        return {
            "question": question,
            "ingestion": workflow.get("base_ingestion"),
            "uploaded_ingestions": workflow.get("uploaded_ingestions", []),
            "response": workflow["response"],
        }

