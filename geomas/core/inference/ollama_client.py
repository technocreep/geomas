from __future__ import annotations

import importlib.util
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence

import ollama

from dotenv import load_dotenv

from geomas.core.inference.chat_utils import (
    SUPPORTED_CHAT_ROLES,
    normalise_chat_messages,
)
from geomas.core.repository.rag_repository import RAGConfig, _deep_update

logger = logging.getLogger(__name__)


class OllamaClient:
    """Chat-oriented client compatible with :class:`StandardRAGPipeline`."""
    _SUPPORTED_ROLES = SUPPORTED_CHAT_ROLES

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
        return normalise_chat_messages(
            messages,
            provider_name="Ollama",
            supported_roles=cls._SUPPORTED_ROLES,
        )

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
    *,
    chat_id: str | None = None,
    global_rag_dir: Path,
    local_rag_dir: Path | None = None,
    settings: OllamaSettings | None = None,
    embedding_model_name: str = "artefucktor/LaBSE_geonames_RU",
    embedding_model_kwargs: dict[str, Any] = {"device": "cuda", "trust_remote_code": True},
) -> RAGConfig:
    resolved_settings = settings or load_ollama_settings()

    if chat_id is not None:
        collection_name = f"{chat_id}_local"
        rag_dir = local_rag_dir
    else:
        collection_name = "global"
        rag_dir = global_rag_dir

    overrides: Dict[str, Any] = {
        "parsing": {"enable_parser": False},
        "database": {
            "client_mode": "persistent",
            "persist_directory": str(rag_dir),
            "collection_name": collection_name,
        },
        "retrieval":  {
            "top_k": 5,
            "embedding_model_name": embedding_model_name,
            "embedding_model_kwargs": embedding_model_kwargs,
        },
        "ranking": {
            "use_llm_reranking": False,
            "chroma": {"enabled": True},
        },
        "vector_store": {
            "client": {
                "persist_directory": str(rag_dir),
            }
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
    base_config = RAGConfig.default().to_dict()
    _deep_update(base_config, overrides)
    return RAGConfig.from_mapping(base_config)
