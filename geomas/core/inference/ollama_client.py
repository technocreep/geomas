from __future__ import annotations

import importlib.util
from typing import Mapping, MutableMapping, Sequence


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

        import ollama

        client_kwargs: dict[str, object] = {}
        if host:
            client_kwargs["host"] = host
        if timeout is not None:
            client_kwargs["timeout"] = timeout

        try:
            self._client = ollama.Client(**client_kwargs)
        except Exception as exc:  # pragma: no cover - defensive for httpx errors
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

