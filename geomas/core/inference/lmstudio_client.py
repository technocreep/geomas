from __future__ import annotations

import logging
from typing import Any, Mapping, MutableMapping, Sequence

import requests

from geomas.core.inference.chat_utils import (
    SUPPORTED_CHAT_ROLES,
    normalise_chat_messages,
)

logger = logging.getLogger(__name__)


class LmStudioClient:
    """OpenAI-compatible client for LM Studio chat and embedding endpoints.

    The client mirrors the :class:`OllamaClient` interface, performing basic
    message validation and surfacing network or payload errors as runtime
    exceptions. Both chat completions and embeddings are supported via the
    OpenAI-compatible routes exposed by LM Studio (``/chat/completions`` and
    ``/embeddings``).

    Args:
        base_url: Root URL of the LM Studio server (e.g. ``http://localhost:1234``).
        model: Model identifier exposed by the LM Studio instance.
        headers: Optional HTTP headers to include with every request. Use this to
            pass custom authentication or tracing metadata.
        api_key: Optional bearer token injected as an ``Authorization`` header
            when supplied.
        timeout: Optional request timeout (in seconds) applied to chat and
            embedding calls.
        session: Optional ``requests.Session`` used for HTTP dispatch. When not
            provided, a fresh session is created.
    """

    _SUPPORTED_ROLES = SUPPORTED_CHAT_ROLES

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        headers: Mapping[str, str] | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        session: requests.Session | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("LM Studio base_url must be provided")
        if not model:
            raise ValueError("LM Studio model must be provided")

        self._base_url = base_url.rstrip("/")
        self._chat_url = f"{self._base_url}/chat/completions"
        self._embeddings_url = f"{self._base_url}/embeddings"
        self._model = model
        self._timeout = timeout

        prepared_headers = {str(key): str(value) for key, value in dict(headers or {}).items()}
        if api_key:
            prepared_headers.setdefault("Authorization", f"Bearer {api_key}")
        self._headers = prepared_headers or None
        self._session = session or requests.Session()

    @classmethod
    def _normalise_messages(
        cls, messages: Sequence[Mapping[str, object]]
    ) -> list[MutableMapping[str, object]]:
        return normalise_chat_messages(
            messages,
            provider_name="LM Studio",
            supported_roles=cls._SUPPORTED_ROLES,
        )

    def _post(self, url: str, *, payload: Mapping[str, object]) -> dict[str, Any]:
        try:
            response = self._session.post(
                url,
                json=payload,
                headers=self._headers,
                timeout=self._timeout,
            )
        except Exception as exc:  # pragma: no cover - network error path
            raise RuntimeError(f"LM Studio request failed: {exc}") from exc

        if not response.ok:
            raise RuntimeError(
                f"LM Studio request failed with status {response.status_code}: {response.reason}"
            )

        try:
            return response.json()
        except ValueError as exc:  # pragma: no cover - malformed response path
            raise RuntimeError("LM Studio response was not valid JSON") from exc

    def generate(
        self,
        messages: Sequence[Mapping[str, object]],
        *,
        temperature: float,
    ) -> str:
        """Send ``messages`` to the LM Studio chat endpoint and return the content."""

        payload = {
            "model": self._model,
            "messages": self._normalise_messages(messages),
            "temperature": float(temperature),
        }

        data = self._post(self._chat_url, payload=payload)

        choices = data.get("choices")
        if not isinstance(choices, Sequence) or not choices:
            raise RuntimeError("LM Studio response was missing completion choices")

        first_choice = choices[0]
        content: object | None = None
        if isinstance(first_choice, Mapping):
            message_payload = first_choice.get("message")
            if isinstance(message_payload, Mapping):
                content = message_payload.get("content")
            if content is None:
                content = first_choice.get("text")

        if content is None:
            raise RuntimeError("LM Studio response was missing completion content")

        return str(content)

    def embed(self, inputs: Sequence[str] | str) -> list[list[float]]:
        """Request embeddings for the provided ``inputs`` from LM Studio.

        Returns:
            A list of embedding vectors in the same order as the supplied inputs.
        """

        if isinstance(inputs, str):
            input_values = [inputs]
        else:
            input_values = [str(value) for value in inputs]
        if not input_values:
            raise ValueError("LM Studio embeddings require at least one input")

        payload = {"model": self._model, "input": input_values}
        data = self._post(self._embeddings_url, payload=payload)

        entries = data.get("data")
        if not isinstance(entries, Sequence) or not entries:
            raise RuntimeError("LM Studio embeddings response was missing data entries")

        vectors: list[list[float]] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise RuntimeError("LM Studio embeddings response was malformed")
            embedding = entry.get("embedding")
            if not isinstance(embedding, Sequence):
                raise RuntimeError("LM Studio embedding payload was not a sequence")
            vectors.append([float(value) for value in embedding])

        return vectors
