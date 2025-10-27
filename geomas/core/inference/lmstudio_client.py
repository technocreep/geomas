from __future__ import annotations

import logging
from typing import Mapping, Sequence

import requests

from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_core.documents import Document

from geomas.core.rag_modules.database.chroma_db import Embeddings

logger = logging.getLogger(__name__)


class LmStudioClient:
    """Thin HTTP client targeting an LM Studio OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("LM Studio base_url must be provided")
        if not model:
            raise ValueError("LM Studio model must be provided")

        self._endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self._model = model
        self._headers = {str(key): str(value) for key, value in dict(headers or {}).items()}
        self._timeout = timeout

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float,
    ) -> str:
        """Send ``messages`` to the LM Studio endpoint and return the response."""

        payload = {
            "model": self._model,
            "messages": [dict(message) for message in messages],
            "temperature": temperature,
        }

        try:
            response = requests.post(
                self._endpoint,
                json=payload,
                headers=self._headers or None,
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to reach LM Studio endpoint: {exc}") from exc

        if response.status_code >= 400:
            snippet = response.text.strip()
            raise RuntimeError(
                "LM Studio request failed with status "
                f"{response.status_code}: {snippet or response.reason}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError("LM Studio response was not valid JSON") from exc

        choices = data.get("choices")
        if not isinstance(choices, Sequence) or not choices:
            raise RuntimeError("LM Studio response did not include any choices")

        first_choice = choices[0]
        if isinstance(first_choice, Mapping):
            message_payload = first_choice.get("message")
            if isinstance(message_payload, Mapping):
                content = message_payload.get("content")
            else:
                content = first_choice.get("text")
        else:
            content = None

        if not content:
            raise RuntimeError("LM Studio response was missing completion content")

        return str(content)