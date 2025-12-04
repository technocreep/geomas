"""Shared helpers for LangChain-compatible chat clients.

The Ollama and LM Studio clients both accept OpenAI-style message payloads with
``role``/``content`` keys. The utilities below keep their validation logic in
one place so the parallel connectors stay aligned and future providers can
reuse the same guard rails without duplicating code.
"""

from __future__ import annotations

from typing import Mapping, MutableMapping, Sequence

SUPPORTED_CHAT_ROLES: frozenset[str] = frozenset({"system", "user", "assistant"})


def normalise_chat_messages(
    messages: Sequence[Mapping[str, object]],
    *,
    provider_name: str,
    supported_roles: frozenset[str] = SUPPORTED_CHAT_ROLES,
) -> list[MutableMapping[str, object]]:
    """Validate and normalise chat ``messages`` for downstream clients.

    Args:
        messages: Sequence of mappings containing ``role`` and ``content`` keys.
        provider_name: Provider identifier used in validation error messages.
        supported_roles: Allowed role values; defaults to ``SUPPORTED_CHAT_ROLES``.

    Returns:
        A list of mutable mappings with lowercase ``role`` values and string
        ``content`` entries suitable for OpenAI-style chat APIs.

    Raises:
        TypeError: If any message payload is not a mapping.
        ValueError: If a message is missing a role/content or uses an
            unsupported role name.
    """

    normalised: list[MutableMapping[str, object]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TypeError(
                f"{provider_name} messages must be mappings with 'role' and 'content' keys"
            )

        role_raw = message.get("role")
        if not isinstance(role_raw, str):
            raise ValueError(
                f"Message at position {index} is missing a textual role"
            )

        role = role_raw.strip().lower()
        if role not in supported_roles:
            raise ValueError(
                f"{provider_name} messages must contain a role from 'system', 'user', or 'assistant'"
            )

        content = message.get("content")
        if content is None:
            raise ValueError(
                f"Message at position {index} is missing completion content"
            )

        normalised.append({"role": role, "content": str(content)})

    return normalised
