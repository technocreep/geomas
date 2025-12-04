from __future__ import annotations

from collections.abc import Iterator
from json import load
from pathlib import Path
from typing import Any

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.load import load as langchain_load


class LangChainDocumentLoader(BaseLoader):
    """Load LangChain ``Document`` instances from a JSON file."""

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)

    def lazy_load(self) -> Iterator[Document]:
        with self.file_path.open("r", encoding="utf-8") as handle:
            payload = load(handle)

        for entry in self._normalise_payload(payload):
            yield entry

    def _normalise_payload(self, payload: Any) -> Iterator[Document]:
        if isinstance(payload, dict):
            values = payload.values()
        elif isinstance(payload, list):
            values = payload
        else:
            raise ValueError("Unsupported JSON payload for LangChainDocumentLoader")

        for item in values:
            document = self._coerce_document(item)
            if document is not None:
                yield document

    @staticmethod
    def _coerce_document(entry: Any) -> Document | None:
        if isinstance(entry, Document):
            return entry
        if isinstance(entry, dict) and "page_content" in entry:
            metadata = entry.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            return Document(page_content=entry["page_content"], metadata=metadata)
        try:
            loaded = langchain_load(entry)
        except Exception as exc:
            raise ValueError("Unsupported entry in LangChainDocumentLoader payload") from exc
        if isinstance(loaded, Document):
            return loaded
        raise ValueError("Unsupported entry in LangChainDocumentLoader payload")
