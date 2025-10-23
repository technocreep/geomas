from __future__ import annotations

import os
import re
from copy import deepcopy
from typing import Any, Dict, Mapping, MutableMapping

from langchain_core.documents import Document
from langchain_text_splitters import HTMLSemanticPreservingSplitter, MarkdownTextSplitter

from geomas.core.repository.constant_repository import ROOT_DIR, USE_S3
from geomas.core.repository.parsing_repository import (
    ChunkingParamsConfig,
    ParsingPatternConfig,
)

# Get PARSE_RESULTS_PATH from environment or use default
PARSE_RESULTS_PATH = os.path.join(
    ROOT_DIR, os.environ.get("PARSE_RESULTS_PATH", "./parse_results")
)


class TextChunker:
    """Chunk raw text into LangChain documents while preserving legacy metadata."""

    _SUPPORTED_TYPES: frozenset[str] = frozenset({"html", "markdown"})

    def __init__(self, chunking_params: Mapping[str, Any] | None = None):
        self.splitter_dict = {
            "html": HTMLSemanticPreservingSplitter,
            "markdown": MarkdownTextSplitter,
        }
        raw_params: Mapping[str, Any]
        if chunking_params is not None:
            raw_params = dict(chunking_params)
        else:
            raw_params = ChunkingParamsConfig.default_chunking_parameters()

        self._raw_chunking_params = self._clone_mapping(raw_params)
        self.chunking_params = self._normalise_chunking_params(raw_params)

    def _custom_table_extractor(self, table_tag: Any) -> str:
        return str(table_tag).replace("\n", "")

    @staticmethod
    def _clone_mapping(value: Mapping[str, Any]) -> Dict[str, Any]:
        return {key: deepcopy(val) for key, val in value.items()}

    @classmethod
    def _normalise_chunking_params(
        cls, params: Mapping[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Return structured chunking parameters split by supported type.

        The legacy configuration exposed a flat mapping with shared keys.  New
        configurations may provide ``defaults`` along with ``html`` and
        ``markdown`` overrides.  This helper merges both representations into a
        stable ``{"defaults": ..., "html": ..., "markdown": ...}`` structure.
        """

        if not isinstance(params, Mapping):
            return {"defaults": {}, "html": {}, "markdown": {}}

        raw: Dict[str, Any] = dict(params)
        defaults: Dict[str, Any] = {}
        html: Dict[str, Any] = {}
        markdown: Dict[str, Any] = {}

        common_keys = {"chunk_size", "max_chunk_size", "chunk_overlap"}
        html_keys = common_keys | {
            "headers_to_split_on",
            "elements_to_preserve",
            "preserve_images",
        }
        markdown_keys = common_keys | {"separators"}

        for alias in ("defaults", "default", "common"):
            source = raw.get(alias)
            if isinstance(source, Mapping):
                defaults.update(
                    {k: deepcopy(v) for k, v in source.items() if k in common_keys}
                )

        if not defaults:
            defaults.update(
                {k: deepcopy(raw[k]) for k in common_keys if k in raw}
            )

        html_source = raw.get("html")
        if isinstance(html_source, Mapping):
            html.update(
                {k: deepcopy(v) for k, v in html_source.items() if k in html_keys}
            )

        if not html:
            html.update(
                {k: deepcopy(raw[k]) for k in html_keys if k in raw and k not in defaults}
            )

        markdown_source = raw.get("markdown")
        if isinstance(markdown_source, Mapping):
            markdown.update(
                {
                    k: deepcopy(v)
                    for k, v in markdown_source.items()
                    if k in markdown_keys
                }
            )

        if not markdown:
            markdown.update(
                {
                    k: deepcopy(raw[k])
                    for k in markdown_keys
                    if k in raw and k not in defaults
                }
            )

        def _ensure_size_alias(target: MutableMapping[str, Any]) -> None:
            if "chunk_size" in target and "max_chunk_size" not in target:
                target["max_chunk_size"] = target["chunk_size"]
            if "max_chunk_size" in target and "chunk_size" not in target:
                target["chunk_size"] = target["max_chunk_size"]

        for bucket in (defaults, html, markdown):
            _ensure_size_alias(bucket)

        return {"defaults": defaults, "html": html, "markdown": markdown}

    def _prepare_splitter_kwargs(
        self, document_type: str
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        defaults = self._clone_mapping(self.chunking_params.get("defaults", {}))
        overrides = self._clone_mapping(self.chunking_params.get(document_type, {}))
        kwargs = {**defaults, **overrides}
        post_inits: Dict[str, Any] = {}

        if document_type == "html":
            chunk_size = kwargs.pop("chunk_size", None)
            if "max_chunk_size" not in kwargs and chunk_size is not None:
                kwargs["max_chunk_size"] = chunk_size
        elif document_type == "markdown":
            separators = kwargs.pop("separators", None)
            if separators is not None:
                post_inits["separators"] = separators
            kwargs.pop("max_chunk_size", None)

        return kwargs, post_inits

    def extract_img_url(self, doc_text: str, p_name: str) -> list[str]:
        """
        Extracts image URLs from a document text related to scientific papers.

        This method identifies image references within the text and constructs their full paths for access.
        It focuses on JPEG images specifically referenced using a specific markdown-like syntax.

        Args:
            doc_text: The text of the scientific document to analyze.
            p_name: The name of the project or paper, used to organize image paths.

        Returns:
            list: A list of strings, where each string is a full path to an image extracted from the document,
            constructed using the project path and the image filename.
        """
        matches = re.findall(ParsingPatternConfig.image_pattern, doc_text)
        return [entry[0] for entry in matches] if USE_S3 \
            else [os.path.join(PARSE_RESULTS_PATH, p_name, entry[0]) for entry in matches]

    def apply_chunking(
        self, raw_text: str, document_name: str, document_type: str
    ):
        if document_type not in self._SUPPORTED_TYPES:
            raise ValueError(f"Unsupported document type: {document_type}")

        splitter_cls = self.splitter_dict[document_type]
        kwargs, post_inits = self._prepare_splitter_kwargs(document_type)

        if document_type == "html":
            kwargs["custom_handlers"] = {"table": self._custom_table_extractor}

        splitter = splitter_cls(**kwargs)

        if document_type == "markdown" and "separators" in post_inits:
            separators = list(post_inits["separators"])
            setattr(splitter, "_separators", separators)
            setattr(splitter, "separators", separators)

        documents = splitter.split_text(raw_text)
        if documents and not hasattr(documents[0], "page_content"):
            documents = [Document(page_content=chunk, metadata={}) for chunk in documents]
        for doc in documents:
            doc.page_content = "passage: " + doc.page_content  # Maybe delete "passage: " addition
            doc.metadata["imgs_in_chunk"] = str(self.extract_img_url(doc.page_content, document_name))
            doc.metadata["source"] = document_name + ".pdf"
        return documents
