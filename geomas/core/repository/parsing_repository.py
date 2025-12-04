from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict

from geomas.core.repository.config_repository import ConfigTemplate


class DatabaseChunkingConfig(ConfigTemplate):
    """Params for finetune multimodal LLM using LoRa"""
    sum_chunk_num: bool = 15
    final_sum_chunk_num: bool = 3
    txt_chunk_num: bool = 15
    img_chunk_num: int = 2


class ChunkingParamsConfig(ConfigTemplate):
    """Default chunking options used by :class:`~chunking.TextChunker`.

    ``defaults`` captures values shared by all splitters while ``html`` and
    ``markdown`` record overrides specific to their respective document types.
    The legacy flat keys (e.g. ``max_chunk_size``) remain available so existing
    configuration files continue to work without changes.
    """

    max_chunk_size: int = 2500
    chunk_overlap: int = 200
    separators: list = ["\n\n", "\n", ". "]
    headers_to_split_on: list = [("h1", "Header 1"), ("h2", "Header 2")]
    elements_to_preserve: list = ["ul", "table", "ol"]
    preserve_images: bool = True

    defaults: dict = {
        "chunk_size": max_chunk_size,
        "max_chunk_size": max_chunk_size,
        "chunk_overlap": chunk_overlap,
    }

    html: dict = {
        "chunk_size": max_chunk_size,
        "max_chunk_size": max_chunk_size,
        "chunk_overlap": chunk_overlap,
        "headers_to_split_on": headers_to_split_on,
        "elements_to_preserve": elements_to_preserve,
        "preserve_images": preserve_images,
    }

    markdown: dict = {
        "chunk_size": max_chunk_size,
        "max_chunk_size": max_chunk_size,
        "chunk_overlap": chunk_overlap,
        "separators": separators,
    }

    @classmethod
    def default_chunking_parameters(cls) -> Dict[str, Any]:
        """Return the default chunking configuration with independent copies."""

        base: Dict[str, Any] = {
            "chunk_size": cls.max_chunk_size,
            "max_chunk_size": cls.max_chunk_size,
            "chunk_overlap": cls.chunk_overlap,
            "separators": deepcopy(cls.separators),
            "headers_to_split_on": deepcopy(cls.headers_to_split_on),
            "elements_to_preserve": deepcopy(cls.elements_to_preserve),
            "preserve_images": cls.preserve_images,
        }

        defaults: Dict[str, Any] = {
            "chunk_size": cls.max_chunk_size,
            "max_chunk_size": cls.max_chunk_size,
            "chunk_overlap": cls.chunk_overlap,
        }

        html: Dict[str, Any] = {
            "chunk_size": cls.max_chunk_size,
            "max_chunk_size": cls.max_chunk_size,
            "chunk_overlap": cls.chunk_overlap,
            "headers_to_split_on": deepcopy(cls.headers_to_split_on),
            "elements_to_preserve": deepcopy(cls.elements_to_preserve),
            "preserve_images": cls.preserve_images,
        }

        markdown: Dict[str, Any] = {
            "chunk_size": cls.max_chunk_size,
            "max_chunk_size": cls.max_chunk_size,
            "chunk_overlap": cls.chunk_overlap,
            "separators": deepcopy(cls.separators),
        }

        structured = dict(base)
        structured["defaults"] = defaults
        structured["html"] = html
        structured["markdown"] = markdown
        return structured


class DataTypeLoaderConfig(ConfigTemplate):
    docx = 'docx'
    doc = 'doc'
    odt = 'odt'
    rtf = 'rtf'
    pdf = 'pdf'
    directory = 'directory'
    zip = 'zip'
    json = 'json'


class ParsingPatternConfig(ConfigTemplate):
    image_pattern = r'!\[image:([^\]]+\.jpeg)\]\(([^)]+\.jpeg)\)'
    table_pattern = r'<table\b[^>]*>.*?</table>'
    ignored_topics = [
        "author information", "associated content", "acknowledgment", "acknowledgement", "acknowledgments",
        "acknowledgements", "references", "data availability", "declaration of competing interest",
        "credit authorship contribution statement", "funding", "ethical statements", "supplementary materials",
        "conflict of interest", "conflicts of interest", "author contributions", "data availability statement",
        "ethics approval", "supplementary information"
    ]
