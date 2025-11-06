"""Markdown specification and validation utilities."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, TypedDict

import yaml

FRONT_MATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


class MarkdownMeta(TypedDict):
    """Required YAML front matter fields for standardized Markdown."""

    source_file: str
    source_mime: str
    page_index: int
    sha256: str
    created_utc: str
    model: str
    model_version: str
    pipeline: str


REQUIRED_FIELDS = {
    "source_file",
    "source_mime",
    "page_index",
    "sha256",
    "created_utc",
    "model",
    "model_version",
    "pipeline",
}


def _parse_front_matter(text: str) -> Dict[str, Any]:
    try:
        return (
            yaml.safe_load(match.group(1)) or {}
            if (match := FRONT_MATTER_RE.match(text))
            else {}
        )
    except yaml.YAMLError:
        return {}


def validate_markdown(text: str) -> bool:
    """Return ``True`` when *text* has valid Markdown front matter."""

    meta = _parse_front_matter(text)
    if REQUIRED_FIELDS - meta.keys():
        return False
    src = meta.get("source_file")
    mime = meta.get("source_mime")
    page = meta.get("page_index")
    digest = meta.get("sha256")
    created = meta.get("created_utc")
    model = meta.get("model")
    model_ver = meta.get("model_version")
    pipeline = meta.get("pipeline")
    if not isinstance(src, str) or not src:
        return False
    if not isinstance(mime, str) or not mime:
        return False
    if not isinstance(page, int) or page < 0:
        return False
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        return False
    if not isinstance(created, str):
        return False
    try:
        datetime.fromisoformat(created)
    except ValueError:
        return False
    if not isinstance(model, str) or not model:
        return False
    if not isinstance(model_ver, str) or not model_ver:
        return False
    if not isinstance(pipeline, str) or not pipeline:
        return False
    return True


__all__ = ["MarkdownMeta", "REQUIRED_FIELDS", "validate_markdown"]
