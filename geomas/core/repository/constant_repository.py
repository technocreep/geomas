from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Iterable, MutableMapping

from dotenv import load_dotenv


logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT_DIR, "config.env")

_ENV_LOADED = False
_REPOSITORY_ENV: MutableMapping[str, Any] = {}

_MANAGED_ENV_VARS = {
    "USE_S3",
    "CHROMA_HOST",
    "CHROMA_PORT",
    "CHROMA_ALLOW_RESET",
    "CHROMA_CLIENT_MODE",
    "CHROMA_PERSIST_PATH",
    "EMBEDDING_HOST",
    "EMBEDDING_PORT",
    "EMBEDDING_ENDPOINT",
    "EMBEDDING_MODE",
    "RERANKING_ENABLED",
    "RERANKER_HOST",
    "RERANKER_PORT",
    "RERANKER_ENDPOINT",
    "PARSE_RESULTS_PATH",
    "PAPERS_STORAGE_PATH",
    "SUMMARY_LLM_URL",
    "VISION_LLM_URL",
    "LLM_SERVICE_URL",
    "LLM_OCR_URL",
    "IMAGE_RESOLUTION_SCALE",
}


def _normalise_key(key: str) -> str:
    return key.strip().upper()


def _get_raw_env(key: str) -> str | None:
    return os.getenv(_normalise_key(key))


def _get_bool(key: str, default: bool = False) -> bool:
    value = _get_raw_env(key)
    if value is None:
        return default
    normalized = value.strip().lower()
    if not normalized:
        return default
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    logger.warning("Invalid boolean value for %s: %s", key, value)
    return default


def _get_int(key: str, default: int | None = None) -> int | None:
    value = _get_raw_env(key)
    if value is None:
        return default
    candidate = value.strip()
    if not candidate:
        return default
    try:
        return int(candidate)
    except ValueError:
        logger.warning("Invalid integer value for %s: %s", key, value)
        return default


def _get_float(key: str, default: float | None = None) -> float | None:
    value = _get_raw_env(key)
    if value is None:
        return default
    candidate = value.strip()
    if not candidate:
        return default
    try:
        return float(candidate)
    except ValueError:
        logger.warning("Invalid float value for %s: %s", key, value)
        return default


def _get_str(key: str, default: str | None = None) -> str | None:
    value = _get_raw_env(key)
    if value is None:
        return default
    candidate = value.strip()
    if not candidate and default is not None:
        return default
    return candidate


def _resolve_path(key: str, default: str) -> str:
    value = _get_str(key, default)
    if value is None:
        value = default
    path = Path(value)
    if not path.is_absolute():
        return str(Path(ROOT_DIR) / path)
    return str(path)


def _clean_empty_values(keys: Iterable[str]) -> None:
    for key in keys:
        raw_value = os.getenv(key)
        if raw_value is not None and not raw_value.strip():
            os.environ.pop(key, None)


def load_repository_env(force: bool = False) -> MutableMapping[str, Any]:
    """Load environment variables from :data:`CONFIG_PATH` once per process."""

    global _ENV_LOADED
    if force or not _ENV_LOADED:
        load_dotenv(CONFIG_PATH, override=False)
        _clean_empty_values(_MANAGED_ENV_VARS)
        _REPOSITORY_ENV.clear()
        _REPOSITORY_ENV.update(
            {
                "use_s3": _get_bool("USE_S3", False),
                "chroma_host": _get_str("CHROMA_HOST"),
                "chroma_port": _get_int("CHROMA_PORT"),
                "chroma_allow_reset": _get_bool("CHROMA_ALLOW_RESET", False),
                "chroma_client_mode": _get_str("CHROMA_CLIENT_MODE", "persistent"),
                "chroma_persist_path": _resolve_path(
                    "CHROMA_PERSIST_PATH", "./chroma_storage"
                ),
                "embedding_host": _get_str("EMBEDDING_HOST"),
                "embedding_port": _get_int("EMBEDDING_PORT"),
                "embedding_endpoint": _get_str("EMBEDDING_ENDPOINT", "/embed"),
                "embedding_mode": _get_str("EMBEDDING_MODE", "local"),
                "reranker_enabled": _get_bool("RERANKING_ENABLED", False),
                "reranker_host": _get_str("RERANKER_HOST"),
                "reranker_port": _get_int("RERANKER_PORT"),
                "reranker_endpoint": _get_str("RERANKER_ENDPOINT", "/rerank"),
                "parse_results_path": _resolve_path(
                    "PARSE_RESULTS_PATH", "./parse_results"
                ),
                "papers_storage_path": _resolve_path(
                    "PAPERS_STORAGE_PATH", "./papers"
                ),
                "vision_llm_url": _get_str("VISION_LLM_URL", ""),
                "summary_llm_url": _get_str("SUMMARY_LLM_URL", ""),
                "llm_service_url": _get_str("LLM_SERVICE_URL", ""),
                "llm_ocr_url": _get_str("LLM_OCR_URL", ""),
                "image_resolution_scale": _get_float("IMAGE_RESOLUTION_SCALE", 2.0)
                or 2.0,
            }
        )
        _ENV_LOADED = True
    return _REPOSITORY_ENV


REPOSITORY_ENV = load_repository_env()

USE_S3: bool = bool(REPOSITORY_ENV["use_s3"])
CHROMA_DB_PATH = REPOSITORY_ENV["chroma_persist_path"]
VISION_LLM_URL = REPOSITORY_ENV["vision_llm_url"] or ""
SUMMARY_LLM_URL = REPOSITORY_ENV["summary_llm_url"] or ""
LLM_SERVICE_URL = REPOSITORY_ENV["llm_service_url"] or ""
IMAGE_RESOLUTION_SCALE = float(REPOSITORY_ENV["image_resolution_scale"])
LLM_OCR_URL = REPOSITORY_ENV["llm_ocr_url"] or ""
PARSE_RESULTS_PATH = REPOSITORY_ENV["parse_results_path"]
PAPERS_STORAGE_PATH = REPOSITORY_ENV["papers_storage_path"]

BOS, EOS = "<|begin_of_text|>", "<|end_of_text|>"
SOH, EOH = "<|start_header_id|>", "<|end_header_id|>\n\n"
EOT = "<|eot_id|>"