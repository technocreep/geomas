from __future__ import annotations

import argparse
import logging
import os
import shutil
import typer
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

from dotenv import load_dotenv

from geomas.api import rag as rag_module
from geomas.api.rag import RagApi
from geomas.core.logging.logger import get_logger
from geomas.core.rag_modules.data_adapter import format_text_context

logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
app = typer.Typer(help="GEOMAS")
logger = get_logger()

QUESTION_1 = (
    "Подробно опиши морфологию рудных тел на территории `Светлое`."
    "Ответь со ссылкой на источник. "
    "В ответе укажи названия файлов."
)

QUESTION_2 = (
    "Подробно опиши морфологию рудных тел на территории `Сенон`."
    "Ответь со ссылкой на источник. "
    "В ответе укажи названия файлов."
)

QUESTION_3 = (
    "Подробно опиши морфологию рудных тел на территории `Сергеевское`."
    "Ответь со ссылкой на источник. "
    "В ответе укажи названия файлов."
)

DEFAULT_MODEL = "lmstudio-community/Meta-Llama-3-8B-Instruct"


@dataclass(frozen=True, slots=True)
class LmStudioSettings:
    base_url: str
    model: str
    temperature: float = 0.2
    timeout: float | None = None
    system_prompt: str | None = "Ответ должен быть на русском"

    def with_overrides(self, overrides: Mapping[str, object]) -> "LmStudioSettings":
        """Return a copy with supported ``overrides`` applied."""

        if not overrides:
            return self

        valid_fields = {"base_url", "model", "temperature", "timeout", "system_prompt"}
        unknown = sorted(set(overrides) - valid_fields)
        if unknown:
            raise ValueError(f"Unsupported LM Studio settings: {', '.join(unknown)}")

        payload = {
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "system_prompt": self.system_prompt,
        }

        for key, value in overrides.items():
            if key == "temperature":
                payload[key] = float(value)
            elif key == "timeout":
                payload[key] = None if value is None else float(value)
            elif key == "base_url":
                payload[key] = str(value).rstrip("/")
            elif key == "model":
                if not value:
                    raise ValueError("LM Studio model must be a non-empty string")
                payload[key] = str(value)
            elif key == "system_prompt":
                payload[key] = None if value in {None, ""} else str(value)

        return LmStudioSettings(**payload)

    def to_inference_params(self) -> dict[str, object]:
        params: dict[str, object] = {
            "provider": "lm_studio",
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
        }
        if self.timeout is not None:
            params["timeout"] = self.timeout
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


def load_lmstudio_settings(
    *,
    use_dotenv: bool = True,
    environ: Mapping[str, str] | None = None,
) -> LmStudioSettings:
    """Load :class:`LmStudioSettings` from environment variables."""

    if use_dotenv:
        try:
            load_dotenv()
        except Exception as exc:  # pragma: no cover - defensive load
            logger.debug("Failed to load .env file: %s", exc)

    env = dict(os.environ)
    if environ is not None:
        env.update(environ)

    base_url = env.get("LM_STUDIO_URL") or env.get("LM_STUDIO_BASE_URL")
    if not base_url:
        host = env.get("LM_STUDIO_HOST", "localhost")
        port = env.get("LM_STUDIO_PORT", "1234")
        base_url = f"http://{host}:{port}"

    model = env.get("LM_STUDIO_MODEL", DEFAULT_MODEL)
    if not model:
        raise RuntimeError("LM_STUDIO_MODEL must be set to a non-empty string")

    temperature = _read_float_env("LM_STUDIO_TEMPERATURE", default=0.2, env=env)
    timeout = _read_optional_float_env("LM_STUDIO_TIMEOUT", env=env)
    system_prompt = env.get("LM_STUDIO_SYSTEM_PROMPT") or "Ответ должен быть на русском"

    return LmStudioSettings(
        base_url=base_url.rstrip("/"),
        model=model,
        temperature=temperature,
        timeout=timeout,
        system_prompt=system_prompt,
    )


def build_lmstudio_rag_config(
    *,
    chat_id: str | None = None,
    global_rag_dir: Path,
    local_rag_dir: Path | None = None,
    settings: LmStudioSettings | None = None,
) -> rag_module.RAGConfig:
    resolved_settings = settings or load_lmstudio_settings()

    if chat_id is not None:
        collection_name = f"{chat_id}_local"
        rag_dir = local_rag_dir
    else:
        collection_name = "global"
        rag_dir = global_rag_dir

    overrides: dict[str, object] = {
        "parsing": {"enable_parser": False},
        "database": {
            "client_mode": "persistent",
            "persist_directory": str(rag_dir),
            "collection_name": collection_name,
        },
        "retrieval": {
            "top_k": 5,
            "text_top_k": 5,
            "embedding_model_name": "ViT-B-32",
            "checkpoint": "laion2b_s34b_b79k",
        },
        "ranking": {
            "use_llm_reranking": False,
            "chroma": {"enabled": True},
        },
        "vector_store": {
            "persist_directory": str(rag_dir),
        },
        "inference": {
            "enable_remote_services": True,
            "provider": "lm_studio",
            "params": resolved_settings.to_inference_params(),
        },
    }
    base_config = rag_module.RAGConfig.default().to_dict()
    rag_module._deep_update(base_config, overrides)
    return rag_module.RAGConfig.from_mapping(base_config)


def default_collection_targets(chat_id: str, include_global: bool = True) -> list[str]:
    targets: list[str] = []
    if include_global:
        targets.append("global")
    targets.append(f"{chat_id}_local")
    return targets


def build_paths(
    documents_dir: Path | str,
    global_rag_dir: Path | str,
    chat_dir: Path | str,
    uploads_dir: Path | str,
    local_rag_dir: Path | str,
) -> dict[str, Path]:
    document_dir = Path(documents_dir)
    global_rag_dir = Path(global_rag_dir)
    chat_dir = Path(chat_dir)
    uploads_dir = Path(uploads_dir)
    local_rag_dir = Path(local_rag_dir)
    document_dir.mkdir(parents=True, exist_ok=True)
    global_rag_dir.mkdir(parents=True, exist_ok=True)
    chat_dir.mkdir(parents=True, exist_ok=True)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    local_rag_dir.mkdir(parents=True, exist_ok=True)
    return dict(
        documents_dir=documents_dir,
        global_rag_dir=global_rag_dir,
        chat_dir=chat_dir,
        uploads_dir=uploads_dir,
        local_rag_dir=local_rag_dir,
    )


def build_chat_api(
    *,
    local_rag_dir: Path | str | None = None,
    global_rag_dir: Path | str,
    chat_id: str | None = None,
    settings_overrides: Mapping[str, object] | LmStudioSettings | None = None,
) -> RagApi:
    settings = load_lmstudio_settings()
    if settings_overrides is not None:
        settings = settings.with_overrides(settings_overrides)
    config = build_lmstudio_rag_config(
        chat_id=chat_id,
        global_rag_dir=global_rag_dir,
        local_rag_dir=local_rag_dir,
        settings=settings,
    )
    return RagApi(config=config)


def initialize_global_rag(
    *,
    paths: dict[str, Path],
    settings: Mapping[str, object] | LmStudioSettings | None = None,
    describe_images: bool = False,
) -> None:
    logger.info("Step 1/4: preparing shared corpus at %s", paths.get("global_rag_dir"))
    if not os.listdir(paths.get("global_rag_dir")):
        with build_chat_api(
            global_rag_dir=paths.get("global_rag_dir"),
            settings_overrides=settings,
        ) as api:
            api.initialize_pipeline(
                paths.get("documents_dir"),
                describe_images=describe_images,
            )
            api.close()


@contextmanager
def create_chat_session(
    *,
    paths: dict[str, Path],
    chat_id: str,
    reset_local_rag: bool = False,
    settings: Mapping[str, object] | LmStudioSettings | None = None,
    describe_images: bool = False,
) -> Iterator[RagApi]:
    if reset_local_rag:
        for entry in list(paths.get("local_rag_dir").iterdir()):
            if entry.is_file():
                os.remove(entry)
            elif entry.is_dir():
                shutil.rmtree(entry)
    logger.info(
        "Step 2/4: preparing chat %s RAG in %s (uploads at %s)",
        chat_id,
        paths.get("local_rag_dir"),
        paths.get("uploads_dir"),
    )
    with build_chat_api(
        local_rag_dir=paths.get("local_rag_dir"),
        global_rag_dir=paths.get("global_rag_dir"),
        chat_id=chat_id,
        settings_overrides=settings,
    ) as api:
        api.initialize_pipeline(
            paths.get("uploads_dir"),
            describe_images=describe_images,
        )
        yield api


def ingest_local_documents(
    api: RagApi,
    *,
    paths: Mapping[str, Path] | object,
    describe_images: bool = False,
) -> None:
    """Ingest chat uploads, optionally generating descriptions for images."""

    logger.info("Preparing chat-local vector store at %s", paths.get("local_rag_dir"))
    api.initialize_pipeline(
        paths.get("uploads_dir"),
        api.config.database.get("collection_name"),
        describe_images=describe_images,
    )


def answer_with_combined_context(
    api: RagApi,
    question: str,
    *,
    chat_id: str,
    query_kwargs: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    logger.info(
        "Step 4/4: querying combined context for chat %s and question: %s",
        chat_id,
        question,
    )
    payload = dict(query_kwargs or {})
    response = api.ask_question(question, **payload)
    raw_context = response.get("text_context", [])
    formatted_rows = format_text_context(raw_context)
    return response, formatted_rows


def show_results(
    response: dict[str, object],
    context_rows: list[dict[str, object]]
) -> None:
    logger.info(f"Answer: {response.get('answer')}")
    text_rows = [row for row in context_rows if row.get("type") != "image"]
    image_rows = [row for row in context_rows if row.get("type") == "image"]

    if text_rows:
        logger.info("\nContext snippets:")
        for entry in text_rows:
            score = entry.get("score")
            if isinstance(score, (int, float)):
                score_display = f"{float(score):.3f}"
            else:
                score_display = str(score)
            scope = entry.get("database_scope")
            scope_suffix = f", scope={scope}" if scope else ""
            logger.info(f"- {entry.get('document')} (score={score_display}{scope_suffix})")
            logger.info(f"  {entry.get('preview')}")

    if image_rows:
        logger.info("\nImage matches:")
        for entry in image_rows:
            score = entry.get("score")
            label = entry.get("document")
            path = entry.get("source_path") or "(inline)"
            score_display = f"{float(score):.3f}" if isinstance(score, (int, float)) else str(score)
            logger.info(f"- {label} [{score_display}] -> {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chat-id", default="demo-chat", help="Name for the chat-local collection")
    parser.add_argument("--documents-dir", default="./data/global/uploads", help="Shared corpus location")
    parser.add_argument(
        "--global-rag-dir",
        default="./data/global/.vector-store",
        help="Persistent directory for the shared Chroma database",
    )
    parser.add_argument("--uploads-dir", default=None, help="Directory containing chat uploads")
    parser.add_argument(
        "--local-rag-dir",
        default=None,
        help="Persistent directory for the chat-local Chroma database",
    )
    parser.add_argument("--lm-studio-host", default="localhost", help="LM Studio host (matches Ollama example comments)")
    parser.add_argument("--lm-studio-port", default="1234", help="LM Studio port")
    parser.add_argument(
        "--lm-studio-model",
        default=DEFAULT_MODEL,
        help="LM Studio model identifier (mirrors the Ollama example structure)",
    )
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature for LM Studio completions")
    parser.add_argument("--system-prompt", default="Ответ должен быть на русском", help="System prompt passed to LM Studio")
    parser.add_argument("--include-global", action="store_true", help="Search both global and chat-local scopes")
    parser.add_argument("--reset-local-rag", action="store_true", help="Clear chat-local vector store before ingesting")
    parser.add_argument(
        "--describe-images",
        action="store_true",
        help="Enable image-description ingestion for both global and chat uploads",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chat_id = args.chat_id
    include_global = args.include_global
    reset_local_rag = args.reset_local_rag

    uploads_dir = args.uploads_dir or f"./data/{chat_id}/uploads"
    local_rag_dir = args.local_rag_dir or f"./data/{chat_id}/.vector-store"
    paths = build_paths(
        documents_dir=args.documents_dir,
        global_rag_dir=args.global_rag_dir,
        chat_dir=f"./data/{chat_id}",
        uploads_dir=uploads_dir,
        local_rag_dir=local_rag_dir,
    )

    settings_overrides: dict[str, object] = {
        "base_url": f"http://{args.lm_studio_host}:{args.lm_studio_port}",
        "model": args.lm_studio_model,
        "temperature": args.temperature,
        "system_prompt": args.system_prompt,
    }

    query_kwargs: dict[str, object] = {
        "text_top_k": 5,
        "rerank_top_k": 5,
    }
    collection_targets = default_collection_targets(chat_id, include_global=include_global)
    query_kwargs["scopes"] = collection_targets

    logger.info("Step 1/4: Initialising shared corpus...")
    initialize_global_rag(paths=paths, settings=settings_overrides, describe_images=args.describe_images)
    logger.info("Step 1/4 complete: shared corpus initialised.")

    logger.info(f"Creating new chat: {chat_id}")
    logger.info("Step 2/4: Creating chat session...")
    logger.info(f"Databases: {collection_targets}")
    with create_chat_session(
        paths=paths,
        chat_id=chat_id,
        settings=settings_overrides,
        reset_local_rag=reset_local_rag,
        describe_images=args.describe_images,
    ) as api:
        logger.info("Step 3/4: Ingesting uploads...")
        ingest_local_documents(
            api,
            paths=paths,
            describe_images=args.describe_images,
        )
        logger.info("Step 3/4 complete.")

        logger.info("Step 4/4: Querying combined context... [1]")
        response, context_rows = answer_with_combined_context(
            api,
            QUESTION_1,
            chat_id=chat_id,
            query_kwargs=query_kwargs,
        )
        show_results(response, context_rows)

        logger.info("Step 4/4: Querying combined context... [2]")
        response, context_rows = answer_with_combined_context(
            api,
            QUESTION_2,
            chat_id=chat_id,
            query_kwargs=query_kwargs,
        )
        show_results(response, context_rows)

        logger.info("Step 4/4: Querying combined context... [3]")
        response, context_rows = answer_with_combined_context(
            api,
            QUESTION_3,
            chat_id=chat_id,
            query_kwargs=query_kwargs,
        )
        show_results(response, context_rows)
        api.close()


if __name__ == "__main__":
    main()
