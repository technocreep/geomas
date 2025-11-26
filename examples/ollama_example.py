from __future__ import annotations

import logging
import os
import shutil
import typer
from contextlib import contextmanager

from pathlib import Path
from typing import Iterator, Mapping

from geomas.api import rag as rag_module
from geomas.api.rag import RagApi
from geomas.core.logging.logger import get_logger
from geomas.core.repository.constant_repository import _resolve_path
from geomas.core.inference.ollama_client import (
    OllamaSettings,
    build_ollama_rag_config,
    load_ollama_settings,
)
from geomas.core.rag_modules.data_adapter import format_text_context

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

logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
app = typer.Typer(help="GEOMAS")
logger = get_logger()

build_rag_config = rag_module.build_ollama_rag_config


def default_collection_targets(
    chat_id: str,
    paths: dict[str, Path],
    include_global: bool = True
) -> dict[str, str]:
    targets: dict = {}
    if include_global:
        targets["global"] = paths.get("global_rag_dir")
    targets[f"{chat_id}_local"] = paths.get("local_rag_dir")
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
    _resolve_path("", document_dir)
    _resolve_path("", global_rag_dir)
    _resolve_path("", chat_dir)
    _resolve_path("", uploads_dir)
    _resolve_path("", local_rag_dir)
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
    settings_overrides: Mapping[str, object] | OllamaSettings | None = None,
) -> RagApi:
    settings = load_ollama_settings()
    if settings_overrides is not None:
        settings = settings.with_overrides(settings_overrides)
    config = build_ollama_rag_config(
        chat_id=chat_id,
        global_rag_dir=global_rag_dir,
        local_rag_dir=local_rag_dir,
        settings=settings,
    )
    return RagApi(config=config)


def initialize_global_rag(
    *,
    paths: dict[str, Path],
    settings: Mapping[str, object] | OllamaSettings | None = None,
) -> None:
    logger.info("Step 1/4: preparing shared corpus at %s", paths.get("global_rag_dir"))
    if not os.listdir(paths.get("global_rag_dir")):
        with build_chat_api(
            global_rag_dir=paths.get("global_rag_dir"),
            settings_overrides=settings,
        ) as api:
            api.initialize_pipeline(paths.get("documents_dir"))
            api.close()


@contextmanager
def create_chat_session(
    *,
    paths: dict[str, Path],
    chat_id: str,
    reset_local_rag: bool = True,
    settings: Mapping[str, object] | OllamaSettings | None = None,
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
        try:
            api.initialize_pipeline(paths.get("uploads_dir"), f"{chat_id}_local")
            yield api
        finally:
            api.close()


def ingest_local_documents(
    api: RagApi,
    *,
    paths: Mapping[str, Path] | object,
    describe_images: bool = True,
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
    history: str = "",
) -> tuple[dict[str, object], list[dict[str, object]]]:
    logger.info(
        "Step 4/4: querying combined context for chat %s and question: %s",
        chat_id,
        question,
    )
    payload = dict(query_kwargs or {})
    response = api.ask_question(question, history, **payload)
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


def main() -> None:
    query_kwargs: dict[str, int] = {
        "top_k": 5,
        "query_images": False,
    }
    settings_overrides: dict[str, object] = {"temperature": 0.1}

    chat_id = "demo-chat"
    include_global = True
    reset_local_rag = True
    describe_images = True
    paths = build_paths(
        documents_dir="./data/global/uploads",
        global_rag_dir="./data/global/.vector-store",
        chat_dir=f"./data/{chat_id}",
        uploads_dir=f"./data/{chat_id}/uploads",
        local_rag_dir=f"./data/{chat_id}/.vector-store",
    )
    logger.info("Step 1/4: Initialising shared corpus...")
    initialize_global_rag(paths=paths)
    logger.info("Step 1/4 complete: shared corpus initialised.")

    logger.info(f"Creating new chat: {chat_id}")
    logger.info("Step 2/4: Creating chat session...")
    collection_targets = default_collection_targets(chat_id, paths=paths, include_global=include_global)
    query_kwargs["scopes"] = collection_targets
    logger.info(f"Databases: {query_kwargs['scopes']}")
    with create_chat_session(
        paths=paths,
        chat_id=chat_id,
        settings=settings_overrides,
        reset_local_rag=reset_local_rag,
    ) as api:
        logger.info("Step 3/4: Ingesting uploads...")
        ingest_local_documents(
            api,
            paths=paths,
            describe_images=describe_images,
        )
        logger.info(f"Step 3/4 complete.")

        logger.info("Step 4/4: Querying combined context... [1]")
        question = QUESTION_1
        response, context_rows = answer_with_combined_context(
            api,
            question,
            chat_id=chat_id,
            query_kwargs=query_kwargs,
        )
        show_results(response, context_rows)

        logger.info("Step 4/4: Querying combined context... [2]")
        question = QUESTION_2
        response, context_rows = answer_with_combined_context(
            api,
            question,
            chat_id=chat_id,
            query_kwargs=query_kwargs,
        )
        show_results(response, context_rows)

        logger.info("Step 4/4: Querying combined context... [3]")
        question = QUESTION_3
        response, context_rows = answer_with_combined_context(
            api,
            question,
            chat_id=chat_id,
            query_kwargs=query_kwargs,
        )
        show_results(response, context_rows)

    previous_chat_id = chat_id
    chat_id = "sfasfpkasfka"
    include_global = True
    reset_local_rag = True
    describe_images = True
    paths = build_paths(
        documents_dir="./data/global/uploads",
        global_rag_dir="./data/global/.vector-store",
        chat_dir=Path(f"./data/{chat_id}"),
        uploads_dir=Path(f"./data/{chat_id}/uploads"),
        local_rag_dir=Path(f"./data/{chat_id}/.vector-store"),
    )
    logger.info(f"Creating new chat: {chat_id}")

    collection_targets = default_collection_targets(chat_id, paths=paths, include_global=include_global)
    collection_targets[f"{previous_chat_id}"] = f"./data/{previous_chat_id}/uploads"
    query_kwargs["scopes"] = collection_targets
    logger.info(f"Databases: {query_kwargs['scopes']}")
    with create_chat_session(
        paths=paths,
        chat_id=chat_id,
        settings=settings_overrides,
        reset_local_rag=reset_local_rag,
    ) as api:
        logger.info("Step 3/4: Ingesting uploads...")
        ingest_local_documents(
            api,
            paths=paths,
            describe_images=describe_images,
        )
        logger.info(f"Step 3/4 complete.")

        logger.info("Step 4/4: Querying combined context... [New chat session]")
        question = QUESTION_1
        response, context_rows = answer_with_combined_context(
            api,
            question,
            chat_id=chat_id,
            query_kwargs=query_kwargs,
        )
        show_results(response, context_rows)


if __name__ == "__main__":
    main()
