from __future__ import annotations

import os
import logging
import shutil
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from geomas.api import rag as rag_module
from geomas.api.rag import RagApi
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


logger = logging.getLogger(__name__)

build_rag_config = rag_module.build_ollama_rag_config


def default_collection_targets(
    chat_id: str,
    *,
    paths: dict[str, Path],
    include_global: bool = True
) -> dict[str, str]:
    targets: dict = {}
    if include_global:
        targets["global"] = paths.get("documents_dir")
    targets[f"{chat_id}_local"] = paths.get("uploads_dir")
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
    local_rag_dir: Path | str  | None = None,
    global_rag_dir: Path | str,
    chat_id: str  | None = None,
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
    reset_local_rag: bool = False,
    settings: Mapping[str, object] | OllamaSettings | None = None,
    collection_targets: dict[str, str] | None = None,
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
        for target, path in collection_targets.items():
            api.initialize_pipeline(str(path), target)
        yield api


def ingest_local_documents(
    api: RagApi,
    *,
    paths: Mapping[str, Path] | object,
) -> None:
    logger.info("Preparing chat-local vector store at %s", paths.get("local_rag_dir"))
    api.initialize_pipeline(paths.get("uploads_dir"), api.config.database.get("collection_name"))


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
    print(f"Answer: {response.get('answer') or 'No answer returned.'}")
    # Retrieval debug

    # print("Context snippets:")
    # for entry in context_rows:
    #     score = entry.get("score")
    #     if isinstance(score, (int, float)):
    #         score_display = f"{float(score):.3f}"
    #     else:
    #         score_display = str(score)
    #     scope = entry.get("database_scope")
    #     scope_suffix = f", scope={scope}" if scope else ""
    #     print(f"- {entry.get('document')} (score={score_display}{scope_suffix})")
    #     print(f"  {entry.get('preview')}")


def main() -> None:
    query_kwargs = {"text_top_k": 4, "rerank_top_k": 3}
    settings_overrides: dict[str, object] = {"temperature": 0.2}

    chat_id = "demo-chat"
    include_global = True
    reset_local_rag = False
    paths = build_paths(
        documents_dir="./data/global/uploads",
        global_rag_dir="./data/global/.vector-store",
        chat_dir=f"./data/{chat_id}",
        uploads_dir=f"./data/{chat_id}/uploads",
        local_rag_dir=f"./data/{chat_id}/.vector-store",
    )
    print("Step 1/4: Initialising shared corpus...")
    initialize_global_rag(paths=paths)
    print("Step 1/4 complete: shared corpus initialised.")

    print(f"Creating new chat: {chat_id}")
    print("Step 2/4: Creating chat session...")
    collection_targets = default_collection_targets(chat_id, paths=paths, include_global=include_global)
    print(f"Databases: {collection_targets}")
    with create_chat_session(
        paths=paths,
        chat_id=chat_id,
        settings=settings_overrides,
        collection_targets=collection_targets,
        reset_local_rag=reset_local_rag,
    ) as api:
        print("Step 3/4: Ingesting uploads...")
        ingest_local_documents(
            api,
            paths=paths,
        )
        print(f"Step 3/4 complete.")

        print("Step 4/4: Querying combined context... [1]")
        question = QUESTION_1
        response, context_rows = answer_with_combined_context(
            api,
            question,
            chat_id=chat_id,
            query_kwargs=query_kwargs,
        )
        show_results(response, context_rows)

        print("Step 4/4: Querying combined context... [2]")
        question = QUESTION_2
        response, context_rows = answer_with_combined_context(
            api,
            question,
            chat_id=chat_id,
            query_kwargs=query_kwargs,
        )
        show_results(response, context_rows)

        print("Step 4/4: Querying combined context... [3]")
        question = QUESTION_3
        response, context_rows = answer_with_combined_context(
            api,
            question,
            chat_id=chat_id,
            query_kwargs=query_kwargs,
        )
        show_results(response, context_rows)
        api.close()

    previous_chat_id = chat_id
    chat_id = "sfasfpkasfka"
    include_global = True
    reset_local_rag = False
    paths = build_paths(
        documents_dir="./data/global/uploads",
        global_rag_dir="./data/global/.vector-store",
        chat_dir=Path(f"./data/{chat_id}"),
        uploads_dir=Path(f"./data/{chat_id}/uploads"),
        local_rag_dir=Path(f"./data/{chat_id}/.vector-store"),
    )
    print(f"Creating new chat: {chat_id}")

    collection_targets = default_collection_targets(chat_id, paths=paths, include_global=include_global)
    collection_targets[f"{previous_chat_id}"] = f"./data/{previous_chat_id}/uploads"

    with create_chat_session(
        paths=paths,
        chat_id=chat_id,
        settings=settings_overrides,
        collection_targets=collection_targets,
        reset_local_rag=reset_local_rag,
    ) as api:
        print("Step 3/4: Ingesting uploads...")
        ingest_local_documents(
            api,
            paths=paths,
        )
        print(f"Step 3/4 complete.")

        print("Step 4/4: Querying combined context... [New chat session]")
        question = QUESTION_1
        response, context_rows = answer_with_combined_context(
            api,
            question,
            chat_id=chat_id,
            query_kwargs=query_kwargs,
        )
        show_results(response, context_rows)
        api.close()

if __name__ == "__main__":
    main()