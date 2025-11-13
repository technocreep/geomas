from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

from geomas.api.rag import RagApi
from geomas.core.inference.ollama_client import (
    OllamaSettings,
    build_ollama_rag_config,
    load_ollama_settings,
)
from geomas.core.rag_modules.data_adapter import format_text_context
from geomas.core.rag_modules.database.chroma_db import ProcessingResult


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
    global_rag_dir: Path,
    chat_id: str | None = None,
    local_rag_dir: Path | None = None,
    settings: Mapping[str, object] | OllamaSettings | None = None,
) -> RagApi:
    base_settings = load_ollama_settings()
    if isinstance(settings, OllamaSettings):
        resolved_settings = settings
    elif isinstance(settings, Mapping):
        resolved_settings = base_settings.with_overrides(settings)
    elif settings is None:
        resolved_settings = base_settings
    config = build_ollama_rag_config(
        chat_id=chat_id,
        global_rag_dir=global_rag_dir,
        local_rag_dir=local_rag_dir,
        settings=resolved_settings,
    )
    return RagApi(config=config)


def step_initialize_shared_corpus(
    *,
    paths: dict[str, Path],
    settings: Mapping[str, object] | OllamaSettings | None = None,
) -> bool:
    with build_chat_api(
        global_rag_dir=paths.get("global_rag_dir"),
        settings=settings,
    ) as api:
        documents_dir = Path(paths.get("documents_dir"))
        pipeline = getattr(api, "pipeline", None)
        has_existing_embeddings = False
        if pipeline is not None:
            store = getattr(pipeline, "global_store", None)
            collection = getattr(store, "collection", None)
            count = getattr(collection, "count", None)
            if callable(count):
                try:
                    has_existing_embeddings = int(count()) > 0
                except Exception:
                    has_existing_embeddings = False
        if has_existing_embeddings:
            logger.info(
                "Shared corpus already available; initialising pipeline without reingestion",
            )
            return bool(api.initialize_pipeline())
        documents_available = False
        if documents_dir.exists():
            if documents_dir.is_file():
                documents_available = True
            elif documents_dir.is_dir():
                documents_available = any(
                    candidate.is_file() for candidate in documents_dir.rglob("*")
                )
        if not documents_available:
            logger.info(
                "No documents found at %s; initialising pipeline without ingestion",
                documents_dir,
            )
            return bool(api.initialize_pipeline())
        logger.info(
            "Ingesting corpus from %s into the shared vector-store",
            documents_dir,
        )
        return bool(api.initialize_pipeline(documents_path=documents_dir))


def step_list_and_ingest_uploads(
    api: RagApi,
    *,
    uploads_dir: Path,
    uploads: Sequence[Path] | None = None,
) -> tuple[list[Path], list[ProcessingResult]]:
    if uploads is None:
        upload_paths = sorted(
            (path.resolve() for path in uploads_dir.rglob("*") if path.is_file()),
            key=lambda path: path.as_posix(),
        )
    else:
        upload_paths = [Path(path).resolve() for path in uploads]
    ingestions: list[ProcessingResult] = []
    for document_path in upload_paths:
        if not document_path.exists():
            raise FileNotFoundError(f"Upload artefact is missing: {document_path}")
        ingestions.append(api._ingest_path(document_path, namespace="local"))
    return upload_paths, ingestions


def step_query_with_combined_context(
    api: RagApi,
    question: str,
    *,
    query_kwargs: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    response = api.ask_question(question, **(dict(query_kwargs) if query_kwargs else {}))
    if not isinstance(response, Mapping):
        raise TypeError("RagApi.ask_question returned an unexpected payload")
    raw_context = response.get("text_context", [])
    metadata_lookup: dict[object, Mapping[str, object]] = {}
    if isinstance(raw_context, Sequence):
        for entry in raw_context:
            if not isinstance(entry, Sequence) or len(entry) < 3:
                continue
            metadata = entry[2] if isinstance(entry[2], Mapping) else {}
            try:
                metadata_lookup.setdefault(entry[0], metadata)
            except TypeError:
                continue
    formatted_rows = format_text_context(raw_context)
    enriched_rows: list[dict[str, object]] = []
    for row in formatted_rows:
        row_metadata = metadata_lookup.get(row.get("id"), {})
        scope_value = None
        if isinstance(row_metadata, Mapping):
            candidate_scope = row_metadata.get("database_scope")
            if isinstance(candidate_scope, str) and candidate_scope:
                scope_value = candidate_scope
        payload = {key: value for key, value in row.items() if key != "id"}
        payload["database_scope"] = scope_value
        enriched_rows.append(payload)
    return dict(response), enriched_rows


def initialize_global_rag(
    *,
    paths: dict[str, Path],
    settings: Mapping[str, object] | OllamaSettings | None = None,
) -> bool:
    logger.info("Step 1/4: preparing shared corpus at %s", paths.get("global_rag_dir"))
    return step_initialize_shared_corpus(
        paths=paths,
        settings=settings,
    )


@contextmanager
def create_chat_session(
    *,
    paths: dict[str, Path],
    chat_id: str,
    settings: Mapping[str, object] | OllamaSettings | None = None,
) -> Iterator[RagApi]:
    logger.info(
        "Step 2/4: preparing chat %s RAG in %s (uploads at %s)",
        chat_id,
        paths.get("local_rag_dir"),
        paths.get("uploads_dir"),
    )
    api = build_chat_api(
        chat_id=chat_id,
        global_rag_dir=paths.get("global_rag_dir"),
        local_rag_dir=paths.get("local_rag_dir"),
        settings=settings,
    )
    try:
        api.initialize_pipeline()
        yield api
    finally:
        try:
            api.close()
        except Exception as exc:
            logger.debug("Failed to close chat session cleanly: %s", exc)


def ingest_local_documents(
    api: RagApi,
    *,
    paths: dict[str, Path],
    uploads: Sequence[Path] | None = None,
) -> tuple[list[Path], list[ProcessingResult]]:
    logger.info("Step 3/4: ingesting uploads from %s", paths.get("uploads_dir"))
    return step_list_and_ingest_uploads(
        api,
        uploads_dir=paths.get("uploads_dir"),
        uploads=uploads,
    )


def answer_with_combined_context(
    api: RagApi,
    question: str,
    *,
    chat_id: str,
    query_kwargs: Mapping[str, object] | None = None,
    context_limit: int = 4,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    logger.info(
        "Step 4/4: querying combined context for chat %s and question: %s",
        chat_id,
        question,
    )
    return step_query_with_combined_context(
        api,
        question,
        query_kwargs=query_kwargs,
    )

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
    paths = build_paths(
        documents_dir="./data",
        global_rag_dir="./data/.vector-store",
        chat_dir=f"./data/{chat_id}",
        uploads_dir=f"./data/{chat_id}/uploads",
        local_rag_dir=f"./data/{chat_id}/.vector-store",
    )
    print("Step 1/4: Initialising shared corpus...")
    global_ready = initialize_global_rag(paths=paths)
    print(f"Step 1/4 complete: shared corpus initialised -> {global_ready}.")

    print(f"Creating new chat: {chat_id}")
    print("Step 2/4: Creating chat session...")
    with create_chat_session(
        paths=paths,
        chat_id=chat_id,
        settings=settings_overrides,
    ) as api:
        print("Step 3/4: Ingesting uploads...")
        upload_paths, ingestions = ingest_local_documents(
            api,
            paths=paths,
        )
        print(f"Step 3/4 complete: {len(ingestions)} upload(s) ingested.")

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


    chat_id = "sfasfpkasfka"
    paths = build_paths(
        documents_dir="./data",
        global_rag_dir="./data/.vector-store",
        chat_dir=f"./data/{chat_id}",
        uploads_dir=f"./data/{chat_id}/uploads",
        local_rag_dir=f"./data/{chat_id}/.vector-store",
    )
    print(f"Creating new chat: {chat_id}")
    with create_chat_session(
        paths=paths,
        chat_id=chat_id,
        settings=settings_overrides,
    ) as api:
        print("Step 3/4: Ingesting uploads...")
        upload_paths, ingestions = ingest_local_documents(
            api,
            paths=paths,
        )
        print(f"Step 3/4 complete: {len(ingestions)} upload(s) ingested.")

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