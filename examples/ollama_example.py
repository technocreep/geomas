from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from shutil import copy2
from datetime import datetime

from geomas.api.rag import RagApi
from geomas.core.inference.ollama_client import (
    OllamaSettings,
    build_ollama_rag_config,
    load_ollama_settings,
    run_ollama_workflow as _run_ollama_workflow,
)
from geomas.core.rag_modules.data_adapter import format_text_context
from geomas.core.rag_modules.database.chroma_db import ProcessingResult

EXAMPLE_DOCUMENTS = Path(__file__).resolve().parent / "data"
CHAT_ID = "demo-chat"
CHAT_ROOT = Path(__file__).resolve().parent / CHAT_ID
DATA_ROOT = CHAT_ROOT / "data"
UPLOADS_ROOT = CHAT_ROOT / "uploads"
CACHE_ROOT = CHAT_ROOT / ".vector-store"
UPLOADS_CACHE_ROOT = CACHE_ROOT / "uploads"
FULL_CACHE_ROOT = CACHE_ROOT / "full"
DEFAULT_QUESTION = (
    "Какие руды присутствуют на территории Рудное поле Светлое? "
    "Ответь со ссылкой на источник. "
    "В ответе укажи названия файлов."
)


# Backwards compatibility alias retained for existing callers.
build_rag_config = build_ollama_rag_config


def run_ollama_workflow(
    question: str = DEFAULT_QUESTION,
    *,
    documents_dir: Path = EXAMPLE_DOCUMENTS,
    uploaded_documents: Sequence[Path] | None = None,
    cache_dir: Path | None = None,
    settings: Mapping[str, object] | OllamaSettings | None = None,
) -> dict[str, object]:
    """Delegate to :func:`geomas.core.inference.ollama_client.run_ollama_workflow` with example defaults."""
    return _run_ollama_workflow(
        question,
        documents_dir=documents_dir,
        uploaded_documents=uploaded_documents,
        cache_dir=cache_dir,
        settings=settings,
    )


def ensure_chat_directories() -> None:
    """Ensure chat-specific directories exist for the demo."""
    for path in (CHAT_ROOT, DATA_ROOT, UPLOADS_ROOT, CACHE_ROOT, UPLOADS_CACHE_ROOT, FULL_CACHE_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def _copy_if_missing(source: Path, destination: Path) -> Path:
    """Copy ``source`` to ``destination`` unless the target already exists."""
    if not destination.exists():
        copy2(source, destination)
    return destination


def populate_uploads() -> list[Path]:
    """Copy a subset of the sample corpus into ``UPLOADS_ROOT`` for uploads."""
    ensure_chat_directories()
    layout: dict[str, Sequence[str]] = {
        # f"{str(datetime.now())[:10]}": ["geomas_field_report.html"],
    }
    uploads: list[Path] = []
    for relative_folder, filenames in layout.items():
        target_folder = UPLOADS_ROOT / relative_folder
        target_folder.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            source_path = EXAMPLE_DOCUMENTS / filename
            if not source_path.exists():
                raise FileNotFoundError(f"Sample document is missing: {source_path}")
            uploads.append(_copy_if_missing(source_path, target_folder / filename))
    return uploads


def build_chat_api(
    *,
    cache_dir: Path,
    settings: Mapping[str, object] | OllamaSettings | None = None,
) -> RagApi:
    """Return a :class:`RagApi` configured for the chat-specific environment."""
    ensure_chat_directories()
    cache_dir.mkdir(parents=True, exist_ok=True)
    base_settings = load_ollama_settings()
    if isinstance(settings, OllamaSettings):
        resolved_settings = settings
    elif isinstance(settings, Mapping):
        resolved_settings = base_settings.with_overrides(settings)
    elif settings is None:
        resolved_settings = base_settings
    else:
        raise TypeError("settings must be a mapping or OllamaSettings instance")

    config = build_ollama_rag_config(
        DATA_ROOT,
        cache_dir=cache_dir,
        settings=resolved_settings,
    )
    return RagApi(config=config)


def run_small_rag(
    question: str,
    uploads: Sequence[Path],
    *,
    settings: Mapping[str, object] | OllamaSettings | None = None,
    query_kwargs: Mapping[str, object] | None = None,
) -> tuple[list[ProcessingResult], dict[str, object]]:
    """Run a lightweight RAG workflow that only ingests uploaded artefacts."""
    with build_chat_api(cache_dir=UPLOADS_CACHE_ROOT, settings=settings) as api:
        api.initialize_pipeline()
        workflow = api.run_workflow(
            question,
            uploaded_documents=uploads,
            query_kwargs=query_kwargs,
        )
    ingestions = [
        result for result in workflow.get("uploaded_ingestions", []) if isinstance(result, ProcessingResult)
    ]
    response_payload = workflow.get("response")
    formatted_response = dict(response_payload) if isinstance(response_payload, Mapping) else {}
    return ingestions, formatted_response


def run_full_rag(
    question: str,
    uploads: Sequence[Path],
    *,
    settings: Mapping[str, object] | OllamaSettings | None = None,
    query_kwargs: Mapping[str, object] | None = None,
) -> tuple[bool, list[ProcessingResult], dict[str, object]]:
    """Run a full RAG workflow ingesting the base corpus and uploaded artefacts."""
    with build_chat_api(cache_dir=FULL_CACHE_ROOT, settings=settings) as api:
        initialised = api.initialize_pipeline(documents_path=DATA_ROOT)
        workflow = api.run_workflow(
            question,
            uploaded_documents=uploads,
            query_kwargs=query_kwargs,
        )
    ingestions = [
        result for result in workflow.get("uploaded_ingestions", []) if isinstance(result, ProcessingResult)
    ]
    response_payload = workflow.get("response")
    formatted_response = dict(response_payload) if isinstance(response_payload, Mapping) else {}
    return bool(initialised), ingestions, formatted_response


def main() -> None:
    ensure_chat_directories()
    uploads = populate_uploads()
    question = DEFAULT_QUESTION
    query_kwargs = {"text_top_k": 4, "rerank_top_k": 3}

    small_ingestions, small_response = run_small_rag(
        question,
        uploads,
        settings={"temperature": 0.2},
        query_kwargs=query_kwargs,
    )

    print("=== Small RAG (uploads only) ===")
    print(f"Question: {question}")
    print(f"Uploads ingested: {len(small_ingestions)}")
    print(f"Answer: {small_response.get('answer') or 'No answer returned.'}")
    print("Context snippets:")
    for entry in format_text_context(small_response.get("text_context", [])):
        print(f"- {entry['document']} (score={entry['score']})")
        print(f"  {entry['preview']}")

    full_initialized, full_ingestions, full_response = run_full_rag(
        question,
        uploads,
        settings={"temperature": 0.2},
        query_kwargs=query_kwargs,
    )

    print("\n=== Full RAG (base corpus + uploads) ===")
    print(f"Pipeline initialised with base corpus: {'yes' if full_initialized else 'no'}")
    print(f"Uploads ingested: {len(full_ingestions)}")
    print(f"Answer: {full_response.get('answer') or 'No answer returned.'}")
    print("Context snippets:")
    for entry in format_text_context(full_response.get("text_context", [])):
        print(f"- {entry['document']} (score={entry['score']})")
        print(f"  {entry['preview']}")


if __name__ == "__main__":
    main()
