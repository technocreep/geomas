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
    run_ollama_workflow as _run_ollama_workflow,
)
from geomas.core.rag_modules.data_adapter import format_text_context
from geomas.core.rag_modules.database.chroma_db import ProcessingResult

BASE_DIR = Path(__file__).resolve().parent
EXAMPLE_DOCUMENTS = BASE_DIR / "data"
DOCUMENTS_ROOT = EXAMPLE_DOCUMENTS
CHAT_ID = "demo-chat"
CHAT_ROOT = BASE_DIR / CHAT_ID
UPLOADS_ROOT = CHAT_ROOT / "uploads"

# Global/static cache for the bundled corpus shared across chats.
GLOBAL_CACHE_ROOT = BASE_DIR / ".vector-store"
# Store base-ingested documents in a dedicated sub-directory.
GLOBAL_CORPUS_CACHE_ROOT = GLOBAL_CACHE_ROOT / "corpus"

# Chat-local cache storing session/uploads specific artefacts.
LOCAL_CACHE_ROOT = CHAT_ROOT / ".vector-store"
LOCAL_UPLOAD_CACHE_ROOT = LOCAL_CACHE_ROOT / "uploads"
DEFAULT_QUESTION = (
    "Какие руды присутствуют на территории Рудное поле Светлое? "
    "Ответь со ссылкой на источник. "
    "В ответе укажи названия файлов."
)

logger = logging.getLogger(__name__)

# Maintain compatibility with the core API helpers.
build_rag_config = build_ollama_rag_config


def run_ollama_workflow(
    question: str = DEFAULT_QUESTION,
    *,
    documents_dir: Path = EXAMPLE_DOCUMENTS,
    uploaded_documents: Sequence[Path] | None = None,
    cache_dir: Path | None = None,
    local_cache_dir: Path | None = None,
    settings: Mapping[str, object] | OllamaSettings | None = None,
) -> dict[str, object]:
    """Delegate to the core workflow helper using example-oriented defaults.

    Args:
        question: Prompt executed against the workflow. Defaults to
            :data:`DEFAULT_QUESTION`.
        documents_dir: Base document directory used for ingestion when the
            pipeline initialises.
        uploaded_documents: Optional iterable of artefacts to ingest alongside
            ``documents_dir``.
        cache_dir: Optional persistent cache location for the shared vector
            store. When ``None`` the core helper falls back to its default
            behaviour.
        local_cache_dir: Optional path configuring a local vector-store cache.
            The value is forwarded directly to
            :func:`geomas.core.inference.ollama_client.build_ollama_rag_config`.
        settings: Optional Ollama configuration overrides applied on top of the
            environment-derived defaults.
    """
    return _run_ollama_workflow(
        question,
        documents_dir=documents_dir,
        uploaded_documents=uploaded_documents,
        cache_dir=cache_dir,
        local_cache_dir=local_cache_dir,
        settings=settings,
    )


def ensure_chat_directories() -> None:
    """Ensure chat-specific directories exist for the demo."""
    for path in (
        CHAT_ROOT,
        UPLOADS_ROOT,
        GLOBAL_CACHE_ROOT,
        GLOBAL_CORPUS_CACHE_ROOT,
        LOCAL_CACHE_ROOT,
        LOCAL_UPLOAD_CACHE_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)

def populate_uploads() -> list[Path]:
    """Ensure directories exist and list absolute file paths in ``UPLOADS_ROOT``.

    Returns:
        A deterministically ordered list of absolute :class:`Path` objects for
        every regular file located beneath :data:`UPLOADS_ROOT`.
    """

    ensure_chat_directories()
    files = [
        path.resolve()
        for path in UPLOADS_ROOT.rglob("*")
        if path.is_file()
    ]
    return sorted(files, key=lambda path: path.as_posix())


def _context_rows(
    response: Mapping[str, object] | None,
    *,
    limit: int = 4,
) -> list[dict[str, object]]:
    """Return formatted context rows enriched with database scope metadata."""
    if not isinstance(response, Mapping):
        return []
    raw_context = response.get("text_context", [])
    if not isinstance(raw_context, Sequence):
        return []

    formatted_rows = format_text_context(raw_context, limit=limit)

    metadata_lookup: dict[object, Mapping[str, object]] = {}
    for entry in raw_context:
        if not isinstance(entry, Sequence) or len(entry) < 3:
            continue
        doc_id = entry[0]
        metadata = entry[2] if isinstance(entry[2], Mapping) else {}
        try:
            metadata_lookup.setdefault(doc_id, metadata)
        except TypeError:
            continue
    enriched_rows: list[dict[str, object]] = []
    for row in formatted_rows:
        row_metadata = metadata_lookup.get(row.get("id"))
        database_scope: str | None = None
        if isinstance(row_metadata, Mapping):
            scope_value = row_metadata.get("database_scope")
            if isinstance(scope_value, str) and scope_value:
                database_scope = scope_value
        row_payload = {key: value for key, value in row.items() if key != "id"}
        row_payload["database_scope"] = database_scope
        enriched_rows.append(row_payload)
    return enriched_rows


def build_chat_api(
    *,
    cache_dir: Path | None,
    local_cache_dir: Path | None,
    settings: Mapping[str, object] | OllamaSettings | None = None,
) -> RagApi:
    """Return a :class:`RagApi` configured for the chat-specific environment.

    Args:
        cache_dir: Persistent cache directory shared across chat sessions. When
            ``None`` the global cache falls back to the workflow default.
        local_cache_dir: Local cache path forwarded to
            :func:`build_ollama_rag_config`. When ``None`` only the global cache
            is configured.
        settings: Optional Ollama overrides applied to the environment-derived
            defaults.
    """
    ensure_chat_directories()
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    if local_cache_dir is not None:
        local_cache_dir.mkdir(parents=True, exist_ok=True)
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
        DOCUMENTS_ROOT,
        cache_dir=cache_dir,
        local_cache_dir=local_cache_dir,
        settings=resolved_settings,
    )
    return RagApi(config=config)


def _ensure_directory(path: Path | None) -> Path | None:
    """Create ``path`` when provided and return the normalised instance."""
    if path is None:
        return None
    normalised = Path(path)
    normalised.mkdir(parents=True, exist_ok=True)
    return normalised


def _store_has_documents(api: RagApi) -> bool:
    """Return ``True`` when the global store already exposes embeddings."""

    pipeline = getattr(api, "pipeline", None)
    if pipeline is None:
        return False

    store = getattr(pipeline, "global_store", None)
    if store is None:
        return False

    collection = getattr(store, "collection", None)
    if collection is None:
        return False

    count = getattr(collection, "count", None)
    if not callable(count):
        return False

    try:
        total = count()
    except Exception:  # pragma: no cover - defensive against client failures
        return False

    try:
        return int(total) > 0
    except (TypeError, ValueError):
        return False


def _documents_available(documents_path: Path) -> bool:
    """Return ``True`` if ``documents_path`` contains at least one file."""

    if not documents_path.exists():
        return False

    if documents_path.is_file():
        return True

    if not documents_path.is_dir():
        return False

    return any(candidate.is_file() for candidate in documents_path.rglob("*"))


def initialize_global_rag(
    *,
    documents_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
    settings: Mapping[str, object] | OllamaSettings | None = None,
) -> bool:
    """Initialise the shared corpus stored in the global vector cache.

    The helper prepares the static dataset once so subsequent chat sessions can
    reuse the cached embeddings without re-ingesting the bundled corpus when it
    is already available.

    Args:
        documents_dir: Directory containing the corpus that should populate the
            shared namespace. Defaults to the example-level corpus located at
            :data:`DOCUMENTS_ROOT` when omitted.
        cache_dir: Persistent cache directory used to store the global vector
            store. Defaults to :data:`GLOBAL_CORPUS_CACHE_ROOT` when ``None``.
        settings: Optional Ollama configuration overrides applied when building
            the shared :class:`RagApi` instance.

    Returns:
        ``True`` when the corpus was successfully ingested into the global
        namespace, ``False`` otherwise.
    """
    ensure_chat_directories()
    documents_path = (
        Path(documents_dir)
        if documents_dir is not None
        else DOCUMENTS_ROOT
    )
    cache_path = (
        _ensure_directory(Path(cache_dir))
        if cache_dir is not None
        else _ensure_directory(GLOBAL_CORPUS_CACHE_ROOT)
    )

    with build_chat_api(
        cache_dir=cache_path,
        local_cache_dir=None,
        settings=settings,
    ) as api:
        if _store_has_documents(api):
            logger.info(
                "Global store already populated; loading cached embeddings without ingestion"
            )
            return bool(api.initialize_pipeline())

        if not _documents_available(documents_path):
            logger.info(
                "No documents found at %s; initialising pipeline without ingestion", documents_path
            )
            return bool(api.initialize_pipeline())

        logger.info(
            "Ingesting example corpus from %s into the global vector store", documents_path
        )
        return bool(api.initialize_pipeline(documents_path=documents_path))


@contextmanager
def create_chat_session(
    *,
    cache_dir: Path | str | None = None,
    local_cache_dir: Path | str | None = LOCAL_UPLOAD_CACHE_ROOT,
    settings: Mapping[str, object] | OllamaSettings | None = None,
) -> Iterator[RagApi]:
    """Yield a ready-to-query :class:`RagApi` bound to chat-specific caches.

    Args:
        cache_dir: Persistent cache directory shared across chat sessions.
            Defaults to :data:`GLOBAL_CORPUS_CACHE_ROOT` when ``None``.
        local_cache_dir: Directory storing chat-local artefacts such as upload
            ingestions. Defaults to :data:`LOCAL_UPLOAD_CACHE_ROOT` when
            omitted.
        settings: Optional Ollama overrides applied when configuring the
            session.

    Yields:
        A :class:`RagApi` instance whose pipeline has been initialised without
        re-ingesting the shared corpus.
    """
    ensure_chat_directories()
    resolved_cache_dir = (
        _ensure_directory(Path(cache_dir))
        if cache_dir is not None
        else _ensure_directory(GLOBAL_CORPUS_CACHE_ROOT)
    )
    resolved_local_cache_dir = _ensure_directory(
        Path(local_cache_dir) if local_cache_dir is not None else None
    )
    with build_chat_api(
        cache_dir=resolved_cache_dir,
        local_cache_dir=resolved_local_cache_dir,
        settings=settings,
    ) as api:
        api.initialize_pipeline()
        yield api


def ingest_local_documents(api: RagApi, uploads: Sequence[Path]) -> list[ProcessingResult]:
    """Ingest uploaded artefacts into the chat-local namespace.

    Args:
        api: Active chat session returned by :func:`create_chat_session`.
        uploads: Collection of artefacts to ingest for the chat.

    Returns:
        A list of :class:`ProcessingResult` entries describing the ingestion
        outcome for each artefact.

    Raises:
        FileNotFoundError: If any provided path does not exist.
    """
    ingestions: list[ProcessingResult] = []
    for document_path in uploads:
        if not document_path.exists():
            raise FileNotFoundError(f"Upload artefact is missing: {document_path}")
        result = api._ingest_path(document_path, namespace="local")
        ingestions.append(result)
    return ingestions


def answer_with_combined_context(
    api: RagApi,
    question: str,
    *,
    query_kwargs: Mapping[str, object] | None = None,
    context_limit: int = 4,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Query the pipeline and return the formatted response context.

    Args:
        api: Active chat session returned by :func:`create_chat_session`.
        question: Natural-language prompt to execute.
        query_kwargs: Optional keyword arguments forwarded to
            :meth:`RagApi.ask_question`.
        context_limit: Maximum number of context rows to format.

    Returns:
        A tuple containing the raw response payload and a list of formatted
        context entries produced by :func:`_context_rows`.
    """
    response = api.ask_question(question, **(dict(query_kwargs) if query_kwargs else {}))
    if not isinstance(response, Mapping):
        raise TypeError("RagApi.ask_question returned an unexpected payload")
    formatted_context = _context_rows(response, limit=context_limit)
    return dict(response), formatted_context


def main() -> None:
    ensure_chat_directories()
    question = DEFAULT_QUESTION
    query_kwargs = {"text_top_k": 4, "rerank_top_k": 3}
    settings_overrides: dict[str, object] = {"temperature": 0.2}

    initialize_global_rag()

    with create_chat_session(settings=settings_overrides) as api:
        uploads = populate_uploads()
        ingestions = ingest_local_documents(api, uploads)

        response, context_rows = answer_with_combined_context(
            api,
            question,
            query_kwargs=query_kwargs,
            context_limit=4,
        )

    print("=== Combined RAG (base corpus + uploads) ===")
    print(f"Uploads ingested: {len(ingestions)}")
    print(f"Answer: {response.get('answer') or 'No answer returned.'}")
    print("Context snippets:")
    for entry in context_rows:
        score = entry.get("score")
        if isinstance(score, (int, float)):
            score_display = f"{float(score):.3f}"
        else:
            score_display = str(score)
        scope = entry.get("database_scope")
        scope_suffix = f", scope={scope}" if scope else ""
        print(f"- {entry.get('document')} (score={score_display}{scope_suffix})")
        print(f"  {entry.get('preview')}")


if __name__ == "__main__":
    main()
