from __future__ import annotations

import os
import logging
import shutil
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import chromadb

from geomas.api import rag as rag_module
from geomas.api.rag import RagApi
from geomas.core.inference.ollama_client import (
    OllamaSettings,
    build_ollama_rag_config,
    load_ollama_settings,
)
from geomas.core.rag_modules.data_adapter import format_text_context
from geomas.core.rag_modules.database.chroma_db import (
    ChromaDatabaseClient,
    ChromaDatabaseStore,
    PartitionedChromaDatabaseStore,
    ProcessingResult,
)


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


@dataclass(frozen=True, slots=True)
class CollectionTarget:
    collection_name: str
    directory: Path
    scope_label: str

    def as_tuple(self) -> tuple[str, Path, str]:
        return (self.collection_name, self.directory, self.scope_label)


def build_collection_target(
    collection_name: str,
    directory: Path | str,
    scope_label: str | None = None,
) -> CollectionTarget:
    resolved_directory = Path(directory).expanduser().resolve()
    name = str(collection_name).strip()
    if not name:
        raise ValueError("collection_name must be a non-empty string")
    scope = str(scope_label).strip() if isinstance(scope_label, str) else name
    if not scope:
        scope = name
    return CollectionTarget(collection_name=name, directory=resolved_directory, scope_label=scope)


def default_collection_targets(
    chat_id: str,
    *,
    paths: dict[str, Path],
    include_global: bool = True,
    include_chat_local: bool = True,
) -> list[CollectionTarget]:
    targets: list[CollectionTarget] = []
    if include_global:
        targets.append(build_collection_target("global", paths.get("global_rag_dir"), "global"))

    if include_chat_local:
        collection_name = chat_id
        scope_label = f"{collection_name}_local"
        targets.append(
            build_collection_target(collection_name, paths.get("local_rag_dir"), scope_label)
        )
    return targets


def _normalise_collection_specs(
    extra_collections: CollectionTarget
    | Mapping[str, object]
    | Sequence[object]
    | None,
) -> list[CollectionTarget]:
    if extra_collections is None:
        return []

    if isinstance(extra_collections, CollectionTarget):
        return [extra_collections]

    targets: list[CollectionTarget] = []

    def _append_target(name: object, directory: object, scope: object | None) -> None:
        targets.append(build_collection_target(name, directory, scope_label=scope))

    def _coerce_mapping_entry(collection_name: object, payload: object) -> None:
        if isinstance(payload, CollectionTarget):
            targets.append(payload)
            return

        if isinstance(payload, Mapping):
            directory = payload.get("directory", payload.get("path"))
            scope_label = payload.get("scope", payload.get("label"))
            if directory is None:
                raise ValueError(
                    "Collection mapping entries must define a 'directory' or 'path' value"
                )
            _append_target(collection_name, directory, scope_label)
            return

        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
            if len(payload) < 1:
                raise ValueError("Collection descriptor sequences must include a directory path")
            directory = payload[0]
            scope_label = payload[1] if len(payload) > 1 else None
            _append_target(collection_name, directory, scope_label)
            return

        _append_target(collection_name, payload, None)

    if isinstance(extra_collections, Mapping):
        for collection_name, payload in extra_collections.items():
            _coerce_mapping_entry(collection_name, payload)
    elif isinstance(extra_collections, Sequence) and not isinstance(
        extra_collections, (str, bytes, bytearray)
    ):
        for descriptor in extra_collections:
            if isinstance(descriptor, CollectionTarget):
                targets.append(descriptor)
                continue

            if isinstance(descriptor, Mapping):
                collection_name = (
                    descriptor.get("collection")
                    or descriptor.get("collection_name")
                    or descriptor.get("name")
                )
                directory = descriptor.get("directory", descriptor.get("path"))
                scope_label = descriptor.get("scope", descriptor.get("label"))
                if collection_name is None or directory is None:
                    raise ValueError(
                        "Collection descriptor mappings must define 'collection' and 'directory'"
                    )
                _append_target(collection_name, directory, scope_label)
                continue

            if isinstance(descriptor, Sequence) and not isinstance(
                descriptor, (str, bytes, bytearray)
            ):
                if len(descriptor) == 3:
                    collection_name, directory, scope_label = descriptor
                elif len(descriptor) == 2:
                    collection_name, directory = descriptor
                    scope_label = None
                else:
                    raise ValueError(
                        "Collection descriptor tuples must contain 2 or 3 elements"
                    )
                _append_target(collection_name, directory, scope_label)
                continue

            raise TypeError("Unsupported collection descriptor format")
    else:
        raise TypeError("extra_collections must be a mapping, sequence, or CollectionTarget")

    unique: dict[tuple[str, str, Path], CollectionTarget] = {}
    for target in targets:
        key = (target.collection_name, target.scope_label, target.directory)
        unique[key] = target

    return list(unique.values())


def _close_chroma_client(client: object) -> None:
    for attr in ("close", "persist", "stop", "shutdown", "dispose", "release"):
        method = getattr(client, attr, None)
        if callable(method):
            method()
            break


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
    global_rag_dir: Path | str | None = None,
    chat_id: str | None = None,
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


def _normalise_scope_filters(raw_filters: object) -> list[str]:
    if raw_filters is None:
        return []

    def _coerce(value: object) -> list[str]:
        if isinstance(value, str):
            candidate = value.strip()
            return [candidate] if candidate else []
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            collected: list[str] = []
            for entry in value:
                collected.extend(_coerce(entry))
            return collected
        return []

    scopes: list[str] = []
    for scope in _coerce(raw_filters):
        if scope not in scopes:
            scopes.append(scope)
    return scopes


def _extract_scope_labels(metadata: Mapping[str, object] | None) -> list[str]:
    if not isinstance(metadata, Mapping):
        return []

    scope_candidates: list[str] = []

    def _append(values: object) -> None:
        if isinstance(values, str):
            candidate = values.strip()
            if candidate and candidate not in scope_candidates:
                scope_candidates.append(candidate)
        elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            for entry in values:
                _append(entry)

    if "scope" in metadata:
        _append(metadata.get("scope"))
    if "database_scope" in metadata:
        _append(metadata.get("database_scope"))

    return scope_candidates


def initialize_global_rag(
    *,
    paths: dict[str, Path],
    settings: Mapping[str, object] | OllamaSettings | None = None,
) -> None:
    documents_dir = paths.get("documents_dir")
    global_rag_dir = paths.get("global_rag_dir")

    logger.info("Step 1/4: preparing shared corpus at %s", global_rag_dir)

    if not os.listdir(global_rag_dir):
        chroma_client = chromadb.PersistentClient(path=str(global_rag_dir))
        with build_chat_api(
            global_rag_dir=global_rag_dir,
            settings=settings,
        ) as api:
            api.initialize_pipeline(documents_dir)
        _close_chroma_client(chroma_client)

@contextmanager
def create_chat_session(
    *,
    paths: dict[str, Path],
    chat_id: str,
    reset_local_store: bool = False,
    settings: Mapping[str, object] | OllamaSettings | None = None,
    extra_collections: CollectionTarget
    | Mapping[str, object]
    | Sequence[object]
    | None = None,
) -> Iterator[RagApi]:
    logger.info(
        "Step 2/4: preparing chat %s RAG in %s (uploads at %s)",
        chat_id,
        paths.get("local_rag_dir"),
        paths.get("uploads_dir"),
    )

    api = build_chat_api(
        local_rag_dir=paths.get("local_rag_dir"),
        global_rag_dir=paths.get("global_rag_dir"),
        chat_id=paths.get("chat_id"),
        settings=settings,
    )
    auxiliary_stores: list[ChromaDatabaseStore] = []
    registered_scopes: list[tuple[object, str]] = []

    try:
        composite_store = None
        pipeline = getattr(api, "pipeline", None)
        if pipeline is not None:
            composite_store = getattr(pipeline, "store", None)

        collection_targets = _normalise_collection_specs(extra_collections)
        embedding = None
        if pipeline is not None:
            embedding = getattr(pipeline, "embedding_function", None)
        if embedding is None and composite_store is not None:
            embedding = getattr(composite_store, "embedding", None)

        if collection_targets:
            for target in collection_targets:
                target.directory.mkdir(parents=True, exist_ok=True)
                client = ChromaDatabaseClient(
                    mode="persistent",
                    persistent_path=str(target.directory),
                )
                store = ChromaDatabaseStore(
                    client=client,
                    collection_name=target.collection_name,
                    embedding=embedding,
                )
                auxiliary_stores.append(store)
                registered_scopes.append((composite_store, None))
        api.initialize_pipeline()
        yield api
    finally:
        for store in auxiliary_stores:
            store.close()
        if reset_local_store:
            for entry in list(local_store_dir.iterdir()):
                try:
                    if entry.is_dir() and not entry.is_symlink():
                        shutil.rmtree(entry)
                    else:
                        entry.unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning(
                        "Failed to remove stale cache entry %s: %s", entry, exc
                    )
        api.close()


def ingest_local_documents(
    api: RagApi,
    *,
    paths: Mapping[str, Path] | object,
    uploads: Sequence[Path] | None = None,
) -> tuple[list[Path], list[ProcessingResult]]:
    uploads_dir = paths.get("uploads_dir")
    local_rag_dir = paths.get("local_rag_dir")

    uploads_dir.mkdir(parents=True, exist_ok=True)
    local_rag_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Preparing chat-local vector store at %s", local_rag_dir)
    chroma_client = chromadb.PersistentClient(path=str(local_rag_dir))

    try:
        upload_paths, ingestions = step_list_and_ingest_uploads(
            api,
            uploads_dir=uploads_dir,
            uploads=uploads,
        )
    finally:
        _close_chroma_client(chroma_client)

    return upload_paths, ingestions


def answer_with_combined_context(
    api: RagApi,
    question: str,
    *,
    chat_id: str | None = None,
    query_kwargs: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    logger.info(
        "Step 4/4: querying combined context for chat %s and question: %s",
        chat_id or "<unspecified>",
        question,
    )
    payload = dict(query_kwargs or {})
    response = api.ask_question(question, **payload)
    raw_context = response.get("text_context", [])
    metadata_lookup: dict[object, dict[str, object | list[str]]] = {}
    if isinstance(raw_context, Sequence):
        for entry in raw_context:
            metadata = entry[2] if isinstance(entry[2], Mapping) else {}
            identifier = entry[0]
            metadata_lookup.setdefault(identifier, {"metadata": metadata, "scopes": []})
            scopes_bucket = metadata_lookup[identifier].setdefault("scopes", [])
            scopes_from_metadata = _extract_scope_labels(metadata)
            for scope in scopes_from_metadata:
                if scope not in scopes_bucket:
                    scopes_bucket.append(scope)
            if "metadata" not in metadata_lookup[identifier]:
                metadata_lookup[identifier]["metadata"] = metadata
    formatted_rows = format_text_context(raw_context)
    enriched_rows: list[dict[str, object]] = []
    seen_ids: set[object] = set()
    for row in formatted_rows:
        identifier = row.get("id")
        if identifier in seen_ids:
            continue
        seen_ids.add(identifier)
        metadata_entry = metadata_lookup.get(identifier, {})
        scopes_bucket = metadata_entry.get("scopes")
        if not isinstance(scopes_bucket, list):
            scopes_bucket = []
        scopes = [scope for scope in scopes_bucket if isinstance(scope, str)]
        if not scopes:
            metadata_obj = metadata_entry.get("metadata")
            scopes = _extract_scope_labels(metadata_obj if isinstance(metadata_obj, Mapping) else {})
        payload_row = {key: value for key, value in row.items() if key != "id"}
        if not scopes:
            payload_row["database_scope"] = None
        elif len(scopes) == 1:
            payload_row["database_scope"] = scopes[0]
        else:
            payload_row["database_scope"] = list(scopes)
        enriched_rows.append(payload_row)
    response_payload = dict(response)
    response_payload["formatted_context"] = enriched_rows
    return response_payload, enriched_rows

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
    collection_targets = default_collection_targets(chat_id, paths=paths)
    with create_chat_session(
        paths=paths,
        chat_id=chat_id,
        settings=settings_overrides,
        extra_collections=collection_targets,
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

    previous_chat_id = chat_id
    chat_id = "sfasfpkasfka"
    paths["chat_dir"] = Path(f"./data/{chat_id}")
    paths["uploads_dir"] = Path(f"./data/{chat_id}/uploads")
    paths["local_rag_dir"] = Path(f"./data/{chat_id}/.vector-store")
    print(f"Creating new chat: {chat_id}")
    collection_targets = default_collection_targets(chat_id, paths=paths)
    collection_targets.append(
        build_collection_target(
            previous_chat_id,
            Path(f"./data/{previous_chat_id}/.vector-store"),
            f"{previous_chat_id}_local",
        )
    )
    with create_chat_session(
        paths=paths,
        chat_id=chat_id,
        settings=settings_overrides,
        extra_collections=collection_targets,
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

if __name__ == "__main__":
    main()