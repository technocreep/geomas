from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import gradio as gr

from geomas.core.rag_modules.data_adapter import format_text_context

from examples import basic_example

DEFAULT_EXAMPLES: tuple[str, ...] = (
    "Подробно опиши морфологию рудных тел на территории `Светлое`."
    "Ответь со ссылкой на источник. "
    "В ответе укажи названия файлов.",

    "Подробно опиши морфологию рудных тел на территории `Сенон`."
    "Ответь со ссылкой на источник. "
    "В ответе укажи названия файлов.",

    "Подробно опиши морфологию рудных тел на территории `Сергеевское`."
    "Ответь со ссылкой на источник. "
    "В ответе укажи названия файлов.",
)


def _format_rows(text_context: list[dict[str, object]], limit: int = 4) -> str:
    rows = format_text_context(text_context, limit=limit)
    return "\n".join(
        f"[{entry['document']}] score={float(entry['score']):.3f}: {entry['preview']}"
        for entry in rows
    )


def prepare_chat_backend(
    *,
    chat_id: str,
    include_global: bool,
    reset_local_rag: bool,
    describe_images: bool,
    settings_overrides: dict[str, object],
    documents_dir: Path,
    global_rag_dir: Path,
    uploads_dir: Path,
    local_rag_dir: Path,
) -> tuple[basic_example.RagApi, dict[str, object]]:
    paths = basic_example.build_paths(
        documents_dir=documents_dir,
        global_rag_dir=global_rag_dir,
        chat_dir=f"./data/{chat_id}",
        uploads_dir=uploads_dir,
        local_rag_dir=local_rag_dir,
    )

    basic_example.initialize_global_rag(
        paths=paths,
        settings=settings_overrides,
        describe_images=describe_images,
    )

    collection_targets = basic_example.default_collection_targets(
        chat_id, include_global=include_global
    )
    query_kwargs: dict[str, object] = {
        "text_top_k": 5,
        "rerank_top_k": 5,
        "scopes": collection_targets,
    }

    session = basic_example.build_chat_api(
        local_rag_dir=paths["local_rag_dir"],
        global_rag_dir=paths["global_rag_dir"],
        chat_id=chat_id,
        settings_overrides=settings_overrides,
    )
    session.initialize_pipeline(paths["uploads_dir"], describe_images=describe_images)
    return session, query_kwargs


def create_responder(
    api: basic_example.RagApi,
    *,
    query_kwargs: dict[str, object],
    chat_id: str,
    context_limit: int = 3,
) -> Callable[[str, list[list[str]] | None], str]:
    def respond(message: str, _history: list[list[str]] | None = None) -> str:
        payload = dict(query_kwargs)
        payload.update({"text_top_k": context_limit, "rerank_top_k": context_limit})
        response, _context_rows = basic_example.answer_with_combined_context(
            api,
            message,
            chat_id=chat_id,
            query_kwargs=payload,
        )
        context_summary = _format_rows(response.get("text_context", []), limit=context_limit)
        answer = response.get("answer") or "LM Studio did not return an answer."
        return f"{answer}\n\nContext:\n{context_summary}" if context_summary else answer

    return respond


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chat-id", default="demo-chat", help="Chat identifier for the session")
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
    parser.add_argument("--lm-studio-host", default="localhost", help="LM Studio host")
    parser.add_argument("--lm-studio-port", default="1234", help="LM Studio port")
    parser.add_argument("--lm-studio-model", default=basic_example.DEFAULT_MODEL, help="LM Studio model identifier")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature for LM Studio completions")
    parser.add_argument("--system-prompt", default="Ответ должен быть на русском", help="System prompt passed to LM Studio")
    parser.add_argument("--include-global", action="store_true", help="Search both global and chat-local scopes")
    parser.add_argument("--reset-local-rag", action="store_true", help="Clear chat-local vector store before ingesting")
    parser.add_argument("--describe-images", action="store_true", help="Enable image-description ingestion")
    return parser.parse_args()


def launch_ui() -> None:
    args = parse_args()
    chat_id = args.chat_id
    uploads_dir = Path(args.uploads_dir or f"./data/{chat_id}/uploads")
    local_rag_dir = Path(args.local_rag_dir or f"./data/{chat_id}/.vector-store")

    settings_overrides: dict[str, object] = {
        "base_url": f"http://{args.lm_studio_host}:{args.lm_studio_port}",
        "model": args.lm_studio_model,
        "temperature": args.temperature,
        "system_prompt": args.system_prompt,
    }

    api, query_kwargs = prepare_chat_backend(
        chat_id=chat_id,
        include_global=args.include_global,
        reset_local_rag=args.reset_local_rag,
        describe_images=args.describe_images,
        settings_overrides=settings_overrides,
        documents_dir=Path(args.documents_dir),
        global_rag_dir=Path(args.global_rag_dir),
        uploads_dir=uploads_dir,
        local_rag_dir=local_rag_dir,
    )
    responder = create_responder(api, query_kwargs=query_kwargs, chat_id=chat_id)

    description = (
        "Ask questions about the bundled demo corpus. "
        "GeoMAS keeps up to four high-similarity chunks (>= 0.85) in context, "
        "and LM Studio generates the final answer. Images in the documents "
        "directory are embedded alongside text automatically; enable the image "
        "description flag to capture inline captions during ingestion."
    )

    try:
        gr.ChatInterface(
            responder,
            title="GeoMAS + LM Studio Demo",
            description=description,
            examples=list(DEFAULT_EXAMPLES),
        ).launch()
    finally:
        api.close()


if __name__ == "__main__":
    launch_ui()
