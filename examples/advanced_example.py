from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Sequence

import gradio as gr

from geomas.core.rag_modules import rag_pipeline
from geomas.core.rag_modules.database.chroma_db import ProcessingResult
from geomas.core.rag_modules.data_adapter import format_text_context

StandardRAGPipeline = rag_pipeline.StandardRAGPipeline

from examples import basic_example

DEFAULT_EXAMPLES: tuple[str, ...] = (
    "Какие руды присутствуют на территории Рудное поле Светлое? Ответь со ссылкой на название источника.",
)


def _format_rows(text_context: Iterable[Sequence[object]], limit: int = 3) -> str:
    rows = format_text_context(text_context, limit=limit)
    return "\n".join(
        f"[{entry['document']}] score={entry['score']:.3f}: {entry['preview']}"
        for entry in rows
    )


def prepare_pipeline(
    *,
    documents_dir: Path = basic_example.EXAMPLE_DOCUMENTS,
    settings: basic_example.LMStudioSettings | None = None,
) -> tuple[StandardRAGPipeline, ProcessingResult]:
    config = basic_example.build_rag_config(documents_dir, settings=settings)
    pipeline = rag_pipeline.create_standard_pipeline(config)
    try:
        result = rag_pipeline.ingest_documents(pipeline, documents_dir)
    except Exception:
        pipeline.close()
        raise
    return pipeline, result


def create_responder(
    pipeline: StandardRAGPipeline,
    *,
    context_limit: int = 3,
) -> Callable[[str, list[list[str]] | None], str]:
    def respond(message: str, _history: list[list[str]] | None = None) -> str:
        payload = pipeline.query(message, text_top_k=context_limit, rerank_top_k=context_limit)
        answer = payload.get("answer") or "The LM Studio connector did not return an answer."
        # context_summary = _format_rows(payload.get("text_context", []), limit=context_limit)
        # if context_summary:
        #     return f"{answer}\n\nContext:\n{context_summary}"
        return answer
    return respond


def launch_ui(
    *,
    documents_dir: Path = basic_example.EXAMPLE_DOCUMENTS,
    settings: basic_example.LMStudioSettings | None = None,
) -> None:
    pipeline, _ = prepare_pipeline(documents_dir=documents_dir, settings=settings)
    responder = create_responder(pipeline)
    description = (
        "Ask questions about the bundled demo corpus. "
        "GeoMAS handles chunking, retrieval, and reranking; LM Studio generates the final answer."
    )
    try:
        gr.ChatInterface(
            responder,
            title="GeoMAS + LM Studio Demo",
            description=description,
            examples=list(DEFAULT_EXAMPLES),
        ).launch()
    finally:
        pipeline.close()


def create_chat_backend(
    *,
    documents_dir: Path = basic_example.EXAMPLE_DOCUMENTS,
    settings: basic_example.LMStudioSettings | None = None,
) -> tuple[
    StandardRAGPipeline,
    Callable[[str, list[list[str]] | None], str],
    ProcessingResult,
]:
    pipeline, result = prepare_pipeline(documents_dir=documents_dir, settings=settings)
    responder = create_responder(pipeline)
    return pipeline, responder, result

if __name__ == "__main__":
    launch_ui()
