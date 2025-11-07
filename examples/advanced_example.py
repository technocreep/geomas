from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Sequence

import gradio as gr

from geomas.api.rag import RagApi
from geomas.core.rag_modules import rag_pipeline
from geomas.core.rag_modules.database.chroma_db import ProcessingResult
from geomas.core.rag_modules.data_adapter import format_text_context

from examples import basic_example

DEFAULT_EXAMPLES: tuple[str, ...] = (
    "Какие руды присутствуют на территории Рудное поле Светлое? Ответь со ссылкой на источник.",
)


def _format_rows(text_context: Iterable[Sequence[object]], limit: int = 4) -> str:
    rows = format_text_context(text_context, limit=limit)
    return "\n".join(
        f"[{entry['document']}] similarity={float(entry['score']):.3f}: {entry['preview']}"
        for entry in rows
    )


def prepare_pipeline(
    *,
    documents_dir: Path = basic_example.EXAMPLE_DOCUMENTS,
    settings: basic_example.LMStudioSettings | None = None,
) -> tuple[RagApi, ProcessingResult]:
    config = basic_example.build_rag_config(documents_dir, settings=settings)
    api = RagApi(config=config)
    override_pipeline = rag_pipeline.create_standard_pipeline(config)
    previous_pipeline = api.pipeline
    api.pipeline = override_pipeline
    api.is_initialized = False
    try:
        try:
            success = api.initialize_pipeline(documents_dir)
            result = override_pipeline.last_ingest_result or ProcessingResult(success=False)
        except TypeError as exc:  # pragma: no cover - compatibility with stub signatures
            if "document_name" not in str(exc):
                raise
            result = rag_pipeline.ingest_documents(override_pipeline, documents_dir)
            success = bool(result.success)
            if result.success:
                api.is_initialized = True
        if not success or not result.success:
            raise RuntimeError(f"Failed to ingest documents from {documents_dir}")
        return api, result
    except Exception:
        api.close()
        raise
    finally:
        if previous_pipeline is not None and previous_pipeline is not override_pipeline:
            try:
                previous_pipeline.close()
            except Exception:  # pragma: no cover - defensive cleanup
                pass


def create_responder(
    api_or_pipeline: RagApi | object,
    *,
    context_limit: int = 3,
) -> Callable[[str, list[list[str]] | None], str]:
    def respond(message: str, _history: list[list[str]] | None = None) -> str:
        if isinstance(api_or_pipeline, RagApi):
            payload = api_or_pipeline.ask_question(
                message,
                text_top_k=context_limit,
                rerank_top_k=context_limit,
            )
        elif hasattr(api_or_pipeline, "query"):
            query_callable = getattr(api_or_pipeline, "query")
            payload = query_callable(
                message,
                text_top_k=context_limit,
                rerank_top_k=context_limit,
            )
        else:
            raise AttributeError("Responder requires a RagApi or pipeline with a query method")
        answer = payload.get("answer") or "The LM Studio connector did not return an answer."
        context_summary = _format_rows(payload.get("text_context", []), limit=context_limit)
        if context_summary:
            return f"{answer}\n\nContext:\n{context_summary}"
        return answer
    return respond


def launch_ui(
    *,
    documents_dir: Path = basic_example.EXAMPLE_DOCUMENTS,
    settings: basic_example.LMStudioSettings | None = None,
) -> None:
    api, _ = prepare_pipeline(documents_dir=documents_dir, settings=settings)
    responder = create_responder(api)
    description = (
        "Ask questions about the bundled demo corpus. "
        "GeoMAS keeps up to four high-similarity chunks (>= 0.85) in context, "
        "and LM Studio generates the final answer."
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


def create_chat_backend(
    *,
    documents_dir: Path = basic_example.EXAMPLE_DOCUMENTS,
    settings: basic_example.LMStudioSettings | None = None,
) -> tuple[
    RagApi,
    Callable[[str, list[list[str]] | None], str],
    ProcessingResult,
]:
    api, result = prepare_pipeline(documents_dir=documents_dir, settings=settings)
    responder = create_responder(api)
    return api, responder, result

if __name__ == "__main__":
    launch_ui()
