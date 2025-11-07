from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from geomas.core.inference.ollama_client import (
    OllamaSettings,
    build_ollama_rag_config,
    load_ollama_settings,
    run_ollama_workflow as _run_ollama_workflow,
)
from geomas.core.rag_modules.data_adapter import format_text_context

EXAMPLE_DOCUMENTS = Path(__file__).resolve().parent / "data"
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


def main() -> None:
    request = (
        "Какие руды присутствуют на территории Рудное поле Светлое? Ответь со ссылкой на источник. "
        "В ответе укажи названия файлов. Ответ должен быть на русском языке."
    )
    result = run_ollama_workflow(
        question=request,
        settings={"temperature": 0.2},
    )
    response = result["response"]
    print(f"Question: {result['question']}")
    print(f"Answer: {response.get('answer') or 'No answer returned.'}")
    print("\nContext snippets:")
    for entry in format_text_context(response.get("text_context", [])):
        print(f"- {entry['document']} (score={entry['score']})")
        print(f"  {entry['preview']}")


if __name__ == "__main__":
    main()
