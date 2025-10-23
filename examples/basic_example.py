from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from geomas.core.rag_modules import rag_pipeline
from geomas.core.rag_modules.data_adapter import format_text_context

StandardRAGPipeline = rag_pipeline.StandardRAGPipeline
from geomas.core.repository.rag_repository import RAGConfig

EXAMPLE_DOCUMENTS = Path(__file__).resolve().parent / "data"
DEFAULT_QUESTION = "Какие руды присутствуют на территории Рудное поле Светлое? Ответь со ссылкой на название источника."


@dataclass(slots=True)
class LMStudioSettings:
    base_url: str
    model: str
    temperature: float = 0.0
    timeout: float | None = None


def load_lmstudio_settings() -> LMStudioSettings:
    load_dotenv()
    base_url = os.getenv("LM_STUDIO_URL")
    model = os.getenv("LM_STUDIO_MODEL")
    temperature = os.getenv("LM_STUDIO_TEMPERATURE")
    timeout = os.getenv("LM_STUDIO_TIMEOUT")
    if not base_url:
        raise RuntimeError("LM Studio base URL could not be determined from the environment")
    if not model:
        raise RuntimeError("LM Studio model must be configured via LM_STUDIO_MODEL")
    return LMStudioSettings(
        base_url=base_url.rstrip("/"),
        model=model,
        temperature=temperature,
        timeout=timeout,
    )


def build_rag_config(
    documents_dir: Path,
    *,
    cache_dir: Path | None = None,
    settings: LMStudioSettings | None = None,
) -> RAGConfig:
    resolved_settings = settings or load_lmstudio_settings()
    persistent_path = cache_dir or (documents_dir / ".vector-store")
    inference_params = {
        "base_url": resolved_settings.base_url,
        "model": resolved_settings.model,
        "temperature": resolved_settings.temperature,
    }
    if resolved_settings.timeout is not None:
        inference_params["timeout"] = resolved_settings.timeout
    overrides = {
        "parsing": {
            "enable_parser": False,
        },
        "database": {
            "client_mode": "persistent",
            "persistent_path": str(persistent_path),
            "collection_name": "geomas",
        },
        "retrieval": {
            "top_k": 5,
            "text_top_k": 5,
            "embedding_model_name": "labse",
        },
        "ranking": {
            "use_llm_reranking": True,
        },
        "inference": {
            "enable_remote_services": True,
            "params": inference_params,
        },
    }
    return RAGConfig.from_mapping(overrides)


def run_basic_workflow(
    question: str = DEFAULT_QUESTION,
    *,
    documents_dir: Path = EXAMPLE_DOCUMENTS,
    settings: LMStudioSettings | None = None,
) -> dict[str, object]:
    config = build_rag_config(documents_dir, settings=settings)
    pipeline = rag_pipeline.create_standard_pipeline(config)
    try:
        ingest_result = rag_pipeline.ingest_documents(pipeline, documents_dir)
        payload = pipeline.query(question, text_top_k=4, rerank_top_k=3)
        return {
            "question": question,
            "ingestion": ingest_result,
            "response": payload,
        }
    finally:
        pipeline.close()


def main() -> None:
    result = run_basic_workflow()
    response = result["response"]
    print(f"Question: {result['question']}")
    print(f"Answer: {response.get('answer') or 'No answer returned.'}")
    # print("\nContext snippets:")
    # for entry in format_text_context(response.get("text_context", [])):
    #     print(f"- {entry['document']} (score={entry['score']})")
    #     print(f"  {entry['preview']}")


if __name__ == "__main__":
    main()

