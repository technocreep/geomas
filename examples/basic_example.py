from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from geomas.core.rag_modules import rag_pipeline
from geomas.core.rag_modules.data_adapter import format_text_context

StandardRAGPipeline = rag_pipeline.StandardRAGPipeline
from geomas.core.repository.rag_repository import RAGConfig

EXAMPLE_DOCUMENTS = Path(__file__).resolve().parent / "data"
DEFAULT_QUESTION = "Какие руды присутствуют на территории Рудное поле Светлое? Ответь со ссылкой на источник."


@dataclass(slots=True)
class LMStudioSettings:
    base_url: str
    model: str
    temperature: float = 0.0
    timeout: float | None = None
    reranker_model: str | None = None
    reranker_inference_kwargs: dict[str, object] = field(default_factory=dict)
    use_llm_reranker: bool = True
    use_chroma_reranker: bool = True
    chroma_function: str | None = None
    chroma_model: str | None = None
    chroma_kwargs: dict[str, object] = field(default_factory=dict)


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalised = value.strip().lower()
    if normalised in {"1", "true", "yes", "on"}:
        return True
    if normalised in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"Environment variable {name} must be a boolean flag, got: {value!r}")


def _json_env(name: str) -> dict[str, object]:
    payload = os.getenv(name)
    if not payload:
        return {}
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(f"Environment variable {name} must contain valid JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Environment variable {name} must contain a JSON object")
    return value


def load_lmstudio_settings() -> LMStudioSettings:
    load_dotenv()
    base_url = os.getenv("LM_STUDIO_URL")
    # if not base_url:
    #     base_url = os.getenv("LM_STUDIO_BASE_URL")
    # if not base_url:
    #     host = os.getenv("LM_STUDIO_HOST")
    #     port = os.getenv("LM_STUDIO_PORT")
    #     if host and port:
    #         base_url = f"http://{host}:{port}"
    model = os.getenv("LM_STUDIO_MODEL")
    temperature = os.getenv("LM_STUDIO_TEMPERATURE")
    timeout = os.getenv("LM_STUDIO_TIMEOUT")
    reranker_model = os.getenv("LM_STUDIO_RERANKER_MODEL")
    if not base_url:
        raise RuntimeError("LM Studio base URL could not be determined from the environment")
    if not model:
        raise RuntimeError("LM Studio model must be configured via LM_STUDIO_MODEL")
    reranker_inference_kwargs = _json_env("LM_STUDIO_RERANKER_INFERENCE_KWARGS")
    use_llm_reranker = _env_flag("GEOMAS_USE_LLM_RERANKER", default=True)
    use_chroma_reranker = _env_flag("GEOMAS_USE_CHROMA_RERANKER", default=True)
    chroma_function = os.getenv("GEOMAS_CHROMA_RERANKER_FUNCTION", default='SentenceTransformersEmbeddingFunction')
    chroma_model = os.getenv("GEOMAS_CHROMA_RERANKER_MODEL", default='labse')
    chroma_kwargs = _json_env("GEOMAS_CHROMA_RERANKER_KWARGS")

    return LMStudioSettings(
        base_url=base_url.rstrip("/"),
        model=model,
        temperature=temperature,
        timeout=timeout,
        reranker_model=reranker_model,
        reranker_inference_kwargs=reranker_inference_kwargs,
        use_llm_reranker=use_llm_reranker,
        use_chroma_reranker=use_chroma_reranker,
        chroma_function=chroma_function,
        chroma_model=chroma_model,
        chroma_kwargs=chroma_kwargs,
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
    reranker_model = resolved_settings.reranker_model or resolved_settings.model
    ranking_params: dict[str, object] = {
        "use_llm_reranking": resolved_settings.use_llm_reranker,
        "chroma": {
            "enabled": resolved_settings.use_chroma_reranker,
        },
    }
    if reranker_model and resolved_settings.use_llm_reranker:
        ranking_params["llm_url"] = reranker_model
    if resolved_settings.reranker_inference_kwargs:
        ranking_params["inference_config"] = dict(resolved_settings.reranker_inference_kwargs)
    if resolved_settings.chroma_function:
        ranking_params["chroma"]["function"] = resolved_settings.chroma_function
    if resolved_settings.chroma_model is not None:
        ranking_params["chroma"]["model_name"] = resolved_settings.chroma_model
    if resolved_settings.chroma_kwargs:
        ranking_params["chroma"]["kwargs"] = dict(resolved_settings.chroma_kwargs)

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
            **ranking_params,
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

