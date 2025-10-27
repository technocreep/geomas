from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from geomas.core.rag_modules import rag_pipeline
from geomas.core.rag_modules.data_adapter import format_text_context
from geomas.core.repository.rag_repository import RAGConfig

EXAMPLE_DOCUMENTS = Path(__file__).resolve().parent / "data"
DEFAULT_QUESTION = "Какие руды присутствуют на территории Рудное поле Светлое? Ответь со ссылкой на источник."


@dataclass(slots=True)
class OllamaSettings:
    base_url: str | None
    model: str
    temperature: float = 0.0
    timeout: float | None = None
    system_prompt: str | None = None


def _float_env(name: str, *, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError as exc:  # pragma: no cover - defensive input validation
        raise RuntimeError(f"Environment variable {name} must be a number") from exc


def _optional_float_env(name: str) -> float | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError as exc:  # pragma: no cover - defensive input validation
        raise RuntimeError(f"Environment variable {name} must be a number") from exc


def load_ollama_settings() -> OllamaSettings:
    load_dotenv()
    base_url = os.getenv("OLLAMA_URL") or os.getenv("OLLAMA_BASE_URL")
    if not base_url:
        host = os.getenv("OLLAMA_HOST")
        port = os.getenv("OLLAMA_PORT")
        if host and port:
            base_url = f"http://{host}:{port}"
    if base_url:
        base_url = base_url.rstrip("/")
    model = os.getenv("OLLAMA_MODEL", default='gpt-oss:20b')
    temperature = _float_env("OLLAMA_TEMPERATURE", default=0.0)
    timeout = _optional_float_env("OLLAMA_TIMEOUT")
    system_prompt = os.getenv("OLLAMA_SYSTEM_PROMPT")
    return OllamaSettings(
        base_url=base_url,
        model=model,
        temperature=temperature,
        timeout=timeout,
        system_prompt=system_prompt,
    )


def build_rag_config(
    documents_dir: Path,
    *,
    cache_dir: Path | None = None,
    settings: OllamaSettings | None = None,
) -> RAGConfig:
    resolved_settings = settings or load_ollama_settings()
    persistent_path = cache_dir or (documents_dir / ".vector-store")
    inference_params: dict[str, object] = {
        "provider": "ollama",
        "model": resolved_settings.model,
        "temperature": resolved_settings.temperature,
    }
    if resolved_settings.base_url:
        inference_params["host"] = resolved_settings.base_url
    if resolved_settings.timeout is not None:
        inference_params["timeout"] = resolved_settings.timeout
    if resolved_settings.system_prompt:
        inference_params["system_prompt"] = resolved_settings.system_prompt

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
            "use_llm_reranking": False,
            "chroma": {
                "enabled": True,
            },
        },
        "inference": {
            "enable_remote_services": True,
            "provider": "ollama",
            "params": inference_params,
        },
        "integrations": {
            "enable_ollama": True,
            "ollama_endpoint": resolved_settings.base_url,
        },
    }
    return RAGConfig.from_mapping(overrides)


def run_ollama_workflow(
    question: str = DEFAULT_QUESTION,
    *,
    documents_dir: Path = EXAMPLE_DOCUMENTS,
    settings: OllamaSettings | None = None,
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
    result = run_ollama_workflow()
    response = result["response"]
    print(f"Question: {result['question']}")
    print(f"Answer: {response.get('answer') or 'No answer returned.'}")
    print("\nContext snippets:")
    for entry in format_text_context(response.get("text_context", [])):
        print(f"- {entry['document']} (score={entry['score']})")
        print(f"  {entry['preview']}")


if __name__ == "__main__":
    main()
