from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from geomas.api.rag import RagApi
from geomas.core.rag_modules import rag_pipeline
from geomas.core.rag_modules.data_adapter import format_text_context
from geomas.core.repository.rag_repository import RAGConfig

# Custom corpora can be nested or symlinked beneath this folder for ingestion.
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
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Environment variable {name} must contain valid JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Environment variable {name} must contain a JSON object")
    return value


def load_lmstudio_settings() -> LMStudioSettings:
    load_dotenv()
    base_url = os.getenv("LM_STUDIO_URL")
    if not base_url:
        base_url = os.getenv("LM_STUDIO_BASE_URL")
    if not base_url:
        host = os.getenv("LM_STUDIO_HOST")
        port = os.getenv("LM_STUDIO_PORT")
        if host and port:
            base_url = f"http://{host}:{port}"
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
    chroma_function = os.getenv("GEOMAS_CHROMA_RERANKER_FUNCTION")
    chroma_model = os.getenv("GEOMAS_CHROMA_RERANKER_MODEL")
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
    """Return a :class:`RAGConfig` wired for the bundled demos.

    The ranking section enables both rerankers by default. Override the knobs in
    ``settings`` or via environment variables to match your deployment:

    .. code-block:: python

        ranking_overrides = {
            "use_llm_reranking": True,
            "llm_url": "http://localhost:1234/v1/rerank",
            "inference_config": {"model": "reranker-model"},
            "chroma": {
                "enabled": True,
                "function": "SentenceTransformerEmbeddingFunction",
                "model_name": "all-MiniLM-L6-v2",
            },
        }

    Retrieval keeps the global similarity threshold of 0.5 unless you supply an
    override. This demo raises it to 0.85 so the console output focuses on
    high-confidence matches while still demonstrating how to customise the
    behaviour.

    Export ``GEOMAS_USE_LLM_RERANKER`` or ``GEOMAS_USE_CHROMA_RERANKER`` with a
    boolean value (``true``/``false``) to toggle each reranker without changing
    the code. ``GEOMAS_CHROMA_RERANKER_FUNCTION``,
    ``GEOMAS_CHROMA_RERANKER_MODEL``, and
    ``GEOMAS_CHROMA_RERANKER_KWARGS`` customise the Chroma reranking pipeline.
    ``LM_STUDIO_RERANKER_INFERENCE_KWARGS`` injects JSON overrides into the LLM
    reranker connector.
    """
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
            "chunk_limit": 4,
            "score_threshold": 0.85,  # default is 0.5; raise to emphasise top matches
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
    with RagApi(config=config) as api:
        override_pipeline = rag_pipeline.create_standard_pipeline(config)
        previous_pipeline = api.pipeline
        api.pipeline = override_pipeline
        api.is_initialized = False
        query_kwargs = {
            "text_top_k": 4,
            "rerank_top_k": 3,
        }
        def _coerce_result(raw: object) -> object:
            if hasattr(raw, "success"):
                return raw
            candidate = getattr(override_pipeline, "last_ingest_result", None)
            if candidate is not None and hasattr(candidate, "success"):
                return candidate
            return rag_pipeline.ProcessingResult(success=bool(raw))
        try:
            try:
                workflow = api.run_workflow(
                    question,
                    documents_path=documents_dir,
                    query_kwargs=query_kwargs,
                )
            except TypeError as exc:
                if "document_name" not in str(exc):
                    raise
                base_ingestion_raw = rag_pipeline.ingest_documents(override_pipeline, documents_dir)
                base_ingestion = _coerce_result(base_ingestion_raw)
                if bool(getattr(base_ingestion, "success", base_ingestion)):
                    api.is_initialized = True
                response = api.ask_question(question, **query_kwargs)
                workflow = {
                    "base_ingestion": base_ingestion,
                    "uploaded_ingestions": [],
                    "response": response,
                }
            return {
                "question": question,
                "ingestion": workflow.get("base_ingestion"),
                "response": workflow["response"],
            }
        finally:
            if previous_pipeline is not None and previous_pipeline is not override_pipeline:
                try:
                    previous_pipeline.close()
                except Exception:
                    pass


def main() -> None:
    result = run_basic_workflow()
    response = result["response"]
    print(f"Question: {result['question']}")
    print(f"Answer: {response.get('answer') or 'No answer returned.'}")
    print("\nContext snippets:")
    for entry in format_text_context(response.get("text_context", [])):
        similarity = float(entry["score"])
        print(f"- {entry['document']} (similarity={similarity:.3f})")
        print(f"  {entry['preview']}")


if __name__ == "__main__":
    main()

