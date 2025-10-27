from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Type, TypeVar

import yaml

from geomas.core.repository.constant_repository import SUMMARY_LLM_URL
from geomas.core.repository.parsing_repository import ChunkingParamsConfig


@dataclass(slots=True)
class ParsingConfigTemplate:
    """Parsing section template."""

    chunk_size: int = ChunkingParamsConfig.max_chunk_size
    chunk_overlap: int = ChunkingParamsConfig.chunk_overlap
    separators: list[str] = field(
        default_factory=lambda: list(ChunkingParamsConfig.separators)
    )
    headers_to_split_on: list[tuple[str, str]] = field(
        default_factory=lambda: list(ChunkingParamsConfig.headers_to_split_on)
    )
    elements_to_preserve: list[str] = field(
        default_factory=lambda: list(ChunkingParamsConfig.elements_to_preserve)
    )
    preserve_images: bool = ChunkingParamsConfig.preserve_images
    use_llm: bool = False
    enable_parser: bool = True

    def chunking_parameters(self) -> Dict[str, Any]:
        """Return chunking configuration with explicit splitter overrides."""

        base = {
            "chunk_size": self.chunk_size,
            "max_chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "separators": list(self.separators),
            "headers_to_split_on": list(self.headers_to_split_on),
            "elements_to_preserve": list(self.elements_to_preserve),
            "preserve_images": self.preserve_images,
        }

        defaults = {
            "chunk_size": self.chunk_size,
            "max_chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }

        html_overrides = {
            "chunk_size": self.chunk_size,
            "max_chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "headers_to_split_on": list(self.headers_to_split_on),
            "elements_to_preserve": list(self.elements_to_preserve),
            "preserve_images": self.preserve_images,
        }

        markdown_overrides = {
            "chunk_size": self.chunk_size,
            "max_chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "separators": list(self.separators),
        }

        structured = dict(base)
        structured["defaults"] = defaults
        structured["html"] = html_overrides
        structured["markdown"] = markdown_overrides
        return structured

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enable_parser": self.enable_parser,
            "use_llm": self.use_llm,
            "chunking": self.chunking_parameters(),
        }


@dataclass(slots=True)
class DatabaseConfigTemplate:
    """Database section template."""

    collection_name: str = "geomas_text_documents"
    client_mode: str = "persistent"
    persistent_path: str = field(
        default_factory=lambda: str(Path.home() / ".cache" / "geomas" / "chroma")
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "collection_name": self.collection_name,
            "client_mode": self.client_mode,
            "persistent_path": self.persistent_path,
        }


@dataclass(slots=True)
class VectorStoreConfigTemplate:
    """Vector store section template."""

    type: str = "chroma"
    client: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "client": dict(self.client),
        }


@dataclass(slots=True)
class RetrievalConfigTemplate:
    """Retrieval section template."""

    top_k: int = 5
    filters: Dict[str, Any] = field(default_factory=dict)
    embedding_model_name: str | None = "all-MiniLM-L6-v2"
    final_top_k: int | None = None
    text_top_k: int | None = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "top_k": self.top_k,
            "filters": dict(self.filters),
            "embedding_model_name": self.embedding_model_name,
        }
        if self.final_top_k is not None:
            data["final_top_k"] = self.final_top_k
        if self.text_top_k is not None:
            data["text_top_k"] = self.text_top_k
        return data


@dataclass(slots=True)
class ChromaRankingConfigTemplate:
    """Configuration template controlling :class:`ChromaReranker`.

    Attributes
    ----------
    enabled:
        Flag enabling Chroma-based reranking when ``True``.
    function:
        Name of the embedding function exposed by
        ``chromadb.utils.embedding_functions``. ``None`` preserves the default.
    model_name:
        Optional model identifier forwarded to the embedding constructor.
    kwargs:
        Arbitrary keyword arguments supplied to the embedding constructor.
    """

    enabled: bool = False
    function: str | None = None
    model_name: str | None = None
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"enabled": self.enabled}
        if self.function is not None:
            data["function"] = self.function
        if self.model_name is not None:
            data["model_name"] = self.model_name
        if self.kwargs:
            data["kwargs"] = dict(self.kwargs)
        return data

    def to_overrides(self) -> Dict[str, Any]:
        """Return Chroma reranker overrides compatible with ``ChromaReranker``."""

        overrides: Dict[str, Any] = {}
        if self.function:
            overrides["embedding_function_name"] = self.function
        if self.model_name is not None:
            overrides["embedding_model_name"] = self.model_name
        if self.kwargs:
            overrides["embedding_function_kwargs"] = dict(self.kwargs)
        return overrides


@dataclass(slots=True)
class RankingConfigTemplate:
    """Ranking section template.

    The template controls both LLM-based and Chroma-based rerankers. The
    ``chroma`` attribute exposes :class:`ChromaRankingConfigTemplate`, providing
    strongly typed accessors for embedding selection.
    """

    use_llm_reranking: bool = False
    llm_url: str = SUMMARY_LLM_URL
    inference_config: Dict[str, Any] = field(default_factory=dict)
    chroma: ChromaRankingConfigTemplate = field(
        default_factory=ChromaRankingConfigTemplate
    )

    @property
    def use_chroma_reranking(self) -> bool:
        return self.chroma.enabled

    @use_chroma_reranking.setter
    def use_chroma_reranking(self, value: bool) -> None:
        self.chroma.enabled = bool(value)

    def to_dict(self) -> Dict[str, Any]:
        chroma_config = self.chroma.to_dict()
        return {
            "use_llm_reranking": self.use_llm_reranking,
            "llm_url": self.llm_url,
            "inference_config": dict(self.inference_config),
            "use_chroma_reranking": self.use_chroma_reranking,
            "chroma": chroma_config,
        }


@dataclass(slots=True)
class DataConfigTemplate:
    """Data section template."""

    loader_type: str = "auto"
    loader_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loader_type": self.loader_type,
            "loader_params": dict(self.loader_params),
        }


@dataclass(slots=True)
class InferenceConfigTemplate:
    """Inference section template for remote LLM providers.

    ``provider`` or ``service`` may be set to ``"lm_studio"`` (default) or
    ``"ollama"`` to control which chat client :class:`StandardRAGPipeline`
    initialises. The ``params`` mapping is forwarded to the selected connector.
    """

    enable_remote_services: bool = True
    provider: str | None = None
    service: str | None = None
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "enable_remote_services": self.enable_remote_services,
            "params": dict(self.params),
        }
        if self.provider is not None:
            data["provider"] = self.provider
        if self.service is not None:
            data["service"] = self.service
        return data


@dataclass(slots=True)
class MonitoringConfigTemplate:
    """Monitoring section template."""

    enable_tracing: bool = False
    exporters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enable_tracing": self.enable_tracing,
            "exporters": dict(self.exporters),
        }


@dataclass(slots=True)
class IntegrationsConfigTemplate:
    """Integration hooks for external tooling."""

    enable_ollama: bool = False
    ollama_endpoint: str | None = None
    enable_gradio: bool = False
    gradio_launch_kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enable_ollama": self.enable_ollama,
            "ollama_endpoint": self.ollama_endpoint,
            "enable_gradio": self.enable_gradio,
            "gradio_launch_kwargs": dict(self.gradio_launch_kwargs),
        }


@dataclass(slots=True)
class RAGConfigTemplate:
    """Container aggregating all RAG configuration sections."""

    parsing: ParsingConfigTemplate = field(default_factory=ParsingConfigTemplate)
    database: DatabaseConfigTemplate = field(default_factory=DatabaseConfigTemplate)
    vector_store: VectorStoreConfigTemplate = field(default_factory=VectorStoreConfigTemplate)
    retrieval: RetrievalConfigTemplate = field(default_factory=RetrievalConfigTemplate)
    ranking: RankingConfigTemplate = field(default_factory=RankingConfigTemplate)
    data: DataConfigTemplate = field(default_factory=DataConfigTemplate)
    inference: InferenceConfigTemplate = field(default_factory=InferenceConfigTemplate)
    monitoring: MonitoringConfigTemplate = field(default_factory=MonitoringConfigTemplate)
    integrations: IntegrationsConfigTemplate = field(default_factory=IntegrationsConfigTemplate)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parsing": self.parsing.to_dict(),
            "database": self.database.to_dict(),
            "vector_store": self.vector_store.to_dict(),
            "retrieval": self.retrieval.to_dict(),
            "ranking": self.ranking.to_dict(),
            "data": self.data.to_dict(),
            "inference": self.inference.to_dict(),
            "monitoring": self.monitoring.to_dict(),
            "integrations": self.integrations.to_dict(),
        }


def get_default_rag_template() -> RAGConfigTemplate:
    """Return the strongly-typed default RAG configuration template."""

    return RAGConfigTemplate()


def get_default_rag_config() -> Dict[str, Any]:
    """Return the default RAG configuration as a serialisable mapping."""

    return get_default_rag_template().to_dict()


def _deep_update(
    original: MutableMapping[str, Any], updates: Mapping[str, Any]
) -> MutableMapping[str, Any]:
    """Recursively merge ``updates`` into ``original`` and return ``original``."""

    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(original.get(key), Mapping):
            _deep_update(original[key], value)
        else:
            original[key] = value
    return original


TemplateT = TypeVar("TemplateT", bound=object)


def _apply_template_updates(template: TemplateT, updates: Mapping[str, Any]) -> TemplateT:
    """Recursively apply ``updates`` onto ``template`` and return ``template``."""

    for key, value in updates.items():
        if isinstance(template, ParsingConfigTemplate) and key == "chunking" and isinstance(value, Mapping):
            _apply_template_updates(template, value)
            continue
        if isinstance(template, RankingConfigTemplate) and key == "use_chroma_reranking":
            template.use_chroma_reranking = bool(value)
            continue
        if isinstance(template, ChromaRankingConfigTemplate):
            if key in {"enabled", "use_chroma_reranking"}:
                template.enabled = bool(value)
                continue
            if key in {"function", "embedding_function_name", "embedding_function"}:
                template.function = None if value is None else str(value)
                continue
            if key in {"model_name", "embedding_model_name"}:
                template.model_name = None if value is None else str(value)
                continue
            if key in {"kwargs", "embedding_function_kwargs"}:
                template.kwargs.clear()
                if isinstance(value, Mapping):
                    template.kwargs.update(deepcopy(value))
                continue
        if not hasattr(template, key):
            continue
        current = getattr(template, key)
        if is_dataclass(current) and isinstance(value, Mapping):
            _apply_template_updates(current, value)
        elif isinstance(current, dict) and isinstance(value, Mapping):
            current.clear()
            current.update(deepcopy(value))
        elif isinstance(current, list) and isinstance(value, list):
            current.clear()
            current.extend(deepcopy(value))
        else:
            setattr(template, key, deepcopy(value))
    return template


def _build_template(
    template_cls: Type[TemplateT], data: Mapping[str, Any] | None
) -> TemplateT:
    """Instantiate ``template_cls`` using ``data`` for overrides."""

    template = template_cls()
    if not data:
        return template
    return _apply_template_updates(template, data)


def _clone_template(template: TemplateT) -> TemplateT:
    """Return a deep copy of ``template`` preserving dataclass types."""

    if hasattr(template, "to_dict"):
        return _build_template(type(template), template.to_dict())
    return deepcopy(template)


@dataclass(slots=True)
class RAGConfig:
    """Light-weight configuration container for :class:`StandardRAGPipeline`."""

    _template: RAGConfigTemplate

    @classmethod
    def default(cls) -> "RAGConfig":
        """Return the opinionated default configuration."""

        return cls(get_default_rag_template())

    @classmethod
    def from_mapping(cls, overrides: Mapping[str, Any] | None) -> "RAGConfig":
        """Create a configuration using ``overrides`` on top of defaults."""

        if not overrides:
            return cls.default()

        merged = get_default_rag_config()
        _deep_update(merged, dict(overrides))
        template = _build_template(RAGConfigTemplate, merged)
        return cls(template)

    @classmethod
    def from_template(cls, template: RAGConfigTemplate) -> "RAGConfig":
        """Wrap a :class:`RAGConfigTemplate` inside ``RAGConfig``."""

        return cls(_clone_template(template))

    @classmethod
    def from_path(cls, path: str | Path) -> "RAGConfig":
        """Create a configuration from a YAML/JSON file."""

        config_path = Path(path).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as file:
            if config_path.suffix.lower() in {".yaml", ".yml"}:
                data = yaml.safe_load(file) or {}
            else:
                data = json.load(file)

        if not isinstance(data, Mapping):
            raise ValueError("Configuration file must define a mapping of sections")

        return cls.from_mapping(data)

    @classmethod
    def ensure(
        cls,
        config: Mapping[str, Any] | "RAGConfig" | RAGConfigTemplate | None,
    ) -> "RAGConfig":
        """Normalise supported configuration inputs into ``RAGConfig``."""

        if config is None:
            return cls.default()
        if isinstance(config, cls):
            return cls.from_mapping(config.to_dict())
        if isinstance(config, RAGConfigTemplate):
            return cls.from_template(config)
        if isinstance(config, Mapping):
            return cls.from_mapping(config)
        if hasattr(config, "to_dict") and callable(getattr(config, "to_dict")):
            return cls.from_mapping(config.to_dict())
        raise TypeError(f"Unsupported configuration type: {type(config)!r}")

    def to_dict(self) -> Dict[str, Any]:
        """Return a deep copy of the configuration as a dictionary."""

        return self._template.to_dict()

    def copy(self) -> "RAGConfig":
        """Return an independent copy of the configuration."""

        return RAGConfig.from_mapping(self.to_dict())

    def as_template(self) -> RAGConfigTemplate:
        """Return a strongly typed template representing the configuration."""

        return _clone_template(self._template)

    @property
    def parsing(self) -> Dict[str, Any]:
        return self._template.parsing.to_dict()

    @property
    def database(self) -> Dict[str, Any]:
        return self._template.database.to_dict()

    @property
    def vector_store(self) -> Dict[str, Any]:
        return self._template.vector_store.to_dict()

    @property
    def retrieval(self) -> Dict[str, Any]:
        return self._template.retrieval.to_dict()

    @property
    def ranking(self) -> Dict[str, Any]:
        return self._template.ranking.to_dict()

    @property
    def data(self) -> Dict[str, Any]:
        return self._template.data.to_dict()

    @property
    def inference(self) -> Dict[str, Any]:
        return self._template.inference.to_dict()

    @property
    def monitoring(self) -> Dict[str, Any]:
        return self._template.monitoring.to_dict()

    @property
    def integrations(self) -> Dict[str, Any]:
        return self._template.integrations.to_dict()
