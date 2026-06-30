"""Configuration management for cementic."""

import os
import sys
from pathlib import Path
from typing import Any, ClassVar

from platformdirs import user_config_dir, user_data_dir
from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
from sqlalchemy.engine import URL, make_url

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python 3.10
    import tomli as tomllib


def resolve_config_path() -> Path | None:
    """Resolve the active config file path, or None if there isn't one.

    Pure given the environment: ``CEMENTIC_CONFIG`` env var, then a project-local
    ``./cementic.toml``, then ``<user_config_dir>/cementic/config.toml``.
    """
    candidates: list[Path] = []
    explicit = os.environ.get("CEMENTIC_CONFIG")
    if explicit:
        candidates.append(Path(explicit))
    candidates.append(Path.cwd() / "cementic.toml")
    candidates.append(Path(user_config_dir("cementic")) / "config.toml")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def default_config_path() -> Path:
    """The canonical location `cementic config init` writes to."""
    return Path(user_config_dir("cementic")) / "config.toml"


def cementic_data_dir(*, ensure_exists: bool = False) -> Path:
    """Return the cementic user-data directory (where artifacts/models live)."""
    return Path(user_data_dir("cementic", ensure_exists=ensure_exists))


def resolve_llama_model_path(model_path: str) -> Path:
    """Resolve a configured llama.cpp model path to a concrete file location.

    The same rule decides where the bootstrapper *downloads* the model and where
    the runtime *loads* it, so the two always agree:

    - an absolute path is honored as-is;
    - a relative path that already exists from the current directory is used as
      given (the in-repo ``./models/...`` development convenience);
    - otherwise it resolves under the cementic data directory, which is where an
      auto-downloaded model is written.
    """
    raw = Path(model_path)
    if raw.is_absolute():
        return raw
    if raw.exists():
        return raw.resolve()
    return (cementic_data_dir() / raw).resolve()


def load_config_file() -> dict[str, Any]:
    """Read the active TOML config file into a dict (empty if none/invalid)."""
    path = resolve_config_path()
    if path is None:
        return {}
    try:
        with open(path, "rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


class _SectionedTomlSource(PydanticBaseSettingsSource):
    """Feed one ``[section]`` of the TOML config file into a settings model.

    Scoped per sub-model and ranked *below* env, so env always overrides the
    file while the file overrides built-in defaults.
    """

    def __init__(self, settings_cls: type[BaseSettings], section: str) -> None:
        super().__init__(settings_cls)
        self._section = section

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        section = load_config_file().get(self._section)
        return dict(section) if isinstance(section, dict) else {}


class _SectionSettings(BaseSettings):
    """Base for config sections: env > config-file section > defaults."""

    #: TOML table this section reads from (e.g. "database").
    _toml_section: ClassVar[str] = ""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources: list[PydanticBaseSettingsSource] = [
            init_settings,
            env_settings,
            dotenv_settings,
        ]
        if cls._toml_section:
            sources.append(_SectionedTomlSource(settings_cls, cls._toml_section))
        sources.append(file_secret_settings)
        return tuple(sources)


class DatabaseConfig(_SectionSettings):
    """PostgreSQL database configuration."""

    model_config = SettingsConfigDict(env_prefix="CEMENTIC_DB_", populate_by_name=True)
    _toml_section = "database"

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    name: str = Field(default="cementic", description="Database name")
    user: str = Field(default="cementic", description="Database user")
    password: SecretStr = Field(
        default=SecretStr("cementic"),
        description="Database password (override CEMENTIC_DB_PASSWORD outside local dev)",
    )
    url_override: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("CEMENTIC_DB_URL"),
        description="Full SQLAlchemy URL; when set it wins over the discrete fields above",
    )

    @property
    def url(self) -> URL:
        """SQLAlchemy database URL (password redacted in repr).

        A ``CEMENTIC_DB_URL`` override takes precedence over the discrete
        host/port/name/user/password fields.
        """
        if self.url_override is not None and self.url_override.get_secret_value():
            return make_url(self.url_override.get_secret_value())
        return URL.create(
            "postgresql",
            username=self.user,
            password=self.password.get_secret_value(),
            host=self.host,
            port=self.port,
            database=self.name,
        )


class LlamaCppConfig(_SectionSettings):
    """llama.cpp configuration."""

    model_config = SettingsConfigDict(env_prefix="CEMENTIC_LLAMA_")
    _toml_section = "llama_cpp"

    model_path: str = Field(
        default="models/nomic-embed-text-v2-moe.Q8_0.gguf",
        description="Path to .gguf model file",
    )
    n_ctx: int = Field(default=512, description="Context window size")
    n_gpu_layers: int = Field(
        default=0,
        description="Number of layers to offload to GPU (-1 for all)",
    )
    embedding_dim: int = Field(
        default=768,
        description="Fallback embedding dimension; the live model's dimension is "
        "probed and stored in the profile when available",
    )
    verbose: bool = Field(default=False, description="Enable verbose output")
    daemon_host: str = Field(default="127.0.0.1", description="llama.cpp daemon host")
    daemon_port: int = Field(default=11555, description="llama.cpp daemon port")
    daemon_start_timeout_seconds: int = Field(
        default=30,
        description="Seconds to wait for the llama.cpp daemon to become ready",
    )
    daemon_autostart: bool = Field(
        default=True,
        description="Automatically start/restart the llama.cpp daemon when a client needs it",
    )
    daemon_pid_file: Path | None = Field(default=None, description="llama.cpp daemon PID path")
    daemon_log_file: Path | None = Field(default=None, description="llama.cpp daemon log path")


class PipelineConfig(_SectionSettings):
    """Pipeline configuration."""

    model_config = SettingsConfigDict(
        env_prefix="CEMENTIC_PIPELINE_",
        populate_by_name=True,
    )
    _toml_section = "pipeline"

    chunk_size: int = Field(default=512, ge=1, description="Tokens per chunk")
    chunk_overlap: int = Field(default=128, ge=0, description="Token overlap between chunks")
    embedding_provider: str = Field(
        default="llama-cpp",
        description="Embedding provider to use (currently llama-cpp)",
    )

    @model_validator(mode="after")
    def _validate_chunk_window(self) -> "PipelineConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class IndexConfig(_SectionSettings):
    """ANN index configuration.

    ``method`` is a serving choice (which index to build/query), not an embedding
    fact — changing it rebuilds an index, never re-embeds. Query-time knobs
    (``hnsw_ef_search``, ``diskann_query_rescore``) are applied per query; the
    others are build-time.
    """

    model_config = SettingsConfigDict(env_prefix="CEMENTIC_INDEX_")
    _toml_section = "index"

    method: str = Field(default="hnsw", description="ANN index method: hnsw or diskann")
    # pgvector HNSW
    hnsw_m: int = Field(default=16, ge=1, description="HNSW max connections per layer (build)")
    hnsw_ef_construction: int = Field(
        default=64, ge=1, description="HNSW build-time candidate list size"
    )
    hnsw_ef_search: int = Field(default=40, ge=1, description="HNSW query-time candidate list size")
    # pgvectorscale DiskANN
    diskann_num_neighbors: int = Field(default=50, ge=1, description="DiskANN graph degree (build)")
    diskann_search_list_size: int = Field(
        default=100, ge=1, description="DiskANN build-time search list size"
    )
    diskann_query_rescore: int = Field(
        default=50, ge=1, description="DiskANN query-time rescore count"
    )

    @field_validator("method")
    @classmethod
    def _validate_method(cls, value: str) -> str:
        if value not in {"hnsw", "diskann"}:
            raise ValueError("index method must be hnsw or diskann")
        return value


class ExtractionConfig(_SectionSettings):
    """Document extraction configuration."""

    model_config = SettingsConfigDict(env_prefix="CEMENTIC_EXTRACT_")
    _toml_section = "extraction"

    backends: dict[str, str] = Field(
        default_factory=dict,
        description="Per-file-type extractor choice, e.g. {'pdf': 'pymupdf4llm'}. "
        "Keys are bare file types; values are registered extractor names. "
        "Unset types fall back to the registry default.",
    )
    use_ocr: bool = Field(default=False, description="Enable OCR when supported")


class StorageConfig(_SectionSettings):
    """Artifact storage configuration."""

    model_config = SettingsConfigDict(env_prefix="CEMENTIC_STORAGE_")
    _toml_section = "storage"

    artifacts_path: Path | None = Field(
        default=None, description="Root path for cached artifacts"
    )


class SourceWatcherConfig(_SectionSettings):
    """Source watcher configuration."""

    model_config = SettingsConfigDict(env_prefix="CEMENTIC_SOURCE_WATCHER_")
    _toml_section = "source_watcher"

    log_file: Path | None = Field(default=None, description="Log file path")
    state_path: Path | None = Field(default=None, description="Source watcher state file path")


class PipelineWorkerConfig(_SectionSettings):
    """Pipeline worker configuration."""

    model_config = SettingsConfigDict(
        env_prefix="CEMENTIC_PIPELINE_WORKER_",
        populate_by_name=True,
    )
    _toml_section = "pipeline_worker"

    log_file: Path | None = Field(default=None, description="Log file path")
    state_path: Path | None = Field(default=None, description="Pipeline worker state file path")
    batch_size: int = Field(
        default=32,
        ge=1,
        le=128,
        description="Number of chunks to embed in one batch",
    )
    poll_interval: float = Field(
        default=1.0,
        description="Seconds between polling for pending chunks",
    )


class BootstrapConfig(_SectionSettings):
    """Runtime bootstrap configuration for the embedding model.

    cementic does not manage containers. Postgres (with pgvector + vectorscale)
    is provisioned externally -- e.g. via the shipped ``compose.yml`` run with
    ``docker compose up -d`` or ``podman compose up -d``, or any Postgres pointed
    at by ``CEMENTIC_DB_URL``.
    """

    model_config = SettingsConfigDict(env_prefix="CEMENTIC_BOOTSTRAP_")
    _toml_section = "bootstrap"

    auto_download_llama_model: bool = Field(
        default=True,
        description="Automatically download the llama.cpp model file if missing",
    )
    llama_model_url: str = Field(
        default=(
            "https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe-GGUF/resolve/main/"
            "nomic-embed-text-v2-moe.Q8_0.gguf"
        ),
        description="Default llama.cpp model download URL",
    )
    llama_model_sha256: str | None = Field(
        default="06e7a7e594a26985523c18383aba4aad39fe6e14f08ffc6ab5b554e1ccdc3cff",
        description="Expected SHA-256 of the llama.cpp model file. Defaults to the digest "
        "of the bundled Nomic model; set empty to disable verification when pointing "
        "llama_model_url at a different file.",
    )


class Config(BaseSettings):
    """Main configuration class."""

    model_config = SettingsConfigDict(
        env_prefix="CEMENTIC_",
    )

    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    llama_cpp: LlamaCppConfig = Field(default_factory=LlamaCppConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    source_watcher: SourceWatcherConfig = Field(default_factory=SourceWatcherConfig)
    pipeline_worker: PipelineWorkerConfig = Field(default_factory=PipelineWorkerConfig)
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Set default paths based on data directory
        data_dir = Path(user_data_dir("cementic", ensure_exists=True))

        if self.source_watcher.log_file is None:
            self.source_watcher.log_file = data_dir / "source_watcher.log"

        if self.source_watcher.state_path is None:
            self.source_watcher.state_path = data_dir / "source_watcher_state.json"

        if self.pipeline_worker.log_file is None:
            self.pipeline_worker.log_file = data_dir / "pipeline_worker.log"

        if self.pipeline_worker.state_path is None:
            self.pipeline_worker.state_path = data_dir / "pipeline_worker_state.json"

        if self.llama_cpp.daemon_pid_file is None:
            self.llama_cpp.daemon_pid_file = data_dir / "llama_cpp_daemon.pid"

        if self.llama_cpp.daemon_log_file is None:
            self.llama_cpp.daemon_log_file = data_dir / "llama_cpp_daemon.log"

        if self.storage.artifacts_path is None:
            self.storage.artifacts_path = data_dir / "artifacts"


def get_config() -> Config:
    """Get or create configuration instance."""
    return Config()
