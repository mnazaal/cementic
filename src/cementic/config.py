"""Configuration management for cementic."""

from pathlib import Path
from typing import Any, Optional

from platformdirs import user_data_dir
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    """PostgreSQL database configuration."""

    model_config = SettingsConfigDict(env_prefix="CEMENTIC_DB_")

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    name: str = Field(default="cementic", description="Database name")
    user: str = Field(default="cementic", description="Database user")
    password: Optional[str] = Field(default="cementic", description="Database password")

    @property
    def url(self) -> str:
        """Generate SQLAlchemy database URL."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class LlamaCppConfig(BaseSettings):
    """llama.cpp configuration."""

    model_config = SettingsConfigDict(env_prefix="CEMENTIC_LLAMA_")

    model_path: str = Field(
        default="models/nomic-embed-text-v2-moe.Q8_0.gguf",
        description="Path to .gguf model file",
    )
    n_ctx: int = Field(default=512, description="Context window size")
    n_gpu_layers: int = Field(
        default=0,
        description="Number of layers to offload to GPU (-1 for all)",
    )
    embedding_dim: int = Field(default=768, description="Embedding dimension")
    verbose: bool = Field(default=False, description="Enable verbose output")
    daemon_host: str = Field(default="127.0.0.1", description="llama.cpp daemon host")
    daemon_port: int = Field(default=11555, description="llama.cpp daemon port")
    daemon_start_timeout_seconds: int = Field(
        default=30,
        description="Seconds to wait for the llama.cpp daemon to become ready",
    )
    daemon_pid_file: Optional[Path] = Field(default=None, description="llama.cpp daemon PID path")
    daemon_log_file: Optional[Path] = Field(default=None, description="llama.cpp daemon log path")


class OllamaConfig(BaseSettings):
    """Ollama server configuration."""

    model_config = SettingsConfigDict(env_prefix="CEMENTIC_OLLAMA_")

    host: str = Field(default="http://localhost:11434", description="Ollama server URL")
    model: str = Field(default="nomic-embed-text", description="Embedding model name")
    embedding_dim: int = Field(default=768, description="Embedding dimension")


class PipelineConfig(BaseSettings):
    """Pipeline configuration."""

    model_config = SettingsConfigDict(
        env_prefix="CEMENTIC_PIPELINE_",
        populate_by_name=True,
    )

    chunk_size: int = Field(default=512, description="Tokens per chunk")
    chunk_overlap: int = Field(default=128, description="Token overlap between chunks")
    embedding_provider: str = Field(
        default="llama-cpp",
        description="Embedding provider to use (llama-cpp or ollama)",
    )


class ExtractionConfig(BaseSettings):
    """Document extraction configuration."""

    model_config = SettingsConfigDict(env_prefix="CEMENTIC_EXTRACT_")

    backend: str = Field(default="pymupdf4llm", description="Extractor backend name")
    use_ocr: bool = Field(default=False, description="Enable OCR when supported")


class StorageConfig(BaseSettings):
    """Artifact storage configuration."""

    model_config = SettingsConfigDict(env_prefix="CEMENTIC_STORAGE_")

    artifacts_path: Optional[Path] = Field(
        default=None, description="Root path for cached artifacts"
    )


class SourceWatcherConfig(BaseSettings):
    """Source watcher configuration."""

    model_config = SettingsConfigDict(env_prefix="CEMENTIC_SOURCE_WATCHER_")

    pid_file: Optional[Path] = Field(default=None, description="PID file path")
    log_file: Optional[Path] = Field(default=None, description="Log file path")
    state_path: Optional[Path] = Field(default=None, description="Source watcher state file path")


class PipelineWorkerConfig(BaseSettings):
    """Pipeline worker configuration."""

    model_config = SettingsConfigDict(
        env_prefix="CEMENTIC_PIPELINE_WORKER_",
        populate_by_name=True,
    )

    pid_file: Optional[Path] = Field(default=None, description="PID file path")
    log_file: Optional[Path] = Field(default=None, description="Log file path")
    state_path: Optional[Path] = Field(default=None, description="Pipeline worker state file path")
    max_workers: int = Field(
        default=1,
        description="Number of concurrent embedding workers",
    )
    batch_size: int = Field(
        default=32,
        description="Number of chunks to embed in one batch",
    )
    poll_interval: float = Field(
        default=1.0,
        description="Seconds between polling for pending chunks",
    )
    processing_stale_seconds: int = Field(
        default=30,
        description="Seconds after which processing chunks are reset to pending",
    )


class BootstrapConfig(BaseSettings):
    """Runtime bootstrap configuration for infra and models."""

    model_config = SettingsConfigDict(env_prefix="CEMENTIC_BOOTSTRAP_")

    auto_start_infra: bool = Field(
        default=True,
        description="Automatically start required containers",
    )
    auto_build_postgres_image: bool = Field(
        default=True,
        description="Automatically build the local Postgres image with vectorscale",
    )
    auto_pull_ollama_model: bool = Field(
        default=True,
        description="Automatically pull Ollama model if missing",
    )
    auto_download_llama_model: bool = Field(
        default=True,
        description="Automatically download llama.cpp model if missing",
    )
    postgres_container: str = Field(
        default="cementic-postgres", description="Postgres container name"
    )
    ollama_container: str = Field(default="cementic-ollama", description="Ollama container name")
    postgres_image: str = Field(
        default="localhost/cementic-postgres-vectorscale:pg18.3-v0.9.0",
        description="Postgres container image",
    )
    postgres_base_image: str = Field(
        default="docker.io/postgres:18.3-bookworm",
        description="Base Postgres image used to build the pgvector + vectorscale image",
    )
    pgvectorscale_version: str = Field(
        default="0.9.0",
        description="Pinned pgvectorscale release used in the Postgres image build",
    )
    ollama_image: str = Field(
        default="docker.io/ollama/ollama:0.20.6",
        description="Ollama container image",
    )
    wait_timeout_seconds: int = Field(default=90, description="Maximum bootstrap wait time")
    wait_interval_seconds: float = Field(default=2.0, description="Polling interval while waiting")
    postgres_data_path: Optional[Path] = Field(
        default=None, description="Path for postgres data volume"
    )
    ollama_data_path: Optional[Path] = Field(
        default=None, description="Path for ollama data volume"
    )
    llama_model_url: str = Field(
        default=(
            "https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe-GGUF/resolve/main/"
            "nomic-embed-text-v2-moe.Q8_0.gguf"
        ),
        description="Default llama.cpp model download URL",
    )


class Config(BaseSettings):
    """Main configuration class."""

    model_config = SettingsConfigDict(
        env_prefix="CEMENTIC_",
    )

    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    llama_cpp: LlamaCppConfig = Field(default_factory=LlamaCppConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    source_watcher: SourceWatcherConfig = Field(default_factory=SourceWatcherConfig)
    pipeline_worker: PipelineWorkerConfig = Field(default_factory=PipelineWorkerConfig)
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Set default paths based on data directory
        data_dir = Path(user_data_dir("cementic", ensure_exists=True))

        if self.source_watcher.pid_file is None:
            self.source_watcher.pid_file = data_dir / "source_watcher.pid"

        if self.source_watcher.log_file is None:
            self.source_watcher.log_file = data_dir / "source_watcher.log"

        if self.source_watcher.state_path is None:
            self.source_watcher.state_path = data_dir / "source_watcher_state.json"

        if self.pipeline_worker.pid_file is None:
            self.pipeline_worker.pid_file = data_dir / "pipeline_worker.pid"

        if self.pipeline_worker.log_file is None:
            self.pipeline_worker.log_file = data_dir / "pipeline_worker.log"

        if self.pipeline_worker.state_path is None:
            self.pipeline_worker.state_path = data_dir / "pipeline_worker_state.json"

        if self.llama_cpp.daemon_pid_file is None:
            self.llama_cpp.daemon_pid_file = data_dir / "llama_cpp_daemon.pid"

        if self.llama_cpp.daemon_log_file is None:
            self.llama_cpp.daemon_log_file = data_dir / "llama_cpp_daemon.log"

        if self.bootstrap.postgres_data_path is None:
            self.bootstrap.postgres_data_path = data_dir / "postgres-data"

        if self.bootstrap.ollama_data_path is None:
            self.bootstrap.ollama_data_path = data_dir / "ollama-data"

        if self.storage.artifacts_path is None:
            self.storage.artifacts_path = data_dir / "artifacts"


def get_config() -> Config:
    """Get or create configuration instance."""
    return Config()
