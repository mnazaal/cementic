"""Configuration management for seman."""

from pathlib import Path
from typing import Optional

from platformdirs import user_data_dir
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    """PostgreSQL database configuration."""

    model_config = SettingsConfigDict(env_prefix="SEMAN_DB_")

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    name: str = Field(default="seman", description="Database name")
    user: str = Field(default="seman", description="Database user")
    password: Optional[str] = Field(default="seman", description="Database password")

    @property
    def url(self) -> str:
        """Generate SQLAlchemy database URL."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class LlamaCppConfig(BaseSettings):
    """llama.cpp configuration."""

    model_config = SettingsConfigDict(env_prefix="SEMAN_LLAMA_")

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


class OllamaConfig(BaseSettings):
    """Ollama server configuration."""

    model_config = SettingsConfigDict(env_prefix="SEMAN_OLLAMA_")

    host: str = Field(default="http://localhost:11434", description="Ollama server URL")
    model: str = Field(default="nomic-embed-text", description="Embedding model name")
    embedding_dim: int = Field(default=768, description="Embedding dimension")


class IndexingConfig(BaseSettings):
    """Indexing configuration."""

    model_config = SettingsConfigDict(env_prefix="SEMAN_INDEX_")

    chunk_size: int = Field(default=512, description="Tokens per chunk")
    chunk_overlap: int = Field(default=128, description="Token overlap between chunks")
    embedder: str = Field(
        default="llama-cpp",
        description="Embedder to use (llama-cpp or ollama)",
    )
    state_path: Optional[Path] = Field(default=None, description="State file path")


class ConverterConfig(BaseSettings):
    """Converter daemon configuration."""

    model_config = SettingsConfigDict(env_prefix="SEMAN_CONVERTER_")

    pid_file: Optional[Path] = Field(default=None, description="PID file path")
    log_file: Optional[Path] = Field(default=None, description="Log file path")


class EmbedderConfig(BaseSettings):
    """Embedder daemon configuration."""

    model_config = SettingsConfigDict(env_prefix="SEMAN_EMBEDDER_")

    pid_file: Optional[Path] = Field(default=None, description="PID file path")
    log_file: Optional[Path] = Field(default=None, description="Log file path")
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


class BootstrapConfig(BaseSettings):
    """Runtime bootstrap configuration for infra and models."""

    model_config = SettingsConfigDict(env_prefix="SEMAN_BOOTSTRAP_")

    auto_start_infra: bool = Field(
        default=True,
        description="Automatically start required containers",
    )
    auto_pull_ollama_model: bool = Field(
        default=True,
        description="Automatically pull Ollama model if missing",
    )
    auto_download_llama_model: bool = Field(
        default=True,
        description="Automatically download llama.cpp model if missing",
    )
    postgres_container: str = Field(default="seman-postgres", description="Postgres container name")
    ollama_container: str = Field(default="seman-ollama", description="Ollama container name")
    postgres_image: str = Field(
        default="docker.io/timescale/timescaledb:latest-pg18",
        description="Postgres container image",
    )
    ollama_image: str = Field(
        default="docker.io/ollama/ollama:latest",
        description="Ollama container image",
    )
    wait_timeout_seconds: int = Field(default=90, description="Maximum bootstrap wait time")
    wait_interval_seconds: float = Field(default=2.0, description="Polling interval while waiting")
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
        env_prefix="SEMAN_",
    )

    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    llama_cpp: LlamaCppConfig = Field(default_factory=LlamaCppConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    converter: ConverterConfig = Field(default_factory=ConverterConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set default paths based on data directory
        data_dir = Path(user_data_dir("seman", ensure_exists=True))

        if self.indexing.state_path is None:
            self.indexing.state_path = data_dir / "state.json"

        if self.converter.pid_file is None:
            self.converter.pid_file = data_dir / "converter.pid"

        if self.converter.log_file is None:
            self.converter.log_file = data_dir / "converter.log"

        if self.embedder.pid_file is None:
            self.embedder.pid_file = data_dir / "embedder.pid"

        if self.embedder.log_file is None:
            self.embedder.log_file = data_dir / "embedder.log"


def get_config() -> Config:
    """Get or create configuration instance."""
    return Config()
