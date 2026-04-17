"""Tests for configuration module."""

import os
from unittest.mock import patch

from cementic.config import Config, DatabaseConfig, LlamaCppConfig, get_config


class TestDatabaseConfig:
    """Test database configuration."""

    def test_default_values(self):
        """Test default database configuration."""
        config = DatabaseConfig()
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.name == "cementic"
        assert config.user == "cementic"
        assert config.password == "cementic"

    def test_database_url(self):
        """Test database URL generation."""
        config = DatabaseConfig(
            host="testhost", port=5433, name="testdb", user="testuser", password="testpass"
        )
        expected_url = "postgresql://testuser:testpass@testhost:5433/testdb"
        assert config.url == expected_url

    def test_environment_override(self):
        """Test environment variable override."""
        with patch.dict(
            os.environ,
            {"CEMENTIC_DB_HOST": "envhost", "CEMENTIC_DB_PORT": "5434"},
        ):
            config = DatabaseConfig()
            assert config.host == "envhost"
            assert config.port == 5434


class TestLlamaCppConfig:
    """Test llama.cpp configuration."""

    def test_default_values(self):
        """Test default llama.cpp configuration."""
        config = LlamaCppConfig()
        assert config.n_ctx == 512
        assert config.n_gpu_layers == 0
        assert config.embedding_dim == 768
        assert config.verbose is False

    def test_model_path(self):
        """Test model path configuration."""
        config = LlamaCppConfig(model_path="/path/to/model.gguf")
        assert config.model_path == "/path/to/model.gguf"


class TestConfig:
    """Test main configuration class."""

    def test_pipeline_env_prefix(self):
        """Pipeline settings should use the pipeline env prefix."""
        with patch.dict(os.environ, {"CEMENTIC_PIPELINE_CHUNK_SIZE": "1024"}):
            config = Config()
            assert config.pipeline.chunk_size == 1024

    def test_pipeline_worker_env_prefix(self):
        """Pipeline worker settings should use the worker env prefix."""
        with patch.dict(os.environ, {"CEMENTIC_PIPELINE_WORKER_BATCH_SIZE": "12"}):
            config = Config()
            assert config.pipeline_worker.batch_size == 12

    def test_config_has_required_sections(self):
        """Test that config has all required sections."""
        config = Config()
        assert hasattr(config, "database")
        assert hasattr(config, "llama_cpp")
        assert hasattr(config, "ollama")
        assert hasattr(config, "pipeline")
        assert hasattr(config, "source_watcher")
        assert hasattr(config, "pipeline_worker")

    def test_extraction_ocr_defaults_off(self):
        """OCR should be opt-in to avoid expensive/sticky extraction by default."""
        config = Config()
        assert config.extraction.use_ocr is False

    def test_bootstrap_defaults_include_vectorscale_image(self):
        """Postgres bootstrap defaults should target the vectorscale image."""
        config = Config()
        assert config.bootstrap.auto_build_postgres_image is True
        assert (
            config.bootstrap.postgres_image
            == "localhost/cementic-postgres-vectorscale:pg18.3-v0.9.0"
        )
        assert config.bootstrap.postgres_base_image == "docker.io/postgres:18.3-bookworm"
        assert config.bootstrap.pgvectorscale_version == "0.9.0"

    def test_bootstrap_defaults_pin_ollama_image(self):
        """Ollama bootstrap defaults should use a pinned image tag."""
        config = Config()
        assert config.bootstrap.ollama_image == "docker.io/ollama/ollama:0.20.6"
        assert not config.bootstrap.ollama_image.endswith(":latest")

    def test_default_paths_set(self, temp_dir):
        """Test that default paths are set."""
        with patch("cementic.config.user_data_dir", return_value=str(temp_dir)):
            config = Config()
            assert config.source_watcher.pid_file is not None
            assert config.pipeline_worker.pid_file is not None
            assert config.llama_cpp.daemon_pid_file is not None
            assert config.llama_cpp.daemon_log_file is not None

    def test_get_config_singleton(self):
        """Test that get_config returns a Config instance."""
        config = get_config()
        assert isinstance(config, Config)

    def test_embedding_provider_selection(self):
        """Test embedding provider configuration."""
        config = Config()
        assert config.pipeline.embedding_provider in ["llama-cpp", "ollama"]
        assert config.pipeline.chunk_size == 512
        assert config.pipeline.chunk_overlap == 128
