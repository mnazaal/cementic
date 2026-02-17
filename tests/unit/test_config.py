"""Tests for configuration module."""

import os
from unittest.mock import patch

from seman.config import Config, DatabaseConfig, LlamaCppConfig, get_config


class TestDatabaseConfig:
    """Test database configuration."""

    def test_default_values(self):
        """Test default database configuration."""
        config = DatabaseConfig()
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.name == "seman"
        assert config.user == "seman"
        assert config.password == "seman"

    def test_database_url(self):
        """Test database URL generation."""
        config = DatabaseConfig(
            host="testhost", port=5433, name="testdb", user="testuser", password="testpass"
        )
        expected_url = "postgresql://testuser:testpass@testhost:5433/testdb"
        assert config.url == expected_url

    def test_environment_override(self):
        """Test environment variable override."""
        with patch.dict(os.environ, {"SEMAN_DB_HOST": "envhost", "SEMAN_DB_PORT": "5434"}):
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

    def test_config_has_required_sections(self):
        """Test that config has all required sections."""
        config = Config()
        assert hasattr(config, "database")
        assert hasattr(config, "llama_cpp")
        assert hasattr(config, "ollama")
        assert hasattr(config, "indexing")
        assert hasattr(config, "converter")
        assert hasattr(config, "embedder")

    def test_default_paths_set(self, temp_dir):
        """Test that default paths are set."""
        with patch("seman.config.user_data_dir", return_value=str(temp_dir)):
            config = Config()
            assert config.indexing.state_path is not None
            assert config.converter.pid_file is not None
            assert config.embedder.pid_file is not None

    def test_get_config_singleton(self):
        """Test that get_config returns a Config instance."""
        config = get_config()
        assert isinstance(config, Config)

    def test_embedder_selection(self):
        """Test embedder configuration."""
        config = Config()
        assert config.indexing.embedder in ["llama-cpp", "ollama"]
        assert config.indexing.chunk_size == 512
        assert config.indexing.chunk_overlap == 128
