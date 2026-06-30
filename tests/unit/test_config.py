"""Tests for configuration module."""

import os
from unittest.mock import patch

import pytest

from cementic.config import (
    Config,
    DatabaseConfig,
    LlamaCppConfig,
    get_config,
    load_config_file,
    resolve_config_path,
)


class TestConfigFile:
    """TOML config file: precedence (defaults < file < env < init) and resolution."""

    def test_file_values_applied(self, tmp_path, monkeypatch) -> None:
        cfg = tmp_path / "cementic.toml"
        cfg.write_text(
            '[database]\nhost = "file-host"\nport = 6000\n[pipeline]\nchunk_size = 999\n'
        )
        monkeypatch.setenv("CEMENTIC_CONFIG", str(cfg))
        monkeypatch.delenv("CEMENTIC_DB_HOST", raising=False)

        config = Config()
        assert config.database.host == "file-host"
        assert config.database.port == 6000
        assert config.pipeline.chunk_size == 999

    def test_env_overrides_file(self, tmp_path, monkeypatch) -> None:
        cfg = tmp_path / "cementic.toml"
        cfg.write_text('[database]\nhost = "file-host"\nport = 6000\n')
        monkeypatch.setenv("CEMENTIC_CONFIG", str(cfg))
        monkeypatch.setenv("CEMENTIC_DB_HOST", "env-host")

        config = Config()
        assert config.database.host == "env-host"  # env wins over file
        assert config.database.port == 6000  # file still fills the gap

    def test_init_kwargs_override_file(self, tmp_path, monkeypatch) -> None:
        cfg = tmp_path / "cementic.toml"
        cfg.write_text("[pipeline]\nchunk_size = 999\n")
        monkeypatch.setenv("CEMENTIC_CONFIG", str(cfg))

        config = Config(pipeline={"chunk_size": 256})
        assert config.pipeline.chunk_size == 256

    def test_defaults_when_no_file(self, monkeypatch) -> None:
        # The autouse fixture already ensures no config file is found.
        monkeypatch.delenv("CEMENTIC_DB_HOST", raising=False)
        config = Config()
        assert config.database.host == "localhost"
        assert config.pipeline.chunk_size == 512

    def test_resolve_path_prefers_explicit_env(self, tmp_path, monkeypatch) -> None:
        path = tmp_path / "explicit.toml"
        path.write_text("")
        monkeypatch.setenv("CEMENTIC_CONFIG", str(path))
        assert resolve_config_path() == path

    def test_resolve_path_project_local(self, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv("CEMENTIC_CONFIG", raising=False)
        monkeypatch.chdir(tmp_path)
        (tmp_path / "cementic.toml").write_text("")
        assert resolve_config_path() == tmp_path / "cementic.toml"

    def test_resolve_path_none_when_absent(self, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv("CEMENTIC_CONFIG", raising=False)
        monkeypatch.chdir(tmp_path)
        assert resolve_config_path() is None

    def test_invalid_toml_is_ignored(self, tmp_path, monkeypatch) -> None:
        bad = tmp_path / "bad.toml"
        bad.write_text("this is := not valid toml")
        monkeypatch.setenv("CEMENTIC_CONFIG", str(bad))
        assert load_config_file() == {}
        # Config falls back to defaults rather than crashing.
        assert Config().database.host == "localhost"

    def test_index_section_defaults(self) -> None:
        config = Config()
        assert config.index.method == "hnsw"
        assert config.index.hnsw_m == 16

    def test_index_section_from_file(self, tmp_path, monkeypatch) -> None:
        cfg = tmp_path / "cementic.toml"
        cfg.write_text('[index]\nmethod = "diskann"\ndiskann_num_neighbors = 64\n')
        monkeypatch.setenv("CEMENTIC_CONFIG", str(cfg))
        config = Config()
        assert config.index.method == "diskann"
        assert config.index.diskann_num_neighbors == 64


class TestDatabaseConfig:
    """Test database configuration."""

    def test_default_values(self):
        """Test default database configuration."""
        with patch.dict(os.environ, {}, clear=True):
            config = DatabaseConfig()
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.name == "cementic"
        assert config.user == "cementic"
        assert config.password.get_secret_value() == "cementic"

    def test_database_url(self):
        """Test database URL generation."""
        config = DatabaseConfig(
            host="testhost", port=5433, name="testdb", user="testuser", password="testpass"
        )
        expected_url = "postgresql://testuser:testpass@testhost:5433/testdb"
        assert config.url.render_as_string(hide_password=False) == expected_url

    def test_environment_override(self):
        """Test environment variable override."""
        with patch.dict(
            os.environ,
            {
                "CEMENTIC_DB_HOST": "envhost",
                "CEMENTIC_DB_PORT": "5434",
                "CEMENTIC_DB_PASSWORD": "test-password",
            },
        ):
            config = DatabaseConfig()
            assert config.host == "envhost"
            assert config.port == 5434
            assert config.password.get_secret_value() == "test-password"


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
        assert hasattr(config, "pipeline")
        assert hasattr(config, "source_watcher")
        assert hasattr(config, "pipeline_worker")

    def test_extraction_ocr_defaults_off(self):
        """OCR should be opt-in to avoid expensive/sticky extraction by default."""
        config = Config()
        assert config.extraction.use_ocr is False

    def test_bootstrap_defaults_are_container_free(self):
        """Bootstrap config carries only model-download settings, no containers."""
        config = Config()
        assert config.bootstrap.auto_download_llama_model is True
        assert not hasattr(config.bootstrap, "postgres_image")
        assert not hasattr(config.bootstrap, "auto_start_infra")

    def test_default_paths_set(self, temp_dir):
        """Test that default paths are set."""
        with patch("cementic.config.user_data_dir", return_value=str(temp_dir)):
            config = Config()
            assert config.source_watcher.log_file is not None
            assert config.pipeline_worker.log_file is not None
            assert config.llama_cpp.daemon_pid_file is not None
            assert config.llama_cpp.daemon_log_file is not None

    def test_get_config_singleton(self):
        """Test that get_config returns a Config instance."""
        config = get_config()
        assert isinstance(config, Config)

    def test_embedding_provider_selection(self):
        """Test embedding provider configuration."""
        config = Config()
        assert config.pipeline.embedding_provider == "llama-cpp"
        assert config.pipeline.chunk_size == 512
        assert config.pipeline.chunk_overlap == 128

    def test_chunk_overlap_must_be_smaller_than_size(self):
        """Pipeline config rejects an overlap that cannot make progress."""
        with pytest.raises(ValueError, match="chunk_overlap must be smaller"):
            Config(pipeline={"chunk_size": 128, "chunk_overlap": 128})

    def test_chunk_size_must_be_positive(self):
        """Pipeline config rejects non-positive chunk sizes."""
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            Config(pipeline={"chunk_size": 0})

    def test_index_method_must_be_supported(self):
        """Index config rejects unknown ANN methods."""
        with pytest.raises(ValueError, match="hnsw or diskann"):
            Config(index={"method": "flat"})

    def test_index_params_must_be_positive(self):
        """Index config rejects non-positive build/query parameters."""
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            Config(index={"hnsw_m": 0})
