"""Tests for configuration module."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from cementic.config import (
    Config,
    ConfigError,
    DatabaseConfig,
    LlamaCppConfig,
    format_config_error,
    get_config,
    load_config_file,
    resolve_config_path,
    resolve_llama_model_path,
)
from cementic.index_strategies import supported_index_methods
from cementic.storage import extracted_document_path


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

    def test_invalid_toml_is_ignored(self, tmp_path, monkeypatch, capsys) -> None:
        bad = tmp_path / "bad.toml"
        bad.write_text("this is := not valid toml")
        monkeypatch.setenv("CEMENTIC_CONFIG", str(bad))
        assert load_config_file() == {}
        # Config falls back to defaults rather than crashing.
        assert Config().database.host == "localhost"
        # ...but the user is warned rather than the error being silently swallowed.
        stderr = capsys.readouterr().err
        assert str(bad) in stderr
        assert "malformed" in stderr.lower()

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

    def test_constructing_config_does_not_create_data_dir(self, temp_dir):
        """Read-only commands must not write to disk just from building a Config."""
        data_dir = temp_dir / "not-yet-created"
        with patch(
            "cementic.config.user_data_dir", return_value=str(data_dir)
        ) as mock_user_data_dir:
            config = Config()
        assert config.source_watcher.log_file == data_dir / "source_watcher.log"
        _, kwargs = mock_user_data_dir.call_args
        assert kwargs["ensure_exists"] is False

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
        """Index config rejects methods the strategy registry does not provide."""
        with pytest.raises(ValueError, match="index method must be one of"):
            Config(index={"method": "flat"})

    def test_index_method_accepts_every_registered_method(self):
        """Validation is driven by the registry, so the two cannot drift."""
        for method in supported_index_methods():
            assert Config(index={"method": method}).index.method == method

    def test_index_params_must_be_positive(self):
        """Index config rejects non-positive build/query parameters."""
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            Config(index={"hnsw_m": 0})


class TestConfigProblemsAreReported:
    """A broken config must be diagnosable.

    A typo inside a known section was already a hard failure, but a typo in the
    section *name* was invisible: the section was dropped and its settings fell
    back to plausible defaults, which is the expensive direction -- a dropped
    `[pipeline] chunk_size` changes the embedding profile, recoverable only by a
    full re-index.
    """

    def _write(self, tmp_path, text: str, monkeypatch) -> Path:
        path = tmp_path / "cementic.toml"
        path.write_text(text, encoding="utf-8")
        monkeypatch.setenv("CEMENTIC_CONFIG", str(path))
        return path

    def test_unknown_section_is_refused_with_a_suggestion(self, tmp_path, monkeypatch):
        self._write(tmp_path, '[llamacpp]\nmodel_path = "/x.gguf"\n', monkeypatch)

        with pytest.raises(ConfigError) as excinfo:
            get_config()
        assert "[llamacpp]" in str(excinfo.value)
        assert "llama_cpp" in str(excinfo.value)

    def test_every_problem_is_reported_at_once(self, tmp_path, monkeypatch):
        """pydantic stops at the first broken section, so a pre-scan is the only
        way the user fixes more than one typo per run."""
        self._write(tmp_path, "[llamacpp]\nx = 1\n\n[pipelines]\nchunk_size = 64\n", monkeypatch)

        with pytest.raises(ConfigError) as excinfo:
            get_config()
        assert "llamacpp" in str(excinfo.value)
        assert "pipelines" in str(excinfo.value)

    def test_unknown_key_in_a_known_section_is_reported(self, tmp_path, monkeypatch):
        self._write(tmp_path, '[llama_cpp]\nmodel_pth = "/x.gguf"\n', monkeypatch)

        with pytest.raises(ConfigError) as excinfo:
            get_config()
        assert "model_pth" in str(excinfo.value)

    def test_top_level_key_is_reported_rather_than_dropped(self, tmp_path, monkeypatch):
        self._write(tmp_path, "chunk_size = 999\n", monkeypatch)

        with pytest.raises(ConfigError):
            get_config()

    def test_valid_config_still_loads(self, tmp_path, monkeypatch):
        self._write(tmp_path, "[pipeline]\nchunk_size = 256\n", monkeypatch)

        assert get_config().pipeline.chunk_size == 256

    def test_nested_table_is_not_a_false_positive(self, tmp_path, monkeypatch):
        self._write(tmp_path, '[extraction.backends]\npdf = "pymupdf4llm"\n', monkeypatch)

        assert get_config().extraction.backends["pdf"] == "pymupdf4llm"


class TestExplicitConfigPathIsHonoured:
    def test_missing_file_is_an_error_not_a_fallback(self, tmp_path, monkeypatch):
        """A stale path in a service unit used to run against entirely different
        settings, with `config path` printing the fallback."""
        monkeypatch.setenv("CEMENTIC_CONFIG", str(tmp_path / "nope.toml"))

        with pytest.raises(ConfigError, match="does not exist"):
            get_config()

    def test_directory_is_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CEMENTIC_CONFIG", str(tmp_path))

        with pytest.raises(ConfigError, match="is a directory"):
            get_config()

    def test_unset_env_var_still_falls_back(self, monkeypatch):
        monkeypatch.delenv("CEMENTIC_CONFIG", raising=False)

        assert get_config() is not None


class TestConfigErrorsDoNotLeakValues:
    """pydantic embeds the offending value in both str(exc) and errors()["input"],
    so a mistyped credential key would otherwise be printed into the terminal and
    into the worker log files the CLI points users at."""

    def test_mistyped_credential_key_does_not_echo_its_value(self, tmp_path, monkeypatch):
        secret = "PLACEHOLDER-CANARY-VALUE"
        path = tmp_path / "cementic.toml"
        path.write_text(f'[database]\npasswrd = "{secret}"\n', encoding="utf-8")
        monkeypatch.setenv("CEMENTIC_CONFIG", str(path))

        with pytest.raises(ConfigError) as excinfo:
            get_config()
        assert secret not in str(excinfo.value)
        assert "passwrd" in str(excinfo.value)

    def test_formatter_never_echoes_a_bad_value(self, monkeypatch):
        monkeypatch.setenv("CEMENTIC_DB_PORT", "PLACEHOLDER-CANARY-VALUE")

        with pytest.raises(ValidationError) as excinfo:
            Config()
        rendered = format_config_error(excinfo.value, None)
        assert "PLACEHOLDER-CANARY-VALUE" not in rendered
        assert "port" in rendered


class TestDatabaseUrlIsValidatedAsConfig:
    def test_malformed_url_is_a_config_error(self, monkeypatch):
        """Left to the `url` property it surfaced as a raw ArgumentError from
        whichever command touched it first -- including the message `start`
        builds to explain the failure."""
        monkeypatch.setenv("CEMENTIC_DB_URL", "not a url")

        with pytest.raises(ValidationError, match="not a valid SQLAlchemy URL"):
            Config()

    def test_empty_url_still_means_unset(self, monkeypatch):
        monkeypatch.setenv("CEMENTIC_DB_URL", "")

        assert Config().database.url.get_backend_name() == "postgresql"

    def test_valid_url_is_accepted(self, monkeypatch):
        monkeypatch.setenv("CEMENTIC_DB_URL", "postgresql://u:p@h:5432/d")

        assert Config().database.url.database == "d"


class TestPathsAreExpandedAndAnchored:
    """TOML and systemd `Environment=` do no shell expansion.

    A hand-written `~/Documents` stayed a literal `~` directory created under the
    working directory, and a relative artifacts root was worse than wrong: it is
    used both to build the paths stored in the database and to resolve them
    again, so a reader in a different directory checked containment against its
    own root and "removed" a file that was never there.
    """

    def test_tilde_in_artifacts_path_expands(self, monkeypatch):
        monkeypatch.setenv("CEMENTIC_STORAGE_ARTIFACTS_PATH", "~/cementic-artifacts")

        path = Config().storage.artifacts_path

        assert "~" not in path.parts
        assert path.is_absolute()

    def test_tilde_in_model_path_expands(self, monkeypatch):
        monkeypatch.setenv("CEMENTIC_LLAMA_MODEL_PATH", "~/models/m.gguf")

        resolved = resolve_llama_model_path(Config().llama_cpp.model_path)

        assert "~" not in resolved.parts
        assert resolved.is_absolute()

    def test_relative_paths_are_anchored_to_the_working_directory(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("CEMENTIC_STORAGE_ARTIFACTS_PATH", "artifacts")

        path = Config().storage.artifacts_path

        assert path.is_absolute()
        assert path.parent == tmp_path

    def test_artifact_paths_derived_from_the_root_are_absolute(self, monkeypatch, tmp_path):
        """Anchoring at load time is what makes containment checks meaningful.

        A relative root stayed relative, so `safe_remove_artifact` resolved both
        the stored path and the root against whatever directory the caller
        happened to be in: containment passed vacuously and the unlink no-opped
        on a path that did not exist, deleting the rows while the bytes stayed.
        A relative setting is still relative to where cementic was started -- it
        is now merely unambiguous, and a genuine mismatch is refused rather than
        silently succeeding.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("CEMENTIC_STORAGE_ARTIFACTS_PATH", "artifacts")

        config = Config()
        artifact = extracted_document_path(config, "coll", 1, 2)

        assert config.storage.artifacts_path.is_absolute()
        assert artifact.is_absolute()
        assert artifact.is_relative_to(tmp_path)

    @pytest.mark.parametrize(
        "env_var",
        [
            "CEMENTIC_SOURCE_WATCHER_LOG_FILE",
            "CEMENTIC_PIPELINE_WORKER_LOG_FILE",
            "CEMENTIC_LLAMA_DAEMON_PID_FILE",
        ],
    )
    def test_every_path_setting_is_expanded(self, monkeypatch, env_var):
        monkeypatch.setenv(env_var, "~/somewhere/file")

        dumped = Config().model_dump(mode="json")

        assert "~" not in json.dumps(dumped)
