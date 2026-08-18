"""Tests for the container-free bootstrap: DB checks and model download."""

import hashlib
from contextlib import contextmanager
from unittest.mock import Mock, patch

import pytest
from pydantic import SecretStr

from cementic import bootstrap as bootstrap_module
from cementic.bootstrap import Bootstrapper
from cementic.config import Config


class TestEnsureRuntime:
    """ensure_for_* orchestration."""

    def test_ensure_for_index_checks_postgres_then_model(self) -> None:
        config = Config()
        config.pipeline.embedding_provider = "llama-cpp"
        bootstrapper = Bootstrapper(config)
        with (
            patch.object(bootstrapper, "_ensure_postgres_ready") as ensure_pg,
            patch.object(bootstrapper, "_ensure_llama_model") as ensure_model,
        ):
            bootstrapper.ensure_for_index()
        ensure_pg.assert_called_once_with()
        ensure_model.assert_called_once_with()

    def test_ensure_for_convert_only_checks_postgres(self) -> None:
        config = Config()
        bootstrapper = Bootstrapper(config)
        with (
            patch.object(bootstrapper, "_ensure_postgres_ready") as ensure_pg,
            patch.object(bootstrapper, "ensure_embedding_runtime") as ensure_embed,
        ):
            bootstrapper.ensure_for_convert()
        ensure_pg.assert_called_once_with()
        ensure_embed.assert_not_called()

    def test_unsupported_embedding_provider_raises(self) -> None:
        config = Config()
        config.pipeline.embedding_provider = "unknown"
        bootstrapper = Bootstrapper(config)
        with (
            patch.object(bootstrapper, "_ensure_postgres_ready"),
            pytest.raises(RuntimeError, match="Unsupported embedding provider"),
        ):
            bootstrapper.ensure_for_index()


class TestPostgresReady:
    """Postgres is external; cementic only probes it."""

    def test_database_error_none_when_reachable(self) -> None:
        config = Config()
        bootstrapper = Bootstrapper(config)
        with patch("cementic.bootstrap.get_engine") as mock_engine:
            mock_engine.return_value.connect.return_value.__enter__.return_value = Mock()
            assert bootstrapper._database_error() is None

    def test_database_error_returns_the_cause(self) -> None:
        config = Config()
        bootstrapper = Bootstrapper(config)
        with patch("cementic.bootstrap.get_engine", side_effect=ConnectionError("refused")):
            error = bootstrapper._database_error()
        assert isinstance(error, ConnectionError)

    def test_ensure_postgres_ready_ok_when_reachable(self) -> None:
        config = Config()
        bootstrapper = Bootstrapper(config)
        with patch.object(bootstrapper, "_database_error", return_value=None):
            bootstrapper._ensure_postgres_ready()  # must not raise

    def test_ensure_postgres_ready_raises_with_compose_hint(self) -> None:
        config = Config()
        bootstrapper = Bootstrapper(config)
        with (
            patch.object(bootstrapper, "_database_error", return_value=ConnectionError("x")),
            pytest.raises(RuntimeError, match="cementic init postgres"),
        ):
            bootstrapper._ensure_postgres_ready()

    def test_failure_message_reports_the_underlying_cause(self) -> None:
        """A wrong password must not read as "cannot reach Postgres".

        Regression: every failure collapsed to the same unreachable message with
        advice to run `init postgres`, which is wrong when the server is up and
        only the credentials are bad.
        """
        config = Config()
        bootstrapper = Bootstrapper(config)
        cause = RuntimeError('password authentication failed for user "cementic"')
        with (
            patch.object(bootstrapper, "_database_error", return_value=cause),
            pytest.raises(RuntimeError, match="password authentication failed"),
        ):
            bootstrapper._ensure_postgres_ready()

    def test_failure_message_does_not_leak_the_password(self) -> None:
        config = Config()
        config.database.password = SecretStr("hunter2-secret")
        bootstrapper = Bootstrapper(config)
        with patch.object(bootstrapper, "_database_error", return_value=ConnectionError("x")):
            with pytest.raises(RuntimeError) as excinfo:
                bootstrapper._ensure_postgres_ready()
        assert "hunter2-secret" not in str(excinfo.value)


class TestEnsureLlamaModel:
    """Model file is fetched (confined to the data dir) when missing."""

    def test_existing_model_is_not_downloaded(self, temp_dir) -> None:
        config = Config()
        config.bootstrap.llama_model_sha256 = None  # mechanics test; skip verification
        model_path = temp_dir / "test.gguf"
        model_path.write_text("fake")
        config.llama_cpp.model_path = str(model_path)
        bootstrapper = Bootstrapper(config)
        with patch("cementic.bootstrap.requests.get") as mock_get:
            bootstrapper._ensure_llama_model()
            mock_get.assert_not_called()

    def test_missing_model_with_download_disabled_raises(self, temp_dir) -> None:
        config = Config()
        config.llama_cpp.model_path = str(temp_dir / "missing.gguf")
        config.bootstrap.auto_download_llama_model = False
        bootstrapper = Bootstrapper(config)
        with pytest.raises(RuntimeError, match="model not found"):
            bootstrapper._ensure_llama_model()

    @patch("cementic.bootstrap.requests.get")
    def test_downloads_into_data_dir(self, mock_get, temp_dir) -> None:
        config = Config()
        config.llama_cpp.model_path = "download.gguf"  # relative -> under data dir
        config.bootstrap.auto_download_llama_model = True
        config.bootstrap.llama_model_sha256 = None  # mechanics test; skip verification
        bootstrapper = Bootstrapper(config)

        response = Mock()
        response.iter_content.return_value = [b"model-bytes"]
        mock_get.return_value.__enter__.return_value = response

        with patch("cementic.config.user_data_dir", return_value=str(temp_dir)):
            bootstrapper._ensure_llama_model()

        downloaded = temp_dir / "download.gguf"
        assert downloaded.exists()
        assert downloaded.read_bytes() == b"model-bytes"

    @patch("cementic.bootstrap.requests.get")
    def test_download_is_atomic(self, mock_get, temp_dir) -> None:
        config = Config()
        config.llama_cpp.model_path = "download.gguf"
        config.bootstrap.auto_download_llama_model = True
        config.bootstrap.llama_model_sha256 = None  # mechanics test; skip verification
        bootstrapper = Bootstrapper(config)

        response = Mock()
        response.iter_content.return_value = [b"model-bytes"]
        mock_get.return_value.__enter__.return_value = response

        with patch("cementic.config.user_data_dir", return_value=str(temp_dir)):
            bootstrapper._ensure_llama_model()

        assert (temp_dir / "download.gguf").read_bytes() == b"model-bytes"
        assert not list(temp_dir.glob(".download.gguf.*tmp"))

    @patch("cementic.bootstrap.requests.get")
    def test_network_failure_is_one_line_not_a_traceback(self, mock_get, temp_dir) -> None:
        """Regression (third-time carry): a connection reset or DNS failure
        propagated as a raw requests traceback into the background log; every
        caller catches RuntimeError and prints "Bootstrap failed: ..."."""
        import requests as requests_module

        config = Config()
        config.llama_cpp.model_path = "download.gguf"
        config.bootstrap.auto_download_llama_model = True
        config.bootstrap.llama_model_sha256 = None
        bootstrapper = Bootstrapper(config)

        mock_get.side_effect = requests_module.exceptions.ConnectionError("reset by peer")

        with (
            patch("cementic.config.user_data_dir", return_value=str(temp_dir)),
            pytest.raises(RuntimeError, match="Model download failed"),
        ):
            bootstrapper._ensure_llama_model()

        assert not list(temp_dir.glob(".download.gguf.*tmp"))

    @patch("cementic.bootstrap.requests.get")
    def test_waiter_reuses_the_winners_file_instead_of_redownloading(
        self, mock_get, temp_dir
    ) -> None:
        """The download runs under a file lock; whoever waited re-checks for
        the winner's file. Simulated by a first bootstrapper completing before
        a second runs: the second must not touch the network at all."""
        config = Config()
        config.llama_cpp.model_path = "download.gguf"
        config.bootstrap.auto_download_llama_model = True
        config.bootstrap.llama_model_sha256 = None
        response = Mock()
        response.iter_content.return_value = [b"model-bytes"]
        mock_get.return_value.__enter__.return_value = response

        with patch("cementic.config.user_data_dir", return_value=str(temp_dir)):
            Bootstrapper(config)._ensure_llama_model()
            assert mock_get.call_count == 1
            Bootstrapper(config)._ensure_llama_model()

        assert mock_get.call_count == 1

    @patch("cementic.bootstrap.requests.get")
    def test_the_re_check_inside_the_lock_is_the_one_that_stops_the_waiter(
        self, mock_get, temp_dir
    ) -> None:
        """The second bootstrapper above returns at the *pre-lock* exists check,
        so it never exercises the in-lock re-check it is named for: deleting that
        re-check would leave a real waiter re-downloading over the winner's file
        and the test still green. Here the file appears only while the lock is
        held, which is exactly the waiter's situation."""
        config = Config()
        config.llama_cpp.model_path = "download.gguf"
        config.bootstrap.auto_download_llama_model = True
        config.bootstrap.llama_model_sha256 = None
        model_path = temp_dir / "download.gguf"

        real_file_lock = bootstrap_module.file_lock

        @contextmanager
        def _winner_finishes_while_we_wait(path, timeout):
            with real_file_lock(path, timeout=timeout):
                model_path.write_bytes(b"the winner's file")
                yield

        with (
            patch("cementic.config.user_data_dir", return_value=str(temp_dir)),
            patch.object(bootstrap_module, "file_lock", _winner_finishes_while_we_wait),
        ):
            Bootstrapper(config)._ensure_llama_model()

        assert mock_get.call_count == 0
        assert model_path.read_bytes() == b"the winner's file"

    @patch("cementic.bootstrap.requests.get")
    def test_download_rejects_checksum_mismatch_and_removes_temp(self, mock_get, temp_dir) -> None:
        config = Config()
        config.llama_cpp.model_path = "download.gguf"
        config.bootstrap.auto_download_llama_model = True
        config.bootstrap.llama_model_sha256 = "0" * 64
        bootstrapper = Bootstrapper(config)

        response = Mock()
        response.iter_content.return_value = [b"model-bytes"]
        mock_get.return_value.__enter__.return_value = response

        with (
            patch("cementic.config.user_data_dir", return_value=str(temp_dir)),
            pytest.raises(RuntimeError, match="Checksum mismatch"),
        ):
            bootstrapper._ensure_llama_model()

        assert not (temp_dir / "download.gguf").exists()
        assert not (temp_dir / ".download.gguf.tmp").exists()

    def test_default_model_path_checksum_mismatch_raises(self, temp_dir, monkeypatch) -> None:
        """The pin protects the file cementic downloads for itself."""
        config = Config()  # model_path left at its default
        model_path = temp_dir / "default.gguf"
        model_path.write_bytes(b"fake")
        monkeypatch.setattr(
            "cementic.bootstrap.resolve_llama_model_path", lambda _path: model_path
        )
        config.bootstrap.llama_model_sha256 = "0" * 64
        bootstrapper = Bootstrapper(config)

        with pytest.raises(RuntimeError, match="Checksum mismatch"):
            bootstrapper._ensure_llama_model()

    def test_mismatch_error_names_the_override(self, temp_dir, monkeypatch) -> None:
        """The message must lead to the fix; the env var was never mentioned."""
        config = Config()
        model_path = temp_dir / "default.gguf"
        model_path.write_bytes(b"fake")
        monkeypatch.setattr(
            "cementic.bootstrap.resolve_llama_model_path", lambda _path: model_path
        )
        config.bootstrap.llama_model_sha256 = "0" * 64

        with pytest.raises(RuntimeError, match="CEMENTIC_BOOTSTRAP_LLAMA_MODEL_SHA256"):
            Bootstrapper(config)._ensure_llama_model()

    def test_user_supplied_model_is_not_checked_against_the_pin(self, temp_dir) -> None:
        """Pointing at your own model must not fail against the bundled digest.

        Regression: llama_model_sha256 defaults to the bundled Nomic digest, so
        `CEMENTIC_LLAMA_MODEL_PATH=/my/model.gguf` -- the documented way to use a
        different model -- hard-failed with a mismatch.
        """
        config = Config()
        model_path = temp_dir / "my-own-model.gguf"
        model_path.write_bytes(b"fake")
        config.llama_cpp.model_path = str(model_path)
        config.bootstrap.llama_model_sha256 = "0" * 64  # digest of the bundled model
        bootstrapper = Bootstrapper(config)

        with patch("cementic.bootstrap.requests.get") as mock_get:
            bootstrapper._ensure_llama_model()  # must not raise

        mock_get.assert_not_called()

    def test_existing_model_checksum_match_skips_download(self, temp_dir) -> None:
        config = Config()
        model_path = temp_dir / "test.gguf"
        model_path.write_bytes(b"fake")
        config.llama_cpp.model_path = str(model_path)
        config.bootstrap.llama_model_sha256 = hashlib.sha256(b"fake").hexdigest()
        bootstrapper = Bootstrapper(config)

        with patch("cementic.bootstrap.requests.get") as mock_get:
            bootstrapper._ensure_llama_model()

        mock_get.assert_not_called()
