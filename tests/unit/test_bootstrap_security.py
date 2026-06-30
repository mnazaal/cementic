"""Security tests for bootstrap: model-download path confinement + integrity."""

import logging
import re
from unittest.mock import Mock, patch

import pytest

from cementic.bootstrap import Bootstrapper
from cementic.config import Config


class TestModelDownloadConfinement:
    """Downloads must never escape the cementic data directory."""

    def test_absolute_path_outside_data_dir_is_rejected(self, temp_dir) -> None:
        config = Config()
        config.llama_cpp.model_path = "/etc/evil.gguf"
        config.bootstrap.auto_download_llama_model = True
        bootstrapper = Bootstrapper(config)
        with (
            patch("cementic.config.user_data_dir", return_value=str(temp_dir)),
            patch("cementic.bootstrap.requests.get") as mock_get,
            pytest.raises(RuntimeError, match="outside the cementic data directory"),
        ):
            bootstrapper._ensure_llama_model()
        mock_get.assert_not_called()

    def test_relative_traversal_outside_data_dir_is_rejected(self, temp_dir) -> None:
        config = Config()
        config.llama_cpp.model_path = "../../etc/evil.gguf"
        config.bootstrap.auto_download_llama_model = True
        bootstrapper = Bootstrapper(config)
        data_dir = temp_dir / "data"
        data_dir.mkdir()
        with (
            patch("cementic.config.user_data_dir", return_value=str(data_dir)),
            patch("cementic.bootstrap.requests.get") as mock_get,
            pytest.raises(RuntimeError, match="outside the cementic data directory"),
        ):
            bootstrapper._ensure_llama_model()
        mock_get.assert_not_called()


class TestModelDownloadIntegrity:
    """The default download is verified against a pinned SHA-256."""

    def test_default_config_pins_model_checksum(self) -> None:
        digest = Config().bootstrap.llama_model_sha256
        assert digest is not None
        assert re.fullmatch(r"[0-9a-f]{64}", digest)

    @patch("cementic.bootstrap.requests.get")
    def test_download_with_default_checksum_rejects_wrong_content(
        self, mock_get, temp_dir
    ) -> None:
        config = Config()  # keeps the pinned default checksum
        config.llama_cpp.model_path = "download.gguf"  # relative -> under data dir
        config.bootstrap.auto_download_llama_model = True
        bootstrapper = Bootstrapper(config)

        response = Mock()
        response.iter_content.return_value = [b"not-the-real-model"]
        mock_get.return_value.__enter__.return_value = response

        with patch("cementic.config.user_data_dir", return_value=str(temp_dir)):
            with pytest.raises(RuntimeError, match="Checksum mismatch"):
                bootstrapper._ensure_llama_model()

        mock_get.assert_called_once()
        # The unverified partial download must not be left behind.
        assert not (temp_dir / "download.gguf").exists()
        assert not (temp_dir / ".download.gguf.tmp").exists()

    @patch("cementic.bootstrap.requests.get")
    def test_unverified_download_warns(self, mock_get, temp_dir, caplog) -> None:
        config = Config()
        config.llama_cpp.model_path = "download.gguf"
        config.bootstrap.auto_download_llama_model = True
        config.bootstrap.llama_model_sha256 = None  # opt out of verification
        bootstrapper = Bootstrapper(config)

        response = Mock()
        response.iter_content.return_value = [b"model-bytes"]
        mock_get.return_value.__enter__.return_value = response

        with patch("cementic.config.user_data_dir", return_value=str(temp_dir)):
            with caplog.at_level(logging.WARNING, logger="cementic.bootstrap"):
                bootstrapper._ensure_llama_model()

        assert any("without integrity verification" in rec.message for rec in caplog.records)
