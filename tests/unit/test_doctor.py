"""Tests for read-only doctor diagnostics."""

from unittest.mock import MagicMock, patch

from cementic.config import Config
from cementic.doctor import collect_doctor_report


def _config_with_model_path(tmp_path, exists: bool):
    config = Config()
    model_path = tmp_path / "model.gguf"
    if exists:
        model_path.write_bytes(b"fake")
    config.llama_cpp.model_path = str(model_path)
    return config


class TestModelCheck:
    """Missing model is a warning (not a hard failure) when auto-download is on."""

    @patch("cementic.doctor._daemon_reachable", return_value=True)
    @patch("cementic.doctor.get_engine")
    def test_missing_model_with_autodownload_is_warning_not_fail(
        self, mock_get_engine, mock_daemon, tmp_path
    ) -> None:
        mock_get_engine.side_effect = Exception("no db in this test")
        config = _config_with_model_path(tmp_path, exists=False)
        config.bootstrap.auto_download_llama_model = True

        report = collect_doctor_report(config)

        assert report["checks"]["model"]["status"] == "warning"
        assert report["checks"]["model"]["exists"] is False

    @patch("cementic.doctor._daemon_reachable", return_value=True)
    @patch("cementic.doctor.get_engine")
    def test_missing_model_without_autodownload_is_fail(
        self, mock_get_engine, mock_daemon, tmp_path
    ) -> None:
        mock_get_engine.side_effect = Exception("no db in this test")
        config = _config_with_model_path(tmp_path, exists=False)
        config.bootstrap.auto_download_llama_model = False

        report = collect_doctor_report(config)

        assert report["checks"]["model"]["status"] == "fail"

    @patch("cementic.doctor._daemon_reachable", return_value=True)
    @patch("cementic.doctor.get_engine")
    def test_present_model_is_ok_regardless_of_autodownload(
        self, mock_get_engine, mock_daemon, tmp_path
    ) -> None:
        mock_get_engine.side_effect = Exception("no db in this test")
        config = _config_with_model_path(tmp_path, exists=True)
        config.bootstrap.auto_download_llama_model = False

        report = collect_doctor_report(config)

        assert report["checks"]["model"]["status"] == "ok"

    @patch("cementic.doctor._daemon_reachable", return_value=True)
    def test_missing_model_with_autodownload_does_not_force_overall_not_ok(
        self, mock_daemon, tmp_path
    ) -> None:
        """A warning-level model check must not by itself flip the overall verdict."""
        config = _config_with_model_path(tmp_path, exists=False)
        config.bootstrap.auto_download_llama_model = True

        mock_conn = MagicMock()
        mock_conn.execute.return_value.scalar.return_value = True
        mock_engine = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn

        with patch("cementic.doctor.get_engine", return_value=mock_engine):
            report = collect_doctor_report(config)

        assert report["checks"]["model"]["status"] == "warning"
        assert report["ok"] is True
