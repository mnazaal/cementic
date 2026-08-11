"""Tests for read-only doctor diagnostics."""

from unittest.mock import MagicMock, patch

from cementic.config import Config
from cementic.doctor import _daemon_state, collect_doctor_report
from cementic.embedding_runtime import DaemonHealth


def _config_with_model_path(tmp_path, exists: bool):
    config = Config()
    model_path = tmp_path / "model.gguf"
    if exists:
        model_path.write_bytes(b"fake")
    config.llama_cpp.model_path = str(model_path)
    return config


class TestConfigCheck:
    """The config check must be able to fail.

    Regression: its status was a literal "ok", so a malformed or unreadable
    config file -- silently discarded, leaving cementic on defaults with the
    wrong database and model -- was reported as fine, pointing at the very file
    that was not being used.
    """

    @patch("cementic.doctor._daemon_state", return_value=(True, "reachable"))
    @patch("cementic.doctor.get_engine")
    def test_malformed_config_file_fails_the_check(
        self, mock_get_engine, mock_daemon, tmp_path, monkeypatch
    ) -> None:
        mock_get_engine.side_effect = Exception("no db in this test")
        bad = tmp_path / "cementic.toml"
        bad.write_text("this is [not valid TOML", encoding="utf-8")
        monkeypatch.setenv("CEMENTIC_CONFIG", str(bad))

        report = collect_doctor_report(Config())

        assert report["checks"]["config"]["status"] == "fail"
        assert "malformed TOML" in report["checks"]["config"]["message"]
        assert report["ok"] is False

    @patch("cementic.doctor._daemon_state", return_value=(True, "reachable"))
    @patch("cementic.doctor.get_engine")
    def test_valid_config_file_passes(
        self, mock_get_engine, mock_daemon, tmp_path, monkeypatch
    ) -> None:
        mock_get_engine.side_effect = Exception("no db in this test")
        good = tmp_path / "cementic.toml"
        good.write_text("[pipeline]\nchunk_size = 256\n", encoding="utf-8")
        monkeypatch.setenv("CEMENTIC_CONFIG", str(good))

        report = collect_doctor_report(Config())

        assert report["checks"]["config"]["status"] == "ok"

    @patch("cementic.doctor._daemon_state", return_value=(True, "reachable"))
    @patch("cementic.doctor.get_engine")
    def test_no_config_file_is_not_a_failure(
        self, mock_get_engine, mock_daemon, tmp_path
    ) -> None:
        """Running without a config file is normal, not an error."""
        mock_get_engine.side_effect = Exception("no db in this test")

        report = collect_doctor_report(Config())

        assert report["checks"]["config"]["status"] == "ok"


class TestDaemonCheck:
    """A busy daemon must not be reported as broken.

    Regression: doctor used the /v1/models probe alone, which llama_cpp.server
    can block for the whole duration of an in-flight embedding batch. Running
    `status --doctor` during indexing therefore said the daemon was unreachable,
    and with autostart disabled exited non-zero.
    """

    @patch("cementic.doctor.probe_daemon", return_value=DaemonHealth.BUSY)
    def test_busy_daemon_is_healthy(self, mock_probe) -> None:
        healthy, message = _daemon_state(Config())
        assert healthy is True
        assert "busy" in message

    @patch("cementic.doctor.probe_daemon", return_value=DaemonHealth.DOWN)
    def test_dead_daemon_is_not_healthy(self, mock_probe) -> None:
        healthy, _message = _daemon_state(Config())
        assert healthy is False

    @patch("cementic.doctor.probe_daemon", return_value=DaemonHealth.DOWN)
    @patch("cementic.doctor.get_engine")
    def test_dead_daemon_fails_when_autostart_disabled(
        self, mock_get_engine, mock_probe, tmp_path
    ) -> None:
        mock_get_engine.side_effect = Exception("no db in this test")
        config = _config_with_model_path(tmp_path, exists=True)
        config.llama_cpp.daemon_autostart = False

        report = collect_doctor_report(config)

        assert report["checks"]["daemon"]["status"] == "fail"


class TestModelCheck:
    """Missing model is a warning (not a hard failure) when auto-download is on."""

    @patch("cementic.doctor.probe_daemon", return_value=DaemonHealth.HEALTHY)
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

    @patch("cementic.doctor.probe_daemon", return_value=DaemonHealth.HEALTHY)
    @patch("cementic.doctor.get_engine")
    def test_missing_model_without_autodownload_is_fail(
        self, mock_get_engine, mock_daemon, tmp_path
    ) -> None:
        mock_get_engine.side_effect = Exception("no db in this test")
        config = _config_with_model_path(tmp_path, exists=False)
        config.bootstrap.auto_download_llama_model = False

        report = collect_doctor_report(config)

        assert report["checks"]["model"]["status"] == "fail"

    @patch("cementic.doctor.probe_daemon", return_value=DaemonHealth.HEALTHY)
    @patch("cementic.doctor.get_engine")
    def test_present_model_is_ok_regardless_of_autodownload(
        self, mock_get_engine, mock_daemon, tmp_path
    ) -> None:
        mock_get_engine.side_effect = Exception("no db in this test")
        config = _config_with_model_path(tmp_path, exists=True)
        config.bootstrap.auto_download_llama_model = False

        report = collect_doctor_report(config)

        assert report["checks"]["model"]["status"] == "ok"

    @patch("cementic.doctor.probe_daemon", return_value=DaemonHealth.HEALTHY)
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
