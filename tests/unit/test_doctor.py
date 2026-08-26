"""Tests for read-only doctor diagnostics."""

from unittest.mock import MagicMock, patch

from cementic.config import Config
from cementic.doctor import _daemon_state, collect_doctor_report
from cementic.embedding_runtime import AmbiguousDaemonPidsError, DaemonHealth


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

    @patch("cementic.doctor._daemon_state", return_value=(True, True, "reachable"))
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

    @patch("cementic.doctor._daemon_state", return_value=(True, True, "reachable"))
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

    @patch("cementic.doctor._daemon_state", return_value=(True, True, "reachable"))
    @patch("cementic.doctor.get_engine")
    def test_no_config_file_is_not_a_failure(
        self, mock_get_engine, mock_daemon, tmp_path
    ) -> None:
        """Running without a config file is normal, not an error."""
        mock_get_engine.side_effect = Exception("no db in this test")

        report = collect_doctor_report(Config())

        assert report["checks"]["config"]["status"] == "ok"


class TestExtensionFailureIsNotADatabaseFailure:
    @patch("cementic.doctor._daemon_state", return_value=(True, True, "reachable"))
    @patch("cementic.doctor._extension_check", side_effect=RuntimeError("permission denied"))
    @patch("cementic.doctor.get_engine")
    def test_extension_inspection_failure_keeps_database_ok(
        self, mock_get_engine, mock_ext, mock_daemon
    ) -> None:
        """Regression: a failure while *inspecting extensions* fell into the
        database handler, which reported an unreachable database -- with the
        init-postgres hint -- for a server that had just answered SELECT 1."""
        conn = MagicMock()
        mock_get_engine.return_value.connect.return_value.__enter__.return_value = conn

        report = collect_doctor_report(Config())

        assert report["checks"]["database"]["status"] == "ok"
        assert report["checks"]["database"]["reachable"] is True
        extensions = report["checks"]["extensions"]
        assert any(
            "could not inspect extensions" in payload["message"]
            for payload in extensions.values()
        )
        assert report["ok"] is False


class TestDaemonCheck:
    """A busy daemon must not be reported as broken.

    Regression: doctor used the /v1/models probe alone, which llama_cpp.server
    can block for the whole duration of an in-flight embedding batch. Running
    `cementic doctor` during indexing therefore said the daemon was unreachable,
    and with autostart disabled exited non-zero.
    """

    @patch("cementic.doctor.probe_daemon", return_value=DaemonHealth.BUSY)
    def test_busy_daemon_is_healthy(self, mock_probe) -> None:
        healthy, repairable, message = _daemon_state(Config())
        assert healthy is True
        assert "busy" in message

    @patch("cementic.doctor.probe_daemon", return_value=DaemonHealth.DOWN)
    def test_dead_daemon_is_not_healthy(self, mock_probe) -> None:
        healthy, repairable, _message = _daemon_state(Config())
        assert healthy is False
        # A stopped daemon is exactly what autostart repairs.
        assert repairable is True

    @patch("cementic.doctor.probe_daemon", return_value=DaemonHealth.WEDGED)
    @patch("cementic.doctor.get_engine")
    def test_a_wedged_daemon_fails_even_with_autostart(
        self, mock_get_engine, mock_probe, tmp_path
    ) -> None:
        """Autostart cannot repair a wedge: the daemon still answers /v1/models
        with the expected fingerprint, so the resolver returns it unrepaired.
        Excusing every unhealthy state behind autostart let a wedged daemon
        pass `doctor` at ok -- the state behind the 21-hour incident."""
        mock_get_engine.side_effect = Exception("no db in this test")
        config = _config_with_model_path(tmp_path, exists=True)
        assert config.llama_cpp.daemon_autostart is True

        report = collect_doctor_report(config)

        assert report["checks"]["daemon"]["status"] == "fail"
        assert "not answering embeddings" in report["checks"]["daemon"]["message"]
        assert report["ok"] is False

    @patch("cementic.doctor.probe_daemon", return_value=DaemonHealth.WRONG_MODEL)
    @patch("cementic.doctor.get_engine")
    def test_a_wrong_model_daemon_is_a_warning_with_autostart(
        self, mock_get_engine, mock_probe, tmp_path
    ) -> None:
        """Unlike a wedge, a mismatch is repairable: autostart stops the stale
        daemon and spawns the configured model on the next embedding use."""
        mock_get_engine.side_effect = Exception("no db in this test")
        config = _config_with_model_path(tmp_path, exists=True)

        report = collect_doctor_report(config)

        assert report["checks"]["daemon"]["status"] == "warning"

    @patch(
        "cementic.doctor.probe_daemon",
        side_effect=AmbiguousDaemonPidsError([111, 222]),
    )
    @patch("cementic.doctor.get_engine")
    def test_ambiguous_daemon_pids_fail_the_check_instead_of_crashing(
        self, mock_get_engine, mock_probe, tmp_path
    ) -> None:
        """The diagnostic tool must report the pathological state, not die of
        it: probe recovery refuses to guess between candidate processes, and
        that refusal used to escape `doctor` as a raw traceback."""
        mock_get_engine.side_effect = Exception("no db in this test")
        config = _config_with_model_path(tmp_path, exists=True)

        report = collect_doctor_report(config)

        assert report["checks"]["daemon"]["status"] == "fail"
        assert "111, 222" in report["checks"]["daemon"]["message"]

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

        # The path has to be one auto-download would actually accept: downloads
        # are confined to the data directory, so make tmp_path be it.
        with patch("cementic.config.user_data_dir", return_value=str(tmp_path)):
            report = collect_doctor_report(config)

        assert report["checks"]["model"]["status"] == "warning"
        assert report["checks"]["model"]["exists"] is False

    @patch("cementic.doctor.probe_daemon", return_value=DaemonHealth.HEALTHY)
    @patch("cementic.doctor.get_engine")
    def test_undownloadable_path_is_not_reported_as_a_pending_download(
        self, mock_get_engine, mock_daemon, tmp_path
    ) -> None:
        """Auto-download is confined to the data directory, so promising one for
        a path outside it passed a config that `cementic start` then died on."""
        mock_get_engine.side_effect = Exception("no db in this test")
        config = _config_with_model_path(tmp_path, exists=False)
        config.bootstrap.auto_download_llama_model = True

        with patch(
            "cementic.config.user_data_dir", return_value=str(tmp_path / "elsewhere")
        ):
            report = collect_doctor_report(config)

        assert report["checks"]["model"]["status"] == "fail"
        assert "outside" in report["checks"]["model"]["message"]
        assert report["ok"] is False

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

        with (
            patch("cementic.doctor.get_engine", return_value=mock_engine),
            patch("cementic.config.user_data_dir", return_value=str(tmp_path)),
        ):
            report = collect_doctor_report(config)

        assert report["checks"]["model"]["status"] == "warning"
        assert report["ok"] is True
