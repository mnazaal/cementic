"""Tests for read-only doctor diagnostics."""

import json
from unittest.mock import MagicMock, patch

import pytest

from cementic.config import Config, ExtractionConfig
from cementic.doctor import (
    _daemon_state,
    _embedding_server_check,
    _extraction_commands_check,
    _extractor_drift_check,
    _ocr_check,
    collect_doctor_report,
)
from cementic.embedding_runtime import AmbiguousDaemonPidsError, DaemonHealth


def _config_with_model_path(tmp_path, exists: bool):
    config = Config()
    model_path = tmp_path / "model.gguf"
    if exists:
        model_path.write_bytes(b"fake")
    config.llama_cpp.model_path = str(model_path)
    return config


@pytest.fixture(autouse=True)
def _stub_embedding_server_check(request):
    """Keep the rest of the suite off the host's PATH.

    `_embedding_server_check` resolves and runs a real binary, so without this
    every test asserting the overall verdict would pass or fail according to
    whether the developer happens to have llama-server installed and working.
    The class that tests the check itself opts out.
    """
    if request.cls is TestEmbeddingServerCheck:
        yield
        return
    with patch(
        "cementic.doctor._embedding_server_check",
        return_value={"status": "ok", "command": "llama-server", "message": "stubbed"},
    ):
        yield


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


class TestOcrCheck:
    """OCR needs two settings and a package to agree; each fails quietly alone."""

    def test_ocr_off_needs_no_backend(self) -> None:
        report = _ocr_check(Config())
        assert report["status"] == "ok"
        assert report["enabled"] is False

    def test_ocr_on_under_a_backend_that_ignores_it_is_a_warning(self) -> None:
        """The silent no-op: pymupdf-raw reads the text layer and never OCRs,
        so `use_ocr = true` changes nothing and scans still extract empty."""
        config = Config()
        config.extraction.use_ocr = True
        config.extraction.backends = {"pdf": "pymupdf-raw"}

        report = _ocr_check(config)
        assert report["status"] == "warning"
        assert "pymupdf-raw" in report["message"]

    def test_missing_engine_under_an_ocr_backend_fails(self) -> None:
        """Not a warning: with OCR reaching extraction, every PDF raises, and
        calling that ready would pass an unusable configuration."""
        config = Config()
        config.extraction.use_ocr = True
        config.extraction.backends = {"pdf": "pymupdf4llm"}

        with patch("cementic.extract.ocr_backend_available", return_value=False):
            report = _ocr_check(config)
        assert report["status"] == "fail"
        assert "rapidocr>=3.6.0" in report["message"]

    def test_installed_engine_under_an_ocr_backend_is_ok(self) -> None:
        config = Config()
        config.extraction.use_ocr = True
        config.extraction.backends = {"pdf": "pymupdf4llm"}

        with patch("cementic.extract.ocr_backend_available", return_value=True):
            report = _ocr_check(config)
        assert report["status"] == "ok"
        assert report["enabled"] is True

    def test_a_failing_ocr_check_fails_the_whole_report(self) -> None:
        """The check has to reach `ok`, or doctor reports ready on a config
        under which no PDF can be extracted."""
        config = Config()
        config.extraction.use_ocr = True
        config.extraction.backends = {"pdf": "pymupdf4llm"}

        with patch("cementic.extract.ocr_backend_available", return_value=False):
            report = collect_doctor_report(config)
        assert report["checks"]["ocr"]["status"] == "fail"
        assert report["ok"] is False


class TestEmbeddingServerCheck:
    """cementic ships no embedding server, so the binary is a prerequisite."""

    @staticmethod
    def _config(command: list[str]) -> Config:
        config = Config()
        config.llama_cpp.daemon_command = command
        return config

    def test_a_name_not_on_path_fails(self) -> None:
        report = _embedding_server_check(self._config(["llama-server-not-installed"]))
        assert report["status"] == "fail"
        assert "not on PATH" in report["message"]

    def test_a_resolvable_binary_that_cannot_run_fails(self, tmp_path, monkeypatch) -> None:
        """The failure `which` alone cannot see.

        A build whose shared library moved resolves fine and dies on exec, so a
        PATH-only check calls it ready and autostart discovers it later, in a
        worker's log file.
        """
        broken = tmp_path / "llama-server"
        broken.write_text("#!/bin/sh\necho 'libllama.so: cannot open' >&2\nexit 1\n")
        broken.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path))

        report = _embedding_server_check(self._config(["llama-server"]))

        assert report["status"] == "fail"
        assert "does not run" in report["message"]
        assert "libllama.so" in report["message"]

    def test_a_working_binary_is_ok(self, tmp_path, monkeypatch) -> None:
        working = tmp_path / "llama-server"
        working.write_text("#!/bin/sh\necho 'version: 1'\n")
        working.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path))

        report = _embedding_server_check(self._config(["llama-server"]))

        assert report["status"] == "ok"
        assert report["command"] == str(working)

    def test_an_env_fronted_command_is_left_alone(self) -> None:
        """env(1) carries setup this check cannot reproduce -- an
        LD_LIBRARY_PATH supplied precisely because the binary needs it -- so
        running the bare binary would report a working setup as broken."""
        report = _embedding_server_check(
            self._config(["env", "LD_LIBRARY_PATH=/opt/llama", "/opt/llama/llama-server"])
        )
        assert report["status"] == "ok"
        assert "not resolved further" in report["message"]

    def test_an_explicit_path_is_left_alone(self) -> None:
        report = _embedding_server_check(self._config(["/opt/llama/llama-server"]))
        assert report["status"] == "ok"
        assert "not resolved further" in report["message"]

    def test_a_failing_server_check_fails_the_whole_report(self) -> None:
        """doctor must not report ready when nothing can serve embeddings."""
        config = self._config(["llama-server-not-installed"])
        mock_conn = MagicMock()
        mock_conn.execute.return_value.scalar.return_value = True
        mock_engine = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn

        with patch("cementic.doctor.get_engine", return_value=mock_engine):
            report = collect_doctor_report(config)

        assert report["checks"]["embedding_server"]["status"] == "fail"
        assert report["ok"] is False


class TestExtractionCommandsCheck:
    """Surfaces the one thing no check can validate: a wrong version flag."""

    @staticmethod
    def _config(version_argv: list[str]) -> Config:
        return Config(
            extraction=ExtractionConfig(
                backends={"pdf": "command"},
                commands={"pdf": ["cat", "{path}"]},
                command_versions={"pdf": version_argv},
            )
        )

    def test_quiet_when_no_command_backend_is_used(self) -> None:
        report = _extraction_commands_check(Config())
        assert report["status"] == "ok"
        assert report["commands"] == {}

    def test_reports_the_version_each_tool_gives(self, tmp_path) -> None:
        script = tmp_path / "v.sh"
        script.write_text("#!/bin/sh\necho 'tool 4.5.6'\n")
        script.chmod(0o755)

        report = _extraction_commands_check(self._config([str(script)]))

        assert report["status"] == "ok"
        assert report["commands"] == {"pdf": "tool 4.5.6"}
        assert "tool 4.5.6" in report["message"]

    def test_a_tool_that_cannot_answer_fails_the_report(self) -> None:
        config = self._config(["definitely-not-a-tool"])
        mock_conn = MagicMock()
        mock_conn.execute.return_value.scalar.return_value = True
        mock_engine = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn

        with patch("cementic.doctor.get_engine", return_value=mock_engine):
            report = collect_doctor_report(config)

        assert report["checks"]["extraction_commands"]["status"] == "fail"
        assert report["ok"] is False

    def test_a_wrong_version_flag_is_shown_rather_than_judged(self, tmp_path) -> None:
        """`pdftotext --version` exits 0 printing an error, so the recorded
        "version" is a constant that never moves on upgrade. Nothing can detect
        that automatically; printing it is what lets a human catch it."""
        script = tmp_path / "v.sh"
        script.write_text("#!/bin/sh\necho \"I/O Error: Couldn't open file '--version'\"\n")
        script.chmod(0o755)

        report = _extraction_commands_check(self._config([str(script)]))

        assert report["status"] == "ok"
        assert "I/O Error" in report["message"]


class TestExtractorDriftCheck:
    """An install whose extractor payload differs from the active revision's
    rebuilds the collection instead of extending it. That is invisible until
    the worker has already re-extracted thousands of documents: the revision
    label is built from the profile *name*, so both revisions print the same
    string while holding different fingerprints."""

    PAYLOAD = {
        "backends": {"pdf": "pymupdf-raw"},
        "use_ocr": False,
        "version": "v1",
        "extraction_libraries": {"pymupdf": "1.27.1", "pymupdf4llm": "0.3.4"},
    }

    def test_nothing_indexed_yet_has_nothing_to_compare(self) -> None:
        report = _extractor_drift_check(self.PAYLOAD, [])

        assert report["status"] == "ok"
        assert report["drift"] == {}
        assert "no active revision" in report["message"]

    def test_a_matching_install_is_ok(self) -> None:
        report = _extractor_drift_check(self.PAYLOAD, [("papers", dict(self.PAYLOAD))])

        assert report["status"] == "ok"
        assert report["drift"] == {}

    def test_a_library_bump_warns_and_names_both_versions(self) -> None:
        """The real case: a second install resolved a newer pymupdf, so
        indexing from it re-extracts the corpus under a new profile."""
        recorded = dict(self.PAYLOAD)
        recorded["extraction_libraries"] = {"pymupdf": "1.28.2", "pymupdf4llm": "1.28.2"}

        report = _extractor_drift_check(self.PAYLOAD, [("papers", recorded)])

        assert report["status"] == "warning"
        assert report["drift"]["papers"] == [
            "extraction_libraries: pymupdf 1.28.2 -> 1.27.1, pymupdf4llm 1.28.2 -> 0.3.4"
        ]
        assert "papers" in report["message"]
        assert "1.27.1" in report["message"]

    def test_a_scalar_key_change_is_reported_whole(self) -> None:
        recorded = dict(self.PAYLOAD, use_ocr=True)

        report = _extractor_drift_check(self.PAYLOAD, [("papers", recorded)])

        assert report["drift"]["papers"] == ["use_ocr: True -> False"]

    def test_each_drifting_collection_is_named(self) -> None:
        recorded = dict(self.PAYLOAD, version="v0")

        report = _extractor_drift_check(
            self.PAYLOAD, [("notes", recorded), ("papers", dict(self.PAYLOAD))]
        )

        assert set(report["drift"]) == {"notes"}
        assert "notes" in report["message"]
        assert "papers" not in report["message"]

    def test_a_warning_does_not_fail_the_overall_report(self, tmp_path) -> None:
        """Rebuilding is the right answer to a real extractor change, so this
        cannot be the check that makes `cementic doctor` exit non-zero."""
        config = _config_with_model_path(tmp_path, exists=True)
        recorded = json.dumps(dict(self.PAYLOAD, version="v0"))
        mock_conn = MagicMock()
        mock_conn.execute.return_value.scalar.return_value = True
        mock_conn.execute.return_value.fetchall.return_value = [("papers", recorded)]
        mock_engine = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn

        with patch("cementic.doctor.get_engine", return_value=mock_engine):
            report = collect_doctor_report(config)

        assert report["checks"]["extractor_profile"]["status"] == "warning"
        assert report["ok"] is True
