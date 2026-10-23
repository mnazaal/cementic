"""Tests for the process supervisor runner module."""

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from cementic.runner import app

runner = CliRunner()


class TestRunnerCommands:
    """Tests for runner Typer commands."""

    def test_source_watcher_help(self) -> None:
        result = runner.invoke(app, ["source-watcher", "--help"])
        assert result.exit_code == 0
        assert "watch" in result.stdout.lower()

    def test_pipeline_worker_help(self) -> None:
        result = runner.invoke(app, ["pipeline-worker", "--help"])
        assert result.exit_code == 0
        assert "worker" in result.stdout.lower()

    def test_root_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "source-watcher" in result.stdout
        assert "pipeline-worker" in result.stdout

    def test_source_watcher_missing_directories(self) -> None:
        result = runner.invoke(app, ["source-watcher"])
        assert result.exit_code != 0

    def test_source_watcher_reports_a_broken_config_on_stderr(self, tmp_path, monkeypatch) -> None:
        """`_load_config` was entirely untested before it shared `render_config_error`
        with the CLI's `cli_config` -- this is the runner-side counterpart of
        `test_cli.py::TestErrorStreamDiscipline::test_config_error_under_json_flag_keeps_stdout_clean`.
        """
        bad = tmp_path / "cementic.toml"
        bad.write_text("this is := not toml", encoding="utf-8")
        monkeypatch.setenv("CEMENTIC_CONFIG", str(bad))

        result = runner.invoke(app, ["source-watcher", str(tmp_path)])

        assert result.exit_code == 1
        assert result.stdout.strip() == ""
        assert "config error" in result.stderr


class TestRunnerErrorExits:
    """The one-line stderr reasons that land in the background log.

    `cementic start` points the user at that log; each of these paths used to
    be uncovered, so a regression back to a raw traceback (or a zero exit)
    would pass the suite.
    """

    def test_source_watcher_rejects_an_invalid_collection_name(self) -> None:
        result = runner.invoke(app, ["source-watcher", "/tmp/x", "-c", "bad name!"])
        assert result.exit_code == 1
        assert "Collection name" in result.stderr

    def test_pipeline_worker_rejects_an_invalid_collection_name(self) -> None:
        result = runner.invoke(app, ["pipeline-worker", "-c", "bad name!"])
        assert result.exit_code == 1
        assert "Collection name" in result.stderr

    @patch("cementic.runner.get_config")
    @patch("cementic.runner.Bootstrapper")
    @patch("cementic.runner.SourceWatcher")
    def test_source_watcher_startup_runtime_error_is_one_stderr_line(
        self, mock_watcher, mock_boot, mock_cfg
    ) -> None:
        mock_watcher.return_value.start.side_effect = RuntimeError("no watchable directories")
        result = runner.invoke(app, ["source-watcher", "/tmp/x"])
        assert result.exit_code == 1
        assert "Source watcher failed: no watchable directories" in result.stderr

    @patch("cementic.runner.get_config")
    @patch("cementic.runner.Bootstrapper")
    @patch("cementic.runner.PipelineWorker")
    def test_pipeline_worker_startup_runtime_error_is_one_stderr_line(
        self, mock_worker, mock_boot, mock_cfg
    ) -> None:
        mock_worker.return_value.start.side_effect = RuntimeError("unindexable dimension")
        mock_worker.return_value.fatal_reason = None
        result = runner.invoke(app, ["pipeline-worker"])
        assert result.exit_code == 1
        assert "Pipeline worker failed: unindexable dimension" in result.stderr

    @patch("cementic.runner.get_config")
    @patch("cementic.runner.Bootstrapper")
    @patch("cementic.runner.SourceWatcher")
    def test_bootstrap_failure_is_one_stderr_line(
        self, mock_watcher, mock_boot, mock_cfg
    ) -> None:
        mock_boot.return_value.ensure_for_convert.side_effect = RuntimeError("db unreachable")
        result = runner.invoke(app, ["source-watcher", "/tmp/x"])
        assert result.exit_code == 1
        assert "Bootstrap failed: db unreachable" in result.stderr


class TestRunnerLifecycle:
    """Folded from the former tests/integration/test_runner.py (2026-10-22)."""

    @pytest.mark.parametrize(
        ("argv", "worker_class", "expected_start"),
        [
            (
                ["source-watcher", "/tmp/a", "-c", "mycoll"],
                "cementic.runner.SourceWatcher",
                ((["/tmp/a"],), {"collection": "mycoll"}),
            ),
            (
                ["pipeline-worker", "-c", "mycoll"],
                "cementic.runner.PipelineWorker",
                ((), {"collection": "mycoll"}),
            ),
        ],
    )
    def test_ctrl_c_during_start_stops_cleanly_with_exit_0(
        self, argv, worker_class, expected_start
    ) -> None:
        with (
            patch("cementic.runner.get_config"),
            patch("cementic.runner.Bootstrapper"),
            patch(worker_class) as worker_cls,
        ):
            worker = worker_cls.return_value
            worker.start.side_effect = KeyboardInterrupt
            worker.fatal_reason = None

            result = runner.invoke(app, argv)

        assert result.exit_code == 0
        args, kwargs = expected_start
        worker.start.assert_called_once_with(*args, **kwargs)
        worker.stop.assert_called_once()

    @patch("cementic.runner.get_config")
    @patch("cementic.runner.Bootstrapper")
    @patch("cementic.runner.PipelineWorker")
    def test_a_fatal_reason_after_start_returns_exits_1(
        self, mock_worker, mock_boot, mock_cfg
    ) -> None:
        """A startup failure that returns instead of raising (already running,
        lock held) used to be indistinguishable from success by exit code."""
        mock_worker.return_value.start.return_value = None
        mock_worker.return_value.fatal_reason = "already running"

        result = runner.invoke(app, ["pipeline-worker"])

        assert result.exit_code == 1
        mock_worker.return_value.stop.assert_not_called()
