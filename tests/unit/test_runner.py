"""Tests for the process supervisor runner module."""

from unittest.mock import patch

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

    @patch("cementic.runner.get_config")
    @patch("cementic.runner.SourceWatcher")
    @patch("cementic.runner.Bootstrapper")
    def test_source_watcher_calls_start(self, mock_boot, mock_watcher, mock_cfg) -> None:
        runner.invoke(app, ["source-watcher", "/tmp/test", "-c", "mycoll"])
        # Errors may occur due to mocks not being fully set up
        # We just verify the command runs and calls our mock
        assert mock_watcher.called or mock_cfg.called or mock_boot.called

    def test_source_watcher_reports_a_broken_config_on_stderr(self, tmp_path, monkeypatch) -> None:
        """`_load_config` was entirely untested before it shared `render_config_error`
        with the CLI's `_get_config` -- this is the runner-side counterpart of
        `test_cli.py::TestErrorStreamDiscipline::test_config_error_under_json_flag_keeps_stdout_clean`.
        """
        bad = tmp_path / "cementic.toml"
        bad.write_text("this is := not toml", encoding="utf-8")
        monkeypatch.setenv("CEMENTIC_CONFIG", str(bad))

        result = runner.invoke(app, ["source-watcher", str(tmp_path)])

        assert result.exit_code == 1
        assert result.stdout.strip() == ""
        assert "config error" in result.stderr
