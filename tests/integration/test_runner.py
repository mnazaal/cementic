"""Integration tests for runner.py subprocess entrypoints."""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from cementic.runner import app, main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestSourceWatcherCommand:
    """Tests for runner source-watcher command."""

    @patch("cementic.runner.SourceWatcher")
    @patch("cementic.runner.Bootstrapper")
    def test_success_path(self, mock_boot_cls, mock_watcher_cls, runner):
        """Bootstrap succeeds, watcher start called, KeyboardInterrupt stops."""
        mock_boot = MagicMock()
        mock_watcher = MagicMock()
        mock_boot_cls.return_value = mock_boot
        mock_watcher_cls.return_value = mock_watcher
        # simulate KeyboardInterrupt during start() to exercise stop() path
        mock_watcher.start.side_effect = KeyboardInterrupt

        result = runner.invoke(app, ["source-watcher", "/tmp/test"])
        # KeyboardInterrupt is caught, so exit 0
        assert result.exit_code == 0
        mock_boot.ensure_for_convert.assert_called_once()
        mock_watcher.start.assert_called_once_with(
            ["/tmp/test"], collection="default"
        )
        mock_watcher.stop.assert_called_once()

    @patch("cementic.runner.SourceWatcher")
    @patch("cementic.runner.Bootstrapper")
    def test_bootstrap_failure(self, mock_boot_cls, mock_watcher_cls, runner):
        """Bootstrap raises RuntimeError → exit code 1, message printed."""
        mock_boot = MagicMock()
        mock_boot.ensure_for_convert.side_effect = RuntimeError("boom")
        mock_boot_cls.return_value = mock_boot

        result = runner.invoke(app, ["source-watcher", "/tmp/test"])
        assert result.exit_code == 1
        assert "Bootstrap failed" in result.stdout

    @patch("cementic.runner.SourceWatcher")
    @patch("cementic.runner.Bootstrapper")
    def test_custom_collection(self, mock_boot_cls, mock_watcher_cls, runner):
        """Collection flag is passed through to watcher.start."""
        mock_boot = MagicMock()
        mock_watcher = MagicMock()
        mock_boot_cls.return_value = mock_boot
        mock_watcher_cls.return_value = mock_watcher
        mock_watcher.start.side_effect = KeyboardInterrupt

        result = runner.invoke(
            app, ["source-watcher", "/tmp/a", "-c", "mycoll"]
        )
        assert result.exit_code == 0
        mock_watcher.start.assert_called_once_with(
            ["/tmp/a"], collection="mycoll"
        )


class TestPipelineWorkerCommand:
    """Tests for runner pipeline-worker command."""

    @patch("cementic.runner.PipelineWorker")
    @patch("cementic.runner.Bootstrapper")
    def test_success_path(self, mock_boot_cls, mock_worker_cls, runner):
        """Bootstrap succeeds, worker start called, KeyboardInterrupt stops."""
        mock_boot = MagicMock()
        mock_worker = MagicMock()
        mock_boot_cls.return_value = mock_boot
        mock_worker_cls.return_value = mock_worker
        mock_worker.start.side_effect = KeyboardInterrupt

        result = runner.invoke(app, ["pipeline-worker"])
        assert result.exit_code == 0
        mock_boot.ensure_for_index.assert_called_once()
        mock_worker.start.assert_called_once_with(collection="default")
        mock_worker.stop.assert_called_once()

    @patch("cementic.runner.PipelineWorker")
    @patch("cementic.runner.Bootstrapper")
    def test_bootstrap_failure(self, mock_boot_cls, mock_worker_cls, runner):
        """Bootstrap raises RuntimeError → exit 1."""
        mock_boot = MagicMock()
        mock_boot.ensure_for_index.side_effect = RuntimeError("no db")
        mock_boot_cls.return_value = mock_boot

        result = runner.invoke(app, ["pipeline-worker"])
        assert result.exit_code == 1
        assert "Bootstrap failed" in result.stdout

    @patch("cementic.runner.PipelineWorker")
    @patch("cementic.runner.Bootstrapper")
    def test_custom_collection(self, mock_boot_cls, mock_worker_cls, runner):
        """Collection flag passed to worker.start."""
        mock_boot = MagicMock()
        mock_worker = MagicMock()
        mock_boot_cls.return_value = mock_boot
        mock_worker_cls.return_value = mock_worker
        mock_worker.start.side_effect = KeyboardInterrupt

        result = runner.invoke(app, ["pipeline-worker", "-c", "mycoll"])
        assert result.exit_code == 0
        mock_worker.start.assert_called_once_with(collection="mycoll")


class TestMainEntrypoint:
    """Tests for the main() function."""

    def test_main_calls_app(self):
        """main() invokes the typer app."""
        with patch("cementic.runner.app") as mock_app:
            main()
            mock_app.assert_called_once()
