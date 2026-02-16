"""Tests for CLI commands."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from seman.cli import app


runner = CliRunner()


class TestInfraCommands:
    """Test infrastructure management commands."""

    @patch("seman.cli.subprocess.run")
    def test_infra_up(self, mock_run):
        """Test infra up command."""
        mock_run.return_value = MagicMock(returncode=0)

        result = runner.invoke(app, ["infra-up"])

        assert result.exit_code == 0
        mock_run.assert_called_once()
        assert "podman-compose" in mock_run.call_args[0][0]
        assert "up" in mock_run.call_args[0][0]

    @patch("seman.cli.subprocess.run")
    def test_infra_up_failure(self, mock_run):
        """Test infra up command failure."""
        from subprocess import CalledProcessError

        mock_run.side_effect = CalledProcessError(1, "cmd")

        result = runner.invoke(app, ["infra-up"])

        assert result.exit_code == 1
        assert "Failed" in result.output

    @patch("seman.cli.subprocess.run")
    def test_infra_down(self, mock_run):
        """Test infra down command."""
        mock_run.return_value = MagicMock(returncode=0)

        result = runner.invoke(app, ["infra-down"])

        assert result.exit_code == 0
        mock_run.assert_called_once()

    @patch("seman.cli.subprocess.run")
    def test_infra_status(self, mock_run):
        """Test infra status command."""
        mock_run.return_value = MagicMock(returncode=0, stdout="CONTAINER STATUS")

        result = runner.invoke(app, ["infra-status"])

        assert result.exit_code == 0
        assert "CONTAINER STATUS" in result.output


class TestConvertCommands:
    """Test converter commands."""

    @patch("seman.cli.ConverterDaemon")
    def test_convert_start(self, mock_daemon_class):
        """Test convert start command."""
        mock_daemon = MagicMock()
        mock_daemon_class.return_value = mock_daemon

        result = runner.invoke(app, ["convert", "start", "/path/to/pdfs"])

        assert result.exit_code == 0
        mock_daemon_class.assert_called_once()
        mock_daemon.start.assert_called_once_with(["/path/to/pdfs"])

    def test_convert_status(self):
        """Test convert status command."""
        with patch("seman.cli.StateManager") as mock_state_manager:
            mock_state = MagicMock()
            mock_state.daemon_state.value = "running"
            mock_state.pid = 1234
            mock_state.watched_directories = ["/path"]
            mock_state.current_file = None
            mock_state.processed_count = 10
            mock_state.failed_count = 0

            mock_state_manager.return_value.load.return_value = mock_state

            result = runner.invoke(app, ["convert", "status"])

            assert result.exit_code == 0
            assert "running" in result.output

    @patch("seman.cli.os.kill")
    def test_convert_stop(self, mock_kill):
        """Test convert stop command."""
        with patch("seman.cli.StateManager") as mock_state_manager:
            mock_state = MagicMock()
            mock_state.pid = 1234
            mock_state_manager.return_value.load.return_value = mock_state

            result = runner.invoke(app, ["convert", "stop"])

            assert result.exit_code == 0
            mock_kill.assert_called_once_with(1234, 15)


class TestEmbedCommands:
    """Test embedder commands."""

    @patch("seman.cli.EmbedderDaemon")
    def test_embed_start(self, mock_daemon_class):
        """Test embed start command."""
        mock_daemon = MagicMock()
        mock_daemon_class.return_value = mock_daemon

        result = runner.invoke(app, ["embed", "start"])

        assert result.exit_code == 0
        mock_daemon_class.assert_called_once()
        mock_daemon.start.assert_called_once()

    def test_embed_status(self):
        """Test embed status command."""
        with patch("seman.cli.StateManager") as mock_state_manager:
            with patch("seman.cli.get_engine"):
                with patch("seman.cli.get_session_factory") as mock_session_factory:
                    mock_session = MagicMock()
                    mock_session.query.return_value.filter_by.return_value.count.return_value = 0
                    mock_session_factory.return_value = lambda: mock_session

                    mock_state = MagicMock()
                    mock_state.daemon_state.value = "stopped"
                    mock_state.pid = None
                    mock_state.current_file = None
                    mock_state_manager.return_value.load.return_value = mock_state

                    result = runner.invoke(app, ["embed", "status"])

                    assert result.exit_code == 0


class TestSearchCommand:
    """Test search command."""

    @patch("seman.cli.Searcher")
    def test_search_basic(self, mock_searcher_class):
        """Test basic search."""
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = [
            {
                "source_path": "/test.pdf",
                "content": "test result",
                "score": 0.95,
                "page_start": 1,
                "page_end": 2,
            }
        ]
        mock_searcher_class.return_value = mock_searcher

        result = runner.invoke(app, ["search", "test query"])

        assert result.exit_code == 0
        mock_searcher.search.assert_called_once_with("test query", top_k=10)
        assert "/test.pdf" in result.output

    @patch("seman.cli.Searcher")
    def test_search_no_results(self, mock_searcher_class):
        """Test search with no results."""
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = []
        mock_searcher_class.return_value = mock_searcher

        result = runner.invoke(app, ["search", "nonexistent"])

        assert result.exit_code == 0
        assert "No results" in result.output

    @patch("seman.cli.Searcher")
    def test_search_with_top_k(self, mock_searcher_class):
        """Test search with custom top_k."""
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = []
        mock_searcher_class.return_value = mock_searcher

        result = runner.invoke(app, ["search", "query", "-n", "20"])

        assert result.exit_code == 0
        mock_searcher.search.assert_called_once_with("query", top_k=20)

    @patch("seman.cli.Searcher")
    def test_search_error(self, mock_searcher_class):
        """Test search handles errors."""
        mock_searcher = MagicMock()
        mock_searcher.search.side_effect = Exception("Search failed")
        mock_searcher_class.return_value = mock_searcher

        result = runner.invoke(app, ["search", "query"])

        assert result.exit_code == 1
        assert "failed" in result.output.lower()
