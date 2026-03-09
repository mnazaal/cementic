"""Tests for CLI commands."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from seman import cli as seman_cli
from seman.cli import app

runner = CliRunner()


class TestRootHelp:
    """Test top-level help behavior."""

    def test_root_help_shown_with_no_args(self, monkeypatch, capsys):
        """Running `seman` with no args should print root help."""
        monkeypatch.setattr(seman_cli.sys, "argv", ["seman"])

        seman_cli.main()
        output = capsys.readouterr().out

        assert "Usage: seman" in output
        assert "Commands:" in output

    def test_root_help_shown_with_short_help_flag(self, monkeypatch, capsys):
        """Running `seman -h` should print root help."""
        monkeypatch.setattr(seman_cli.sys, "argv", ["seman", "-h"])

        seman_cli.main()
        output = capsys.readouterr().out

        assert "Usage: seman" in output
        assert "Commands:" in output


class TestLegacyCommands:
    """Test removed legacy command surface."""

    def test_convert_subcommands_removed(self):
        """Legacy convert command should not be available."""
        result = runner.invoke(app, ["convert", "start", "/path/to/pdfs"])
        assert result.exit_code != 0

    def test_index_subcommands_removed(self):
        """Legacy index command should not be available."""
        result = runner.invoke(app, ["index", "start"])
        assert result.exit_code != 0

    def test_reset_command_removed(self):
        """Reset command is removed from simplified CLI."""
        result = runner.invoke(app, ["reset", "--force"])
        assert result.exit_code != 0


class TestSearchCommand:
    """Test search command."""

    def test_search_subcommand_help_disabled(self):
        """Subcommand-level --help is intentionally disabled."""
        result = runner.invoke(app, ["search", "--help"])

        assert result.exit_code != 0
        assert "No such option" in result.output

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
        mock_searcher.search.assert_called_once_with("test query", top_k=10, collections=None)
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
        mock_searcher.search.assert_called_once_with("query", top_k=20, collections=None)

    def test_search_rejects_removed_top_k_long_flag(self):
        """Search no longer supports --top-k long option."""
        result = runner.invoke(app, ["search", "query", "--top-k", "20"])

        assert result.exit_code != 0
        assert "No such option" in result.output

    @patch("seman.cli.Searcher")
    def test_search_with_collection_filter(self, mock_searcher_class):
        """Test search with collection filters."""
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = []
        mock_searcher_class.return_value = mock_searcher

        result = runner.invoke(
            app,
            ["search", "query", "--collection", "work", "--collection", "personal"],
        )

        assert result.exit_code == 0
        mock_searcher.search.assert_called_once_with(
            "query",
            top_k=10,
            collections=["work", "personal"],
        )

    @patch("seman.cli.Searcher")
    def test_search_with_space_separated_collection_filter(self, mock_searcher_class):
        """Test search with one collection flag and multiple values."""
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = []
        mock_searcher_class.return_value = mock_searcher

        result = runner.invoke(
            app,
            ["search", "query", "--collection", "work", "personal"],
        )

        assert result.exit_code == 0
        mock_searcher.search.assert_called_once_with(
            "query",
            top_k=10,
            collections=["work", "personal"],
        )

    @patch("seman.cli.Searcher")
    def test_search_with_short_space_separated_collection_filter(self, mock_searcher_class):
        """Test search with one short collection flag and multiple values."""
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = []
        mock_searcher_class.return_value = mock_searcher

        result = runner.invoke(
            app,
            ["search", "query", "-c", "work", "personal"],
        )

        assert result.exit_code == 0
        mock_searcher.search.assert_called_once_with(
            "query",
            top_k=10,
            collections=["work", "personal"],
        )

    def test_search_with_trailing_collection_without_flag_errors(self):
        """Search should error when trailing collections omit --collection."""
        result = runner.invoke(app, ["search", "query", "personal"])

        assert result.exit_code != 0
        assert "Use --collection/-c" in result.output

    @patch("seman.cli.Searcher")
    def test_search_error(self, mock_searcher_class):
        """Test search handles errors."""
        mock_searcher = MagicMock()
        mock_searcher.search.side_effect = Exception("Search failed")
        mock_searcher_class.return_value = mock_searcher

        result = runner.invoke(app, ["search", "query"])

        assert result.exit_code == 1
        assert "failed" in result.output.lower()


class TestBackgroundCommands:
    """Test top-level background process commands."""

    def test_status_subcommand_help_disabled(self):
        """Subcommand-level --help is intentionally disabled."""
        result = runner.invoke(app, ["status", "--help"])

        assert result.exit_code != 0
        assert "No such option" in result.output

    @patch("seman.cli._spawn_detached")
    def test_start_background(self, mock_spawn, temp_dir: Path):
        """Start command spawns converter and indexer."""
        mock_spawn.side_effect = [1111, 2222]

        with patch("seman.cli.supervisor_state_path", temp_dir / "supervisor.json"):
            result = runner.invoke(app, ["start", str(temp_dir), "--collection", "test"])

        assert result.exit_code == 0
        assert mock_spawn.call_count == 2
        assert "Started seman in background" in result.output

    @patch("seman.cli._spawn_detached")
    def test_start_background_accepts_multiple_directories(self, mock_spawn, temp_dir: Path):
        """Start command accepts multiple directories."""
        mock_spawn.side_effect = [1111, 2222]
        second_dir = temp_dir / "second"
        second_dir.mkdir()

        with patch("seman.cli.supervisor_state_path", temp_dir / "supervisor.json"):
            result = runner.invoke(app, ["start", str(temp_dir), str(second_dir)])

        assert result.exit_code == 0
        first_command = mock_spawn.call_args_list[0].args[0]
        assert first_command[1:4] == ["-m", "seman.runner", "converter"]
        assert first_command[4:6] == [str(temp_dir), str(second_dir)]
        assert first_command[6:] == ["--collection", "default"]

    def test_start_background_missing_directory(self, temp_dir: Path):
        """Start command errors on missing directory."""
        with patch("seman.cli.supervisor_state_path", temp_dir / "supervisor.json"):
            result = runner.invoke(app, ["start", "/nonexistent/path", "--collection", "test"])

        assert result.exit_code == 1
        assert "does not exist" in result.output

    def test_stop_background_no_state(self, temp_dir: Path):
        """Stop command handles missing state file."""
        with patch("seman.cli.supervisor_state_path", temp_dir / "supervisor.json"):
            result = runner.invoke(app, ["stop"])

        assert result.exit_code == 0
        assert "No background seman processes found" in result.output

    @patch("seman.cli._wait_for_exit", return_value=[])
    @patch("seman.cli.os.kill")
    def test_stop_background_stops_and_clears_state(self, mock_kill, mock_wait, temp_dir: Path):
        """Stop command waits for exit and removes supervisor state."""
        state_path = temp_dir / "supervisor.json"
        state_path.write_text('{"processes": [{"name": "converter", "pid": 1234}]}')

        with patch("seman.cli.supervisor_state_path", state_path):
            result = runner.invoke(app, ["stop"])

        assert result.exit_code == 0
        mock_kill.assert_called_once_with(1234, 15)
        assert "Stopped 1 process(es)" in result.output
        assert not state_path.exists()

    @patch("seman.cli._wait_for_exit", return_value=[1234])
    @patch("seman.cli.os.kill")
    def test_stop_background_keeps_state_when_timeout(self, mock_kill, mock_wait, temp_dir: Path):
        """Stop command keeps state for processes that did not stop yet."""
        state_path = temp_dir / "supervisor.json"
        state_path.write_text('{"processes": [{"name": "converter", "pid": 1234}]}')

        with patch("seman.cli.supervisor_state_path", state_path):
            result = runner.invoke(app, ["stop"])

        assert result.exit_code == 0
        assert "Stop timed out" in result.output
        assert state_path.exists()

    @patch("seman.cli.get_session_factory")
    @patch("seman.cli.get_engine")
    @patch("seman.cli._embedder_state_manager")
    @patch("seman.cli._converter_state_manager")
    def test_status_command(
        self,
        mock_converter_state_manager,
        mock_embedder_state_manager,
        mock_get_engine,
        mock_get_session_factory,
        temp_dir: Path,
    ):
        """Status command renders daemon and queue summaries."""
        converter_state = MagicMock()
        converter_state.daemon_state = "running"
        converter_state.pid = 1001
        embedder_state = MagicMock()
        embedder_state.daemon_state = "running"
        embedder_state.pid = 1002

        mock_converter_state_manager.return_value.load.return_value = converter_state
        mock_embedder_state_manager.return_value.load.return_value = embedder_state

        mock_session = MagicMock()
        mock_session.query.return_value.filter_by.return_value.count.side_effect = [1, 2, 3, 4, 1]
        mock_get_session_factory.return_value = lambda: mock_session

        state_path = temp_dir / "supervisor.json"
        state_path.write_text('{"processes": [{"name": "converter", "pid": 1001}]}')

        with patch("seman.cli.supervisor_state_path", state_path):
            with patch("seman.cli._is_pid_running", return_value=True):
                result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "Converter Status" in result.output
        assert "Indexer Status" in result.output
        assert "Supervisor" in result.output
        assert "Embedding Queue" in result.output


class TestCollectionCommands:
    """Test collection management commands."""

    @patch("seman.cli.get_session_factory")
    @patch("seman.cli.get_engine")
    def test_delete_collection_success(self, mock_get_engine, mock_get_session_factory):
        """Delete collection removes matching documents and chunks."""
        mock_doc = MagicMock()
        mock_doc.id = 1

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.filter_by.return_value.all.return_value = [mock_doc]
        mock_session.query.return_value.filter.return_value.delete.side_effect = [5, 1]
        mock_get_session_factory.return_value = lambda: mock_session

        result = runner.invoke(app, ["delete-collection", "test", "--force"])

        assert result.exit_code == 0
        assert "Deleted collection 'test'" in result.output

    @patch("seman.cli.get_session_factory")
    @patch("seman.cli.get_engine")
    def test_delete_collection_not_found(self, mock_get_engine, mock_get_session_factory):
        """Deleting missing collection prints warning."""
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.filter_by.return_value.all.return_value = []
        mock_get_session_factory.return_value = lambda: mock_session

        result = runner.invoke(app, ["delete-collection", "missing", "--force"])

        assert result.exit_code == 0
        assert "not found" in result.output.lower()
