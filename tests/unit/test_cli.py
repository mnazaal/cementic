"""Tests for CLI commands."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sqlalchemy.exc import OperationalError
from typer.testing import CliRunner

from cementic import cli as cementic_cli
from cementic.cli import app

runner = CliRunner()


class TestRootHelp:
    """Test top-level help behavior."""

    def test_root_help_shown_with_no_args(self, monkeypatch, capsys):
        monkeypatch.setattr(cementic_cli.sys, "argv", ["cementic"])
        cementic_cli.main()
        output = capsys.readouterr().out
        assert "Usage: cementic" in output
        assert "collection" in output
        assert "Index and semantically search PDF collections" in output

    def test_collection_namespace_shows_help_with_no_subcommand(self, monkeypatch, capsys):
        monkeypatch.setattr(cementic_cli.sys, "argv", ["cementic", "collection"])
        cementic_cli.main()
        output = capsys.readouterr().out
        assert "Inspect and manage collections" in output
        assert "list" in output
        assert "promote" in output
        assert "revisions" in output
        assert "remove" in output

    def test_start_shows_help_with_no_args(self, monkeypatch, capsys):
        monkeypatch.setattr(cementic_cli.sys, "argv", ["cementic", "start"])
        cementic_cli.main()
        output = capsys.readouterr().out
        assert "Start background indexing" in output
        assert "Usage: cementic start" in output

    def test_search_shows_help_with_no_args(self, monkeypatch, capsys):
        monkeypatch.setattr(cementic_cli.sys, "argv", ["cementic", "search"])
        cementic_cli.main()
        output = capsys.readouterr().out
        assert "Run semantic search over indexed chunks" in output
        assert "Usage: cementic search" in output

    def test_status_shows_help_with_no_args(self, monkeypatch, capsys):
        monkeypatch.setattr(cementic_cli.sys, "argv", ["cementic", "status"])
        with patch("cementic.cli.app") as mock_app:
            cementic_cli.main()
        output = capsys.readouterr().out
        assert output == ""
        mock_app.assert_called_once()

    @patch("cementic.cli._llama_daemon_runtime_status", return_value="stopped")
    @patch("cementic.cli.list_collections", return_value=[])
    @patch("cementic.cli.get_session_factory")
    @patch("cementic.cli.get_engine")
    @patch("cementic.cli.load_worker_statuses")
    @patch("cementic.cli._load_supervisor_state")
    @patch("cementic.cli.build_supervisor_status")
    def test_bare_status_runs_status_command(
        self,
        mock_build_supervisor_status,
        mock_load_supervisor_state,
        mock_load_worker_statuses,
        mock_get_engine,
        mock_get_session_factory,
        mock_list_collections,
        mock_daemon_status,
    ):
        mock_load_supervisor_state.return_value = {
            "collection": "research",
            "directories": ["/docs"],
            "processes": [],
        }
        mock_load_worker_statuses.return_value = (
            SimpleNamespace(
                state="running",
                pid="111",
                process="running",
                current_file="None",
                watched_directories=["/docs"],
                processed_count=3,
                failed_count=1,
            ),
            SimpleNamespace(
                state="running",
                pid="222",
                process="running",
                current_file="None",
                watched_directories=[],
                processed_count=0,
                failed_count=0,
            ),
        )
        mock_build_supervisor_status.return_value = SimpleNamespace(
            state="2/2 running",
            collection="research",
            directories=["/docs"],
        )
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = False
        mock_get_session_factory.return_value = lambda: mock_session

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "session collection: research" in result.output
        assert "collections:" in result.output
        assert "- none" in result.output
        mock_list_collections.assert_called_once_with(mock_session)
        mock_daemon_status.assert_called_once_with()

    def test_collection_promote_shows_help_with_no_args(self, monkeypatch, capsys):
        monkeypatch.setattr(cementic_cli.sys, "argv", ["cementic", "collection", "promote"])
        cementic_cli.main()
        output = capsys.readouterr().out
        assert "Promote a ready revision" in output
        assert "Usage: cementic collection promote <COLLECTION>" in output

    def test_collection_revisions_shows_help_with_no_args(self, monkeypatch, capsys):
        monkeypatch.setattr(cementic_cli.sys, "argv", ["cementic", "collection", "revisions"])
        cementic_cli.main()
        output = capsys.readouterr().out
        assert "Show revision history" in output
        assert "Usage: cementic collection revisions <COLLECTION>" in output

    def test_collection_remove_shows_help_with_no_args(self, monkeypatch, capsys):
        monkeypatch.setattr(cementic_cli.sys, "argv", ["cementic", "collection", "remove"])
        cementic_cli.main()
        output = capsys.readouterr().out
        assert "Remove one collection" in output
        assert "Usage: cementic collection remove <COLLECTION>" in output


class TestSearchCommand:
    """Test search command."""

    @patch("cementic.cli.Searcher")
    def test_search_basic(self, mock_searcher_class):
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
        assert "query: test query" in result.output
        assert "results:" in result.output
        assert "rank=1" in result.output
        assert "source=/test.pdf" in result.output
        mock_searcher.search.assert_called_once_with("test query", top_k=10, collections=None)

    @patch("cementic.cli.Searcher")
    def test_search_shows_database_hint_when_database_unavailable(self, mock_searcher_class):
        mock_searcher = MagicMock()
        mock_searcher.search.side_effect = OperationalError("statement", {}, Exception("down"))
        mock_searcher_class.return_value = mock_searcher

        result = runner.invoke(app, ["search", "test query"])

        assert result.exit_code == 1
        assert "search failed: database not reachable" in result.output
        assert "run `cementic start` to start infrastructure" in result.output


class TestBackgroundCommands:
    """Test top-level background process commands."""

    @patch("cementic.cli._llama_daemon_runtime_status", return_value="running, pid=333")
    @patch("cementic.cli.list_collections")
    @patch("cementic.cli.load_pipeline_status")
    @patch("cementic.cli.get_session_factory")
    @patch("cementic.cli.get_engine")
    @patch("cementic.cli.load_worker_statuses")
    @patch("cementic.cli._load_supervisor_state")
    @patch("cementic.cli.build_supervisor_status")
    def test_status_command_shows_global_overview(
        self,
        mock_build_supervisor_status,
        mock_load_supervisor_state,
        mock_load_worker_statuses,
        mock_get_engine,
        mock_get_session_factory,
        mock_load_pipeline_status,
        mock_list_collections,
        mock_daemon_status,
    ):
        mock_load_supervisor_state.return_value = {
            "collection": "research",
            "directories": ["/docs"],
            "processes": [],
        }
        mock_load_worker_statuses.return_value = (
            SimpleNamespace(
                state="running",
                pid="111",
                process="running",
                current_file="None",
                watched_directories=["/docs"],
                processed_count=3,
                failed_count=1,
            ),
            SimpleNamespace(
                state="running",
                pid="222",
                process="running",
                current_file="/docs/a.pdf",
                watched_directories=[],
                processed_count=0,
                failed_count=0,
            ),
        )
        mock_build_supervisor_status.return_value = SimpleNamespace(
            state="2/2 running",
            collection="research",
            directories=["/docs"],
        )
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = False
        mock_get_session_factory.return_value = lambda: mock_session
        mock_list_collections.return_value = [
            SimpleNamespace(
                name="research",
                documents=10,
                active_revision_label="rev-1",
                building_revision_label="rev-2",
            )
        ]
        mock_load_pipeline_status.side_effect = [
            SimpleNamespace(
                documents=10,
                extracted_done=8,
                chunked_done=7,
                pending_embeddings=5,
                processing_embeddings=1,
                done_embeddings=20,
                failed_embeddings=2,
                active_revision_label="rev-1",
                building_revision_label="rev-2",
            )
        ]

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "session collection: research" in result.output
        assert "workers: 2/2 running" in result.output
        assert "source watcher:" in result.output
        assert "pipeline worker:" in result.output
        assert "search daemon: running, pid=333" in result.output
        assert "collections:" in result.output
        assert "name=research" in result.output
        assert "extracted=8" in result.output
        assert "chunked=7" in result.output
        assert "embedded=20" in result.output
        assert "active=rev-1" in result.output
        mock_list_collections.assert_called_once_with(mock_session)
        mock_daemon_status.assert_called_once_with()
        mock_load_pipeline_status.assert_called_once_with(cementic_cli.config, "research")

    @patch("cementic.cli._llama_daemon_runtime_status", return_value="stopped")
    @patch("cementic.cli.load_pipeline_status")
    @patch("cementic.cli.load_worker_statuses")
    @patch("cementic.cli._load_supervisor_state")
    @patch("cementic.cli.build_supervisor_status")
    def test_status_command_shows_one_collection_when_requested(
        self,
        mock_build_supervisor_status,
        mock_load_supervisor_state,
        mock_load_worker_statuses,
        mock_load_pipeline_status,
        mock_daemon_status,
    ):
        mock_load_supervisor_state.return_value = {
            "collection": "research",
            "directories": ["/docs"],
            "processes": [],
        }
        mock_load_worker_statuses.return_value = (
            SimpleNamespace(
                state="running",
                pid="111",
                process="running",
                current_file="None",
                watched_directories=["/docs"],
                processed_count=3,
                failed_count=1,
            ),
            SimpleNamespace(
                state="running",
                pid="222",
                process="running",
                current_file="/docs/a.pdf",
                watched_directories=[],
                processed_count=0,
                failed_count=0,
            ),
        )
        mock_build_supervisor_status.return_value = SimpleNamespace(
            state="2/2 running",
            collection="research",
            directories=["/docs"],
        )
        mock_load_pipeline_status.return_value = SimpleNamespace(
            documents=10,
            extracted_done=8,
            chunked_done=7,
            pending_embeddings=5,
            processing_embeddings=1,
            done_embeddings=20,
            failed_embeddings=2,
            active_revision_label="rev-1",
            building_revision_label="rev-2",
        )

        result = runner.invoke(app, ["status", "--collection", "research"])

        assert result.exit_code == 0
        assert "session collection: research" in result.output
        assert "collection: research" in result.output
        assert "pipeline:" in result.output
        assert "documents=10" in result.output
        assert "active=rev-1" in result.output
        mock_load_pipeline_status.assert_called_once_with(cementic_cli.config, "research")
        mock_daemon_status.assert_called_once_with()

    @patch("cementic.cli._llama_daemon_runtime_status", return_value="stopped")
    @patch("cementic.cli.load_pipeline_status")
    @patch("cementic.cli.load_worker_statuses")
    @patch("cementic.cli._load_supervisor_state")
    @patch("cementic.cli.build_supervisor_status")
    def test_status_command_accepts_collection_short_flag(
        self,
        mock_build_supervisor_status,
        mock_load_supervisor_state,
        mock_load_worker_statuses,
        mock_load_pipeline_status,
        mock_daemon_status,
    ):
        mock_load_supervisor_state.return_value = {
            "collection": "research",
            "directories": ["/docs"],
            "processes": [],
        }
        mock_load_worker_statuses.return_value = (
            SimpleNamespace(
                state="running",
                pid="111",
                process="running",
                current_file="None",
                watched_directories=["/docs"],
                processed_count=3,
                failed_count=1,
            ),
            SimpleNamespace(
                state="running",
                pid="222",
                process="running",
                current_file="None",
                watched_directories=[],
                processed_count=0,
                failed_count=0,
            ),
        )
        mock_build_supervisor_status.return_value = SimpleNamespace(
            state="2/2 running",
            collection="research",
            directories=["/docs"],
        )
        mock_load_pipeline_status.return_value = SimpleNamespace(
            documents=1,
            extracted_done=1,
            chunked_done=1,
            pending_embeddings=0,
            processing_embeddings=0,
            done_embeddings=1,
            failed_embeddings=0,
            active_revision_label="rev-1",
            building_revision_label="None",
        )

        result = runner.invoke(app, ["status", "-c", "research"])

        assert result.exit_code == 0
        assert "collection: research" in result.output
        mock_load_pipeline_status.assert_called_once_with(cementic_cli.config, "research")

    @patch("cementic.cli._spawn_detached")
    def test_start_background(self, mock_spawn, temp_dir: Path):
        mock_spawn.side_effect = [1111, 2222]

        with patch("cementic.cli.supervisor_state_path", temp_dir / "supervisor.json"):
            with patch("cementic.cli.Bootstrapper") as mock_bootstrapper:
                mock_bootstrapper.return_value.ensure_for_convert.return_value = None
                mock_bootstrapper.return_value.ensure_for_index.return_value = None
                result = runner.invoke(app, ["start", str(temp_dir), "--collection", "test"])

        assert result.exit_code == 0
        assert mock_spawn.call_count == 2
        first_command = mock_spawn.call_args_list[0].args[0]
        second_command = mock_spawn.call_args_list[1].args[0]
        assert first_command[1:4] == ["-m", "cementic.runner", "source-watcher"]
        assert second_command[1:4] == ["-m", "cementic.runner", "pipeline-worker"]
        assert second_command[4:] == ["--collection", "test"]

    @patch("cementic.cli._spawn_detached")
    def test_start_background_mentions_default_collection(self, mock_spawn, temp_dir: Path):
        mock_spawn.side_effect = [1111, 2222]

        with patch("cementic.cli.supervisor_state_path", temp_dir / "supervisor.json"):
            with patch("cementic.cli.Bootstrapper") as mock_bootstrapper:
                mock_bootstrapper.return_value.ensure_for_convert.return_value = None
                mock_bootstrapper.return_value.ensure_for_index.return_value = None
                result = runner.invoke(app, ["start", str(temp_dir)])

        assert result.exit_code == 0
        assert "collection: default" in result.output
        assert "documents will be indexed into 'default'" in result.output

    @patch("cementic.cli.get_session_factory")
    @patch("cementic.cli.get_engine")
    def test_collection_promote_command(self, mock_get_engine, mock_get_session_factory):
        revision = SimpleNamespace(collection="research", status="ready", label="rev-1")
        revision_query = MagicMock()
        revision_query.filter_by.return_value.order_by.return_value.first.return_value = revision
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = False
        mock_session.query.return_value = revision_query
        mock_get_session_factory.return_value = lambda: mock_session

        with patch("cementic.cli.promote_ready_revision", return_value=revision) as mock_promote:
            result = runner.invoke(app, ["collection", "promote", "research"])

        assert result.exit_code == 0
        assert "collection: research" in result.output
        assert "status: promoted" in result.output
        assert "revision: rev-1" in result.output
        mock_promote.assert_called_once_with(mock_session, "research")

    @patch("cementic.cli.get_session_factory")
    @patch("cementic.cli.get_engine")
    def test_collection_revisions_command(self, mock_get_engine, mock_get_session_factory):
        extractor_profile = SimpleNamespace(name="pymupdf4llm")
        chunk_profile = SimpleNamespace(fingerprint="abcdef123456")
        embedding_profile = SimpleNamespace(provider="ollama", fingerprint="fedcba654321")
        revision = SimpleNamespace(
            id=3,
            status="active",
            label="rev-3",
            extractor_profile=extractor_profile,
            chunk_profile=chunk_profile,
            embedding_profile=embedding_profile,
        )
        revision_query = MagicMock()
        revision_query.filter_by.return_value.order_by.return_value.all.return_value = [revision]
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = False
        mock_session.query.return_value = revision_query
        mock_get_session_factory.return_value = lambda: mock_session

        result = runner.invoke(app, ["collection", "revisions", "research"])

        assert result.exit_code == 0
        assert "collection: research" in result.output
        assert "revisions:" in result.output
        assert "id=3" in result.output
        assert "status=active" in result.output
        assert "rev-3" in result.output
        assert "pymupdf4llm" in result.output

    @patch("cementic.cli.get_session_factory")
    @patch("cementic.cli.get_engine")
    def test_collection_list_command(self, mock_get_engine, mock_get_session_factory):
        summary = SimpleNamespace(
            name="research",
            documents=10,
            active_revision_label="rev-1",
            building_revision_label="rev-2",
        )
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = False
        mock_get_session_factory.return_value = lambda: mock_session

        with patch("cementic.cli.list_collections", return_value=[summary]) as mock_list:
            result = runner.invoke(app, ["collection", "list"])

        assert result.exit_code == 0
        assert "collections:" in result.output
        assert "name=research" in result.output
        assert "documents=10" in result.output
        assert "active=rev-1" in result.output
        assert "building=rev-2" in result.output
        mock_list.assert_called_once_with(mock_session)

    @patch("cementic.cli.get_engine")
    def test_collection_list_shows_database_hint_when_database_unavailable(self, mock_get_engine):
        mock_get_engine.side_effect = OperationalError("statement", {}, Exception("down"))

        result = runner.invoke(app, ["collection", "list"])

        assert result.exit_code == 1
        assert "collection list failed: database not reachable" in result.output
        assert "run `cementic start` to start infrastructure" in result.output

    @patch("cementic.cli.get_engine")
    def test_collection_promote_shows_database_hint_when_database_unavailable(
        self, mock_get_engine
    ):
        mock_get_engine.side_effect = OperationalError("statement", {}, Exception("down"))

        result = runner.invoke(app, ["collection", "promote", "research"])

        assert result.exit_code == 1
        assert "collection promote failed: database not reachable" in result.output
        assert "run `cementic start` to start infrastructure" in result.output


class TestCollectionCommands:
    """Test collection management commands."""

    @patch("cementic.cli.get_session_factory")
    @patch("cementic.cli.get_engine")
    def test_collection_remove_not_found(self, mock_get_engine, mock_get_session_factory):
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = False
        mock_session.query.return_value.filter_by.return_value.all.return_value = []
        mock_get_session_factory.return_value = lambda: mock_session

        with patch("cementic.cli.delete_collection_records", return_value=None):
            result = runner.invoke(app, ["collection", "remove", "missing", "--force"])

        assert result.exit_code == 0
        assert "collection: missing" in result.output
        assert "status: not found" in result.output
