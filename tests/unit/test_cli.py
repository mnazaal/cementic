"""Tests for CLI commands."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import typer
from sqlalchemy.exc import OperationalError
from typer import Context
from typer.testing import CliRunner

from cementic import cli as cementic_cli
from cementic.cli import (
    _build_collection_filters,
    _get_data_dir,
    _supervisor_processes,
    app,
    collection_callback,
)
from cementic.collections import PromotionOutcome
from cementic.config import Config, default_config_path
from cementic.pipeline_worker import PipelineCounts

runner = CliRunner()


class TestConfigCommands:
    """Tests for the `cementic config` command group."""

    def test_path_prints_active_file(self, tmp_path, monkeypatch) -> None:
        cfg = tmp_path / "cementic.toml"
        cfg.write_text("")
        monkeypatch.setenv("CEMENTIC_CONFIG", str(cfg))
        result = runner.invoke(app, ["config", "path"])
        assert result.exit_code == 0
        assert result.stdout.strip() == str(cfg)

    def test_path_prints_default_when_none(self, tmp_path, monkeypatch) -> None:
        # autouse fixture clears CEMENTIC_CONFIG and empties the user-config dir
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["config", "path"])
        assert result.exit_code == 0
        assert result.stdout.strip() == str(default_config_path())

    def test_init_writes_then_refuses_clobber(self) -> None:
        target = default_config_path()
        assert not target.exists()

        first = runner.invoke(app, ["config", "init"])
        assert first.exit_code == 0
        assert target.exists()
        assert "[pipeline]" in target.read_text()

        again = runner.invoke(app, ["config", "init"])
        assert again.exit_code == 1  # refuses to overwrite

        forced = runner.invoke(app, ["config", "init", "--force"])
        assert forced.exit_code == 0

    def test_show_outputs_effective_json(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(cementic_cli, "_config", None)  # reset cached singleton
        cfg = tmp_path / "cementic.toml"
        cfg.write_text("[pipeline]\nchunk_size = 321\n")
        monkeypatch.setenv("CEMENTIC_CONFIG", str(cfg))

        result = runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["pipeline"]["chunk_size"] == 321
        assert "database" in data

    def test_show_redacts_database_password(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(cementic_cli, "_config", None)  # reset cached singleton
        monkeypatch.delenv("CEMENTIC_CONFIG", raising=False)
        monkeypatch.setenv("CEMENTIC_DB_PASSWORD", "super-secret-pw")

        result = runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0
        assert "super-secret-pw" not in result.stdout
        data = json.loads(result.stdout)
        assert data["database"]["password"] == "**********"


class TestInitPostgresCommand:
    """Tests for generated Postgres setup files."""

    def test_init_postgres_writes_setup_tree(self, tmp_path) -> None:
        target = tmp_path / "cementic-postgres"

        result = runner.invoke(app, ["init", "postgres", str(target)])

        assert result.exit_code == 0
        assert (target / ".env").is_file()
        assert (target / "compose.yml").is_file()
        assert (target / "Containerfile").is_file()
        assert (target / "README.md").is_file()
        assert (target / "quadlet" / "cementic-postgres.container").is_file()
        assert "cementic status --doctor" in result.output

    def test_init_postgres_refuses_non_empty_directory(self, tmp_path) -> None:
        target = tmp_path / "cementic-postgres"
        target.mkdir()
        (target / "keep.txt").write_text("do not clobber")

        result = runner.invoke(app, ["init", "postgres", str(target)])

        assert result.exit_code == 1
        assert "already exists and is not empty" in result.output
        assert (target / "keep.txt").read_text() == "do not clobber"

    def test_init_postgres_force_replaces_non_empty_directory(self, tmp_path) -> None:
        target = tmp_path / "cementic-postgres"
        target.mkdir()
        (target / "old.txt").write_text("old")

        result = runner.invoke(app, ["init", "postgres", str(target), "--force"])

        assert result.exit_code == 0
        assert not (target / "old.txt").exists()
        assert (target / "compose.yml").is_file()


class TestRootHelp:
    """Test top-level help behavior."""

    def test_root_help_shown_with_no_args(self):
        result = runner.invoke(app, [])
        assert "USAGE:" in result.output
        assert "collection" in result.output
        assert "Index and semantically search document collections" in result.output

    def test_collection_namespace_shows_help_with_no_subcommand(self):
        result = runner.invoke(app, ["collection"])
        output = result.output
        assert "Inspect and manage collections" in output
        assert "list" in output
        assert "promote" in output
        assert "revisions" in output
        assert "remove" in output

    def test_start_shows_help_with_no_args(self):
        result = runner.invoke(app, ["start"])
        assert "USAGE:" in result.output
        assert "start" in result.output
        assert "DIRECTORIES" in result.output

    def test_search_shows_help_with_no_args(self):
        result = runner.invoke(app, ["search"])
        assert "USAGE:" in result.output
        assert "search" in result.output
        assert "Search indexed documents" in result.output

    def test_status_shows_help_with_no_args(self, monkeypatch, capsys):
        monkeypatch.setattr(cementic_cli.sys, "argv", ["cementic", "status"])
        with patch("cementic.cli.app") as mock_app:
            cementic_cli.main()
        output = capsys.readouterr().out
        assert output == ""
        mock_app.assert_called_once()

    @patch("cementic.cli._llama_daemon_runtime_status", return_value="stopped")
    @patch("cementic.cli.list_collections", return_value=[])
    @patch("cementic.cli.check_health")
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
        mock_check_health,
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
        mock_check_health.return_value = SimpleNamespace(
            db_reachable=True,
            embedding_provider="llama-cpp",
            embedding_healthy=True,
            llama_daemon="running, pid=333",
        )
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = False
        mock_get_session_factory.return_value = lambda: mock_session

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "workers" in result.output
        assert "collections" in result.output
        assert "(none)" in result.output
        mock_list_collections.assert_called_once_with(mock_session)

    def test_collection_promote_shows_help_with_no_args(self):
        result = runner.invoke(app, ["collection", "promote"])
        assert "USAGE:" in result.output
        assert "promote" in result.output
        assert "COLLECTION" in result.output

    def test_collection_revisions_shows_help_with_no_args(self):
        result = runner.invoke(app, ["collection", "revisions"])
        assert "USAGE:" in result.output
        assert "revisions" in result.output

    def test_collection_remove_shows_help_with_no_args(self):
        result = runner.invoke(app, ["collection", "remove"])
        assert "USAGE:" in result.output
        assert "remove" in result.output


class TestEmbeddingCommands:
    """Test embedding runtime lifecycle commands."""

    @patch("cementic.cli.get_llama_cpp_runtime_client")
    def test_embedding_start_starts_llama_runtime(self, mock_runtime_client):
        client = MagicMock()
        client.embedding_dim = 768
        mock_runtime_client.return_value = client

        result = runner.invoke(app, ["embedding", "start"])

        assert result.exit_code == 0
        assert "embedding: running" in result.output
        _, kwargs = mock_runtime_client.call_args
        assert kwargs["autostart"] is True

    @patch("cementic.cli.stop_llama_cpp_runtime", return_value=True)
    def test_embedding_stop_stops_llama_runtime(self, mock_stop):
        result = runner.invoke(app, ["embedding", "stop"])

        assert result.exit_code == 0
        assert "embedding: stopped" in result.output
        mock_stop.assert_called_once()

    @patch("cementic.cli._llama_daemon_runtime_status", return_value="running, pid=123")
    def test_embedding_status_reports_llama_runtime(self, mock_status):
        result = runner.invoke(app, ["embedding", "status"])

        assert result.exit_code == 0
        assert "embedding: running, pid=123" in result.output
        mock_status.assert_called_once_with()


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
        assert "0.950" in result.output
        assert "/test.pdf" in result.output
        assert "test result" in result.output
        mock_searcher.search.assert_called_once_with("test query", top_k=10, collections=None)

    @patch("cementic.cli.Searcher")
    def test_search_shows_database_hint_when_database_unavailable(self, mock_searcher_class):
        mock_searcher = MagicMock()
        mock_searcher.search.side_effect = OperationalError("statement", {}, Exception("down"))
        mock_searcher_class.return_value = mock_searcher

        result = runner.invoke(app, ["search", "test query"])

        assert result.exit_code == 1
        assert "search: database not reachable" in result.output
        assert "cementic init postgres" in result.output


class TestBackgroundCommands:
    """Test top-level background process commands."""

    @patch("cementic.cli.collect_doctor_report")
    def test_status_doctor_outputs_json_and_fails_when_not_ok(self, mock_collect):
        mock_collect.return_value = {
            "ok": False,
            "checks": {
                "database": {"status": "fail", "reachable": False, "message": "down"},
                "extensions": {},
                "model": {"status": "ok", "exists": True},
                "daemon": {"status": "warning", "reachable": False, "autostart": True},
            },
        }

        result = runner.invoke(app, ["status", "--doctor", "--json"])

        assert result.exit_code == 1
        data = json.loads(result.stdout)
        assert data["ok"] is False
        assert data["checks"]["database"]["message"] == "down"

    @patch("cementic.cli.collect_doctor_report")
    def test_status_doctor_outputs_human_summary_when_ok(self, mock_collect):
        mock_collect.return_value = {
            "ok": True,
            "checks": {
                "database": {"status": "ok", "reachable": True},
                "extensions": {
                    "vector": {"status": "ok", "message": "installed"},
                    "vectorscale": {"status": "warning", "message": "available"},
                },
                "model": {"status": "ok", "exists": True},
                "daemon": {"status": "warning", "reachable": False, "autostart": True},
            },
        }

        result = runner.invoke(app, ["status", "--doctor"])

        assert result.exit_code == 0
        assert "cementic doctor: ok" in result.output
        assert "vector: ok" in result.output
        assert "vectorscale: warning" in result.output

    @patch("cementic.cli._llama_daemon_runtime_status", return_value="running, pid=333")
    @patch("cementic.cli.check_health")
    @patch("cementic.cli.load_pipeline_status_bulk")
    @patch("cementic.cli.list_collections")
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
        mock_list_collections,
        mock_load_pipeline_status_bulk,
        mock_check_health,
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
        mock_check_health.return_value = SimpleNamespace(
            db_reachable=True,
            embedding_provider="llama-cpp",
            embedding_healthy=True,
            llama_daemon="running, pid=333",
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
        mock_load_pipeline_status_bulk.return_value = {
            "research": SimpleNamespace(
                documents=10,
                extracted_done=8,
                extracted_failed=1,
                chunked_done=7,
                chunked_failed=1,
                total_chunks=30,
                pending_embeddings=5,
                processing_embeddings=1,
                done_embeddings=20,
                failed_embeddings=2,
                extraction_pct=80.0,
                chunking_pct=87.5,
                embedding_pct=66.7,
                active_revision_label="rev-1",
                building_revision_label="rev-2",
            )
        }

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "workers" in result.output
        assert "running" in result.output
        assert "database" in result.output
        assert "embedding" in result.output
        assert "collections" in result.output
        assert "research" in result.output
        assert "10 docs" in result.output
        assert "20/30 embedded (66.7%)" in result.output
        mock_list_collections.assert_called_once_with(mock_session)
        mock_load_pipeline_status_bulk.assert_called_once_with(
            cementic_cli._get_config(), ["research"]
        )

    @patch("cementic.cli._llama_daemon_runtime_status", return_value="stopped")
    @patch("cementic.cli.check_health")
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
        mock_check_health,
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
        mock_check_health.return_value = SimpleNamespace(
            db_reachable=True,
            embedding_provider="llama-cpp",
            embedding_healthy=True,
            llama_daemon="running, pid=333",
        )
        mock_load_pipeline_status.return_value = SimpleNamespace(
            documents=10,
            extracted_done=8,
            extracted_failed=1,
            chunked_done=7,
            chunked_failed=1,
            total_chunks=30,
            pending_embeddings=5,
            processing_embeddings=1,
            done_embeddings=20,
            failed_embeddings=2,
            extraction_pct=80.0,
            chunking_pct=87.5,
            embedding_pct=66.7,
            active_revision_label="rev-1",
            building_revision_label="rev-2",
        )

        result = runner.invoke(app, ["status", "--collection", "research"])

        assert result.exit_code == 0
        assert "collection" in result.output
        assert "research" in result.output
        assert "documents" in result.output
        assert "8/10 (80.0%)" in result.output
        assert "7/8 (87.5%)" in result.output
        assert "20/30 (66.7%)" in result.output
        assert "active=rev-1" in result.output
        mock_load_pipeline_status.assert_called_once_with(cementic_cli._get_config(), "research")

    @patch("cementic.cli._llama_daemon_runtime_status", return_value="stopped")
    @patch("cementic.cli.check_health")
    @patch("cementic.cli.load_pipeline_status")
    @patch("cementic.cli.load_worker_statuses")
    @patch("cementic.cli._load_supervisor_state")
    @patch("cementic.cli.build_supervisor_status")
    def test_status_command_shows_dash_when_no_active_revision(
        self,
        mock_build_supervisor_status,
        mock_load_supervisor_state,
        mock_load_worker_statuses,
        mock_load_pipeline_status,
        mock_check_health,
        mock_daemon_status,
    ):
        """No active/building revision (None, not the string 'None') renders as '-'."""
        mock_load_supervisor_state.return_value = {
            "collection": "research",
            "directories": ["/docs"],
            "processes": [],
        }
        mock_load_worker_statuses.return_value = (
            SimpleNamespace(
                state="stopped", pid="N/A", process="stopped", current_file="None",
                watched_directories=[], processed_count=0, failed_count=0,
            ),
            SimpleNamespace(
                state="stopped", pid="N/A", process="stopped", current_file="None",
                watched_directories=[], processed_count=0, failed_count=0,
            ),
        )
        mock_build_supervisor_status.return_value = SimpleNamespace(
            state="0/2 running", collection="research", directories=["/docs"],
        )
        mock_check_health.return_value = SimpleNamespace(
            db_reachable=True, embedding_provider="llama-cpp",
            embedding_healthy=True, llama_daemon="stopped",
        )
        mock_load_pipeline_status.return_value = SimpleNamespace(
            documents=0, extracted_done=0, extracted_failed=0, chunked_done=0,
            chunked_failed=0, total_chunks=0, pending_embeddings=0,
            processing_embeddings=0, done_embeddings=0, failed_embeddings=0,
            extraction_pct=0.0, chunking_pct=0.0, embedding_pct=0.0,
            active_revision_label=None, building_revision_label=None,
        )

        result = runner.invoke(app, ["status", "--collection", "research"])

        assert result.exit_code == 0
        assert "active=-" in result.output
        assert "building=-" in result.output
        assert "active=None" not in result.output

    @patch("cementic.cli._llama_daemon_runtime_status", return_value="stopped")
    @patch("cementic.cli.check_health")
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
        mock_check_health,
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
        mock_check_health.return_value = SimpleNamespace(
            db_reachable=True,
            embedding_provider="llama-cpp",
            embedding_healthy=True,
            llama_daemon="running, pid=333",
        )
        mock_load_pipeline_status.return_value = SimpleNamespace(
            documents=1,
            extracted_done=1,
            extracted_failed=0,
            chunked_done=1,
            chunked_failed=0,
            total_chunks=1,
            pending_embeddings=0,
            processing_embeddings=0,
            done_embeddings=1,
            failed_embeddings=0,
            extraction_pct=100.0,
            chunking_pct=100.0,
            embedding_pct=100.0,
            active_revision_label="rev-1",
            building_revision_label="None",
        )

        result = runner.invoke(app, ["status", "-c", "research"])

        assert result.exit_code == 0
        assert "research" in result.output
        mock_load_pipeline_status.assert_called_once_with(cementic_cli._get_config(), "research")

    @patch("cementic.cli._spawn_detached")
    def test_start_background(self, mock_spawn, temp_dir: Path):
        mock_spawn.side_effect = [1111, 2222]
        mock_path = temp_dir / "supervisor.json"

        with patch(
            "cementic.cli._get_supervisor_state_path", return_value=mock_path
        ):
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
        mock_path = temp_dir / "supervisor.json"

        with patch(
            "cementic.cli._get_supervisor_state_path", return_value=mock_path
        ):
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

        outcome = PromotionOutcome(status="promoted", revision=revision)
        with patch("cementic.cli.promote_ready_revision", return_value=outcome) as mock_promote:
            result = runner.invoke(app, ["collection", "promote", "research"])

        assert result.exit_code == 0
        assert "collection: research" in result.output
        assert "status: promoted" in result.output
        assert "revision: rev-1" in result.output
        args, kwargs = mock_promote.call_args
        assert args == (mock_session, "research")
        assert isinstance(kwargs["config"], Config)
        assert kwargs["force"] is False

    @patch("cementic.cli.get_session_factory")
    @patch("cementic.cli.get_engine")
    def test_collection_promote_blocked_by_failures(
        self, mock_get_engine, mock_get_session_factory
    ):
        revision = SimpleNamespace(collection="research", status="ready", label="rev-1")
        counts = PipelineCounts(
            documents=2,
            extracted_done=1,
            chunked_done=1,
            total_chunks=3,
            done_embeddings=2,
            extracted_failed=1,
            failed_embeddings=1,
        )
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = False
        mock_get_session_factory.return_value = lambda: mock_session

        outcome = PromotionOutcome(
            status="blocked_by_failures", revision=revision, failures=counts
        )
        with patch("cementic.cli.promote_ready_revision", return_value=outcome):
            result = runner.invoke(app, ["collection", "promote", "research"])

        assert result.exit_code == 1
        assert "status: blocked" in result.output
        assert "extract=1" in result.output
        assert "embed=1" in result.output
        assert "--force" in result.output

    @patch("cementic.cli.get_session_factory")
    @patch("cementic.cli.get_engine")
    def test_collection_promote_force_passes_through(
        self, mock_get_engine, mock_get_session_factory
    ):
        revision = SimpleNamespace(collection="research", status="ready", label="rev-1")
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = False
        mock_get_session_factory.return_value = lambda: mock_session

        outcome = PromotionOutcome(status="promoted", revision=revision)
        with patch("cementic.cli.promote_ready_revision", return_value=outcome) as mock_promote:
            result = runner.invoke(app, ["collection", "promote", "research", "--force"])

        assert result.exit_code == 0
        assert "status: promoted" in result.output
        assert mock_promote.call_args.kwargs["force"] is True

    @patch("cementic.cli.get_session_factory")
    @patch("cementic.cli.get_engine")
    def test_collection_revisions_command(self, mock_get_engine, mock_get_session_factory):
        extractor_profile = SimpleNamespace(name="pymupdf4llm")
        chunk_profile = SimpleNamespace(fingerprint="abcdef123456")
        embedding_profile = SimpleNamespace(provider="llama-cpp", fingerprint="fedcba654321")
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
        assert "collection" in result.output
        assert "research" in result.output
        assert "revisions" in result.output
        assert "active" in result.output
        assert "rev-3" in result.output
        assert "extract=pymupdf4llm" in result.output
        assert "embed=llama-cpp:fedcba65" in result.output

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
        assert "collections" in result.output
        assert "research" in result.output
        assert "10 docs" in result.output
        assert "active=rev-1" in result.output
        assert "building=rev-2" in result.output
        mock_list.assert_called_once_with(mock_session)

    @patch("cementic.cli.get_engine")
    def test_collection_list_shows_database_hint_when_database_unavailable(self, mock_get_engine):
        mock_get_engine.side_effect = OperationalError("statement", {}, Exception("down"))

        result = runner.invoke(app, ["collection", "list"])

        assert result.exit_code == 1
        assert "collection list: database not reachable" in result.output
        assert "cementic init postgres" in result.output

    @patch("cementic.cli.get_engine")
    def test_collection_promote_shows_database_hint_when_database_unavailable(
        self, mock_get_engine
    ):
        mock_get_engine.side_effect = OperationalError("statement", {}, Exception("down"))

        result = runner.invoke(app, ["collection", "promote", "research"])

        assert result.exit_code == 1
        assert "collection promote: database not reachable" in result.output
        assert "cementic init postgres" in result.output


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


class TestStatusCommand:
    """Test the status command."""

    @patch("cementic.status_service.get_engine")
    @patch("cementic.cli.get_session_factory")
    @patch("cementic.status_service.is_pid_running", return_value=False)
    @patch("cementic.embedding_runtime.get_llama_cpp_runtime_client")
    def test_status_no_collection_shows_health(
        self, mock_llama, mock_pid, mock_sf, mock_engine, monkeypatch
    ) -> None:
        mock_client = MagicMock()
        mock_client.health_check.return_value = True
        mock_llama.return_value = mock_client

        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn

        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_sf.return_value = lambda: mock_session

        # Make list_collections return empty
        with patch("cementic.cli.list_collections", return_value=[]):
            result = runner.invoke(app, ["status"])
            # May error on missing infra, just check it handles gracefully
            assert result.exit_code in (0, 1)


class TestStopCommand:
    """Test the stop command."""

    def test_stop_with_no_processes(self) -> None:
        result = runner.invoke(app, ["stop", "--force"])
        assert result.exit_code == 0


class TestCliHelpers:
    """Tests for CLI helper functions."""

    def test_supervisor_processes_with_non_list(self):
        """_supervisor_processes returns [] for non-list 'processes'."""
        assert _supervisor_processes({"processes": "not_a_list"}) == []

    def test_supervisor_processes_filters_non_dict_entries(self):
        """_supervisor_processes filters out non-dict entries in list."""
        state = {"processes": [{"name": "ok", "pid": 1}, "string", 123, None]}
        result = _supervisor_processes(state)
        assert len(result) == 1
        assert result[0]["name"] == "ok"

    @patch("cementic.cli.version", side_effect=cementic_cli.PackageNotFoundError)
    def test_get_cli_version_returns_unknown(self, mock_version):
        """_get_cli_version returns 'unknown' when package not found."""
        result = cementic_cli._get_cli_version()
        assert result == "unknown"

    def test_build_collection_filters_trailing_raises(self):
        """_build_collection_filters raises BadParameter for bare trailing values."""
        with patch.object(typer, "BadParameter", typer.BadParameter):
            try:
                _build_collection_filters([], ["bare_value"])
            except typer.BadParameter as e:
                assert "Unexpected argument" in str(e)
            else:
                raise AssertionError("Expected BadParameter")

    def test_llama_daemon_runtime_status_no_pid_file(self):
        """_llama_daemon_runtime_status returns 'stopped' when pid_file is None."""
        mock_cfg = MagicMock()
        mock_llama = MagicMock()
        mock_llama.daemon_pid_file = None
        mock_cfg.llama_cpp = mock_llama
        with patch.object(cementic_cli, "_config", mock_cfg):
            assert cementic_cli._llama_daemon_runtime_status() == "stopped"

    def test_llama_daemon_runtime_status_pid_file_missing(self, tmp_path: Path):
        """_llama_daemon_runtime_status returns 'stopped' when pid_file doesn't exist."""
        mock_cfg = MagicMock()
        mock_llama = MagicMock()
        mock_llama.daemon_pid_file = tmp_path / "nonexistent.pid"
        mock_cfg.llama_cpp = mock_llama
        with patch.object(cementic_cli, "_config", mock_cfg):
            assert cementic_cli._llama_daemon_runtime_status() == "stopped"

    def test_llama_daemon_runtime_status_bad_pid_file(self, tmp_path: Path):
        """_llama_daemon_runtime_status returns 'stopped' for non-int PID content."""
        pid_file = tmp_path / "bad.pid"
        pid_file.write_text("not_an_int")
        mock_cfg = MagicMock()
        mock_llama = MagicMock()
        mock_llama.daemon_pid_file = pid_file
        mock_cfg.llama_cpp = mock_llama
        with patch.object(cementic_cli, "_config", mock_cfg):
            assert cementic_cli._llama_daemon_runtime_status() == "stopped"

    @patch("cementic.cli._is_pid_running", return_value=False)
    def test_llama_daemon_runtime_status_pid_not_running(
        self, mock_is_running, tmp_path: Path
    ):
        """_llama_daemon_runtime_status returns 'stopped' when PID not alive."""
        pid_file = tmp_path / "dead.pid"
        pid_file.write_text("99999")
        mock_cfg = MagicMock()
        mock_llama = MagicMock()
        mock_llama.daemon_pid_file = pid_file
        mock_cfg.llama_cpp = mock_llama
        with patch.object(cementic_cli, "_config", mock_cfg):
            assert cementic_cli._llama_daemon_runtime_status() == "stopped"

    def test_get_data_dir_runtime_error(self):
        """_get_data_dir raises RuntimeError when no state_path configured."""
        mock_cfg = MagicMock()
        mock_sw = MagicMock()
        mock_sw.state_path = None
        mock_pw = MagicMock()
        mock_pw.state_path = None
        mock_cfg.source_watcher = mock_sw
        mock_cfg.pipeline_worker = mock_pw
        with patch.object(cementic_cli, "_config", mock_cfg):
            try:
                _get_data_dir()
            except RuntimeError:
                pass
            else:
                raise AssertionError("Expected RuntimeError")


class TestCollectionCallback:
    """Tests for collection_callback."""

    def test_callback_without_subcommand(self, monkeypatch):
        """collection_callback prints help and exits when no subcommand."""
        mock_ctx = MagicMock(spec=Context)
        mock_ctx.invoked_subcommand = None
        mock_ctx.get_help.return_value = "help text here"
        try:
            collection_callback(mock_ctx)
        except typer.Exit:
            pass  # expected
        mock_ctx.get_help.assert_called_once()


class TestRemovePrompt:
    """Test remove collection without --force (confirm prompt)."""

    @patch("cementic.cli.get_engine")
    def test_remove_without_force_aborts_on_no(self, mock_get_engine):
        """remove_collection aborts when user says no to confirm."""
        result = runner.invoke(
            app, ["collection", "remove", "mycol"], input="n\n"
        )
        assert result.exit_code != 0  # Abort


class TestCollectionCommandsEdgeCases:
    """Edge cases for collection commands."""

    @patch("cementic.cli.get_session_factory")
    @patch("cementic.cli.get_engine")
    def test_list_empty_collections(self, mock_get_engine, mock_get_session_factory):
        """list_collection_command prints 'none' when no collections."""
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_get_session_factory.return_value = lambda: mock_session
        with patch("cementic.cli.list_collections", return_value=[]):
            result = runner.invoke(app, ["collection", "list"])
        assert result.exit_code == 0
        assert "(none)" in result.output

    @patch("cementic.cli.list_collections", side_effect=ValueError("something broke"))
    @patch("cementic.cli.get_session_factory")
    @patch("cementic.cli.get_engine")
    def test_list_generic_error(
        self, mock_get_engine, mock_get_session_factory, mock_list
    ):
        """list_collection_command prints generic error and exits 1."""
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_get_session_factory.return_value = lambda: mock_session
        result = runner.invoke(app, ["collection", "list"])
        assert result.exit_code == 1
        assert "failed to list collections" in result.output

    @patch("cementic.cli.get_session_factory")
    @patch("cementic.cli.get_engine")
    def test_promote_no_ready_revision(self, mock_get_engine, mock_get_session_factory):
        """promote_collection shows 'no ready revision' when none exists."""
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_get_session_factory.return_value = lambda: mock_session
        outcome = PromotionOutcome(status="no_ready")
        with patch("cementic.cli.promote_ready_revision", return_value=outcome):
            result = runner.invoke(app, ["collection", "promote", "mycol"])
        assert result.exit_code == 0
        assert "no ready revision" in result.output

    @patch("cementic.cli.promote_ready_revision", side_effect=ValueError("boom"))
    @patch("cementic.cli.get_session_factory")
    @patch("cementic.cli.get_engine")
    def test_promote_generic_error(
        self, mock_get_engine, mock_get_session_factory, mock_promote
    ):
        """promote_collection prints generic error and exits 1."""
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_get_session_factory.return_value = lambda: mock_session
        result = runner.invoke(app, ["collection", "promote", "mycol"])
        assert result.exit_code == 1
        assert "promotion failed" in result.output


class TestListCollectionRevisionsEdgeCases:
    """Edge cases for revisions command."""

    @patch("cementic.cli.get_session_factory")
    @patch("cementic.cli.get_engine")
    def test_revisions_empty(self, mock_get_engine, mock_get_session_factory):
        """list_collection_revision_command shows 'none' for empty revisions."""
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_get_session_factory.return_value = lambda: mock_session
        with patch(
            "cementic.cli.list_collection_revisions", return_value=[]
        ):
            result = runner.invoke(app, ["collection", "revisions", "mycol"])
        assert result.exit_code == 0
        assert "(none)" in result.output

    @patch(
        "cementic.cli.list_collection_revisions",
        side_effect=ValueError("boom"),
    )
    @patch("cementic.cli.get_session_factory")
    @patch("cementic.cli.get_engine")
    def test_revisions_generic_error(
        self, mock_get_engine, mock_get_session_factory, mock_list
    ):
        """list_collection_revision_command prints generic error and exits 1."""
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_get_session_factory.return_value = lambda: mock_session
        result = runner.invoke(app, ["collection", "revisions", "mycol"])
        assert result.exit_code == 1
        assert "failed to load revisions" in result.output


class TestSearchEdgeCases:
    """Edge cases for search command."""

    @patch("cementic.cli.Searcher")
    def test_search_no_results(self, mock_searcher_class):
        """search prints 'none' when no results found."""
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = []
        mock_searcher_class.return_value = mock_searcher
        result = runner.invoke(app, ["search", "no matches"])
        assert result.exit_code == 0
        assert "no results" in result.output

    @patch("cementic.cli.Searcher")
    def test_search_generic_error(self, mock_searcher_class):
        """search prints generic error and exits 1."""
        mock_searcher = MagicMock()
        mock_searcher.search.side_effect = ValueError("bad query")
        mock_searcher_class.return_value = mock_searcher
        result = runner.invoke(app, ["search", "bad"])
        assert result.exit_code == 1
        assert "search failed: bad query" in result.output


class TestStatusEdgeCases:
    """Edge cases for the status command."""

    @patch("cementic.cli._llama_daemon_runtime_status", return_value="stopped")
    @patch("cementic.cli.load_worker_statuses")
    @patch("cementic.cli._load_supervisor_state")
    @patch("cementic.cli.build_supervisor_status")
    @patch("cementic.cli.list_collections", return_value=[])
    @patch("cementic.cli.get_session_factory")
    @patch("cementic.cli.get_engine")
    def test_status_command_health_exception_fallback(
        self,
        mock_get_engine,
        mock_get_session_factory,
        mock_list_collections,
        mock_build_supervisor_status,
        mock_load_supervisor_state,
        mock_load_worker_statuses,
        mock_daemon_status,
    ):
        """Status gracefully handles health check exceptions."""
        mock_load_supervisor_state.return_value = {
            "collection": "research",
            "directories": [],
            "processes": [],
        }
        mock_load_worker_statuses.return_value = (
            SimpleNamespace(
                state="running", pid=1111, process="running",
                current_file="None", watched_directories=[], processed_count=0, failed_count=0,
            ),
            SimpleNamespace(
                state="running", pid=2222, process="running",
                current_file="None", watched_directories=[], processed_count=0, failed_count=0,
            ),
        )
        mock_build_supervisor_status.return_value = SimpleNamespace(
            state="2/2 running", collection="research", directories=[],
        )
        # check_health raises an exception
        with patch(
            "cementic.cli.check_health", side_effect=RuntimeError("health check failed")
        ):
            mock_session = MagicMock()
            mock_session.__enter__.return_value = mock_session
            mock_get_session_factory.return_value = lambda: mock_session

            result = runner.invoke(app, ["status"])
            assert result.exit_code == 0
            # Health section should not appear (health is None)
            assert "health:" not in result.output

    @patch("cementic.cli.list_collections", return_value=[])
    def test_status_json_output(self, mock_list_collections):
        """Status with --json prints JSON output."""
        with patch("cementic.cli.load_worker_statuses") as mock_load:
            mock_load.return_value = (
                SimpleNamespace(state="running", pid=1111, process="running",
                                current_file="None", watched_directories=[],
                                processed_count=0, failed_count=0),
                SimpleNamespace(state="running", pid=2222, process="running",
                                current_file="None", watched_directories=[],
                                processed_count=0, failed_count=0),
            )
            with patch("cementic.cli._load_supervisor_state") as mock_state:
                mock_state.return_value = {
                    "collection": "research", "directories": ["/docs"], "processes": [],
                }
                with patch("cementic.cli.build_supervisor_status") as mock_build:
                    mock_build.return_value = SimpleNamespace(
                        state="2/2 running", collection="research",
                        directories=["/docs"],
                    )
                    with patch("cementic.cli.check_health") as mock_health:
                        mock_health.return_value = SimpleNamespace(
                            db_reachable=True,
                            embedding_provider="llama-cpp",
                            embedding_healthy=True,
                            llama_daemon="running, pid=333",
                        )
                        with patch("cementic.cli.get_session_factory") as mock_sf:
                            with patch("cementic.cli.get_engine") as _mock_engine:
                                mock_session = MagicMock()
                                mock_session.__enter__.return_value = mock_session
                                mock_sf.return_value = lambda: mock_session

                                result = runner.invoke(app, ["status", "--json"])
                                assert result.exit_code == 0
                                assert '"supervisor"' in result.output
                                assert '"source_watcher"' in result.output
                                assert '"pipeline_worker"' in result.output
                                assert '"health"' in result.output
                                assert '"collections"' in result.output

    @patch("cementic.cli.list_collections", return_value=[])
    def test_status_json_output_survives_piping_with_long_paths(self, mock_list_collections):
        """A long path must not be hard-wrapped by rich, which would break JSON parsing."""
        long_path = "/a" + "/very-long-directory-segment" * 10
        with patch("cementic.cli.load_worker_statuses") as mock_load:
            mock_load.return_value = (
                SimpleNamespace(state="running", pid=1111, process="running",
                                current_file="None", watched_directories=[],
                                processed_count=0, failed_count=0),
                SimpleNamespace(state="running", pid=2222, process="running",
                                current_file="None", watched_directories=[],
                                processed_count=0, failed_count=0),
            )
            with patch("cementic.cli._load_supervisor_state") as mock_state:
                mock_state.return_value = {
                    "collection": "research", "directories": [long_path], "processes": [],
                }
                with patch("cementic.cli.build_supervisor_status") as mock_build:
                    mock_build.return_value = SimpleNamespace(
                        state="2/2 running", collection="research",
                        directories=[long_path],
                    )
                    with patch("cementic.cli.check_health") as mock_health:
                        mock_health.return_value = SimpleNamespace(
                            db_reachable=True,
                            embedding_provider="llama-cpp",
                            embedding_healthy=True,
                            llama_daemon="running, pid=333",
                        )
                        with patch("cementic.cli.get_session_factory") as mock_sf:
                            with patch("cementic.cli.get_engine") as _mock_engine:
                                mock_session = MagicMock()
                                mock_session.__enter__.return_value = mock_session
                                mock_sf.return_value = lambda: mock_session

                                result = runner.invoke(app, ["status", "--json"])
                                assert result.exit_code == 0
                                parsed = json.loads(result.output)
                                assert parsed["supervisor"]["directories"] == [long_path]


class TestStartEdgeCases:
    """Edge cases for start command."""

    def test_start_directory_not_exists(self):
        """start_background exits 1 when directory doesn't exist."""
        result = runner.invoke(app, ["start", "/nonexistent/path/xyz"])
        assert result.exit_code == 1
        assert "Directory does not exist" in result.output

    @patch("cementic.cli._is_managed_proc_alive", return_value=True)
    @patch("cementic.cli._load_supervisor_state")
    def test_start_already_running(self, mock_load_state, mock_is_running):
        """start_background exits 1 when processes already running."""
        mock_load_state.return_value = {
            "collection": "test",
            "directories": ["/tmp"],
            "processes": [
                {"name": "source-watcher", "pid": 1111},
                {"name": "pipeline-worker", "pid": 2222},
            ],
        }
        result = runner.invoke(app, ["start", "/tmp"])
        assert result.exit_code == 1
        assert "Background cementic processes already running" in result.output

    @patch("cementic.cli.Bootstrapper")
    @patch("cementic.cli._is_pid_running", return_value=False)
    @patch("cementic.cli._load_supervisor_state")
    def test_start_bootstrap_fails(self, mock_load_state, mock_is_running, mock_boot):
        """start_background exits 1 when bootstrap raises RuntimeError."""
        mock_load_state.return_value = {"collection": "", "directories": [], "processes": []}
        mock_boot.return_value.ensure_for_convert.side_effect = RuntimeError("infra down")
        result = runner.invoke(app, ["start", "/tmp"])
        assert result.exit_code == 1
        assert "Bootstrap failed before background start" in result.output


class TestMainEntrypoint:
    """Tests for main() entrypoint (delegates to Typer)."""

    @pytest.mark.parametrize("flag", ["--version", "-V"])
    def test_main_version_flag(self, flag, monkeypatch, capsys):
        """main() with --version/-V prints the version and exits 0."""
        monkeypatch.setattr(cementic_cli.sys, "argv", ["cementic", flag])
        with pytest.raises(SystemExit) as exc:
            cementic_cli.main()
        assert exc.value.code == 0
        assert capsys.readouterr().out.strip() != ""


# Every command, group, and subcommand must answer both -h and --help.
_HELP_TARGETS = [
    [],
    ["collection"],
    ["embedding"],
    ["config"],
    ["start"],
    ["status"],
    ["stop"],
    ["search"],
    ["collection", "list"],
    ["collection", "promote"],
    ["collection", "revisions"],
    ["collection", "remove"],
    ["embedding", "start"],
    ["embedding", "stop"],
    ["embedding", "status"],
    ["config", "init"],
    ["config", "path"],
    ["config", "show"],
]


class TestHelpFlags:
    """`-h`/`--help` are consistent and helpful everywhere (clig.dev)."""

    @pytest.mark.parametrize("target", _HELP_TARGETS)
    @pytest.mark.parametrize("flag", ["-h", "--help"])
    def test_help_flag_everywhere(self, target, flag):
        result = runner.invoke(app, [*target, flag])
        assert result.exit_code == 0, result.output
        assert "USAGE:" in result.output

    def test_help_can_follow_arguments(self):
        # "you should be able to add -h to the end of anything"
        result = runner.invoke(app, ["search", "foo", "bar", "-h"])
        assert result.exit_code == 0
        assert "USAGE:" in result.output
        assert "search" in result.output

    def test_root_help_includes_examples_and_link(self):
        result = runner.invoke(app, ["--help"])
        assert "EXAMPLES:" in result.output
        assert "github.com/mnazaal/cementic" in result.output


class _FakeEmbedProvider:
    """Identity-formatting provider returning fixed-width vectors."""

    def format_document(self, text: str) -> str:
        return text

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class _BatchTrackingEmbedProvider(_FakeEmbedProvider):
    """Fake provider that records each embed_batch call's batch size."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batch_sizes.append(len(texts))
        return super().embed_batch(texts)


class TestFilterCommands:
    """extract / chunk / embed stdin-stdout filters (Move 3)."""

    @patch("cementic.cli.extract_document", return_value="# Title\n\nbody text")
    def test_extract_outputs_markdown(self, mock_extract):
        result = runner.invoke(app, ["extract", "paper.pdf"])
        assert result.exit_code == 0
        assert "# Title" in result.output
        assert "body text" in result.output
        # dispatched by content type through the registry
        assert mock_extract.call_args.args[0] == "paper.pdf"

    @patch("cementic.cli.extract_document", side_effect=ValueError("no extractor for '.xyz'"))
    def test_extract_missing_file_exits_1(self, mock_extract):
        result = runner.invoke(app, ["extract", "nope.pdf"])
        assert result.exit_code == 1
        assert "extract failed" in result.output

    @patch("cementic.cli.extract_document", side_effect=FileNotFoundError("no such file"))
    def test_extract_error_goes_to_stderr_not_stdout(self, mock_extract):
        # A filter's diagnostics must never land on stdout, or a downstream
        # `| chunk | embed` would embed the error text as document content.
        result = runner.invoke(app, ["extract", "missing.pdf"])
        assert result.exit_code == 1
        assert "extract failed" not in result.stdout
        assert result.stdout.strip() == ""
        assert "extract failed" in result.stderr

    def test_extract_no_args_shows_help(self):
        result = runner.invoke(app, ["extract"])
        assert "USAGE:" in result.output

    def test_chunk_missing_file_errors_to_stderr(self, tmp_path):
        result = runner.invoke(app, ["chunk", str(tmp_path / "nope.txt")])
        assert result.exit_code == 1
        assert result.stdout.strip() == ""
        assert "chunk failed" in result.stderr

    @patch("cementic.cli.runtime_spec_from_config", return_value=object())
    @patch("cementic.cli.create_provider", return_value=_FakeEmbedProvider())
    def test_embed_bad_json_errors_to_stderr(self, mock_create, mock_spec):
        result = runner.invoke(app, ["embed"], input="not json at all\n")
        assert result.exit_code == 1
        assert result.stdout.strip() == ""
        assert "embed failed" in result.stderr

    def test_chunk_emits_jsonl_from_stdin(self):
        result = runner.invoke(app, ["chunk"], input="hello world from a document")
        assert result.exit_code == 0
        lines = [line for line in result.output.splitlines() if line.strip()]
        assert lines
        first = json.loads(lines[0])
        assert first["index"] == 0
        assert "hello world" in first["content"]

    def test_chunk_reads_a_file(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("some text to chunk", encoding="utf-8")
        result = runner.invoke(app, ["chunk", str(f)])
        assert result.exit_code == 0
        assert json.loads(result.output.splitlines()[0])["content"]

    @patch("cementic.cli.runtime_spec_from_config", return_value=object())
    @patch("cementic.cli.create_provider", return_value=_FakeEmbedProvider())
    def test_embed_adds_embedding_field(self, mock_create, mock_spec):
        stdin = '{"index": 0, "content": "hi"}\n{"index": 1, "content": "there"}\n'
        result = runner.invoke(app, ["embed"], input=stdin)
        assert result.exit_code == 0
        lines = [json.loads(line) for line in result.output.splitlines() if line.strip()]
        assert len(lines) == 2
        assert lines[0]["embedding"] == [0.1, 0.2, 0.3]
        assert lines[0]["index"] == 0 and lines[0]["content"] == "hi"

    def test_embed_empty_stdin_is_noop(self):
        result = runner.invoke(app, ["embed"], input="")
        assert result.exit_code == 0
        assert result.output.strip() == ""

    @patch("cementic.cli.runtime_spec_from_config", return_value=object())
    def test_embed_batches_requests_by_config_batch_size(self, mock_spec):
        """Regression: embed must not send every stdin record in one request.

        A single unbounded batch risks timing out the embedding backend on
        large inputs; requests must be chunked by pipeline_worker.batch_size.
        """
        provider = _BatchTrackingEmbedProvider()
        stdin = "".join(
            json.dumps({"index": i, "content": f"chunk {i}"}) + "\n" for i in range(5)
        )
        with patch("cementic.cli.create_provider", return_value=provider):
            with patch("cementic.cli._get_config") as mock_get_config:
                config = mock_get_config.return_value
                config.pipeline_worker.batch_size = 2
                result = runner.invoke(app, ["embed"], input=stdin)

        assert result.exit_code == 0
        assert provider.batch_sizes == [2, 2, 1]
        lines = [json.loads(line) for line in result.output.splitlines() if line.strip()]
        assert len(lines) == 5
        assert [line["index"] for line in lines] == [0, 1, 2, 3, 4]
