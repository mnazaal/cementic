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
from cementic.collections import PromotionOutcome, ReindexOutcome
from cementic.config import Config, default_config_path
from cementic.pipeline_worker import PipelineCounts
from cementic.state import StateManager
from cementic.status_service import WorkerStatus
from cementic.supervisor import ManagedProcess

runner = CliRunner()


@pytest.fixture(autouse=True)
def _collection_exists_by_default():
    """Assume a named collection exists unless a test says otherwise.

    Most tests here drive the CLI against a mocked session whose `.query()` is
    stubbed for one specific call, so the real existence check cannot run
    against it. Tests covering the unknown-collection path patch this to False.
    """
    with patch("cementic.cli.collection_exists", return_value=True):
        yield


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

    def test_path_reports_unusable_explicit_config(self, tmp_path, monkeypatch) -> None:
        """Regression: `config path` printed the fallback for a CEMENTIC_CONFIG
        pointing nowhere -- the exact symptom config_path_error exists to
        diagnose, hidden by the one command a user would run to check it."""
        monkeypatch.setenv("CEMENTIC_CONFIG", str(tmp_path / "nope.toml"))
        result = runner.invoke(app, ["config", "path"])
        assert result.exit_code == 1
        assert "does not exist" in result.output
        assert str(default_config_path()) not in result.output

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

    def test_init_postgres_compose_persists_data_in_a_named_volume(self, tmp_path) -> None:
        """Without a named volume the database lives in the container's writable
        layer, so the `compose down` this directory's README documents would
        delete every indexed document, chunk and embedding."""
        target = tmp_path / "cementic-postgres"

        result = runner.invoke(app, ["init", "postgres", str(target)])

        assert result.exit_code == 0
        compose = (target / "compose.yml").read_text()
        assert "cementic-postgres-data:/var/lib/postgresql" in compose
        assert "\nvolumes:\n  cementic-postgres-data:" in compose

    def test_init_postgres_refuses_non_empty_directory(self, tmp_path) -> None:
        target = tmp_path / "cementic-postgres"
        target.mkdir()
        (target / "keep.txt").write_text("do not clobber")

        result = runner.invoke(app, ["init", "postgres", str(target)])

        assert result.exit_code == 1
        assert "already exists and is not empty" in result.output
        assert (target / "keep.txt").read_text() == "do not clobber"

    def test_init_postgres_force_overwrites_setup_files(self, tmp_path) -> None:
        target = tmp_path / "cementic-postgres"
        target.mkdir()
        (target / "compose.yml").write_text("stale template")

        result = runner.invoke(app, ["init", "postgres", str(target), "--force"])

        assert result.exit_code == 0
        assert (target / "compose.yml").read_text() != "stale template"

    def test_init_postgres_force_leaves_unrelated_files_alone(self, tmp_path) -> None:
        """`--force` must not be a recursive delete of whatever it is pointed at.

        It used to `rmtree` the target, so `cementic init postgres ~ --force`
        destroyed the user's home directory before writing five files into it.
        """
        target = tmp_path / "cementic-postgres"
        target.mkdir()
        (target / "irreplaceable.txt").write_text("keep me")

        result = runner.invoke(app, ["init", "postgres", str(target), "--force"])

        assert result.exit_code == 0
        assert (target / "irreplaceable.txt").read_text() == "keep me"
        assert (target / "compose.yml").is_file()

    def test_init_postgres_rejects_an_existing_file_target(self, tmp_path) -> None:
        target = tmp_path / "not-a-directory"
        target.write_text("i am a file")

        result = runner.invoke(app, ["init", "postgres", str(target)])

        assert result.exit_code == 1
        assert "is not a directory" in result.output
        assert target.read_text() == "i am a file"


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
            WorkerStatus(
                state="running",
                pid=("111"),
                process="running",
                current_file="None",
                watched_directories=["/docs"],
                processed_count=3,
                failed_count=1,
            ),
            WorkerStatus(
                state="running",
                pid=("222"),
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
        # The watcher's failed_count is headline information, not --verbose
        # detail: a skipped file never becomes a document, so no pipeline
        # percentage can ever reveal it.
        assert "skipped" in result.output
        assert "1 file(s) not indexed" in result.output

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
    @patch("cementic.cli.Bootstrapper")
    def test_embedding_start_starts_llama_runtime(self, mock_bootstrapper, mock_runtime_client):
        mock_bootstrapper.return_value.ensure_embedding_runtime.return_value = None
        client = MagicMock()
        client.embedding_dim = 768
        mock_runtime_client.return_value = client

        result = runner.invoke(app, ["embedding", "start"])

        assert result.exit_code == 0
        assert "embedding: running" in result.output
        mock_bootstrapper.return_value.ensure_embedding_runtime.assert_called_once()
        _, kwargs = mock_runtime_client.call_args
        assert kwargs["autostart"] is True

    @patch("cementic.cli.get_llama_cpp_runtime_client")
    @patch("cementic.cli.Bootstrapper")
    def test_embedding_start_fails_fast_when_model_missing(
        self, mock_bootstrapper, mock_runtime_client
    ):
        """A missing model must fail with a clear message, not a bare daemon timeout."""
        mock_bootstrapper.return_value.ensure_embedding_runtime.side_effect = RuntimeError(
            "llama.cpp model not found at /models/x.gguf"
        )

        result = runner.invoke(app, ["embedding", "start"])

        assert result.exit_code == 1
        assert "model not found" in result.output
        mock_runtime_client.assert_not_called()

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


class TestStopFallsBackToWorkerStateFiles:
    """`cementic stop` must still work when supervisor.json is gone.

    Regression: the supervisor record was the only source of PIDs, and it is
    deleted by every stop and overwritten by every start. Losing it made
    `cementic stop` report "no background cementic processes found" while both
    workers kept running, leaving `ps` and `kill` as the only recourse.
    """

    def _config_with_state(self, tmp_path, pid):
        config = Config()
        config.source_watcher.state_path = tmp_path / "sw.json"
        config.pipeline_worker.state_path = tmp_path / "pw.json"
        StateManager(config.pipeline_worker.state_path).update(pid=pid, start_token=None)
        return config

    def test_recovers_live_workers_when_supervisor_record_is_missing(self, tmp_path):
        config = self._config_with_state(tmp_path, pid=4242)

        with (
            patch("cementic.cli._get_config", return_value=config),
            patch("cementic.cli._load_supervisor_state", return_value={}),
            patch("cementic.cli._get_supervisor_state_path", return_value=tmp_path / "sup.json"),
            patch("cementic.cli.is_managed_process_alive", return_value=True),
            patch("cementic.cli.is_pid_running", return_value=False),
            patch("cementic.cli.os.kill") as mock_kill,
            patch("cementic.cli.wait_for_exit", return_value=[]),
        ):
            result = runner.invoke(app, ["stop"])

        assert result.exit_code == 0
        assert "supervisor record missing" in result.output
        mock_kill.assert_called_once_with(4242, 15)

    def test_reports_nothing_when_no_worker_is_alive(self, tmp_path):
        config = self._config_with_state(tmp_path, pid=4242)

        with (
            patch("cementic.cli._get_config", return_value=config),
            patch("cementic.cli._load_supervisor_state", return_value={}),
            patch("cementic.cli._get_supervisor_state_path", return_value=tmp_path / "sup.json"),
            patch("cementic.cli.is_managed_process_alive", return_value=False),
        ):
            result = runner.invoke(app, ["stop"])

        assert result.exit_code == 0
        assert "no background cementic processes found" in result.output

    def test_eperm_workers_are_not_cleared_as_stale(self, tmp_path):
        """Regression: PermissionError from os.kill was folded into "already
        exited", so stop printed "cleared stale state", exited 0, and deleted
        the state files of workers that kept indexing under another uid --
        the exact case is_pid_running was fixed to call alive."""
        config = self._config_with_state(tmp_path, pid=4242)

        with (
            patch("cementic.cli._get_config", return_value=config),
            patch("cementic.cli._load_supervisor_state", return_value={}),
            patch("cementic.cli._get_supervisor_state_path", return_value=tmp_path / "sup.json"),
            patch("cementic.cli.is_managed_process_alive", return_value=True),
            patch(
                "cementic.cli.os.kill",
                side_effect=PermissionError(1, "Operation not permitted"),
            ),
            patch("cementic.cli.wait_for_exit", return_value=[]),
        ):
            result = runner.invoke(app, ["stop"])

        assert result.exit_code == 1
        assert "permission denied" in result.output
        assert "cleared stale state" not in result.output
        # The worker's own state file must survive, or nothing can ever find
        # this pid again.
        assert config.pipeline_worker.state_path.exists()


class TestChunkCommand:
    """Test the stdin/stdout chunk filter."""

    def test_chunk_size_below_configured_overlap_reports_an_error(self):
        """Regression: this printed a full traceback.

        Passing only --chunk-size leaves --chunk-overlap at the configured
        default (128), so `cementic chunk --chunk-size 4` -- an entirely
        reasonable invocation -- tripped chunk_text's validation outside any
        handler.
        """
        result = runner.invoke(app, ["chunk", "--chunk-size", "4"], input="hello world")

        assert result.exit_code == 1
        assert "Traceback" not in result.output
        assert "chunk_overlap must be smaller than chunk_size" in result.output
        assert "chunk_size=4" in result.output
        assert "--chunk-overlap defaults to the config value" in result.output

    def test_explicit_consistent_flags_succeed(self):
        result = runner.invoke(
            app,
            ["chunk", "--chunk-size", "4", "--chunk-overlap", "1"],
            input="hello world this is a test",
        )

        assert result.exit_code == 0
        assert json.loads(result.output.splitlines()[0])["index"] == 0


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

    @patch("cementic.cli.Searcher")
    def test_search_json_reports_unknown_collection_on_stderr(self, mock_searcher_class):
        """An unindexed collection must not look like a clean no-match.

        Regression: --json returned before the unknown-collection check, so a
        typo'd collection produced empty stdout and exit 0 -- indistinguishable
        from a query that genuinely matched nothing.
        """
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = []
        mock_searcher.unsearchable_collections.return_value = ["typo"]
        mock_searcher_class.return_value = mock_searcher

        result = runner.invoke(app, ["search", "q", "-c", "typo", "--json"], catch_exceptions=False)

        assert result.exit_code == 1
        assert "no indexed revision for typo" in result.output

    @patch("cementic.cli.Searcher")
    def test_search_json_no_match_still_succeeds(self, mock_searcher_class):
        """A real no-match against an indexed collection stays exit 0."""
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = []
        mock_searcher.unsearchable_collections.return_value = []
        mock_searcher_class.return_value = mock_searcher

        result = runner.invoke(app, ["search", "q", "-c", "indexed", "--json"])

        assert result.exit_code == 0

    @pytest.mark.parametrize("extra_args", [[], ["--json"]])
    @patch("cementic.cli.Searcher")
    def test_unknown_collection_is_reported_even_when_others_match(
        self, mock_searcher_class, extra_args
    ):
        """A typo'd collection must not be hidden by a sibling that matched.

        The check was gated on an empty result set, so `-c work -c persnal`
        said nothing about the typo as long as `work` returned a hit: half the
        query was dropped invisibly, at exit 0, in both output modes.
        """
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = [
            {
                "collection": "work",
                "source_path": "/docs/a.md",
                "content": "a match",
                "score": 0.9,
                "distance": 0.1,
                "score_kind": "cosine_similarity",
            }
        ]
        mock_searcher.unsearchable_collections.return_value = ["persnal"]
        mock_searcher_class.return_value = mock_searcher

        result = runner.invoke(
            app, ["search", "q", "-c", "work", "-c", "persnal", *extra_args]
        )

        assert "no indexed revision for persnal" in result.output
        assert result.exit_code == 1

    @pytest.mark.parametrize(
        "argv",
        [
            ["status", "-c", "nosuch"],
            ["status", "-c", "nosuch", "--json"],
            ["collection", "revisions", "nosuch"],
            ["collection", "promote", "nosuch"],
            ["collection", "reindex", "nosuch"],
        ],
    )
    @patch("cementic.cli.get_session_factory")
    @patch("cementic.cli.get_engine")
    def test_unknown_collection_is_an_error_not_an_empty_result(
        self, mock_get_engine, mock_get_session_factory, argv
    ):
        """A typo must not read as a real collection that has nothing yet.

        These each reported success for a name cementic had never heard of: a
        zero-filled status report, an empty revision list, "no ready revision"
        -- all at exit 0, so no script could tell a typo from an idle
        collection.
        """
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = False
        mock_get_session_factory.return_value = lambda: mock_session

        with patch("cementic.cli.collection_exists", return_value=False):
            result = runner.invoke(app, argv)

        assert result.exit_code == 1
        assert "unknown collection" in result.output

    @patch("cementic.cli.get_session_factory")
    @patch("cementic.cli.get_engine")
    def test_status_json_unknown_collection_has_no_pipeline_block(
        self, mock_get_engine, mock_get_session_factory
    ):
        """Regression: `status -c typo --json` emitted a full zero-filled
        pipeline block at exit 0 -- the exact defect the human path had already
        fixed, alive in the mode scripts actually consume."""
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = False
        mock_get_session_factory.return_value = lambda: mock_session

        with patch("cementic.cli.collection_exists", return_value=False):
            result = runner.invoke(app, ["status", "-c", "nosuch", "--json"])

        assert result.exit_code == 1
        parsed = json.loads(result.output)
        assert "unknown collection" in parsed["error"]
        assert "pipeline" not in parsed

    @patch("cementic.cli.Searcher")
    def test_human_and_json_modes_agree_on_the_unknown_collection_exit_code(
        self, mock_searcher_class
    ):
        """Identical input must not change contract with the output format."""
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = []
        mock_searcher.unsearchable_collections.return_value = ["typo"]
        mock_searcher_class.return_value = mock_searcher

        human = runner.invoke(app, ["search", "q", "-c", "typo"])
        json_mode = runner.invoke(app, ["search", "q", "-c", "typo", "--json"])

        assert human.exit_code == json_mode.exit_code == 1

    @patch("cementic.cli.Searcher")
    def test_search_json_emits_one_json_object_per_line(self, mock_searcher_class):
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = [
            {
                "collection": "research",
                "source_path": "/a.pdf",
                "content": "a",
                "score": 0.9,
                "distance": 0.1,
                "score_kind": "cosine_similarity",
            },
            {
                "collection": "math",
                "source_path": "/b.pdf",
                "content": "b",
                "score": 0.8,
                "distance": 0.2,
                "score_kind": "cosine_similarity",
            },
        ]
        mock_searcher_class.return_value = mock_searcher

        result = runner.invoke(app, ["search", "test query", "--json"])

        assert result.exit_code == 0
        lines = [json.loads(line) for line in result.output.splitlines() if line.strip()]
        assert len(lines) == 2
        assert lines[0]["collection"] == "research"
        assert lines[1]["collection"] == "math"

    @patch("cementic.cli.Searcher")
    def test_search_json_no_results_prints_nothing(self, mock_searcher_class):
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = []
        mock_searcher_class.return_value = mock_searcher

        result = runner.invoke(app, ["search", "test query", "--json"])

        assert result.exit_code == 0
        assert result.output.strip() == ""

    @patch("cementic.cli.Searcher")
    def test_search_json_error_goes_to_stderr_not_stdout(self, mock_searcher_class):
        mock_searcher = MagicMock()
        mock_searcher.search.side_effect = RuntimeError("boom")
        mock_searcher_class.return_value = mock_searcher

        result = runner.invoke(app, ["search", "test query", "--json"])

        assert result.exit_code == 1
        assert result.stdout.strip() == ""
        assert "search failed" in result.stderr

    @patch("cementic.cli.Searcher")
    def test_search_top_k_long_form_aliases(self, mock_searcher_class):
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = []
        mock_searcher_class.return_value = mock_searcher

        runner.invoke(app, ["search", "test query", "--top-k", "5"])
        mock_searcher.search.assert_called_with("test query", top_k=5, collections=None)

        runner.invoke(app, ["search", "test query", "--limit", "7"])
        mock_searcher.search.assert_called_with("test query", top_k=7, collections=None)


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

    def test_status_doctor_still_reports_when_config_is_broken(self, tmp_path, monkeypatch):
        """Regression: --doctor died on the config error with no report at all
        -- the one command that exists to diagnose a broken setup produced
        *less* output than plain `status`. It now emits a failing report (valid
        JSON under --json) with the precise error on stderr."""
        bad = tmp_path / "cementic.toml"
        bad.write_text("this is := not toml", encoding="utf-8")
        monkeypatch.setenv("CEMENTIC_CONFIG", str(bad))
        monkeypatch.setattr(cementic_cli, "_config", None)

        result = runner.invoke(app, ["status", "--doctor", "--json"])

        assert result.exit_code == 1
        assert "config error" in result.stderr
        data = json.loads(result.stdout)
        assert data["ok"] is False
        assert data["checks"]["config"]["status"] == "fail"

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
            WorkerStatus(
                state="running",
                pid=("111"),
                process="running",
                current_file="None",
                watched_directories=["/docs"],
                processed_count=3,
                failed_count=1,
            ),
            WorkerStatus(
                state="running",
                pid=("222"),
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
                ready_revision_label=None,
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
                ready_revision_label=None,
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
            WorkerStatus(
                state="running",
                pid=("111"),
                process="running",
                current_file="None",
                watched_directories=["/docs"],
                processed_count=3,
                failed_count=1,
            ),
            WorkerStatus(
                state="running",
                pid=("222"),
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
            ready_revision_label=None,
            building_revision_label="rev-2",
        )

        result = runner.invoke(app, ["status", "--collection", "research"])

        assert result.exit_code == 0
        assert "collection" in result.output
        assert "research" in result.output
        assert "documents" in result.output
        assert "8/10 (80.0%)" in result.output
        assert "7/8 (87.5%)" in result.output
        # Only 7 of 10 documents are chunked, so the embedding denominator is
        # "chunks that exist so far" and the percentage is qualified.
        assert "20/30 (66.7% of chunks created so far)" in result.output
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
            WorkerStatus(
                state="stopped",
                pid=("N/A"),
                process="stopped",
                current_file="None",
                watched_directories=[],
                processed_count=0,
                failed_count=0,
            ),
            WorkerStatus(
                state="stopped",
                pid=("N/A"),
                process="stopped",
                current_file="None",
                watched_directories=[],
                processed_count=0,
                failed_count=0,
            ),
        )
        mock_build_supervisor_status.return_value = SimpleNamespace(
            state="0/2 running",
            collection="research",
            directories=["/docs"],
        )
        mock_check_health.return_value = SimpleNamespace(
            db_reachable=True,
            embedding_provider="llama-cpp",
            embedding_healthy=True,
            llama_daemon="stopped",
        )
        mock_load_pipeline_status.return_value = SimpleNamespace(
            documents=0,
            extracted_done=0,
            extracted_failed=0,
            chunked_done=0,
            chunked_failed=0,
            total_chunks=0,
            pending_embeddings=0,
            processing_embeddings=0,
            done_embeddings=0,
            failed_embeddings=0,
            extraction_pct=0.0,
            chunking_pct=0.0,
            embedding_pct=0.0,
            active_revision_label=None,
            ready_revision_label=None,
            building_revision_label=None,
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
            WorkerStatus(
                state="running",
                pid=("111"),
                process="running",
                current_file="None",
                watched_directories=["/docs"],
                processed_count=3,
                failed_count=1,
            ),
            WorkerStatus(
                state="running",
                pid=("222"),
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
            ready_revision_label=None,
            building_revision_label="None",
        )

        result = runner.invoke(app, ["status", "-c", "research"])

        assert result.exit_code == 0
        assert "research" in result.output
        mock_load_pipeline_status.assert_called_once_with(cementic_cli._get_config(), "research")

    @patch("cementic.cli._wait_for_worker_startup", return_value=[])
    @patch("cementic.cli.spawn_detached")
    def test_start_background(self, mock_spawn, mock_startup, temp_dir: Path):
        mock_spawn.side_effect = [1111, 2222]
        mock_path = temp_dir / "supervisor.json"

        with patch("cementic.cli._get_supervisor_state_path", return_value=mock_path):
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

    @patch("cementic.cli.spawn_detached")
    def test_concurrent_start_is_refused_rather_than_racing(self, mock_spawn, temp_dir: Path):
        """A second `cementic start` must not spawn while one is mid-flight.

        Regression: check-then-spawn-then-record was unsynchronised, so two
        concurrent runs could both see nothing running, both spawn, and the
        second's supervisor record replace the first's -- leaving the first pair
        running and invisible to `cementic stop`.
        """
        from cementic.filelock import file_lock

        lock_path = temp_dir / "start.lock"
        with (
            patch("cementic.cli._get_start_lock_path", return_value=lock_path),
            patch("cementic.cli.Bootstrapper"),
        ):
            with file_lock(lock_path, timeout=0):  # stand in for the other process
                result = runner.invoke(app, ["start", str(temp_dir), "--collection", "test"])

        assert result.exit_code == 1
        assert "already in progress" in result.output
        mock_spawn.assert_not_called()

    @patch("cementic.cli.spawn_detached")
    def test_start_stops_the_survivor_when_one_worker_dies(self, mock_spawn, temp_dir: Path):
        """A partial startup must not leave an unmanaged worker running.

        Regression: only the dead worker was reported. The survivor kept running
        and holding the collection's advisory lock, and because the supervisor
        state file had already been written, the next `cementic start` refused
        with "already running" -- contradicting the failure just printed.
        """
        mock_spawn.side_effect = [1111, 2222]
        state_path = temp_dir / "supervisor.json"
        dead = ManagedProcess("source-watcher", 1111, "/tmp/sw.log", None)

        with (
            patch("cementic.cli._get_supervisor_state_path", return_value=state_path),
            patch("cementic.cli.Bootstrapper"),
            patch("cementic.cli._wait_for_worker_startup", return_value=[dead]),
            patch("cementic.cli._terminate_managed") as mock_terminate,
        ):
            result = runner.invoke(app, ["start", str(temp_dir), "--collection", "test"])

        assert result.exit_code == 1
        assert "failed to start" in result.output
        # The pipeline worker survived, so it must be stopped and reported.
        mock_terminate.assert_called_once()
        survivors = mock_terminate.call_args.args[0]
        assert [proc.pid for proc in survivors] == [2222]
        assert "stopped pipeline-worker (PID 2222)" in result.output
        # ...and the stale record removed, so the next start is not refused.
        assert not state_path.exists()

    @patch("cementic.cli._wait_for_worker_startup", return_value=[])
    @patch("cementic.cli.spawn_detached")
    def test_start_passes_directories_after_end_of_options(
        self, mock_spawn, mock_startup, temp_dir: Path
    ):
        """Directories are absolute and separated by `--`.

        A directory whose name begins with "-" would otherwise be parsed as a
        flag by the runner, and a relative path means something different in the
        detached worker's working directory.
        """
        mock_spawn.side_effect = [1111, 2222]
        odd_dir = temp_dir / "-notes"
        odd_dir.mkdir()

        with (
            patch("cementic.cli._get_supervisor_state_path", return_value=temp_dir / "s.json"),
            patch("cementic.cli.Bootstrapper"),
        ):
            # `--` is needed at this level too, for the same reason.
            result = runner.invoke(app, ["start", "--collection", "test", "--", str(odd_dir)])

        assert result.exit_code == 0
        watcher_command = mock_spawn.call_args_list[0].args[0]
        assert watcher_command[-2] == "--"
        assert watcher_command[-1] == str(odd_dir.resolve())

    @patch("cementic.cli._wait_for_worker_startup", return_value=[])
    @patch("cementic.cli.spawn_detached")
    def test_start_background_mentions_default_collection(
        self, mock_spawn, mock_startup, temp_dir: Path
    ):
        mock_spawn.side_effect = [1111, 2222]
        mock_path = temp_dir / "supervisor.json"

        with patch("cementic.cli._get_supervisor_state_path", return_value=mock_path):
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
    def test_collection_promote_warns_about_leftover_artifacts_at_exit_zero(
        self, mock_get_engine, mock_get_session_factory
    ):
        """The promote is durable, so leftovers are a warning, not a failure --
        but they must be *printed*, as `collection remove` already does."""
        revision = SimpleNamespace(collection="research", status="active", label="rev-1")
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = False
        mock_get_session_factory.return_value = lambda: mock_session

        outcome = PromotionOutcome(
            status="promoted",
            revision=revision,
            unremoved_artifacts=["/artifacts/old.zst"],
            cleanup_error="DROP TABLE failed: disk error",
        )
        with patch("cementic.cli.promote_ready_revision", return_value=outcome):
            result = runner.invoke(app, ["collection", "promote", "research"])

        assert result.exit_code == 0
        assert "status: promoted" in result.output
        assert "cleanup failed" in result.output
        assert "could not be removed" in result.output
        assert "/artifacts/old.zst" in result.output

    @patch("cementic.cli.get_session_factory")
    @patch("cementic.cli.get_engine")
    def test_collection_reindex_reports_the_method_change(
        self, mock_get_engine, mock_get_session_factory
    ):
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = False
        mock_get_session_factory.return_value = lambda: mock_session

        outcome = ReindexOutcome("reindexed", method="diskann", previous_method="hnsw")
        with patch("cementic.cli.reindex_collection", return_value=outcome) as mock_reindex:
            result = runner.invoke(app, ["collection", "reindex", "research"])

        assert result.exit_code == 0
        assert "hnsw -> diskann" in result.output
        # The rebuild can run for minutes; saying so beforehand is the point.
        assert "several minutes" in result.output
        assert mock_reindex.call_args.kwargs["force"] is False

    @patch("cementic.cli.get_session_factory")
    @patch("cementic.cli.get_engine")
    def test_collection_reindex_without_an_active_revision_fails(
        self, mock_get_engine, mock_get_session_factory
    ):
        """Nothing promoted is a different answer from nothing to do."""
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = False
        mock_get_session_factory.return_value = lambda: mock_session

        with patch("cementic.cli.reindex_collection", return_value=ReindexOutcome("no_active")):
            result = runner.invoke(app, ["collection", "reindex", "research"])

        assert result.exit_code == 1
        assert "no active revision" in result.output

    @patch("cementic.cli.get_session_factory")
    @patch("cementic.cli.get_engine")
    def test_collection_reindex_says_how_to_rebuild_an_unchanged_method(
        self, mock_get_engine, mock_get_session_factory
    ):
        """hnsw_m and ef_construction only take effect on a forced rebuild, so an
        unchanged method must point at --force rather than just reporting nothing."""
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = False
        mock_get_session_factory.return_value = lambda: mock_session

        outcome = ReindexOutcome("reindexed", method="hnsw", previous_method="hnsw")
        with patch("cementic.cli.reindex_collection", return_value=outcome):
            result = runner.invoke(app, ["collection", "reindex", "research"])

        assert result.exit_code == 0
        assert "status: unchanged" in result.output
        assert "--force" in result.output

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

        outcome = PromotionOutcome(status="blocked_by_failures", revision=revision, counts=counts)
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
            ready_revision_label=None,
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

    @patch("cementic.cli.get_session_factory")
    @patch("cementic.cli.get_engine")
    def test_collection_remove_warns_when_workers_still_running(
        self, mock_get_engine, mock_get_session_factory
    ):
        """Regression: removing a collection under running workers succeeded in
        silence -- the watcher then re-registered the files and resurrected the
        collection, while the pipeline worker sat on the deleted revision."""
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = False
        mock_get_session_factory.return_value = lambda: mock_session
        deletion = SimpleNamespace(
            deleted_docs=1, deleted_chunks=2, artifact_paths=[], vector_profile_ids=[]
        )

        def invoke(supervisor_state):
            with (
                patch("cementic.cli.delete_collection_records", return_value=deletion),
                patch("cementic.cli.remove_artifacts", return_value=[]),
                patch("cementic.cli.drop_orphan_vector_tables"),
                patch(
                    "cementic.cli._load_supervisor_state", return_value=supervisor_state
                ),
                patch("cementic.cli._is_managed_proc_alive", return_value=True),
            ):
                return runner.invoke(app, ["collection", "remove", "mycol", "--force"])

        running = invoke(
            {"collection": "mycol", "processes": [{"name": "watcher", "pid": 1}]}
        )
        assert running.exit_code == 0
        assert "status: deleted" in running.output
        assert "still watching this collection" in running.output
        assert "cementic stop" in running.output

        idle = invoke({})
        assert idle.exit_code == 0
        assert "still watching" not in idle.output


class TestStatusCommand:
    """Test the status command."""

    @patch("cementic.status_service.get_engine")
    @patch("cementic.cli.get_session_factory")
    @patch("cementic.status_service.is_managed_process_alive", return_value=False)
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

    @patch("cementic.cli.is_pid_running", return_value=False)
    def test_llama_daemon_runtime_status_pid_not_running(self, mock_is_running, tmp_path: Path):
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
        result = runner.invoke(app, ["collection", "remove", "mycol"], input="n\n")
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
    def test_list_generic_error(self, mock_get_engine, mock_get_session_factory, mock_list):
        """list_collection_command prints generic error and exits 1."""
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_get_session_factory.return_value = lambda: mock_session
        result = runner.invoke(app, ["collection", "list"])
        assert result.exit_code == 1
        assert "collection list failed" in result.output

    @patch("cementic.cli.get_session_factory")
    @patch("cementic.cli.get_engine")
    def test_promote_no_ready_revision(self, mock_get_engine, mock_get_session_factory):
        """Nothing was promoted, so the exit code says so.

        Regression: this was the one promoted-nothing outcome that exited 0
        (empty, incomplete and blocked all exit 1), so a script chaining
        `promote && search` proceeded as if a revision had been published.
        """
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_get_session_factory.return_value = lambda: mock_session
        outcome = PromotionOutcome(status="no_ready")
        with patch("cementic.cli.promote_ready_revision", return_value=outcome):
            result = runner.invoke(app, ["collection", "promote", "mycol"])
        assert result.exit_code == 1
        assert "no ready revision" in result.output

    @patch("cementic.cli.promote_ready_revision", side_effect=ValueError("boom"))
    @patch("cementic.cli.get_session_factory")
    @patch("cementic.cli.get_engine")
    def test_promote_generic_error(self, mock_get_engine, mock_get_session_factory, mock_promote):
        """promote_collection prints generic error and exits 1."""
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_get_session_factory.return_value = lambda: mock_session
        result = runner.invoke(app, ["collection", "promote", "mycol"])
        assert result.exit_code == 1
        assert "collection promote failed" in result.output


class TestListCollectionRevisionsEdgeCases:
    """Edge cases for revisions command."""

    @patch("cementic.cli.get_session_factory")
    @patch("cementic.cli.get_engine")
    def test_revisions_empty(self, mock_get_engine, mock_get_session_factory):
        """list_collection_revision_command shows 'none' for empty revisions."""
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_get_session_factory.return_value = lambda: mock_session
        with patch("cementic.cli.list_collection_revisions", return_value=[]):
            result = runner.invoke(app, ["collection", "revisions", "mycol"])
        assert result.exit_code == 0
        assert "(none)" in result.output

    @patch(
        "cementic.cli.list_collection_revisions",
        side_effect=ValueError("boom"),
    )
    @patch("cementic.cli.get_session_factory")
    @patch("cementic.cli.get_engine")
    def test_revisions_generic_error(self, mock_get_engine, mock_get_session_factory, mock_list):
        """list_collection_revision_command prints generic error and exits 1."""
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_get_session_factory.return_value = lambda: mock_session
        result = runner.invoke(app, ["collection", "revisions", "mycol"])
        assert result.exit_code == 1
        assert "collection revisions failed" in result.output


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


class TestStatusSurfacesWorkerErrors:
    """A worker looping on a permanent failure must be visible in status.

    Regression: the pipeline worker's only failure channel was its log file, so
    a worker retrying forever looked exactly like a healthy idle one.
    """

    def _invoke(self, worker_error, args):
        with (
            patch("cementic.cli._llama_daemon_runtime_status", return_value="stopped"),
            patch("cementic.cli.check_health") as mock_health,
            patch("cementic.cli.load_worker_statuses") as mock_workers,
            patch("cementic.cli._load_supervisor_state") as mock_state,
            patch("cementic.cli.get_engine"),
            patch("cementic.cli.get_session_factory"),
            patch("cementic.cli.list_collections", return_value=[]),
            patch("cementic.cli.load_pipeline_status_bulk", return_value={}),
        ):
            mock_state.return_value = {"collection": "c", "directories": [], "processes": []}
            healthy = WorkerStatus(
                state="running", pid="1", process="running", current_file="None",
                watched_directories=[], processed_count=0, failed_count=0,
            )
            failing = WorkerStatus(
                state="running", pid="2", process="running", current_file="None",
                watched_directories=[], processed_count=0, failed_count=0,
                last_error=worker_error, last_error_at="2026-08-06T00:00:00+00:00",
            )
            mock_workers.return_value = (healthy, failing)
            mock_health.return_value = SimpleNamespace(
                db_reachable=True, embedding_provider="llama-cpp",
                embedding_healthy=True, llama_daemon="running",
            )
            return runner.invoke(app, args)

    def test_last_error_is_shown_without_verbose(self):
        result = self._invoke("ProgrammingError: relation does not exist", ["status"])
        assert "last error" in result.output
        assert "pipeline worker" in result.output
        assert "relation does not exist" in result.output

    def test_last_error_appears_in_json(self):
        result = self._invoke("ProgrammingError: boom", ["status", "--json"])
        payload = json.loads(result.output)
        assert payload["pipeline_worker"]["last_error"] == "ProgrammingError: boom"
        assert payload["source_watcher"]["last_error"] is None


class TestStatusExitCodes:
    """`cementic status` must exit non-zero when it could not report status."""

    @patch("cementic.cli._llama_daemon_runtime_status", return_value="stopped")
    @patch("cementic.cli.check_health")
    @patch("cementic.cli.load_worker_statuses")
    @patch("cementic.cli._load_supervisor_state")
    def test_status_exits_nonzero_when_database_unreachable(
        self, mock_state, mock_workers, mock_health, mock_daemon
    ):
        """Regression: this printed the hint and exited 0, so
        `cementic status && ...` succeeded against an unreachable database."""
        mock_state.return_value = {"collection": "c", "directories": [], "processes": []}
        worker = WorkerStatus(
            state="stopped",
            pid=("N/A"),
            process="stopped",
            current_file="None",
            watched_directories=[],
            processed_count=0,
            failed_count=0,
        )
        mock_workers.return_value = (worker, worker)
        mock_health.return_value = SimpleNamespace(
            db_reachable=False,
            embedding_provider="llama-cpp",
            embedding_healthy=False,
            llama_daemon="stopped",
        )

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 1
        assert "cementic init postgres" in result.output


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
            WorkerStatus(
                state="running",
                pid=str(1111),
                process="running",
                current_file="None",
                watched_directories=[],
                processed_count=0,
                failed_count=0,
            ),
            WorkerStatus(
                state="running",
                pid=str(2222),
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
            directories=[],
        )
        # check_health raises an exception
        with patch("cementic.cli.check_health", side_effect=RuntimeError("health check failed")):
            mock_session = MagicMock()
            mock_session.__enter__.return_value = mock_session
            mock_get_session_factory.return_value = lambda: mock_session

            result = runner.invoke(app, ["status"])
            assert result.exit_code == 0
            # No health section in the report itself...
            assert "health:" not in result.stdout
            # ...but the crash is explained on stderr rather than the whole
            # section silently vanishing (regression).
            assert "health: unavailable" in result.stderr

    @patch("cementic.cli.list_collections", return_value=[])
    def test_status_json_output(self, mock_list_collections):
        """Status with --json prints JSON output."""
        with patch("cementic.cli.load_worker_statuses") as mock_load:
            mock_load.return_value = (
                WorkerStatus(
                    state="running",
                    pid=str(1111),
                    process="running",
                    current_file="None",
                    watched_directories=[],
                    processed_count=0,
                    failed_count=0,
                ),
                WorkerStatus(
                    state="running",
                    pid=str(2222),
                    process="running",
                    current_file="None",
                    watched_directories=[],
                    processed_count=0,
                    failed_count=0,
                ),
            )
            with patch("cementic.cli._load_supervisor_state") as mock_state:
                mock_state.return_value = {
                    "collection": "research",
                    "directories": ["/docs"],
                    "processes": [],
                }
                with patch("cementic.cli.build_supervisor_status") as mock_build:
                    mock_build.return_value = SimpleNamespace(
                        state="2/2 running",
                        collection="research",
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
                                parsed = json.loads(result.stdout)
                                assert "supervisor" in parsed
                                assert "health" in parsed
                                assert "collections" in parsed
                                # Regression: --json omitted current_file, so
                                # the one field showing live progress existed
                                # only in the human output.
                                assert "current_file" in parsed["source_watcher"]
                                assert "current_file" in parsed["pipeline_worker"]

    @patch("cementic.cli.list_collections", return_value=[])
    def test_status_json_output_survives_piping_with_long_paths(self, mock_list_collections):
        """A long path must not be hard-wrapped by rich, which would break JSON parsing."""
        long_path = "/a" + "/very-long-directory-segment" * 10
        with patch("cementic.cli.load_worker_statuses") as mock_load:
            mock_load.return_value = (
                WorkerStatus(
                    state="running",
                    pid=str(1111),
                    process="running",
                    current_file="None",
                    watched_directories=[],
                    processed_count=0,
                    failed_count=0,
                ),
                WorkerStatus(
                    state="running",
                    pid=str(2222),
                    process="running",
                    current_file="None",
                    watched_directories=[],
                    processed_count=0,
                    failed_count=0,
                ),
            )
            with patch("cementic.cli._load_supervisor_state") as mock_state:
                mock_state.return_value = {
                    "collection": "research",
                    "directories": [long_path],
                    "processes": [],
                }
                with patch("cementic.cli.build_supervisor_status") as mock_build:
                    mock_build.return_value = SimpleNamespace(
                        state="2/2 running",
                        collection="research",
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
        assert "does not exist" in result.stderr

    def test_start_on_a_file_says_not_a_directory(self, tmp_path):
        """Regression: `cementic start some.pdf` said "Directory does not
        exist" about a path the user could plainly see existed."""
        file_path = tmp_path / "some.pdf"
        file_path.write_bytes(b"pdf")
        result = runner.invoke(app, ["start", str(file_path)])
        assert result.exit_code == 1
        assert "is not a directory" in result.stderr
        assert "does not exist" not in result.stderr

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
    @patch("cementic.cli.is_pid_running", return_value=False)
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


class TestErrorStreamDiscipline:
    """Errors go to stderr; stdout carries only the command's data.

    Regression (§4.2-4.3 of the fifth review): config and database errors
    printed to *stdout*, so `search q --json | jq` choked on "config error:
    ..." as if it were data -- and rich's off-TTY 80-column fallback
    hard-wrapped paths and hints mid-word in piped output.
    """

    def test_config_error_under_json_flag_keeps_stdout_clean(self, tmp_path, monkeypatch):
        bad = tmp_path / "cementic.toml"
        bad.write_text("this is := not toml", encoding="utf-8")
        monkeypatch.setenv("CEMENTIC_CONFIG", str(bad))
        # _get_config caches process-wide; an earlier test's good config would
        # mask the broken file this test plants.
        monkeypatch.setattr(cementic_cli, "_config", None)

        result = runner.invoke(app, ["search", "q", "--json"])

        assert result.exit_code == 1
        assert result.stdout.strip() == ""
        assert "config error" in result.stderr

    @patch("cementic.cli.get_session_factory")
    @patch("cementic.cli.get_engine")
    def test_unknown_collection_error_is_on_stderr(
        self, mock_get_engine, mock_get_session_factory
    ):
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = False
        mock_get_session_factory.return_value = lambda: mock_session

        with patch("cementic.cli.collection_exists", return_value=False):
            result = runner.invoke(app, ["status", "-c", "nosuch"])

        assert result.exit_code == 1
        assert "unknown collection" in result.stderr
        assert "unknown collection" not in result.stdout

    def test_long_paths_are_not_hard_wrapped_off_tty(self):
        long_dir = "/very-long" + "/segment-of-a-directory-path" * 5
        result = runner.invoke(app, ["start", long_dir])

        assert result.exit_code == 1
        # The whole path must survive contiguously; the 80-column fallback
        # used to split it mid-word.
        assert long_dir in result.stderr

    @patch("cementic.cli.get_engine")
    def test_database_hint_is_one_unwrapped_stderr_line(self, mock_get_engine):
        from sqlalchemy.exc import OperationalError

        from cementic.cli import _DB_HINT

        mock_get_engine.side_effect = OperationalError("statement", {}, Exception("down"))

        result = runner.invoke(app, ["collection", "list"])

        assert result.exit_code == 1
        # The full >100-char hint as one contiguous substring pins soft_wrap;
        # its presence on stderr pins the stream.
        assert _DB_HINT in result.stderr
        assert result.stdout.strip() == ""


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

    @pytest.mark.parametrize(
        "error",
        [
            IsADirectoryError("Is a directory"),
            PermissionError("Permission denied"),
            OSError(40, "Too many levels of symbolic links"),
        ],
        ids=["directory", "unreadable", "symlink-loop"],
    )
    def test_extract_reports_ordinary_os_errors_without_a_traceback(self, error):
        """Only FileNotFoundError was caught, so a directory named `notes.md`,
        an unreadable file, or a symlink loop produced a full traceback."""
        with patch("cementic.cli.extract_document", side_effect=error):
            result = runner.invoke(app, ["extract", "thing.md"])

        assert result.exit_code == 1
        assert "extract failed" in result.stderr
        assert "Traceback" not in result.output

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

    def test_chunk_size_and_overlap_flags_override_config(self):
        text = " ".join(f"word{i}" for i in range(50))
        default_result = runner.invoke(app, ["chunk"], input=text)
        flagged_result = runner.invoke(
            app, ["chunk", "--chunk-size", "5", "--chunk-overlap", "0"], input=text
        )
        assert default_result.exit_code == 0
        assert flagged_result.exit_code == 0
        default_lines = [line for line in default_result.output.splitlines() if line.strip()]
        flagged_lines = [line for line in flagged_result.output.splitlines() if line.strip()]
        assert len(flagged_lines) > len(default_lines)

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
    @patch("cementic.cli.create_provider", return_value=_FakeEmbedProvider())
    def test_embed_error_names_the_real_stdin_line(self, mock_create, mock_spec):
        """Regression: blank lines were dropped before numbering, so the error
        for line 4 was reported as line 2."""
        stdin = '{"content": "ok"}\n\n\n{"content": null}\n'
        result = runner.invoke(app, ["embed"], input=stdin)
        assert result.exit_code == 1
        assert "line 4" in result.stderr

    @patch("cementic.cli.runtime_spec_from_config", return_value=object())
    @patch("cementic.cli.create_provider", return_value=_FakeEmbedProvider())
    def test_embed_binary_stdin_is_an_error_not_a_traceback(self, mock_create, mock_spec):
        """Regression: only JSONDecodeError was caught, so binary stdin
        surfaced as a raw UnicodeDecodeError traceback."""
        result = runner.invoke(app, ["embed"], input=b"\x89PNG\r\n\x1a\n\xff\xfe")
        assert result.exit_code == 1
        assert "Traceback" not in result.output
        assert "not text" in result.stderr

    def test_chunk_empty_path_argument_is_an_error_not_stdin(self, tmp_path):
        """Regression: `chunk ""` fell through the truthiness check to stdin
        and sat there looking hung."""
        result = runner.invoke(app, ["chunk", ""], input="should not be read")
        assert result.exit_code == 1
        assert "chunk failed" in result.stderr
        assert result.stdout.strip() == ""

    def test_extract_directory_is_a_clear_error(self, tmp_path):
        """Regression: a directory said "no extractor for '(none)'"."""
        result = runner.invoke(app, ["extract", str(tmp_path)])
        assert result.exit_code == 1
        assert "is a directory" in result.stderr

    @pytest.mark.parametrize(
        "record",
        ['{"content": ""}', '{"content": "   \\n\\t  "}'],
        ids=["empty", "whitespace-only"],
    )
    @patch("cementic.cli.runtime_spec_from_config", return_value=object())
    @patch("cementic.cli.create_provider", return_value=_FakeEmbedProvider())
    def test_embed_rejects_content_with_no_text(self, mock_create, mock_spec, record):
        """Regression: {"content": ""} embedded into a plausible-looking vector
        at exit 0 -- the same hole the missing/null fix closed, one layer down.
        The pipeline worker never embeds whitespace-only chunks either."""
        result = runner.invoke(app, ["embed"], input=record + "\n")
        assert result.exit_code == 1
        assert result.stdout.strip() == ""
        assert "no text to embed" in result.stderr

    @patch("cementic.cli.runtime_spec_from_config", return_value=object())
    def test_embed_batches_requests_by_config_batch_size(self, mock_spec):
        """Regression: embed must not send every stdin record in one request.

        A single unbounded batch risks timing out the embedding backend on
        large inputs; requests must be chunked by pipeline_worker.batch_size.
        """
        provider = _BatchTrackingEmbedProvider()
        stdin = "".join(json.dumps({"index": i, "content": f"chunk {i}"}) + "\n" for i in range(5))
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


class TestStopExplainsAnIndexBuild:
    """A stop that times out mid-index-build must say why.

    The worker cannot answer SIGTERM from inside CREATE INDEX, so the grace
    period always expires and the message read as a hung process. It is not
    hung, and the difference matters: the statement is not resumable, so
    --force throws away however many minutes it had accumulated.
    """

    @patch("cementic.cli._pipeline_worker_activity", return_value="building hnsw index")
    @patch("cementic.cli.wait_for_exit", return_value=[4242])
    @patch("cementic.cli._is_managed_proc_alive", return_value=True)
    @patch("cementic.cli.os.kill")
    @patch("cementic.cli._save_supervisor_state")
    @patch(
        "cementic.cli._load_supervisor_state",
        return_value={"processes": [{"name": "pipeline-worker", "pid": 4242}]},
    )
    def test_timeout_names_the_build_and_its_cost(
        self, mock_state, mock_save, mock_kill, mock_alive, mock_wait, mock_activity
    ):
        result = runner.invoke(app, ["stop"])

        assert result.exit_code == 1
        assert "building hnsw index" in result.output
        assert "discard" in result.output

    @patch("cementic.cli._pipeline_worker_activity", return_value=None)
    @patch("cementic.cli.wait_for_exit", return_value=[4242])
    @patch("cementic.cli._is_managed_proc_alive", return_value=True)
    @patch("cementic.cli.os.kill")
    @patch("cementic.cli._save_supervisor_state")
    @patch(
        "cementic.cli._load_supervisor_state",
        return_value={"processes": [{"name": "pipeline-worker", "pid": 4242}]},
    )
    def test_ordinary_timeout_keeps_the_plain_advice(
        self, mock_state, mock_save, mock_kill, mock_alive, mock_wait, mock_activity
    ):
        result = runner.invoke(app, ["stop"])

        assert result.exit_code == 1
        assert "use --force" in result.output
