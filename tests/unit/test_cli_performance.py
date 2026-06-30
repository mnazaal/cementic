"""Runtime-budget tests for CLI command dispatch."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from cementic.cli import app

runner = CliRunner()


def _assert_fast(args: list[str], budget_seconds: float) -> str:
    start = time.perf_counter()
    result = runner.invoke(app, args)
    elapsed = time.perf_counter() - start
    assert result.exit_code == 0, result.output
    assert elapsed < budget_seconds, f"{args} took {elapsed:.3f}s > {budget_seconds:.3f}s"
    return result.output


def test_namespace_help_commands_stay_fast() -> None:
    """Help-only commands should never touch slow external services."""
    for args in (["collection"], ["start"], ["search"]):
        start = time.perf_counter()
        result = runner.invoke(app, list(args))
        elapsed = time.perf_counter() - start
        assert result.exit_code in {0, 2}, result.output
        assert elapsed < 0.20, f"{args} took {elapsed:.3f}s > 0.200s"


def test_root_help_stays_fast() -> None:
    """Root help must not touch slow external services."""
    start = time.perf_counter()
    result = runner.invoke(app, ["--help"])
    elapsed = time.perf_counter() - start
    assert result.exit_code == 0, result.output
    assert "USAGE:" in result.output
    assert elapsed < 0.20, f"--help took {elapsed:.3f}s"


@patch("cementic.cli._get_cli_version", return_value="0.1.0")
def test_version_stays_fast(mock_version: MagicMock) -> None:
    """Version dispatch stays below budget."""
    start = time.perf_counter()
    result = runner.invoke(app, ["--version"])
    elapsed = time.perf_counter() - start
    assert result.exit_code == 0, result.output
    assert "0.1.0" in result.output
    assert elapsed < 0.20, f"--version took {elapsed:.3f}s"


@patch("cementic.cli.Searcher")
def test_search_command_stays_fast_with_mocked_backend(mock_searcher_class: MagicMock) -> None:
    """Search CLI dispatch stays fast when backend is mocked."""
    mock_searcher = MagicMock()
    mock_searcher.search.return_value = [
        {"score": 0.99, "source_path": "/tmp/doc.pdf", "content": "fast result"}
    ]
    mock_searcher_class.return_value = mock_searcher

    output = _assert_fast(["search", "fast query", "-n", "1"], 0.50)

    assert "fast result" in output


@patch("cementic.cli._llama_daemon_runtime_status", return_value="stopped")
@patch("cementic.cli.list_collections", return_value=[])
@patch("cementic.cli.get_session_factory")
@patch("cementic.cli.get_engine")
@patch("cementic.cli.check_health", return_value=None)
@patch("cementic.cli.load_worker_statuses")
@patch("cementic.cli._load_supervisor_state", return_value={"processes": [], "directories": []})
@patch("cementic.cli.build_supervisor_status")
def test_status_command_stays_fast_with_mocked_backend(
    mock_build: MagicMock,
    mock_state: MagicMock,
    mock_workers: MagicMock,
    mock_health: MagicMock,
    mock_engine: MagicMock,
    mock_session_factory: MagicMock,
    mock_collections: MagicMock,
    mock_daemon: MagicMock,
) -> None:
    """Status CLI dispatch stays fast when DB and health checks are mocked."""
    mock_workers.return_value = (
        SimpleNamespace(
            state="running", pid=1, process="running", current_file="None",
            watched_directories=[], processed_count=0, failed_count=0,
        ),
        SimpleNamespace(
            state="running", pid=2, process="running", current_file="None",
            watched_directories=[], processed_count=0, failed_count=0,
        ),
    )
    mock_build.return_value = SimpleNamespace(
        state="2/2 running", collection="default", directories=[]
    )
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_session_factory.return_value = lambda: mock_session

    output = _assert_fast(["status"], 0.50)

    assert "collections" in output


@patch("cementic.cli.list_collections", return_value=[])
@patch("cementic.cli.get_session_factory")
@patch("cementic.cli.get_engine")
def test_collection_list_stays_fast_with_mocked_backend(
    mock_engine: MagicMock, mock_session_factory: MagicMock, mock_collections: MagicMock
) -> None:
    """Collection list dispatch stays fast when DB is mocked."""
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_session_factory.return_value = lambda: mock_session

    output = _assert_fast(["collection", "list"], 0.50)

    assert "(none)" in output


@patch("cementic.cli._load_supervisor_state", return_value={"processes": []})
def test_stop_stays_fast(mock_state: MagicMock) -> None:
    """Stop should not wait when no processes exist."""
    output = _assert_fast(["stop"], 0.75)

    assert "no background cementic processes found" in output
