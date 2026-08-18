"""Runtime-budget tests for CLI command dispatch."""

from __future__ import annotations

import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from cementic.cli import app
from cementic.status_service import WorkerStatus

runner = CliRunner()


def _assert_fast(args: list[str], budget_seconds: float) -> str:
    start = time.perf_counter()
    result = runner.invoke(app, args)
    elapsed = time.perf_counter() - start
    assert result.exit_code == 0, result.output
    assert elapsed < budget_seconds, f"{args} took {elapsed:.3f}s > {budget_seconds:.3f}s"
    return result.output


def test_cli_import_does_not_pull_in_the_pdf_stack() -> None:
    """Importing the CLI must not load the PDF layout model.

    `cli` imports `extract`, which used to import `pymupdf.layout` at module
    scope -- an ONNX layout analyser plus networkx, about 0.5s warm and over a
    second cold, paid by `cementic --version` and every other command. The other
    budgets in this file cannot catch it: they time dispatch, by which point the
    test module has already imported the CLI. Asserted structurally rather than
    by clock, so it does not depend on machine load.
    """
    probe = (
        "import sys; import cementic.cli; "
        "print(','.join(m for m in ('pymupdf.layout', 'pymupdf4llm', 'networkx') "
        "if m in sys.modules))"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout.strip()

    assert loaded == "", f"cementic.cli eagerly imported: {loaded}"


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
@patch(
    "cementic.cli.check_health",
    # A real health object, not None: None now means the probe failed, which
    # exits 1 rather than printing a summary two rows short.
    return_value=SimpleNamespace(
        db_reachable=True,
        embedding_provider="llama-cpp",
        embedding_healthy=True,
        llama_daemon="running",
    ),
)
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
        WorkerStatus(
            state="running",
            pid=str(1),
            process="running",
            current_file=None,
            watched_directories=[],
            processed_count=0,
            failed_count=0,
        ),
        WorkerStatus(
            state="running",
            pid=str(2),
            process="running",
            current_file=None,
            watched_directories=[],
            processed_count=0,
            failed_count=0,
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
