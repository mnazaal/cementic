"""Structural tests standing in for the CLI's former wall-clock budgets.

The wall-clock assertions these replaced (`elapsed < budget_seconds`) stood in
for two distinct properties, and each conversion below targets the one that
applied:

- Help-only/eager-exit paths (namespace help, root `--help`, `--version`) must
  never reach `_get_config()` -- every slow external call in `cli.py` (DB
  engine, health probe, embedding provider) is downstream of it, so "never
  called" is a direct, machine-load-independent proxy for "never touches a
  slow external service" (see the docstrings this file used to carry).
- Commands whose backend is already fully mocked (search/status/collection
  list dispatch, `stop`) cannot be made slow by the mocked collaborator
  itself; what a hidden bug could still do is call it more than once (a
  retry loop, a duplicated probe) -- which is exactly the pattern that would
  turn slow once real. Each of those tests now asserts the mock's call count
  instead of timing dispatch.
"""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from cementic.cli import app
from cementic.status_service import WorkerStatus

runner = CliRunner()


def _invoke_ok(args: list[str]) -> str:
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
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


@patch("cementic.cli._get_config", side_effect=AssertionError("_get_config must not be called"))
def test_namespace_help_commands_never_load_config(mock_get_config: MagicMock) -> None:
    """Help-only commands should never touch slow external services.

    Every slow call in `cli.py` (DB engine, health probe, embedding provider)
    is reached through `_get_config()`, so asserting it was never called is a
    direct proxy for "never touches a slow external service".
    """
    for args in (["collection"], ["start"], ["search"]):
        result = runner.invoke(app, list(args))
        assert result.exit_code in {0, 2}, result.output
    assert mock_get_config.call_count == 0


@patch("cementic.cli._get_config", side_effect=AssertionError("_get_config must not be called"))
def test_root_help_never_loads_config(mock_get_config: MagicMock) -> None:
    """Root help must not touch slow external services."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "USAGE:" in result.output
    assert mock_get_config.call_count == 0


@patch("cementic.cli._get_config", side_effect=AssertionError("_get_config must not be called"))
@patch("cementic.cli._get_cli_version", return_value="0.1.0")
def test_version_never_loads_config(mock_version: MagicMock, mock_get_config: MagicMock) -> None:
    """`--version` is an eager exit and must not touch slow external services."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0, result.output
    assert "0.1.0" in result.output
    assert mock_get_config.call_count == 0


@patch("cementic.cli.Searcher")
def test_search_command_calls_backend_search_exactly_once(mock_searcher_class: MagicMock) -> None:
    """Search dispatch must not loop/retry against the backend."""
    mock_searcher = MagicMock()
    mock_searcher.search.return_value = [
        {"score": 0.99, "source_path": "/tmp/doc.pdf", "content": "fast result"}
    ]
    mock_searcher_class.return_value = mock_searcher

    output = _invoke_ok(["search", "fast query", "-n", "1"])

    assert "fast result" in output
    assert mock_searcher.search.call_count == 1


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
def test_status_command_probes_health_and_workers_exactly_once(
    mock_build: MagicMock,
    mock_state: MagicMock,
    mock_workers: MagicMock,
    mock_health: MagicMock,
    mock_engine: MagicMock,
    mock_session_factory: MagicMock,
    mock_collections: MagicMock,
    mock_daemon: MagicMock,
) -> None:
    """Status dispatch must not re-probe health or re-load worker state.

    A repeated probe is exactly the pattern that is cheap against a mock and
    slow (or, per the comment above, potentially minutes slow) against the
    real health check.
    """
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

    output = _invoke_ok(["status"])

    assert "collections" in output
    assert mock_health.call_count == 1
    assert mock_workers.call_count == 1


@patch("cementic.cli.list_collections", return_value=[])
@patch("cementic.cli.get_session_factory")
@patch("cementic.cli.get_engine")
def test_collection_list_queries_backend_exactly_once(
    mock_engine: MagicMock, mock_session_factory: MagicMock, mock_collections: MagicMock
) -> None:
    """Collection list dispatch must not loop/retry against the DB."""
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_session_factory.return_value = lambda: mock_session

    output = _invoke_ok(["collection", "list"])

    assert "(none)" in output
    assert mock_collections.call_count == 1


@patch("cementic.cli._load_supervisor_state", return_value={"processes": []})
def test_stop_does_not_wait_when_no_processes_exist(mock_state: MagicMock) -> None:
    """Stop should not wait (poll/sleep) when no processes exist."""
    output = _invoke_ok(["stop"])

    assert "no background cementic processes found" in output
    assert mock_state.call_count == 1
