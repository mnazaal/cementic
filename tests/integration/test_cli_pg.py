"""PostgreSQL-backed CLI integration tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import event
from sqlalchemy.engine import Engine
from typer.testing import CliRunner

from cementic import cli as cementic_cli
from cementic.cli import app
from cementic.status_service import WorkerStatus
from tests.integration.test_pg_helpers import cleanup_pg_tables, seed_active_vector_collection

runner = CliRunner()


class _QueryCounter:
    """Counts SELECT statements issued across all engines while registered.

    Stands in for a wall-clock budget: a per-row (rather than per-collection)
    query pattern -- the actual hazard a "this must stay fast" test guards
    against -- shows up as an unbounded query count long before it shows up as
    an unbounded clock, and unlike a clock this is immune to machine load.
    """

    def __init__(self) -> None:
        self.select_count = 0

    def __enter__(self) -> "_QueryCounter":
        event.listen(Engine, "before_cursor_execute", self._on_execute)
        return self

    def __exit__(self, *exc_info: object) -> None:
        event.remove(Engine, "before_cursor_execute", self._on_execute)

    def _on_execute(self, conn, cursor, statement, *args, **kwargs) -> None:
        if statement.lstrip().upper().startswith("SELECT"):
            self.select_count += 1


class FakeSearchEmbeddingClient:
    """Deterministic query embedding client for CLI PG tests."""

    def health_check(self) -> bool:
        return True

    def format_query(self, text: str) -> str:
        return text

    def format_document(self, text: str) -> str:
        return text

    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]


@pytest.mark.pg
def test_pg_cli_search_returns_real_results(
    pg_session, pg_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    cleanup_pg_tables(pg_session)
    seed_active_vector_collection(
        pg_session,
        collection="cli-cs",
        source_path="/docs/cli-cs.pdf",
        chunks=[("cli neural vector result", [1.0, 0.0, 0.0, 0.0])],
    )
    seed_active_vector_collection(
        pg_session,
        collection="cli-bio",
        source_path="/docs/cli-bio.pdf",
        chunks=[("cli biology result", [0.0, 1.0, 0.0, 0.0])],
    )
    pg_session.commit()
    monkeypatch.setattr(cementic_cli, "_config", pg_config)
    monkeypatch.setattr(
        "cementic.search._create_embedding_provider",
        lambda config_json, config=None: FakeSearchEmbeddingClient(),
    )

    with _QueryCounter() as counter:
        result = runner.invoke(app, ["search", "neural", "-n", "1", "-c", "cli-cs"])

    assert result.exit_code == 0, result.output
    # Bounded well above the handful of per-collection queries a single-collection
    # search actually issues: a per-row query pattern would blow straight through
    # this even with the two rows seeded here.
    assert counter.select_count < 30, (
        f"search issued {counter.select_count} SELECTs -- looks like a per-row query pattern"
    )
    assert "cli neural vector result" in result.output
    assert "cli biology result" not in result.output
    cleanup_pg_tables(pg_session)


@pytest.mark.pg
def test_pg_cli_status_verbose_reports_real_collection(
    pg_session, pg_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    cleanup_pg_tables(pg_session)
    seed_active_vector_collection(
        pg_session,
        collection="cli-status",
        source_path="/docs/status.pdf",
        chunks=[("status vector result", [1.0, 0.0, 0.0, 0.0])],
    )
    pg_session.commit()
    monkeypatch.setattr(cementic_cli, "_config", pg_config)
    # A real health object, not None: None now means the probe *failed*, which
    # exits 1 rather than printing a summary two rows short.
    monkeypatch.setattr(
        "cementic.cli.check_health",
        lambda config: SimpleNamespace(
            db_reachable=True,
            embedding_provider="llama-cpp",
            embedding_healthy=True,
            llama_daemon="running",
        ),
    )
    def _stopped_worker() -> WorkerStatus:
        return WorkerStatus(
            state="stopped",
            pid="N/A",
            process="stopped",
            current_file=None,
            watched_directories=[],
            processed_count=0,
            failed_count=0,
        )

    monkeypatch.setattr(
        "cementic.cli.load_worker_statuses",
        lambda config: (_stopped_worker(), _stopped_worker()),
    )
    monkeypatch.setattr("cementic.cli._load_supervisor_state", lambda: {"processes": []})

    with _QueryCounter() as counter:
        result = runner.invoke(app, ["status", "--verbose", "-c", "cli-status"])

    assert result.exit_code == 0, result.output
    # Same bound and rationale as the search test above.
    assert counter.select_count < 30, (
        f"status --verbose issued {counter.select_count} SELECTs -- "
        "looks like a per-row query pattern"
    )
    assert "cli-status" in result.output
    assert "files:" in result.output
    assert "/docs/status.pdf" in result.output
    cleanup_pg_tables(pg_session)
