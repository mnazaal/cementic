"""PostgreSQL-backed CLI integration tests."""

from __future__ import annotations

import time

import pytest
from typer.testing import CliRunner

from cementic import cli as cementic_cli
from cementic.cli import app
from tests.integration.test_pg_helpers import cleanup_pg_tables, seed_active_vector_collection

runner = CliRunner()


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

    start = time.perf_counter()
    result = runner.invoke(app, ["search", "neural", "-n", "1", "-c", "cli-cs"])
    elapsed = time.perf_counter() - start

    assert result.exit_code == 0, result.output
    assert elapsed < 2.0
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
    monkeypatch.setattr("cementic.cli.check_health", lambda config: None)
    monkeypatch.setattr(
        "cementic.cli.load_worker_statuses",
        lambda config: (
            type("Status", (), {
                "state": "stopped", "pid": None, "process": "stopped",
                "current_file": None, "watched_directories": [],
                "processed_count": 0, "failed_count": 0,
            })(),
            type("Status", (), {
                "state": "stopped", "pid": None, "process": "stopped",
                "current_file": None, "watched_directories": [],
                "processed_count": 0, "failed_count": 0,
            })(),
        ),
    )
    monkeypatch.setattr("cementic.cli._load_supervisor_state", lambda: {"processes": []})

    start = time.perf_counter()
    result = runner.invoke(app, ["status", "--verbose", "-c", "cli-status"])
    elapsed = time.perf_counter() - start

    assert result.exit_code == 0, result.output
    assert elapsed < 2.0
    assert "cli-status" in result.output
    assert "files:" in result.output
    assert "/docs/status.pdf" in result.output
    cleanup_pg_tables(pg_session)
