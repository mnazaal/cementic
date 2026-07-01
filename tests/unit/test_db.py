"""Tests for database helpers."""

from unittest.mock import MagicMock, patch

import pytest

from cementic.db import (
    REQUIRED_DB_EXTENSIONS,
    _ensure_ann_access_method,
    create_tables,
    ensure_embedding_ann_index,
    ensure_vector_extensions,
    get_engine,
)
from cementic.index_strategies import IndexParams


@patch("cementic.db.create_engine")
def test_get_engine_is_pure(mock_create_engine):
    """get_engine should only construct engine without DB side effects."""
    mock_engine = MagicMock()
    mock_create_engine.return_value = mock_engine

    engine = get_engine("postgresql://user:pass@localhost:5432/cementic")

    assert engine is mock_engine
    mock_create_engine.assert_called_once()
    assert not mock_engine.connect.called


@patch("cementic.db.Base.metadata.create_all")
@patch("cementic.db.ensure_vector_extensions")
def test_create_tables_calls_extension_setup(mock_ensure_extensions, mock_create_all):
    """create_tables should ensure extensions before creating tables."""
    mock_engine = MagicMock()

    create_tables(mock_engine)

    mock_ensure_extensions.assert_called_once_with(mock_engine)
    mock_create_all.assert_called_once_with(mock_engine)


def _make_exec_result(value: object) -> MagicMock:
    """Return an execute result whose .scalar() returns *value*."""
    result = MagicMock()
    result.scalar.return_value = value
    return result


def _make_engine_conn_mock(execute_side_effect: list) -> MagicMock:
    """Return an Engine mock whose connect().__enter__() yields a conn with given
    execute side-effects."""
    mock_conn = MagicMock()
    mock_conn.execute.side_effect = execute_side_effect
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_engine.connect.return_value.__exit__.return_value = False
    return mock_engine


class TestEnsureVectorExtensions:
    """Behavior tests for ensure_vector_extensions branch coverage."""

    def test_required_extensions_are_centralized(self) -> None:
        assert REQUIRED_DB_EXTENSIONS == ("vector", "vectorscale")

    def test_postgres_creates_required_extensions(self) -> None:
        """Postgres setup should create both required extensions before tables."""
        engine = _make_engine_conn_mock([None, None])
        engine.dialect.name = "postgresql"

        ensure_vector_extensions(engine)

        assert engine.connect.return_value.__enter__.return_value.execute.call_count == 2
        execute_calls = engine.connect.return_value.__enter__.return_value.execute.call_args_list
        first = str(execute_calls[0][0][0])
        second = str(execute_calls[1][0][0])
        assert "CREATE EXTENSION IF NOT EXISTS vector" in first
        assert "CREATE EXTENSION IF NOT EXISTS vectorscale" in second
        engine.connect.return_value.__enter__.return_value.commit.assert_called_once()

    def test_non_postgres_skips_extension_setup(self) -> None:
        engine = _make_engine_conn_mock([])
        engine.dialect.name = "sqlite"

        ensure_vector_extensions(engine)

        engine.connect.assert_not_called()

    def test_extension_create_failure_is_actionable(self) -> None:
        engine = _make_engine_conn_mock([RuntimeError("permission denied")])
        engine.dialect.name = "postgresql"

        with pytest.raises(Exception):
            ensure_vector_extensions(engine)
        engine.connect.return_value.__enter__.return_value.rollback.assert_called_once()


def test_ensure_embedding_ann_index_builds_hnsw_on_profile_table() -> None:
    mock_conn = MagicMock()
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_engine.connect.return_value.__exit__.return_value = False

    ensure_embedding_ann_index(mock_engine, profile_id=7, method="hnsw", params=IndexParams())

    statement = str(mock_conn.execute.call_args[0][0])
    assert "ON embedding_vectors_p7" in statement
    assert "USING hnsw (embedding vector_cosine_ops)" in statement
    mock_conn.commit.assert_called_once()


def test_ensure_embedding_ann_index_builds_diskann() -> None:
    mock_conn = MagicMock()
    mock_conn.execute.return_value.scalar.return_value = True
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_engine.connect.return_value.__exit__.return_value = False

    ensure_embedding_ann_index(mock_engine, profile_id=3, method="diskann", params=IndexParams())

    statement = str(mock_conn.execute.call_args_list[-1][0][0])
    assert "ON embedding_vectors_p3" in statement
    assert "USING diskann (embedding vector_cosine_ops)" in statement


def test_ensure_ann_access_method_rejects_missing_diskann() -> None:
    mock_conn = MagicMock()
    mock_conn.execute.return_value.scalar.return_value = False

    with pytest.raises(RuntimeError, match="vectorscale"):
        _ensure_ann_access_method(mock_conn, "diskann")


def test_ensure_ann_access_method_skips_hnsw_preflight() -> None:
    mock_conn = MagicMock()

    _ensure_ann_access_method(mock_conn, "hnsw")

    mock_conn.execute.assert_not_called()


def test_ensure_embedding_ann_index_rejects_unsupported_metric() -> None:
    """Unsupported distance metric raises ValueError before any DB call."""
    mock_engine = MagicMock()
    with pytest.raises(ValueError, match="Unsupported distance metric"):
        ensure_embedding_ann_index(
            mock_engine,
            profile_id=1,
            method="hnsw",
            params=IndexParams(),
            distance_metric="euclidean",
        )
    mock_engine.connect.assert_not_called()
