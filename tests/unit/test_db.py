"""Tests for database helpers."""

from unittest.mock import MagicMock, patch

from cementic.db import create_tables, embedding_index_name, ensure_embedding_ann_index, get_engine


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


def test_embedding_index_name_is_deterministic() -> None:
    assert embedding_index_name(42) == "ix_chunk_embeddings_ann_profile_42"


def test_ensure_embedding_ann_index_uses_partial_expression_index() -> None:
    mock_conn = MagicMock()
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_engine.connect.return_value.__exit__.return_value = False

    ensure_embedding_ann_index(mock_engine, profile_id=7, embedding_dim=1536)

    statement = str(mock_conn.execute.call_args[0][0])
    assert "CREATE INDEX IF NOT EXISTS ix_chunk_embeddings_ann_profile_7" in statement
    assert "embedding::vector(1536)" in statement
    assert "vector_cosine_ops" in statement
    assert "WHERE embedding_profile_id = 7" in statement
    mock_conn.commit.assert_called_once()
