"""Tests for database helpers."""

from unittest.mock import MagicMock, patch

from seman.db import create_tables, get_engine


@patch("seman.db.create_engine")
def test_get_engine_is_pure(mock_create_engine):
    """get_engine should only construct engine without DB side effects."""
    mock_engine = MagicMock()
    mock_create_engine.return_value = mock_engine

    engine = get_engine("postgresql://user:pass@localhost:5432/seman")

    assert engine is mock_engine
    mock_create_engine.assert_called_once()
    assert not mock_engine.connect.called


@patch("seman.db.Base.metadata.create_all")
@patch("seman.db.ensure_vector_extensions")
def test_create_tables_calls_extension_setup(mock_ensure_extensions, mock_create_all):
    """create_tables should ensure extensions before creating tables."""
    mock_engine = MagicMock()

    create_tables(mock_engine)

    mock_ensure_extensions.assert_called_once_with(mock_engine)
    mock_create_all.assert_called_once_with(mock_engine)
