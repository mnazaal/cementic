"""Integration tests for CLI collection commands backed by SQLite."""

from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typer.testing import CliRunner

from cementic.cli import app
from cementic.config import get_config
from cementic.db import Base, PipelineRevision, SourceDocument
from cementic.profiles import (
    get_or_create_chunk_profile,
    get_or_create_embedding_profile,
    get_or_create_extractor_profile,
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def sqlite_engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def sqlite_session(sqlite_engine):
    session_maker = sessionmaker(bind=sqlite_engine)
    session = session_maker()
    yield session
    session.close()


@pytest.fixture(autouse=True)
def _skip_ann_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip ANN index creation — SQLite doesn't support pgvector HNSW."""
    monkeypatch.setattr(
        "cementic.pipeline_worker.ensure_revision_ann_index",
        lambda session, revision: None,
    )


class TestRemoveCollectionCommand:
    """Tests for the remove-collection CLI command."""

    def test_remove_nonexistent_collection(self, runner, sqlite_engine):
        with patch("cementic.cli.get_engine", return_value=sqlite_engine):
            result = runner.invoke(app, ["collection", "remove", "ghost", "--force"])
        assert result.exit_code == 0
        assert "not found" in result.stdout.lower()

    def test_remove_existing_collection(self, runner, sqlite_engine, sqlite_session):
        doc = SourceDocument(source_path="/tmp/test.pdf", collection="delme")
        doc.file_hash = "abc"
        doc.status = "pending"
        sqlite_session.add(doc)
        sqlite_session.commit()

        with patch("cementic.cli.get_engine", return_value=sqlite_engine):
            result = runner.invoke(app, ["collection", "remove", "delme", "--force"])
        assert result.exit_code == 0


class TestListCollectionsCommand:
    """Tests for the list-collections CLI command."""

    def test_list_collections_empty(self, runner, sqlite_engine):
        with patch("cementic.cli.get_engine", return_value=sqlite_engine):
            result = runner.invoke(app, ["collection", "list"])
        assert result.exit_code == 0

    def test_list_collections_with_data(self, runner, sqlite_engine, sqlite_session):
        doc = SourceDocument(source_path="/tmp/a.pdf", collection="c1")
        doc.file_hash = "a"
        doc.status = "pending"
        sqlite_session.add(doc)
        sqlite_session.commit()

        with patch("cementic.cli.get_engine", return_value=sqlite_engine):
            result = runner.invoke(app, ["collection", "list"])
        assert result.exit_code == 0
        assert "c1" in result.stdout


class TestPromoteCommand:
    """Tests for the promote CLI command."""

    def test_promote_ready_revision(self, runner, sqlite_engine, sqlite_session):
        config = get_config()
        extractor_profile = get_or_create_extractor_profile(sqlite_session, config)
        chunk_profile = get_or_create_chunk_profile(sqlite_session, config)
        embedding_profile = get_or_create_embedding_profile(sqlite_session, config)

        revision = PipelineRevision(
            collection="pcol",
            extractor_profile_id=extractor_profile.id,
            chunk_profile_id=chunk_profile.id,
            embedding_profile_id=embedding_profile.id,
            status="ready",
            label="test-rev",
        )
        sqlite_session.add(revision)
        sqlite_session.commit()

        with patch("cementic.cli.get_engine", return_value=sqlite_engine):
            result = runner.invoke(app, ["collection", "promote", "pcol"])
        assert result.exit_code == 0


class TestListRevisionCommand:
    """Tests for the list-collection-revisions CLI command."""

    def test_list_revisions_empty(self, runner, sqlite_engine):
        with patch("cementic.cli.get_engine", return_value=sqlite_engine):
            result = runner.invoke(app, ["collection", "revisions", "nocol"])
        assert result.exit_code == 0

    def test_list_revisions_with_data(self, runner, sqlite_engine, sqlite_session):
        config = get_config()
        extractor_profile = get_or_create_extractor_profile(sqlite_session, config)
        chunk_profile = get_or_create_chunk_profile(sqlite_session, config)
        embedding_profile = get_or_create_embedding_profile(sqlite_session, config)

        rev = PipelineRevision(
            collection="revcol",
            extractor_profile_id=extractor_profile.id,
            chunk_profile_id=chunk_profile.id,
            embedding_profile_id=embedding_profile.id,
            status="active",
            label="v1",
        )
        sqlite_session.add(rev)
        sqlite_session.commit()

        with patch("cementic.cli.get_engine", return_value=sqlite_engine):
            result = runner.invoke(app, ["collection", "revisions", "revcol"])
        assert result.exit_code == 0
        assert "v1" in result.stdout
