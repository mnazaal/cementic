"""Integration tests for CLI collection commands backed by SQLite."""

from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typer.testing import CliRunner

from cementic.cli import app
from cementic.config import get_config
from cementic.db import (
    Base,
    Chunk,
    ChunkedDocument,
    ChunkEmbedding,
    ExtractedDocument,
    PipelineRevision,
    SourceDocument,
)
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


def _seed_ready_revision(session, collection: str, *, with_document: bool) -> PipelineRevision:
    """Create a ready revision, optionally with one fully-processed document.

    Promotion re-checks completeness against current counts, so a revision with
    no work in it is refused. `with_document=False` builds exactly that case.
    """
    config = get_config()
    extractor_profile = get_or_create_extractor_profile(session, config)
    chunk_profile = get_or_create_chunk_profile(session, config)
    embedding_profile = get_or_create_embedding_profile(session, config)

    revision = PipelineRevision(
        collection=collection,
        extractor_profile_id=extractor_profile.id,
        chunk_profile_id=chunk_profile.id,
        embedding_profile_id=embedding_profile.id,
        status="ready",
        label="test-rev",
    )
    session.add(revision)

    if with_document:
        document = SourceDocument(
            collection=collection, source_path=f"/{collection}.pdf", file_hash="a"
        )
        document.status = "pending"
        session.add(document)
        session.flush()
        extracted = ExtractedDocument(
            document_id=document.id,
            extractor_profile_id=extractor_profile.id,
            status="done",
            source_file_hash="a",
            content_hash="c",
        )
        session.add(extracted)
        session.flush()
        chunked = ChunkedDocument(
            extracted_document_id=extracted.id,
            chunk_profile_id=chunk_profile.id,
            status="done",
            source_content_hash="c",
        )
        session.add(chunked)
        session.flush()
        chunk = Chunk(
            document_id=document.id,
            chunked_document_id=chunked.id,
            chunk_index=0,
            content="chunk text",
        )
        session.add(chunk)
        session.flush()
        session.add(
            ChunkEmbedding(
                chunk_id=chunk.id,
                embedding_profile_id=embedding_profile.id,
                status="done",
            )
        )

    session.commit()
    return revision


class TestPromoteCommand:
    """Tests for the promote CLI command."""

    def test_promote_ready_revision(self, runner, sqlite_engine, sqlite_session):
        _seed_ready_revision(sqlite_session, "pcol", with_document=True)

        with patch("cementic.cli.get_engine", return_value=sqlite_engine):
            result = runner.invoke(app, ["collection", "promote", "pcol"])
        assert result.exit_code == 0
        assert "promoted" in result.stdout

    def test_promote_refuses_a_revision_with_no_documents(
        self, runner, sqlite_engine, sqlite_session
    ):
        """Promotion retires whatever is active, so publishing an empty revision
        removes search coverage rather than merely adding none."""
        _seed_ready_revision(sqlite_session, "emptycol", with_document=False)

        with patch("cementic.cli.get_engine", return_value=sqlite_engine):
            result = runner.invoke(app, ["collection", "promote", "emptycol"])
        assert result.exit_code == 1
        assert "no documents" in result.stdout


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
