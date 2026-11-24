"""Every path that deletes ``Chunk`` rows takes embeddings and vectors with it.

SQLite enforces no foreign keys (nothing issues ``PRAGMA foreign_keys=ON``), so
the unit suite can show that a path deletes chunks but never that the
``ON DELETE CASCADE`` from ``chunks_v2`` clears ``chunk_embeddings`` and the
per-profile ``embedding_vectors_p*`` table. Each test here calls the real
production function against PostgreSQL and counts the rows left in all three
tables, after first asserting they were populated.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import text

from cementic.collections import delete_collection_records
from cementic.config import Config
from cementic.db import (
    Chunk,
    ChunkedDocument,
    ChunkEmbedding,
    ChunkProfile,
    ExtractedDocument,
    PipelineRevision,
    SourceDocument,
)
from cementic.pipeline_worker import PipelineWorker, _purge_superseded_chunks
from cementic.revisions import prune_collection_history, purge_all_chunks
from cementic.source_watcher import _purge_document_chunks
from cementic.vector_store import vector_table_name
from tests.integration.pg_helpers import cleanup_pg_tables, seed_active_vector_collection

CHUNKS = [
    ("one", [1.0, 0.0, 0.0, 0.0]),
    ("two", [0.0, 1.0, 0.0, 0.0]),
    ("three", [0.0, 0.0, 1.0, 0.0]),
]


def _vector_rows(session, profile_id: int) -> int:
    table = vector_table_name(profile_id)
    return int(session.execute(text(f"SELECT count(*) FROM {table}")).scalar_one())


def _seed(session, collection: str = "cascade", **kwargs) -> int:
    """Seed one collection; return its embedding profile id (the revision may be deleted)."""
    cleanup_pg_tables(session)
    revision = seed_active_vector_collection(
        session,
        collection=collection,
        source_path=f"/docs/{collection}.pdf",
        chunks=CHUNKS,
        **kwargs,
    )
    session.commit()
    assert session.query(Chunk).count() == 3
    assert session.query(ChunkEmbedding).count() == 3
    assert _vector_rows(session, revision.embedding_profile_id) == 3
    return revision.embedding_profile_id


def _assert_all_gone(session, profile_id: int) -> None:
    session.expire_all()
    assert session.query(Chunk).count() == 0
    assert session.query(ChunkEmbedding).count() == 0
    assert _vector_rows(session, profile_id) == 0


def _worker(pg_engine, pg_config: Config, tmp_path: Path) -> PipelineWorker:
    pg_config.pipeline_worker.log_file = tmp_path / "worker.log"
    pg_config.pipeline_worker.state_path = tmp_path / "worker.json"
    from sqlalchemy.orm import sessionmaker

    worker = PipelineWorker(pg_config)
    worker.Session = sessionmaker(bind=pg_engine)
    return worker


@pytest.mark.pg
def test_superseded_purge_takes_embeddings_and_vectors(pg_session) -> None:
    profile_id = _seed(pg_session)
    extracted = pg_session.query(ExtractedDocument).one()
    _purge_superseded_chunks(pg_session, extracted.id, "a-different-content-hash")
    pg_session.commit()
    _assert_all_gone(pg_session, profile_id)


@pytest.mark.pg
def test_purge_all_chunks_takes_embeddings_and_vectors(pg_session) -> None:
    profile_id = _seed(pg_session)
    extracted = pg_session.query(ExtractedDocument).one()
    purge_all_chunks(pg_session, extracted.id)
    pg_session.commit()
    _assert_all_gone(pg_session, profile_id)


@pytest.mark.pg
def test_failed_extraction_takes_embeddings_and_vectors(
    pg_engine, pg_session, pg_config, tmp_path
) -> None:
    profile_id = _seed(pg_session)
    extracted_id = pg_session.query(ExtractedDocument.id).scalar()
    _worker(pg_engine, pg_config, tmp_path)._fail_artifact(
        ExtractedDocument,
        extracted_id,
        ValueError("boom"),
        "extract",
        scope_hash_field="source_file_hash",
        scope_hash="h",
    )
    _assert_all_gone(pg_session, profile_id)


@pytest.mark.pg
def test_failed_chunking_takes_embeddings_and_vectors(
    pg_engine, pg_session, pg_config, tmp_path
) -> None:
    profile_id = _seed(pg_session)
    chunked_id = pg_session.query(ChunkedDocument.id).scalar()
    _worker(pg_engine, pg_config, tmp_path)._fail_artifact(
        ChunkedDocument,
        chunked_id,
        ValueError("boom"),
        "chunk",
        scope_hash_field="source_content_hash",
        scope_hash="h",
    )
    _assert_all_gone(pg_session, profile_id)


@pytest.mark.pg
def test_rechunk_write_back_takes_old_embeddings_and_vectors(
    pg_engine, pg_session, pg_config, tmp_path
) -> None:
    """A re-chunk that fails replaces the old chunks with none."""
    profile_id = _seed(pg_session)
    extracted_id = pg_session.query(ExtractedDocument.id).scalar()
    chunked_id = pg_session.query(ChunkedDocument.id).scalar()
    _worker(pg_engine, pg_config, tmp_path)._write_back_chunks(
        extracted_id, chunked_id, [], "failed", "boom"
    )
    _assert_all_gone(pg_session, profile_id)


@pytest.mark.pg
def test_watcher_purge_takes_embeddings_and_vectors(pg_session) -> None:
    profile_id = _seed(pg_session)
    document_id = pg_session.query(SourceDocument.id).scalar()
    _purge_document_chunks(pg_session, [document_id])
    pg_session.commit()
    _assert_all_gone(pg_session, profile_id)


@pytest.mark.pg
def test_collection_remove_takes_embeddings_and_vectors(pg_session) -> None:
    profile_id = _seed(pg_session)
    result = delete_collection_records(pg_session, "cascade")
    assert result is not None and result.deleted_chunks == 3
    _assert_all_gone(pg_session, profile_id)


@pytest.mark.pg
def test_prune_takes_embeddings_and_vectors_of_a_kept_embedding_profile(pg_session) -> None:
    """Pruning a superseded chunking, while its embedding profile stays in use.

    The explicit embedding delete excludes kept embedding profiles, so here the
    only thing that can clear the chunk embeddings and vectors is the chunk
    delete's cascade. The explicit ``Chunk`` delete is itself redundant on
    PostgreSQL, because the ``ChunkedDocument`` delete right after it cascades
    to the same rows; the test fails only when both are removed.
    """
    profile_id = _seed(pg_session, status="superseded")
    revision = pg_session.query(PipelineRevision).one()
    new_chunk_profile = ChunkProfile(config_json="{}", fingerprint="chunk-new")
    pg_session.add(new_chunk_profile)
    pg_session.flush()
    pg_session.add(
        PipelineRevision(
            collection="cascade",
            label="kept",
            extractor_profile_id=revision.extractor_profile_id,
            chunk_profile_id=new_chunk_profile.id,
            embedding_profile_id=revision.embedding_profile_id,
            status="active",
        )
    )
    pg_session.commit()

    prune_collection_history(pg_session, "cascade")
    pg_session.commit()

    _assert_all_gone(pg_session, profile_id)
