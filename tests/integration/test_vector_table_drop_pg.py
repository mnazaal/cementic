"""Deferred vector-table cleanup must not drop a table another collection uses.

``delete_collection_records`` and pruning work out which per-profile vector
tables are droppable *before* they commit, and the caller drops them after. A
collection created on the same embedding profile in between made that decision
stale, and the drop was unconditional.
"""

from __future__ import annotations

import threading

import pytest
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from cementic.collections import delete_collection_records, drop_orphan_vector_tables
from cementic.db import (
    Chunk,
    ChunkedDocument,
    ChunkEmbedding,
    ChunkProfile,
    ExtractedDocument,
    ExtractorProfile,
    PipelineRevision,
    SourceDocument,
)
from cementic.revisions import ensure_revision_vector_table
from cementic.vector_store import upsert_vectors, vector_table_exists, vector_table_name
from tests.integration.pg_helpers import cleanup_pg_tables, seed_active_vector_collection


def _add_revision(session, collection: str, profile_id: int) -> PipelineRevision:
    """A building revision for ``collection`` on an existing embedding profile."""
    extractor = ExtractorProfile(name="x", config_json="{}", fingerprint=f"x-{collection}")
    chunk_profile = ChunkProfile(config_json="{}", fingerprint=f"c-{collection}")
    session.add_all([extractor, chunk_profile])
    session.flush()
    revision = PipelineRevision(
        collection=collection,
        label=f"rev-{collection}",
        extractor_profile_id=extractor.id,
        chunk_profile_id=chunk_profile.id,
        embedding_profile_id=profile_id,
        status="building",
    )
    session.add(revision)
    session.flush()
    return revision


def _populate_on_profile(session, collection: str, profile_id: int) -> None:
    """Create ``collection`` on an existing profile, with one embedded chunk."""
    revision = _add_revision(session, collection, profile_id)
    source = SourceDocument(
        collection=collection, source_path=f"/docs/{collection}.pdf", file_hash="h", status="done"
    )
    session.add(source)
    session.flush()
    extracted = ExtractedDocument(
        document_id=source.id,
        extractor_profile_id=revision.extractor_profile_id,
        source_file_hash="h",
        content_hash="c",
        status="done",
    )
    session.add(extracted)
    session.flush()
    chunked = ChunkedDocument(
        extracted_document_id=extracted.id,
        chunk_profile_id=revision.chunk_profile_id,
        source_content_hash="c",
        status="done",
        total_chunks=1,
    )
    session.add(chunked)
    session.flush()
    chunk = Chunk(
        document_id=source.id, chunked_document_id=chunked.id, chunk_index=0, content="kept"
    )
    session.add(chunk)
    session.flush()
    session.add(
        ChunkEmbedding(
            chunk_id=chunk.id,
            embedding_profile_id=profile_id,
            status="done",
            collection=collection,
            extractor_profile_id=revision.extractor_profile_id,
            chunk_profile_id=revision.chunk_profile_id,
        )
    )
    upsert_vectors(
        session.connection(),
        profile_id,
        [(chunk.id, [0.0, 0.0, 0.0, 1.0])],
        collection=collection,
        extractor_profile_id=revision.extractor_profile_id,
        chunk_profile_id=revision.chunk_profile_id,
    )


def _seed_and_delete(pg_session) -> tuple[int, list[int]]:
    """Seed collection ``old``, delete it, and return (profile id, retained cleanup list)."""
    cleanup_pg_tables(pg_session)
    revision = seed_active_vector_collection(
        pg_session,
        collection="old",
        source_path="/docs/old.pdf",
        chunks=[("old", [1.0, 0.0, 0.0, 0.0])],
    )
    profile_id = revision.embedding_profile_id
    pg_session.commit()
    result = delete_collection_records(pg_session, "old")
    assert result is not None
    assert result.vector_profile_ids == [profile_id]
    return profile_id, result.vector_profile_ids


def _vector_rows(session, profile_id: int) -> int:
    table = vector_table_name(profile_id)
    return int(session.execute(text(f"SELECT count(*) FROM {table}")).scalar_one())


@pytest.mark.pg
def test_cleanup_keeps_a_table_a_new_collection_now_uses(pg_engine, pg_session) -> None:
    profile_id, stale_cleanup = _seed_and_delete(pg_session)
    _populate_on_profile(pg_session, "new", profile_id)
    pg_session.commit()
    assert _vector_rows(pg_session, profile_id) == 1
    pg_session.commit()  # release the read lock, or the drop below would wait on it

    drop_orphan_vector_tables(pg_engine, stale_cleanup)

    pg_session.expire_all()
    assert vector_table_exists(pg_session.connection(), profile_id)
    assert _vector_rows(pg_session, profile_id) == 1


@pytest.mark.pg
def test_cleanup_still_drops_a_true_orphan(pg_engine, pg_session) -> None:
    profile_id, cleanup = _seed_and_delete(pg_session)

    drop_orphan_vector_tables(pg_engine, cleanup)

    pg_session.expire_all()
    assert not vector_table_exists(pg_session.connection(), profile_id)


@pytest.mark.pg
def test_a_drop_waits_for_an_uncommitted_reuse_and_then_spares_the_table(
    pg_engine, pg_session
) -> None:
    """The recheck alone leaves a window; the lock closes it.

    The reusing transaction has inserted its revision but not committed, so a
    recheck by the dropper cannot see it. The dropper must therefore block on
    the per-profile lock the reuse holds, and see the revision once it commits.
    """
    profile_id, cleanup = _seed_and_delete(pg_session)

    reuse = sessionmaker(bind=pg_engine)()
    try:
        revision = _add_revision(reuse, "new", profile_id)
        ensure_revision_vector_table(reuse, revision)  # takes the per-profile lock

        dropper = threading.Thread(target=drop_orphan_vector_tables, args=(pg_engine, cleanup))
        dropper.start()
        dropper.join(timeout=1.0)
        assert dropper.is_alive(), "the drop did not wait for the in-flight reuse"

        reuse.commit()
        dropper.join(timeout=10.0)
        assert not dropper.is_alive()
    finally:
        reuse.close()

    pg_session.expire_all()
    assert vector_table_exists(pg_session.connection(), profile_id)


@pytest.mark.pg
def test_a_drop_blocked_by_an_open_reader_gives_up_and_a_later_drop_succeeds(
    pg_engine, pg_session, monkeypatch
) -> None:
    """DROP TABLE waits on any open transaction that touched the table.

    Without a lock timeout it would hold the profile lock and queue every new
    reader behind it. The drop must give up, leave the table, and succeed once
    the blocker is gone.
    """
    import time

    import cementic
    from cementic import vector_store

    print("cementic imported from", cementic.__file__)
    monkeypatch.setattr(vector_store, "DROP_LOCK_TIMEOUT_MS", 500)
    profile_id, cleanup = _seed_and_delete(pg_session)
    pg_session.commit()  # release this session's own locks

    reader = pg_engine.connect()
    try:
        reader.execute(
            text(f"SELECT count(*) FROM {vector_table_name(profile_id)}")
        ).scalar_one()  # the open transaction now holds ACCESS SHARE
        started = time.monotonic()
        dropped = vector_store.drop_vector_table_if_unreferenced(pg_engine, profile_id)
        elapsed = time.monotonic() - started
    finally:
        reader.rollback()
        reader.close()

    assert dropped is False
    assert elapsed < 5.0
    assert vector_table_exists(pg_session.connection(), profile_id)
    pg_session.commit()

    assert drop_orphan_vector_tables(pg_engine, cleanup) == 1
    pg_session.expire_all()
    assert not vector_table_exists(pg_session.connection(), profile_id)
