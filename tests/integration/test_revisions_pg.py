"""PostgreSQL integration tests for revision ANN indexes and pruning."""

from __future__ import annotations

import json

import pytest
from sqlalchemy import text

from cementic.collections import promote_ready_revision, reindex_collection
from cementic.config import Config
from cementic.db import (
    Chunk,
    ChunkedDocument,
    ChunkEmbedding,
    ChunkProfile,
    EmbeddingProfile,
    ExtractedDocument,
    ExtractorProfile,
    PipelineRevision,
    SourceDocument,
)
from cementic.revisions import ensure_revision_ann_index, ensure_revision_vector_table
from cementic.vector_store import (
    create_table_sql,
    index_access_method,
    upsert_vectors,
    vector_index_name,
    vector_table_exists,
    vector_table_name,
)
from tests.integration.test_pg_helpers import (
    VECTOR_DIM,
    cleanup_pg_tables,
    seed_active_vector_collection,
)


@pytest.mark.pg
def test_ensure_revision_ann_index_creates_real_pg_index(pg_engine, pg_session) -> None:
    cleanup_pg_tables(pg_session)
    revision = seed_active_vector_collection(
        pg_session,
        collection="ann",
        source_path="/docs/ann.pdf",
        chunks=[("ann vector", [1.0, 0.0, 0.0, 0.0])],
    )
    pg_session.commit()
    index_name = vector_index_name(revision.embedding_profile_id)

    config = Config()
    ensure_revision_ann_index(pg_session, revision, config)
    ensure_revision_ann_index(pg_session, revision, config)

    with pg_engine.connect() as conn:
        rows = conn.execute(
            text("SELECT indexname FROM pg_indexes WHERE indexname = :name"),
            {"name": index_name},
        ).all()

    assert [row[0] for row in rows] == [index_name]
    cleanup_pg_tables(pg_session)


@pytest.mark.pg
def test_ensure_revision_ann_index_skips_missing_vector_table(pg_engine, pg_session) -> None:
    """A revision that embedded nothing has no vector table; indexing must no-op.

    Regression: this used to raise UndefinedTable, which the pipeline worker's
    retry loop swallowed, leaving the revision stuck in `building` forever.
    """
    cleanup_pg_tables(pg_session)
    revision = seed_active_vector_collection(
        pg_session,
        collection="novectors",
        source_path="/docs/novectors.pdf",
        chunks=[("unused", [1.0, 0.0, 0.0, 0.0])],
    )
    pg_session.commit()
    profile_id = revision.embedding_profile_id
    with pg_engine.connect() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {vector_table_name(profile_id)}"))
        conn.commit()

    ensure_revision_ann_index(pg_session, revision, Config())

    with pg_engine.connect() as conn:
        assert not vector_table_exists(conn, profile_id)
    cleanup_pg_tables(pg_session)


@pytest.mark.pg
def test_ensure_revision_vector_table_creates_table_up_front(pg_engine, pg_session) -> None:
    """The vector table exists before any embedding succeeds, so the ANN index
    built at promotion time covers every vector inserted afterwards."""
    cleanup_pg_tables(pg_session)
    revision = seed_active_vector_collection(
        pg_session,
        collection="eager",
        source_path="/docs/eager.pdf",
        chunks=[("unused", [1.0, 0.0, 0.0, 0.0])],
    )
    pg_session.commit()
    profile_id = revision.embedding_profile_id
    with pg_engine.connect() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {vector_table_name(profile_id)}"))
        conn.commit()

    ensure_revision_vector_table(pg_session, revision)
    pg_session.commit()

    with pg_engine.connect() as conn:
        assert vector_table_exists(conn, profile_id)
    cleanup_pg_tables(pg_session)


@pytest.mark.pg
def test_ensure_revision_ann_index_rejects_bad_metric(pg_session) -> None:
    cleanup_pg_tables(pg_session)
    revision = seed_active_vector_collection(
        pg_session,
        collection="badmetric",
        source_path="/docs/bad.pdf",
        chunks=[("bad vector", [1.0, 0.0, 0.0, 0.0])],
    )
    revision.embedding_profile.distance_metric = "euclidean"
    pg_session.commit()

    with pytest.raises(ValueError, match="Unsupported distance metric"):
        ensure_revision_ann_index(pg_session, revision, Config())

    cleanup_pg_tables(pg_session)


def _seed_model_swap_with_vectors(session, collection: str) -> list[int]:
    """One document embedded by three models in turn; returns their profile ids.

    The extractor and chunk profiles never change -- only the model does, which
    is what a `model_path` edit produces. Each model gets a real vector table
    holding a real vector, so a leaked table is observable rather than inferred.
    """
    extractor = ExtractorProfile(name="x", config_json="{}", fingerprint=f"ex-{collection}")
    chunk_profile = ChunkProfile(config_json="{}", fingerprint=f"cp-{collection}")
    source = SourceDocument(
        collection=collection, source_path=f"/{collection}.pdf", file_hash="h", status="done"
    )
    embeddings = [
        EmbeddingProfile(
            provider="llama-cpp",
            model_identifier=f"model-{suffix}",
            embedding_dim=VECTOR_DIM,
            distance_metric="cosine",
            config_json=json.dumps({"provider": "llama-cpp", "embedding_dim": VECTOR_DIM}),
            fingerprint=f"ep-{collection}-{suffix}",
        )
        for suffix in ("a", "b", "c")
    ]
    session.add_all([extractor, chunk_profile, source, *embeddings])
    session.flush()

    extracted = ExtractedDocument(
        document_id=source.id,
        extractor_profile_id=extractor.id,
        source_file_hash="h",
        content_hash="ch",
        status="done",
    )
    session.add(extracted)
    session.flush()
    chunked = ChunkedDocument(
        extracted_document_id=extracted.id,
        chunk_profile_id=chunk_profile.id,
        source_content_hash="ch",
        status="done",
        total_chunks=1,
    )
    session.add(chunked)
    session.flush()
    chunk = Chunk(
        document_id=source.id, chunked_document_id=chunked.id, chunk_index=0, content="text"
    )
    session.add(chunk)
    session.flush()

    connection = session.connection()
    # Oldest retired, newest retired, then the ready one about to be promoted:
    # promotion retires the middle revision, leaving the oldest prunable.
    for embedding, status in zip(embeddings, ("retired", "active", "ready")):
        session.add(
            ChunkEmbedding(
                chunk_id=chunk.id, embedding_profile_id=embedding.id, status="done"
            )
        )
        session.add(
            PipelineRevision(
                collection=collection,
                label=f"rev-{embedding.model_identifier}",
                extractor_profile_id=extractor.id,
                chunk_profile_id=chunk_profile.id,
                embedding_profile_id=embedding.id,
                status=status,
            )
        )
        connection.execute(text(create_table_sql(embedding.id, VECTOR_DIM)))
        upsert_vectors(connection, embedding.id, [(chunk.id, [1.0, 0.0, 0.0, 0.0])])
    session.commit()
    return [embedding.id for embedding in embeddings]


@pytest.mark.pg
def test_promotion_drops_the_retired_model_vector_table(pg_engine, pg_session, pg_config) -> None:
    """A model swap must not leave a full copy of the corpus's vectors behind.

    Regression: pruning scoped its embedding delete through the *chunk* profiles
    being removed, so a swap that changed only the model matched nothing. The
    retired model kept its `chunk_embeddings` rows and its entire
    `embedding_vectors_p{id}` table -- one whole copy of the corpus per swap.
    """
    cleanup_pg_tables(pg_session)
    dropped, kept_retired, promoted = _seed_model_swap_with_vectors(pg_session, "swap")

    outcome = promote_ready_revision(pg_session, "swap", config=pg_config)

    assert outcome.status == "promoted"
    with pg_engine.connect() as conn:
        assert not vector_table_exists(conn, dropped)
        # The rollback step and the newly promoted model both stay searchable.
        assert vector_table_exists(conn, kept_retired)
        assert vector_table_exists(conn, promoted)
    surviving = {
        row.embedding_profile_id for row in pg_session.query(ChunkEmbedding).all()
    }
    assert surviving == {kept_retired, promoted}
    cleanup_pg_tables(pg_session)


@pytest.mark.pg
def test_reindex_switches_the_index_method(pg_engine, pg_session, pg_config) -> None:
    """`index.method` was inert once a collection had been built.

    The ANN index is created when a revision first completes, so editing the
    method afterwards changed config and nothing else -- with no command to ask
    for reconciliation. Search compensated by tuning for the index that actually
    existed, which kept results correct but left the setting permanently unmet.
    """
    cleanup_pg_tables(pg_session)
    revision = seed_active_vector_collection(
        pg_session,
        collection="reindexed",
        source_path="/docs/reindexed.pdf",
        chunks=[("a vector", [1.0, 0.0, 0.0, 0.0])],
    )
    pg_session.commit()
    profile_id = revision.embedding_profile_id

    pg_config.index.method = "hnsw"
    first = reindex_collection(pg_session, "reindexed", config=pg_config)
    assert first.status == "reindexed"
    with pg_engine.connect() as conn:
        assert index_access_method(conn, vector_index_name(profile_id)) == "hnsw"

    pg_config.index.method = "diskann"
    second = reindex_collection(pg_session, "reindexed", config=pg_config)

    assert second.previous_method == "hnsw"
    assert second.method == "diskann"
    with pg_engine.connect() as conn:
        assert index_access_method(conn, vector_index_name(profile_id)) == "diskann"
    cleanup_pg_tables(pg_session)


@pytest.mark.pg
def test_reindex_reports_a_collection_with_nothing_promoted(pg_session, pg_config) -> None:
    """Distinguishable from a successful no-op, which is what silence would look like."""
    cleanup_pg_tables(pg_session)

    outcome = reindex_collection(pg_session, "never-built", config=pg_config)

    assert outcome.status == "no_active"
    cleanup_pg_tables(pg_session)


@pytest.mark.pg
def test_reindex_reports_an_active_revision_with_no_vectors(
    pg_engine, pg_session, pg_config
) -> None:
    """A revision can legitimately complete having embedded nothing."""
    cleanup_pg_tables(pg_session)
    revision = seed_active_vector_collection(
        pg_session,
        collection="novecs",
        source_path="/docs/novecs.pdf",
        chunks=[("unused", [1.0, 0.0, 0.0, 0.0])],
    )
    pg_session.commit()
    with pg_engine.connect() as conn:
        conn.execute(
            text(f"DROP TABLE IF EXISTS {vector_table_name(revision.embedding_profile_id)}")
        )
        conn.commit()

    outcome = reindex_collection(pg_session, "novecs", config=pg_config)

    assert outcome.status == "no_vectors"
    cleanup_pg_tables(pg_session)


@pytest.mark.pg
def test_force_rebuilds_even_when_the_method_is_unchanged(
    pg_engine, pg_session, pg_config
) -> None:
    """hnsw_m and ef_construction are fixed when the index is built.

    `CREATE INDEX IF NOT EXISTS` keeps the existing graph, so without a drop a
    changed build parameter would report success and alter nothing. Proven by
    relfilenode: a rebuilt index occupies new storage.
    """
    cleanup_pg_tables(pg_session)
    revision = seed_active_vector_collection(
        pg_session,
        collection="forced",
        source_path="/docs/forced.pdf",
        chunks=[("a vector", [1.0, 0.0, 0.0, 0.0])],
    )
    pg_session.commit()
    index_name = vector_index_name(revision.embedding_profile_id)
    pg_config.index.method = "hnsw"
    reindex_collection(pg_session, "forced", config=pg_config)

    def _relfilenode() -> int:
        with pg_engine.connect() as conn:
            return conn.execute(
                text("SELECT relfilenode FROM pg_class WHERE relname = :name"),
                {"name": index_name},
            ).scalar()

    before = _relfilenode()
    pg_config.index.hnsw_m = pg_config.index.hnsw_m + 4

    unforced = reindex_collection(pg_session, "forced", config=pg_config)
    assert unforced.previous_method == "hnsw"
    assert _relfilenode() == before, "an unforced run must not rebuild"

    reindex_collection(pg_session, "forced", config=pg_config, force=True)

    assert _relfilenode() != before
    cleanup_pg_tables(pg_session)
