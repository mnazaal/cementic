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
from cementic.revisions import (
    ensure_revision_ann_index,
    ensure_revision_ann_index_up_front,
    ensure_revision_vector_table,
    materialize_pending_embeddings,
)
from cementic.vector_store import (
    create_table_sql,
    ensure_vector_table_schema,
    index_access_method,
    upsert_vectors,
    vector_index_name,
    vector_table_exists,
    vector_table_name,
)
from tests.integration.pg_helpers import (
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
def test_ann_index_is_created_up_front_on_an_empty_table(pg_engine, pg_session) -> None:
    """HNSW has no training step, so the index is created before any inserts.

    Every insert then maintains the graph incrementally: no build stall at the
    ready transition, per-batch resumability, and a searchable index during
    ingestion (measured equal recall/latency to the build-after-load order).
    """
    cleanup_pg_tables(pg_session)
    revision = seed_active_vector_collection(
        pg_session,
        collection="upfront",
        source_path="/docs/upfront.pdf",
        chunks=[("seed", [1.0, 0.0, 0.0, 0.0])],
    )
    pg_session.commit()
    profile_id = revision.embedding_profile_id
    with pg_engine.connect() as conn:
        conn.execute(text(f"DELETE FROM {vector_table_name(profile_id)}"))
        conn.execute(text(f"DROP INDEX IF EXISTS {vector_index_name(profile_id)}"))
        conn.commit()

    ensure_revision_ann_index_up_front(pg_session, revision, Config())

    with pg_engine.connect() as conn:
        assert index_access_method(conn, vector_index_name(profile_id)) == "hnsw"
    cleanup_pg_tables(pg_session)


@pytest.mark.pg
def test_up_front_index_leaves_a_populated_table_to_the_ready_build(
    pg_engine, pg_session
) -> None:
    """Rows but no index means a resumed build from an older version.

    Creating the index here would run the bulk build synchronously at worker
    startup -- the exact stall the up-front path exists to avoid -- so it is
    left to the ready transition, where it is announced and expected.
    """
    cleanup_pg_tables(pg_session)
    revision = seed_active_vector_collection(
        pg_session,
        collection="resumed",
        source_path="/docs/resumed.pdf",
        chunks=[("existing vector", [1.0, 0.0, 0.0, 0.0])],
    )
    pg_session.commit()
    profile_id = revision.embedding_profile_id
    with pg_engine.connect() as conn:
        conn.execute(text(f"DROP INDEX IF EXISTS {vector_index_name(profile_id)}"))
        conn.commit()

    ensure_revision_ann_index_up_front(pg_session, revision, Config())

    with pg_engine.connect() as conn:
        assert index_access_method(conn, vector_index_name(profile_id)) is None
    cleanup_pg_tables(pg_session)


@pytest.mark.pg
def test_up_front_index_only_applies_to_hnsw(pg_engine, pg_session) -> None:
    """DiskANN keeps today's build-at-ready behavior; only HNSW was measured."""
    cleanup_pg_tables(pg_session)
    revision = seed_active_vector_collection(
        pg_session,
        collection="diskann-upfront",
        source_path="/docs/diskann-upfront.pdf",
        chunks=[("seed", [1.0, 0.0, 0.0, 0.0])],
    )
    pg_session.commit()
    profile_id = revision.embedding_profile_id
    with pg_engine.connect() as conn:
        conn.execute(text(f"DELETE FROM {vector_table_name(profile_id)}"))
        conn.execute(text(f"DROP INDEX IF EXISTS {vector_index_name(profile_id)}"))
        conn.commit()

    config = Config()
    config.index.method = "diskann"
    ensure_revision_ann_index_up_front(pg_session, revision, config)

    with pg_engine.connect() as conn:
        assert index_access_method(conn, vector_index_name(profile_id)) is None
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
        upsert_vectors(
            connection,
            embedding.id,
            [(chunk.id, [1.0, 0.0, 0.0, 0.0])],
            collection=collection,
            extractor_profile_id=extractor.id,
            chunk_profile_id=chunk_profile.id,
        )
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


@pytest.mark.pg
def test_migrating_an_already_current_table_is_a_no_op(pg_engine, pg_session) -> None:
    """Runs on every embed batch, so it must not rewrite the table each time."""
    cleanup_pg_tables(pg_session)
    revision = seed_active_vector_collection(
        pg_session,
        collection="current",
        source_path="/docs/current.pdf",
        chunks=[("text", [1.0, 0.0, 0.0, 0.0])],
    )
    pg_session.commit()
    profile_id = revision.embedding_profile_id

    with pg_engine.begin() as conn:
        ensure_vector_table_schema(conn, profile_id, VECTOR_DIM)
        ensure_vector_table_schema(conn, profile_id, VECTOR_DIM)

    with pg_engine.connect() as conn:
        assert conn.execute(
            text(f"SELECT count(*) FROM {vector_table_name(profile_id)}")
        ).scalar() == 1
    cleanup_pg_tables(pg_session)


def test_materialize_pending_embeddings_is_idempotent(pg_session) -> None:
    """Callers run it whenever work might have appeared, so a no-op must be free.

    It replaces an anti-join that ran once per 32-chunk batch and cost 178,681
    buffers; the whole point is that it can be called liberally and settles to
    inserting nothing.
    """
    cleanup_pg_tables(pg_session)
    revision = seed_active_vector_collection(
        pg_session,
        collection="materialise",
        source_path="/docs/a.pdf",
        chunks=[(f"chunk {i}", [0.1] * VECTOR_DIM) for i in range(5)],
    )
    pg_session.commit()
    # The fixture seeds finished embeddings; clear them so there is work to find.
    pg_session.query(ChunkEmbedding).delete()
    pg_session.commit()

    added = materialize_pending_embeddings(pg_session, "materialise", revision)
    pg_session.commit()
    assert added == 5
    assert {status for (status,) in pg_session.query(ChunkEmbedding.status).all()} == {"pending"}

    again = materialize_pending_embeddings(pg_session, "materialise", revision)
    pg_session.commit()
    assert again == 0
    assert pg_session.query(ChunkEmbedding).count() == 5


def test_materialize_pending_embeddings_leaves_finished_work_alone(pg_session) -> None:
    """A row that already exists must not be reset to pending and re-embedded."""
    cleanup_pg_tables(pg_session)
    revision = seed_active_vector_collection(
        pg_session,
        collection="materialise-done",
        source_path="/docs/b.pdf",
        chunks=[(f"chunk {i}", [0.1] * VECTOR_DIM) for i in range(3)],
    )
    pg_session.commit()

    added = materialize_pending_embeddings(pg_session, "materialise-done", revision)
    pg_session.commit()
    assert added == 0
    assert {status for (status,) in pg_session.query(ChunkEmbedding.status).all()} == {"done"}


def test_materialize_pending_embeddings_skips_deleted_documents(pg_session) -> None:
    """A document the watcher marked deleted is not work, and must not be queued."""
    cleanup_pg_tables(pg_session)
    revision = seed_active_vector_collection(
        pg_session,
        collection="materialise-deleted",
        source_path="/docs/c.pdf",
        chunks=[(f"chunk {i}", [0.1] * VECTOR_DIM) for i in range(4)],
    )
    pg_session.commit()
    pg_session.query(ChunkEmbedding).delete()
    pg_session.query(SourceDocument).update({"status": "deleted"}, synchronize_session=False)
    pg_session.commit()

    assert materialize_pending_embeddings(pg_session, "materialise-deleted", revision) == 0


def test_the_embedding_claim_never_scans_more_than_a_batch(pg_session) -> None:
    """The claim must read `batch_size` rows however much of the corpus is done.

    This is the property the whole denormalisation exists for, and it is easy to
    lose: adding an ORDER BY makes PostgreSQL read every pending row and top-N
    sort it before honouring the LIMIT, and putting a scope filter back on a
    joined table makes it drive from the collection and walk the finished
    prefix. Measured at 300k chunks / 90% embedded, the joined shape cost
    1,895,124 buffers against 2,804 for this one.
    """
    cleanup_pg_tables(pg_session)
    revision = seed_active_vector_collection(
        pg_session,
        collection="claimscan",
        source_path="/docs/scan.pdf",
        chunks=[(f"chunk {i}", [0.1] * VECTOR_DIM) for i in range(200)],
    )
    pg_session.commit()
    pg_session.query(ChunkEmbedding).delete()
    pg_session.commit()
    materialize_pending_embeddings(pg_session, "claimscan", revision)
    # Drive it to 90% done: the regime where a joined claim degrades worst.
    done = [
        row_id
        for (row_id,) in pg_session.query(ChunkEmbedding.id).order_by(ChunkEmbedding.id).limit(180)
    ]
    pg_session.query(ChunkEmbedding).filter(ChunkEmbedding.id.in_(done)).update(
        {"status": "done"}, synchronize_session=False
    )
    pg_session.commit()
    pg_session.execute(text("ANALYZE chunk_embeddings"))

    plan = "\n".join(
        row[0]
        for row in pg_session.execute(
            text(
                "EXPLAIN (ANALYZE, BUFFERS, TIMING OFF) "
                "SELECT ce.chunk_id FROM chunk_embeddings ce "
                "WHERE ce.embedding_profile_id = :m AND ce.status IN ('pending','processing') "
                "AND ce.collection = :c AND ce.extractor_profile_id = :e "
                "AND ce.chunk_profile_id = :k LIMIT 32"
            ),
            {
                "m": revision.embedding_profile_id,
                "c": "claimscan",
                "e": revision.extractor_profile_id,
                "k": revision.chunk_profile_id,
            },
        )
    )
    assert "Sort" not in plan, f"a sort defeats early termination:\n{plan}"
    scanned = max(int(float(part.split()[0])) for part in plan.split("actual rows=")[1:])
    assert scanned <= 32, f"claim read {scanned} rows for a 32-row batch:\n{plan}"
