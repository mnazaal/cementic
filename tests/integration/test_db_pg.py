"""PostgreSQL integration tests for db.py: vector extensions and ANN indexes."""

from typing import NamedTuple

import pytest
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from cementic import vector_store
from cementic.db import (
    LEXICAL_INDEX_NAME,
    Chunk,
    ChunkedDocument,
    ChunkProfile,
    EmbeddingProfile,
    ExtractedDocument,
    ExtractorProfile,
    SourceDocument,
    _lexical_index_exists,
    create_tables,
    ensure_embedding_ann_index,
    ensure_vector_extensions,
)
from cementic.index_strategies import IndexParams


class _MinimalPipeline(NamedTuple):
    """Ids a vector row needs: its chunk, its profile, and its routing."""

    chunk_id: int
    embedding_profile_id: int
    collection: str
    extractor_profile_id: int
    chunk_profile_id: int


def _create_minimal_pipeline(session, *, embedding_dim: int = 4) -> "_MinimalPipeline":
    """Create minimal DB state: SourceDoc → ExtractedDoc → ChunkedDoc → Chunk."""
    src = SourceDocument(
        source_path="/test/minimal.pdf",
        file_hash="minimal-hash",
        collection="test-col",
        status="pending",
    )
    ext_prof = ExtractorProfile(name="test-ext", config_json="{}", fingerprint="ep-fp")
    chunk_prof = ChunkProfile(config_json="{}", fingerprint="cp-fp")
    emb_prof = EmbeddingProfile(
        provider="llama-cpp",
        model_identifier="test-model",
        embedding_dim=embedding_dim,
        config_json="{}",
        fingerprint="emb-fp",
    )
    session.add_all([src, ext_prof, chunk_prof, emb_prof])
    session.flush()

    ext_doc = ExtractedDocument(
        document_id=src.id,
        extractor_profile_id=ext_prof.id,
        artifact_path="/test/minimal.json",
        status="done",
    )
    session.add(ext_doc)
    session.flush()

    chunked = ChunkedDocument(
        extracted_document_id=ext_doc.id,
        chunk_profile_id=chunk_prof.id,
        status="done",
    )
    session.add(chunked)
    session.flush()

    chunk = Chunk(
        document_id=src.id,
        chunked_document_id=chunked.id,
        chunk_index=1,
        content="test chunk for ann index",
    )
    session.add(chunk)
    session.flush()

    return _MinimalPipeline(
        chunk_id=chunk.id,
        embedding_profile_id=emb_prof.id,
        collection=src.collection,
        extractor_profile_id=ext_prof.id,
        chunk_profile_id=chunk_prof.id,
    )


@pytest.mark.pg
class TestVectorExtensions:
    """Test ensure_vector_extensions with real PostgreSQL."""

    def test_creates_vectorscale_extension(self, pg_engine):
        """ensure_vector_extensions creates vectorscale and vector extensions."""
        ensure_vector_extensions(pg_engine)

        with pg_engine.connect() as conn:
            result = conn.execute(
                text("SELECT extname FROM pg_extension WHERE extname IN ('vectorscale', 'vector')")
            )
            extensions = {row[0] for row in result}
            assert "vector" in extensions
            assert "vectorscale" in extensions

    def test_idempotent_call(self, pg_engine):
        """ensure_vector_extensions can be called multiple times safely."""
        ensure_vector_extensions(pg_engine)
        ensure_vector_extensions(pg_engine)
        # Should not raise


def _cleanup(pg_engine, emb_prof_id: int) -> None:
    vector_store.drop_vector_table(pg_engine, emb_prof_id)
    with pg_engine.connect() as conn:
        for table in [
            "chunk_embeddings", "chunks_v2", "chunked_documents",
            "extracted_documents", "source_documents",
            "embedding_profiles", "extractor_profiles", "chunk_profiles",
        ]:
            conn.execute(text(f"DELETE FROM {table}"))
        conn.commit()


def _create_vector_table(engine, profile_id: int, dim: int) -> None:
    """Create a profile's vector table the same way the embed step does."""
    with engine.begin() as conn:
        conn.execute(text(vector_store.create_table_sql(profile_id, dim)))


@pytest.mark.pg
class TestAnnIndex:
    """Build each ANN index method on a real per-profile vector table and search it."""

    @pytest.mark.parametrize("method", ["hnsw", "diskann"])
    def test_build_index_and_search(self, pg_engine, method):
        session_factory = sessionmaker(bind=pg_engine)
        with session_factory() as session:
            pipeline = _create_minimal_pipeline(session, embedding_dim=4)
            chunk_id, emb_prof_id = pipeline.chunk_id, pipeline.embedding_profile_id
            session.commit()

        try:
            _create_vector_table(pg_engine, emb_prof_id, 4)
            with pg_engine.begin() as conn:
                vector_store.upsert_vectors(
                    conn,
                    emb_prof_id,
                    [(chunk_id, [0.1, 0.2, 0.3, 0.4])],
                    collection=pipeline.collection,
                    extractor_profile_id=pipeline.extractor_profile_id,
                    chunk_profile_id=pipeline.chunk_profile_id,
                )

            ensure_embedding_ann_index(
                pg_engine, profile_id=emb_prof_id, method=method, params=IndexParams()
            )

            table = vector_store.vector_table_name(emb_prof_id)
            with pg_engine.connect() as conn:
                indexes = [
                    row[0]
                    for row in conn.execute(
                        text("SELECT indexname FROM pg_indexes WHERE tablename = :t"),
                        {"t": table},
                    )
                ]
                assert any("ann" in name for name in indexes), f"no ANN index: {indexes}"

                results = list(
                    conn.execute(
                        text(
                            f"SELECT chunk_id FROM {table} "
                            "ORDER BY embedding <=> (:q)::vector LIMIT 5"
                        ),
                        {"q": vector_store.to_vector_literal([0.1, 0.2, 0.3, 0.4])},
                    )
                )
            assert results and results[0][0] == chunk_id
        finally:
            _cleanup(pg_engine, emb_prof_id)

    def test_idempotent_ann_index(self, pg_engine):
        session_factory = sessionmaker(bind=pg_engine)
        with session_factory() as session:
            pipeline = _create_minimal_pipeline(session, embedding_dim=4)
            chunk_id, emb_prof_id = pipeline.chunk_id, pipeline.embedding_profile_id
            session.commit()

        try:
            _create_vector_table(pg_engine, emb_prof_id, 4)
            with pg_engine.begin() as conn:
                vector_store.upsert_vectors(
                    conn,
                    emb_prof_id,
                    [(chunk_id, [0.5, 0.6, 0.7, 0.8])],
                    collection=pipeline.collection,
                    extractor_profile_id=pipeline.extractor_profile_id,
                    chunk_profile_id=pipeline.chunk_profile_id,
                )
            ensure_embedding_ann_index(
                pg_engine, profile_id=emb_prof_id, method="hnsw", params=IndexParams()
            )
            ensure_embedding_ann_index(
                pg_engine, profile_id=emb_prof_id, method="hnsw", params=IndexParams()
            )  # should not raise
        finally:
            _cleanup(pg_engine, emb_prof_id)

    def test_failed_force_rebuild_keeps_the_old_index(self, pg_engine):
        """Regression (fourth review, previously untested): the drop used to
        commit separately from the create, so a rebuild that failed left the
        collection with no ANN index at all -- permanently and silently, since
        search keeps working by sequential scan."""
        from unittest.mock import patch

        session_factory = sessionmaker(bind=pg_engine)
        with session_factory() as session:
            pipeline = _create_minimal_pipeline(session, embedding_dim=4)
            emb_prof_id = pipeline.embedding_profile_id
            session.commit()

        try:
            _create_vector_table(pg_engine, emb_prof_id, 4)
            ensure_embedding_ann_index(
                pg_engine, profile_id=emb_prof_id, method="hnsw", params=IndexParams()
            )

            with (
                patch(
                    "cementic.db.build_index_ddl",
                    return_value="CREATE INDEX this is not valid SQL",
                ),
                pytest.raises(Exception),
            ):
                ensure_embedding_ann_index(
                    pg_engine,
                    profile_id=emb_prof_id,
                    method="hnsw",
                    params=IndexParams(),
                    force_rebuild=True,
                )

            table = vector_store.vector_table_name(emb_prof_id)
            with pg_engine.connect() as conn:
                indexes = [
                    row[0]
                    for row in conn.execute(
                        text("SELECT indexname FROM pg_indexes WHERE tablename = :t"),
                        {"t": table},
                    )
                ]
            assert any("ann" in name for name in indexes), (
                f"the old ANN index must survive a failed rebuild: {indexes}"
            )
        finally:
            _cleanup(pg_engine, emb_prof_id)

    def test_build_memory_does_not_leak_into_the_pool(self, pg_engine):
        """Regression: a session-level SET pinned maintenance_work_mem on the
        pooled connection -- the code comment claimed the connection was
        discarded, but engine.connect() returns it to the pool, so every later
        borrower ran with the build's gigabytes. SET LOCAL scopes it to the
        build transaction."""
        session_factory = sessionmaker(bind=pg_engine)
        with session_factory() as session:
            pipeline = _create_minimal_pipeline(session, embedding_dim=4)
            emb_prof_id = pipeline.embedding_profile_id
            session.commit()

        try:
            _create_vector_table(pg_engine, emb_prof_id, 4)
            with pg_engine.connect() as conn:
                default_value = conn.execute(text("SHOW maintenance_work_mem")).scalar()
            build_memory = "1234MB"
            assert build_memory != default_value

            ensure_embedding_ann_index(
                pg_engine,
                profile_id=emb_prof_id,
                method="hnsw",
                params=IndexParams(),
                build_memory=build_memory,
            )

            # Drain the whole pool, not a hard-coded three: the build's
            # connection returns to the *back* of the FIFO queue, so checking
            # out three happened to reach it only while occupancy was low. With
            # four or more idle connections -- another pg test holding some, or
            # -n parallelism -- the tainted connection was never sampled and
            # this test silently stopped testing anything.
            pool = pg_engine.pool
            checkout_count = pool.size() + pool.overflow() + 1
            conns = [pg_engine.connect() for _ in range(max(checkout_count, 3))]
            try:
                values = {
                    conn.execute(text("SHOW maintenance_work_mem")).scalar()
                    for conn in conns
                }
            finally:
                for conn in conns:
                    conn.close()
            assert values == {default_value}
        finally:
            _cleanup(pg_engine, emb_prof_id)

    def test_switch_index_method_rebuilds(self, pg_engine):
        """Switching index.method must rebuild the index, not silently keep the old one."""
        session_factory = sessionmaker(bind=pg_engine)
        with session_factory() as session:
            pipeline = _create_minimal_pipeline(session, embedding_dim=4)
            chunk_id, emb_prof_id = pipeline.chunk_id, pipeline.embedding_profile_id
            session.commit()

        index_name = vector_store.vector_index_name(emb_prof_id)
        try:
            _create_vector_table(pg_engine, emb_prof_id, 4)
            with pg_engine.begin() as conn:
                vector_store.upsert_vectors(
                    conn,
                    emb_prof_id,
                    [(chunk_id, [0.1, 0.2, 0.3, 0.4])],
                    collection=pipeline.collection,
                    extractor_profile_id=pipeline.extractor_profile_id,
                    chunk_profile_id=pipeline.chunk_profile_id,
                )

            ensure_embedding_ann_index(
                pg_engine, profile_id=emb_prof_id, method="hnsw", params=IndexParams()
            )
            with pg_engine.connect() as conn:
                assert vector_store.index_access_method(conn, index_name) == "hnsw"

            ensure_embedding_ann_index(
                pg_engine, profile_id=emb_prof_id, method="diskann", params=IndexParams()
            )
            with pg_engine.connect() as conn:
                assert vector_store.index_access_method(conn, index_name) == "diskann"

            ensure_embedding_ann_index(
                pg_engine, profile_id=emb_prof_id, method="hnsw", params=IndexParams()
            )
            with pg_engine.connect() as conn:
                assert vector_store.index_access_method(conn, index_name) == "hnsw"
        finally:
            _cleanup(pg_engine, emb_prof_id)

    def test_unsupported_distance_metric(self, pg_engine):
        with pytest.raises(ValueError, match="Unsupported distance metric"):
            ensure_embedding_ann_index(
                pg_engine,
                profile_id=1,
                method="hnsw",
                params=IndexParams(),
                distance_metric="euclidean",
            )


@pytest.mark.pg
class TestActiveRevisionIndexConcurrency:
    """create_tables runs in both workers at once; the index DDL must tolerate it."""

    def test_two_concurrent_create_tables_calls_both_succeed(self, pg_engine):
        """Regression: `CREATE UNIQUE INDEX IF NOT EXISTS` checks the catalog
        *before* taking its lock, so the two processes `cementic start` spawns
        can both pass the check and race. The loser got a duplicate-key error on
        pg_class -- not a RuntimeError, so runner.py let it through as a raw
        traceback and the worker "exited immediately" on the first start after
        every upgrade."""
        import threading

        from cementic.db import _active_revision_index_exists, create_tables

        with pg_engine.begin() as conn:
            conn.execute(text("DROP INDEX IF EXISTS uq_pipeline_revisions_one_active"))

        outcomes: list[str] = []
        barrier = threading.Barrier(4)

        def run() -> None:
            barrier.wait()
            try:
                create_tables(pg_engine)
                outcomes.append("ok")
            except Exception as error:  # noqa: BLE001 - the failure is the point
                outcomes.append(type(error).__name__)

        threads = [threading.Thread(target=run) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert outcomes == ["ok"] * 4, outcomes
        assert _active_revision_index_exists(pg_engine)


class TestLexicalIndexPg:
    """The full-text index, against a real PostgreSQL."""

    def test_create_tables_leaves_a_valid_index(self, pg_engine):
        """It must exist *and* be valid.

        A failed CONCURRENTLY build leaves an INVALID index that the planner
        silently ignores, so checking only for presence would pass while every
        lexical search fell back to a sequential scan.
        """
        create_tables(pg_engine)

        assert _lexical_index_exists(pg_engine)
        with pg_engine.connect() as conn:
            valid = conn.execute(
                text(
                    "select indisvalid from pg_index "
                    "where indexrelid = to_regclass(:name)"
                ),
                {"name": LEXICAL_INDEX_NAME},
            ).scalar()
        assert valid is True

    def test_is_idempotent(self, pg_engine):
        create_tables(pg_engine)
        create_tables(pg_engine)

        assert _lexical_index_exists(pg_engine)

    def test_the_planner_actually_uses_it(self, pg_engine):
        """The index is only worth having if the query reaches it.

        The text search configuration in the index and in the query must match;
        when they drift the planner ignores the index without saying so, which
        is invisible until someone times a search on a full corpus.
        """
        create_tables(pg_engine)

        with pg_engine.connect() as conn:
            conn.execute(text("set enable_seqscan = off"))
            plan = "\n".join(
                row[0]
                for row in conn.execute(
                    text(
                        "explain select id from chunks_v2 where "
                        "to_tsvector('english', content) @@ "
                        "plainto_tsquery('english', 'kalman')"
                    )
                ).fetchall()
            )

        assert LEXICAL_INDEX_NAME in plan, plan
