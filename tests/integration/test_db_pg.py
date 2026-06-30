"""PostgreSQL integration tests for db.py: vector extensions and ANN indexes."""

import pytest
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from cementic import vector_store
from cementic.db import (
    Chunk,
    ChunkedDocument,
    ChunkProfile,
    EmbeddingProfile,
    ExtractedDocument,
    ExtractorProfile,
    SourceDocument,
    ensure_embedding_ann_index,
    ensure_vector_extensions,
)
from cementic.index_strategies import IndexParams


def _create_minimal_pipeline(session, *, embedding_dim: int = 4) -> tuple[int, int]:
    """Create minimal DB state: SourceDoc → ExtractedDoc → ChunkedDoc → Chunk.

    Returns (chunk_id, embedding_profile_id) for use in ChunkEmbedding.
    """
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
        page_start=1,
        page_end=1,
    )
    session.add(chunk)
    session.flush()

    return chunk.id, emb_prof.id


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


@pytest.mark.pg
class TestAnnIndex:
    """Build each ANN index method on a real per-profile vector table and search it."""

    @pytest.mark.parametrize("method", ["hnsw", "diskann"])
    def test_build_index_and_search(self, pg_engine, method):
        session_factory = sessionmaker(bind=pg_engine)
        with session_factory() as session:
            chunk_id, emb_prof_id = _create_minimal_pipeline(session, embedding_dim=4)
            session.commit()

        try:
            vector_store.ensure_vector_table(pg_engine, emb_prof_id, 4)
            with pg_engine.begin() as conn:
                vector_store.upsert_vectors(conn, emb_prof_id, [(chunk_id, [0.1, 0.2, 0.3, 0.4])])

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
            chunk_id, emb_prof_id = _create_minimal_pipeline(session, embedding_dim=4)
            session.commit()

        try:
            vector_store.ensure_vector_table(pg_engine, emb_prof_id, 4)
            with pg_engine.begin() as conn:
                vector_store.upsert_vectors(conn, emb_prof_id, [(chunk_id, [0.5, 0.6, 0.7, 0.8])])
            ensure_embedding_ann_index(
                pg_engine, profile_id=emb_prof_id, method="hnsw", params=IndexParams()
            )
            ensure_embedding_ann_index(
                pg_engine, profile_id=emb_prof_id, method="hnsw", params=IndexParams()
            )  # should not raise
        finally:
            _cleanup(pg_engine, emb_prof_id)

    def test_switch_index_method_rebuilds(self, pg_engine):
        """Switching index.method must rebuild the index, not silently keep the old one."""
        session_factory = sessionmaker(bind=pg_engine)
        with session_factory() as session:
            chunk_id, emb_prof_id = _create_minimal_pipeline(session, embedding_dim=4)
            session.commit()

        index_name = vector_store.vector_index_name(emb_prof_id)
        try:
            vector_store.ensure_vector_table(pg_engine, emb_prof_id, 4)
            with pg_engine.begin() as conn:
                vector_store.upsert_vectors(conn, emb_prof_id, [(chunk_id, [0.1, 0.2, 0.3, 0.4])])

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
