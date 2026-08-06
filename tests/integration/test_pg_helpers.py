"""Shared helpers for PostgreSQL integration tests."""

from __future__ import annotations

import json

from sqlalchemy import text
from sqlalchemy.orm import Session

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
from cementic.vector_store import create_table_sql, upsert_vectors

VECTOR_DIM = 4


def seed_active_vector_collection(
    session: Session,
    *,
    collection: str,
    source_path: str,
    chunks: list[tuple[str, list[float]]],
    status: str = "active",
) -> PipelineRevision:
    """Create one searchable collection with active revision and done embeddings."""
    suffix = f"{collection}-{source_path}"
    source = SourceDocument(
        collection=collection,
        source_path=source_path,
        file_hash=f"hash-{suffix}",
        status="done",
    )
    extractor = ExtractorProfile(
        name="test-extractor",
        config_json="{}",
        fingerprint=f"extractor-{suffix}",
    )
    chunk_profile = ChunkProfile(config_json="{}", fingerprint=f"chunk-{suffix}")
    embedding_profile = EmbeddingProfile(
        provider="llama-cpp",
        model_identifier="test-model",
        embedding_dim=VECTOR_DIM,
        distance_metric="cosine",
        config_json=json.dumps(
            {
                "provider": "llama-cpp",
                "model_identifier": "test-model",
                "embedding_dim": VECTOR_DIM,
                "n_ctx": 512,
                "n_gpu_layers": 0,
                "verbose": False,
            }
        ),
        fingerprint=f"embedding-{suffix}",
    )
    session.add_all([source, extractor, chunk_profile, embedding_profile])
    session.flush()

    extracted = ExtractedDocument(
        document_id=source.id,
        extractor_profile_id=extractor.id,
        artifact_path=f"/tmp/{suffix}.json",
        status="done",
    )
    session.add(extracted)
    session.flush()

    chunked = ChunkedDocument(
        extracted_document_id=extracted.id,
        chunk_profile_id=chunk_profile.id,
        status="done",
        total_chunks=len(chunks),
    )
    session.add(chunked)
    session.flush()

    vector_rows: list[tuple[int, list[float]]] = []
    for index, (content, vector) in enumerate(chunks):
        chunk = Chunk(
            document_id=source.id,
            chunked_document_id=chunked.id,
            chunk_index=index,
            content=content,
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
        vector_rows.append((chunk.id, vector))

    # Vectors live in the per-profile table, not on ChunkEmbedding.
    conn = session.connection()
    conn.execute(text(create_table_sql(embedding_profile.id, VECTOR_DIM)))
    upsert_vectors(conn, embedding_profile.id, vector_rows)

    revision = PipelineRevision(
        collection=collection,
        label=f"revision-{collection}",
        extractor_profile_id=extractor.id,
        chunk_profile_id=chunk_profile.id,
        embedding_profile_id=embedding_profile.id,
        status=status,
    )
    session.add(revision)
    session.flush()
    return revision


def cleanup_pg_tables(session: Session) -> None:
    """Remove rows from all pipeline tables between PG integration tests."""
    session.execute(
        text(
            "TRUNCATE chunk_embeddings, chunks_v2, chunked_documents, "
            "extracted_documents, source_documents, pipeline_revisions, "
            "embedding_profiles, extractor_profiles, chunk_profiles RESTART IDENTITY CASCADE"
        )
    )
    # Drop the dynamic per-profile vector tables too.
    vector_tables = session.execute(
        text("SELECT tablename FROM pg_tables WHERE tablename LIKE 'embedding_vectors_p%'")
    ).all()
    for (table_name,) in vector_tables:
        session.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
    session.commit()
