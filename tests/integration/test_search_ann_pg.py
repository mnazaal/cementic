"""The ANN index must actually be used, and must still return top_k rows.

Before the filter columns moved onto the vector table, cementic's search query
filtered on joined tables and the planner never chose the ANN index: measured at
100k rows and 768 dimensions, 407ms of exact scan versus about 1ms once the
filters sat on the vector row. Exact, but linear in corpus size.

These tests are small -- the planner's choice, not its speed, is what regressed,
and that is visible with a forced plan on a few thousand rows. The timing
evidence is in README's "Measurements behind the defaults".
"""

from __future__ import annotations

import json
import random

import pytest
from sqlalchemy import insert, text

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
from cementic.search import Searcher
from cementic.vector_store import (
    create_table_sql,
    knn_sql,
    pgvector_version,
    supports_hnsw_iterative_scan,
    upsert_vectors,
    vector_index_name,
    vector_table_name,
)
from tests.integration.test_pg_helpers import _fake_hash, cleanup_pg_tables

DIM = 16
CROWD = 4_000
SLICE_ROWS = 30
TOP_K = 10


def _unit(values) -> list[float]:
    vector = list(values)
    norm = sum(value * value for value in vector) ** 0.5
    return [value / norm for value in vector]


class _FixedClient:
    def health_check(self) -> bool:
        return True

    def format_query(self, value: str) -> str:
        return value

    def format_document(self, value: str) -> str:
        return value

    def embed(self, value: str) -> list[float]:
        return _unit(random.Random(3).gauss(0, 1) for _ in range(DIM))


def _seed_shared_vector_table(session) -> int:
    """Two collections sharing one embedding profile, hence one vector table."""
    rng = random.Random(17)
    chunk_profile = ChunkProfile(config_json="{}", fingerprint="ann-cp")
    embedding_profile = EmbeddingProfile(
        provider="llama-cpp",
        model_identifier="ann-model",
        embedding_dim=DIM,
        distance_metric="cosine",
        config_json=json.dumps(
            {
                "provider": "llama-cpp",
                "model_identifier": "ann-model",
                "embedding_dim": DIM,
                "n_ctx": 512,
            }
        ),
        fingerprint="ann-ep",
    )
    extractor = ExtractorProfile(name="x", config_json="{}", fingerprint="ann-ex")
    session.add_all([chunk_profile, embedding_profile, extractor])
    session.flush()
    session.connection().execute(text(create_table_sql(embedding_profile.id, DIM)))

    for collection, count in (("crowd", CROWD), ("slice", SLICE_ROWS)):
        file_hash, content_hash = _fake_hash(f"f{collection}"), _fake_hash(f"c{collection}")
        source = SourceDocument(
            collection=collection,
            source_path=f"/docs/{collection}.pdf",
            file_hash=file_hash,
            status="done",
        )
        session.add(source)
        session.flush()
        extracted = ExtractedDocument(
            document_id=source.id,
            extractor_profile_id=extractor.id,
            source_file_hash=file_hash,
            content_hash=content_hash,
            status="done",
        )
        session.add(extracted)
        session.flush()
        chunked = ChunkedDocument(
            extracted_document_id=extracted.id,
            chunk_profile_id=chunk_profile.id,
            source_content_hash=content_hash,
            status="done",
            total_chunks=count,
        )
        session.add(chunked)
        session.flush()
        chunk_ids = [
            row[0]
            for row in session.execute(
                insert(Chunk).returning(Chunk.id),
                [
                    {
                        "document_id": source.id,
                        "chunked_document_id": chunked.id,
                        "chunk_index": index,
                        "content": f"{collection} chunk {index}",
                    }
                    for index in range(count)
                ],
            ).all()
        ]
        session.execute(
            insert(ChunkEmbedding),
            [
                {
                    "chunk_id": chunk_id,
                    "embedding_profile_id": embedding_profile.id,
                    "status": "done",
                }
                for chunk_id in chunk_ids
            ],
        )
        upsert_vectors(
            session.connection(),
            embedding_profile.id,
            [(chunk_id, _unit(rng.gauss(0, 1) for _ in range(DIM))) for chunk_id in chunk_ids],
            collection=collection,
            extractor_profile_id=extractor.id,
            chunk_profile_id=chunk_profile.id,
        )
        session.add(
            PipelineRevision(
                collection=collection,
                label=f"rev-{collection}",
                extractor_profile_id=extractor.id,
                chunk_profile_id=chunk_profile.id,
                embedding_profile_id=embedding_profile.id,
                status="active",
            )
        )
    session.flush()
    session.connection().execute(
        text(
            f"CREATE INDEX {vector_index_name(embedding_profile.id)} ON "
            f"{vector_table_name(embedding_profile.id)} "
            "USING hnsw (embedding vector_cosine_ops)"
        )
    )
    session.connection().execute(text(f"ANALYZE {vector_table_name(embedding_profile.id)}"))
    session.commit()
    return embedding_profile.id


@pytest.mark.pg
def test_the_search_query_can_use_the_ann_index(pg_engine, pg_session) -> None:
    """Regression: it could not, at any corpus size or dimensionality.

    Every filter sat on a joined table, so the planner drove from
    `chunked_documents` and probed the vector table by primary key -- reading
    every row and sorting. Asserted on the plan rather than the clock, because
    what regressed is which plan is *available*, and a timing threshold on a
    small fixture would be noise.
    """
    cleanup_pg_tables(pg_session)
    profile_id = _seed_shared_vector_table(pg_session)
    query = "[" + ",".join(
        repr(value) for value in _unit(random.Random(3).gauss(0, 1) for _ in range(DIM))
    ) + "]"

    with pg_engine.connect() as conn:
        conn.execute(text("SET LOCAL hnsw.ef_search = 40"))
        plan = "\n".join(
            row[0]
            for row in conn.execute(
                text("EXPLAIN (COSTS OFF) " + knn_sql(profile_id, distance_operator="<=>")),
                {
                    "query": query,
                    "collection": "crowd",
                    "chunk_profile_id": 1,
                    "extractor_profile_id": 1,
                    "k": TOP_K,
                },
            ).all()
        )

    assert vector_index_name(profile_id) in plan, plan
    cleanup_pg_tables(pg_session)


@pytest.mark.pg
def test_search_returns_a_full_page_through_the_ann_path(
    pg_engine, pg_session, pg_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end over the plan the previous test pins: still top_k, still right.

    Deliberately the dominant collection. There is no test here for the
    thin-slice recall failure that `hnsw.iterative_scan` prevents: reproducing
    it needs roughly 100k rows, because below that the planner picks an exact
    sequential scan for a selective filter -- which returns the right answer and
    would make the test pass for the wrong reason. The measurement (0 of 10
    with the setting off, 10 of 10 with it on) is in README's "Measurements
    behind the defaults".
    """
    cleanup_pg_tables(pg_session)
    _seed_shared_vector_table(pg_session)
    with pg_engine.connect() as conn:
        if not supports_hnsw_iterative_scan(pgvector_version(conn)):
            pytest.skip("pgvector predates iterative scan")

    monkeypatch.setattr(
        "cementic.search._create_embedding_provider",
        lambda config_json, config=None: _FixedClient(),
    )

    results = Searcher(pg_config).search("anything", top_k=TOP_K, collections=["crowd"])

    assert len(results) == TOP_K
    assert {result["collection"] for result in results} == {"crowd"}
    cleanup_pg_tables(pg_session)
