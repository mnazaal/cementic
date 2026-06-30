"""PostgreSQL integration tests for revision ANN indexes."""

from __future__ import annotations

import pytest
from sqlalchemy import text

from cementic.config import Config
from cementic.revisions import ensure_revision_ann_index
from cementic.vector_store import vector_index_name
from tests.integration.test_pg_helpers import cleanup_pg_tables, seed_active_vector_collection


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
