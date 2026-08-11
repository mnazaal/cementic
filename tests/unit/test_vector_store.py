"""Tests for the pure SQL builders in vector_store."""

import pytest

from cementic.vector_store import (
    create_table_sql,
    drop_table_sql,
    query_tuning_sql,
    to_vector_literal,
    upsert_sql,
    vector_index_name,
    vector_table_name,
)


def test_table_and_index_names() -> None:
    assert vector_table_name(3) == "embedding_vectors_p3"
    assert vector_index_name(3) == "ix_embedding_vectors_p3_ann"


@pytest.mark.parametrize("bad", [-1, True, "1"])
def test_profile_id_validation(bad) -> None:
    with pytest.raises(ValueError, match="profile_id"):
        vector_table_name(bad)


def test_to_vector_literal() -> None:
    assert to_vector_literal([1, 2, 3]) == "[1.0,2.0,3.0]"


def test_create_table_sql() -> None:
    sql = create_table_sql(5, 768)
    assert "CREATE TABLE IF NOT EXISTS embedding_vectors_p5" in sql
    assert "embedding vector(768) NOT NULL" in sql
    assert "REFERENCES chunks_v2(id) ON DELETE CASCADE" in sql


@pytest.mark.parametrize("bad", [0, -4, 9000, True])
def test_create_table_sql_rejects_bad_dim(bad) -> None:
    with pytest.raises(ValueError, match="dim"):
        create_table_sql(1, bad)


def test_drop_table_sql() -> None:
    assert drop_table_sql(2) == "DROP TABLE IF EXISTS embedding_vectors_p2"


def test_upsert_sql() -> None:
    sql = upsert_sql(1)
    assert "INSERT INTO embedding_vectors_p1 (chunk_id, embedding)" in sql
    assert "(:embedding)::vector" in sql
    assert "ON CONFLICT (chunk_id) DO UPDATE" in sql


def test_query_tuning_sql() -> None:
    assert query_tuning_sql("hnsw", hnsw_ef_search=80, diskann_query_rescore=50) == (
        "SET LOCAL hnsw.ef_search = 80"
    )
    assert query_tuning_sql("diskann", hnsw_ef_search=40, diskann_query_rescore=120) == (
        "SET LOCAL diskann.query_rescore = 120"
    )
    assert query_tuning_sql("other", hnsw_ef_search=40, diskann_query_rescore=50) is None


def test_hnsw_ef_search_is_never_below_the_requested_result_count() -> None:
    """An HNSW scan yields at most ef_search candidates, so a configured value
    under top_k silently caps the result count: the default 40 sits below the
    documented maximum of 50 results."""
    assert query_tuning_sql(
        "hnsw", hnsw_ef_search=40, diskann_query_rescore=50, top_k=50
    ) == "SET LOCAL hnsw.ef_search = 50"


def test_hnsw_ef_search_keeps_a_configured_value_above_the_result_count() -> None:
    assert query_tuning_sql(
        "hnsw", hnsw_ef_search=200, diskann_query_rescore=50, top_k=10
    ) == "SET LOCAL hnsw.ef_search = 200"
