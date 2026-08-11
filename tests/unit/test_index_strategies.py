"""Tests for the pure ANN index-DDL builders."""

import pytest

from cementic.index_strategies import (
    IndexParams,
    build_index_ddl,
    index_dimension_error,
    max_indexable_dim,
    supported_index_methods,
)


def test_hnsw_ddl() -> None:
    sql = build_index_ddl(
        method="hnsw",
        index_name="ix_t",
        table="embedding_vectors_p1",
        column="embedding",
        metric="cosine",
        params=IndexParams(hnsw_m=32, hnsw_ef_construction=128),
    )
    assert sql.startswith("CREATE INDEX IF NOT EXISTS ix_t ON embedding_vectors_p1 ")
    assert "USING hnsw (embedding vector_cosine_ops)" in sql
    assert "m = 32" in sql
    assert "ef_construction = 128" in sql


def test_diskann_ddl() -> None:
    sql = build_index_ddl(
        method="diskann",
        index_name="ix_t",
        table="embedding_vectors_p1",
        column="embedding",
        metric="cosine",
        params=IndexParams(diskann_num_neighbors=64, diskann_search_list_size=200),
    )
    assert "USING diskann (embedding vector_cosine_ops)" in sql
    assert "num_neighbors = 64" in sql
    assert "search_list_size = 200" in sql


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="Unknown index method"):
        build_index_ddl(
            method="nope",
            index_name="ix",
            table="t",
            column="embedding",
            metric="cosine",
            params=IndexParams(),
        )


def test_unknown_metric_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported distance metric"):
        build_index_ddl(
            method="hnsw",
            index_name="ix",
            table="t",
            column="embedding",
            metric="weird",
            params=IndexParams(),
        )


def test_supported_methods() -> None:
    assert set(supported_index_methods()) == {"hnsw", "diskann"}


class TestDimensionLimits:
    """pgvector's HNSW index stops at 2000 dimensions while the column holds far
    more, so an oversized model used to embed the entire corpus and only then
    fail at CREATE INDEX -- looping on that failure with the work already done."""

    def test_hnsw_refuses_a_dimension_it_cannot_index(self):
        message = index_dimension_error("hnsw", 2560)

        assert message is not None
        assert "2000" in message and "2560" in message
        assert "diskann" in message

    def test_hnsw_accepts_a_dimension_at_the_limit(self):
        assert index_dimension_error("hnsw", 2000) is None

    def test_diskann_has_no_dimension_limit(self):
        assert index_dimension_error("diskann", 4096) is None
        assert max_indexable_dim("diskann") is None

    def test_common_models_are_accepted(self):
        """768 (nomic) and 1024 (bge-m3) must keep working."""
        assert index_dimension_error("hnsw", 768) is None
        assert index_dimension_error("hnsw", 1024) is None
