"""Tests for the pure SQL builders in vector_store."""

import pytest

from cementic.vector_store import (
    create_table_sql,
    drop_table_sql,
    parse_extension_version,
    query_tuning_statements,
    supports_hnsw_iterative_scan,
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
    assert (
        "INSERT INTO embedding_vectors_p1 "
        "(chunk_id, embedding, collection, extractor_profile_id, chunk_profile_id)"
    ) in sql
    assert "(:embedding)::vector" in sql
    assert "ON CONFLICT (chunk_id) DO UPDATE" in sql


def test_query_tuning_statements() -> None:
    assert query_tuning_statements("hnsw", hnsw_ef_search=80, diskann_query_rescore=50) == [
        "SET LOCAL hnsw.ef_search = 80"
    ]
    assert query_tuning_statements("diskann", hnsw_ef_search=40, diskann_query_rescore=120) == [
        "SET LOCAL diskann.query_rescore = 120"
    ]
    assert query_tuning_statements("other", hnsw_ef_search=40, diskann_query_rescore=50) == []


def test_hnsw_ef_search_is_never_below_the_requested_result_count() -> None:
    """An HNSW scan yields at most ef_search candidates, so a configured value
    under top_k silently caps the result count: the default 40 sits below the
    documented maximum of 50 results."""
    assert query_tuning_statements(
        "hnsw", hnsw_ef_search=40, diskann_query_rescore=50, top_k=50
    ) == ["SET LOCAL hnsw.ef_search = 50"]


def test_hnsw_ef_search_keeps_a_configured_value_above_the_result_count() -> None:
    assert query_tuning_statements(
        "hnsw", hnsw_ef_search=200, diskann_query_rescore=50, top_k=10
    ) == ["SET LOCAL hnsw.ef_search = 200"]


class TestIterativeScanGating:
    """The gate is what keeps an old pgvector from losing all search.

    PostgreSQL accepts an unknown *qualified* setting as a placeholder until the
    defining module loads on that connection, then rejects it with InvalidName.
    Behind a connection pool that makes an ungated `SET LOCAL
    hnsw.iterative_scan` fail on exactly those connections that had already run
    a vector query -- intermittently, not honestly.
    """

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("0.8.3", (0, 8, 3)),
            ("0.7.4", (0, 7, 4)),
            ("1.0", (1, 0)),
            ("0.8.0-rc1", (0, 8)),
            ("", None),
            (None, None),
            ("unknown", None),
        ],
    )
    def test_version_parsing(self, raw, expected) -> None:
        assert parse_extension_version(raw) == expected

    @pytest.mark.parametrize(
        "version,supported",
        [((0, 8, 3), True), ((0, 8, 0), True), ((0, 7, 4), False), ((1, 2), True), (None, False)],
    )
    def test_support_threshold(self, version, supported) -> None:
        assert supports_hnsw_iterative_scan(version) is supported

    def test_an_unsupported_server_gets_only_ef_search(self) -> None:
        assert query_tuning_statements(
            "hnsw", hnsw_ef_search=40, diskann_query_rescore=50, hnsw_iterative_scan=None
        ) == ["SET LOCAL hnsw.ef_search = 40"]

    def test_a_supported_server_also_gets_the_scan_mode(self) -> None:
        assert query_tuning_statements(
            "hnsw",
            hnsw_ef_search=40,
            diskann_query_rescore=50,
            hnsw_iterative_scan="relaxed_order",
        ) == [
            "SET LOCAL hnsw.ef_search = 40",
            "SET LOCAL hnsw.iterative_scan = relaxed_order",
        ]

    def test_an_unrecognised_mode_is_dropped_rather_than_sent(self) -> None:
        """An invalid value aborts the SET, and with it the query it was tuning."""
        assert query_tuning_statements(
            "hnsw",
            hnsw_ef_search=40,
            diskann_query_rescore=50,
            hnsw_iterative_scan="'; DROP TABLE chunks_v2; --",
        ) == ["SET LOCAL hnsw.ef_search = 40"]
