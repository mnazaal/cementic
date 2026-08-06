"""PostgreSQL integration tests for real vector search."""

from __future__ import annotations

import pytest

from cementic.search import Searcher
from tests.integration.test_pg_helpers import (
    cleanup_pg_tables,
    seed_active_vector_collection,
    seed_two_extractor_profiles_sharing_a_vector_table,
)


class FakeSearchEmbeddingClient:
    """Deterministic query embedding client for PG search tests."""

    def health_check(self) -> bool:
        return True

    def format_query(self, text: str) -> str:
        return text

    def format_document(self, text: str) -> str:
        return text

    def embed(self, text: str) -> list[float]:
        lowered = text.lower()
        if "biology" in lowered or "cell" in lowered:
            return [0.0, 1.0, 0.0, 0.0]
        if "history" in lowered or "rome" in lowered:
            return [0.0, 0.0, 1.0, 0.0]
        return [1.0, 0.0, 0.0, 0.0]


@pytest.mark.pg
def test_pg_search_cosine_ranking_and_topic_separation(
    pg_engine, pg_session, pg_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    cleanup_pg_tables(pg_session)
    seed_active_vector_collection(
        pg_session,
        collection="research",
        source_path="/docs/topics.pdf",
        chunks=[
            ("neural networks and semantic vectors", [1.0, 0.0, 0.0, 0.0]),
            ("cell biology and proteins", [0.0, 1.0, 0.0, 0.0]),
        ],
    )
    pg_session.commit()
    monkeypatch.setattr(
        "cementic.search._create_embedding_provider",
        lambda config_json, config=None: FakeSearchEmbeddingClient(),
    )

    results = Searcher(pg_config).search("neural vector search", top_k=2)

    assert len(results) == 2
    assert "neural networks" in results[0]["content"]
    assert results[0]["score"] > results[1]["score"]

    cleanup_pg_tables(pg_session)


@pytest.mark.pg
def test_pg_search_collection_filter_isolates_results(
    pg_engine, pg_session, pg_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    cleanup_pg_tables(pg_session)
    seed_active_vector_collection(
        pg_session,
        collection="cs",
        source_path="/docs/cs.pdf",
        chunks=[("neural networks", [1.0, 0.0, 0.0, 0.0])],
    )
    seed_active_vector_collection(
        pg_session,
        collection="bio",
        source_path="/docs/bio.pdf",
        chunks=[("cell biology", [0.0, 1.0, 0.0, 0.0])],
    )
    pg_session.commit()
    monkeypatch.setattr(
        "cementic.search._create_embedding_provider",
        lambda config_json, config=None: FakeSearchEmbeddingClient(),
    )

    results = Searcher(pg_config).search("neural vector search", top_k=5, collections=["bio"])

    assert [result["collection"] for result in results] == ["bio"]
    assert "cell biology" in results[0]["content"]

    cleanup_pg_tables(pg_session)


@pytest.mark.pg
def test_pg_search_excludes_a_retired_extractor_profile(
    pg_engine, pg_session, pg_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Search must not return superseded extractions of the same document.

    Regression: the per-profile vector table is keyed by *embedding* profile, so
    it also holds vectors from other revisions using that model. Restricting to
    the chunk profile was not enough -- a revision whose extractor changed
    reuses the same chunk and embedding profiles, and pruning deliberately keeps
    the most recent retired revision. Every document therefore came back twice,
    once from the retired extraction and once from the active one, spending half
    of top_k on superseded text.
    """
    cleanup_pg_tables(pg_session)
    revision = seed_two_extractor_profiles_sharing_a_vector_table(
        pg_session,
        collection="extractorchange",
        source_path="/docs/paper.pdf",
        old_content="superseded extraction text",
        new_content="current extraction text",
        vector=[1.0, 0.0, 0.0, 0.0],
    )
    pg_session.commit()
    monkeypatch.setattr(
        "cementic.search._create_embedding_provider",
        lambda config_json, config=None: FakeSearchEmbeddingClient(),
    )

    results = Searcher(pg_config).search(
        "neural networks", top_k=10, collections=["extractorchange"]
    )

    contents = [result["content"] for result in results]
    assert contents == ["current extraction text"]
    assert revision.status == "active"
    cleanup_pg_tables(pg_session)
