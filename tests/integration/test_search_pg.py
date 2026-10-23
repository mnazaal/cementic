"""PostgreSQL integration tests for real vector search."""

from __future__ import annotations

import pytest
from sqlalchemy import text

from cementic.db import ExtractedDocument, SourceDocument
from cementic.pipeline_worker import _purge_superseded_chunks
from cementic.search import Searcher, split_arms
from cementic.source_watcher import _purge_document_chunks
from tests.integration.pg_helpers import (
    CURRENT_CONTENT_SQL,
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

    searcher = Searcher(pg_config)
    # Both chunks are one document, which a search returns as one row; the
    # ranking between them is visible on the vector arm's chunk rows.
    vector_arm, _ = split_arms(searcher.candidates("neural vector search", top_k=2))
    results = searcher.search("neural vector search", top_k=2)

    assert len(vector_arm) == 2
    assert "neural networks" in vector_arm[0]["content"]
    assert vector_arm[0]["score"] > vector_arm[1]["score"]
    assert len(results) == 1
    assert "neural networks" in results[0]["content"]  # the best chunk represents it

    cleanup_pg_tables(pg_session)


@pytest.mark.pg
@pytest.mark.parametrize("hybrid", [False, True])
def test_pg_search_results_carry_the_document_id(
    pg_engine, pg_session, pg_config, monkeypatch: pytest.MonkeyPatch, hybrid: bool
) -> None:
    """Both arms project `source_documents.id`, so a result names its document."""
    cleanup_pg_tables(pg_session)
    seed_active_vector_collection(
        pg_session,
        collection="ids",
        source_path="/docs/ids.pdf",
        chunks=[
            ("neural networks and semantic vectors", [1.0, 0.0, 0.0, 0.0]),
            ("cell biology and proteins", [0.0, 1.0, 0.0, 0.0]),
        ],
    )
    pg_session.commit()
    document_id = pg_session.execute(
        text("SELECT id FROM source_documents WHERE collection = 'ids'")
    ).scalar_one()
    monkeypatch.setattr(
        "cementic.search._create_embedding_provider",
        lambda config_json, config=None: FakeSearchEmbeddingClient(),
    )
    pg_config.search.hybrid = hybrid

    searcher = Searcher(pg_config)
    # The candidate pool is where each arm's SQL surfaces: assert on it so a
    # lexical row is covered even when fusion would show the vector row.
    pool = searcher.candidates("proteins", top_k=2)
    results = searcher.search("proteins", top_k=2)

    assert pool and results
    assert {row["document_id"] for row in pool} == {document_id}
    # Every path merges to one row per document, so the two chunks give one row.
    assert [row["document_id"] for row in results] == [document_id]
    if hybrid:
        assert any(row["score_kind"] == "lexical_rank" for row in pool)

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


@pytest.mark.pg
def test_the_seed_helper_produces_rows_the_pipeline_could_produce(pg_session) -> None:
    """The fixture must satisfy the same freshness invariant the worker maintains.

    Regression: the helper left `source_file_hash`, `content_hash` and
    `source_content_hash` NULL. Once search began requiring
    `ed.source_file_hash = sd.file_hash` -- NULL fails it, since NULL = NULL is
    not true -- every seeded corpus became unfindable, and three PG tests went
    red on main while the unit and non-PG jobs stayed green.

    Asserted against the production predicate itself, so the fixture cannot
    drift away from it again without this failing.
    """
    cleanup_pg_tables(pg_session)
    seed_active_vector_collection(
        pg_session,
        collection="invariant",
        source_path="/docs/invariant.pdf",
        chunks=[("some text", [1.0, 0.0, 0.0, 0.0])],
    )
    pg_session.commit()

    # `IS NOT TRUE`, not `NOT (...)`: with a NULL hash the predicate evaluates
    # to NULL, and `NOT NULL` is NULL rather than true -- so the obvious spelling
    # counts nothing and passes against exactly the fixture it exists to reject.
    stale = pg_session.execute(
        text(
            "SELECT count(*) FROM extracted_documents ed "
            "JOIN source_documents sd ON ed.document_id = sd.id "
            "JOIN chunked_documents cd ON cd.extracted_document_id = ed.id "
            "WHERE sd.collection = 'invariant' "
            "AND (" + CURRENT_CONTENT_SQL + ") IS NOT TRUE"
        )
    ).scalar()

    assert stale == 0, "seeded rows do not satisfy the freshness predicate search requires"
    cleanup_pg_tables(pg_session)


@pytest.mark.pg
def test_search_does_not_return_a_deleted_document(
    pg_engine, pg_session, pg_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guarantee that used to come from `sd.status <> 'deleted'` in the query.

    That filter lived on a joined table, which is why the planner could never
    use the ANN index. It is now enforced by deleting the document's chunks,
    whose cascade clears the vectors -- so this test pins the behaviour rather
    than the mechanism, and would fail if the purge were dropped.
    """
    cleanup_pg_tables(pg_session)
    seed_active_vector_collection(
        pg_session,
        collection="deleted",
        source_path="/docs/gone.pdf",
        chunks=[("content that should vanish", [1.0, 0.0, 0.0, 0.0])],
    )
    pg_session.commit()
    monkeypatch.setattr(
        "cementic.search._create_embedding_provider",
        lambda config_json, config=None: FakeSearchEmbeddingClient(),
    )
    assert Searcher(pg_config).search("neural", top_k=5, collections=["deleted"])

    document = pg_session.query(SourceDocument).filter_by(collection="deleted").one()
    document.status = "deleted"
    document.file_hash = None
    _purge_document_chunks(pg_session, [document.id])
    pg_session.commit()

    assert Searcher(pg_config).search("neural", top_k=5, collections=["deleted"]) == []
    cleanup_pg_tables(pg_session)


@pytest.mark.pg
def test_search_does_not_return_superseded_content(
    pg_engine, pg_session, pg_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guarantee that used to come from the freshness join.

    A file whose re-extraction produced different text must stop returning its
    previous contents even before re-chunking runs -- otherwise a stopped worker
    or a failed chunking serves stale text indefinitely.
    """
    cleanup_pg_tables(pg_session)
    seed_active_vector_collection(
        pg_session,
        collection="superseded",
        source_path="/docs/edited.pdf",
        chunks=[("the old text", [1.0, 0.0, 0.0, 0.0])],
    )
    pg_session.commit()
    monkeypatch.setattr(
        "cementic.search._create_embedding_provider",
        lambda config_json, config=None: FakeSearchEmbeddingClient(),
    )
    assert Searcher(pg_config).search("neural", top_k=5, collections=["superseded"])

    extracted = pg_session.query(ExtractedDocument).one()
    _purge_superseded_chunks(pg_session, extracted.id, "a-different-content-hash")
    pg_session.commit()

    assert Searcher(pg_config).search("neural", top_k=5, collections=["superseded"]) == []
    cleanup_pg_tables(pg_session)


@pytest.mark.pg
def test_the_rarity_probe_counts_only_the_searched_collection(
    pg_engine, pg_session, pg_config
) -> None:
    """A token rare in the searched collection leads, however common elsewhere."""
    cleanup_pg_tables(pg_session)
    threshold = pg_config.search.lexical_lead_max_documents
    seed_active_vector_collection(
        pg_session,
        collection="narrow",
        source_path="/docs/only.pdf",
        chunks=[("the zyxqterm appears once here", [1.0, 0.0, 0.0, 0.0])],
    )
    for i in range(threshold + 1):
        seed_active_vector_collection(
            pg_session,
            collection=f"broad{i}",
            source_path=f"/docs/broad{i}.pdf",
            chunks=[("the zyxqterm appears everywhere", [1.0, 0.0, 0.0, 0.0])],
        )
    pg_session.commit()
    searcher = Searcher(pg_config)

    assert searcher.lexical_should_lead("zyxqterm", ["narrow"])
    assert not searcher.lexical_should_lead("zyxqterm")
    cleanup_pg_tables(pg_session)


@pytest.mark.pg
def test_a_full_rarity_sample_does_not_claim_the_term_is_rare(
    pg_engine, pg_session, pg_config
) -> None:
    """Past the probe's cap the sample is unscoped, so its documents all count."""
    cleanup_pg_tables(pg_session)
    pg_config.search.lexical_lead_max_documents = 1
    cap = 20  # the probe fetches threshold * 20 chunks
    seed_active_vector_collection(
        pg_session,
        collection="narrow",
        source_path="/docs/only.pdf",
        chunks=[("the zyxqterm appears once here", [1.0, 0.0, 0.0, 0.0])],
    )
    for i in range(cap + 5):
        seed_active_vector_collection(
            pg_session,
            collection=f"broad{i}",
            source_path=f"/docs/broad{i}.pdf",
            chunks=[("the zyxqterm appears everywhere", [1.0, 0.0, 0.0, 0.0])],
        )
    pg_session.commit()

    assert not Searcher(pg_config).lexical_should_lead("zyxqterm", ["narrow"])
    cleanup_pg_tables(pg_session)


@pytest.mark.pg
def test_a_typo_reaches_the_lexical_arm_through_the_fallback(
    pg_engine, pg_session, pg_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No chunk holds every term of a typo'd query; most of them still match."""
    cleanup_pg_tables(pg_session)
    seed_active_vector_collection(
        pg_session,
        collection="typo",
        source_path="/docs/typo.pdf",
        chunks=[("neural networks and semantic vectors", [1.0, 0.0, 0.0, 0.0])],
    )
    pg_session.commit()
    monkeypatch.setattr(
        "cementic.search._create_embedding_provider",
        lambda config_json, config=None: FakeSearchEmbeddingClient(),
    )
    pg_config.search.hybrid = True
    searcher = Searcher(pg_config)

    _, lexical = split_arms(searcher.candidates("nueral networks semantic", top_k=2))
    assert [row["source_path"] for row in lexical] == ["/docs/typo.pdf"]

    # An exact match never reaches the fallback.
    monkeypatch.setattr(
        searcher, "_typo_fallback", lambda *a: pytest.fail("fallback ran on an exact match")
    )
    _, lexical = split_arms(searcher.candidates("neural networks semantic", top_k=2))
    assert lexical
    cleanup_pg_tables(pg_session)
