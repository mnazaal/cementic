"""PostgreSQL integration tests for real vector search."""

from __future__ import annotations

import pytest
from sqlalchemy import text

from cementic.db import ExtractedDocument, SourceDocument
from cementic.pipeline_worker import _purge_superseded_chunks
from cementic.revisions import CURRENT_CONTENT_SQL
from cementic.search import Searcher
from cementic.source_watcher import _purge_document_chunks
from cementic.vector_store import FILTER_COLUMNS, vector_table_name
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
def test_search_works_against_a_pre_migration_vector_table(
    pg_engine, pg_session, pg_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upgrading must not break reading until someone happens to write.

    The KNN query names the filter columns, so on a database written by an
    older cementic search would fail with UndefinedColumn until the worker next
    ran and migrated the table. Search migrates it too.
    """
    cleanup_pg_tables(pg_session)
    revision = seed_active_vector_collection(
        pg_session,
        collection="premigration",
        source_path="/docs/old.pdf",
        chunks=[("still findable", [1.0, 0.0, 0.0, 0.0])],
    )
    pg_session.commit()
    table = vector_table_name(revision.embedding_profile_id)
    with pg_engine.begin() as conn:
        for column in FILTER_COLUMNS:
            conn.execute(text(f"ALTER TABLE {table} DROP COLUMN {column}"))

    monkeypatch.setattr(
        "cementic.search._create_embedding_provider",
        lambda config_json, config=None: FakeSearchEmbeddingClient(),
    )

    results = Searcher(pg_config).search("neural", top_k=5, collections=["premigration"])

    assert [result["content"] for result in results] == ["still findable"]
    cleanup_pg_tables(pg_session)
