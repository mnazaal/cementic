"""Tests for revision-aware search functionality."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cementic.config import Config
from cementic.db import Base, PipelineRevision
from cementic.search import (
    Searcher,
    SearchResult,
    _create_embedding_provider,
    _distance_operator,
    _reject_query_over_context,
    _score_from_distance,
    _searchable_revisions,
)


class TestRevisionChoiceAgreesWithAndWithoutDashC:
    """Regression (independently derived twice): naming a collection with -c
    took the newest in-flight revision by id, so a newer `building` revision
    out-ranked a `ready` one -- while the same collection searched without -c
    ranked active > ready > building. Same data, different revision, silently
    different results depending only on how the collection was named."""

    def _searcher_and_session(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session = sessionmaker(bind=engine, expire_on_commit=False)()
        with (
            patch("cementic.search.get_engine"),
            patch("cementic.search.get_session_factory"),
        ):
            searcher = Searcher(config=Config())
        return searcher, session

    _next_chunk_profile = iter(range(1, 100))

    def _revision(self, collection: str, status: str) -> PipelineRevision:
        # sqlite does not enforce the profile FKs, which are irrelevant here;
        # chunk_profile_id varies because (collection, profile triple) is
        # unique -- as in reality, where a new revision means a changed profile.
        return PipelineRevision(
            collection=collection,
            extractor_profile_id=1,
            chunk_profile_id=next(self._next_chunk_profile),
            embedding_profile_id=1,
            status=status,
            label=f"{collection}-{status}",
        )

    def test_named_collection_prefers_ready_over_newer_building(self):
        searcher, session = self._searcher_and_session()
        session.add(self._revision("papers", "ready"))
        session.add(self._revision("papers", "building"))  # newer id
        session.commit()

        named = searcher._load_searchable_revisions(session, ["papers"])
        unnamed = searcher._load_searchable_revisions(session, None)

        assert [revision.status for revision in named] == ["ready"]
        assert [(revision.collection, revision.status) for revision in named] == [
            (revision.collection, revision.status) for revision in unnamed
        ]

    def test_named_collection_still_falls_back_to_building_only(self):
        searcher, session = self._searcher_and_session()
        session.add(self._revision("papers", "building"))
        session.commit()

        named = searcher._load_searchable_revisions(session, ["papers"])

        assert [revision.status for revision in named] == ["building"]

    def test_named_collection_prefers_active_over_everything(self):
        searcher, session = self._searcher_and_session()
        session.add(self._revision("papers", "active"))
        session.add(self._revision("papers", "ready"))
        session.add(self._revision("papers", "building"))
        session.commit()

        named = searcher._load_searchable_revisions(session, ["papers"])

        assert [revision.status for revision in named] == ["active"]


class TestSearchActivityLease:
    """The search-side half of query-first embed scheduling."""

    def _session_factory(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        return sessionmaker(bind=engine, expire_on_commit=False)

    def test_record_search_activity_upserts_the_single_lease_row(self):
        from cementic.db import SearchActivity
        from cementic.search import record_search_activity

        session_factory = self._session_factory()
        record_search_activity(session_factory)
        with session_factory() as session:
            first = session.get(SearchActivity, 1).last_search_at
        record_search_activity(session_factory)
        with session_factory() as session:
            rows = session.query(SearchActivity).all()
            assert len(rows) == 1
            assert rows[0].last_search_at >= first

    def test_record_search_activity_tolerates_a_database_without_the_table(self):
        """The lease is best-effort: search must not fail on a pre-lease schema.

        The table is created by the worker paths (create_tables); a search
        against a database whose worker predates the lease must still search.
        """
        from cementic.db import SearchActivity
        from cementic.search import record_search_activity

        session_factory = self._session_factory()
        SearchActivity.__table__.drop(session_factory.kw["bind"])
        record_search_activity(session_factory)  # must not raise


class TestSearcher:
    """Test search functionality."""

    class RevisionQuery:
        def __init__(self, rows):
            self.rows = rows

        def filter(self, *args, **kwargs):
            return self

        def filter_by(self, **kwargs):
            return self

        def order_by(self, *args):
            return self

        def first(self):
            return self.rows[0] if self.rows else None

        def all(self):
            return self.rows

    @patch("cementic.search.get_engine")
    @patch("cementic.search.get_session_factory")
    def test_search_returns_empty_without_searchable_revisions(
        self, mock_session_factory, mock_get_engine
    ):
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = False
        mock_session.query.return_value = self.RevisionQuery([])
        mock_session_factory.return_value = lambda: mock_session

        searcher = Searcher()
        assert searcher.search("test") == []

    @patch("cementic.search.get_engine")
    @patch("cementic.search.get_session_factory")
    @patch("cementic.search._create_embedding_provider")
    def test_search_uses_building_revision_when_no_active_exists(
        self, mock_create_embedding_provider, mock_session_factory, mock_get_engine
    ):
        building_revision = SimpleNamespace(
            collection="papers",
            status="building",
            embedding_profile_id=1,
            chunk_profile_id=2,
            extractor_profile_id=1,
            embedding_profile=SimpleNamespace(
                config_json=(
                    '{"provider": "llama-cpp", '
                    '"model_identifier": "nomic-embed-text", "embedding_dim": 768}'
                ),
                model_identifier="nomic-embed-text",
                embedding_dim=768,
                distance_metric="cosine",
            ),
        )
        revision_query = self.RevisionQuery([building_revision])

        row = SimpleNamespace(
            collection="papers",
            source_path="/tmp/papers.pdf",
            content="partial chunk text",
            distance=0.1,
        )
        exec_result = MagicMock()
        exec_result.scalar.return_value = 1  # vector table exists
        exec_result.__iter__.return_value = iter([row])

        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = False
        mock_session.query.side_effect = [revision_query]
        mock_session.execute.return_value = exec_result
        mock_session_factory.return_value = lambda: mock_session

        embedding_provider = MagicMock()
        embedding_provider.health_check.return_value = True
        embedding_provider.embed.return_value = [0.1] * 768
        mock_create_embedding_provider.return_value = embedding_provider

        searcher = Searcher()
        results = searcher.search("hello", collections=["papers"])

        assert results[0]["source_path"] == "/tmp/papers.pdf"
        embedding_provider.embed.assert_called_once()

    # (A test simulating the -c path's former two-query shape -- active first,
    # then a separate in-flight fallback -- was removed when revision choice was
    # unified through _searchable_revisions; the fallback-to-building behaviour
    # it pinned is covered by test_search_uses_building_revision_when_no_active_exists
    # and TestRevisionChoiceAgreesWithAndWithoutDashC.)

    @patch("cementic.search.get_engine")
    @patch("cementic.search.get_session_factory")
    def test_search_rejects_mixed_active_models(self, mock_session_factory, mock_get_engine):
        """The refusal must name the collections, their models and their statuses.

        It used to say only "different active embedding models; search them
        separately" -- naming no collection, so finding the odd one out meant
        reading `collection list` and comparing revision labels by hand. It also
        said "active" about revisions that were merely `building` or `ready`.
        """
        revision_query = self.RevisionQuery(
            [
                SimpleNamespace(
                    collection="a",
                    status="active",
                    embedding_profile_id=1,
                    embedding_profile=SimpleNamespace(model_identifier="nomic-v1.5"),
                ),
                SimpleNamespace(
                    collection="b",
                    status="building",
                    embedding_profile_id=2,
                    embedding_profile=SimpleNamespace(model_identifier="minilm"),
                ),
            ]
        )
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = False
        mock_session.query.return_value = revision_query
        mock_session_factory.return_value = lambda: mock_session

        searcher = Searcher()
        with patch("cementic.search._create_embedding_provider"):
            with pytest.raises(RuntimeError) as excinfo:
                searcher.search("test")

        message = str(excinfo.value)
        assert "nomic-v1.5: a (active)" in message
        assert "minilm: b (building)" in message
        assert "-c " in message

    @patch("cementic.search.get_engine")
    @patch("cementic.search.get_session_factory")
    @patch("cementic.search._create_embedding_provider")
    def test_search_uses_active_revision_model(
        self, mock_create_embedding_provider, mock_session_factory, mock_get_engine
    ):
        active_revision = SimpleNamespace(
            collection="default",
            embedding_profile_id=1,
            chunk_profile_id=2,
            extractor_profile_id=1,
            embedding_profile=SimpleNamespace(
                config_json=(
                    '{"provider": "llama-cpp", '
                    '"model_identifier": "nomic-embed-text", "embedding_dim": 768}'
                ),
                model_identifier="nomic-embed-text",
                embedding_dim=768,
                distance_metric="cosine",
            ),
        )
        revision_query = self.RevisionQuery([active_revision])

        row = SimpleNamespace(
            collection="default",
            source_path="/tmp/test.pdf",
            content="chunk text",
            distance=0.05,
        )
        exec_result = MagicMock()
        exec_result.scalar.return_value = 1
        exec_result.__iter__.return_value = iter([row])

        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = False
        mock_session.query.side_effect = [revision_query]
        mock_session.execute.return_value = exec_result
        mock_session_factory.return_value = lambda: mock_session

        embedding_provider = MagicMock()
        embedding_provider.health_check.return_value = True
        embedding_provider.embed.return_value = [0.1] * 768
        mock_create_embedding_provider.return_value = embedding_provider

        searcher = Searcher()
        results = searcher.search("hello")

        assert results[0]["source_path"] == "/tmp/test.pdf"
        assert results[0]["score"] == 0.95
        assert results[0]["distance"] == 0.05
        assert results[0]["score_kind"] == "cosine_similarity"
        embedding_provider.embed.assert_called_once()

    def test_search_result_type(self):
        result = SearchResult(
            collection="default",
            source_path="/test.pdf",
            content="test content",
            score=0.95,
            distance=0.05,
            score_kind="cosine_similarity",
        )

        assert result["collection"] == "default"
        assert result["source_path"] == "/test.pdf"

    def test_score_from_distance_has_metric_specific_semantics(self):
        assert _score_from_distance("cosine", 0.25) == (0.75, "cosine_similarity")
        assert _score_from_distance("l2", 2.5) == (-2.5, "negative_l2_distance")
        assert _score_from_distance("ip", -3.0) == (3.0, "inner_product")

    def test_distance_operator_matches_metric(self):
        assert _distance_operator("cosine") == "<=>"
        assert _distance_operator("l2") == "<->"
        assert _distance_operator("ip") == "<#>"

    def test_searchable_revisions_prefers_active_per_collection(self):
        building = SimpleNamespace(collection="papers", status="building", id=1)
        active = SimpleNamespace(collection="papers", status="active", id=2)
        ready = SimpleNamespace(collection="notes", status="ready", id=3)

        revisions = _searchable_revisions([building, active, ready])

        assert revisions == [active, ready]

    def test_searchable_revisions_keeps_building_only_collection_with_active_other(self):
        """Revision selection is per collection, not all-active globally."""
        active = SimpleNamespace(collection="papers", status="active", id=2)
        building_only = SimpleNamespace(collection="notes", status="building", id=3)

        revisions = _searchable_revisions([active, building_only])

        assert revisions == [active, building_only]

    def test_searchable_revisions_prefers_ready_over_building(self):
        building = SimpleNamespace(collection="papers", status="building", id=1)
        ready = SimpleNamespace(collection="papers", status="ready", id=2)

        revisions = _searchable_revisions([building, ready])

        assert revisions == [ready]

    @patch("cementic.search.get_engine")
    @patch("cementic.search.get_session_factory")
    @patch("cementic.search._create_embedding_provider")
    def test_search_surfaces_a_provider_that_cannot_be_reached(
        self, mock_create_embedding_provider, mock_session_factory, mock_get_engine
    ):
        active_revision = SimpleNamespace(
            collection="default",
            status="active",
            embedding_profile_id=1,
            chunk_profile_id=2,
            extractor_profile_id=1,
            embedding_profile=SimpleNamespace(
                config_json=(
                    '{"provider": "llama-cpp", '
                    '"model_identifier": "nomic-embed-text", "embedding_dim": 768}'
                ),
                model_identifier="nomic-embed-text",
                embedding_dim=768,
                distance_metric="cosine",
            ),
        )
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = False
        mock_session.query.return_value = self.RevisionQuery([active_revision])
        mock_session_factory.return_value = lambda: mock_session

        # Search no longer re-probes health: _create_embedding_provider only
        # returns once the served fingerprint is confirmed, and raises with a
        # more specific reason otherwise. The second probe re-asked a question
        # already answered, and could itself wait out an in-flight batch.
        mock_create_embedding_provider.side_effect = RuntimeError(
            "llama.cpp embedding daemon appears busy"
        )

        searcher = Searcher()
        with pytest.raises(RuntimeError, match="busy"):
            searcher.search("hello")

    @patch("cementic.search.get_engine")
    @patch("cementic.search.get_session_factory")
    def test_search_prefers_active_revisions_without_loading_building_ones(
        self, mock_session_factory, mock_get_engine
    ):
        active_default = SimpleNamespace(
            collection="default",
            status="active",
            embedding_profile_id=1,
            chunk_profile_id=2,
            extractor_profile_id=1,
            embedding_profile=SimpleNamespace(
                config_json=(
                    '{"provider": "llama-cpp", '
                    '"model_identifier": "nomic-embed-text", "embedding_dim": 768}'
                ),
                model_identifier="nomic-embed-text",
                embedding_dim=768,
                distance_metric="cosine",
            ),
        )
        active_test = SimpleNamespace(
            collection="test",
            status="active",
            embedding_profile_id=1,
            chunk_profile_id=2,
            extractor_profile_id=1,
            embedding_profile=active_default.embedding_profile,
        )
        active_query = self.RevisionQuery([active_default, active_test])

        exec_result = MagicMock()
        exec_result.scalar.return_value = 1
        exec_result.__iter__.return_value = iter([])

        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = False
        mock_session.query.side_effect = [active_query]
        mock_session.execute.return_value = exec_result
        mock_session_factory.return_value = lambda: mock_session

        with patch("cementic.search._create_embedding_provider") as mock_create_provider:
            provider = MagicMock()
            provider.health_check.return_value = True
            provider.embed.return_value = [0.1] * 768
            mock_create_provider.return_value = provider

            searcher = Searcher()
            searcher.search("bayes")

        # Only the active-revisions query runs; building revisions are not loaded,
        # and the vector search now goes through session.execute (not .query).
        assert mock_session.query.call_count == 1


class TestSearchOnlyServesCurrentContent:
    """Search must apply the same definition of "current" as the rest of cementic.

    That definition used to live in the query, as a join against the source and
    extraction hashes. It now lives in the *data*: superseded and deleted chunks
    are removed when they become stale, so the query filters on the vector row
    alone. The behavioural guarantee is unchanged and is pinned end-to-end in
    tests/integration/test_search_pg.py; what this class pins is the query
    shape, because a filter on a joined table stops the planner from using the
    ANN index at all -- measured at 407ms versus 1ms on 100k rows.
    """

    @patch("cementic.search.get_engine")
    @patch("cementic.search.get_session_factory")
    @patch("cementic.search._create_embedding_provider")
    def test_knn_query_filters_on_the_vector_row_alone(
        self, mock_create_embedding_provider, mock_session_factory, mock_get_engine
    ):
        revision = SimpleNamespace(
            collection="default",
            embedding_profile_id=1,
            chunk_profile_id=2,
            extractor_profile_id=1,
            embedding_profile=SimpleNamespace(
                config_json=(
                    '{"provider": "llama-cpp", '
                    '"model_identifier": "nomic-embed-text", "embedding_dim": 768}'
                ),
                model_identifier="nomic-embed-text",
                embedding_dim=768,
                distance_metric="cosine",
            ),
        )
        exec_result = MagicMock()
        exec_result.scalar.return_value = 1
        exec_result.__iter__.return_value = iter([])

        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = False
        mock_session.query.side_effect = [TestSearcher.RevisionQuery([revision])]
        mock_session.execute.return_value = exec_result
        mock_session_factory.return_value = lambda: mock_session

        embedding_provider = MagicMock()
        embedding_provider.health_check.return_value = True
        embedding_provider.embed.return_value = [0.1] * 768
        mock_create_embedding_provider.return_value = embedding_provider

        Searcher().search("hello")

        executed = " ".join(
            str(call.args[0]) for call in mock_session.execute.call_args_list if call.args
        )
        # Every filter is on `ev`, the vector table itself.
        assert "ev.collection = :collection" in executed
        assert "ev.chunk_profile_id = :chunk_profile_id" in executed
        assert "ev.extractor_profile_id = :extractor_profile_id" in executed
        # The tables that no longer appear are the ones that only ever carried
        # filters; chunks_v2 and source_documents remain, for content and path.
        assert "chunked_documents" not in executed
        assert "extracted_documents" not in executed


class TestQueryContextBound:
    """A query over the model's context window must be refused, not truncated.

    The server silently drops everything past the limit, so the tail of a long
    query stopped affecting results: two queries sharing a long prefix and
    differing only in their final words returned bit-identical scores.
    """

    def test_query_over_the_context_window_is_rejected(self):
        query = "vector database design and retrieval augmented generation " * 130

        with pytest.raises(ValueError, match="context window"):
            _reject_query_over_context(query, 512)

    def test_ordinary_query_is_accepted(self):
        _reject_query_over_context("transformer inference latency", 512)

    def test_query_quoting_a_control_token_spelling_is_accepted(self):
        """A pasted `<|endoftext|>` is prose to count, not a token to refuse.

        tiktoken raises on literal control-token spellings unless told
        otherwise (same landmine as chunk.py and the embed budget guard).
        """
        _reject_query_over_context("papers about the <|endoftext|> delimiter", 512)

    def test_bound_follows_the_configured_context_window(self):
        """The char-based cap could not do this: it was fixed at 8000 chars
        regardless of the window the model was actually loaded with."""
        query = "retrieval augmented generation " * 100

        _reject_query_over_context(query, 4096)
        with pytest.raises(ValueError, match="context window"):
            _reject_query_over_context(query, 256)


class TestEmbeddingProviderVerboseComesFromLiveConfig:
    """`verbose` was removed from the embedding profile fingerprint (it is a
    launch argument, not a fact about the vectors), so a spec rebuilt from a
    stored profile's JSON no longer carries a real value for it. Search must
    source it from the current config instead of silently defaulting to
    False regardless of what the user has configured -- otherwise a daemon
    started via search never honours `llama_cpp.verbose`.
    """

    def test_verbose_is_taken_from_current_config_not_the_stored_profile(self):
        config = Config()
        config.llama_cpp.verbose = True
        config_json = (
            '{"provider": "llama-cpp", "model_identifier": "model.gguf", '
            '"n_ctx": 512, "n_gpu_layers": 0, "embedding_dim": 768}'
        )

        with patch("cementic.search.create_provider") as mock_create_provider:
            _create_embedding_provider(config_json, config)

        (spec_arg, config_arg), _ = mock_create_provider.call_args
        assert spec_arg.verbose is True
        assert config_arg is config

    def test_false_config_verbose_overrides_a_stale_true_in_the_profile(self):
        """A profile built before this change may still carry `verbose: true`
        in its stored JSON; live config must win regardless of direction."""
        config = Config()
        config.llama_cpp.verbose = False
        config_json = (
            '{"provider": "llama-cpp", "model_identifier": "model.gguf", '
            '"n_ctx": 512, "n_gpu_layers": 0, "embedding_dim": 768, "verbose": true}'
        )

        with patch("cementic.search.create_provider") as mock_create_provider:
            _create_embedding_provider(config_json, config)

        (spec_arg, _config_arg), _ = mock_create_provider.call_args
        assert spec_arg.verbose is False


class TestEmptyQueryIsRejected:
    """An empty query embeds to a real vector, so it produced real-looking results.

    `cementic search ""` and `cementic search "   "` each returned a confidently
    ranked top-k -- the nearest neighbours of nothing -- with exit 0 and no
    indication that the query was empty.
    """

    @pytest.mark.parametrize("query", ["", "   ", "\n\t "])
    @patch("cementic.search.get_engine")
    @patch("cementic.search.get_session_factory")
    def test_blank_queries_are_refused(self, mock_session_factory, mock_get_engine, query):
        searcher = Searcher()

        with pytest.raises(ValueError, match="query cannot be empty"):
            searcher.search(query)

    @patch("cementic.search.get_engine")
    @patch("cementic.search.get_session_factory")
    def test_rejection_happens_before_any_database_work(
        self, mock_session_factory, mock_get_engine
    ):
        searcher = Searcher()
        mock_session_factory.reset_mock()

        with pytest.raises(ValueError, match="query cannot be empty"):
            searcher.search("  ")

        mock_session_factory.return_value.assert_not_called()
