"""Tests for revision-aware search functionality."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cementic.search import (
    Searcher,
    SearchResult,
    _distance_operator,
    _reject_query_over_context,
    _score_from_distance,
    _searchable_revisions,
)


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

    @patch("cementic.search.get_engine")
    @patch("cementic.search.get_session_factory")
    @patch("cementic.search._create_embedding_provider")
    def test_search_falls_back_to_building_when_active_query_empty(
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
        row = SimpleNamespace(
            collection="papers",
            source_path="/tmp/papers.pdf",
            content="partial chunk text",
            distance=0.1,
        )
        exec_result = MagicMock()
        exec_result.scalar.return_value = 1
        exec_result.__iter__.return_value = iter([row])

        # active query → empty, fallback query → building revision
        active_query = self.RevisionQuery([])
        fallback_query = self.RevisionQuery([building_revision])

        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = False
        mock_session.query.side_effect = [active_query, fallback_query]
        mock_session.execute.return_value = exec_result
        mock_session_factory.return_value = lambda: mock_session

        embedding_provider = MagicMock()
        embedding_provider.health_check.return_value = True
        embedding_provider.embed.return_value = [0.1] * 768
        mock_create_embedding_provider.return_value = embedding_provider

        searcher = Searcher()
        results = searcher.search("hello", collections=["papers"])

        assert results[0]["source_path"] == "/tmp/papers.pdf"

    @patch("cementic.search.get_engine")
    @patch("cementic.search.get_session_factory")
    def test_search_rejects_mixed_active_models(self, mock_session_factory, mock_get_engine):
        revision_query = self.RevisionQuery(
            [
                SimpleNamespace(collection="a", embedding_profile_id=1),
                SimpleNamespace(collection="b", embedding_profile_id=2),
            ]
        )
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = False
        mock_session.query.return_value = revision_query
        mock_session_factory.return_value = lambda: mock_session

        searcher = Searcher()
        with patch("cementic.search._create_embedding_provider"):
            try:
                searcher.search("test")
            except RuntimeError as error:
                assert "different active embedding models" in str(error)
            else:
                raise AssertionError("Expected RuntimeError")

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
    def test_search_rejects_unhealthy_embedding_provider(
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

        embedding_provider = MagicMock()
        embedding_provider.health_check.return_value = False
        mock_create_embedding_provider.return_value = embedding_provider

        searcher = Searcher()
        try:
            searcher.search("hello")
        except RuntimeError as error:
            assert "not healthy" in str(error)
        else:
            raise AssertionError("Expected RuntimeError")

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

    Filtering on profile ids alone serves the *old* content of a file whose
    re-extraction failed: the superseded rows keep ``status='done'`` and match
    the profile ids, so stale text ranks normally and indefinitely.
    """

    @patch("cementic.search.get_engine")
    @patch("cementic.search.get_session_factory")
    @patch("cementic.search._create_embedding_provider")
    def test_knn_query_requires_hashes_to_match_the_source(
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

        knn_sql = " ".join(
            str(call.args[0]) for call in mock_session.execute.call_args_list if call.args
        )
        assert "ed.source_file_hash = sd.file_hash" in knn_sql
        assert "cd.source_content_hash = ed.content_hash" in knn_sql
        assert "ed.status = 'done'" in knn_sql
        assert "cd.status = 'done'" in knn_sql


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

    def test_bound_follows_the_configured_context_window(self):
        """The char-based cap could not do this: it was fixed at 8000 chars
        regardless of the window the model was actually loaded with."""
        query = "retrieval augmented generation " * 100

        _reject_query_over_context(query, 4096)
        with pytest.raises(ValueError, match="context window"):
            _reject_query_over_context(query, 256)
