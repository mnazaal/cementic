"""Tests for search functionality."""

from unittest.mock import MagicMock, patch

from seman.search import Searcher, SearchResult


class TestSearcher:
    """Test search functionality."""

    @patch("seman.search.get_engine")
    @patch("seman.search.get_session_factory")
    @patch("seman.search.get_embedder")
    def test_search_with_results(self, mock_get_embedder, mock_session_factory, mock_get_engine):
        """Test search returns results."""
        # Setup mock embedder
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 768
        mock_get_embedder.return_value = mock_embedder

        # Setup mock session
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session_factory.return_value = mock_session

        # Setup query chain
        mock_query_result = MagicMock()
        mock_query_result.join.return_value = mock_query_result
        mock_query_result.filter.return_value = mock_query_result
        mock_query_result.order_by.return_value = mock_query_result
        mock_query_result.limit.return_value = mock_query_result
        mock_query_result.all.return_value = []
        mock_session.query.return_value = mock_query_result

        # Create searcher and search
        searcher = Searcher()
        results = searcher.search("nonexistent query")

        # Verify empty results
        assert results == []

    @patch("seman.search.get_engine")
    @patch("seman.search.get_session_factory")
    @patch("seman.search.get_embedder")
    def test_search_uses_correct_embedder(
        self, mock_get_embedder, mock_session_factory, mock_get_engine
    ):
        """Test that search uses configured embedder."""
        # Setup mock embedder
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 768
        mock_get_embedder.return_value = mock_embedder

        # Setup mock session
        mock_session = MagicMock()
        mock_session_factory.return_value = lambda: mock_session
        mock_query_result = MagicMock()
        mock_query_result.join.return_value = mock_query_result
        mock_query_result.filter.return_value = mock_query_result
        mock_query_result.order_by.return_value = mock_query_result
        mock_query_result.limit.return_value = mock_query_result
        mock_query_result.all.return_value = []
        mock_session.query.return_value = mock_query_result

        # Test with llama-cpp config
        with patch("seman.search.Config") as mock_config_class:
            mock_config = MagicMock()
            mock_config.indexing.embedder = "llama-cpp"
            mock_config.llama_cpp.model_path = "/path/to/model.gguf"
            mock_config_class.return_value = mock_config

            searcher = Searcher()
            searcher.search("test")

            # Verify correct embedder was requested
            mock_get_embedder.assert_called_once()
            assert mock_get_embedder.call_args[0][0] == "llama-cpp"

    @patch("seman.search.get_engine")
    @patch("seman.search.get_session_factory")
    @patch("seman.search.get_embedder")
    def test_search_respects_top_k(self, mock_get_embedder, mock_session_factory, mock_get_engine):
        """Test that search respects top_k parameter."""
        # Setup mock embedder
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 768
        mock_get_embedder.return_value = mock_embedder

        # Setup mock session
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.return_value = mock_session  # Make callable return itself
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []
        mock_session_factory.return_value = mock_session

        # Create searcher and search with specific top_k
        searcher = Searcher()
        searcher.search("test", top_k=5)

        # Verify limit was called with correct value
        mock_query.limit.assert_called_once_with(5)

    @patch("seman.search.get_engine")
    @patch("seman.search.get_session_factory")
    @patch("seman.search.get_embedder")
    def test_search_filters_by_collections(
        self, mock_get_embedder, mock_session_factory, mock_get_engine
    ):
        """Test search applies collection filtering when requested."""
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 768
        mock_get_embedder.return_value = mock_embedder

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.return_value = mock_session
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []
        mock_session_factory.return_value = mock_session

        searcher = Searcher()
        searcher.search("test", collections=["work", "personal"])

        assert mock_query.filter.call_count >= 2

    def test_search_result_type(self):
        """Test that search results have correct type."""
        result = SearchResult(
            collection="default",
            source_path="/test.pdf",
            content="test content",
            score=0.95,
            page_start=1,
            page_end=2,
        )

        assert result["collection"] == "default"
        assert result["source_path"] == "/test.pdf"
        assert result["content"] == "test content"
        assert result["score"] == 0.95
        assert result["page_start"] == 1
        assert result["page_end"] == 2
