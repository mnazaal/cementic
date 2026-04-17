"""Tests for revision-aware search functionality."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cementic.search import Searcher, SearchResult


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

    class ResultQuery:
        def __init__(self, rows):
            self.rows = rows

        def join(self, *args, **kwargs):
            return self

        def filter(self, *args, **kwargs):
            return self

        def params(self, *args, **kwargs):
            return self

        def order_by(self, *args, **kwargs):
            return self

        def limit(self, *args, **kwargs):
            return self

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
            embedding_profile=SimpleNamespace(
                config_json=(
                    '{"provider": "ollama", "host": "http://localhost:11434", '
                    '"model_identifier": "nomic-embed-text", "embedding_dim": 768}'
                ),
                model_identifier="nomic-embed-text",
                embedding_dim=768,
            ),
        )
        revision_query = self.RevisionQuery([building_revision])

        chunk = SimpleNamespace(
            document=SimpleNamespace(collection="papers", source_path="/tmp/papers.pdf"),
            content="partial chunk text",
            page_start=1,
            page_end=1,
        )
        result_query = self.ResultQuery([(chunk, 0.1)])

        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = False
        mock_session.query.side_effect = [revision_query, result_query]
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
            embedding_profile=SimpleNamespace(
                config_json=(
                    '{"provider": "ollama", "host": "http://localhost:11434", '
                    '"model_identifier": "nomic-embed-text", "embedding_dim": 768}'
                ),
                model_identifier="nomic-embed-text",
                embedding_dim=768,
            ),
        )
        revision_query = self.RevisionQuery([active_revision])

        chunk = SimpleNamespace(
            document=SimpleNamespace(collection="default", source_path="/tmp/test.pdf"),
            content="chunk text",
            page_start=1,
            page_end=2,
        )
        result_query = self.ResultQuery([(chunk, 0.05)])

        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = False
        mock_session.query.side_effect = [revision_query, result_query]
        mock_session_factory.return_value = lambda: mock_session

        embedding_provider = MagicMock()
        embedding_provider.health_check.return_value = True
        embedding_provider.embed.return_value = [0.1] * 768
        mock_create_embedding_provider.return_value = embedding_provider

        searcher = Searcher()
        results = searcher.search("hello")

        assert results[0]["source_path"] == "/tmp/test.pdf"
        embedding_provider.embed.assert_called_once()

    def test_search_result_type(self):
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
            embedding_profile=SimpleNamespace(
                config_json=(
                    '{"provider": "ollama", "host": "http://localhost:11434", '
                    '"model_identifier": "nomic-embed-text", "embedding_dim": 768}'
                ),
                model_identifier="nomic-embed-text",
                embedding_dim=768,
            ),
        )
        active_test = SimpleNamespace(
            collection="test",
            status="active",
            embedding_profile_id=1,
            chunk_profile_id=2,
            embedding_profile=active_default.embedding_profile,
        )
        active_query = self.RevisionQuery([active_default, active_test])
        result_query = self.ResultQuery([])

        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = False
        mock_session.query.side_effect = [active_query, result_query, result_query]
        mock_session_factory.return_value = lambda: mock_session

        with patch("cementic.search._create_embedding_provider") as mock_create_provider:
            provider = MagicMock()
            provider.health_check.return_value = True
            provider.embed.return_value = [0.1] * 768
            mock_create_provider.return_value = provider

            searcher = Searcher()
            searcher.search("bayes")

        assert mock_session.query.call_count == 3

    @patch("cementic.search.get_llama_cpp_runtime_client")
    def test_create_provider_uses_llama_runtime_client(self, mock_runtime_client):
        from cementic.config import Config
        from cementic.search import _create_embedding_provider

        config = Config()
        _create_embedding_provider(
            (
                '{"provider": "llama-cpp", "model_identifier": "model.gguf", '
                '"n_ctx": 512, "n_gpu_layers": 0, "embedding_dim": 768, '
                '"verbose": false}'
            ),
            config,
        )

        mock_runtime_client.assert_called_once_with(config)
