"""Tests for embedding provider base class and implementations."""

from unittest.mock import MagicMock, patch

import pytest

from cementic.embedding_providers import get_embedding_provider
from cementic.embedding_providers.base import EmbeddingProvider
from cementic.embedding_providers.llama_cpp import LlamaCppEmbeddingProvider
from cementic.embedding_providers.ollama import OllamaEmbeddingProvider


class TestEmbeddingProviderInterface:
    """Test the abstract embedding provider interface."""

    def test_provider_is_abstract(self):
        """Test that EmbeddingProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            EmbeddingProvider()  # type: ignore[abstract]

    def test_provider_subclass_must_implement_methods(self):
        """Test that subclasses must implement abstract methods."""

        class IncompleteEmbeddingProvider(EmbeddingProvider):
            pass

        with pytest.raises(TypeError):
            IncompleteEmbeddingProvider()  # type: ignore[abstract]


class TestEmbeddingProviderFactory:
    """Test the embedding provider factory function."""

    def test_get_llama_cpp_provider(self, temp_dir):
        """Test factory creates llama-cpp embedding provider."""
        # Create a dummy model file
        model_path = temp_dir / "test.gguf"
        model_path.write_bytes(b"dummy model")

        # Import is inside __init__, so we need to patch at the source
        with patch("llama_cpp.Llama") as mock_llama:
            mock_llama.return_value = MagicMock()
            provider = get_embedding_provider("llama-cpp", model_path=str(model_path))
            assert isinstance(provider, LlamaCppEmbeddingProvider)

    def test_get_ollama_provider(self):
        """Test factory creates Ollama embedding provider."""
        provider = get_embedding_provider("ollama", host="http://test:11434", model="test-model")
        assert isinstance(provider, OllamaEmbeddingProvider)

    def test_get_unknown_provider_raises_error(self):
        """Test factory raises error for unknown embedding provider type."""
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            get_embedding_provider("unknown")


class TestOllamaEmbeddingProvider:
    """Test Ollama embedding provider implementation."""

    def test_init(self):
        """Test Ollama embedding provider initialization."""
        provider = OllamaEmbeddingProvider(
            host="http://test:11434", model="nomic-embed-text", embedding_dim=768
        )
        assert provider.host == "http://test:11434"
        assert provider.model == "nomic-embed-text"
        assert provider.embedding_dim == 768

    @patch("cementic.embedding_providers.ollama.requests.post")
    def test_embed_single(self, mock_post):
        """Test embedding a single text."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.1] * 768}
        mock_post.return_value = mock_response

        provider = OllamaEmbeddingProvider()
        result = provider.embed("test text")

        assert len(result) == 768
        assert result[0] == 0.1
        mock_post.assert_called_once()

    @patch("cementic.embedding_providers.ollama.requests.post")
    def test_embed_batch(self, mock_post):
        """Test embedding multiple texts."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.2] * 768}
        mock_post.return_value = mock_response

        provider = OllamaEmbeddingProvider()
        texts = ["text1", "text2", "text3"]
        results = provider.embed_batch(texts)

        assert len(results) == 3
        assert all(r is not None and len(r) == 768 for r in results)
        assert mock_post.call_count == 3

    @patch("cementic.embedding_providers.ollama.requests.post")
    def test_embed_batch_handles_failures(self, mock_post):
        """Test batch embedding handles individual failures."""
        mock_post.side_effect = [
            Exception("Failed"),
            MagicMock(json=lambda: {"embedding": [0.1] * 768}),
        ]

        provider = OllamaEmbeddingProvider()
        texts = ["fail", "success"]
        results = provider.embed_batch(texts)

        assert results[0] is None
        assert results[1] is not None

    @patch("cementic.embedding_providers.ollama.requests.get")
    def test_health_check_success(self, mock_get):
        """Test health check with healthy server."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        provider = OllamaEmbeddingProvider()
        assert provider.health_check() is True

    @patch("cementic.embedding_providers.ollama.requests.get")
    def test_health_check_failure(self, mock_get):
        """Test health check with unhealthy server."""
        import requests

        mock_get.side_effect = requests.RequestException("Connection failed")

        provider = OllamaEmbeddingProvider()
        assert provider.health_check() is False


class TestLlamaCppEmbeddingProvider:
    """Test llama.cpp embedding provider implementation."""

    def test_init_raises_if_model_not_found(self, temp_dir):
        """Test initialization fails if model file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            LlamaCppEmbeddingProvider(model_path=str(temp_dir / "nonexistent.gguf"))

    @patch("llama_cpp.Llama")
    def test_init_with_model_file(self, mock_llama_class, temp_dir):
        """Test initialization with valid model file."""
        model_path = temp_dir / "test.gguf"
        model_path.write_bytes(b"dummy model")

        mock_llama = MagicMock()
        mock_llama_class.return_value = mock_llama

        provider = LlamaCppEmbeddingProvider(
            model_path=str(model_path), n_ctx=512, embedding_dim=768
        )

        assert provider.model_path == model_path
        assert provider.embedding_dim == 768
        mock_llama_class.assert_called_once()

    @patch("llama_cpp.Llama")
    def test_embed_single(self, mock_llama_class, temp_dir):
        """Test embedding a single text."""
        model_path = temp_dir / "test.gguf"
        model_path.write_bytes(b"dummy model")

        mock_llama = MagicMock()
        mock_llama.create_embedding.return_value = {"data": [{"embedding": [0.3] * 768}]}
        mock_llama_class.return_value = mock_llama

        provider = LlamaCppEmbeddingProvider(model_path=str(model_path))
        result = provider.embed("test text")

        assert len(result) == 768
        assert result[0] == 0.3
        mock_llama.create_embedding.assert_called_once_with("test text")

    @patch("llama_cpp.Llama")
    def test_embed_batch(self, mock_llama_class, temp_dir):
        """Test embedding multiple texts."""
        model_path = temp_dir / "test.gguf"
        model_path.write_bytes(b"dummy model")

        mock_llama = MagicMock()
        mock_llama.create_embedding.return_value = {"data": [{"embedding": [0.4] * 768}]}
        mock_llama_class.return_value = mock_llama

        provider = LlamaCppEmbeddingProvider(model_path=str(model_path))
        texts = ["text1", "text2"]
        results = provider.embed_batch(texts)

        assert len(results) == 2
        assert all(r is not None and len(r) == 768 for r in results)
        assert mock_llama.create_embedding.call_count == 2

    @patch("llama_cpp.Llama")
    def test_health_check(self, mock_llama_class, temp_dir):
        """Test health check."""
        model_path = temp_dir / "test.gguf"
        model_path.write_bytes(b"dummy model")

        mock_llama = MagicMock()
        mock_llama_class.return_value = mock_llama

        provider = LlamaCppEmbeddingProvider(model_path=str(model_path))
        assert provider.health_check() is True
        mock_llama.create_embedding.assert_not_called()

    def test_health_check_failure_without_model(self):
        """Test health check returns false without a loaded model."""
        provider = object.__new__(LlamaCppEmbeddingProvider)
        assert provider.health_check() is False
