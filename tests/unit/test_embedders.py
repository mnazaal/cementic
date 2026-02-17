"""Tests for embedder base class and implementations."""

from unittest.mock import MagicMock, patch

import pytest

from seman.embedders import get_embedder
from seman.embedders.base import Embedder
from seman.embedders.llama_cpp import LlamaCppEmbedder
from seman.embedders.ollama import OllamaEmbedder


class TestEmbedderInterface:
    """Test the abstract embedder interface."""

    def test_embedder_is_abstract(self):
        """Test that Embedder cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Embedder()

    def test_embedder_subclass_must_implement_methods(self):
        """Test that subclasses must implement abstract methods."""

        class IncompleteEmbedder(Embedder):
            pass

        with pytest.raises(TypeError):
            IncompleteEmbedder()


class TestEmbedderFactory:
    """Test the embedder factory function."""

    def test_get_llama_cpp_embedder(self, temp_dir):
        """Test factory creates llama-cpp embedder."""
        # Create a dummy model file
        model_path = temp_dir / "test.gguf"
        model_path.write_bytes(b"dummy model")

        # Import is inside __init__, so we need to patch at the source
        with patch("llama_cpp.Llama") as mock_llama:
            mock_llama.return_value = MagicMock()
            embedder = get_embedder("llama-cpp", model_path=str(model_path))
            assert isinstance(embedder, LlamaCppEmbedder)

    def test_get_ollama_embedder(self):
        """Test factory creates Ollama embedder."""
        embedder = get_embedder("ollama", host="http://test:11434", model="test-model")
        assert isinstance(embedder, OllamaEmbedder)

    def test_get_unknown_embedder_raises_error(self):
        """Test factory raises error for unknown embedder type."""
        with pytest.raises(ValueError, match="Unknown embedder"):
            get_embedder("unknown")


class TestOllamaEmbedder:
    """Test Ollama embedder implementation."""

    def test_init(self):
        """Test Ollama embedder initialization."""
        embedder = OllamaEmbedder(
            host="http://test:11434", model="nomic-embed-text", embedding_dim=768
        )
        assert embedder.host == "http://test:11434"
        assert embedder.model == "nomic-embed-text"
        assert embedder.embedding_dim == 768

    @patch("seman.embedders.ollama.requests.post")
    def test_embed_single(self, mock_post):
        """Test embedding a single text."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.1] * 768}
        mock_post.return_value = mock_response

        embedder = OllamaEmbedder()
        result = embedder.embed("test text")

        assert len(result) == 768
        assert result[0] == 0.1
        mock_post.assert_called_once()

    @patch("seman.embedders.ollama.requests.post")
    def test_embed_batch(self, mock_post):
        """Test embedding multiple texts."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.2] * 768}
        mock_post.return_value = mock_response

        embedder = OllamaEmbedder()
        texts = ["text1", "text2", "text3"]
        results = embedder.embed_batch(texts)

        assert len(results) == 3
        assert all(r is not None and len(r) == 768 for r in results)
        assert mock_post.call_count == 3

    @patch("seman.embedders.ollama.requests.post")
    def test_embed_batch_handles_failures(self, mock_post):
        """Test batch embedding handles individual failures."""
        mock_post.side_effect = [
            Exception("Failed"),
            MagicMock(json=lambda: {"embedding": [0.1] * 768}),
        ]

        embedder = OllamaEmbedder()
        texts = ["fail", "success"]
        results = embedder.embed_batch(texts)

        assert results[0] is None
        assert results[1] is not None

    @patch("seman.embedders.ollama.requests.get")
    def test_health_check_success(self, mock_get):
        """Test health check with healthy server."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        embedder = OllamaEmbedder()
        assert embedder.health_check() is True

    @patch("seman.embedders.ollama.requests.get")
    def test_health_check_failure(self, mock_get):
        """Test health check with unhealthy server."""
        import requests

        mock_get.side_effect = requests.RequestException("Connection failed")

        embedder = OllamaEmbedder()
        assert embedder.health_check() is False


class TestLlamaCppEmbedder:
    """Test llama.cpp embedder implementation."""

    def test_init_raises_if_model_not_found(self, temp_dir):
        """Test initialization fails if model file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            LlamaCppEmbedder(model_path=str(temp_dir / "nonexistent.gguf"))

    @patch("llama_cpp.Llama")
    def test_init_with_model_file(self, mock_llama_class, temp_dir):
        """Test initialization with valid model file."""
        model_path = temp_dir / "test.gguf"
        model_path.write_bytes(b"dummy model")

        mock_llama = MagicMock()
        mock_llama_class.return_value = mock_llama

        embedder = LlamaCppEmbedder(model_path=str(model_path), n_ctx=512, embedding_dim=768)

        assert embedder.model_path == model_path
        assert embedder.embedding_dim == 768
        mock_llama_class.assert_called_once()

    @patch("llama_cpp.Llama")
    def test_embed_single(self, mock_llama_class, temp_dir):
        """Test embedding a single text."""
        model_path = temp_dir / "test.gguf"
        model_path.write_bytes(b"dummy model")

        mock_llama = MagicMock()
        mock_llama.create_embedding.return_value = {"data": [{"embedding": [0.3] * 768}]}
        mock_llama_class.return_value = mock_llama

        embedder = LlamaCppEmbedder(model_path=str(model_path))
        result = embedder.embed("test text")

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

        embedder = LlamaCppEmbedder(model_path=str(model_path))
        texts = ["text1", "text2"]
        results = embedder.embed_batch(texts)

        assert len(results) == 2
        assert all(r is not None and len(r) == 768 for r in results)
        assert mock_llama.create_embedding.call_count == 2

    @patch("llama_cpp.Llama")
    def test_health_check(self, mock_llama_class, temp_dir):
        """Test health check."""
        model_path = temp_dir / "test.gguf"
        model_path.write_bytes(b"dummy model")

        mock_llama = MagicMock()
        mock_llama.create_embedding.return_value = {"data": [{"embedding": [0.1] * 768}]}
        mock_llama_class.return_value = mock_llama

        embedder = LlamaCppEmbedder(model_path=str(model_path))
        assert embedder.health_check() is True

    @patch("llama_cpp.Llama")
    def test_health_check_failure(self, mock_llama_class, temp_dir):
        """Test health check with unhealthy model."""
        model_path = temp_dir / "test.gguf"
        model_path.write_bytes(b"dummy model")

        mock_llama = MagicMock()
        mock_llama.create_embedding.side_effect = Exception("Model error")
        mock_llama_class.return_value = mock_llama

        embedder = LlamaCppEmbedder(model_path=str(model_path))
        assert embedder.health_check() is False
