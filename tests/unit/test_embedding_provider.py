"""Tests for the embedding provider base class."""

import pytest

from cementic.embedding_provider import EmbeddingProvider


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
