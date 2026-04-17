"""Abstract base class for embedding providers."""

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (None for failed embeddings)
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the embedding provider is available and working.

        Returns:
            True if healthy, False otherwise
        """
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        pass
