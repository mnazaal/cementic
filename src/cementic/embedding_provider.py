"""Abstract base class for embedding providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingFacts:
    """Self-described, immutable facts about an embedding backend."""

    name: str
    embedding_dim: int
    distance_metric: str


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    A provider is a self-describing value: it knows its backend ``name``, the
    ``distance_metric`` its vectors are built for, its ``embedding_dim``, and how
    to format a document vs. a query before embedding. Callers select a provider
    by data and then ask it -- they never branch on its identity.
    """

    #: Backend name, e.g. "llama-cpp". Overridden per provider.
    name: str = "embedding"

    #: Vector distance metric this provider's embeddings are built for.
    distance_metric: str = "cosine"

    def format_document(self, text: str) -> str:
        """Format document text before embedding. Pure; default is identity."""
        return text

    def format_query(self, text: str) -> str:
        """Format query text before embedding. Pure; default is identity."""
        return text

    def describe(self) -> EmbeddingFacts:
        """Return this provider's self-described facts.

        The default reports declared values. A provider backed by a live runtime
        may override this to report what the model actually produces (e.g. by
        probing the embedding dimension).
        """
        return EmbeddingFacts(
            name=self.name,
            embedding_dim=self.embedding_dim,
            distance_metric=self.distance_metric,
        )

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

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        pass
