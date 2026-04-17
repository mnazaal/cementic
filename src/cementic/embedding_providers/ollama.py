"""Ollama embedding provider implementation."""

# mypy: disable-error-code=import-untyped

import requests

from cementic.embedding_providers.base import EmbeddingProvider


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using the Ollama API."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        embedding_dim: int = 768,
    ) -> None:
        """Initialize the Ollama embedding provider.

        Args:
            host: Ollama server URL
            model: Model name to use for embeddings
            embedding_dim: Expected embedding dimension
        """
        self.host = host.rstrip("/")
        self.model = model
        self._embedding_dim = embedding_dim

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        response = requests.post(
            f"{self.host}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        embedding = payload["embedding"]
        return [float(value) for value in embedding]

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        """Generate embeddings for multiple texts."""
        results: list[list[float] | None] = []

        for text in texts:
            try:
                embedding = self.embed(text)
                results.append(embedding)
            except Exception:
                results.append(None)

        return results

    def health_check(self) -> bool:
        """Check if Ollama server is accessible."""
        try:
            response = requests.get(
                f"{self.host}/api/tags",
                timeout=5,
            )
            return bool(response.status_code == 200)
        except requests.RequestException:
            return False

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self._embedding_dim
