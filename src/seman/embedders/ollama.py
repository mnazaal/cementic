"""Ollama embedder implementation."""

from typing import List, Optional

import requests

from seman.embedders.base import Embedder


class OllamaEmbedder(Embedder):
    """Embedder using Ollama API."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        embedding_dim: int = 768,
    ) -> None:
        """Initialize Ollama embedder.

        Args:
            host: Ollama server URL
            model: Model name to use for embeddings
            embedding_dim: Expected embedding dimension
        """
        self.host = host.rstrip("/")
        self.model = model
        self._embedding_dim = embedding_dim

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        response = requests.post(
            f"{self.host}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts."""
        results = []

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
            return response.status_code == 200
        except requests.RequestException:
            return False

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self._embedding_dim
