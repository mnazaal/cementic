"""Vectorization using Ollama API."""

from typing import List

import requests

from seman.config import OllamaConfig


class Vectorizer:
    """Client for Ollama embedding API."""

    def __init__(self, config: OllamaConfig) -> None:
        """Initialize with Ollama configuration."""
        self.config = config
        self.base_url = config.host.rstrip("/")
        self.model = config.model

    def embed(self, text: str) -> List[float]:
        """Generate embedding for single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={
                "model": self.model,
                "prompt": text,
            },
            timeout=60,
        )
        response.raise_for_status()

        data = response.json()
        return data["embedding"]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = []

        # Process in batches to avoid overwhelming Ollama
        batch_size = self.config.batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            for text in batch:
                try:
                    embedding = self.embed(text)
                    embeddings.append(embedding)
                except Exception as e:
                    # Return None for failed embeddings
                    embeddings.append(None)

        return embeddings

    def health_check(self) -> bool:
        """Check if Ollama server is accessible."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False
