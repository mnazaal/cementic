"""Embedding provider factory and exports."""

from typing import Any, Type

from cementic.embedding_providers.base import EmbeddingProvider
from cementic.embedding_providers.llama_cpp import LlamaCppEmbeddingProvider
from cementic.embedding_providers.ollama import OllamaEmbeddingProvider

__all__ = [
    "EmbeddingProvider",
    "LlamaCppEmbeddingProvider",
    "OllamaEmbeddingProvider",
    "get_embedding_provider",
]

_EMBEDDING_PROVIDER_REGISTRY: dict[str, Type[EmbeddingProvider]] = {
    "llama-cpp": LlamaCppEmbeddingProvider,
    "ollama": OllamaEmbeddingProvider,
}


def get_embedding_provider(name: str, **kwargs: Any) -> EmbeddingProvider:
    """Get embedding provider instance by name.

    Args:
        name: Embedding provider type name (llama-cpp or ollama)
        **kwargs: Constructor arguments for the provider

    Returns:
        EmbeddingProvider instance

    Raises:
        ValueError: If provider type is not found
    """
    if name not in _EMBEDDING_PROVIDER_REGISTRY:
        available = list(_EMBEDDING_PROVIDER_REGISTRY.keys())
        raise ValueError(f"Unknown embedding provider: {name}. Available: {available}")

    provider_class = _EMBEDDING_PROVIDER_REGISTRY[name]
    return provider_class(**kwargs)
