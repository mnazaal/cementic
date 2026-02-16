"""Embedder factory and exports."""

from typing import Type

from seman.embedders.base import Embedder
from seman.embedders.llama_cpp import LlamaCppEmbedder
from seman.embedders.ollama import OllamaEmbedder

__all__ = ["Embedder", "LlamaCppEmbedder", "OllamaEmbedder", "get_embedder"]

_EMBEDDER_REGISTRY: dict[str, Type[Embedder]] = {
    "llama-cpp": LlamaCppEmbedder,
    "ollama": OllamaEmbedder,
}


def get_embedder(name: str, **kwargs) -> Embedder:
    """Get embedder instance by name.

    Args:
        name: Embedder type name (llama-cpp or ollama)
        **kwargs: Constructor arguments for the embedder

    Returns:
        Embedder instance

    Raises:
        ValueError: If embedder type not found
    """
    if name not in _EMBEDDER_REGISTRY:
        raise ValueError(f"Unknown embedder: {name}. Available: {list(_EMBEDDER_REGISTRY.keys())}")

    embedder_class = _EMBEDDER_REGISTRY[name]
    return embedder_class(**kwargs)
