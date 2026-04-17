"""llama.cpp embedding provider using llama-cpp-python."""

from pathlib import Path
from typing import cast

from cementic.embedding_providers.base import EmbeddingProvider


class LlamaCppEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using llama.cpp via llama-cpp-python."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 512,
        n_gpu_layers: int = 0,
        embedding_dim: int = 768,
        verbose: bool = False,
    ) -> None:
        """Initialize the llama.cpp embedding provider.

        Args:
            model_path: Path to the .gguf model file
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
            embedding_dim: Expected embedding dimension
            verbose: Enable verbose output
        """
        from llama_cpp import Llama

        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self._embedding_dim = embedding_dim
        self.verbose = verbose

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Initialize the model with embedding=True
        self._llm = Llama(
            model_path=str(self.model_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            embedding=True,
            verbose=verbose,
        )

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        result = self._llm.create_embedding(text)
        embedding = cast(list[float], result["data"][0]["embedding"])
        return embedding

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
        """Check if model is loaded and working."""
        return hasattr(self, "_llm") and self._llm is not None

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self._embedding_dim
