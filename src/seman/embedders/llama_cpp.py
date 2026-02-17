"""llama.cpp embedder implementation using llama-cpp-python."""

from pathlib import Path
from typing import List, Optional

from seman.embedders.base import Embedder


class LlamaCppEmbedder(Embedder):
    """Embedder using llama.cpp via llama-cpp-python."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 512,
        n_gpu_layers: int = 0,
        embedding_dim: int = 768,
        verbose: bool = False,
    ) -> None:
        """Initialize llama.cpp embedder.

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

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        result = self._llm.create_embedding(text)
        return result["data"][0]["embedding"]

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
        """Check if model is loaded and working."""
        try:
            # Try to create a simple embedding
            _ = self.embed("test")
            return True
        except Exception:
            return False

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self._embedding_dim
