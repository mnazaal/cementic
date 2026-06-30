"""Test fixtures and configuration."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from cementic.embedding_provider import EmbeddingProvider

# Ensure DB password env var is set for all tests
os.environ.setdefault("CEMENTIC_DB_PASSWORD", "test-password")


@pytest.fixture(autouse=True)
def _isolate_user_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Keep tests hermetic: never pick up a developer's real config file.

    Clears CEMENTIC_CONFIG and points the user-config dir at an empty temp dir,
    so resolve_config_path() returns None unless a test opts in. Tests that
    exercise the config file set CEMENTIC_CONFIG themselves.
    """
    monkeypatch.delenv("CEMENTIC_CONFIG", raising=False)
    empty = tmp_path_factory.mktemp("cementic-no-user-config")
    monkeypatch.setattr("cementic.config.user_config_dir", lambda *a, **k: str(empty))


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_pdf_content() -> bytes:
    """Provide sample PDF content (minimal valid PDF structure)."""
    return (
        b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
        b"2 0 obj\n<<\n/Type /Pages\n/Kids []\n/Count 0\n>>\nendobj\nxref\n0 3\n"
        b"0000000000 65535 f\n0000000009 00000 n\n0000000058 00000 n\n"
        b"trailer\n<<\n/Size 3\n/Root 1 0 R\n>>\nstartxref\n107\n%%EOF"
    )


@pytest.fixture
def mock_embedding() -> list:
    """Provide a mock embedding vector."""
    return [0.1] * 768


class FakeEmbeddingClient(EmbeddingProvider):
    """Tiny deterministic embedding client for integration tests."""

    TERMS = ["computer", "symbiosis", "man", "time", "machine", "learning",
             "vector", "semantic", "neural", "network"]

    def health_check(self) -> bool:
        return True

    def embed(self, text: str) -> list[float]:
        lowered = text.lower().replace("search_document: ", "").replace("search_query: ", "")
        return [float(lowered.count(term)) for term in self.TERMS]

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        return [self.embed(text) for text in texts]

    @property
    def embedding_dim(self) -> int:
        return len(self.TERMS)


@pytest.fixture
def fake_embedding_client() -> FakeEmbeddingClient:
    """Provide a deterministic fake embedding client."""
    return FakeEmbeddingClient()


@pytest.fixture
def pdf_fixtures_dir() -> Path:
    """Path to the PDF fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_config_dict(temp_dir: Path) -> dict:
    """Provide a sample configuration dictionary."""
    return {
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "test_seman",
            "user": "test_user",
            "password": "test_pass",
        },
        "llama_cpp": {
            "model_path": str(temp_dir / "test_model.gguf"),
            "n_ctx": 512,
            "n_gpu_layers": 0,
            "embedding_dim": 768,
            "verbose": False,
        },
        "pipeline": {
            "chunk_size": 512,
            "chunk_overlap": 128,
            "embedding_provider": "llama-cpp",
        },
        "source_watcher": {
            "log_file": str(temp_dir / "source_watcher.log"),
        },
        "pipeline_worker": {
            "log_file": str(temp_dir / "pipeline_worker.log"),
            "batch_size": 16,
            "poll_interval": 0.1,
        },
    }
