"""Test fixtures and configuration."""

import tempfile
from pathlib import Path
from typing import Generator

import pytest


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
        "ollama": {
            "host": "http://localhost:11434",
            "model": "nomic-embed-text",
            "embedding_dim": 768,
        },
        "indexing": {
            "chunk_size": 512,
            "chunk_overlap": 128,
            "embedder": "llama-cpp",
            "state_path": str(temp_dir / "state.json"),
        },
        "converter": {
            "pid_file": str(temp_dir / "converter.pid"),
            "log_file": str(temp_dir / "converter.log"),
        },
        "embedder": {
            "pid_file": str(temp_dir / "embedder.pid"),
            "log_file": str(temp_dir / "embedder.log"),
            "max_workers": 2,
            "batch_size": 16,
            "poll_interval": 0.1,
        },
    }
