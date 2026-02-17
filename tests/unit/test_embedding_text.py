"""Tests for embedding text formatting helpers."""

from seman.config import get_config
from seman.embedding_text import format_document_text, format_query_text


def test_adds_nomic_v2_document_prefix() -> None:
    config = get_config()
    config.indexing.embedder = "llama-cpp"
    config.llama_cpp.model_path = "models/nomic-embed-text-v2-moe.Q8_0.gguf"

    assert format_document_text("hello", config) == "search_document: hello"


def test_adds_nomic_v2_query_prefix() -> None:
    config = get_config()
    config.indexing.embedder = "llama-cpp"
    config.llama_cpp.model_path = "models/nomic-embed-text-v2-moe.Q8_0.gguf"

    assert format_query_text("hello", config) == "search_query: hello"


def test_does_not_change_non_nomic_v2_text() -> None:
    config = get_config()
    config.indexing.embedder = "ollama"
    config.ollama.model = "nomic-embed-text"

    assert format_document_text("hello", config) == "hello"
    assert format_query_text("hello", config) == "hello"
