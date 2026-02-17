"""Utilities for embedding input text formatting."""

from __future__ import annotations

from pathlib import Path

from seman.config import Config


def format_document_text(text: str, config: Config) -> str:
    """Format document text before embedding."""
    return _apply_nomic_v2_prefix(text, "search_document", config)


def format_query_text(text: str, config: Config) -> str:
    """Format query text before embedding."""
    return _apply_nomic_v2_prefix(text, "search_query", config)


def _apply_nomic_v2_prefix(text: str, task: str, config: Config) -> str:
    if not _uses_nomic_v2_model(config):
        return text

    prefix = f"{task}: "
    if text.startswith(prefix):
        return text
    return f"{prefix}{text}"


def _uses_nomic_v2_model(config: Config) -> bool:
    if config.indexing.embedder == "ollama":
        model_name = config.ollama.model.lower()
    else:
        model_name = Path(config.llama_cpp.model_path).name.lower()

    return "nomic-embed-text-v2" in model_name
