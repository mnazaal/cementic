"""Utilities for embedding input text formatting."""

from __future__ import annotations

from pathlib import Path

from cementic.config import Config


def format_document_text(text: str, config: Config) -> str:
    """Format document text before embedding."""
    return format_document_text_for_model(text, _model_name_from_config(config))


def format_query_text(text: str, config: Config) -> str:
    """Format query text before embedding."""
    return format_query_text_for_model(text, _model_name_from_config(config))


def format_document_text_for_model(text: str, model_identifier: str) -> str:
    """Format document text for a specific embedding model."""
    return _apply_nomic_v2_prefix(text, "search_document", model_identifier)


def format_query_text_for_model(text: str, model_identifier: str) -> str:
    """Format query text for a specific embedding model."""
    return _apply_nomic_v2_prefix(text, "search_query", model_identifier)


def _apply_nomic_v2_prefix(text: str, task: str, model_identifier: str) -> str:
    if not _uses_nomic_v2_model(model_identifier):
        return text

    prefix = f"{task}: "
    if text.startswith(prefix):
        return text
    return f"{prefix}{text}"


def _model_name_from_config(config: Config) -> str:
    if config.pipeline.embedding_provider == "ollama":
        return config.ollama.model
    return config.llama_cpp.model_path


def _uses_nomic_v2_model(model_identifier: str) -> bool:
    model_name = Path(model_identifier).name.lower()
    return "nomic-embed-text-v2" in model_name
