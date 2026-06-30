"""Pure, model-keyed policy for formatting embedding input text.

The decision of *whether* to apply a model-specific prefix belongs to the
embedding provider (which knows its own model); these functions are the pure
helpers a provider delegates to. Nothing here reads global config.
"""

from __future__ import annotations

from pathlib import Path


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


def _uses_nomic_v2_model(model_identifier: str) -> bool:
    model_name = Path(model_identifier).name.lower()
    return "nomic-embed-text-v2" in model_name
