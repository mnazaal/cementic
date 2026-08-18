"""Pure, model-keyed policy for formatting embedding input text.

The decision of *whether* to apply a model-specific prefix belongs to the
embedding provider (which knows its own model); these functions are the pure
helpers a provider delegates to. Nothing here reads global config.

Asymmetric models (the Nomic family among them) are trained with different prefixes for
documents and queries, and embedding either without its prefix measurably
degrades retrieval -- silently, since every query still returns *something*.
The policy is therefore part of the embedding profile's identity: see
``EMBEDDING_TEXT_FORMAT_VERSION`` in ``profiles.py``, which must be bumped
whenever the rules below change, or old and new vectors would be mixed in one
profile.
"""

from __future__ import annotations

from pathlib import Path

#: Substring identifying models that require asymmetric task prefixes.
#:
#: The whole nomic-embed-text family (v1, v1.5, v2) is trained with the same
#: ``search_document:`` / ``search_query:`` prefixes. Matching only the v2
#: filename silently embedded v1/v1.5 without them -- measurably worse
#: retrieval, no error -- with the v1.5 GGUF sitting in this very repo.
#:
#: Matching on the model *filename* is a known weakness: renaming the GGUF, or
#: pointing llama_model_url at a mirror that serves a different filename, turns
#: prefixing off with no error and no log line -- search keeps working, just
#: worse. Reading the model's real identity from GGUF metadata or /v1/models
#: would be sounder; until then ``describe_text_policy`` exists so callers can
#: report which policy was actually selected.
_NOMIC_FAMILY_MARKER = "nomic-embed-text"

#: Policy name reported when no model-specific formatting applies.
PLAIN_POLICY = "plain"
NOMIC_TASK_PREFIX_POLICY = "nomic-task-prefix"


def format_document_text_for_model(text: str, model_identifier: str) -> str:
    """Format document text for a specific embedding model."""
    return _apply_nomic_prefix(text, "search_document", model_identifier)


def format_query_text_for_model(text: str, model_identifier: str) -> str:
    """Format query text for a specific embedding model."""
    return _apply_nomic_prefix(text, "search_query", model_identifier)


def describe_text_policy(model_identifier: str) -> str:
    """Name the formatting policy selected for a model (pure).

    Lets callers surface the choice, so a model whose filename stopped matching
    is visible rather than quietly degrading retrieval quality.
    """
    return NOMIC_TASK_PREFIX_POLICY if _uses_nomic_model(model_identifier) else PLAIN_POLICY


def _apply_nomic_prefix(text: str, task: str, model_identifier: str) -> str:
    if not _uses_nomic_model(model_identifier):
        return text

    prefix = f"{task}: "
    if text.startswith(prefix):
        return text
    return f"{prefix}{text}"


def _uses_nomic_model(model_identifier: str) -> bool:
    model_name = Path(model_identifier).name.lower()
    return _NOMIC_FAMILY_MARKER in model_name
