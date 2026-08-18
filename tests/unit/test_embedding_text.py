"""Tests for embedding text formatting policy and provider delegation.

Formatting is a pure, model-keyed policy that providers delegate to; nothing
here reads global config.
"""

from cementic.embedding_provider import EmbeddingProvider
from cementic.embedding_text import (
    format_document_text_for_model,
    format_query_text_for_model,
)

NOMIC_V2 = "models/nomic-embed-text-v2-moe.Q8_0.gguf"
NOMIC_V15 = "models/nomic-embed-text-v1.5.f16.gguf"


def test_adds_nomic_v2_document_prefix() -> None:
    assert format_document_text_for_model("hello", NOMIC_V2) == "search_document: hello"


def test_adds_nomic_v2_query_prefix() -> None:
    assert format_query_text_for_model("hello", NOMIC_V2) == "search_query: hello"


def test_v1_and_v15_models_get_the_same_prefixes() -> None:
    """Regression: the marker matched only v2 filenames, so pointing model_path
    at the v1.5 GGUF in this repo silently embedded without the prefixes the
    whole nomic-embed-text family is trained with -- measurably worse
    retrieval, no error. The old behaviour was even pinned by a test here,
    which made the defect read as intentional."""
    assert format_document_text_for_model("hello", NOMIC_V15) == "search_document: hello"
    assert format_query_text_for_model("hello", NOMIC_V15) == "search_query: hello"
    assert (
        format_document_text_for_model("hello", "nomic-embed-text-v1.Q4_0.gguf")
        == "search_document: hello"
    )


def test_does_not_change_non_nomic_text() -> None:
    assert format_document_text_for_model("hello", "all-MiniLM-L6-v2.gguf") == "hello"
    assert format_query_text_for_model("hello", "bge-small-en.gguf") == "hello"


def test_no_double_prefix_document_text() -> None:
    """Already-prefixed text should not get a second prefix."""
    assert (
        format_document_text_for_model("search_document: hello", NOMIC_V2)
        == "search_document: hello"
    )


def test_no_double_prefix_query_text() -> None:
    """Already-prefixed query text should not get a second prefix."""
    assert format_query_text_for_model("search_query: hello", NOMIC_V2) == "search_query: hello"


def test_provider_default_format_is_identity() -> None:
    """The base contract's formatting defaults to identity."""

    class _Stub(EmbeddingProvider):
        def embed(self, text: str) -> list[float]:
            return []

        def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
            return []

        def health_check(self) -> bool:
            return True

        @property
        def embedding_dim(self) -> int:
            return 1

    provider = _Stub()
    assert provider.format_document("hello") == "hello"
    assert provider.format_query("hello") == "hello"


def test_provider_delegates_nomic_prefix_by_its_own_model() -> None:
    """A provider applies its model's prefix without any config branching."""

    class _NomicProvider(EmbeddingProvider):
        name = "stub"

        def __init__(self, model: str) -> None:
            self._model = model

        def format_document(self, text: str) -> str:
            return format_document_text_for_model(text, self._model)

        def format_query(self, text: str) -> str:
            return format_query_text_for_model(text, self._model)

        def embed(self, text: str) -> list[float]:
            return []

        def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
            return []

        def health_check(self) -> bool:
            return True

        @property
        def embedding_dim(self) -> int:
            return 1

    provider = _NomicProvider("nomic-embed-text-v2-moe")
    assert provider.format_document("hello") == "search_document: hello"
    assert provider.format_query("hello") == "search_query: hello"
