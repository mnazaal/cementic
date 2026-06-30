"""Contract tests for the data-keyed registries (extractor + embedding provider).

These assert the *seam* itself rather than any one implementation: every
registered entry must satisfy the contract its registry promises. They make
"open for extension" a tested invariant, so a second backend gets these checks
for free instead of relying on the single current implementation.
"""

from __future__ import annotations

import pytest

from cementic.config import Config
from cementic.embedding_provider import EmbeddingProvider
from cementic.embedding_runtime import (
    _PROVIDER_FACTORIES,
    RemoteEmbeddingClient,
    create_provider,
)
from cementic.extract import (
    _EXTRACTORS,
    extractor_for,
    extractor_registry_payload,
    supported_extensions,
)


class TestExtractorRegistryContract:
    """Every `_EXTRACTORS` entry must be a well-formed, dispatchable extractor."""

    @pytest.mark.parametrize("name", list(_EXTRACTORS))
    def test_spec_is_well_formed(self, name: str) -> None:
        spec, fn = _EXTRACTORS[name]
        assert spec.name == name, "registry key must equal the spec name"
        assert spec.version >= 1
        assert spec.extensions, "an extractor must handle at least one extension"
        for ext in spec.extensions:
            assert ext.startswith("."), f"{ext!r} must be a dotted suffix"
            assert ext == ext.lower(), f"{ext!r} must be lowercase"
        assert callable(fn)

    def test_supported_extensions_is_the_union(self) -> None:
        expected = {ext for spec, _ in _EXTRACTORS.values() for ext in spec.extensions}
        assert supported_extensions() == expected

    @pytest.mark.parametrize("name", list(_EXTRACTORS))
    def test_each_declared_extension_dispatches(self, name: str) -> None:
        config = Config()
        spec, _ = _EXTRACTORS[name]
        for ext in spec.extensions:
            resolved = extractor_for(f"/some/file{ext}", config)
            assert resolved is not None, f"{ext} should resolve to an extractor"
            _resolved_name, fn = resolved
            assert callable(fn)

    def test_fingerprint_payload_covers_every_extractor(self) -> None:
        payload_names = {entry["name"] for entry in extractor_registry_payload()}
        assert payload_names == set(_EXTRACTORS)


class TestEmbeddingProviderRegistryContract:
    """The provider factory seam must honor the ABC contract for every backend."""

    @pytest.mark.parametrize("name", list(_PROVIDER_FACTORIES))
    def test_runtime_factory_is_callable(self, name: str) -> None:
        assert callable(_PROVIDER_FACTORIES[name])

    @pytest.mark.parametrize("cls", [RemoteEmbeddingClient])
    def test_provider_class_fully_implements_abc(
        self, cls: type[EmbeddingProvider]
    ) -> None:
        assert issubclass(cls, EmbeddingProvider)
        # No leftover abstract methods => the class is a complete provider.
        assert cls.__abstractmethods__ == frozenset(), (
            f"{cls.__name__} leaves abstract methods unimplemented: "
            f"{sorted(cls.__abstractmethods__)}"
        )

    def test_unknown_provider_is_rejected(self) -> None:
        from cementic.embedding_runtime import EmbeddingRuntimeSpec

        spec = EmbeddingRuntimeSpec(
            provider="does-not-exist",
            model_identifier="x",
            embedding_dim=8,
        )
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            create_provider(spec, config=Config())
