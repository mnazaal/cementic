"""Contract tests for the data-keyed registries (extractor + embedding provider).

These assert the *seam* itself rather than any one implementation: every
registered entry must satisfy the contract its registry promises. They make
"open for extension" a tested invariant, so a second backend gets these checks
for free instead of relying on the single current implementation.
"""

from __future__ import annotations

import pytest

from cementic.config import Config, ExtractionConfig
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

#: The registry holds two kinds of entry, and most contracts apply to one of
#: them. Split once here rather than branching inside every test.
_STATIC_EXTRACTORS = [n for n, (spec, _) in _EXTRACTORS.items() if spec.extensions is not None]
_CONFIG_DRIVEN_EXTRACTORS = [n for n, (spec, _) in _EXTRACTORS.items() if spec.extensions is None]


class TestExtractorRegistryContract:
    """Every `_EXTRACTORS` entry must be a well-formed, dispatchable extractor."""

    @pytest.mark.parametrize("name", list(_EXTRACTORS))
    def test_spec_is_well_formed(self, name: str) -> None:
        spec, fn = _EXTRACTORS[name]
        assert spec.name == name, "registry key must equal the spec name"
        assert spec.version >= 1
        assert callable(fn)
        if spec.extensions is None:
            return  # config-driven; its file types are asserted below
        assert spec.extensions, "a static extractor must handle at least one extension"
        for ext in spec.extensions:
            assert ext.startswith("."), f"{ext!r} must be a dotted suffix"
            assert ext == ext.lower(), f"{ext!r} must be lowercase"

    def test_supported_extensions_is_the_union(self) -> None:
        expected = {
            ext for spec, _ in _EXTRACTORS.values() if spec.extensions for ext in spec.extensions
        }
        assert supported_extensions() == expected

    @pytest.mark.parametrize("name", _STATIC_EXTRACTORS)
    def test_each_declared_extension_dispatches(self, name: str) -> None:
        config = Config()
        spec, _ = _EXTRACTORS[name]
        assert spec.extensions is not None
        for ext in spec.extensions:
            resolved = extractor_for(f"/some/file{ext}", config)
            assert resolved is not None, f"{ext} should resolve to an extractor"
            _resolved_name, fn = resolved
            assert callable(fn)

    def test_fingerprint_payload_covers_every_static_extractor(self) -> None:
        payload_names = {entry["name"] for entry in extractor_registry_payload()}
        assert payload_names == set(_STATIC_EXTRACTORS)


class TestConfigDrivenExtractorContract:
    """The second kind of registry entry: file types come from config.

    A config-driven extractor exists so a command can handle a type no built-in
    extractor knows. That freedom is why it must stay out of both the fallback
    and the unselected fingerprint -- "handles anything" would otherwise make it
    the default for everything, and re-version every revision on the day it was
    added.
    """

    @staticmethod
    def _configured(file_type: str = "pdf") -> Config:
        return Config(
            extraction=ExtractionConfig(
                backends={file_type: "command"},
                commands={file_type: ["true", "{path}"]},
                command_versions={file_type: ["true"]},
            )
        )

    @pytest.mark.parametrize("name", _CONFIG_DRIVEN_EXTRACTORS)
    def test_declares_no_static_extensions(self, name: str) -> None:
        assert _EXTRACTORS[name][0].extensions is None

    def test_is_never_reached_by_fallback(self) -> None:
        """Unnamed, it must not capture a type a built-in extractor handles."""
        name, _fn = extractor_for("/some/file.pdf", Config(extraction=ExtractionConfig()))
        assert name not in _CONFIG_DRIVEN_EXTRACTORS

    def test_an_unknown_type_is_unhandled_until_configured(self) -> None:
        assert extractor_for("/some/file.epub", Config(extraction=ExtractionConfig())) is None

    def test_dispatches_once_named(self) -> None:
        resolved = extractor_for("/some/file.epub", self._configured("epub"))
        assert resolved is not None
        assert resolved[0] == "command"

    def test_config_extends_supported_extensions(self) -> None:
        assert ".epub" not in supported_extensions()
        assert ".epub" in supported_extensions(self._configured("epub"))

    def test_absent_from_the_payload_until_named(self) -> None:
        """Excluded when unselected, or adding it would rebuild every corpus."""
        bare = Config(extraction=ExtractionConfig())
        unselected = {entry["name"] for entry in extractor_registry_payload(bare)}
        assert unselected == set(_STATIC_EXTRACTORS)

        selected = {entry["name"] for entry in extractor_registry_payload(self._configured())}
        assert selected == set(_STATIC_EXTRACTORS) | {"command"}


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
