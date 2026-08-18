"""Tests for profile fingerprinting and resolution helpers."""

from unittest.mock import MagicMock

from cementic.config import Config
from cementic.profiles import (
    _fingerprint,
    _stable_json,
    build_chunk_profile_payload,
    build_embedding_profile_payload,
    build_extractor_profile_payload,
    get_or_create_chunk_profile,
    get_or_create_embedding_profile,
    get_or_create_extractor_profile,
)


class TestHelpers:
    """Tests for internal profile helpers."""

    def test_stable_json_is_deterministic(self) -> None:
        payload = {"z": 1, "a": 2, "nested": {"c": 3, "b": 4}}
        out1 = _stable_json(payload)
        out2 = _stable_json(payload)
        assert out1 == out2
        assert "z" not in out1.split(",")[0]  # keys sorted alphabetically

    def test_fingerprint_is_stable(self) -> None:
        fp1 = _fingerprint({"key": "value"})
        fp2 = _fingerprint({"key": "value"})
        assert fp1 == fp2
        assert len(fp1) == 64  # sha256 hex

    def test_fingerprint_changes_with_contents(self) -> None:
        fp1 = _fingerprint({"key": "a"})
        fp2 = _fingerprint({"key": "b"})
        assert fp1 != fp2


class TestBuildExtractorProfilePayload:
    """Tests for build_extractor_profile_payload."""

    def test_uses_config_fields(self) -> None:
        config = Config()
        config.extraction.backends = {"pdf": "pdfplumber"}
        config.extraction.use_ocr = True
        payload = build_extractor_profile_payload(config)
        assert payload["backends"] == {"pdf": "pdfplumber"}
        assert payload["use_ocr"] is True
        assert "version" in payload


class TestBuildChunkProfilePayload:
    """Tests for build_chunk_profile_payload."""

    def test_uses_config_fields(self) -> None:
        config = Config()
        config.pipeline.chunk_size = 256
        config.pipeline.chunk_overlap = 64
        payload = build_chunk_profile_payload(config)
        assert payload["chunk_size"] == 256
        assert payload["chunk_overlap"] == 64
        assert payload["tokenizer"] == "cl100k_base"
        assert "version" in payload


class TestBuildEmbeddingProfilePayload:
    """Tests for build_embedding_profile_payload."""

    def test_llama_cpp_provider(self) -> None:
        config = Config()
        config.pipeline.embedding_provider = "llama-cpp"
        config.llama_cpp.model_path = "/models/test.gguf"
        config.llama_cpp.embedding_dim = 384
        config.llama_cpp.n_ctx = 512
        config.llama_cpp.n_gpu_layers = 0
        payload = build_embedding_profile_payload(config)
        assert payload["provider"] == "llama-cpp"
        assert payload["model_identifier"].endswith("test.gguf")
        assert payload["embedding_dim"] == 384
        assert payload["n_ctx"] == 512
        assert payload["distance_metric"] == "cosine"

    def test_provider_facts_override_config_dim(self) -> None:
        """A live provider's self-described facts take precedence over config."""
        from cementic.embedding_provider import EmbeddingFacts

        config = Config()
        config.pipeline.embedding_provider = "llama-cpp"
        config.llama_cpp.embedding_dim = 768

        provider = MagicMock()
        provider.describe.return_value = EmbeddingFacts(
            name="llama-cpp", embedding_dim=1024, distance_metric="cosine"
        )

        payload = build_embedding_profile_payload(config, provider)
        # Dimension comes from the provider (probed), not config.
        assert payload["embedding_dim"] == 1024
        # Runtime-identity fields still come from the config-derived spec.
        assert payload["n_ctx"] == config.llama_cpp.n_ctx


class TestGetOrCreateExtractorProfile:
    """Tests for get_or_create_extractor_profile."""

    def test_returns_existing(self) -> None:
        session = MagicMock()
        session.query().filter_by().first.return_value = MagicMock()
        config = Config()

        result = get_or_create_extractor_profile(session, config)
        assert result is not None

    def test_creates_new(self) -> None:
        session = MagicMock()
        session.query().filter_by().first.return_value = None
        config = Config()

        result = get_or_create_extractor_profile(session, config)
        assert result is not None
        assert session.add.called
        assert session.flush.called


class TestGetOrCreateChunkProfile:
    """Tests for get_or_create_chunk_profile."""

    def test_returns_existing(self) -> None:
        session = MagicMock()
        session.query().filter_by().first.return_value = MagicMock()
        config = Config()

        result = get_or_create_chunk_profile(session, config)
        assert result is not None

    def test_creates_new(self) -> None:
        session = MagicMock()
        session.query().filter_by().first.return_value = None
        config = Config()

        result = get_or_create_chunk_profile(session, config)
        assert result is not None
        assert session.add.called
        assert session.flush.called


class TestGetOrCreateEmbeddingProfile:
    """Tests for get_or_create_embedding_profile."""

    def test_returns_existing(self) -> None:
        session = MagicMock()
        session.query().filter_by().first.return_value = MagicMock()
        config = Config()

        result = get_or_create_embedding_profile(session, config)
        assert result is not None

    def test_creates_new_uses_payload_embedding_dim(self) -> None:
        """Regression: embedding_dim must come from payload, not config directly."""
        session = MagicMock()
        session.query().filter_by().first.return_value = None
        config = Config()
        config.pipeline.embedding_provider = "llama-cpp"
        config.llama_cpp.embedding_dim = 999

        result = get_or_create_embedding_profile(session, config)
        assert result is not None
        # The result object was instantiated with embedding_dim from payload
        call_kwargs = session.add.call_args
        assert call_kwargs is not None
        profile = call_kwargs[0][0]
        # Value comes from payload (which equals config since freshly built),
        # not hardcoded from config attribute path
        assert profile.embedding_dim == 999


class TestTextPolicyIsPartOfEmbeddingIdentity:
    """The task-prefix policy is selected from the model *filename*.

    Renaming a GGUF, or mirroring it under another name, silently switches to
    plain text. While the policy was absent from the payload, prefixed and
    unprefixed corpora shared one profile and one vector table -- two
    incompatible vector spaces in a single index, which is what the sibling
    text_format_version exists to prevent.
    """

    def _payload_for(self, model_path: str) -> dict:
        config = Config()
        config.llama_cpp.model_path = model_path
        return build_embedding_profile_payload(config)

    def test_policy_is_recorded(self):
        payload = self._payload_for("models/nomic-embed-text-v2-moe.Q8_0.gguf")

        assert payload["text_policy"] == "nomic-task-prefix"

    def test_a_renamed_model_records_a_different_policy(self):
        payload = self._payload_for("models/renamed.gguf")

        assert payload["text_policy"] == "plain"

    def test_a_rename_forks_the_profile_instead_of_sharing_its_table(self):
        prefixed = self._payload_for("models/nomic-embed-text-v2-moe.Q8_0.gguf")
        renamed = self._payload_for("models/renamed.gguf")

        assert _fingerprint(prefixed) != _fingerprint(renamed)
