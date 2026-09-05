"""Tests for the embedding-server canary."""

from __future__ import annotations

import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from cementic.canary import (
    CANARY_TEXTS,
    CanaryVerdict,
    canary_payload,
    canary_request_texts,
    capture_canary,
    compare_canary_vectors,
    decode_texts,
    decode_vectors,
    describe_canary_verdict,
    load_canary,
)
from cementic.db import Base, EmbeddingProfile, EmbeddingProfileCanary
from cementic.embedding_provider import EmbeddingProvider


class FakeProvider(EmbeddingProvider):
    """A provider whose vectors and failures the test dictates."""

    name = "fake"

    def __init__(self, vectors: list[list[float]] | None = None, error: Exception | None = None):
        self._vectors = vectors or [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]
        self._error = error
        self.exact_calls: list[list[str]] = []

    def format_document(self, text: str) -> str:
        return f"search_document: {text}"

    def embed(self, text: str) -> list[float]:
        return self._vectors[0]

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        raise AssertionError("the canary must not travel the throughput path")

    def embed_exact(self, texts: list[str]) -> list[list[float]]:
        if self._error is not None:
            raise self._error
        self.exact_calls.append(list(texts))
        return self._vectors

    def server_build(self) -> str | None:
        return "b10605-a130532ae"

    @property
    def embedding_dim(self) -> int:
        return 2


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)()


def _profile(session: Session) -> EmbeddingProfile:
    profile = EmbeddingProfile(
        fingerprint="fp-1",
        provider="llama-cpp",
        model_identifier="nomic-embed-text-v2-moe.Q8_0.gguf",
        embedding_dim=2,
        config_json="{}",
    )
    session.add(profile)
    session.flush()
    return profile


class TestComparison:
    """The comparison is strict: any difference is reported, and its size only
    decides how the difference is described. Measurement is what earns that --
    one build answering one fixed request is bitwise reproducible, so there is
    no rounding band to forgive."""

    STORED = [[1.0, 0.0], [0.0, 1.0]]

    def test_identical_vectors_are_identical(self) -> None:
        verdict = compare_canary_vectors(self.STORED, [[1.0, 0.0], [0.0, 1.0]])

        assert verdict.identical
        assert verdict.kind == "identical"
        assert verdict.worst_abs_diff == 0.0

    def test_the_servers_own_scheduling_noise_stays_comparable(self) -> None:
        """Measured on the live server: replaying a canary after a 3-text
        request moved it to cosine 0.999908. Bit equality would report that as
        drift, so the check has to sit above the server's own scheduling."""
        verdict = compare_canary_vectors(self.STORED, [[1.0, 1e-9], [0.0, 1.0]])

        assert not verdict.identical
        assert verdict.kind == "scheduling"
        assert verdict.comparable
        assert verdict.worst_abs_diff == pytest.approx(1e-9)

    def test_a_changed_function_is_reported_as_semantic(self) -> None:
        """A tokenizer or pooling change moves vectors far past rounding."""
        verdict = compare_canary_vectors(self.STORED, [[0.0, 1.0], [0.0, 1.0]])

        assert verdict.kind == "semantic"
        assert not verdict.comparable
        assert verdict.worst_cosine == pytest.approx(0.0)

    def test_a_shape_mismatch_is_an_error_not_drift(self) -> None:
        with pytest.raises(ValueError, match="server returned 1"):
            compare_canary_vectors(self.STORED, [[1.0, 0.0]])

    def test_a_dimension_mismatch_names_the_vector(self) -> None:
        with pytest.raises(ValueError, match="canary vector 1"):
            compare_canary_vectors(self.STORED, [[1.0, 0.0], [0.0, 1.0, 0.0]])


class TestStorageRoundTrip:
    """The strict comparison is only as exact as storage is. JSON is chosen
    because it round-trips float64 exactly; if that ever stopped holding, every
    canary would report drift that never happened."""

    def test_json_round_trip_preserves_bit_exactness(self) -> None:
        vectors = [[0.1, 0.2, 1 / 3], [-1e-17, 5e300, 0.30000000000000004]]

        restored = decode_vectors(canary_payload(vectors))

        assert restored == vectors
        assert compare_canary_vectors(vectors, restored).identical

    def test_texts_round_trip(self) -> None:
        texts = ["search_document: a\nb", ". . . 117 C.5.4"]

        assert decode_texts(json.dumps(texts)) == texts


class TestRequestShape:
    """Capture stores the strings actually sent, so a replay re-sends the same
    request rather than re-deriving it through a formatting policy that may
    have moved in between."""

    def test_the_request_is_the_documents_as_the_model_wants_them(self) -> None:
        texts = canary_request_texts(FakeProvider())

        assert texts == [f"search_document: {text}" for text in CANARY_TEXTS]

    def test_capture_stores_that_exact_request(self, session: Session) -> None:
        provider = FakeProvider()

        canary = capture_canary(session, _profile(session), provider, server_build="b1")

        assert canary is not None
        assert decode_texts(canary.texts_json) == provider.exact_calls[0]


class TestCapture:
    """Capture is lazy and once per profile, so a profile built before canaries
    existed gains one on the next indexing run instead of staying permanently
    uncheckable."""

    def test_capture_records_vectors_and_the_server_build(self, session: Session) -> None:
        profile = _profile(session)

        canary = capture_canary(session, profile, FakeProvider(), server_build="b10605-a130532ae")

        assert canary is not None
        assert canary.server_build == "b10605-a130532ae"
        assert decode_vectors(canary.vectors_json)[0] == [1.0, 0.0]
        assert load_canary(session, profile.id) is not None

    def test_a_second_capture_keeps_the_first(self, session: Session) -> None:
        """Re-stamping must be deliberate: silently overwriting on every run
        would make the canary agree with whatever server ran last, which is
        precisely the drift it exists to notice."""
        profile = _profile(session)
        capture_canary(session, profile, FakeProvider(), server_build="b10605")

        again = capture_canary(
            session, profile, FakeProvider(vectors=[[9.0, 9.0]] * 4), server_build="b10818"
        )

        assert again is not None
        assert again.server_build == "b10605"
        assert decode_vectors(again.vectors_json)[0] == [1.0, 0.0]
        assert session.query(EmbeddingProfileCanary).count() == 1

    def test_an_unreachable_server_does_not_block_indexing(self, session: Session) -> None:
        profile = _profile(session)

        canary = capture_canary(
            session,
            profile,
            FakeProvider(error=RuntimeError("connection refused")),
            server_build=None,
        )

        assert canary is None
        assert session.query(EmbeddingProfileCanary).count() == 0


class TestDescription:
    """Every message has to say what to do next; a diagnostic that reports a
    number and no action gets ignored."""

    def test_identical_names_the_build_it_was_recorded_under(self) -> None:
        message = describe_canary_verdict(CanaryVerdict(True, 1.0, 0.0), "b10605-a130532ae")

        assert "reproduces" in message
        assert "b10605-a130532ae" in message

    def test_scheduling_noise_reads_as_reproduction_not_drift(self) -> None:
        message = describe_canary_verdict(CanaryVerdict(False, 0.9999999, 2e-9), "b10605")

        assert "scheduling noise" in message
        assert "rebuild" not in message

    def test_semantic_drift_says_the_vectors_are_not_comparable(self) -> None:
        message = describe_canary_verdict(CanaryVerdict(False, 0.7, 0.4), "b10605")

        assert "not comparable" in message
        assert "rebuild" in message


class TestProviderDefault:
    """`embed_exact` must not depend on configuration. The default implementation
    is one text at a time; the remote client overrides it with one request."""

    def test_the_default_never_reaches_the_batch_path(self) -> None:
        class OneAtATime(FakeProvider):
            embed_exact = EmbeddingProvider.embed_exact  # type: ignore[assignment]

        assert OneAtATime().embed_exact(["a", "b"]) == [[1.0, 0.0], [1.0, 0.0]]

    def test_a_provider_without_a_runtime_reports_no_build(self) -> None:
        class Bare(FakeProvider):
            server_build = EmbeddingProvider.server_build  # type: ignore[assignment]

        assert Bare().server_build() is None
