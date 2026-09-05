"""Reference vectors that catch an embedding server changing under a profile.

The embedding profile fingerprints the model, its digest, the dimension, the
metric and the context window -- everything cementic *configures*. It cannot
fingerprint the llama.cpp build that ran them, which is knowable only by asking
a running server, so swapping `llama-server` for a newer one rewrites vectors
under an unchanged fingerprint.

Recording the build in the profile would fork the corpus on every upgrade, and
measurement says that is the wrong trade: b10605 against b10818, 213 builds
apart, returned bitwise-identical vectors for 200 real chunks on both CPU and
Vulkan. Drift is therefore rare enough to *detect* rather than to version
against -- which is what this module does. A profile stores a handful of texts
with the vectors one server produced for them; `cementic doctor` replays that
exact request and compares.

The comparison is by cosine, not by bits, and the reason is measured. A fixed
request is *not* reproducible on a busy server: llama-server packs concurrent
slot work into unified batches, so a canary replayed after a 3-text request
differs from one replayed after none (cosine 0.999908 against 1.0, measured
2026-09-05). Batch composition moves vectors the same way, at ~2.3e-3. Bit
equality would therefore fire on the server's own scheduling, which is the
false alarm that gets a check ignored.

What the noise cannot do is cross the gap to a *changed function*: a tokenizer,
pooling or normalisation change moves cosine to the low 0.9s or worse, two
orders of magnitude past the worst noise observed here. `SEMANTIC_COSINE_FLOOR`
sits in that gap.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy.orm import Session

from cementic.db import EmbeddingProfile, EmbeddingProfileCanary
from cementic.embedding_provider import EmbeddingProvider

#: Texts embedded as one fixed request to characterise a server's behaviour.
#: Chosen to span what the corpus actually contains -- prose, mathematics, code,
#: and a dot-leader contents line -- because a tokenizer change shows up in
#: punctuation and symbols long before it shows up in plain English. They are
#: never re-generated: changing this tuple invalidates every stored canary, so
#: treat it as frozen data, not a knob.
CANARY_TEXTS: tuple[str, ...] = (
    "The variational autoencoder optimises the evidence lower bound.",
    "Let $X_1, \\dots, X_n$ be i.i.d. with $\\mathbb{E}[X_i] = \\mu < \\infty$.",
    "def forward(self, x: Tensor) -> Tensor:\n    return self.norm(x + self.attn(x))",
    ". . . . . . . . . . . . . . . 117 C.5.4 Proof of Claim 19",
)

#: Below this cosine, the server is computing a different function rather than
#: scheduling differently -- a tokenizer, pooling or normalisation change.
#: Measured noise to clear: 2.0e-4 (1 - cosine) from slot/batch state left by
#: preceding traffic, and 2.3e-4 .. 2.3e-3 from request composition. The floor
#: sits ~100x above that and still ~10x below any plausible semantic change,
#: because both bounds were sampled from a handful of shapes, not derived.
SEMANTIC_COSINE_FLOOR = 0.99


@dataclass(frozen=True)
class CanaryVerdict:
    """What replaying a stored canary against the live server showed."""

    #: True when every vector matched bit for bit, which is the expected result.
    identical: bool
    #: Cosine of the worst-matching pair; 1.0 when identical.
    worst_cosine: float
    #: Largest single-component difference across all pairs.
    worst_abs_diff: float

    @property
    def kind(self) -> str:
        """`identical`, `scheduling` (the server's own noise), or `semantic`."""
        if self.identical:
            return "identical"
        return "scheduling" if self.worst_cosine >= SEMANTIC_COSINE_FLOOR else "semantic"

    @property
    def comparable(self) -> bool:
        """Whether these vectors still belong in the same space as the indexed ones."""
        return self.kind != "semantic"


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine between two vectors; 0.0 if either has no magnitude."""
    dot = sum(x * y for x, y in zip(left, right, strict=True))
    norms = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(y * y for y in right))
    return dot / norms if norms else 0.0


def compare_canary_vectors(
    stored: Sequence[Sequence[float]], observed: Sequence[Sequence[float]]
) -> CanaryVerdict:
    """Compare a stored canary against a freshly embedded one (pure).

    Raises when the two disagree in shape: that is a programming error (a
    changed `CANARY_TEXTS`, or a model of another dimension), not drift, and
    reporting it as drift would send the reader looking at the wrong thing.
    """
    if len(stored) != len(observed):
        raise ValueError(f"canary has {len(stored)} vectors, server returned {len(observed)}")
    for index, (was, now) in enumerate(zip(stored, observed, strict=True)):
        if len(was) != len(now):
            raise ValueError(
                f"canary vector {index} has {len(was)} dimensions, server returned {len(now)}"
            )
    if list(stored) == [list(vector) for vector in observed]:
        return CanaryVerdict(identical=True, worst_cosine=1.0, worst_abs_diff=0.0)
    pairs = list(zip(stored, observed, strict=True))
    return CanaryVerdict(
        identical=False,
        worst_cosine=min(cosine_similarity(was, now) for was, now in pairs),
        worst_abs_diff=max(
            max(abs(x - y) for x, y in zip(was, now, strict=True)) for was, now in pairs
        ),
    )


def describe_canary_verdict(verdict: CanaryVerdict, recorded_build: str | None) -> str:
    """One line saying what moved and what to do about it (pure)."""
    if verdict.identical:
        served = f" (recorded under {recorded_build})" if recorded_build else ""
        return f"the server reproduces this profile's vectors exactly{served}"
    origin = f" recorded under {recorded_build}" if recorded_build else ""
    if verdict.kind == "scheduling":
        return (
            f"the server reproduces this profile's vectors{origin} to within its own "
            f"scheduling noise (worst cosine {verdict.worst_cosine:.9f}, worst component "
            f"{verdict.worst_abs_diff:.2e})"
        )
    return (
        f"the server computes different embeddings from those{origin} this profile was built "
        f"with (worst cosine {verdict.worst_cosine:.9f}); its vectors are not comparable to the "
        "indexed ones, so rebuild the collection or restore the previous server"
    )


def canary_payload(vectors: Sequence[Sequence[float]]) -> str:
    """Serialise vectors for storage.

    JSON rather than a packed binary: `json.dumps` round-trips float64 exactly
    (it writes `repr`), so the strict comparison survives storage, and a stored
    canary stays readable by anything that can read the row.
    """
    return json.dumps([list(vector) for vector in vectors])


def decode_vectors(vectors_json: str) -> list[list[float]]:
    """The vectors a stored canary holds."""
    return [[float(value) for value in vector] for vector in json.loads(vectors_json)]


def decode_texts(texts_json: str) -> list[str]:
    """The exact request a stored canary was taken from."""
    return [str(text) for text in json.loads(texts_json)]


def canary_request_texts(provider: EmbeddingProvider) -> list[str]:
    """The strings actually sent for a canary: `CANARY_TEXTS` as documents.

    Formatting is applied at capture time and the *result* is stored, so a
    replay re-sends the same bytes. Prefix-policy drift is deliberately out of
    scope here -- it changes `model_identifier`, which the profile fingerprint
    and the daemon's model check already cover.
    """
    return [provider.format_document(text) for text in CANARY_TEXTS]


def capture_canary(
    session: Session,
    profile: EmbeddingProfile,
    provider: EmbeddingProvider,
    *,
    server_build: str | None,
) -> EmbeddingProfileCanary | None:
    """Store this profile's canary if it has none yet; return it either way.

    Capture is once per profile and lazy, so profiles built before this existed
    (including the one behind the live corpus) gain a canary on the next
    indexing run. That canary attests to the server present *then*, not to the
    one that built the corpus -- which is why `server_build` is recorded beside
    it rather than assumed.

    A failure to reach the server is not a failure to index: the canary is a
    diagnostic, and refusing to build a revision because a reference vector
    could not be taken would trade a real job for a nicety.
    """
    existing = (
        session.query(EmbeddingProfileCanary).filter_by(embedding_profile_id=profile.id).first()
    )
    if existing is not None:
        return existing
    texts = canary_request_texts(provider)
    try:
        vectors = provider.embed_exact(texts)
    except Exception:
        return None
    canary = EmbeddingProfileCanary(
        embedding_profile_id=profile.id,
        texts_json=json.dumps(texts),
        vectors_json=canary_payload(vectors),
        server_build=server_build,
    )
    session.add(canary)
    session.flush()
    return canary


def load_canary(session: Session, embedding_profile_id: int) -> EmbeddingProfileCanary | None:
    """The stored canary for one embedding profile, if it has one."""
    return (
        session.query(EmbeddingProfileCanary)
        .filter_by(embedding_profile_id=embedding_profile_id)
        .first()
    )
