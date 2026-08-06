"""Profile fingerprinting and resolution helpers."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import cast

from sqlalchemy.orm import Session

from cementic.chunk import TOKENIZER
from cementic.config import Config
from cementic.db import ChunkProfile, EmbeddingProfile, ExtractorProfile
from cementic.embedding_provider import EmbeddingFacts, EmbeddingProvider
from cementic.embedding_runtime import runtime_spec_from_config

#: Identity of the embedding-input formatting rules in embedding_text.py. Bump
#: this whenever those rules change: it is part of the embedding profile
#: fingerprint, and without a bump old and new vectors would be mixed in one
#: profile with no way to tell them apart.
EMBEDDING_TEXT_FORMAT_VERSION = "v1"
# v2: chunk_text no longer emits a duplicate tail chunk when a chunk ends exactly
# at the end of the text, so chunk output changed for boundary-length documents.
CHUNKING_VERSION = "v2"
EXTRACTION_VERSION = "v1"


def _stable_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _fingerprint(payload: dict[str, object]) -> str:
    return sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def build_extractor_profile_payload(config: Config) -> dict[str, object]:
    """Build extractor profile payload from config.

    Records the extractor registry identity so adding or revving a content-type
    extractor re-versions the revision. Imported locally to keep this module's
    import graph free of the (heavier) extraction backend.
    """
    from cementic.extract import extractor_registry_payload

    return {
        "backends": dict(sorted(config.extraction.backends.items())),
        "use_ocr": config.extraction.use_ocr,
        "version": EXTRACTION_VERSION,
        "extractors": extractor_registry_payload(),
    }


def build_chunk_profile_payload(config: Config) -> dict[str, object]:
    """Build chunk profile payload from config."""
    return {
        "chunk_size": config.pipeline.chunk_size,
        "chunk_overlap": config.pipeline.chunk_overlap,
        "tokenizer": TOKENIZER,
        "version": CHUNKING_VERSION,
    }


def build_embedding_profile_payload(
    config: Config, provider: EmbeddingProvider | None = None
) -> dict[str, object]:
    """Build embedding profile payload from config and the resolved provider.

    The provider self-describes its facts (backend name, embedding dimension,
    distance metric); with a live provider the dimension is what the model
    actually produces. Runtime-identity fields that change the vectors (model,
    context window, GPU offload) come from the config-derived spec. The
    payload is fully generic -- there is no per-provider branching.
    """
    spec = runtime_spec_from_config(config)
    if provider is not None:
        facts = provider.describe()
    else:
        facts = EmbeddingFacts(
            name=spec.provider,
            embedding_dim=spec.embedding_dim,
            distance_metric=spec.distance_metric,
        )
    return {
        "provider": facts.name,
        "model_identifier": spec.model_identifier,
        "embedding_dim": facts.embedding_dim,
        "distance_metric": facts.distance_metric,
        "n_ctx": spec.n_ctx,
        "n_gpu_layers": spec.n_gpu_layers,
        "verbose": spec.verbose,
        "text_format_version": EMBEDDING_TEXT_FORMAT_VERSION,
    }


def _extractor_profile_name(payload: dict[str, object]) -> str:
    """A short display name; the fingerprint carries the real identity."""
    backends = payload.get("backends") or {}
    if not isinstance(backends, dict) or not backends:
        return "default"
    return ",".join(f"{key}={value}" for key, value in sorted(backends.items()))


def get_or_create_extractor_profile(session: Session, config: Config) -> ExtractorProfile:
    """Resolve immutable extractor profile."""
    payload = build_extractor_profile_payload(config)
    fingerprint = _fingerprint(payload)
    profile = session.query(ExtractorProfile).filter_by(fingerprint=fingerprint).first()
    if profile is None:
        profile = ExtractorProfile(
            fingerprint=fingerprint,
            name=_extractor_profile_name(payload),
            config_json=_stable_json(payload),
        )
        session.add(profile)
        session.flush()
    return profile


def get_or_create_chunk_profile(session: Session, config: Config) -> ChunkProfile:
    """Resolve immutable chunk profile."""
    payload = build_chunk_profile_payload(config)
    fingerprint = _fingerprint(payload)
    profile = session.query(ChunkProfile).filter_by(fingerprint=fingerprint).first()
    if profile is None:
        profile = ChunkProfile(fingerprint=fingerprint, config_json=_stable_json(payload))
        session.add(profile)
        session.flush()
    return profile


def get_or_create_embedding_profile(
    session: Session, config: Config, provider: EmbeddingProvider | None = None
) -> EmbeddingProfile:
    """Resolve immutable embedding profile."""
    payload = build_embedding_profile_payload(config, provider)
    fingerprint = _fingerprint(payload)
    provider_name = str(payload["provider"])
    model_identifier = str(payload["model_identifier"])
    embedding_dim = int(cast(int, payload["embedding_dim"]))
    distance_metric = str(payload["distance_metric"])
    profile = session.query(EmbeddingProfile).filter_by(fingerprint=fingerprint).first()
    if profile is None:
        profile = EmbeddingProfile(
            fingerprint=fingerprint,
            provider=provider_name,
            model_identifier=model_identifier,
            embedding_dim=embedding_dim,
            distance_metric=distance_metric,
            config_json=_stable_json(payload),
        )
        session.add(profile)
        session.flush()
    return profile
