"""Profile fingerprinting and resolution helpers."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from sqlalchemy.orm import Session

from cementic.config import Config
from cementic.db import ChunkProfile, EmbeddingProfile, ExtractorProfile

EMBEDDING_TEXT_FORMAT_VERSION = "v1"
CHUNKING_VERSION = "v1"
EXTRACTION_VERSION = "v1"


def _stable_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _fingerprint(payload: dict[str, object]) -> str:
    return sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def build_extractor_profile_payload(config: Config) -> dict[str, object]:
    """Build extractor profile payload from config."""
    return {
        "backend": config.extraction.backend,
        "use_ocr": config.extraction.use_ocr,
        "version": EXTRACTION_VERSION,
    }


def build_chunk_profile_payload(config: Config) -> dict[str, object]:
    """Build chunk profile payload from config."""
    return {
        "chunk_size": config.pipeline.chunk_size,
        "chunk_overlap": config.pipeline.chunk_overlap,
        "tokenizer": "cl100k_base",
        "version": CHUNKING_VERSION,
    }


def build_embedding_profile_payload(config: Config) -> dict[str, object]:
    """Build embedding profile payload from config."""
    if config.pipeline.embedding_provider == "ollama":
        model_identifier = config.ollama.model
        embedding_dim = config.ollama.embedding_dim
        provider = "ollama"
        payload = {
            "provider": provider,
            "host": config.ollama.host,
            "model_identifier": model_identifier,
            "embedding_dim": embedding_dim,
            "distance_metric": "cosine",
            "text_format_version": EMBEDDING_TEXT_FORMAT_VERSION,
        }
    else:
        model_identifier = str(Path(config.llama_cpp.model_path))
        embedding_dim = config.llama_cpp.embedding_dim
        provider = "llama-cpp"
        payload = {
            "provider": provider,
            "model_identifier": model_identifier,
            "embedding_dim": embedding_dim,
            "distance_metric": "cosine",
            "n_ctx": config.llama_cpp.n_ctx,
            "n_gpu_layers": config.llama_cpp.n_gpu_layers,
            "verbose": config.llama_cpp.verbose,
            "text_format_version": EMBEDDING_TEXT_FORMAT_VERSION,
        }

    return payload


def get_or_create_extractor_profile(session: Session, config: Config) -> ExtractorProfile:
    """Resolve immutable extractor profile."""
    payload = build_extractor_profile_payload(config)
    fingerprint = _fingerprint(payload)
    profile = session.query(ExtractorProfile).filter_by(fingerprint=fingerprint).first()
    if profile is None:
        profile = ExtractorProfile(
            fingerprint=fingerprint,
            name=str(payload["backend"]),
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


def get_or_create_embedding_profile(session: Session, config: Config) -> EmbeddingProfile:
    """Resolve immutable embedding profile."""
    payload = build_embedding_profile_payload(config)
    fingerprint = _fingerprint(payload)
    provider = str(payload["provider"])
    model_identifier = str(payload["model_identifier"])
    embedding_dim = (
        config.ollama.embedding_dim if provider == "ollama" else config.llama_cpp.embedding_dim
    )
    distance_metric = str(payload["distance_metric"])
    profile = session.query(EmbeddingProfile).filter_by(fingerprint=fingerprint).first()
    if profile is None:
        profile = EmbeddingProfile(
            fingerprint=fingerprint,
            provider=provider,
            model_identifier=model_identifier,
            embedding_dim=embedding_dim,
            distance_metric=distance_metric,
            config_json=_stable_json(payload),
        )
        session.add(profile)
        session.flush()
    return profile
