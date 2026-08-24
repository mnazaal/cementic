"""Profile fingerprinting and resolution helpers."""

from __future__ import annotations

import json
from hashlib import sha256
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from typing import cast

from sqlalchemy.orm import Session

from cementic.chunk import TOKENIZER
from cementic.config import Config, resolve_llama_model_path
from cementic.db import ChunkProfile, EmbeddingProfile, ExtractorProfile
from cementic.embedding_provider import EmbeddingFacts, EmbeddingProvider
from cementic.embedding_runtime import EmbeddingRuntimeSpec, runtime_spec_from_config
from cementic.embedding_text import describe_text_policy
from cementic.model_digest import model_content_digest

#: Identity of the embedding-input formatting rules in embedding_text.py. Bump
#: this whenever those rules change: it is part of the embedding profile
#: fingerprint, and without a bump old and new vectors would be mixed in one
#: profile with no way to tell them apart.
EMBEDDING_TEXT_FORMAT_VERSION = "v2"
# v2: task prefixes extended from nomic-embed-text-v2 filenames to the whole
# nomic-embed-text family -- v1/v1.5 are trained with the same asymmetric
# prefixes and were silently embedded without them. Vectors from a v1/v1.5
# model under the old rule are unprefixed and must not share a profile with
# prefixed ones.
# v2: chunk_text no longer emits a duplicate tail chunk when a chunk ends exactly
# at the end of the text, so chunk output changed for boundary-length documents.
# v3: covers three later changes to chunk_text output that each shipped without a
# bump, so collections built before them share a fingerprint with today's while
# holding different text: chunk boundaries aligned to whole characters (chunks
# that previously contained U+FFFD now hold correct text), empty slices skipped,
# and chunk_index made contiguous.
CHUNKING_VERSION = "v3"
#: Manual override for changes to cementic's *own* extraction wrapper code
#: (extract.py) that are not a pymupdf/pymupdf4llm version bump -- e.g. a
#: different `header`/`footer`/`page_chunks` call, or a change to how OCR is
#: invoked. It deliberately does NOT track the libraries' own versions; those
#: are recorded separately in "extraction_libraries" below and move the
#: fingerprint on their own when either package is bumped, since a bump
#: changes the Markdown pymupdf4llm produces whether or not this literal was
#: remembered to be typed.
EXTRACTION_VERSION = "v1"

#: The installed packages that actually produce the extracted Markdown.
#: pymupdf-layout is included because the pymupdf4llm backend requires it (it
#: raises if `pymupdf._get_layout` is unavailable) for improved page layout
#: analysis, not merely pulled in incidentally. It stays listed even though the
#: pymupdf-raw backend never loads it: which backend produced a revision's text
#: is recorded by "backends" in the payload, so a raw-backend revision is
#: already distinct from a pymupdf4llm one, and pinning the version here costs
#: only a rebuild that a layout-model upgrade should cause anyway.
_EXTRACTION_LIBRARY_NAMES = ("pymupdf4llm", "pymupdf", "pymupdf-layout")


def _stable_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _fingerprint(payload: dict[str, object]) -> str:
    return sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _installed_extraction_library_versions() -> dict[str, str | None]:
    """Installed version of each extraction library, or None if not installed.

    None rather than a raised exception: a bare `importlib.metadata.version()`
    call raises `PackageNotFoundError` for an uninstalled optional package
    (pymupdf-layout is required at runtime by extract.py, but profile
    construction itself must never crash on what happens to be installed).
    """
    versions: dict[str, str | None] = {}
    for name in _EXTRACTION_LIBRARY_NAMES:
        try:
            versions[name] = _package_version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def build_extractor_profile_payload(config: Config) -> dict[str, object]:
    """Build extractor profile payload from config.

    Records the extractor registry identity so adding or revving a content-type
    extractor re-versions the revision. Imported locally to keep this module's
    import graph free of the (heavier) extraction backend.

    The Markdown is actually produced by pymupdf/pymupdf4llm, not by this
    module's own code -- "extraction_libraries" records their installed
    versions so a dependency bump moves the fingerprint on its own instead of
    silently changing extraction output under an unchanged fingerprint.
    """
    from cementic.extract import extractor_registry_payload

    return {
        "backends": dict(sorted(config.extraction.backends.items())),
        "use_ocr": config.extraction.use_ocr,
        "version": EXTRACTION_VERSION,
        "extractors": extractor_registry_payload(),
        "extraction_libraries": _installed_extraction_library_versions(),
    }


def build_chunk_profile_payload(config: Config) -> dict[str, object]:
    """Build chunk profile payload from config."""
    return {
        "chunk_size": config.pipeline.chunk_size,
        "chunk_overlap": config.pipeline.chunk_overlap,
        "tokenizer": TOKENIZER,
        "version": CHUNKING_VERSION,
    }


def _model_identity(spec: EmbeddingRuntimeSpec) -> tuple[str, str]:
    """(display label, content-identity digest) for one runtime spec's model.

    Only llama-cpp resolves to a local file that can be hashed; a
    hypothetical future non-file provider falls back to its raw identifier as
    both, since there is nothing on disk to hash by.

    The digest -- not the path string -- is what makes two profiles the same
    or different: ``resolve_llama_model_path`` honours a relative path that
    exists from the current directory, so two directories each holding a
    different GGUF at the same relative path used to fingerprint as one
    model. The label stays a plain basename (not the full resolved path) so
    that the *same* file reached via two different absolute paths still
    fingerprints identically -- only content and display name matter, not
    where it happens to sit on disk.
    """
    if spec.provider != "llama-cpp":
        return spec.model_identifier, spec.model_identifier

    resolved = resolve_llama_model_path(spec.model_identifier)
    digest = model_content_digest(spec.model_identifier)
    if digest is None:
        # Not downloaded yet, or a test/fixture path that never exists.
        # Falling back to the resolved path keeps profile construction from
        # crashing; production never reaches this branch because the daemon
        # serving this model must already be running by the time a profile
        # is resolved for real indexing or search.
        digest = f"unreadable:{resolved}"
    return resolved.name, digest


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
    model_label, model_digest = _model_identity(spec)
    return {
        "provider": facts.name,
        "model_identifier": model_label,
        "model_digest": model_digest,
        "embedding_dim": facts.embedding_dim,
        "distance_metric": facts.distance_metric,
        "n_ctx": spec.n_ctx,
        "n_gpu_layers": spec.n_gpu_layers,
        # `verbose` is deliberately absent: it is a launch argument (see
        # embedding_runtime.llama_cpp_runtime_fingerprint, where it correctly
        # stays), not a fact about the vectors this profile identifies.
        # Toggling it must not fork the corpus into two vector spaces.
        "text_format_version": EMBEDDING_TEXT_FORMAT_VERSION,
        # The task-prefix policy is chosen from the model *filename*, so
        # renaming a GGUF (or mirroring it under another name) silently switches
        # to plain text. Left out of the payload, prefixed and unprefixed
        # corpora shared one profile and one vector table -- two incompatible
        # vector spaces mixed in a single index, which is exactly what
        # text_format_version beside it exists to prevent. Recording it makes a
        # rename fork the profile and rebuild instead.
        #
        # Deliberately keyed off spec.model_identifier (the configured path),
        # not the resolved/digest identity above: the prefix convention is a
        # filename heuristic, unrelated to which bytes are on disk.
        "text_policy": describe_text_policy(spec.model_identifier),
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


def build_embedding_runtime_payload(
    config: Config, provider: EmbeddingProvider | None = None
) -> dict[str, object]:
    """The profile payload plus enough to relaunch the model it names.

    Split from ``build_embedding_profile_payload`` because the two answer
    different questions and had been sharing one dict. The *fingerprint* must be
    path-independent -- that is the whole point of identifying the model by
    content digest, so the same file under two paths is one profile. The stored
    ``config_json`` must be path-*dependent*, because ``search`` rebuilds a
    daemon launch spec from it and needs a path that resolves.

    Collapsing them meant the basename went into the launch spec, where
    ``resolve_llama_model_path`` resolved it against the data dir root and
    produced a file that does not exist -- so every search after promoting a
    new-scheme revision died with "Model path does not exist". Found by running
    the CLI against the live corpus; no unit test covered profile JSON reaching
    a real daemon launch.
    """
    payload = build_embedding_profile_payload(config, provider)
    spec = runtime_spec_from_config(config)
    if spec.provider != "llama-cpp":
        return payload
    # Launch needs the resolved file, not the identity. The digest stays in the
    # payload as the identity; this key is what makes the profile replayable.
    return {**payload, "model_identifier": str(resolve_llama_model_path(spec.model_identifier))}


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
            config_json=_stable_json(build_embedding_runtime_payload(config, provider)),
        )
        session.add(profile)
        session.flush()
    return profile
