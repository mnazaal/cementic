"""Content-identity for embedding model files.

The embedding profile fingerprint must identify a model by its bytes, not by
the path string that happened to resolve to it: ``resolve_llama_model_path``
honours a relative path that exists from the current working directory, so
running cementic from two directories that each hold a different GGUF at the
same relative path used to fingerprint both as "the same model" -- exactly
what ``search.py``'s mixed-model refusal exists to prevent, and it was walked
past because the fingerprint claimed the models were identical (see
``profiles.build_embedding_profile_payload``).

Hashing a multi-hundred-MB GGUF on every profile resolution is too expensive
to do unconditionally, so the digest is cached under the cementic data
directory, keyed on ``(resolved absolute path, size, mtime_ns)``: any of the
three changing forces a re-hash. A missing or corrupt cache re-hashes rather
than failing -- it is a pure performance optimisation, never a source of
truth.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from cementic.config import cementic_data_dir, resolve_llama_model_path
from cementic.filelock import file_lock
from cementic.hashing import sha256_file

_logger = logging.getLogger("cementic.model_digest")

_CACHE_FILENAME = "model_digest_cache.json"


def _cache_path() -> Path:
    """Where the digest cache lives. Does not create the data directory."""
    return cementic_data_dir(ensure_exists=False) / _CACHE_FILENAME


def _cache_key(resolved_path: Path, size: int, mtime_ns: int) -> str:
    return f"{resolved_path}:{size}:{mtime_ns}"


def _load_cache(path: Path) -> dict[str, str]:
    """The cached digest map, or empty when the file is missing/corrupt."""
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items() if isinstance(value, str)}


def _save_cache(path: Path, cache: dict[str, str]) -> None:
    try:
        path.write_text(json.dumps(cache), encoding="utf-8")
    except OSError:
        # Best effort: a failed write only costs a re-hash next time, not a
        # wrong answer -- the cache is never the source of truth.
        _logger.debug("Could not write model digest cache to %s", path, exc_info=True)


def _hash_model(resolved: Path) -> str | None:
    """Hash the model itself; None when its bytes cannot be read."""
    try:
        return sha256_file(resolved)
    except OSError:
        _logger.debug("Could not read model %s", resolved, exc_info=True)
        return None


def model_content_digest(model_path: str) -> str | None:
    """SHA-256 hex digest of the resolved model file, or None if unreadable.

    ``model_path`` is resolved the same way the runtime loads it
    (``resolve_llama_model_path``), so the digest always describes the file
    that will actually be served.

    Returns None -- never raises -- only when the *model* cannot be stat'd or
    read (missing, not downloaded yet, permission denied). A failure of the
    digest cache (lock cannot be taken, file unwritable, e.g. a read-only data
    directory) never changes the answer: the file is hashed without the cache,
    which is slower but identical, because the cache is never a source of
    truth.

    A None result must not be turned into a persisted identity. The daemon
    alias (``embedding_runtime.llama_cpp_runtime_fingerprint``) substitutes a
    placeholder so ``doctor``/``status`` keep working; profile construction
    (``profiles._model_identity``) refuses instead.
    """
    resolved = resolve_llama_model_path(model_path)
    try:
        stat = resolved.stat()
    except OSError:
        return None

    key = _cache_key(resolved, stat.st_size, stat.st_mtime_ns)
    cache_file = _cache_path()
    lock_path = cache_file.with_name(f"{cache_file.name}.lock")
    try:
        with file_lock(lock_path):
            cache = _load_cache(cache_file)
            cached = cache.get(key)
            if cached is not None:
                return cached
            digest = _hash_model(resolved)
            if digest is not None:
                cache[key] = digest
                _save_cache(cache_file, cache)
            return digest
    except OSError:
        # The cache lock (or its directory) is unavailable: not the model's
        # fault, so the digest must not change -- hash without the cache.
        _logger.debug("Digest cache unavailable for %s", resolved, exc_info=True)
        return _hash_model(resolved)
