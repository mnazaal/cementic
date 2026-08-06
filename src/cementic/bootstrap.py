"""Runtime bootstrap helpers: verify external services and fetch the model.

cementic does not manage containers. Postgres (with pgvector + vectorscale) is
provisioned externally -- e.g. via ``cementic init postgres ./cementic-postgres``
and its generated setup, or any Postgres pointed at by ``CEMENTIC_DB_URL``. Here
we only verify Postgres is reachable and fetch the llama.cpp model file when
needed.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import requests
from sqlalchemy import text

from cementic.config import (
    Config,
    LlamaCppConfig,
    cementic_data_dir,
    resolve_llama_model_path,
)
from cementic.db import get_engine

_logger = logging.getLogger("cementic.bootstrap")

_COMPOSE_HINT = (
    "Run `cementic init postgres ./cementic-postgres` once and follow its README, "
    "or set CEMENTIC_DB_URL to an existing Postgres with the pgvector and "
    "vectorscale extensions."
)


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_sha256(path: Path, expected: str | None) -> None:
    """Validate a file checksum when an expected digest is configured.

    An empty or unset ``expected`` disables verification (the opt-out for a
    custom ``llama_model_url``).
    """
    if not expected:
        return
    actual = _sha256_file(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(
            f"Checksum mismatch for {path}: expected {expected}, got {actual}. "
            "Set CEMENTIC_BOOTSTRAP_LLAMA_MODEL_SHA256 to the digest of this file, "
            "or to an empty value to disable verification."
        )


class Bootstrapper:
    """Verifies external runtime dependencies and fetches the embedding model."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def ensure_for_convert(self) -> None:
        """Ensure runtime dependencies for conversion."""
        self._ensure_postgres_ready()

    def ensure_for_index(self) -> None:
        """Ensure runtime dependencies for indexing."""
        self._ensure_postgres_ready()
        self.ensure_embedding_runtime()

    def ensure_embedding_runtime(self) -> None:
        """Ensure the embedding runtime's model is present (no database needed)."""
        provider = self.config.pipeline.embedding_provider
        if provider == "llama-cpp":
            self._ensure_llama_model()
            return
        raise RuntimeError(f"Unsupported embedding provider: {provider}")

    def _ensure_postgres_ready(self) -> None:
        if self._database_ready():
            return
        raise RuntimeError(
            f"Cannot reach Postgres at {self.config.database.url}. {_COMPOSE_HINT}"
        )

    def _model_path_is_default(self) -> bool:
        """Whether model_path still points at the file cementic would download."""
        default = LlamaCppConfig.model_fields["model_path"].default
        return bool(self.config.llama_cpp.model_path == default)

    def _ensure_llama_model(self) -> None:
        # Resolve to the same physical path the runtime will load from, so a
        # successful download is exactly what the daemon/provider opens later.
        model_path = resolve_llama_model_path(self.config.llama_cpp.model_path)
        expected_sha256 = self.config.bootstrap.llama_model_sha256
        if model_path.exists():
            # The pin is a supply-chain control on what cementic *downloads*, so
            # it only applies to a file at the default model path. Enforcing the
            # bundled Nomic digest against a model the user supplied themselves
            # made `CEMENTIC_LLAMA_MODEL_PATH=/my/model.gguf` -- the documented
            # way to use a different model -- fail with a mismatch.
            if self._model_path_is_default():
                _ensure_sha256(model_path, expected_sha256)
            return

        if not self.config.bootstrap.auto_download_llama_model:
            raise RuntimeError(
                f"llama.cpp model not found at {model_path}. "
                "Set CEMENTIC_LLAMA_MODEL_PATH to an existing file or enable "
                "CEMENTIC_BOOTSTRAP_AUTO_DOWNLOAD_LLAMA_MODEL=true."
            )

        # Restrict auto-downloads to the cementic data directory.
        allowed_root = cementic_data_dir(ensure_exists=True).resolve()
        try:
            model_path.resolve().relative_to(allowed_root)
        except ValueError:
            raise RuntimeError(
                f"llama.cpp model path {model_path} is outside the "
                f"cementic data directory {allowed_root}"
            )

        if not expected_sha256:
            _logger.warning(
                "Downloading llama.cpp model from %s without integrity verification; "
                "set CEMENTIC_BOOTSTRAP_LLAMA_MODEL_SHA256 to pin the expected SHA-256.",
                self.config.bootstrap.llama_model_url,
            )

        model_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = model_path.with_name(f".{model_path.name}.tmp")
        temp_path.unlink(missing_ok=True)
        max_download_bytes = 5 * 1024 * 1024 * 1024  # 5 GiB
        downloaded = 0
        with requests.get(
            self.config.bootstrap.llama_model_url, stream=True, timeout=60
        ) as response:
            response.raise_for_status()
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if downloaded > max_download_bytes:
                            f.close()
                            temp_path.unlink(missing_ok=True)
                            raise RuntimeError(
                                f"Model download exceeded maximum size "
                                f"({max_download_bytes} bytes)"
                            )
        try:
            _ensure_sha256(temp_path, expected_sha256)
            os.replace(temp_path, model_path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    def _database_ready(self) -> bool:
        try:
            engine = get_engine(self.config.database.url)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
