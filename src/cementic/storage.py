"""Artifact storage helpers."""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

from cementic.config import Config
from cementic.validation import validate_collection_name


def extracted_document_path(
    config: Config,
    collection: str,
    document_id: int,
    extractor_profile_id: int,
) -> Path:
    """Return the gzip path for one extracted document artifact."""
    _ = validate_collection_name(collection)
    root = config.storage.artifacts_path
    if root is None:
        raise RuntimeError("Artifact storage path is not configured")
    return root / collection / "extracted" / str(extractor_profile_id) / f"{document_id}.md.gz"


def write_extracted_text(path: Path, content: str) -> str:
    """Write extracted text as gzip and return its content hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.is_symlink():
        raise ValueError(f"Refusing to write through symlink: {path}")
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_bytes(gzip.compress(content.encode("utf-8")))
    tmp_path.replace(path)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def read_extracted_text(path: Path) -> str:
    """Read extracted text from a gzip artifact."""
    return gzip.decompress(path.read_bytes()).decode("utf-8")


def safe_remove_artifact(config: Config, artifact_path: str) -> None:
    """Remove an artifact file, rejecting paths outside the artifacts root.

    This is a best-effort helper that swallows OSError for idempotent cleanup.
    """
    root = config.storage.artifacts_path
    if root is None:
        raise RuntimeError("Artifact storage path is not configured")
    resolved = Path(artifact_path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        raise ValueError(
            f"Artifact path {resolved} is outside the artifacts root {root}"
        ) from None
    path = Path(artifact_path)
    if path.is_symlink():
        raise ValueError(f"Refusing to remove symlink artifact: {path}")
    try:
        resolved.unlink(missing_ok=True)
    except OSError:
        pass
