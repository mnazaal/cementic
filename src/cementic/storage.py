"""Artifact storage helpers."""

from __future__ import annotations

import gzip
import hashlib
import os
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
    # Use the normalized name, not the raw argument: "  foo  " validates but
    # would otherwise create a directory whose name does not match the
    # collection recorded in the database.
    collection = validate_collection_name(collection)
    root = config.storage.artifacts_path
    if root is None:
        raise RuntimeError("Artifact storage path is not configured")
    return root / collection / "extracted" / str(extractor_profile_id) / f"{document_id}.md.gz"


def write_extracted_text(path: Path, content: str) -> str:
    """Write extracted text as gzip and return its content hash.

    Written to a per-process temp name and renamed, so a reader never sees a
    partial artifact. The temp file is removed if the write fails: a worker
    killed mid-write previously left a ``.<name>.tmp`` behind forever, since
    nothing sweeps them and `collection remove` only deletes paths recorded in
    the database. The pid suffix keeps two workers from writing the same temp.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"Refusing to write through symlink: {path}")
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_bytes(gzip.compress(content.encode("utf-8")))
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def read_extracted_text(path: Path) -> str:
    """Read extracted text from a gzip artifact."""
    return gzip.decompress(path.read_bytes()).decode("utf-8")


def safe_remove_artifact(config: Config, artifact_path: str) -> None:
    """Remove an artifact file, rejecting paths outside the artifacts root.

    Idempotent: a file that is already gone is success. A file that is *there*
    and cannot be removed raises, so the caller can report it -- swallowing
    OSError here meant `collection remove` printed "status: deleted" with every
    artifact still on disk and the database rows that named them gone, leaving
    nothing that could ever find them again.
    """
    root = config.storage.artifacts_path
    if root is None:
        raise RuntimeError("Artifact storage path is not configured")
    path = Path(artifact_path)
    # Symlink check first, and unlink the *given* path rather than the resolved
    # one: resolve() follows links, so the old order checked containment on the
    # target and then unlinked the target -- one swapped link away from deleting
    # a file the containment check never saw.
    if path.is_symlink():
        raise ValueError(f"Refusing to remove symlink artifact: {path}")
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        raise ValueError(
            f"Artifact path {resolved} is outside the artifacts root {root}"
        ) from None
    path.unlink(missing_ok=True)
