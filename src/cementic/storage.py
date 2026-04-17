"""Artifact storage helpers."""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

from cementic.config import Config


def extracted_document_path(
    config: Config,
    collection: str,
    document_id: int,
    extractor_profile_id: int,
) -> Path:
    """Return the gzip path for one extracted document artifact."""
    root = config.storage.artifacts_path
    if root is None:
        raise RuntimeError("Artifact storage path is not configured")
    return root / collection / "extracted" / str(extractor_profile_id) / f"{document_id}.md.gz"


def write_extracted_text(path: Path, content: str) -> str:
    """Write extracted text as gzip and return its content hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(content.encode("utf-8")))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def read_extracted_text(path: Path) -> str:
    """Read extracted text from a gzip artifact."""
    return gzip.decompress(path.read_bytes()).decode("utf-8")
