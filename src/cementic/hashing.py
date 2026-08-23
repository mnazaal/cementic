"""Shared file-hashing helper.

One hasher, reused wherever cementic needs a file's content identity:
``bootstrap.py`` verifies a downloaded model against its pinned checksum, and
``model_digest.py`` fingerprints a configured model by content rather than by
the path string that happens to resolve to it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, streamed in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
