"""Tests for content-identity hashing/caching of embedding model files."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from cementic import model_digest


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the digest cache at a throwaway directory for every test."""
    cache_dir = tmp_path / "cementic-data"
    monkeypatch.setattr(
        model_digest, "cementic_data_dir", lambda *, ensure_exists=False: cache_dir
    )
    return cache_dir


class TestModelContentDigest:
    """The core defect: identity must follow content, not the path string."""

    def test_two_different_files_have_different_digests(self, tmp_path: Path) -> None:
        """Two GGUFs at the *same relative path* from two working directories.

        Simulated here by two absolute files that stand in for "models/x.gguf"
        resolved from two different cwds -- the scenario `resolve_llama_model_path`
        cannot tell apart by path string alone.
        """
        dir_a = tmp_path / "run-from-a" / "models"
        dir_b = tmp_path / "run-from-b" / "models"
        dir_a.mkdir(parents=True)
        dir_b.mkdir(parents=True)
        file_a = dir_a / "embed.gguf"
        file_b = dir_b / "embed.gguf"
        file_a.write_bytes(b"model-one-bytes")
        file_b.write_bytes(b"model-two-bytes-different")

        digest_a = model_digest.model_content_digest(str(file_a))
        digest_b = model_digest.model_content_digest(str(file_b))

        assert digest_a is not None
        assert digest_b is not None
        assert digest_a != digest_b

    def test_same_file_at_two_absolute_paths_has_the_same_digest(self, tmp_path: Path) -> None:
        dir_1 = tmp_path / "one"
        dir_2 = tmp_path / "two"
        dir_1.mkdir()
        dir_2.mkdir()
        file_1 = dir_1 / "embed.gguf"
        file_2 = dir_2 / "embed.gguf"
        file_1.write_bytes(b"identical-model-bytes")
        file_2.write_bytes(b"identical-model-bytes")

        assert model_digest.model_content_digest(str(file_1)) == model_digest.model_content_digest(
            str(file_2)
        )

    def test_missing_file_returns_none_rather_than_raising(self, tmp_path: Path) -> None:
        missing = tmp_path / "does-not-exist.gguf"
        assert model_digest.model_content_digest(str(missing)) is None


class TestDigestCache:
    """A cache hit must avoid re-hashing; a stat change must force it."""

    def test_cache_hit_avoids_rehashing(self, tmp_path: Path) -> None:
        model_file = tmp_path / "embed.gguf"
        model_file.write_bytes(b"some model bytes")

        with patch(
            "cementic.model_digest.sha256_file", wraps=model_digest.sha256_file
        ) as hasher:
            first = model_digest.model_content_digest(str(model_file))
            second = model_digest.model_content_digest(str(model_file))

        assert first == second
        assert hasher.call_count == 1

    def test_changed_size_forces_rehash(self, tmp_path: Path) -> None:
        model_file = tmp_path / "embed.gguf"
        model_file.write_bytes(b"short")
        model_digest.model_content_digest(str(model_file))

        model_file.write_bytes(b"a much longer replacement payload")

        with patch(
            "cementic.model_digest.sha256_file", wraps=model_digest.sha256_file
        ) as hasher:
            digest = model_digest.model_content_digest(str(model_file))

        assert hasher.call_count == 1
        assert digest == model_digest.sha256_file(model_file)

    def test_changed_mtime_forces_rehash_even_with_identical_content(
        self, tmp_path: Path
    ) -> None:
        model_file = tmp_path / "embed.gguf"
        model_file.write_bytes(b"same bytes")
        model_digest.model_content_digest(str(model_file))

        # Same size, same bytes, only the mtime moves -- the cache key must
        # still miss, because mtime is one of the three fields that invalidate it.
        stat = model_file.stat()
        os.utime(model_file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

        with patch(
            "cementic.model_digest.sha256_file", wraps=model_digest.sha256_file
        ) as hasher:
            model_digest.model_content_digest(str(model_file))

        assert hasher.call_count == 1

    def test_corrupt_cache_file_is_ignored_not_fatal(self, tmp_path: Path) -> None:
        model_file = tmp_path / "embed.gguf"
        model_file.write_bytes(b"payload")

        cache_path = model_digest._cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("{not valid json", encoding="utf-8")

        digest = model_digest.model_content_digest(str(model_file))

        assert digest == model_digest.sha256_file(model_file)
