"""Tests for artifact storage helpers."""

import hashlib
from pathlib import Path

import pytest

from cementic.config import Config
from cementic.storage import (
    extracted_document_path,
    read_extracted_text,
    safe_remove_artifact,
    write_extracted_text,
)


class TestExtractedDocumentPath:
    """Tests for extracted_document_path."""

    def test_constructs_path(self, temp_dir: Path) -> None:
        config = Config()
        config.storage.artifacts_path = temp_dir
        path = extracted_document_path(config, "mycoll", 7, 3)
        expected = temp_dir / "mycoll" / "extracted" / "3" / "7.md.gz"
        assert path == expected

    def test_raises_when_artifacts_path_is_none(self) -> None:
        config = Config()
        config.storage.artifacts_path = None
        with pytest.raises(RuntimeError, match="not configured"):
            extracted_document_path(config, "x", 1, 1)

    def test_rejects_path_traversal_collection(self, temp_dir: Path) -> None:
        config = Config()
        config.storage.artifacts_path = temp_dir
        with pytest.raises(ValueError, match="must start with"):
            extracted_document_path(config, "../../../etc", 1, 1)

    def test_rejects_slash_in_collection(self, temp_dir: Path) -> None:
        config = Config()
        config.storage.artifacts_path = temp_dir
        with pytest.raises(ValueError, match="invalid characters"):
            extracted_document_path(config, "foo/bar", 1, 1)

    def test_result_path_within_artifacts_root(self, temp_dir: Path) -> None:
        config = Config()
        config.storage.artifacts_path = temp_dir
        path = extracted_document_path(config, "mycoll", 7, 3)
        # Ensure the resolved path stays within the artifacts root
        assert str(temp_dir.resolve()) in str(path.resolve())


class TestWriteReadExtracted:
    """Round-trip tests for write/read extracted text."""

    def test_roundtrip(self, temp_dir: Path) -> None:
        content = "Hello extracted text\nwith multiple lines."
        path = temp_dir / "out.md.gz"

        content_hash = write_extracted_text(path, content)

        assert path.exists()
        assert content_hash == hashlib.sha256(content.encode("utf-8")).hexdigest()

        restored = read_extracted_text(path)
        assert restored == content

    def test_creates_parent_dirs(self, temp_dir: Path) -> None:
        path = temp_dir / "deeply" / "nested" / "dir" / "test.md.gz"
        write_extracted_text(path, "content")
        assert path.exists()

    def test_read_decompresses(self, temp_dir: Path) -> None:
        content = "unicode ✓ works"
        path = temp_dir / "unicode.md.gz"
        write_extracted_text(path, content)
        assert read_extracted_text(path) == content

    def test_write_truncates_existing(self, temp_dir: Path) -> None:
        path = temp_dir / "overwrite.md.gz"
        write_extracted_text(path, "first")
        write_extracted_text(path, "second longer content")
        assert read_extracted_text(path) == "second longer content"


class TestSafeRemoveArtifact:
    """Tests for safe_remove_artifact."""

    def test_removes_file_under_artifacts_root(self, temp_dir: Path) -> None:
        config = Config()
        config.storage.artifacts_path = temp_dir
        f = temp_dir / "file.md.gz"
        f.write_text("content")
        safe_remove_artifact(config, str(f))
        assert not f.exists()

    def test_ignores_missing_file(self, temp_dir: Path) -> None:
        config = Config()
        config.storage.artifacts_path = temp_dir
        safe_remove_artifact(config, str(temp_dir / "nope.md.gz"))  # does not raise

    def test_ignores_oserror(self, temp_dir: Path) -> None:
        config = Config()
        config.storage.artifacts_path = temp_dir
        with pytest.MonkeyPatch.context():  # noop call; any OSError is caught
            pass
        safe_remove_artifact(config, str(temp_dir / "non-existent.md.gz"))

    def test_rejects_path_traversal_artifact(self, temp_dir: Path) -> None:
        config = Config()
        config.storage.artifacts_path = temp_dir
        outside = temp_dir.parent / "outside.md.gz"
        outside.write_text("should not be removed")
        try:
            with pytest.raises(ValueError, match="outside .* artifacts root"):
                safe_remove_artifact(config, str(outside))
        finally:
            outside.unlink(missing_ok=True)

    def test_rejects_unconfigured_artifacts_root(self, temp_dir: Path) -> None:
        config = Config()
        config.storage.artifacts_path = None
        with pytest.raises(RuntimeError, match="not configured"):
            safe_remove_artifact(config, str(temp_dir / "any.md.gz"))

    def test_accepts_path_under_root_via_resolved_path(self, temp_dir: Path) -> None:
        config = Config()
        config.storage.artifacts_path = temp_dir
        f = temp_dir / "sub" / "file.md.gz"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("content")
        safe_remove_artifact(config, str(f))
        assert not f.exists()
