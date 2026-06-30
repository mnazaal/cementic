"""Tests for input validation utilities."""

import pytest

from cementic.validation import (
    MAX_COLLECTION_NAME_LENGTH,
    validate_collection_name,
)


class TestValidateCollectionName:
    """Tests for collection name validation."""

    def test_accepts_simple_alphanumeric(self) -> None:
        assert validate_collection_name("research") == "research"
        assert validate_collection_name("papers2024") == "papers2024"
        assert validate_collection_name("my-collection") == "my-collection"
        assert validate_collection_name("work_papers") == "work_papers"

    def test_accepts_default(self) -> None:
        assert validate_collection_name("default") == "default"

    def test_accepts_single_char(self) -> None:
        assert validate_collection_name("a") == "a"

    def test_accepts_max_length(self) -> None:
        name = "x" * MAX_COLLECTION_NAME_LENGTH
        assert validate_collection_name(name) == name

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_collection_name("")

    def test_rejects_whitespace_only(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_collection_name("   ")

    def test_rejects_too_long(self) -> None:
        long_name = "x" * (MAX_COLLECTION_NAME_LENGTH + 1)
        with pytest.raises(ValueError, match="too long"):
            validate_collection_name(long_name)

    def test_rejects_path_traversal_dot_dot(self) -> None:
        with pytest.raises(ValueError, match="must start with"):
            validate_collection_name("../../../etc")

    def test_rejects_path_traversal_single(self) -> None:
        with pytest.raises(ValueError, match="must start with"):
            validate_collection_name("../data")

    def test_rejects_forward_slash(self) -> None:
        with pytest.raises(ValueError, match="invalid characters"):
            validate_collection_name("foo/bar")

    def test_rejects_backslash(self) -> None:
        with pytest.raises(ValueError, match="invalid characters"):
            validate_collection_name("foo\\bar")

    def test_rejects_null_byte(self) -> None:
        with pytest.raises(ValueError, match="invalid characters"):
            validate_collection_name("bad\x00name")

    def test_rejects_spaces(self) -> None:
        with pytest.raises(ValueError, match="invalid characters"):
            validate_collection_name("my collection")

    def test_rejects_shell_metacharacters(self) -> None:
        for char in (";", "|", "&", "$", "`", "(", ")", "!", "<", ">", "'", '"'):
            with pytest.raises(ValueError, match="invalid characters"):
                validate_collection_name(f"bad{char}name")

    def test_rejects_leading_hyphen(self) -> None:
        with pytest.raises(ValueError, match="must start with"):
            validate_collection_name("-collection")

    def test_rejects_trailing_hyphen(self) -> None:
        with pytest.raises(ValueError, match="must end with"):
            validate_collection_name("collection-")

    def test_rejects_leading_underscore(self) -> None:
        with pytest.raises(ValueError, match="must start with"):
            validate_collection_name("_collection")

    def test_strips_and_validates(self) -> None:
        assert validate_collection_name("  papers  ") == "papers"
