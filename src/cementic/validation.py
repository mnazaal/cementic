"""Input validation utilities for cementic."""

import re

MAX_COLLECTION_NAME_LENGTH = 100

# Collection names: alphanumeric, hyphens, underscores only.
# Must start and end with alphanumeric.
_COLLECTION_NAME_RE = re.compile(r"^[a-zA-Z0-9]([a-zA-Z0-9_-]*[a-zA-Z0-9])?$")


def validate_collection_name(name: str) -> str:
    """Validate and normalize a collection name.

    Args:
        name: Raw collection name from user input.

    Returns:
        Stripped, validated collection name.

    Raises:
        ValueError: If the name is empty, too long, or contains invalid characters.
    """
    stripped = name.strip()
    if not stripped:
        raise ValueError("Collection name cannot be empty")

    if len(stripped) > MAX_COLLECTION_NAME_LENGTH:
        raise ValueError(
            f"Collection name too long: {len(stripped)} characters "
            f"(max {MAX_COLLECTION_NAME_LENGTH})"
        )

    if not _COLLECTION_NAME_RE.match(stripped):
        if not stripped[0].isalnum():
            raise ValueError(
                f"Collection name must start with a letter or digit: {stripped!r}"
            )
        if not stripped[-1].isalnum():
            raise ValueError(
                f"Collection name must end with a letter or digit: {stripped!r}"
            )
        raise ValueError(
            f"Collection name contains invalid characters: {stripped!r}. "
            "Use only letters, digits, hyphens, and underscores."
        )

    return stripped
