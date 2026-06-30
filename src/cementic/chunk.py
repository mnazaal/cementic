"""Text chunking using tiktoken."""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken


@dataclass
class TextChunk:
    """Represents a chunk of text."""

    content: str
    chunk_index: int
    page_start: int | None = None
    page_end: int | None = None


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 128,
    model: str = "cl100k_base",
) -> list[TextChunk]:
    """Chunk text into overlapping segments using tiktoken.

    Args:
        text: Input text to chunk
        chunk_size: Number of tokens per chunk
        chunk_overlap: Number of overlapping tokens between chunks
        model: Tiktoken model encoding to use

    Returns:
        List of TextChunk objects
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    encoding = tiktoken.get_encoding(model)
    tokens = encoding.encode(text)

    chunks: list[TextChunk] = []

    # Early return for empty input
    if len(tokens) == 0:
        return chunks
    start = 0
    chunk_index = 0

    while start < len(tokens):
        # Get chunk tokens
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]

        # Decode back to text
        chunk_text = encoding.decode(chunk_tokens)

        # Create chunk
        chunk = TextChunk(
            content=chunk_text,
            chunk_index=chunk_index,
        )
        chunks.append(chunk)

        # Move to next chunk with overlap
        start += chunk_size - chunk_overlap
        chunk_index += 1

    return chunks
