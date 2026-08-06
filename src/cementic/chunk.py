"""Text chunking using tiktoken."""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken

#: Tokenizer used for chunking. Also recorded in chunk profiles (see profiles.py),
#: so changing it re-versions every chunk profile.
TOKENIZER = "cl100k_base"


@dataclass
class TextChunk:
    """Represents a chunk of text."""

    content: str
    chunk_index: int


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 128,
) -> list[TextChunk]:
    """Chunk text into overlapping segments using tiktoken.

    Args:
        text: Input text to chunk
        chunk_size: Number of tokens per chunk
        chunk_overlap: Number of overlapping tokens between chunks

    Returns:
        List of TextChunk objects
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    encoding = tiktoken.get_encoding(TOKENIZER)
    tokens = encoding.encode(text)

    chunks: list[TextChunk] = []
    start = 0
    chunk_index = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(
            TextChunk(content=encoding.decode(tokens[start:end]), chunk_index=chunk_index)
        )
        # Stop once a chunk reaches the end of the text: stepping again would
        # emit a chunk that is purely a suffix of this one (duplicate content).
        if end == len(tokens):
            break
        start += chunk_size - chunk_overlap
        chunk_index += 1

    return chunks
