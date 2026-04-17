"""Text chunking using tiktoken."""

from dataclasses import dataclass
from typing import List, Optional

import tiktoken


@dataclass
class TextChunk:
    """Represents a chunk of text."""

    content: str
    chunk_index: int
    page_start: Optional[int] = None
    page_end: Optional[int] = None


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 128,
    model: str = "cl100k_base",
) -> List[TextChunk]:
    """Chunk text into overlapping segments using tiktoken.

    Args:
        text: Input text to chunk
        chunk_size: Number of tokens per chunk
        chunk_overlap: Number of overlapping tokens between chunks
        model: Tiktoken model encoding to use

    Returns:
        List of TextChunk objects
    """
    encoding = tiktoken.get_encoding(model)
    tokens = encoding.encode(text)

    chunks = []
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

        # Avoid infinite loop if chunk_size <= overlap
        if chunk_size <= chunk_overlap:
            break

    return chunks
