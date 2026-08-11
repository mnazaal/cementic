"""Text chunking using tiktoken."""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken

#: Tokenizer used for chunking. Also recorded in chunk profiles (see profiles.py),
#: so changing it re-versions every chunk profile.
TOKENIZER = "cl100k_base"


def _character_boundary(data: bytes, index: int) -> int:
    """Advance ``index`` to the next UTF-8 character boundary in ``data`` (pure).

    Continuation bytes (``0b10xxxxxx``) are the tail of a character that starts
    earlier, so an index pointing at one is mid-character.
    """
    while index < len(data) and data[index] & 0xC0 == 0x80:
        index += 1
    return index


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

    # Byte offset of every token boundary. The tokenizer is byte-level, so a
    # token slice can cut through a multi-byte character; decoding that slice
    # directly renders each fragment as U+FFFD, and the corruption is what gets
    # embedded and shown in search results. Working in bytes lets each chunk
    # end on a whole character by keeping the few bytes that complete it.
    data = encoding.decode_bytes(tokens)
    offsets = [0]
    for token in tokens:
        offsets.append(offsets[-1] + len(encoding.decode_single_token_bytes(token)))

    chunks: list[TextChunk] = []
    start = 0
    chunk_index = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        # Widening both ends to the next character boundary keeps concatenation
        # of non-overlapping chunks lossless: the bytes one chunk borrows to
        # finish a character are exactly the ones the next chunk skips.
        first_byte = _character_boundary(data, offsets[start])
        last_byte = _character_boundary(data, offsets[end])
        chunks.append(
            TextChunk(
                content=data[first_byte:last_byte].decode("utf-8", errors="replace"),
                chunk_index=chunk_index,
            )
        )
        # Stop once a chunk reaches the end of the text: stepping again would
        # emit a chunk that is purely a suffix of this one (duplicate content).
        if end == len(tokens):
            break
        start += chunk_size - chunk_overlap
        chunk_index += 1

    return chunks
