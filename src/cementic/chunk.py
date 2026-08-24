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
    # Match the shipped config defaults: 512/128 was exactly the pairing the
    # config module documents as unsafe against the default model's window.
    chunk_size: int = 320,
    chunk_overlap: int = 80,
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
    # `disallowed_special=()` because tiktoken otherwise refuses any text
    # containing a literal control-token spelling -- `<|endoftext|>` and its
    # siblings -- and raises instead of encoding. That string is ordinary prose
    # in a corpus of NLP papers, and the raise is terminal for the document:
    # the chunk step fails it, `requeue_interrupted_artifacts` re-queues it on
    # the next start, and it fails again forever. Here the text is being
    # measured for splitting, never fed to a model as control tokens, so
    # treating the spelling as the bytes it literally is the correct reading.
    tokens = encoding.encode(text, disallowed_special=())

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
        content = data[first_byte:last_byte].decode("utf-8", errors="replace")
        # Widening both ends can empty a slice whose every token was part of one
        # character -- reachable at small chunk_size, which the config and the
        # `chunk` command both accept. Storing and embedding "" would spend a
        # result slot on a blank preview.
        if content:
            chunks.append(TextChunk(content=content, chunk_index=chunk_index))
            # Advance the index only when a chunk was actually emitted. Bumping
            # it every iteration left gaps whenever a slice came back empty --
            # `chunk_text("😀" * 20, chunk_size=1, chunk_overlap=0)` produced 20
            # chunks numbered 0, 2, 4 ... 38 -- so chunk_index no longer agreed
            # with position, and total_chunks agreed with neither.
            chunk_index += 1
        # Stop once a chunk reaches the end of the text: stepping again would
        # emit a chunk that is purely a suffix of this one (duplicate content).
        if end == len(tokens):
            break
        start += chunk_size - chunk_overlap

    return chunks
