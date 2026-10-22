"""Text chunking using tiktoken."""

from __future__ import annotations

import functools
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

import tiktoken

from cementic.config import cementic_data_dir

#: Tokenizer used for chunking. Also recorded in chunk profiles (see profiles.py),
#: so changing it re-versions every chunk profile.
TOKENIZER = "cl100k_base"

#: Where tiktoken fetches ``TOKENIZER``'s BPE ranks on first use. Mirrors
#: ``tiktoken_ext.openai_public.cl100k_base``, and a test pins the two together,
#: because `doctor` derives the cache file's name from it.
TOKENIZER_URL = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"


class TokenizerUnavailableError(RuntimeError):
    """The tokenizer file is not cached and could not be downloaded.

    Not a property of any document, so the worker must not stamp a chunk failed
    for it: every document in the corpus would fail the same way.
    """


def tokenizer_cache_dir() -> Path:
    """Where the tokenizer file is cached (pure apart from reading the environment).

    tiktoken's own overrides win when set. Otherwise cementic's data dir, not
    tiktoken's default of ``$TMPDIR/data-gym-cache``: that is wiped at reboot,
    so every boot silently needed the network again, and an offline machine
    failed every chunk and every search (2026-10-22 audit).
    """
    for variable in ("TIKTOKEN_CACHE_DIR", "DATA_GYM_CACHE_DIR"):
        if variable in os.environ:
            return Path(os.environ[variable])
    return cementic_data_dir() / "tiktoken"


def tokenizer_cache_file() -> Path:
    """The file tiktoken caches ``TOKENIZER`` in: its cache key is the URL's SHA-1."""
    return tokenizer_cache_dir() / hashlib.sha1(TOKENIZER_URL.encode()).hexdigest()


@functools.cache
def tokenizer() -> tiktoken.Encoding:
    """The ``TOKENIZER`` encoding, cached under cementic's data dir.

    The one place that loads it, so the cache location is set before tiktoken
    first looks. Setting the variable is the only side effect, and only when
    the user has not chosen a location themselves.
    """
    if not any(v in os.environ for v in ("TIKTOKEN_CACHE_DIR", "DATA_GYM_CACHE_DIR")):
        os.environ["TIKTOKEN_CACHE_DIR"] = str(tokenizer_cache_dir())
    try:
        return tiktoken.get_encoding(TOKENIZER)
    except Exception as error:
        # tiktoken fetches with `requests` and verifies a hash; any failure
        # here means the ranks are neither cached nor fetchable.
        raise TokenizerUnavailableError(
            f"could not load the {TOKENIZER} tokenizer: it is downloaded once on "
            f"first use and cached in {tokenizer_cache_dir()}, and the download "
            f"failed ({type(error).__name__}: {error})"
        ) from error

#: Longest run of non-whitespace characters `chunk_text` will hand the
#: tokenizer. tiktoken's pre-tokenizer splits on whitespace and then runs BPE
#: over each piece, and that merge loop is superlinear in the piece's length: a
#: single unbroken 160k-character run measured 12.2 s, inside a Rust call that
#: does not observe the shutdown event, so `cementic stop` cannot interrupt it.
#: Four times that run length is over three minutes of unkillable CPU.
#:
#: The cap is deliberately far above real prose. Nothing in a document with
#: ordinary spacing comes near it; what does is a PDF whose text layer lost its
#: spacing, or a minified line in a source file -- text that tokenizes into
#: garbage and retrieves nothing even when it succeeds.
MAX_UNBROKEN_RUN_CHARS = 100_000


def longest_unbroken_run(text: str) -> int:
    """Length of the longest run of non-whitespace characters (pure)."""
    return max((match.end() - match.start() for match in re.finditer(r"\S+", text)), default=0)


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
    chunk_size: int,
    chunk_overlap: int,
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

    # Checked before encoding, because the cost being avoided is inside
    # `encode`. The chunk step turns this into a failed document carrying the
    # message, which `status --verbose` lists ahead of the healthy ones -- a
    # stated failure the reader can act on, rather than a worker that appears
    # to hang on one file and cannot be stopped.
    run = longest_unbroken_run(text)
    if run > MAX_UNBROKEN_RUN_CHARS:
        raise ValueError(
            f"unbroken run of {run:,} characters exceeds the {MAX_UNBROKEN_RUN_CHARS:,} "
            "chunking limit; the text layer has probably lost its spacing "
            "(try a different [extraction.backends] pdf backend)"
        )

    encoding = tokenizer()
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
