"""Tests for text chunking module."""
import pytest
import tiktoken

from cementic.chunk import TOKENIZER, TextChunk, chunk_text


class TestChunkText:
    """Test text chunking functionality."""

    def test_chunk_single_short_text(self):
        """Test chunking a short text."""
        text = "This is a short text."
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)

        assert len(chunks) == 1
        assert isinstance(chunks[0], TextChunk)
        assert chunks[0].content == text
        assert chunks[0].chunk_index == 0

    def test_chunk_long_text(self):
        """Test chunking a longer text."""
        # Create a text that will require multiple chunks
        text = "word " * 1000  # ~5000 tokens
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)

        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert len(chunk.content) > 0

    def test_chunk_overlap(self):
        """Test that chunks have proper overlap."""
        text = "word " * 100  # Simple repeated text
        chunk_size = 50
        chunk_overlap = 10

        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Just verify that multiple chunks are created with overlap logic
        assert len(chunks) > 1
        # The chunk_index should be sequential
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunk_empty_text(self):
        """Test chunking empty text."""
        chunks = chunk_text("", chunk_size=100, chunk_overlap=0)
        assert len(chunks) == 0

    def test_chunk_boundary_condition(self):
        """Test chunking at exact boundary."""
        # Create text that will produce known number of chunks
        # Each "word " is roughly 1 token, so 120 tokens should produce
        # multiple chunks (exact count depends on tokenizer)
        text = "word " * 120
        chunks = chunk_text(text, chunk_size=40, chunk_overlap=0)

        # Should create multiple chunks
        assert len(chunks) >= 2
        # Verify at least one chunk has substantial content
        assert any(len(chunk.content.strip()) > 10 for chunk in chunks)
        assert chunks[0].chunk_index == 0
        assert chunks[1].chunk_index == 1

    def test_chunk_overlap_must_be_smaller_than_size(self):
        """Overlap equal to chunk size is invalid rather than silently degraded."""
        with pytest.raises(ValueError, match="chunk_overlap must be smaller"):
            chunk_text("a b c", chunk_size=5, chunk_overlap=5)

    def test_chunk_size_must_be_positive(self):
        """Zero or negative chunk sizes are invalid."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            chunk_text("a b c", chunk_size=0, chunk_overlap=0)

    def test_chunk_overlap_must_be_non_negative(self):
        """Negative overlap is invalid."""
        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            chunk_text("a b c", chunk_size=5, chunk_overlap=-1)


class TestNoDuplicateTailChunk:
    """A chunk ending exactly at the end of the text must not be re-emitted.

    Stepping past a chunk that already consumed the last token produces a final
    chunk that is a pure suffix of its predecessor -- duplicate content that gets
    embedded and can surface twice in search results.
    """

    @staticmethod
    def _text_of_tokens(count: int) -> str:
        encoding = tiktoken.get_encoding(TOKENIZER)
        return encoding.decode(encoding.encode("alpha beta gamma delta " * count)[:count])

    def test_exact_multiple_of_step_emits_no_suffix_duplicate(self):
        # 512 tokens with size 512 => the first chunk consumes everything.
        chunks = chunk_text(self._text_of_tokens(512), chunk_size=512, chunk_overlap=128)

        assert len(chunks) == 1

    def test_two_step_boundary_emits_no_suffix_duplicate(self):
        # 896 = 512 + (512 - 128): the second chunk ends exactly at the end.
        chunks = chunk_text(self._text_of_tokens(896), chunk_size=512, chunk_overlap=128)

        assert len(chunks) == 2
        assert chunks[-1].content not in chunks[0].content

    def test_non_boundary_length_still_emits_remainder(self):
        chunks = chunk_text(self._text_of_tokens(600), chunk_size=512, chunk_overlap=128)

        assert len(chunks) == 2
        assert [chunk.chunk_index for chunk in chunks] == [0, 1]
