"""Tests for text chunking module."""
import pytest
import tiktoken

from cementic.chunk import (
    MAX_UNBROKEN_RUN_CHARS,
    TOKENIZER,
    TextChunk,
    chunk_text,
    longest_unbroken_run,
)


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

    def test_chunk_index_stays_contiguous_when_slices_come_back_empty(self):
        """Widening to character boundaries can empty a slice; the index must not skip.

        At a small chunk_size over multi-byte text every token can belong to one
        character, so the widened slice is empty and no chunk is emitted. The
        counter used to advance anyway, numbering 20 emoji chunks 0, 2, 4 ... 38
        -- so chunk_index disagreed with position, ChunkedDocument.total_chunks
        agreed with neither, and `cementic chunk` emitted JSONL with gaps.
        """
        chunks = chunk_text("😀" * 20, chunk_size=1, chunk_overlap=0)

        assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))

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


class TestMultiByteCharacterBoundaries:
    """Chunk boundaries must not cut through multi-byte characters.

    The tokenizer is byte-level, so a token slice can split one character.
    Decoding such a slice directly emitted U+FFFD, and that corruption was what
    got embedded and shown in search results -- silently, for any corpus with
    emoji or non-Latin scripts.
    """

    CASES = {
        "emoji": "Vector search is great \U0001f600\U0001f601\U0001f602\U0001f923 " * 60,
        "devanagari": "सूचनांक खोज की दुनिया " * 60,
        "cjk": "上下文窗口と埋め込み検索 " * 60,
    }

    @pytest.mark.parametrize("name", sorted(CASES))
    def test_no_replacement_characters_at_production_settings(self, name):
        chunks = chunk_text(self.CASES[name], chunk_size=512, chunk_overlap=128)

        assert chunks
        assert not any("�" in chunk.content for chunk in chunks)

    @pytest.mark.parametrize("name", sorted(CASES))
    def test_no_replacement_characters_at_tiny_chunk_sizes(self, name):
        """Small chunks split characters far more often, so they are the
        sensitive case: at 512/128 some scripts happen not to trip it."""
        chunks = chunk_text(self.CASES[name], chunk_size=10, chunk_overlap=3)

        assert not any("�" in chunk.content for chunk in chunks)

    @pytest.mark.parametrize("name", sorted(CASES))
    def test_non_overlapping_chunks_reassemble_losslessly(self, name):
        text = self.CASES[name]

        chunks = chunk_text(text, chunk_size=16, chunk_overlap=0)

        assert "".join(chunk.content for chunk in chunks) == text


class TestUnbrokenRunGuard:
    """The cap that keeps one spaceless run out of tiktoken's BPE loop."""

    def test_measures_the_longest_run(self):
        assert longest_unbroken_run("ab cdef g") == 4
        assert longest_unbroken_run("   ") == 0
        assert longest_unbroken_run("") == 0

    def test_ordinary_text_is_untouched(self):
        """Whatever the guard does, it must not change output for real prose.

        Same total length as the refused case below, spaced normally.
        """
        text = ("word " * (MAX_UNBROKEN_RUN_CHARS // 5)) + "x"

        chunks = chunk_text(text, chunk_size=512, chunk_overlap=0)

        assert "".join(chunk.content for chunk in chunks) == text

    def test_one_long_run_is_refused_with_a_reason(self):
        text = "x" * (MAX_UNBROKEN_RUN_CHARS + 1)

        with pytest.raises(ValueError, match="unbroken run"):
            chunk_text(text, chunk_size=512, chunk_overlap=0)

    def test_a_run_at_the_limit_is_allowed(self, monkeypatch):
        """The cap is exclusive: exactly the limit still chunks.

        Run against a lowered cap rather than the real one. Encoding a genuine
        100,000-character run costs 3.5 s -- which is the cost this guard exists
        to bound, so paying it once per test run to assert a boundary is the
        wrong trade.
        """
        monkeypatch.setattr("cementic.chunk.MAX_UNBROKEN_RUN_CHARS", 1_000)
        text = "x" * 1_000

        chunks = chunk_text(text, chunk_size=512, chunk_overlap=0)

        assert "".join(chunk.content for chunk in chunks) == text

    def test_the_cap_is_read_at_call_time(self, monkeypatch):
        monkeypatch.setattr("cementic.chunk.MAX_UNBROKEN_RUN_CHARS", 1_000)

        with pytest.raises(ValueError, match="1,001 characters"):
            chunk_text("x" * 1_001, chunk_size=512, chunk_overlap=0)
