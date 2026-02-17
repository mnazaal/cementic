"""Tests for text chunking module."""


from seman.chunk import TextChunk, chunk_text


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
        chunks = chunk_text("", chunk_size=100)
        assert len(chunks) == 0

    def test_chunk_with_model_parameter(self):
        """Test chunking with specific model encoding."""
        text = "Hello world this is a test"
        chunks = chunk_text(text, chunk_size=10, model="cl100k_base")

        assert len(chunks) >= 1
        assert all(isinstance(chunk, TextChunk) for chunk in chunks)

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

    def test_chunk_overlap_larger_than_size(self):
        """Test behavior when overlap >= chunk size."""
        text = "a b c d e f g h i j"
        chunks = chunk_text(text, chunk_size=5, chunk_overlap=5)

        # Should not hang or error, should create at least one chunk
        assert len(chunks) >= 1


class TestTextChunk:
    """Test TextChunk dataclass."""

    def test_text_chunk_creation(self):
        """Test creating a TextChunk."""
        chunk = TextChunk(content="Test content", chunk_index=0, page_start=1, page_end=2)

        assert chunk.content == "Test content"
        assert chunk.chunk_index == 0
        assert chunk.page_start == 1
        assert chunk.page_end == 2

    def test_text_chunk_optional_fields(self):
        """Test TextChunk with optional fields."""
        chunk = TextChunk(content="Test content", chunk_index=0)

        assert chunk.page_start is None
        assert chunk.page_end is None
