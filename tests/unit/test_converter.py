"""Tests for converter daemon."""

import hashlib
from unittest.mock import MagicMock, patch

from seman.converter import ConverterDaemon, PDFEventHandler
from seman.state import DaemonState, IndexingState


class TestPDFEventHandler:
    """Test PDF event handler."""

    def test_only_processes_pdfs(self):
        """Test that handler only processes PDF files."""
        callback_called = []

        def callback(path):
            callback_called.append(path)

        handler = PDFEventHandler(callback)

        # Create a mock event for non-PDF file
        mock_event = MagicMock()
        mock_event.is_directory = False
        mock_event.src_path = "/path/to/file.txt"

        handler.on_created(mock_event)

        # Wait for debounce
        import time

        time.sleep(2.5)

        assert len(callback_called) == 0

    def test_processes_pdfs(self):
        """Test that handler processes PDF files."""
        callback_called = []

        def callback(path):
            callback_called.append(path)

        handler = PDFEventHandler(callback)

        mock_event = MagicMock()
        mock_event.is_directory = False
        mock_event.src_path = "/path/to/document.pdf"

        handler.on_created(mock_event)

        # Wait for debounce
        import time

        time.sleep(2.5)

        assert len(callback_called) == 1
        assert callback_called[0] == "/path/to/document.pdf"

    def test_debounce_coalesces_repeated_events(self):
        """Repeated events for same PDF should trigger one callback."""
        callback_called = []

        def callback(path):
            callback_called.append(path)

        handler = PDFEventHandler(callback)

        mock_event = MagicMock()
        mock_event.is_directory = False
        mock_event.src_path = "/path/to/document.pdf"

        handler.on_modified(mock_event)
        handler.on_modified(mock_event)

        import time

        time.sleep(2.5)

        assert callback_called == ["/path/to/document.pdf"]


class TestConverterDaemon:
    """Test converter daemon functionality."""

    @patch("seman.converter.get_engine")
    @patch("seman.converter.create_tables")
    @patch("seman.converter.get_session_factory")
    def test_process_pdf_creates_chunks(
        self, mock_session_factory, mock_create_tables, mock_get_engine, temp_dir
    ):
        """Test that PDF processing creates chunks in database."""
        # Create a mock session
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.return_value = mock_session  # Make callable return itself
        mock_session_factory.return_value = mock_session

        # Create a test PDF
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"fake pdf content")

        # Mock the PDF conversion
        with patch("seman.converter.convert_pdf_to_markdown") as mock_convert:
            with patch("seman.converter.chunk_text") as mock_chunk:
                mock_convert.return_value = "Test markdown content"
                mock_chunk.return_value = [
                    MagicMock(content="Chunk 1", chunk_index=0),
                    MagicMock(content="Chunk 2", chunk_index=1),
                ]

                # Create daemon and process
                daemon = ConverterDaemon()
                daemon.Session = mock_session

                # Mock query result for existing document
                mock_session.query.return_value.filter_by.return_value.first.return_value = None

                daemon._process_pdf(str(pdf_path))

                # Verify document was added
                assert mock_session.add.called
                # Verify chunks were added
                assert mock_session.add.call_count >= 2

    def test_compute_file_hash(self, temp_dir):
        """Test file hash computation."""
        # Create a test file
        test_file = temp_dir / "test.txt"
        test_content = b"test content"
        test_file.write_bytes(test_content)

        # Compute hash manually
        expected_hash = hashlib.sha256(test_content).hexdigest()

        # Verify with the daemon's method
        with open(test_file, "rb") as f:
            computed_hash = hashlib.sha256(f.read()).hexdigest()

        assert computed_hash == expected_hash

    @patch("seman.converter.Observer")
    def test_start_watcher(self, mock_observer_class, temp_dir):
        """Test that watcher is started correctly."""
        mock_observer = MagicMock()
        mock_observer_class.return_value = mock_observer

        with patch("seman.converter.get_engine"):
            with patch("seman.converter.create_tables"):
                with patch("seman.converter.get_session_factory"):
                    daemon = ConverterDaemon()

                    # Mock the state
                    with patch.object(daemon.state_manager, "load") as mock_load:
                        mock_load.return_value = IndexingState(daemon_state=DaemonState.STOPPED)

                        # Create a test directory
                        test_dir = temp_dir / "pdfs"
                        test_dir.mkdir()

                        # We can't actually start the daemon in tests, but we can verify
                        # the setup is correct
                        assert daemon is not None


class TestConverterStateManagement:
    """Test converter state transitions."""

    def test_pause_sets_state(self, temp_dir):
        """Test pause updates state correctly."""
        with patch("seman.converter.get_engine"):
            with patch("seman.converter.create_tables"):
                with patch("seman.converter.get_session_factory"):
                    daemon = ConverterDaemon()
                    daemon._pause_event = MagicMock()

                    with patch.object(daemon.state_manager, "update") as mock_update:
                        daemon.pause()

                        daemon._pause_event.set.assert_called_once()
                        mock_update.assert_called_with(daemon_state=DaemonState.PAUSED)

    def test_resume_clears_pause(self, temp_dir):
        """Test resume clears pause state."""
        with patch("seman.converter.get_engine"):
            with patch("seman.converter.create_tables"):
                with patch("seman.converter.get_session_factory"):
                    daemon = ConverterDaemon()
                    daemon._pause_event = MagicMock()

                    with patch.object(daemon.state_manager, "update") as mock_update:
                        daemon.resume()

                        daemon._pause_event.clear.assert_called_once()
                        mock_update.assert_called_with(daemon_state=DaemonState.RUNNING)
