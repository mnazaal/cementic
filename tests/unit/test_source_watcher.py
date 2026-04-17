"""Tests for document watcher daemon."""

import hashlib
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cementic.source_watcher import PDFEventHandler, SourceWatcher
from cementic.state import DaemonState


class TestPDFEventHandler:
    """Test PDF event handler."""

    def test_only_processes_pdfs(self):
        callback_called = []

        def callback(path):
            callback_called.append(path)

        handler = PDFEventHandler(callback)
        mock_event = MagicMock(is_directory=False, src_path="/path/to/file.txt")
        handler.on_created(mock_event)
        time.sleep(2.5)
        assert callback_called == []

    def test_processes_pdfs(self):
        callback_called = []

        def callback(path):
            callback_called.append(path)

        handler = PDFEventHandler(callback)
        mock_event = MagicMock(is_directory=False, src_path="/path/to/document.pdf")
        handler.on_created(mock_event)
        time.sleep(2.5)
        assert callback_called == ["/path/to/document.pdf"]


class TestSourceWatcher:
    """Test document watcher behavior."""

    def test_register_pdf_updates_source_document(self, temp_dir):
        pdf_path = temp_dir / "test.pdf"
        pdf_bytes = b"fake pdf content"
        pdf_path.write_bytes(pdf_bytes)

        document = None
        added = []

        class QueryMock:
            def filter_by(self, **kwargs):
                return self

            def first(self):
                return document

        class SessionMock:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def query(self, model):
                return QueryMock()

            def add(self, value):
                nonlocal document
                document = value
                added.append(value)

            def commit(self):
                return None

        daemon = SourceWatcher()
        daemon.Session = MagicMock(return_value=SessionMock())
        daemon.state_manager.update = MagicMock()

        daemon._register_pdf(str(pdf_path))

        assert added
        assert document is not None
        assert document.file_hash == hashlib.sha256(pdf_bytes).hexdigest()
        assert document.status == "pending"

    @patch("cementic.source_watcher.create_tables")
    @patch("cementic.source_watcher.get_session_factory")
    @patch("cementic.source_watcher.get_engine")
    def test_start_watcher_sets_running_state(
        self, mock_get_engine, mock_session_factory, mock_create_tables
    ):
        daemon = SourceWatcher()
        daemon.state_manager.load = MagicMock(
            return_value=SimpleNamespace(daemon_state=DaemonState.STOPPED, pid=None)
        )
        daemon.state_manager.update = MagicMock()
        daemon._start_watcher = MagicMock()
        daemon._shutdown_event = MagicMock(is_set=MagicMock(side_effect=[True]))
        mock_session_factory.return_value = MagicMock()

        daemon.start(["/tmp"], collection="research")

        assert daemon.collection == "research"
        daemon._start_watcher.assert_called_once_with(["/tmp"])


class TestSourceWatcherStateManagement:
    """Test source watcher state transitions."""

    def test_stop_sets_shutdown(self):
        daemon = SourceWatcher()
        daemon._shutdown_event = MagicMock()
        daemon.watcher = None

        with patch.object(daemon.state_manager, "update") as mock_update:
            daemon.stop()

        daemon._shutdown_event.set.assert_called_once()
        mock_update.assert_called_with(daemon_state=DaemonState.STOPPED, pid=None)
