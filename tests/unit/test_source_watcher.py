"""Tests for document watcher daemon."""

import hashlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cementic.source_watcher import DocumentEventHandler, SourceWatcher
from cementic.state import DaemonState


class TestDocumentEventHandler:
    """Test content-type filtering in the document event handler."""

    def test_ignores_unsupported_types(self):
        callback_called = []
        handler = DocumentEventHandler(callback_called.append)
        mock_event = MagicMock(is_directory=False, src_path="/path/to/file.docx")
        handler.on_created(mock_event)
        assert callback_called == []

    def test_processes_pdfs(self):
        callback_called = []
        handler = DocumentEventHandler(callback_called.append)
        handler._debounce_seconds = 0
        mock_event = MagicMock(is_directory=False, src_path="/path/to/document.pdf")
        handler.on_created(mock_event)
        handler.cancel_all()
        assert callback_called == ["/path/to/document.pdf"]

    def test_processes_markdown(self):
        callback_called = []
        handler = DocumentEventHandler(callback_called.append)
        handler._debounce_seconds = 0
        mock_event = MagicMock(is_directory=False, src_path="/notes/readme.md")
        handler.on_created(mock_event)
        handler.cancel_all()
        assert callback_called == ["/notes/readme.md"]

    def test_deleted_supported_file_calls_delete_callback(self):
        deleted = []
        handler = DocumentEventHandler(lambda _: None, deleted.append)
        mock_event = MagicMock(is_directory=False, src_path="/notes/readme.md")

        handler.on_deleted(mock_event)

        assert deleted == ["/notes/readme.md"]

    def test_moved_file_deletes_old_path_and_processes_new_path(self):
        detected = []
        deleted = []
        handler = DocumentEventHandler(detected.append, deleted.append)
        handler._debounce_seconds = 0
        mock_event = MagicMock(
            is_directory=False,
            src_path="/notes/old.md",
            dest_path="/notes/new.md",
        )

        handler.on_moved(mock_event)
        handler.cancel_all()

        assert deleted == ["/notes/old.md"]
        assert detected == ["/notes/new.md"]


class TestSourceWatcher:
    """Test document watcher behavior."""

    def test_register_document_updates_source_document(self, temp_dir):
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

        daemon._register_document(str(pdf_path))

        assert added
        assert document is not None
        assert document.file_hash == hashlib.sha256(pdf_bytes).hexdigest()
        assert document.status == "pending"

    def test_mark_document_deleted_updates_existing_source_document(self, temp_dir):
        pdf_path = temp_dir / "test.pdf"
        document = SimpleNamespace(
            source_path=str(pdf_path),
            collection="default",
            file_hash="hash",
            status="pending",
            error_message="old",
        )

        class QueryMock:
            def filter_by(self, **kwargs):
                assert kwargs == {"source_path": str(pdf_path.resolve()), "collection": "default"}
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

            def commit(self):
                return None

        daemon = SourceWatcher()
        daemon.Session = MagicMock(return_value=SessionMock())
        daemon._watched_roots = [temp_dir.resolve()]

        daemon._mark_document_deleted(str(pdf_path))

        assert document.status == "deleted"
        assert document.file_hash is None
        assert document.error_message is None

    def test_setup_logging_reuses_existing_file_handler(self, temp_dir):
        """Repeated construction should not duplicate file handlers."""
        config = MagicMock()
        config.source_watcher.log_file = temp_dir / "watcher.log"
        config.source_watcher.state_path = temp_dir / "watcher_state.json"

        first = SourceWatcher(config)
        second = SourceWatcher(config)

        handlers = [
            handler for handler in second._logger.handlers
            if getattr(handler, "baseFilename", None) == str(config.source_watcher.log_file)
        ]
        assert first._logger is second._logger
        assert len(handlers) == 1

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
