"""Tests for document watcher daemon."""

import hashlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cementic.db import (
    Base,
    Chunk,
    ChunkedDocument,
    ChunkProfile,
    ExtractedDocument,
    ExtractorProfile,
    SourceDocument,
)
from cementic.source_watcher import DocumentEventHandler, SourceWatcher
from cementic.state import DaemonState


class _NonClosingSession:
    """Hand the same session to code that uses `with self.Session() as s`.

    The context manager would otherwise close the session and detach the rows
    the test still needs to inspect.
    """

    def __init__(self, session):
        self._session = session

    def __enter__(self):
        return self._session

    def __exit__(self, exc_type, exc, tb):
        return False


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
        """Deleting a document must take its chunks with it, not just flag the row.

        Search filters on the vector row alone now -- the freshness and
        `status <> 'deleted'` joins were what stopped the planner using the ANN
        index -- so a deleted document whose chunks survive is a deleted
        document that still comes back in results.
        """
        pdf_path = temp_dir / "test.pdf"
        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        session = sessionmaker(bind=engine, expire_on_commit=False)()

        document = SourceDocument(
            source_path=str(pdf_path.resolve()),
            collection="default",
            file_hash="hash",
            status="pending",
        )
        session.add(document)
        session.flush()
        extractor = ExtractorProfile(name="x", fingerprint="ex", config_json="{}")
        chunk_profile = ChunkProfile(fingerprint="cp", config_json="{}")
        session.add_all([extractor, chunk_profile])
        session.flush()
        extracted = ExtractedDocument(
            document_id=document.id, extractor_profile_id=extractor.id, status="done"
        )
        session.add(extracted)
        session.flush()
        chunked = ChunkedDocument(
            extracted_document_id=extracted.id, chunk_profile_id=chunk_profile.id, status="done"
        )
        session.add(chunked)
        session.flush()
        session.add(
            Chunk(
                document_id=document.id,
                chunked_document_id=chunked.id,
                chunk_index=0,
                content="text",
            )
        )
        session.commit()

        daemon = SourceWatcher()
        daemon.Session = MagicMock(return_value=_NonClosingSession(session))
        daemon._watched_roots = [temp_dir.resolve()]

        daemon._mark_document_deleted(str(pdf_path))

        assert document.status == "deleted"
        assert document.file_hash is None
        assert session.query(Chunk).count() == 0

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

    def test_shutdown_handler_does_not_touch_the_state_lock(self, tmp_path):
        """The SIGTERM handler must only set the flag.

        Regression: it used to call stop(), which takes StateManager's lock. A
        signal delivered while the main thread was already inside update() --
        which happens on every registered document -- re-entered that lock and
        hung the process. Because the deadlocked handler *was* the SIGTERM
        handler, `cementic stop` could then never stop it; only SIGKILL worked.
        """
        daemon = SourceWatcher()
        daemon.state_manager.state_path = tmp_path / "state.json"

        with patch.object(daemon.state_manager, "update") as mock_update:
            # Simulate the signal landing while the main thread holds the lock.
            with daemon.state_manager._lock:
                daemon._handle_shutdown(15, None)

        assert daemon._shutdown_event.is_set()
        mock_update.assert_not_called()

    def test_scan_existing_stops_on_shutdown(self, tmp_path):
        """A shutdown mid-scan abandons the walk instead of indexing the rest."""
        for name in ("a.md", "b.md", "c.md"):
            (tmp_path / name).write_text("content", encoding="utf-8")
        daemon = SourceWatcher()
        seen: list[str] = []

        def record(path: str) -> None:
            seen.append(path)
            daemon._shutdown_event.set()

        with patch.object(daemon, "_on_file_detected", side_effect=record):
            daemon._scan_existing(tmp_path)

        assert len(seen) == 1
