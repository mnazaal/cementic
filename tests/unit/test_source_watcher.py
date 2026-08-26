"""Tests for document watcher daemon."""

import hashlib
import os
from pathlib import Path
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
from cementic.supervisor import process_start_token


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

    def test_stop_stops_the_observer_before_cancelling_timers(self, tmp_path):
        """Regression: cancel_all ran first, so an event landing before the
        observer stopped re-armed a debounce timer, which then fired into a
        watcher whose state already said STOPPED."""
        daemon = SourceWatcher()
        order: list[str] = []
        handler = MagicMock()
        handler.cancel_all.side_effect = lambda: order.append("cancel_all")
        observer = MagicMock()
        observer.stop.side_effect = lambda: order.append("observer.stop")
        observer.join.side_effect = lambda: order.append("observer.join")
        daemon._event_handler = handler
        daemon.watcher = observer

        with patch.object(daemon.state_manager, "update"):
            daemon.stop()

        assert order == ["observer.stop", "observer.join", "cancel_all"]

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

    def test_scan_existing_passes_symlinks_through_to_be_recorded(self, tmp_path):
        """Regression: the scan filtered symlinks *before* _on_file_detected, so
        `cementic start` on a tree of symlinks recorded nothing -- no counter,
        no skip entry -- while a live event for the same file was recorded by
        _register_document. Broken symlinks fail is_file() and vanished too."""
        (tmp_path / "real.md").write_text("content", encoding="utf-8")
        (tmp_path / "link.md").symlink_to(tmp_path / "real.md")
        (tmp_path / "dangling.md").symlink_to(tmp_path / "nowhere.md")
        daemon = SourceWatcher()
        seen: list[str] = []

        with patch.object(daemon, "_on_file_detected", side_effect=seen.append):
            daemon._scan_existing(tmp_path)

        assert sorted(Path(p).name for p in seen) == ["dangling.md", "link.md", "real.md"]

    def test_scan_walk_error_records_the_unreadable_path(self, tmp_path):
        """Regression: an unreadable subtree bumped failed_count but recorded no
        path, so `status --verbose` could not say *what* was missing."""
        daemon = SourceWatcher()

        def fake_walk(directory, onerror=None):
            onerror(PermissionError(13, "Permission denied", str(tmp_path / "locked")))
            return iter([])

        with patch("cementic.source_watcher.os.walk", side_effect=fake_walk):
            with patch.object(daemon.state_manager, "record_skipped") as mock_skip:
                daemon._scan_existing(tmp_path)

        mock_skip.assert_called_once()
        path, reason = mock_skip.call_args.args
        assert path == str(tmp_path / "locked")
        assert "unreadable during scan" in reason

    def test_registration_failure_records_the_path(self, tmp_path):
        """Regression: an exception in _register_document bumped failed_count
        but left the path only in the log file."""
        daemon = SourceWatcher()

        with patch.object(daemon, "_register_document", side_effect=RuntimeError("db down")):
            with patch.object(daemon.state_manager, "record_skipped") as mock_skip:
                daemon._on_file_detected(str(tmp_path / "doc.md"))

        mock_skip.assert_called_once()
        path, reason = mock_skip.call_args.args
        assert path == str(tmp_path / "doc.md")
        assert "registration failed" in reason
        assert "db down" in reason

    def test_file_deletion_failure_records_the_path(self, tmp_path):
        """Regression: an exception in _mark_document_deleted was only logged

        and dropped, unlike a registration failure three lines above it, which
        already routed through record_skipped. A file that fails to be marked
        deleted then kept matching searches with nothing visible in `status`.
        """
        daemon = SourceWatcher()

        with patch.object(daemon, "_mark_document_deleted", side_effect=RuntimeError("db down")):
            with patch.object(daemon.state_manager, "record_skipped") as mock_skip:
                daemon._on_file_deleted(str(tmp_path / "doc.md"))

        mock_skip.assert_called_once()
        path, reason = mock_skip.call_args.args
        assert path == str(tmp_path / "doc.md")
        assert "deletion failed" in reason
        assert "db down" in reason

    def test_directory_deletion_failure_records_the_path(self, tmp_path):
        """Same regression as the file-deletion path, for a moved-out directory."""
        daemon = SourceWatcher()

        with patch.object(
            daemon, "_mark_documents_deleted_under", side_effect=RuntimeError("db down")
        ):
            with patch.object(daemon.state_manager, "record_skipped") as mock_skip:
                daemon._on_directory_deleted(str(tmp_path / "sub"))

        mock_skip.assert_called_once()
        path, reason = mock_skip.call_args.args
        assert path == str(tmp_path / "sub")
        assert "directory deletion failed" in reason
        assert "db down" in reason

    def test_handler_failure_publishes_last_error(self, tmp_path):
        """Regression: the watcher never wrote `last_error` at all, so the

        "last error" row `cli.py:748-755` renders for it was permanently dead
        -- an operational failure here was only visible in a log file the user
        has to know to check.
        """
        daemon = SourceWatcher()
        daemon.state_manager.state_path = tmp_path / "state.json"

        with patch.object(daemon, "_register_document", side_effect=RuntimeError("db down")):
            daemon._on_file_detected(str(tmp_path / "doc.md"))

        state = daemon.state_manager.load()
        assert state.last_error is not None
        assert "db down" in state.last_error
        assert state.last_error_at is not None


class TestErrorStateIsRetracted:
    """A published last_error must not outlive the condition that caused it."""

    def test_a_clean_registration_clears_a_published_error(self, tmp_path):
        """Regression: the watcher published last_error and never retracted it,
        so `cementic status` reported one transient failure forever. Found by
        running the CLI against the live corpus, not by the suite."""
        watcher = SourceWatcher()
        watcher.state_manager.state_path = tmp_path / "state.json"

        with patch.object(watcher, "_register_document", side_effect=RuntimeError("db down")):
            watcher._on_file_detected(str(tmp_path / "a.md"))
        assert watcher.state_manager.load().last_error is not None

        with patch.object(watcher, "_register_document"):
            watcher._on_file_detected(str(tmp_path / "b.md"))

        state = watcher.state_manager.load()
        assert state.last_error is None
        assert state.last_error_at is None

    def test_a_clean_registration_does_not_write_when_nothing_was_reported(self, tmp_path):
        """The retraction is gated: registering a file must not rewrite the
        state file when no error is standing."""
        watcher = SourceWatcher()
        watcher.state_manager.state_path = tmp_path / "state.json"

        with patch.object(watcher, "_register_document"):
            with patch.object(watcher.state_manager, "update") as mock_update:
                watcher._on_file_detected(str(tmp_path / "a.md"))

        mock_update.assert_not_called()


class TestFatalStartupReasonReachesTheStateFile:
    """A startup failure past `cementic start`'s 2 s grace only shows up in

    `cementic status` if it lands in `last_error` -- `_STARTUP_GRACE_SECONDS`
    (cli.py) is far shorter than the source watcher's own startup work, so a
    fatal condition discovered after `start` already reported success was
    previously reported nowhere `status` or `doctor` look. Regression:
    `report_fatal` logged and echoed to stderr but never touched the state
    file.

    The one exception is the already-running fatal: that state file belongs to
    the *running* worker, whose clear paths are gated on its own in-process
    flags, so a losing duplicate start publishing there left `status`
    reporting a healthy worker with a standing error until the next restart.
    """

    def test_already_running_failure_is_not_published_to_the_live_worker(self, tmp_path):
        watcher = SourceWatcher()
        watcher.state_manager.state_path = tmp_path / "state.json"
        watcher.state_manager.update(
            daemon_state=DaemonState.RUNNING,
            pid=os.getpid(),
            start_token=process_start_token(os.getpid()),
        )

        watcher.start([str(tmp_path)], collection="c")

        # The loser still reports the reason (runner exit code, stderr) --
        # just not into the state file the running worker owns.
        assert watcher.fatal_reason is not None
        assert "already running" in watcher.fatal_reason
        state = watcher.state_manager.load()
        assert state.last_error is None
        assert state.last_error_at is None

    @patch("cementic.source_watcher.create_tables")
    @patch("cementic.source_watcher.get_session_factory")
    @patch("cementic.source_watcher.get_engine")
    def test_a_clean_start_after_a_failed_one_clears_last_error(
        self, mock_get_engine, mock_session_factory, mock_create_tables, tmp_path
    ):
        """The clean start is the first one in this file to get *past* the
        already-running check, so it is the first to reach the database. Unit
        tests have no server: patch the engine seam, as the sibling start test
        does, or this passes only on a developer machine with the container up.
        """
        state_path = tmp_path / "state.json"
        stale = SourceWatcher()
        stale.state_manager.state_path = state_path
        # A previous run's published fatal (any publishing fatal; the
        # already-running one deliberately does not publish).
        stale.state_manager.update(
            daemon_state=DaemonState.STOPPED,
            pid=None,
            last_error="injected fatal from a previous run",
            last_error_at="2026-01-01T00:00:00+00:00",
        )
        assert stale.state_manager.load().last_error is not None

        watch_dir = tmp_path / "watched"
        watch_dir.mkdir()
        watcher = SourceWatcher()
        watcher.state_manager.state_path = state_path
        # The wait loop past startup is irrelevant here; skip it.
        watcher._shutdown_event.set()

        watcher.start([str(watch_dir)], collection="c")

        state = watcher.state_manager.load()
        assert state.last_error is None
        assert state.last_error_at is None
