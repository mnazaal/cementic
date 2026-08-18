"""Integration tests for source watcher thread lifecycle."""

import os
import signal
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cementic.config import Config
from cementic.db import Base, SourceDocument
from cementic.source_watcher import DocumentEventHandler, SourceWatcher


@pytest.fixture
def watcher_config(temp_dir: Path) -> Config:
    """Create watcher config with temp files."""
    config = Config()
    config.source_watcher.log_file = temp_dir / "watcher.log"
    config.source_watcher.state_path = temp_dir / "watcher_state.json"
    return config


@pytest.fixture
def watcher_db(temp_dir: Path):
    """Create SQLite engine + tables for watcher tests."""
    db_path = temp_dir / "watcher_test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    return engine, session_factory, db_path


class TestIgnoredDirectories:
    """Dependency, build and VCS trees are not search corpora.

    Pointing `cementic start` at a project directory otherwise indexed every
    README and note under .git, .venv and node_modules -- thousands of files no
    one meant to search, each costing a hash, an extraction and an embedding.
    """

    def test_scan_skips_ignored_directories(
        self, watcher_config: Config, temp_dir: Path
    ) -> None:
        (temp_dir / "docs").mkdir()
        (temp_dir / "docs" / "keep.md").write_text("keep", encoding="utf-8")
        for ignored in (".git", ".venv", "node_modules"):
            (temp_dir / ignored).mkdir()
            (temp_dir / ignored / "README.md").write_text("skip", encoding="utf-8")
        # Nested inside a kept directory: pruning must apply at any depth.
        (temp_dir / "docs" / "node_modules").mkdir()
        (temp_dir / "docs" / "node_modules" / "readme.md").write_text("skip", encoding="utf-8")

        sw = SourceWatcher(watcher_config)
        seen: list[str] = []
        with patch.object(sw, "_on_file_detected", side_effect=seen.append):
            sw._scan_existing(temp_dir)

        assert [Path(p).name for p in seen] == ["keep.md"]

    def test_live_events_in_ignored_directories_are_dropped(
        self, watcher_config: Config
    ) -> None:
        """A file written into node_modules arrives by inotify, not the walk."""
        handler = DocumentEventHandler(
            lambda path: None, ignore_directories=["node_modules", ".git"]
        )
        assert handler._should_process("/repo/docs/paper.md") is True
        assert handler._should_process("/repo/node_modules/pkg/readme.md") is False
        assert handler._should_process("/repo/.git/notes.md") is False

    def test_empty_ignore_list_indexes_everything(self, watcher_config: Config) -> None:
        handler = DocumentEventHandler(lambda path: None, ignore_directories=[])
        assert handler._should_process("/repo/node_modules/pkg/readme.md") is True


class TestWatchDirectoryPreconditions:
    """A watcher with nothing to watch must fail, not idle.

    Regression: skipping every directory was log-only, so the watcher published
    RUNNING, survived `cementic start`'s startup check, and `cementic status`
    showed healthy workers indexing nothing indefinitely.
    """

    def test_no_watchable_directory_raises(self, watcher_config: Config, temp_dir: Path) -> None:
        sw = SourceWatcher(watcher_config)
        with pytest.raises(RuntimeError, match="No watchable directories"):
            sw._start_watcher([str(temp_dir / "does-not-exist")])

    def test_a_surviving_directory_is_enough(
        self, watcher_config: Config, watcher_db, temp_dir: Path
    ) -> None:
        _engine, session_factory, _db_path = watcher_db
        good = temp_dir / "present"
        good.mkdir()
        sw = SourceWatcher(watcher_config)
        sw.Session = session_factory  # _start_watcher reconciles deletions at the end
        try:
            sw._start_watcher([str(temp_dir / "gone"), str(good)])
            assert sw._watched_roots == [good.resolve()]
            # Regression: the missing directory used to set fatal_reason, so a
            # run that watched the surviving directory for hours exited 1 as a
            # "startup failure" when it finally shut down cleanly.
            assert sw.fatal_reason is None
        finally:
            sw.stop()


class TestConcurrentRegistration:
    """The scan thread and debounce timers can register the same file at once."""

    def test_duplicate_registration_updates_instead_of_failing(
        self, watcher_config: Config, watcher_db, temp_dir: Path
    ) -> None:
        """Regression: read-then-insert raced the unique index on
        (collection, source_path), and the loser's IntegrityError surfaced as
        "Failed to register ..." plus a permanently inflated failure count."""
        _engine, session_factory, _db_path = watcher_db
        doc = temp_dir / "paper.md"
        doc.write_text("content", encoding="utf-8")

        sw = SourceWatcher(watcher_config)
        sw.Session = session_factory
        sw.collection = "docs"
        sw._watched_roots = [temp_dir.resolve()]

        sw._register_document(str(doc))
        sw._register_document(str(doc))  # same file again, as the overlap does

        with session_factory() as session:
            assert session.query(SourceDocument).count() == 1


class TestOfflineDeletionReconciliation:
    """Files deleted while cementic was stopped must drop out of search.

    Regression: deletion was only noticed through a live filesystem event, so a
    file removed between runs kept status="pending" forever and kept matching
    searches with a source_path that no longer existed.
    """

    def test_missing_file_is_marked_deleted_on_startup(
        self, watcher_config: Config, watcher_db, temp_dir: Path
    ) -> None:
        _engine, session_factory, _db_path = watcher_db
        watched = temp_dir / "docs"
        watched.mkdir()
        gone = watched / "gone.md"
        gone.write_text("content", encoding="utf-8")
        kept = watched / "kept.md"
        kept.write_text("content", encoding="utf-8")

        sw = SourceWatcher(watcher_config)
        sw.Session = session_factory
        sw.collection = "docs"
        sw._watched_roots = [watched.resolve()]
        sw._register_document(str(gone))
        sw._register_document(str(kept))

        gone.unlink()  # removed while cementic was not running
        sw._reconcile_deletions()

        with session_factory() as session:
            by_path = {
                Path(doc.source_path).name: doc.status
                for doc in session.query(SourceDocument).all()
            }
        assert by_path["gone.md"] == "deleted"
        assert by_path["kept.md"] == "pending"

    def test_documents_outside_the_watched_roots_are_untouched(
        self, watcher_config: Config, watcher_db, temp_dir: Path
    ) -> None:
        """Only this run's roots are reconciled.

        A document indexed from a directory this run is not watching must not be
        marked deleted merely because it is not visible here.
        """
        _engine, session_factory, _db_path = watcher_db
        watched = temp_dir / "docs"
        watched.mkdir()
        other = temp_dir / "elsewhere"
        other.mkdir()
        outside = other / "outside.md"
        outside.write_text("content", encoding="utf-8")

        sw = SourceWatcher(watcher_config)
        sw.Session = session_factory
        sw.collection = "docs"
        sw._watched_roots = [other.resolve()]
        sw._register_document(str(outside))

        outside.unlink()
        sw._watched_roots = [watched.resolve()]  # a run watching a different root
        sw._reconcile_deletions()

        with session_factory() as session:
            document = session.query(SourceDocument).one()
        assert document.status == "pending"


class TestSourceWatcherLifecycle:
    """Tests for SourceWatcher thread lifecycle."""

    def test_init(self, watcher_config: Config) -> None:
        sw = SourceWatcher(watcher_config)
        assert sw.config is watcher_config
        assert sw.watcher is None
        assert sw._event_handler is None
        assert not sw._shutdown_event.is_set()

    def test_setup_logging_requires_log_file(self, temp_dir: Path) -> None:
        config = Config()
        config.source_watcher.log_file = temp_dir / "watcher.log"
        config.source_watcher.state_path = temp_dir / "state.json"
        sw = SourceWatcher(config)
        sw.config.source_watcher.log_file = None
        with pytest.raises(RuntimeError, match="log file"):
            sw._setup_logging()

    def test_setup_logging_creates_handler(self, watcher_config: Config) -> None:
        sw = SourceWatcher(watcher_config)
        logger = sw._setup_logging()
        assert logger.name == "cementic.source_watcher"

    def test_register_document_sqlite(
        self, watcher_config: Config, watcher_db, temp_dir: Path
    ) -> None:
        """_register_document writes SourceDocument to SQLite DB."""
        engine, session_factory, db_path = watcher_db
        sw = SourceWatcher(watcher_config)
        sw.Session = session_factory

        # Create a minimal PDF file
        pdf_path = temp_dir / "test_watcher.pdf"
        pdf_path.write_text("fake pdf content")

        sw._register_document(str(pdf_path))

        with sw.Session() as session:
            doc = session.query(SourceDocument).first()
            assert doc is not None
            assert doc.source_path == str(pdf_path)
            assert doc.collection == "default"
            assert doc.status == "pending"

    def test_register_document_duplicate_updates(
        self, watcher_config: Config, watcher_db, temp_dir: Path
    ) -> None:
        """Re-registering same PDF updates existing document."""
        engine, session_factory, db_path = watcher_db
        sw = SourceWatcher(watcher_config)
        sw.Session = session_factory

        pdf_path = temp_dir / "dup.pdf"
        pdf_path.write_text("content v1")

        sw._register_document(str(pdf_path))

        # Update content and re-register
        pdf_path.write_text("content v2")
        sw._register_document(str(pdf_path))

        with sw.Session() as session:
            docs = session.query(SourceDocument).all()
            assert len(docs) == 1  # Still one document
            assert docs[0].status == "pending"

    def test_start_already_running_detected(
        self, watcher_config: Config, watcher_db, temp_dir: Path
    ) -> None:
        """start() detects already-running watcher via state file."""
        engine, session_factory, db_path = watcher_db

        # Set up state file with running status and a real PID
        from cementic.state import DaemonState
        sw1 = SourceWatcher(watcher_config)
        sw1.state_manager.update(
            daemon_state=DaemonState.RUNNING,
            pid=os.getpid(),  # our own PID is always running
        )

        sw = SourceWatcher(watcher_config)
        # Patch _start_watcher so it doesn't actually start a watchdog observer
        with patch.object(sw, "_start_watcher"):
            sw.start([str(temp_dir)], collection="test")
        # Should have detected running PID and returned early
        assert sw.watcher is None

    def test_stop_cleans_up(self, watcher_config: Config) -> None:
        """stop() sets shutdown event and writes STOPPED state."""
        sw = SourceWatcher(watcher_config)
        sw.stop()
        assert sw._shutdown_event.is_set()
        state = sw.state_manager.load()
        from cementic.state import DaemonState
        assert state.daemon_state == DaemonState.STOPPED

    def test_handle_shutdown_only_sets_the_flag(self, watcher_config: Config) -> None:
        """_handle_shutdown sets the shutdown event without calling stop().

        stop() takes StateManager's lock and joins the observer thread; doing
        either from a signal handler deadlocked the process against its own
        SIGTERM. start()'s `finally` performs the cleanup instead.
        """
        sw = SourceWatcher(watcher_config)
        called = []
        sw.stop = lambda: called.append(True)  # type: ignore[method-assign]
        sw._handle_shutdown(signal.SIGTERM, None)
        assert called == []
        assert sw._shutdown_event.is_set()


class TestDocumentEventHandler:
    """Tests for DocumentEventHandler debounce and file detection."""

    def test_should_process_pdf(self) -> None:
        callback_called = []
        handler = DocumentEventHandler(lambda p: callback_called.append(p))
        assert handler._should_process("test.pdf") is True
        assert handler._should_process("test.PDF") is True
        assert handler._should_process("test.docx") is False
        assert handler._should_process("noext") is False

    def test_debounced_process_fires(self) -> None:
        callback_called = []
        handler = DocumentEventHandler(lambda p: callback_called.append(p))
        handler._debounce_seconds = 0.01

        handler._debounced_process("/tmp/test.pdf")

        # Wait briefly for timer to fire
        time.sleep(0.1)
        assert len(callback_called) == 1
        assert callback_called[0] == "/tmp/test.pdf"

    def test_debounced_cancel_replaced(self) -> None:
        """Second trigger on same path cancels first timer."""
        callback_called = []
        handler = DocumentEventHandler(lambda p: callback_called.append(p))
        handler._debounce_seconds = 0.03

        handler._debounced_process("/tmp/test.pdf")
        handler._debounced_process("/tmp/test.pdf")  # Same path → cancels first

        time.sleep(0.2)
        # Only one callback should have fired (first was cancelled)
        assert len(callback_called) == 1

    def test_cancel_all_clears_timers(self) -> None:
        callback_called = []
        handler = DocumentEventHandler(lambda p: callback_called.append(p))
        handler._debounce_seconds = 0.05

        handler._debounced_process("/tmp/test.pdf")
        handler.cancel_all()

        time.sleep(0.1)
        assert len(callback_called) == 0
        assert len(handler._timers) == 0

    def test_on_created_non_pdf_ignored(self) -> None:
        callback_called = []
        handler = DocumentEventHandler(lambda p: callback_called.append(p))
        from watchdog.events import FileCreatedEvent
        event = FileCreatedEvent("/tmp/test.docx")
        handler.on_created(event)
        # No callback should have fired for non-PDF
        assert len(callback_called) == 0

    def test_on_modified_debounces(self) -> None:
        callback_called = []
        handler = DocumentEventHandler(lambda p: callback_called.append(p))
        handler._debounce_seconds = 0.01
        from watchdog.events import FileModifiedEvent
        event = FileModifiedEvent("/tmp/test.pdf")
        handler.on_modified(event)
        time.sleep(0.1)
        assert len(callback_called) == 1


class TestDirectoryMoveDeletion:
    """A directory moved out of the watched tree must drop out of search live.

    Measured against watchdog's inotify backend (2026-08-18): `mv watch/sub
    ../outside/` delivers one DirDeletedEvent and **no per-file deletions**, so
    the per-file on_deleted path never fires and every document under the moved
    directory stayed "present" until the next restart's reconciliation --
    searches kept matching paths that no longer existed (fifth review §2.9).
    Moves *within* the tree and moves *in* deliver per-file events and were
    already covered; the watched root itself moving delivers nothing at all
    (documented limitation -- restart reconciles).
    """

    def _watcher_with_registered_docs(self, watcher_config, session_factory, watched):
        sub = watched / "sub"
        sub.mkdir()
        inside = sub / "doc.md"
        inside.write_text("content", encoding="utf-8")
        sibling = watched / "sibling.md"
        sibling.write_text("content", encoding="utf-8")
        sw = SourceWatcher(watcher_config)
        sw.Session = session_factory
        sw.collection = "docs"
        sw._watched_roots = [watched.resolve()]
        sw._register_document(str(inside))
        sw._register_document(str(sibling))
        return sw

    def test_documents_under_a_removed_directory_are_marked_deleted(
        self, watcher_config: Config, watcher_db, temp_dir: Path
    ) -> None:
        _engine, session_factory, _db_path = watcher_db
        watched = temp_dir / "docs"
        watched.mkdir()
        sw = self._watcher_with_registered_docs(watcher_config, session_factory, watched)

        outside = temp_dir / "outside"
        (watched / "sub").rename(outside)  # the move-out that emits no file events
        sw._on_directory_deleted(str(watched / "sub"))

        with session_factory() as session:
            by_path = {
                Path(doc.source_path).name: doc.status
                for doc in session.query(SourceDocument).all()
            }
        assert by_path["doc.md"] == "deleted"
        assert by_path["sibling.md"] == "pending"

    def test_a_prefix_sharing_sibling_directory_is_not_claimed(
        self, watcher_config: Config, watcher_db, temp_dir: Path
    ) -> None:
        """`/a/docs` vanishing must not delete `/a/docs-archive`'s documents."""
        _engine, session_factory, _db_path = watcher_db
        watched = temp_dir / "docs"
        watched.mkdir()
        archive = watched / "sub-archive"
        archive.mkdir()
        keeper = archive / "keep.md"
        keeper.write_text("content", encoding="utf-8")
        sw = self._watcher_with_registered_docs(watcher_config, session_factory, watched)
        sw._register_document(str(keeper))

        sw._on_directory_deleted(str(watched / "sub"))

        with session_factory() as session:
            by_path = {
                Path(doc.source_path).name: doc.status
                for doc in session.query(SourceDocument).all()
            }
        assert by_path["keep.md"] == "pending"
        assert by_path["doc.md"] == "deleted"

    def test_the_live_event_stream_reaches_the_prefix_delete(
        self, watcher_config: Config, watcher_db, temp_dir: Path
    ) -> None:
        """End to end through a real observer: the DirDeletedEvent produced by
        an actual move-out must arrive at the directory callback."""
        import time as time_module

        from watchdog.observers import Observer

        from cementic.source_watcher import DocumentEventHandler

        watched = temp_dir / "docs"
        sub = watched / "sub"
        sub.mkdir(parents=True)
        (sub / "doc.md").write_text("content", encoding="utf-8")

        deleted_dirs: list[str] = []
        handler = DocumentEventHandler(
            lambda path: None,
            lambda path: None,
            watched_roots=[watched.resolve()],
            delete_directory_callback=deleted_dirs.append,
        )
        observer = Observer()
        observer.schedule(handler, str(watched), recursive=True)
        observer.start()
        try:
            time_module.sleep(0.3)
            sub.rename(temp_dir / "outside")
            deadline = time_module.monotonic() + 5.0
            while not deleted_dirs and time_module.monotonic() < deadline:
                time_module.sleep(0.05)
        finally:
            observer.stop()
            observer.join(timeout=5)

        assert deleted_dirs and deleted_dirs[0].endswith("sub")
