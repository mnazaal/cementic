"""Document watcher that registers source files for downstream pipeline processing."""

from __future__ import annotations

import hashlib
import logging
import os
import signal
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from watchdog.events import DirMovedEvent, FileMovedEvent, FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from cementic.config import Config, get_config
from cementic.db import SourceDocument, create_tables, get_engine, get_session_factory
from cementic.extract import supported_extensions
from cementic.state import DaemonState, StateManager
from cementic.supervisor import is_managed_process_alive, process_start_token


class DocumentEventHandler(FileSystemEventHandler):
    """Handles file system events for any supported document type."""

    def __init__(
        self,
        callback: Callable[[str], None],
        delete_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.callback = callback
        self.delete_callback = delete_callback
        self._timers: dict[str, Any] = {}
        self._debounce_seconds = 2.0

    def _should_process(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in supported_extensions()

    def _debounced_process(self, file_path: str) -> None:
        existing_timer = self._timers.pop(file_path, None)
        if existing_timer:
            existing_timer.cancel()

        timer = threading.Timer(self._debounce_seconds, self._run_callback, args=(file_path,))
        timer.daemon = True
        self._timers[file_path] = timer
        timer.start()

    def _run_callback(self, file_path: str) -> None:
        self._timers.pop(file_path, None)
        self.callback(file_path)

    def cancel_all(self) -> None:
        """Cancel all pending debounce timers, preventing fire after shutdown."""
        for file_path, timer in list(self._timers.items()):
            timer.cancel()
        self._timers.clear()

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        src_path = event.src_path.decode() if isinstance(event.src_path, bytes) else event.src_path
        if self._should_process(src_path):
            self._debounced_process(src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        src_path = event.src_path.decode() if isinstance(event.src_path, bytes) else event.src_path
        if self._should_process(src_path):
            self._debounced_process(src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        src_path = event.src_path.decode() if isinstance(event.src_path, bytes) else event.src_path
        timer = self._timers.pop(src_path, None)
        if timer:
            timer.cancel()
        if self.delete_callback and self._should_process(src_path):
            self.delete_callback(src_path)

    def on_moved(self, event: DirMovedEvent | FileMovedEvent) -> None:
        if event.is_directory:
            return
        src_path = event.src_path.decode() if isinstance(event.src_path, bytes) else event.src_path
        dest_path = (
            event.dest_path.decode() if isinstance(event.dest_path, bytes) else event.dest_path
        )
        if self.delete_callback and self._should_process(src_path):
            self.delete_callback(src_path)
        if self._should_process(dest_path):
            self._debounced_process(dest_path)


class SourceWatcher:
    """Background watcher that registers source documents for pipeline processing."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or get_config()
        self.state_manager = StateManager(self.config.source_watcher.state_path)
        self._shutdown_event = threading.Event()
        self._shutdown_signal: int | None = None
        self.watcher: Any = None
        self._event_handler: DocumentEventHandler | None = None
        self._logger = self._setup_logging()
        self.Session: Any = None
        self.collection = "default"
        self._watched_roots: list[Path] = []

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("cementic.source_watcher")
        logger.setLevel(logging.INFO)
        log_file = self.config.source_watcher.log_file
        if log_file is None:
            raise RuntimeError("Source watcher log file is not configured")
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        if not any(
            isinstance(handler, logging.FileHandler)
            and handler.baseFilename == str(log_file)
            for handler in logger.handlers
        ):
            log_file.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(log_file)
            handler.setLevel(logging.INFO)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.propagate = False
        return logger

    def start(self, directories: list[str], collection: str = "default") -> None:
        self.collection = collection
        state = self.state_manager.load()
        # PID + start-token: a recycled PID after `stop --force` must not block
        # a fresh start.
        if (
            state.daemon_state == DaemonState.RUNNING
            and state.pid
            and is_managed_process_alive(state.pid, state.start_token)
        ):
            self._fatal("Source watcher already running with PID %s", state.pid)
            return

        engine = get_engine(self.config.database.url)
        create_tables(engine)
        self.Session = get_session_factory(engine)

        # Counters are per-session: reset so `status` reports this run, not an
        # ever-growing total across restarts.
        self.state_manager.update(
            daemon_state=DaemonState.RUNNING,
            watched_directories=directories,
            pid=os.getpid(),
            start_token=process_start_token(os.getpid()),
            processed_count=0,
            failed_count=0,
            current_file=None,
        )

        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

        try:
            self._start_watcher(directories)
            self._logger.info(
                "Source watcher started watching: %s (collection=%s)", directories, collection
            )
            # The signal handlers set the shutdown event, so a plain wait loop
            # suffices past this point.
            while not self._shutdown_event.is_set():
                self._shutdown_event.wait(1)
        finally:
            # However this run ends, the state file must not be left claiming
            # the watcher is RUNNING.
            self.stop()

    def _fatal(self, message: str, *args: Any) -> None:
        """Log a fatal startup reason and echo it to stderr.

        The module logger writes to its own file, but `cementic start` points the
        user at the spawned process's stdout/stderr log. Without this echo the
        user is sent to a file that cannot explain why the worker exited.
        """
        self._logger.error(message, *args)
        print(message % args if args else message, file=sys.stderr, flush=True)

    def _start_watcher(self, directories: list[str]) -> None:
        observer = Observer()
        event_handler = DocumentEventHandler(self._on_file_detected, self._on_file_deleted)
        self._event_handler = event_handler
        self._watched_roots = []
        for directory in directories:
            path = Path(directory).resolve()
            if path.exists():
                self._watched_roots.append(path)
                observer.schedule(event_handler, str(path), recursive=True)
            else:
                self._logger.error("Watch directory does not exist, skipping: %s", directory)
        # Start observing *before* the initial scan: the scan hashes every
        # existing file and can take minutes, and events are only delivered
        # after start(). Registration is idempotent, so double-seeing a file
        # during the overlap is harmless.
        observer.start()
        self.watcher = observer
        for root in self._watched_roots:
            if self._shutdown_event.is_set():
                return
            self._scan_existing(root)
        if not self._shutdown_event.is_set():
            self._reconcile_deletions()

    def _reconcile_deletions(self) -> None:
        """Mark documents whose files vanished while cementic was not running.

        Deletion is otherwise only noticed through a live filesystem event, so a
        file removed between runs kept ``status="pending"`` forever and kept
        matching searches with a path that no longer exists. Scoped to the
        currently-watched roots so documents indexed from other directories (or
        other collections) are never touched.
        """
        if not self._watched_roots:
            return
        with self.Session() as session:
            documents = (
                session.query(SourceDocument)
                .filter(
                    SourceDocument.collection == self.collection,
                    SourceDocument.status != "deleted",
                )
                .all()
            )
            missing = [
                document
                for document in documents
                if self._is_under_watched_roots(document.source_path)
                and not Path(document.source_path).exists()
            ]
            # Read the paths before committing: ORM attributes expire on commit
            # and these instances are detached once the session closes.
            missing_paths = [document.source_path for document in missing]
            for document in missing:
                document.status = "deleted"
                document.file_hash = None
            if missing:
                session.commit()
        for source_path in missing_paths:
            self._logger.info(
                "Marked document deleted while stopped: %s (collection=%s)",
                source_path,
                self.collection,
            )

    def _scan_existing(self, directory: Path) -> None:
        extensions = supported_extensions()
        for file_path in directory.rglob("*"):
            # The scan can walk a large tree for minutes; without this a
            # `cementic stop` during startup waits out its whole grace period
            # and then reports a timeout, while the watcher keeps indexing.
            if self._shutdown_event.is_set():
                return
            if (
                file_path.is_file()
                and not file_path.is_symlink()
                and file_path.suffix.lower() in extensions
            ):
                self._on_file_detected(str(file_path))

    def _on_file_detected(self, file_path: str) -> None:
        try:
            self._register_document(file_path)
        except Exception as error:
            self._logger.error("Failed to register %s: %s", file_path, error)
            state = self.state_manager.load()
            self.state_manager.update(failed_count=state.failed_count + 1, current_file=None)

    def _on_file_deleted(self, file_path: str) -> None:
        try:
            self._mark_document_deleted(file_path)
        except Exception as error:
            self._logger.error("Failed to mark deleted %s: %s", file_path, error)

    def _is_under_watched_roots(self, file_path: str) -> bool:
        """Whether a stored path lies under a root this run is watching.

        Deliberately does not resolve: the path is already stored resolved, and
        the file may no longer exist.
        """
        path = Path(file_path)
        return any(path.is_relative_to(root) for root in self._watched_roots)

    def _normalize_watched_path(self, file_path: str, *, must_exist: bool) -> str | None:
        path = Path(file_path)
        try:
            resolved_path = path.resolve(strict=must_exist)
        except OSError:
            self._logger.error("Cannot resolve file: %s", file_path)
            return None
        if self._watched_roots and not any(
            resolved_path.is_relative_to(root) for root in self._watched_roots
        ):
            self._logger.error("File is outside watched roots: %s", file_path)
            return None
        return str(resolved_path)

    def _register_document(self, file_path: str) -> None:
        path = Path(file_path)
        if path.is_symlink():
            self._logger.error("Refusing symlinked file: %s", file_path)
            return
        normalized_path = self._normalize_watched_path(file_path, must_exist=True)
        if normalized_path is None:
            return
        file_path = normalized_path
        # Guard against exceedingly large files
        max_size_bytes = 512 * 1024 * 1024  # 512 MiB
        try:
            file_size = path.stat().st_size
        except OSError:
            self._logger.error("Cannot stat file: %s", file_path)
            return
        if file_size > max_size_bytes:
            self._logger.error(
                "File too large (%d bytes, max %d): %s", file_size, max_size_bytes, file_path
            )
            return

        sha256 = hashlib.sha256()
        with open(file_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                sha256.update(chunk)
        file_hash = sha256.hexdigest()

        with self.Session() as session:
            document = (
                session.query(SourceDocument)
                .filter_by(source_path=file_path, collection=self.collection)
                .first()
            )
            if document is None:
                document = SourceDocument(source_path=file_path, collection=self.collection)
                session.add(document)

            self.state_manager.update(current_file=file_path)
            document.file_hash = file_hash
            document.status = "pending"
            session.commit()

        state = self.state_manager.load()
        self.state_manager.update(processed_count=state.processed_count + 1, current_file=None)
        self._logger.info("Registered document: %s (collection=%s)", file_path, self.collection)

    def _mark_document_deleted(self, file_path: str) -> None:
        normalized_path = self._normalize_watched_path(file_path, must_exist=False)
        if normalized_path is None:
            return
        with self.Session() as session:
            document = (
                session.query(SourceDocument)
                .filter_by(source_path=normalized_path, collection=self.collection)
                .first()
            )
            if document is None:
                return
            document.status = "deleted"
            document.file_hash = None
            session.commit()
        self._logger.info(
            "Marked document deleted: %s (collection=%s)", normalized_path, self.collection
        )

    def _handle_shutdown(self, signum: int, frame: object) -> None:
        """Signal handler: set the shutdown flag and nothing else.

        Python runs handlers on the main thread between bytecodes, so anything
        that takes a lock the main thread may already hold deadlocks the process
        -- and since this *is* the SIGTERM handler, a deadlocked process can then
        only be killed with SIGKILL. `stop()` takes the state-file lock and joins
        the observer thread, so it runs from `start()`'s `finally` instead.
        """
        self._shutdown_signal = signum
        self._shutdown_event.set()

    def stop(self) -> None:
        self._shutdown_event.set()
        if self._shutdown_signal is not None:
            self._logger.info("Received signal %s, shutting down...", self._shutdown_signal)
            self._shutdown_signal = None
        if self._event_handler:
            self._event_handler.cancel_all()
        if self.watcher:
            self.watcher.stop()
            self.watcher.join()
        self.state_manager.update(daemon_state=DaemonState.STOPPED, pid=None)
        self._logger.info("Source watcher stopped")
