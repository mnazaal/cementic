"""Document watcher that registers PDFs for downstream pipeline processing."""

from __future__ import annotations

import hashlib
import logging
import os
import signal
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from cementic.config import Config, get_config
from cementic.db import SourceDocument, create_tables, get_engine, get_session_factory
from cementic.state import DaemonState, StateManager


class PDFEventHandler(FileSystemEventHandler):
    """Handles PDF file system events."""

    def __init__(self, callback: Callable[[str], None]) -> None:
        self.callback = callback
        self._timers: dict[str, Any] = {}
        self._debounce_seconds = 2.0

    def _should_process(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() == ".pdf"

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


class SourceWatcher:
    """Background watcher that registers source PDFs for pipeline processing."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or get_config()
        self.state_manager = StateManager(self.config.source_watcher.state_path)
        self._shutdown_event = threading.Event()
        self.watcher: Any = None
        self._logger = self._setup_logging()
        self.Session: Any = None
        self.collection = "default"

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("cementic.source_watcher")
        logger.setLevel(logging.INFO)
        log_file = self.config.source_watcher.log_file
        if log_file is None:
            raise RuntimeError("Source watcher log file is not configured")
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def start(self, directories: list[str], collection: str = "default") -> None:
        self.collection = collection
        state = self.state_manager.load()
        if state.daemon_state == DaemonState.RUNNING and state.pid:
            try:
                os.kill(state.pid, 0)
                self._logger.error("Source watcher already running with PID %s", state.pid)
                return
            except (OSError, ProcessLookupError):
                pass

        engine = get_engine(self.config.database.url)
        create_tables(engine)
        self.Session = get_session_factory(engine)

        self.state_manager.update(
            daemon_state=DaemonState.RUNNING,
            watched_directories=directories,
            pid=os.getpid(),
        )

        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

        self._start_watcher(directories)
        self._logger.info(
            "Source watcher started watching: %s (collection=%s)", directories, collection
        )

        try:
            while not self._shutdown_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def _start_watcher(self, directories: list[str]) -> None:
        observer = Observer()
        event_handler = PDFEventHandler(self._on_pdf_detected)
        for directory in directories:
            path = Path(directory).resolve()
            if path.exists():
                observer.schedule(event_handler, str(path), recursive=True)
                self._scan_existing(path)
        observer.start()
        self.watcher = observer

    def _scan_existing(self, directory: Path) -> None:
        for pdf_file in directory.rglob("*.pdf"):
            if pdf_file.is_file():
                self._on_pdf_detected(str(pdf_file))

    def _on_pdf_detected(self, pdf_path: str) -> None:
        try:
            self._register_pdf(pdf_path)
        except Exception as error:
            self._logger.error("Failed to register %s: %s", pdf_path, error)

    def _register_pdf(self, pdf_path: str) -> None:
        sha256 = hashlib.sha256()
        with open(pdf_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                sha256.update(chunk)
        file_hash = sha256.hexdigest()

        with self.Session() as session:
            document = (
                session.query(SourceDocument)
                .filter_by(source_path=pdf_path, collection=self.collection)
                .first()
            )
            if document is None:
                document = SourceDocument(source_path=pdf_path, collection=self.collection)
                session.add(document)

            self.state_manager.update(current_file=pdf_path)
            document.file_hash = file_hash
            document.status = "pending"
            document.error_message = None
            session.commit()

        state = self.state_manager.load()
        self.state_manager.update(processed_count=state.processed_count + 1, current_file=None)
        self._logger.info("Registered PDF: %s (collection=%s)", pdf_path, self.collection)

    def _handle_shutdown(self, signum: int, frame: object) -> None:
        self._logger.info("Received signal %s, shutting down...", signum)
        self.stop()

    def stop(self) -> None:
        self._shutdown_event.set()
        if self.watcher:
            self.watcher.stop()
            self.watcher.join()
        self.state_manager.update(daemon_state=DaemonState.STOPPED, pid=None)
        self._logger.info("Source watcher stopped")
