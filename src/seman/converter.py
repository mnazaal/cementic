"""Converter daemon that watches PDFs and converts to chunks."""

import hashlib
import logging
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any, List, Optional

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from seman.chunk import chunk_text
from seman.config import Config, get_config
from seman.convert import convert_pdf_to_markdown
from seman.db import Chunk, Document, create_tables, get_engine, get_session_factory
from seman.state import DaemonState, StateManager


class PDFEventHandler(FileSystemEventHandler):
    """Handles PDF file system events."""

    def __init__(self, callback) -> None:
        """Initialize handler with callback."""
        self.callback = callback
        self._timers: dict[str, Any] = {}
        self._debounce_seconds = 2.0

    def _should_process(self, file_path: str) -> bool:
        """Check if file should be processed."""
        path = Path(file_path)
        if path.suffix.lower() != ".pdf":
            return False
        return True

    def _debounced_process(self, file_path: str) -> None:
        """Process file after debounce period."""
        existing_timer = self._timers.pop(file_path, None)
        if existing_timer:
            existing_timer.cancel()

        timer = threading.Timer(self._debounce_seconds, self._run_callback, args=(file_path,))
        timer.daemon = True
        self._timers[file_path] = timer
        timer.start()

    def _run_callback(self, file_path: str) -> None:
        """Run callback for a debounced file path."""
        self._timers.pop(file_path, None)
        self.callback(file_path)

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events."""
        if event.is_directory:
            return

        src_path = event.src_path.decode() if isinstance(event.src_path, bytes) else event.src_path
        if self._should_process(src_path):
            self._debounced_process(src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events."""
        if event.is_directory:
            return

        src_path = event.src_path.decode() if isinstance(event.src_path, bytes) else event.src_path
        if self._should_process(src_path):
            self._debounced_process(src_path)


class ConverterDaemon:
    """Daemon that watches for PDFs and converts them to chunks."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialize converter daemon."""
        self.config = config or get_config()
        self.state_manager = StateManager(self.config.converter.state_path)
        self._shutdown_event = threading.Event()
        self.watcher: Optional[Observer] = None
        self._logger = self._setup_logging()
        self.Session = None
        self.collection = "default"

    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger("seman.converter")
        logger.setLevel(logging.INFO)

        handler = logging.FileHandler(self.config.converter.log_file)
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        return logger

    def start(self, directories: List[str], collection: str = "default") -> None:
        """Start the converter daemon."""
        self.collection = collection
        # Check if already running
        state = self.state_manager.load()
        if state.daemon_state == DaemonState.RUNNING and state.pid:
            try:
                os.kill(state.pid, 0)
                self._logger.error(f"Converter already running with PID {state.pid}")
                return
            except (OSError, ProcessLookupError):
                pass

        # Initialize database
        engine = get_engine(self.config.database.url)
        create_tables(engine)
        self.Session = get_session_factory(engine)

        # Update state
        self.state_manager.update(
            daemon_state=DaemonState.RUNNING,
            watched_directories=directories,
            pid=os.getpid(),
        )

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

        # Start file watcher
        self._start_watcher(directories)

        self._logger.info(
            f"Converter daemon started watching: {directories} (collection={collection})"
        )

        # Wait for shutdown
        try:
            while not self._shutdown_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def _start_watcher(self, directories: List[str]) -> None:
        """Start watching directories."""
        self.watcher = Observer()
        event_handler = PDFEventHandler(self._on_pdf_detected)

        for directory in directories:
            path = Path(directory).resolve()
            if path.exists():
                self.watcher.schedule(event_handler, str(path), recursive=True)
                self._scan_existing(path)

        self.watcher.start()

    def _scan_existing(self, directory: Path) -> None:
        """Scan directory for existing PDFs."""
        for pdf_file in directory.rglob("*.pdf"):
            if pdf_file.is_file():
                self._on_pdf_detected(str(pdf_file))

    def _on_pdf_detected(self, pdf_path: str) -> None:
        """Handle detected PDF file."""
        try:
            self._process_pdf(pdf_path)
        except Exception as e:
            self._logger.error(f"Failed to process {pdf_path}: {e}")

    def _process_pdf(self, pdf_path: str) -> None:
        """Process a single PDF file."""
        # Compute file hash
        sha256 = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        file_hash = sha256.hexdigest()

        with self.Session() as session:
            # Check if already processed
            existing = (
                session.query(Document)
                .filter_by(
                    source_path=pdf_path,
                    collection=self.collection,
                )
                .first()
            )
            if existing:
                if existing.file_hash == file_hash:
                    chunk_count = session.query(Chunk).filter_by(document_id=existing.id).count()
                    expected_chunks = existing.total_chunks or 0
                    is_complete = (
                        existing.status in {"converted", "completed"}
                        and expected_chunks > 0
                        and chunk_count == expected_chunks
                    )
                    if is_complete:
                        self._logger.debug(f"Skipping unchanged file: {pdf_path}")
                        return

                    self._logger.info(
                        "Reprocessing incomplete unchanged file: %s "
                        "(status=%s, chunks=%s, expected=%s)",
                        pdf_path,
                        existing.status,
                        chunk_count,
                        expected_chunks,
                    )

                # File changed or incomplete previous attempt: rebuild chunks
                session.query(Chunk).filter_by(document_id=existing.id).delete()
                document = existing
            else:
                document = Document(source_path=pdf_path, collection=self.collection)
                session.add(document)
                session.flush()

            # Update document metadata
            document.file_hash = file_hash
            document.status = "processing"
            document.total_chunks = None
            document.error_message = None
            session.commit()

            self.state_manager.update(current_file=pdf_path)

            try:
                # Convert PDF to markdown
                markdown_content = convert_pdf_to_markdown(pdf_path)

                # Chunk the text
                chunks = chunk_text(
                    markdown_content,
                    chunk_size=self.config.indexing.chunk_size,
                    chunk_overlap=self.config.indexing.chunk_overlap,
                )

                # Create chunk records
                for chunk_data in chunks:
                    chunk = Chunk(
                        document_id=document.id,
                        chunk_index=chunk_data.chunk_index,
                        content=chunk_data.content,
                        embedding_status="pending",
                        page_start=chunk_data.page_start,
                        page_end=chunk_data.page_end,
                    )
                    session.add(chunk)

                document.status = "converted"
                document.total_chunks = len(chunks)
                session.commit()

                state = self.state_manager.load()
                self.state_manager.update(processed_count=state.processed_count + 1)

                self._logger.info(
                    f"Converted: {pdf_path} ({len(chunks)} chunks, collection={self.collection})"
                )

            except Exception as e:
                document.status = "failed"
                document.error_message = str(e)
                session.commit()

                state = self.state_manager.load()
                self.state_manager.update(failed_count=state.failed_count + 1)

                raise

            finally:
                self.state_manager.update(current_file=None)

    def _handle_shutdown(self, signum, frame) -> None:
        """Handle shutdown signals."""
        self._logger.info(f"Received signal {signum}, shutting down...")
        self.stop()

    def stop(self) -> None:
        """Stop the daemon gracefully."""
        self._shutdown_event.set()

        if self.watcher:
            self.watcher.stop()
            self.watcher.join()

        self.state_manager.update(daemon_state=DaemonState.STOPPED, pid=None)
        self._logger.info("Converter daemon stopped")
