"""File system watcher for PDF files."""

import hashlib
import time
from pathlib import Path
from typing import Callable, List, Set

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from seman.queue import JobQueue


class PDFEventHandler(FileSystemEventHandler):
    """Handles PDF file system events."""

    def __init__(
        self,
        queue: JobQueue,
        on_file_added: Callable[[str], None] = None,
    ) -> None:
        """Initialize handler with queue and callback."""
        self.queue = queue
        self.on_file_added = on_file_added
        self._pending_files: Set[str] = set()
        self._last_event_time: dict = {}
        self._debounce_seconds = 2.0  # Wait 2s after last event before processing

    def _compute_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _should_process(self, file_path: str) -> bool:
        """Check if file should be processed."""
        path = Path(file_path)

        # Only process PDFs
        if path.suffix.lower() != ".pdf":
            return False

        # Check if already in queue
        existing = self.queue.get_job_by_path(file_path)
        if existing and existing.status in ("queued", "processing"):
            return False

        return True

    def _debounced_process(self, file_path: str) -> None:
        """Process file after debounce period."""
        current_time = time.time()
        self._last_event_time[file_path] = current_time
        self._pending_files.add(file_path)

        # Wait for debounce period
        time.sleep(self._debounce_seconds)

        # Check if this is still the most recent event for this file
        if self._last_event_time.get(file_path) == current_time:
            self._pending_files.discard(file_path)
            self._process_file(file_path)

    def _process_file(self, file_path: str) -> None:
        """Process a PDF file."""
        if not self._should_process(file_path):
            return

        try:
            file_hash = self._compute_hash(file_path)
            if self.queue.add_job(file_path, file_hash=file_hash):
                if self.on_file_added:
                    self.on_file_added(file_path)
        except Exception:
            # File might not be fully written yet, will retry on next event
            pass

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events."""
        if not event.is_directory:
            self._debounced_process(event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events."""
        if not event.is_directory:
            self._debounced_process(event.src_path)


class PDFWatcher:
    """Watches directories for PDF files."""

    def __init__(
        self,
        queue: JobQueue,
        directories: List[str],
        recursive: bool = True,
    ) -> None:
        """Initialize watcher with queue and directories."""
        self.queue = queue
        self.directories = [Path(d).resolve() for d in directories]
        self.recursive = recursive
        self.observer = Observer()
        self.event_handler = PDFEventHandler(queue)
        self._running = False

    def start(self) -> None:
        """Start watching directories."""
        if self._running:
            return

        for directory in self.directories:
            if directory.exists():
                self.observer.schedule(
                    self.event_handler,
                    str(directory),
                    recursive=self.recursive,
                )

                # Add existing PDFs to queue
                self._scan_existing(directory)

        self.observer.start()
        self._running = True

    def _scan_existing(self, directory: Path) -> None:
        """Scan directory for existing PDFs."""
        pattern = "**/*.pdf" if self.recursive else "*.pdf"
        for pdf_file in directory.glob(pattern):
            if pdf_file.is_file():
                try:
                    sha256 = hashlib.sha256()
                    with open(pdf_file, "rb") as f:
                        for chunk in iter(lambda: f.read(8192), b""):
                            sha256.update(chunk)
                    file_hash = sha256.hexdigest()
                    self.queue.add_job(str(pdf_file), file_hash=file_hash)
                except Exception:
                    pass

    def stop(self) -> None:
        """Stop watching."""
        if not self._running:
            return

        self.observer.stop()
        self.observer.join()
        self._running = False

    def is_running(self) -> bool:
        """Check if watcher is running."""
        return self._running
