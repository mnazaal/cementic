"""Main daemon that orchestrates PDF indexing."""

import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

from seman.chunk import TextChunk, chunk_text
from seman.config import Config, get_config
from seman.convert import convert_pdf_to_markdown
from seman.db import Chunk, Document, create_tables, get_engine, get_session_factory
from seman.queue import JobQueue
from seman.state import DaemonState, IndexingState, StateManager
from seman.vectorize import Vectorizer
from seman.watcher import PDFWatcher


class IndexingDaemon:
    """Daemon that watches for PDFs and indexes them."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialize daemon with configuration."""
        self.config = config or get_config()
        self.state_manager = StateManager(self.config.indexing.state_path)
        self.queue = JobQueue(self.config.indexing.queue_db_path)
        self.watcher: Optional[PDFWatcher] = None
        self._shutdown_event = threading.Event()
        self._pause_event = threading.Event()
        self._processor_thread: Optional[threading.Thread] = None
        self._logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger("seman")
        logger.setLevel(logging.INFO)

        handler = logging.FileHandler(self.config.daemon.log_file)
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        return logger

    def start(self, directories: List[str]) -> None:
        """Start the daemon."""
        # Check if already running
        state = self.state_manager.load()
        if state.daemon_state == DaemonState.RUNNING and state.pid:
            try:
                import os

                os.kill(state.pid, 0)  # Check if process exists
                self._logger.error(f"Daemon already running with PID {state.pid}")
                return
            except (OSError, ProcessLookupError):
                pass  # Process doesn't exist, continue

        # Reset any stuck processing jobs
        reset_count = self.queue.reset_processing_jobs()
        if reset_count > 0:
            self._logger.info(f"Reset {reset_count} stuck processing jobs")

        # Update state
        self.state_manager.update(
            daemon_state=DaemonState.RUNNING,
            watched_directories=directories,
            pid=Path("/proc/self").stat().st_ino if sys.platform != "win32" else None,
        )

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

        # Initialize database
        engine = get_engine(self.config.database.url)
        create_tables(engine)
        self.Session = get_session_factory(engine)

        # Initialize vectorizer
        self.vectorizer = Vectorizer(self.config.ollama)

        # Start file watcher
        self.watcher = PDFWatcher(
            self.queue,
            directories,
        )
        self.watcher.start()
        self._logger.info(f"Started watching directories: {directories}")

        # Start processor thread
        self._processor_thread = threading.Thread(target=self._process_loop)
        self._processor_thread.start()

        self._logger.info("Daemon started successfully")

        # Wait for shutdown
        try:
            while not self._shutdown_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def _handle_shutdown(self, signum, frame) -> None:
        """Handle shutdown signals."""
        self._logger.info(f"Received signal {signum}, shutting down...")
        self.stop()

    def stop(self) -> None:
        """Stop the daemon gracefully."""
        self._shutdown_event.set()
        self._pause_event.set()  # Unpause if paused

        if self.watcher:
            self.watcher.stop()

        if self._processor_thread:
            self._processor_thread.join(timeout=5)

        # Reset any processing jobs
        self.queue.reset_processing_jobs()

        # Update state
        self.state_manager.update(daemon_state=DaemonState.STOPPED, pid=None)

        self._logger.info("Daemon stopped")

    def pause(self) -> None:
        """Pause processing but keep watching."""
        self._pause_event.set()
        self.state_manager.update(daemon_state=DaemonState.PAUSED)
        self._logger.info("Indexing paused")

    def resume(self) -> None:
        """Resume processing."""
        self._pause_event.clear()
        self.state_manager.update(daemon_state=DaemonState.RUNNING)
        self._logger.info("Indexing resumed")

    def _process_loop(self) -> None:
        """Main processing loop."""
        while not self._shutdown_event.is_set():
            # Check if paused
            if self._pause_event.is_set():
                time.sleep(1)
                continue

            # Get next job from queue
            job = self.queue.get_next_job()

            if job is None:
                time.sleep(1)
                continue

            try:
                self.state_manager.update(current_file=job.source_path)
                self._process_file(job.source_path)
                self.queue.mark_completed(job.id)

                # Update stats
                state = self.state_manager.load()
                self.state_manager.update(processed_count=state.processed_count + 1)

            except Exception as e:
                self._logger.error(f"Failed to process {job.source_path}: {e}")
                self.queue.mark_failed(job.id, str(e))

                state = self.state_manager.load()
                self.state_manager.update(failed_count=state.failed_count + 1)

            finally:
                self.state_manager.update(current_file=None)

    def _process_file(self, pdf_path: str) -> None:
        """Process a single PDF file."""
        self._logger.info(f"Processing: {pdf_path}")

        # Convert PDF to markdown
        markdown_content = convert_pdf_to_markdown(pdf_path)

        # Chunk the text
        chunks = chunk_text(
            markdown_content,
            chunk_size=self.config.indexing.chunk_size,
            chunk_overlap=self.config.indexing.chunk_overlap,
        )

        # Generate embeddings
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = self.vectorizer.embed_batch(chunk_texts)

        # Store in database
        with self.Session() as session:
            # Create or update document record
            document = session.query(Document).filter_by(source_path=pdf_path).first()

            if document is None:
                document = Document(
                    source_path=pdf_path,
                    status="completed",
                    markdown_content=markdown_content,
                    total_chunks=len(chunks),
                )
                session.add(document)
            else:
                document.status = "completed"
                document.markdown_content = markdown_content
                document.total_chunks = len(chunks)
                # Delete old chunks
                session.query(Chunk).filter_by(document_id=document.id).delete()

            session.flush()

            # Create chunk records with embeddings
            for chunk_data, embedding in zip(chunks, embeddings):
                if embedding is not None:
                    chunk = Chunk(
                        document_id=document.id,
                        chunk_index=chunk_data.chunk_index,
                        content=chunk_data.content,
                        embedding=embedding,
                        page_start=chunk_data.page_start,
                        page_end=chunk_data.page_end,
                    )
                    session.add(chunk)

            session.commit()

        self._logger.info(f"Successfully indexed: {pdf_path} ({len(chunks)} chunks)")
