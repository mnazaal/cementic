"""Embedder daemon that generates embeddings for chunks."""

import logging
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from seman.config import Config, get_config
from seman.db import Chunk, Document, get_engine, get_session_factory
from seman.embedders import get_embedder
from seman.embedders.base import Embedder
from seman.embedding_text import format_document_text
from seman.state import DaemonState, StateManager


class EmbedderDaemon:
    """Daemon that generates embeddings for pending chunks."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialize embedder daemon."""
        self.config = config or get_config()
        self.state_manager = StateManager(self.config.indexing.state_path)
        self._shutdown_event = threading.Event()
        self._pause_event = threading.Event()
        self._logger = self._setup_logging()
        self.Session = None
        self.embedder: Optional[Embedder] = None

    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger("seman.embedder")
        logger.setLevel(logging.INFO)

        handler = logging.FileHandler(self.config.embedder.log_file)
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        return logger

    def _init_embedder(self) -> Embedder:
        """Initialize the embedder based on config."""
        embedder_type = self.config.indexing.embedder

        if embedder_type == "llama-cpp":
            return get_embedder(
                "llama-cpp",
                model_path=self.config.llama_cpp.model_path,
                n_ctx=self.config.llama_cpp.n_ctx,
                n_gpu_layers=self.config.llama_cpp.n_gpu_layers,
                embedding_dim=self.config.llama_cpp.embedding_dim,
                verbose=self.config.llama_cpp.verbose,
            )
        elif embedder_type == "ollama":
            return get_embedder(
                "ollama",
                host=self.config.ollama.host,
                model=self.config.ollama.model,
                embedding_dim=self.config.ollama.embedding_dim,
            )
        else:
            raise ValueError(f"Unknown embedder type: {embedder_type}")

    def start(self) -> None:
        """Start the embedder daemon."""
        # Check if already running
        state = self.state_manager.load()
        if state.daemon_state == DaemonState.RUNNING and state.pid:
            try:
                os.kill(state.pid, 0)
                self._logger.error(f"Embedder already running with PID {state.pid}")
                return
            except (OSError, ProcessLookupError):
                pass

        # Initialize database
        engine = get_engine(self.config.database.url)
        self.Session = get_session_factory(engine)

        # Initialize embedder
        try:
            self.embedder = self._init_embedder()
            if not self.embedder.health_check():
                self._logger.error("Embedder health check failed")
                return
        except Exception as e:
            self._logger.error(f"Failed to initialize embedder: {e}")
            return

        # Update state
        self.state_manager.update(
            daemon_state=DaemonState.RUNNING,
            pid=os.getpid(),
        )

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

        self._logger.info(
            f"Embedder daemon started with {self.config.embedder.max_workers} workers"
        )

        # Start processing loop with thread pool
        self._run_processing_loop()

    def _run_processing_loop(self) -> None:
        """Main processing loop with worker pool."""
        with ThreadPoolExecutor(max_workers=self.config.embedder.max_workers) as executor:
            pending_futures = {}

            while not self._shutdown_event.is_set():
                # Check if paused
                if self._pause_event.is_set():
                    time.sleep(0.5)
                    continue

                # Submit new batches if we have capacity
                while len(pending_futures) < self.config.embedder.max_workers:
                    batch = self._get_pending_batch()
                    if not batch:
                        break

                    future = executor.submit(self._process_batch, batch)
                    pending_futures[future] = batch

                # Check completed futures
                done_futures = [f for f in pending_futures if f.done()]
                for future in done_futures:
                    batch = pending_futures.pop(future)
                    try:
                        future.result()
                    except Exception as e:
                        self._logger.error(f"Batch processing failed: {e}")
                        # Mark chunks as failed
                        self._mark_batch_failed(batch, str(e))

                if not pending_futures:
                    # No work to do, sleep before polling again
                    time.sleep(self.config.embedder.poll_interval)

        # Clean up any remaining futures
        for future in pending_futures:
            future.cancel()

    def _get_pending_batch(self) -> Optional[List[Chunk]]:
        """Get a batch of pending chunks from the database."""
        with self.Session() as session:
            chunks = (
                session.query(Chunk)
                .filter_by(embedding_status="pending")
                .limit(self.config.embedder.batch_size)
                .all()
            )

            if not chunks:
                return None

            # Mark as processing
            chunk_ids = [c.id for c in chunks]
            session.query(Chunk).filter(Chunk.id.in_(chunk_ids)).update(
                {"embedding_status": "processing"}
            )
            session.commit()

            return chunks

    def _process_batch(self, chunks: List[Chunk]) -> None:
        """Process a batch of chunks."""
        if not chunks:
            return

        texts = [format_document_text(chunk.content, self.config) for chunk in chunks]

        try:
            embeddings = self.embedder.embed_batch(texts)

            with self.Session() as session:
                for chunk, embedding in zip(chunks, embeddings):
                    if embedding is not None:
                        session.query(Chunk).filter_by(id=chunk.id).update(
                            {"embedding": embedding, "embedding_status": "done"}
                        )
                    else:
                        session.query(Chunk).filter_by(id=chunk.id).update(
                            {
                                "embedding_status": "failed",
                                "error_message": "Failed to generate embedding",
                            }
                        )

                session.commit()

            self._logger.info(f"Embedded batch of {len(chunks)} chunks")

            # Update document status if all chunks are done
            self._update_document_status(chunks)

        except Exception as e:
            self._logger.error(f"Failed to embed batch: {e}")
            raise

    def _mark_batch_failed(self, chunks: List[Chunk], error: str) -> None:
        """Mark a batch of chunks as failed."""
        with self.Session() as session:
            for chunk in chunks:
                session.query(Chunk).filter_by(id=chunk.id).update(
                    {"embedding_status": "failed", "error_message": error}
                )
            session.commit()

    def _update_document_status(self, chunks: List[Chunk]) -> None:
        """Update document status if all chunks are embedded."""
        document_ids = {chunk.document_id for chunk in chunks}

        with self.Session() as session:
            for doc_id in document_ids:
                total_chunks = session.query(Chunk).filter_by(document_id=doc_id).count()
                done_chunks = (
                    session.query(Chunk)
                    .filter_by(document_id=doc_id, embedding_status="done")
                    .count()
                )

                if total_chunks == done_chunks:
                    session.query(Document).filter_by(id=doc_id).update({"status": "completed"})
                    session.commit()

                    doc = session.query(Document).filter_by(id=doc_id).first()
                    if doc:
                        self._logger.info(f"Document completed: {doc.source_path}")

    def _handle_shutdown(self, signum, frame) -> None:
        """Handle shutdown signals."""
        self._logger.info(f"Received signal {signum}, shutting down...")
        self.stop()

    def stop(self) -> None:
        """Stop the daemon gracefully."""
        self._shutdown_event.set()
        self.state_manager.update(daemon_state=DaemonState.STOPPED, pid=None)
        self._logger.info("Embedder daemon stopped")

    def pause(self) -> None:
        """Pause processing."""
        self._pause_event.set()
        self.state_manager.update(daemon_state=DaemonState.PAUSED)
        self._logger.info("Embedder paused")

    def resume(self) -> None:
        """Resume processing."""
        self._pause_event.clear()
        self.state_manager.update(daemon_state=DaemonState.RUNNING)
        self._logger.info("Embedder resumed")


if __name__ == "__main__":
    daemon = EmbedderDaemon()
    daemon.start()
