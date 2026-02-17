"""Embedder daemon that generates embeddings for chunks."""

import logging
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

from sqlalchemy import text

from seman.config import Config, get_config
from seman.db import Chunk, Document, get_engine, get_session_factory
from seman.embedders import get_embedder
from seman.embedders.base import Embedder
from seman.embedding_text import format_document_text
from seman.state import DaemonState, StateManager


@dataclass
class ClaimedChunk:
    """Chunk payload claimed for embedding processing."""

    id: int
    document_id: int
    content: str


class EmbedderDaemon:
    """Daemon that generates embeddings for pending chunks."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialize embedder daemon."""
        self.config = config or get_config()
        self.state_manager = StateManager(self.config.embedder.state_path)
        self._shutdown_event = threading.Event()
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
        self._recover_stale_processing_chunks()

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

    def _recover_stale_processing_chunks(self) -> None:
        """Reset stale processing chunks so they can be retried."""
        stale_before = datetime.now() - timedelta(
            seconds=self.config.embedder.processing_stale_seconds
        )
        with self.Session() as session:
            session.query(Chunk).filter(
                Chunk.embedding_status == "processing",
                Chunk.updated_at < stale_before,
            ).update(
                {
                    "embedding_status": "pending",
                    "error_message": "Recovered from stale processing state",
                },
                synchronize_session=False,
            )
            session.commit()

    def _get_pending_batch(self) -> Optional[List[ClaimedChunk]]:
        """Get a batch of pending chunks from the database."""
        with self.Session() as session:
            result = session.execute(
                text(
                    "WITH claimed AS ("
                    "  SELECT id FROM chunks "
                    "  WHERE embedding_status = 'pending' "
                    "  ORDER BY id "
                    "  FOR UPDATE SKIP LOCKED "
                    "  LIMIT :batch_size"
                    ") "
                    "UPDATE chunks "
                    "SET embedding_status = 'processing', updated_at = NOW() "
                    "WHERE id IN (SELECT id FROM claimed) "
                    "RETURNING id, document_id, content"
                ),
                {"batch_size": self.config.embedder.batch_size},
            )
            rows = result.fetchall()
            session.commit()

            if not rows:
                return None

            return [
                ClaimedChunk(id=row.id, document_id=row.document_id, content=row.content)
                for row in rows
            ]

    def _process_batch(self, chunks: List[ClaimedChunk]) -> None:
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
            self._update_document_status({chunk.document_id for chunk in chunks})

        except Exception as e:
            self._logger.error(f"Failed to embed batch: {e}")
            raise

    def _mark_batch_failed(self, chunks: List[ClaimedChunk], error: str) -> None:
        """Mark a batch of chunks as failed."""
        with self.Session() as session:
            for chunk in chunks:
                session.query(Chunk).filter_by(id=chunk.id).update(
                    {"embedding_status": "failed", "error_message": error}
                )
            session.commit()

    def _update_document_status(self, document_ids: set[int]) -> None:
        """Update document status if all chunks are embedded."""
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


if __name__ == "__main__":
    daemon = EmbedderDaemon()
    daemon.start()
