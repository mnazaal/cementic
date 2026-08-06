"""Pipeline worker that builds extraction, chunking, and embeddings for one revision."""

from __future__ import annotations

import logging
import os
import signal
import threading
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from sqlalchemy import and_, or_, text
from sqlalchemy.orm import Session

from cementic.chunk import chunk_text
from cementic.config import Config, get_config
from cementic.db import (
    Chunk,
    ChunkedDocument,
    ChunkEmbedding,
    ExtractedDocument,
    PipelineRevision,
    SourceDocument,
    create_tables,
    get_engine,
    get_session_factory,
)
from cementic.embedding_provider import EmbeddingProvider
from cementic.embedding_runtime import create_provider, runtime_spec_from_config
from cementic.extract import extract_document
from cementic.revisions import (
    ensure_revision_ann_index,
    ensure_revision_vector_table,
    get_target_revision,
    mark_revision_ready,
    requeue_interrupted_artifacts,
)
from cementic.state import DaemonState, StateManager
from cementic.storage import extracted_document_path, read_extracted_text, write_extracted_text
from cementic.supervisor import is_managed_process_alive, process_start_token
from cementic.vector_store import create_table_sql, upsert_vectors

PIPELINE_WORKER_LOCK_NAMESPACE = 0xC3E17C


def _pipeline_worker_lock_key(collection: str) -> int:
    """Return a stable PostgreSQL advisory lock key for a collection worker."""
    collection_key = zlib.crc32(collection.encode("utf-8"))
    return (PIPELINE_WORKER_LOCK_NAMESPACE << 32) | collection_key


def _try_acquire_pipeline_worker_lock(engine: Any, collection: str) -> Any | None:
    """Acquire a PostgreSQL session advisory lock, returning the holding connection.

    The lock is bound to the returned connection's backend session: it is held
    for as long as that connection lives, and released when the worker process
    exits. If the connection drops (e.g. Postgres restart) the lock goes with
    it, so this is a best-effort guard against two workers on one collection,
    backed by the per-run state file, not a distributed mutex.
    """
    if engine.dialect.name != "postgresql":
        return None
    connection = engine.connect()
    acquired = bool(
        connection.execute(
            text("SELECT pg_try_advisory_lock(:lock_key)"),
            {"lock_key": _pipeline_worker_lock_key(collection)},
        ).scalar()
    )
    if not acquired:
        connection.close()
        return None
    return connection


@dataclass(frozen=True)
class PipelineCounts:
    """Immutable counts for checking whether a pipeline revision is complete."""

    documents: int
    extracted_done: int
    chunked_done: int
    total_chunks: int
    done_embeddings: int
    extracted_failed: int = 0
    chunked_failed: int = 0
    failed_embeddings: int = 0


def _revision_is_complete(counts: PipelineCounts) -> bool:
    """Return True when every pipeline stage has finished for all documents."""
    if counts.extracted_done + counts.extracted_failed != counts.documents:
        return False
    if counts.chunked_done + counts.chunked_failed != counts.extracted_done:
        return False
    return counts.done_embeddings + counts.failed_embeddings == counts.total_chunks


def revision_failure_total(counts: PipelineCounts) -> int:
    """Total failed artifacts across all stages (pure)."""
    return counts.extracted_failed + counts.chunked_failed + counts.failed_embeddings


def compute_revision_counts(
    session: Session, collection: str, revision: PipelineRevision
) -> PipelineCounts:
    """Public accessor for a revision's current progress counts."""
    return _compute_revision_counts(session, collection, revision)


def _compute_revision_counts(
    session: Session, collection: str, revision: PipelineRevision
) -> PipelineCounts:
    """Query the database for current revision progress counts."""
    documents = (
        session.query(SourceDocument)
        .filter(SourceDocument.collection == collection, SourceDocument.status != "deleted")
        .count()
    )
    extracted_done = (
        session.query(ExtractedDocument)
        .join(SourceDocument, ExtractedDocument.document_id == SourceDocument.id)
        .filter(
            SourceDocument.collection == collection,
            SourceDocument.status != "deleted",
            ExtractedDocument.extractor_profile_id == revision.extractor_profile_id,
            ExtractedDocument.status == "done",
            ExtractedDocument.source_file_hash == SourceDocument.file_hash,
        )
        .count()
    )
    extracted_failed = (
        session.query(ExtractedDocument)
        .join(SourceDocument, ExtractedDocument.document_id == SourceDocument.id)
        .filter(
            SourceDocument.collection == collection,
            SourceDocument.status != "deleted",
            ExtractedDocument.extractor_profile_id == revision.extractor_profile_id,
            ExtractedDocument.status == "failed",
            ExtractedDocument.source_file_hash == SourceDocument.file_hash,
        )
        .count()
    )
    # Both chunked counts must partition exactly the set _step_chunk drains
    # (extractions with status "done", current content hash): asymmetric scoping
    # here makes `chunked_done + chunked_failed == extracted_done` unreachable
    # and wedges the revision in "building" forever.
    chunked_scope = (
        SourceDocument.collection == collection,
        SourceDocument.status != "deleted",
        ExtractedDocument.extractor_profile_id == revision.extractor_profile_id,
        ExtractedDocument.status == "done",
        ChunkedDocument.chunk_profile_id == revision.chunk_profile_id,
        ChunkedDocument.source_content_hash == ExtractedDocument.content_hash,
    )
    chunked_done = (
        session.query(ChunkedDocument)
        .join(ExtractedDocument, ChunkedDocument.extracted_document_id == ExtractedDocument.id)
        .join(SourceDocument, ExtractedDocument.document_id == SourceDocument.id)
        .filter(*chunked_scope, ChunkedDocument.status == "done")
        .count()
    )
    chunked_failed = (
        session.query(ChunkedDocument)
        .join(ExtractedDocument, ChunkedDocument.extracted_document_id == ExtractedDocument.id)
        .join(SourceDocument, ExtractedDocument.document_id == SourceDocument.id)
        .filter(*chunked_scope, ChunkedDocument.status == "failed")
        .count()
    )
    total_chunks = (
        session.query(Chunk)
        .join(ChunkedDocument, Chunk.chunked_document_id == ChunkedDocument.id)
        .join(ExtractedDocument, ChunkedDocument.extracted_document_id == ExtractedDocument.id)
        .join(SourceDocument, Chunk.document_id == SourceDocument.id)
        .filter(
            SourceDocument.collection == collection,
            SourceDocument.status != "deleted",
            ExtractedDocument.extractor_profile_id == revision.extractor_profile_id,
            ChunkedDocument.chunk_profile_id == revision.chunk_profile_id,
        )
        .count()
    )
    # Embedding counts must be scoped through the revision's chunk chain, exactly
    # like total_chunks. The same embedding profile can be shared with an older
    # revision's chunks (e.g. after a chunk_size change with the same model);
    # counting those too makes `done + failed == total_chunks` unreachable.
    embedding_scope = (
        SourceDocument.collection == collection,
        SourceDocument.status != "deleted",
        ExtractedDocument.extractor_profile_id == revision.extractor_profile_id,
        ChunkedDocument.chunk_profile_id == revision.chunk_profile_id,
        ChunkEmbedding.embedding_profile_id == revision.embedding_profile_id,
    )
    done_embeddings = (
        session.query(ChunkEmbedding)
        .join(Chunk, ChunkEmbedding.chunk_id == Chunk.id)
        .join(ChunkedDocument, Chunk.chunked_document_id == ChunkedDocument.id)
        .join(ExtractedDocument, ChunkedDocument.extracted_document_id == ExtractedDocument.id)
        .join(SourceDocument, Chunk.document_id == SourceDocument.id)
        .filter(*embedding_scope, ChunkEmbedding.status == "done")
        .count()
    )
    failed_embeddings = (
        session.query(ChunkEmbedding)
        .join(Chunk, ChunkEmbedding.chunk_id == Chunk.id)
        .join(ChunkedDocument, Chunk.chunked_document_id == ChunkedDocument.id)
        .join(ExtractedDocument, ChunkedDocument.extracted_document_id == ExtractedDocument.id)
        .join(SourceDocument, Chunk.document_id == SourceDocument.id)
        .filter(*embedding_scope, ChunkEmbedding.status == "failed")
        .count()
    )
    return PipelineCounts(
        documents=documents,
        extracted_done=extracted_done,
        extracted_failed=extracted_failed,
        chunked_done=chunked_done,
        chunked_failed=chunked_failed,
        total_chunks=total_chunks,
        done_embeddings=done_embeddings,
        failed_embeddings=failed_embeddings,
    )


class PipelineWorker:
    """Builds the target pipeline revision for one collection."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or get_config()
        self.state_manager = StateManager(self.config.pipeline_worker.state_path)
        self._shutdown_event = threading.Event()
        self._shutdown_signal: int | None = None
        self._logger = self._setup_logging()
        self.Session: Any = None
        self.embedding_client: EmbeddingProvider | None = None
        self.collection = "default"

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("cementic.pipeline")
        logger.setLevel(logging.INFO)
        log_file = self.config.pipeline_worker.log_file
        if log_file is None:
            raise RuntimeError("Pipeline worker log file is not configured")
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

    def _create_embedding_client(self) -> EmbeddingProvider:
        spec = runtime_spec_from_config(self.config)
        return create_provider(spec, self.config)

    def start(self, collection: str = "default") -> None:
        self.collection = collection
        state = self.state_manager.load()
        # PID + start-token: a recycled PID after `stop --force` must not block
        # a fresh start. The Postgres advisory lock below is the authoritative
        # mutual exclusion; this is just a fast local pre-check.
        if (
            state.daemon_state == DaemonState.RUNNING
            and state.pid
            and is_managed_process_alive(state.pid, state.start_token)
        ):
            self._logger.error("Pipeline worker already running with PID %s", state.pid)
            return

        engine = get_engine(self.config.database.url)
        create_tables(engine)
        self.Session = get_session_factory(engine)
        worker_lock = _try_acquire_pipeline_worker_lock(engine, self.collection)
        if engine.dialect.name == "postgresql" and worker_lock is None:
            self._logger.error(
                "Pipeline worker already holds DB lock for collection=%s", self.collection
            )
            return

        try:
            self.embedding_client = self._create_embedding_client()
            if not self.embedding_client.health_check():
                self._logger.error("Embedding provider health check failed")
                return
        except Exception as error:
            self._logger.error("Failed to initialize embedding provider: %s", error)
            return

        self.state_manager.update(
            daemon_state=DaemonState.RUNNING,
            pid=os.getpid(),
            start_token=process_start_token(os.getpid()),
            current_file=None,
        )
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        self._logger.info("Pipeline worker started for collection=%s", self.collection)

        # The target revision is a function of config, which is read once per
        # process, so resolve it once here and thread the id through the loop.
        revision_id = self._ensure_target_revision()

        # Re-queue any artifacts a previous run left failed or interrupted, so a
        # fresh start retries them (failures are terminal only within one run).
        with self.Session() as session:
            revision = session.get(PipelineRevision, revision_id)
            if revision is not None:
                requeue_interrupted_artifacts(session, self.collection, revision)
                session.commit()

        try:
            self._run_processing_loop(revision_id)
        finally:
            try:
                if worker_lock is not None:
                    worker_lock.close()
            finally:
                self.stop()

    def _run_processing_loop(self, revision_id: int) -> None:
        while not self._shutdown_event.is_set():
            try:
                if self._step_extract(revision_id):
                    continue
                if self._step_chunk(revision_id):
                    continue
                if self._step_embed(revision_id):
                    continue
                self._mark_revision_ready_if_complete(revision_id)
            except Exception:
                # A transient failure (Postgres restart, network blip) must not
                # kill the worker: log, back off, retry. Interrupted rows are
                # re-queued on the next pass or the next `cementic start`.
                self._logger.exception("Pipeline step failed; retrying after backoff")
                self._shutdown_event.wait(5 * self.config.pipeline_worker.poll_interval)
                continue
            self._shutdown_event.wait(self.config.pipeline_worker.poll_interval)

    def _ensure_target_revision(self) -> int:
        with self.Session() as session:
            revision = get_target_revision(
                session, self.collection, self.config, self.embedding_client
            )
            ensure_revision_vector_table(session, revision)
            session.commit()
            return revision.id

    def _step_extract(self, revision_id: int) -> bool:
        with self.Session() as session:
            revision = session.get(PipelineRevision, revision_id)
            if revision is None:
                return False

            profile_id = revision.extractor_profile_id
            candidate_row = (
                session.query(SourceDocument, ExtractedDocument)
                .outerjoin(
                    ExtractedDocument,
                    and_(
                        ExtractedDocument.document_id == SourceDocument.id,
                        ExtractedDocument.extractor_profile_id == profile_id,
                    ),
                )
                .filter(
                    SourceDocument.collection == self.collection,
                    SourceDocument.status != "deleted",
                )
                .filter(
                    or_(
                        ExtractedDocument.id.is_(None),
                        ExtractedDocument.source_file_hash != SourceDocument.file_hash,
                        ExtractedDocument.status.notin_(["done", "failed"]),
                    )
                )
                .order_by(SourceDocument.id)
                .first()
            )
            if candidate_row is None:
                return False
            document, extracted = candidate_row

            if extracted is None:
                extracted = ExtractedDocument(
                    document_id=document.id,
                    extractor_profile_id=profile_id,
                    status="processing",
                )
                session.add(extracted)
                session.flush()
            else:
                extracted.status = "processing"
                extracted.error_message = None

            session.commit()

            source_path = document.source_path
            file_hash = document.file_hash
            extracted_id = extracted.id
            artifact_path = extracted_document_path(
                self.config, self.collection, document.id, profile_id
            )

        self.state_manager.update(current_file=source_path)
        try:
            content = extract_document(source_path, self.config)
            content_hash = write_extracted_text(artifact_path, content)
            status = "done"
            error_message = None
        except Exception as error:
            content_hash = None
            status = "failed"
            error_message = str(error)

        with self.Session() as session:
            extracted = session.get(ExtractedDocument, extracted_id)
            # Deliberately no SourceDocument.status write here: per-artifact
            # status lives on ExtractedDocument, and overwriting the source row
            # would resurrect a document the watcher marked "deleted" while the
            # extraction was running.
            if extracted is not None:
                extracted.source_file_hash = file_hash
                extracted.artifact_path = (
                    str(artifact_path) if content_hash is not None else extracted.artifact_path
                )
                extracted.content_hash = content_hash
                extracted.status = status
                extracted.error_message = error_message
            session.commit()

        self.state_manager.update(current_file=None)
        return True

    def _step_chunk(self, revision_id: int) -> bool:
        with self.Session() as session:
            revision = session.get(PipelineRevision, revision_id)
            if revision is None:
                return False

            extractor_profile_id = revision.extractor_profile_id
            chunk_profile_id = revision.chunk_profile_id
            candidate_row = (
                session.query(ExtractedDocument, ChunkedDocument)
                .join(SourceDocument, ExtractedDocument.document_id == SourceDocument.id)
                .outerjoin(
                    ChunkedDocument,
                    and_(
                        ChunkedDocument.extracted_document_id == ExtractedDocument.id,
                        ChunkedDocument.chunk_profile_id == chunk_profile_id,
                    ),
                )
                .filter(
                    SourceDocument.collection == self.collection,
                    SourceDocument.status != "deleted",
                    ExtractedDocument.extractor_profile_id == extractor_profile_id,
                    ExtractedDocument.status == "done",
                )
                .filter(
                    or_(
                        ChunkedDocument.id.is_(None),
                        ChunkedDocument.source_content_hash != ExtractedDocument.content_hash,
                        ChunkedDocument.status.notin_(["done", "failed"]),
                    )
                )
                .order_by(ExtractedDocument.id)
                .first()
            )
            if candidate_row is None:
                return False
            extracted, chunked = candidate_row

            if chunked is None:
                chunked = ChunkedDocument(
                    extracted_document_id=extracted.id,
                    chunk_profile_id=chunk_profile_id,
                    status="processing",
                )
                session.add(chunked)
                session.flush()
            else:
                chunked.status = "processing"
                chunked.error_message = None

            session.commit()

            extracted_id = extracted.id
            chunked_id = chunked.id
            extracted_path = extracted.artifact_path

        if extracted_path is None:
            with self.Session() as session:
                chunked = session.get(ChunkedDocument, chunked_id)
                if chunked is not None:
                    # Record the source hash so the failure is counted by the
                    # hash-scoped revision counts (else the build never settles).
                    chunked.source_content_hash = extracted.content_hash
                    chunked.status = "failed"
                    chunked.error_message = "Missing extracted artifact path"
                    session.commit()
            return True

        self.state_manager.update(current_file=extracted_path)
        try:
            content = read_extracted_text(Path(extracted_path))
            chunk_items = chunk_text(
                content,
                chunk_size=self.config.pipeline.chunk_size,
                chunk_overlap=self.config.pipeline.chunk_overlap,
            )
            status = "done"
            error_message = None
        except Exception as error:
            chunk_items = []
            status = "failed"
            error_message = str(error)

        with self.Session() as session:
            extracted = session.get(ExtractedDocument, extracted_id)
            chunked = session.get(ChunkedDocument, chunked_id)
            if chunked is None:
                self.state_manager.update(current_file=None)
                return True

            session.query(Chunk).filter_by(chunked_document_id=chunked_id).delete(
                synchronize_session=False
            )
            if extracted is not None:
                chunked.source_content_hash = extracted.content_hash
            if status == "done" and extracted is not None:
                for item in chunk_items:
                    session.add(
                        Chunk(
                            document_id=extracted.document_id,
                            chunked_document_id=chunked_id,
                            chunk_index=item.chunk_index,
                            content=item.content,
                        )
                    )
                chunked.total_chunks = len(chunk_items)
            chunked.status = status
            chunked.error_message = error_message
            session.commit()

        self.state_manager.update(current_file=None)
        return True

    def _step_embed(self, revision_id: int) -> bool:
        with self.Session() as session:
            revision = session.get(PipelineRevision, revision_id)
            if revision is None:
                return False

            profile_id = revision.embedding_profile_id
            embedding_dim = revision.embedding_profile.embedding_dim

            candidates = (
                session.query(Chunk, ChunkEmbedding)
                .join(ChunkedDocument, Chunk.chunked_document_id == ChunkedDocument.id)
                .join(
                    ExtractedDocument, ChunkedDocument.extracted_document_id == ExtractedDocument.id
                )
                .join(SourceDocument, Chunk.document_id == SourceDocument.id)
                .outerjoin(
                    ChunkEmbedding,
                    (ChunkEmbedding.chunk_id == Chunk.id)
                    & (ChunkEmbedding.embedding_profile_id == revision.embedding_profile_id),
                )
                .filter(
                    SourceDocument.collection == self.collection,
                    SourceDocument.status != "deleted",
                    ExtractedDocument.extractor_profile_id == revision.extractor_profile_id,
                    ChunkedDocument.chunk_profile_id == revision.chunk_profile_id,
                    ChunkedDocument.status == "done",
                    # `processing` is re-picked deliberately, matching
                    # _step_extract and _step_chunk. A row is claimed in one
                    # transaction and written back in another, so a failure in
                    # between would otherwise strand it as `processing` for the
                    # life of the process -- `done + failed` could never reach
                    # `total_chunks` and the revision would never complete.
                    # One worker per collection holds the advisory lock and the
                    # vector upsert is idempotent, so re-claiming is safe.
                    (ChunkEmbedding.id.is_(None))
                    | (ChunkEmbedding.status.in_(["pending", "processing"])),
                )
                .order_by(Chunk.id)
                .limit(self.config.pipeline_worker.batch_size)
                .all()
            )

            claimed: list[tuple[int, str]] = []
            for chunk, existing in candidates:
                if existing is None:
                    existing = ChunkEmbedding(
                        chunk_id=chunk.id,
                        embedding_profile_id=revision.embedding_profile_id,
                        status="processing",
                    )
                    session.add(existing)
                    session.flush()
                    claimed.append((chunk.id, chunk.content))
                elif existing.status in ("pending", "processing"):
                    existing.status = "processing"
                    existing.error_message = None
                    claimed.append((chunk.id, chunk.content))

            if not claimed:
                session.commit()
                return False

            session.commit()

        provider = self.embedding_client
        texts = (
            [provider.format_document(content) for _, content in claimed]
            if provider is not None
            else []
        )
        # Message stamped on any row that ends up without a vector: the batch-wide
        # exception if the whole call failed, or a per-row note if the batch
        # succeeded but an individual embedding came back missing.
        embeddings: list[list[float] | None]
        try:
            raw = provider.embed_batch(texts) if provider is not None else []
            embeddings = list(raw)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as error:
            # Provider connectivity failure — a fact about the daemon, not these
            # texts. Release the claim instead of stamping terminal failures,
            # try to bring the provider back (autostart per config), and let the
            # poll interval provide the backoff.
            self._logger.error("Embedding provider unreachable: %s", error)
            self._release_claimed_embeddings([chunk_id for chunk_id, _ in claimed], profile_id)
            try:
                self.embedding_client = self._create_embedding_client()
            except Exception as restart_error:
                self._logger.error("Embedding provider restart failed: %s", restart_error)
            return False
        except Exception as error:
            embeddings = [None for _ in claimed]
            failure_message = str(error)
        else:
            if len(embeddings) != len(claimed):
                failure_message = (
                    f"Embedding provider returned {len(embeddings)} embeddings "
                    f"for {len(claimed)} chunks"
                )
                embeddings = [None for _ in claimed]
            else:
                failure_message = "Failed to generate embedding"

        successes = [
            (chunk_id, embedding)
            for (chunk_id, _), embedding in zip(claimed, embeddings)
            if embedding is not None
        ]
        # A claim is a lease: every exit path from here must either write a
        # terminal status or return the rows to `pending`. Leaving them
        # `processing` would stall the revision short of completion with no
        # error surfaced anywhere.
        try:
            with self.Session() as session:
                if successes:
                    conn = session.connection()
                    conn.execute(text(create_table_sql(profile_id, embedding_dim)))
                    upsert_vectors(conn, profile_id, successes)
                for (chunk_id, _), embedding in zip(claimed, embeddings):
                    row = (
                        session.query(ChunkEmbedding)
                        .filter_by(chunk_id=chunk_id, embedding_profile_id=profile_id)
                        .first()
                    )
                    if row is None:
                        continue
                    if embedding is None:
                        row.status = "failed"
                        row.error_message = failure_message
                    else:
                        row.status = "done"
                        row.error_message = None
                session.commit()
        except Exception:
            self._release_claimed_embeddings([chunk_id for chunk_id, _ in claimed], profile_id)
            raise
        return True

    def _release_claimed_embeddings(self, chunk_ids: list[int], profile_id: int) -> None:
        """Return a claimed-but-unembedded batch to ``pending`` (provider outage)."""
        with self.Session() as session:
            session.query(ChunkEmbedding).filter(
                ChunkEmbedding.chunk_id.in_(chunk_ids),
                ChunkEmbedding.embedding_profile_id == profile_id,
                ChunkEmbedding.status == "processing",
            ).update({"status": "pending"}, synchronize_session=False)
            session.commit()

    def _mark_revision_ready_if_complete(self, revision_id: int) -> None:
        with self.Session() as session:
            revision = session.get(PipelineRevision, revision_id)
            if revision is None or revision.status != "building":
                return
            if not self._revision_complete(session, revision):
                return
            ensure_revision_ann_index(session, revision, self.config)
            mark_revision_ready(session, revision)
            session.commit()

    def _revision_complete(self, session: Session, revision: PipelineRevision) -> bool:
        counts = _compute_revision_counts(session, self.collection, revision)
        return _revision_is_complete(counts)

    def _handle_shutdown(self, signum: int, frame: object) -> None:
        """Signal handler: set the shutdown flag and nothing else.

        Python runs handlers on the main thread between bytecodes, so anything
        that takes a lock the main thread may already hold deadlocks the process
        -- and since this *is* the SIGTERM handler, a deadlocked process can then
        only be killed with SIGKILL. `stop()` takes the state-file lock, so it
        runs from `start()`'s `finally` instead.
        """
        self._shutdown_signal = signum
        self._shutdown_event.set()

    def stop(self) -> None:
        self._shutdown_event.set()
        if self._shutdown_signal is not None:
            self._logger.info("Received signal %s, shutting down...", self._shutdown_signal)
            self._shutdown_signal = None
        self.state_manager.update(daemon_state=DaemonState.STOPPED, pid=None)
        self._logger.info("Pipeline worker stopped")


if __name__ == "__main__":
    worker = PipelineWorker()
    worker.start()
