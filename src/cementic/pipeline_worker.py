"""Pipeline worker that builds extraction, chunking, and embeddings for one revision."""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn

import requests
from sqlalchemy import and_, or_, select, text
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
from cementic.index_strategies import index_dimension_error
from cementic.revisions import (
    chunk_scope,
    chunked_scope,
    embedding_scope,
    ensure_revision_ann_index,
    ensure_revision_ann_index_up_front,
    ensure_revision_vector_table,
    extracted_scope,
    get_target_revision,
    mark_revision_ready,
    requeue_interrupted_artifacts,
)
from cementic.state import DaemonState, StateManager
from cementic.storage import extracted_document_path, read_extracted_text, write_extracted_text
from cementic.supervisor import is_managed_process_alive, process_start_token
from cementic.vector_store import ensure_vector_table_schema, upsert_vectors

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


def _release_pipeline_worker_lock(connection: Any | None, collection: str) -> None:
    """Release the collection's advisory lock, then return the connection.

    ``Connection.close()`` only returns the connection to the pool; a
    ``pg_advisory_lock`` is bound to the backend session and survives that. The
    lock did go away when the process exited, so this was harmless in practice
    -- but the ``finally`` block read as if it released, and because the engine
    is cached process-wide, a second in-process ``start()`` could get the same
    pooled backend and see its own lock as someone else's.
    """
    if connection is None:
        return
    try:
        connection.execute(
            text("SELECT pg_advisory_unlock(:lock_key)"),
            {"lock_key": _pipeline_worker_lock_key(collection)},
        )
        connection.commit()
    except Exception:  # pragma: no cover - the connection may already be dead
        pass
    finally:
        connection.close()


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


#: HTTP statuses that describe the server's condition rather than the request:
#: overload, rate limiting, a model reload, an OOM kill. Retrying the same texts
#: later is expected to succeed.
_RETRYABLE_HTTP_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


def is_retryable_embed_error(error: BaseException) -> bool:
    """Whether an embedding failure is about the provider, not the texts (pure).

    Only ConnectionError and Timeout used to count as retryable, but
    ``raise_for_status()`` raises ``HTTPError`` -- so a momentary 503 from
    llama.cpp (reloading, out of memory, overloaded) stamped every chunk in the
    batch as permanently ``failed``. Those failures then block
    ``collection promote``, and forcing past them silently omits the chunks from
    the published index. A transient server condition must not be recorded as a
    property of the documents.
    """
    if isinstance(error, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return True
    if isinstance(error, requests.exceptions.ChunkedEncodingError):
        return True
    if isinstance(error, requests.exceptions.HTTPError):
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
        return status in _RETRYABLE_HTTP_STATUSES
    return False


def revision_is_complete(counts: PipelineCounts) -> bool:
    """Return True when every pipeline stage has finished for all documents."""
    if counts.documents == 0:
        # All-zero counts satisfy every equality below, so without this a
        # revision reaches `ready` before the watcher has registered its first
        # document. On a fresh `cementic start` that is the normal race rather
        # than a rare one: both workers spawn together and the pipeline
        # worker's first pass usually wins. Promoting the result publishes an
        # empty index while reporting zero failures.
        return False
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


def _purge_superseded_chunks(
    session: Session, extracted_document_id: int, content_hash: str
) -> None:
    """Drop chunks produced from an older version of a re-extracted document.

    The file changed on disk, so those chunks describe text that is no longer
    there. ``_step_chunk`` would delete them anyway when it re-chunks, but the
    gap between the two steps is not bounded: if the worker stops, or chunking
    fails, they persist. That used to be covered by search's freshness join,
    which is exactly the filter that had to move off a joined table for the ANN
    index to be usable -- so the stale rows are now removed at the moment they
    become stale rather than filtered out later.

    Deleting the chunks also removes their embeddings and vectors through the
    ``ON DELETE CASCADE`` on ``chunks_v2``. The visible consequence is that a
    document whose re-extraction succeeded but whose re-chunking has not run
    yet returns nothing rather than its previous contents.
    """
    stale_chunked_ids = select(ChunkedDocument.id).where(
        ChunkedDocument.extracted_document_id == extracted_document_id,
        or_(
            ChunkedDocument.source_content_hash.is_(None),
            ChunkedDocument.source_content_hash != content_hash,
        ),
    )
    session.query(Chunk).filter(Chunk.chunked_document_id.in_(stale_chunked_ids)).delete(
        synchronize_session=False
    )


def _compute_revision_counts(
    session: Session, collection: str, revision: PipelineRevision
) -> PipelineCounts:
    """Query the database for current revision progress counts."""
    live = (SourceDocument.collection == collection, SourceDocument.status != "deleted")
    documents = session.query(SourceDocument).filter(*live).count()
    extracted_done = (
        session.query(ExtractedDocument)
        .join(SourceDocument, ExtractedDocument.document_id == SourceDocument.id)
        .filter(*live, *extracted_scope(revision), ExtractedDocument.status == "done")
        .count()
    )
    extracted_failed = (
        session.query(ExtractedDocument)
        .join(SourceDocument, ExtractedDocument.document_id == SourceDocument.id)
        .filter(*live, *extracted_scope(revision), ExtractedDocument.status == "failed")
        .count()
    )
    chunked_done = (
        session.query(ChunkedDocument)
        .join(ExtractedDocument, ChunkedDocument.extracted_document_id == ExtractedDocument.id)
        .join(SourceDocument, ExtractedDocument.document_id == SourceDocument.id)
        .filter(*live, *chunked_scope(revision), ChunkedDocument.status == "done")
        .count()
    )
    chunked_failed = (
        session.query(ChunkedDocument)
        .join(ExtractedDocument, ChunkedDocument.extracted_document_id == ExtractedDocument.id)
        .join(SourceDocument, ExtractedDocument.document_id == SourceDocument.id)
        .filter(*live, *chunked_scope(revision), ChunkedDocument.status == "failed")
        .count()
    )
    total_chunks = (
        session.query(Chunk)
        .join(ChunkedDocument, Chunk.chunked_document_id == ChunkedDocument.id)
        .join(ExtractedDocument, ChunkedDocument.extracted_document_id == ExtractedDocument.id)
        .join(SourceDocument, Chunk.document_id == SourceDocument.id)
        .filter(*live, *chunk_scope(revision))
        .count()
    )
    embedding_conditions = (*live, *embedding_scope(revision))
    done_embeddings = (
        session.query(ChunkEmbedding)
        .join(Chunk, ChunkEmbedding.chunk_id == Chunk.id)
        .join(ChunkedDocument, Chunk.chunked_document_id == ChunkedDocument.id)
        .join(ExtractedDocument, ChunkedDocument.extracted_document_id == ExtractedDocument.id)
        .join(SourceDocument, Chunk.document_id == SourceDocument.id)
        .filter(*embedding_conditions, ChunkEmbedding.status == "done")
        .count()
    )
    failed_embeddings = (
        session.query(ChunkEmbedding)
        .join(Chunk, ChunkEmbedding.chunk_id == Chunk.id)
        .join(ChunkedDocument, Chunk.chunked_document_id == ChunkedDocument.id)
        .join(ExtractedDocument, ChunkedDocument.extracted_document_id == ExtractedDocument.id)
        .join(SourceDocument, Chunk.document_id == SourceDocument.id)
        .filter(*embedding_conditions, ChunkEmbedding.status == "failed")
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
        #: Why startup aborted, or None. ``start()`` returns normally on a fatal
        #: startup failure, so without this the runner exits 0 and any systemd
        #: unit or CI check keying on exit status concludes the worker is fine.
        self.fatal_reason: str | None = None

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
        # A previous run's reason must not make this start look failed.
        self.fatal_reason = None
        state = self.state_manager.load()
        # PID + start-token: a recycled PID after `stop --force` must not block
        # a fresh start. The Postgres advisory lock below is the authoritative
        # mutual exclusion; this is just a fast local pre-check.
        if (
            state.daemon_state == DaemonState.RUNNING
            and state.pid
            and is_managed_process_alive(state.pid, state.start_token)
        ):
            self._fatal("Pipeline worker already running with PID %s", state.pid)
            return

        engine = get_engine(self.config.database.url)
        create_tables(engine)
        self.Session = get_session_factory(engine)
        worker_lock = _try_acquire_pipeline_worker_lock(engine, self.collection)
        if engine.dialect.name == "postgresql" and worker_lock is None:
            self._fatal(
                "Pipeline worker already holds DB lock for collection=%s", self.collection
            )
            return

        try:
            self.embedding_client = self._create_embedding_client()
            # describe(), not health_check(): the latter only asks whether the
            # daemon *lists* the expected model, so one that answers /v1/models
            # but fails every embed passed this gate and then failed every batch
            # -- and those failures are retryable, so the worker looped on them.
            # describe() performs a real embed round-trip, which is the property
            # actually required here, and its cost is paid once per start.
            self.embedding_client.describe()
        except Exception as error:
            self._fatal("Embedding provider cannot embed: %s", error)
            return

        self.state_manager.update(
            daemon_state=DaemonState.RUNNING,
            pid=os.getpid(),
            start_token=process_start_token(os.getpid()),
            current_file=None,
            # Clear any error left by a previous run. The in-loop clear only
            # resets errors this process recorded, so without this a failure
            # from an earlier worker would be reported by `cementic status`
            # forever -- making a healthy worker look permanently broken.
            last_error=None,
            last_error_at=None,
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
                _release_pipeline_worker_lock(worker_lock, self.collection)
            finally:
                self.stop()

    def _fatal(self, message: str, *args: Any) -> None:
        """Log a fatal startup reason and echo it to stderr.

        The module logger writes to its own file, but `cementic start` points the
        user at the spawned process's stdout/stderr log. Without this echo the
        user is sent to a file that cannot explain why the worker exited.
        """
        self._logger.error(message, *args)
        rendered = message % args if args else message
        self.fatal_reason = rendered
        print(rendered, file=sys.stderr, flush=True)

    def _run_processing_loop(self, revision_id: int) -> None:
        reported_error = False
        while not self._shutdown_event.is_set():
            try:
                if self._step_extract(revision_id):
                    continue
                if self._step_chunk(revision_id):
                    continue
                if self._step_embed(revision_id):
                    continue
                self._mark_revision_ready_if_complete(revision_id)
            except Exception as error:
                # A transient failure (Postgres restart, network blip) must not
                # kill the worker: log, back off, retry. Interrupted rows are
                # re-queued on the next pass or the next `cementic start`.
                self._logger.exception("Pipeline step failed; retrying after backoff")
                # Publish it too: a worker looping on a permanent failure is
                # otherwise indistinguishable from a healthy idle one, and the
                # only evidence lives in a log file the user has to know about.
                self._record_loop_error(error)
                reported_error = True
                self._shutdown_event.wait(5 * self.config.pipeline_worker.poll_interval)
                continue
            if reported_error:
                self.state_manager.update(last_error=None, last_error_at=None)
                reported_error = False
            self._shutdown_event.wait(self.config.pipeline_worker.poll_interval)

    def _record_loop_error(self, error: Exception) -> None:
        """Publish a processing-loop failure to the worker state file."""
        message = f"{type(error).__name__}: {error}"
        try:
            self.state_manager.update(
                last_error=message[:500],
                last_error_at=datetime.now(timezone.utc).isoformat(),
            )
        except Exception:  # pragma: no cover - state file must never mask the real error
            self._logger.exception("Could not record pipeline error to the state file")

    def _ensure_target_revision(self) -> int:
        with self.Session() as session:
            revision = get_target_revision(
                session, self.collection, self.config, self.embedding_client
            )
            # Before any embedding: the ANN index is not built until the revision
            # first completes, so an unindexable dimension otherwise costs the
            # whole corpus and then loops on CREATE INDEX forever.
            dimension_error = index_dimension_error(
                self.config.index.method, revision.embedding_profile.embedding_dim
            )
            if dimension_error is not None:
                raise RuntimeError(dimension_error)
            ensure_revision_vector_table(session, revision)
            session.commit()
            # After the commit: the index DDL runs on its own connection, so the
            # table must already be visible outside this session's transaction.
            ensure_revision_ann_index_up_front(session, revision, self.config)
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
                previous_content_hash = extracted.content_hash
                extracted.source_file_hash = file_hash
                extracted.artifact_path = (
                    str(artifact_path) if content_hash is not None else extracted.artifact_path
                )
                extracted.content_hash = content_hash
                extracted.status = status
                extracted.error_message = error_message
                if content_hash is not None and previous_content_hash != content_hash:
                    _purge_superseded_chunks(session, extracted.id, content_hash)
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
            # Read while the session is open: these are written onto every
            # vector row so search can filter without joining back.
            extractor_profile_id = revision.extractor_profile_id
            chunk_profile_id = revision.chunk_profile_id

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
        except Exception as error:
            if is_retryable_embed_error(error):
                self._release_after_provider_failure(error, claimed, profile_id)
            # A genuine data failure. Retry one text at a time so a single bad
            # chunk is marked failed on its own instead of taking the other
            # batch_size - 1 down with it and blocking promotion.
            try:
                embeddings, failure_message = self._embed_individually(provider, texts, error)
            except Exception as retry_error:
                # The provider went away mid-retry: still a provider fact, so
                # the claim must be released rather than stranded.
                self._release_after_provider_failure(retry_error, claimed, profile_id)
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
                    ensure_vector_table_schema(conn, profile_id, embedding_dim)
                    upsert_vectors(
                        conn,
                        profile_id,
                        successes,
                        collection=self.collection,
                        extractor_profile_id=extractor_profile_id,
                        chunk_profile_id=chunk_profile_id,
                    )
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

    def _release_after_provider_failure(
        self, error: Exception, claimed: list[tuple[int, str]], profile_id: int
    ) -> NoReturn:
        """Return a claimed batch to ``pending``, then re-raise the failure.

        The failure describes the daemon, not these texts, so nothing is stamped
        terminal. Reconnects (autostart per config), then re-raises into the
        processing loop, which records the reason and backs off.

        Re-raising is what makes the failure visible. Returning ``False`` read as
        "no work this pass", so a provider that was down for days was
        indistinguishable from an idle worker: `cementic status` showed the
        workers running with no last error while progress had simply stopped,
        and the same 32 chunks were re-claimed every poll forever.

        Publishing ``last_error`` from here instead would leave the loop's
        ``reported_error`` flag unset, so the clear-on-recovery path would never
        run and a worker that recovered would look broken indefinitely.
        """
        self._logger.error("Embedding provider unavailable: %s", error)
        self._release_claimed_embeddings([chunk_id for chunk_id, _ in claimed], profile_id)
        try:
            self.embedding_client = self._create_embedding_client()
        except Exception as restart_error:
            self._logger.error("Embedding provider restart failed: %s", restart_error)
        raise error

    def _embed_individually(
        self, provider: EmbeddingProvider | None, texts: list[str], batch_error: Exception
    ) -> tuple[list[list[float] | None], str]:
        """Re-embed a failed batch one text at a time.

        A batch call is all-or-nothing, so one unembeddable chunk previously
        failed its whole batch (default 32). Those failures are terminal for the
        run and block ``collection promote``, so isolating the offender keeps
        the other chunks in the index. A retryable error here re-raises so the
        caller's release-and-back-off path still applies.
        """
        if provider is None:
            return [None for _ in texts], str(batch_error)
        results: list[list[float] | None] = []
        for text_value in texts:
            try:
                results.append(provider.embed(text_value))
            except Exception as error:
                if is_retryable_embed_error(error):
                    raise
                self._logger.warning("Chunk could not be embedded: %s", error)
                results.append(None)
        return results, str(batch_error)

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
            # The build occupies this loop for minutes at a time -- tens of
            # minutes on a large corpus -- while writing nothing else, so
            # `cementic status` would otherwise show a running worker with no
            # current file: indistinguishable from an idle one. Announced before
            # it starts, because nothing can be published from inside it.
            self._logger.info(
                "Building the %s index for collection=%s; the worker does no "
                "other work until it finishes, and stopping now discards it",
                self.config.index.method,
                self.collection,
            )
            self.state_manager.update(
                current_activity=f"building {self.config.index.method} index"
            )
            try:
                ensure_revision_ann_index(session, revision, self.config)
            finally:
                self.state_manager.update(current_activity=None)
            mark_revision_ready(session, revision)
            session.commit()

    def _revision_complete(self, session: Session, revision: PipelineRevision) -> bool:
        counts = _compute_revision_counts(session, self.collection, revision)
        return revision_is_complete(counts)

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
