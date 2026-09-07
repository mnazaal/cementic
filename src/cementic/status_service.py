"""Status query helpers for background workers and pipeline revisions."""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import ColumnElement, and_, case, func, or_, text

from cementic.config import Config
from cementic.db import (
    Chunk,
    ChunkedDocument,
    ChunkEmbedding,
    ExtractedDocument,
    PipelineRevision,
    SourceDocument,
    get_engine,
    get_session_factory,
)
from cementic.embedding_runtime import DaemonHealth
from cementic.revisions import (
    BUILDING_STATUSES,
    bucket_revisions_by_status,
    chunk_scope,
    chunked_scope,
    embedding_scope_denormalised,
    extracted_scope,
    get_active_revision,
)
from cementic.state import StateManager, WorkerState
from cementic.supervisor import (
    is_managed_process_alive,
    managed_process_pid,
    managed_process_start_token,
)


@dataclass(frozen=True)
class WorkerStatus:
    """Rendered status for one background worker."""

    state: str
    pid: str
    process: str
    #: The file being processed, or None when idle. Kept as None rather than a
    #: "None" string: the string leaked into `status --json`, where every
    #: consumer testing truthiness or `is not None` saw a stopped worker as busy.
    current_file: str | None
    watched_directories: list[str]
    processed_count: int
    failed_count: int
    last_error: str | None = None
    last_error_at: str | None = None
    #: Long-running work that is not a file, so a worker mid-index-build is not
    #: reported as idle.
    current_activity: str | None = None
    #: Files the watcher refused to register, most recent last. They never
    #: become documents, so no pipeline count can show them.
    skipped_files: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SupervisorStatus:
    """Rendered status for the process supervisor."""

    state: str
    collection: str
    directories: list[str]


@dataclass(frozen=True)
class PipelineStatus:
    """Rendered status for the active/building pipeline."""

    documents: int
    extracted_done: int
    extracted_failed: int
    chunked_done: int
    chunked_failed: int
    total_chunks: int
    pending_embeddings: int
    processing_embeddings: int
    done_embeddings: int
    failed_embeddings: int
    extraction_pct: float
    chunking_pct: float
    embedding_pct: float
    active_revision_label: str | None
    ready_revision_label: str | None
    building_revision_label: str | None


@dataclass(frozen=True)
class HealthStatus:
    """Health check status for runtime dependencies."""

    db_reachable: bool
    embedding_provider: str
    embedding_healthy: bool
    llama_daemon: str
    #: The probe's classification; None means the probe itself failed and
    #: ``llama_daemon`` carries the reason.
    llama_daemon_health: DaemonHealth | None = None


@dataclass(frozen=True)
class FileProgress:
    """Per-file pipeline progress for verbose status output."""

    source_path: str
    extraction_status: str
    chunking_status: str
    embeddings_done: int
    embeddings_failed: int
    embeddings_total: int
    error_message: str | None = None


def _safe_pct(done: int, total: int) -> float:
    """Return percentage, or 0.0 if total is zero.

    Rounding is not allowed to reach 100.0 while anything is outstanding. One
    decimal place over 2.16M chunks is a resolution of ~2,162 chunks, so a build
    with thousands still pending rounded up and displayed as finished -- and
    "finished" is the one thing a reader acts on. Holding at 99.9 until the last
    item lands keeps 100.0 meaning done.
    """
    if total <= 0:
        return 0.0
    pct = round((done / total) * 100, 1)
    if pct >= 100.0 and done < total:
        return 99.9
    return pct


def _select_target_revision(
    building_revision: PipelineRevision | None,
    active_revision: PipelineRevision | None,
) -> PipelineRevision | None:
    """Prefer in-flight revision for status; fall back to active revision."""
    return building_revision or active_revision


def daemon_state_text(value: object) -> str:
    """Normalize worker state for display."""
    if hasattr(value, "value"):
        return str(getattr(value, "value"))
    return str(value)


def build_worker_status(state: WorkerState) -> WorkerStatus:
    """Build display status from one worker state object."""
    watched_directories = state.watched_directories or []
    if not isinstance(watched_directories, list):
        watched_directories = [str(watched_directories)]

    # Token-checked liveness: a recycled PID after a force-kill must not show
    # as "running".
    running = bool(state.pid and is_managed_process_alive(state.pid, state.start_token))
    return WorkerStatus(
        state=daemon_state_text(state.daemon_state),
        pid=str(state.pid) if state.pid else "N/A",
        process="running" if running else "stopped",
        current_file=str(state.current_file) if state.current_file else None,
        watched_directories=[str(path) for path in watched_directories],
        processed_count=state.processed_count,
        failed_count=state.failed_count,
        last_error=state.last_error,
        last_error_at=state.last_error_at,
        current_activity=state.current_activity,
        skipped_files=list(state.skipped_files),
    )


def load_worker_statuses(config: Config) -> tuple[WorkerStatus, WorkerStatus]:
    """Load source watcher and pipeline worker status from state files."""
    source_watcher_state = StateManager(config.source_watcher.state_path).load()
    pipeline_state = StateManager(config.pipeline_worker.state_path).load()
    return build_worker_status(source_watcher_state), build_worker_status(pipeline_state)


def build_supervisor_status(supervisor_state: dict[str, object]) -> SupervisorStatus:
    """Build supervisor status from stored process metadata."""
    processes = supervisor_state.get("processes", [])
    if not isinstance(processes, list) or not processes:
        return SupervisorStatus(state="not started", collection="N/A", directories=[])

    process_rows = [proc for proc in processes if isinstance(proc, dict)]
    if not process_rows:
        return SupervisorStatus(state="not started", collection="N/A", directories=[])

    running_count = sum(
        1
        for proc in process_rows
        if is_managed_process_alive(managed_process_pid(proc), managed_process_start_token(proc))
    )
    directories = supervisor_state.get("directories", [])
    directory_list = directories if isinstance(directories, list) else []
    return SupervisorStatus(
        state=f"{running_count}/{len(process_rows)} running",
        collection=str(supervisor_state.get("collection", "N/A")),
        directories=[str(path) for path in directory_list],
    )


def load_pipeline_status(config: Config, collection: str) -> PipelineStatus:
    """Load DB-backed pipeline status for one collection."""
    return load_pipeline_status_bulk(config, [collection])[collection]


def load_pipeline_status_bulk(config: Config, collections: list[str]) -> dict[str, PipelineStatus]:
    """Load DB-backed pipeline status for many collections in a bounded number of queries.

    A naive per-collection call to a single-collection status loader costs
    ~11 queries per collection, which stops scaling once a deployment has many
    collections. This groups every count by collection instead, so the total
    query count stays constant regardless of how many collections are asked for.
    """
    if not collections:
        return {}

    engine = get_engine(config.database.url)
    session_factory = get_session_factory(engine)
    with session_factory() as session:
        revision_rows = (
            session.query(PipelineRevision)
            .filter(
                PipelineRevision.collection.in_(collections),
                PipelineRevision.status.in_(["active", "ready", "building"]),
            )
            .order_by(PipelineRevision.collection, PipelineRevision.id.desc())
            .all()
        )

        active_by_collection, ready_by_collection, building_by_collection = (
            bucket_revisions_by_status(revision_rows)
        )
        # Target selection wants the newest not-yet-active revision whatever
        # its status: the newer of the two per-status newest ones.
        in_flight_by_collection: dict[str, PipelineRevision] = {}
        for name in collections:
            candidates = [
                revision
                for revision in (
                    ready_by_collection.get(name),
                    building_by_collection.get(name),
                )
                if revision is not None
            ]
            if candidates:
                in_flight_by_collection[name] = max(candidates, key=lambda r: r.id)

        target_by_collection = {
            collection: _select_target_revision(
                in_flight_by_collection.get(collection), active_by_collection.get(collection)
            )
            for collection in collections
        }

        documents_by_collection: dict[str, int] = {
            collection: count
            for collection, count in session.query(
                SourceDocument.collection, func.count(SourceDocument.id)
            )
            .filter(SourceDocument.collection.in_(collections), SourceDocument.status != "deleted")
            .group_by(SourceDocument.collection)
            .all()
        }

        targets = [
            (collection, revision)
            for collection, revision in target_by_collection.items()
            if revision is not None
        ]

        extracted_done: dict[str, int] = {}
        extracted_failed: dict[str, int] = {}
        chunked_done: dict[str, int] = {}
        chunked_failed: dict[str, int] = {}
        total_chunks: dict[str, int] = {}
        pending_embeddings: dict[str, int] = {}
        processing_embeddings: dict[str, int] = {}
        done_embeddings: dict[str, int] = {}
        failed_embeddings: dict[str, int] = {}

        if targets:
            # Every scope below is the same definition the pipeline worker's own
            # completeness check uses. They were separate copies and drifted:
            # status omitted the source/content hash equalities and counted every
            # non-"done" chunking as failed, so `cementic status` reported queued
            # work as failures and showed 100% extracted for files that had
            # changed on disk and still owed a re-extraction.
            extractor_conditions = [
                and_(SourceDocument.collection == collection, *extracted_scope(revision))
                for collection, revision in targets
            ]
            for collection, status, count in (
                session.query(
                    SourceDocument.collection,
                    ExtractedDocument.status,
                    func.count(ExtractedDocument.id),
                )
                .join(SourceDocument, ExtractedDocument.document_id == SourceDocument.id)
                .filter(
                    SourceDocument.status != "deleted",
                    ExtractedDocument.status.in_(["done", "failed"]),
                    or_(*extractor_conditions),
                )
                .group_by(SourceDocument.collection, ExtractedDocument.status)
                .all()
            ):
                (extracted_done if status == "done" else extracted_failed)[collection] = count

            chunked_conditions = [
                and_(SourceDocument.collection == collection, *chunked_scope(revision))
                for collection, revision in targets
            ]
            chunk_conditions = [
                and_(SourceDocument.collection == collection, *chunk_scope(revision))
                for collection, revision in targets
            ]
            for collection, status, count in (
                session.query(
                    SourceDocument.collection,
                    ChunkedDocument.status,
                    func.count(ChunkedDocument.id),
                )
                .join(
                    ExtractedDocument, ChunkedDocument.extracted_document_id == ExtractedDocument.id
                )
                .join(SourceDocument, ExtractedDocument.document_id == SourceDocument.id)
                .filter(
                    SourceDocument.status != "deleted",
                    ChunkedDocument.status.in_(["done", "failed"]),
                    or_(*chunked_conditions),
                )
                .group_by(SourceDocument.collection, ChunkedDocument.status)
                .all()
            ):
                (chunked_done if status == "done" else chunked_failed)[collection] = count

            total_chunks = {
                collection: count
                for collection, count in session.query(
                    SourceDocument.collection, func.count(Chunk.id)
                )
                .select_from(Chunk)
                .join(ChunkedDocument, Chunk.chunked_document_id == ChunkedDocument.id)
                .join(
                    ExtractedDocument, ChunkedDocument.extracted_document_id == ExtractedDocument.id
                )
                .join(SourceDocument, Chunk.document_id == SourceDocument.id)
                .filter(SourceDocument.status != "deleted", or_(*chunk_conditions))
                .group_by(SourceDocument.collection)
                .all()
            }

            # Single table by design, like `_step_embed`'s claim: the joined
            # form of this count scans every chunk to reach three columns that
            # `chunk_embeddings` already carries. See
            # `embedding_scope_denormalised` for why the two conditions it
            # cannot express exclude nothing.
            embedding_conditions = [
                and_(*embedding_scope_denormalised(collection, revision))
                for collection, revision in targets
            ]
            embedding_status_map = {
                "pending": pending_embeddings,
                "processing": processing_embeddings,
                "done": done_embeddings,
                "failed": failed_embeddings,
            }
            for collection, status, count in (
                session.query(
                    ChunkEmbedding.collection, ChunkEmbedding.status, func.count(ChunkEmbedding.id)
                )
                .filter(or_(*embedding_conditions))
                .group_by(ChunkEmbedding.collection, ChunkEmbedding.status)
                .all()
            ):
                embedding_status_map[status][collection] = count

    result: dict[str, PipelineStatus] = {}
    for collection in collections:
        documents = documents_by_collection.get(collection, 0)
        e_done = extracted_done.get(collection, 0)
        c_done = chunked_done.get(collection, 0)
        t_chunks = total_chunks.get(collection, 0)
        d_embeddings = done_embeddings.get(collection, 0)
        result[collection] = PipelineStatus(
            documents=documents,
            extracted_done=e_done,
            extracted_failed=extracted_failed.get(collection, 0),
            chunked_done=c_done,
            chunked_failed=chunked_failed.get(collection, 0),
            total_chunks=t_chunks,
            pending_embeddings=pending_embeddings.get(collection, 0),
            processing_embeddings=processing_embeddings.get(collection, 0),
            done_embeddings=d_embeddings,
            failed_embeddings=failed_embeddings.get(collection, 0),
            extraction_pct=_safe_pct(e_done, documents),
            chunking_pct=_safe_pct(c_done, e_done),
            embedding_pct=_safe_pct(d_embeddings, t_chunks),
            active_revision_label=getattr(active_by_collection.get(collection), "label", None),
            ready_revision_label=getattr(ready_by_collection.get(collection), "label", None),
            building_revision_label=getattr(
                building_by_collection.get(collection), "label", None
            ),
        )
    return result


def check_health(config: Config) -> HealthStatus:
    """Check health of runtime dependencies."""
    db_reachable = False
    try:
        engine = get_engine(config.database.url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            db_reachable = True
    except Exception:
        db_reachable = False

    embedding_provider = config.pipeline.embedding_provider

    from cementic.embedding_runtime import llama_daemon_status

    llama_daemon = llama_daemon_status(config)

    # One quick probe, no waiting. This used to build the client through
    # `create_provider`, whose own busy-poll blocks for up to two minutes -- and
    # then the result was overridden from the pid file anyway, which was already
    # known above. A status read must not wait out an in-flight batch.
    from cementic.embedding_runtime import (
        EMBED_PROBE_SECONDS,
        build_llama_cpp_client,
        describe_daemon_health,
        probe_daemon,
    )

    embedding_healthy = False
    llama_daemon_health: DaemonHealth | None
    try:
        health = probe_daemon(
            build_llama_cpp_client(config),
            config,
            wait_seconds=0.0,
            # The second stage exercises the embedding path itself: the
            # model list is served without the model lock, so on its own it
            # called a daemon healthy that had answered no embedding for 21
            # hours. Costs up to EMBED_PROBE_SECONDS, and only when the
            # first stage already said healthy.
            embed_probe_seconds=EMBED_PROBE_SECONDS,
        )
    except Exception as error:
        # Keep the reason instead of flattening to DOWN: an ambiguous-PID
        # refusal used to render as a benign "stopped (autostarts when
        # needed)" -- for a state where autostart raises the same refusal.
        llama_daemon = f"unknown ({error})"
        llama_daemon_health = None
    else:
        llama_daemon_health = health
        # BUSY counts as healthy: the process is confirmed alive and merely
        # mid-batch. WRONG_MODEL does not -- the daemon answered, and what it
        # serves is not what this config asks for. Treating that as healthy is
        # what let a stale daemon look fine to `status` while search and the
        # worker restarted it from under each other. WEDGED does not either:
        # it answers listings but no embeddings, with no worker load to explain
        # the silence.
        embedding_healthy = health in (DaemonHealth.HEALTHY, DaemonHealth.BUSY)
        if health in (DaemonHealth.WRONG_MODEL, DaemonHealth.WEDGED):
            llama_daemon = describe_daemon_health(health)

    return HealthStatus(
        db_reachable=db_reachable,
        embedding_provider=embedding_provider,
        embedding_healthy=embedding_healthy,
        llama_daemon=llama_daemon,
        llama_daemon_health=llama_daemon_health,
    )


def _file_progress_rank() -> ColumnElement[int]:
    """Order files by how much they want a reader: failed, then unfinished, then done.

    Written against the outer-joined tables, so a missing artifact row (no
    extraction attempted yet) is checked by id before its status is compared --
    a NULL status is neither equal nor unequal to 'done' and would otherwise
    fall through to the done bucket.
    """
    return case(
        (ExtractedDocument.status == "failed", 0),
        (ChunkedDocument.status == "failed", 0),
        (ExtractedDocument.id.is_(None), 1),
        (ExtractedDocument.status != "done", 1),
        (ChunkedDocument.id.is_(None), 1),
        (ChunkedDocument.status != "done", 1),
        else_=2,
    )


def load_file_progress(
    config: Config, collection: str, limit: int | None = None
) -> list[FileProgress]:
    """Load per-file pipeline progress for verbose status output.

    ``limit`` caps the rows read, and None means every document. A corpus-scale
    collection is the reason the cap exists: 23k documents is one ORM object per
    document plus a per-chunk count for each, which is tens of megabytes loaded
    to print a page of terminal output nobody reads past.

    The cap is only useful if the interesting rows survive it, so failures sort
    first and never-started work second -- the two states a reader opens this
    view to find. Ordering runs in SQL rather than over the result, because
    sorting after the LIMIT would sort whichever arbitrary rows the cap let
    through. Embedding failures are not part of the rank: they are counted per
    chunked document in a second query below, and the summary line above this
    listing already names them.
    """
    engine = get_engine(config.database.url)
    session_factory = get_session_factory(engine)
    with session_factory() as session:
        building_revision = (
            session.query(PipelineRevision)
            .filter(
                PipelineRevision.collection == collection,
                PipelineRevision.status.in_(BUILDING_STATUSES),
            )
            .order_by(PipelineRevision.id.desc())
            .first()
        )
        active_revision = get_active_revision(session, collection)
        target_revision = _select_target_revision(building_revision, active_revision)

        if target_revision is None:
            return []

        rows = (
            session.query(SourceDocument, ExtractedDocument, ChunkedDocument)
            # Matching on profile ids alone made this view disagree with the
            # summary directly above it: after a file changed on disk the counts
            # said "extracted 9/10" while every row here still read
            # "extract=done". The hash equalities are what the counts mean by
            # current, so the per-file view has to apply them too, and a
            # superseded artifact then shows as pending -- which is what it is.
            #
            # Only the ChunkedDocument half of `chunked_scope` is taken: it also
            # requires the extraction to be done, and forcing that into an outer
            # join would drop failed extractions from the listing entirely.
            .outerjoin(
                ExtractedDocument,
                and_(
                    ExtractedDocument.document_id == SourceDocument.id,
                    *extracted_scope(target_revision),
                ),
            )
            .outerjoin(
                ChunkedDocument,
                and_(
                    ChunkedDocument.extracted_document_id == ExtractedDocument.id,
                    ChunkedDocument.chunk_profile_id == target_revision.chunk_profile_id,
                    ChunkedDocument.source_content_hash == ExtractedDocument.content_hash,
                ),
            )
            .filter(SourceDocument.collection == collection, SourceDocument.status != "deleted")
            .order_by(_file_progress_rank(), SourceDocument.source_path)
            .limit(limit)
            .all()
        )

        chunked_ids = [chunked.id for _, _, chunked in rows if chunked is not None]

        total_by_chunked: dict[int, int] = {}
        done_by_chunked: dict[int, int] = {}
        failed_by_chunked: dict[int, int] = {}
        if chunked_ids:
            total_by_chunked = {
                chunked_document_id: count
                for chunked_document_id, count in session.query(
                    Chunk.chunked_document_id, func.count(Chunk.id)
                )
                .filter(Chunk.chunked_document_id.in_(chunked_ids))
                .group_by(Chunk.chunked_document_id)
                .all()
            }
            status_counts = (
                session.query(
                    Chunk.chunked_document_id, ChunkEmbedding.status, func.count(ChunkEmbedding.id)
                )
                .join(Chunk, ChunkEmbedding.chunk_id == Chunk.id)
                .filter(
                    Chunk.chunked_document_id.in_(chunked_ids),
                    ChunkEmbedding.embedding_profile_id == target_revision.embedding_profile_id,
                    ChunkEmbedding.status.in_(["done", "failed"]),
                )
                .group_by(Chunk.chunked_document_id, ChunkEmbedding.status)
                .all()
            )
            for chunked_document_id, status, count in status_counts:
                if status == "done":
                    done_by_chunked[chunked_document_id] = count
                elif status == "failed":
                    failed_by_chunked[chunked_document_id] = count

        result: list[FileProgress] = []
        for doc, extraction, chunking in rows:
            extraction_status = extraction.status if extraction is not None else "pending"
            chunking_status = chunking.status if chunking is not None else "pending"

            embeddings_total = total_by_chunked.get(chunking.id, 0) if chunking is not None else 0
            embeddings_done = done_by_chunked.get(chunking.id, 0) if chunking is not None else 0
            embeddings_failed = (
                failed_by_chunked.get(chunking.id, 0) if chunking is not None else 0
            )

            error_message = None
            if extraction_status == "failed" and extraction is not None:
                error_message = extraction.error_message
            elif chunking_status == "failed" and chunking is not None:
                error_message = chunking.error_message

            result.append(
                FileProgress(
                    source_path=doc.source_path,
                    extraction_status=extraction_status,
                    chunking_status=chunking_status,
                    embeddings_done=embeddings_done,
                    embeddings_failed=embeddings_failed,
                    embeddings_total=embeddings_total,
                    error_message=error_message,
                )
            )

    return result
