"""Status query helpers for background workers and pipeline revisions."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import and_, func, or_, text

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
from cementic.state import StateManager, WorkerState
from cementic.supervisor import (
    is_managed_process_alive,
    is_pid_running,
    managed_process_pid,
    managed_process_start_token,
)


@dataclass(frozen=True)
class WorkerStatus:
    """Rendered status for one background worker."""

    state: str
    pid: str
    process: str
    current_file: str
    watched_directories: list[str]
    processed_count: int
    failed_count: int


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
    building_revision_label: str | None


@dataclass(frozen=True)
class HealthStatus:
    """Health check status for runtime dependencies."""

    db_reachable: bool
    embedding_provider: str
    embedding_healthy: bool
    llama_daemon: str


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
    """Return percentage, or 0.0 if total is zero."""
    return round((done / total) * 100, 1) if total > 0 else 0.0


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

    running = bool(state.pid and is_pid_running(state.pid))
    return WorkerStatus(
        state=daemon_state_text(state.daemon_state),
        pid=str(state.pid) if state.pid else "N/A",
        process="running" if running else "stopped",
        current_file=str(state.current_file or "None"),
        watched_directories=[str(path) for path in watched_directories],
        processed_count=state.processed_count,
        failed_count=state.failed_count,
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

        active_by_collection: dict[str, PipelineRevision] = {}
        building_by_collection: dict[str, PipelineRevision] = {}
        for revision in revision_rows:
            if revision.status == "active":
                active_by_collection.setdefault(revision.collection, revision)
            else:
                building_by_collection.setdefault(revision.collection, revision)

        target_by_collection = {
            collection: _select_target_revision(
                building_by_collection.get(collection), active_by_collection.get(collection)
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
            extractor_conditions = [
                and_(
                    SourceDocument.collection == collection,
                    ExtractedDocument.extractor_profile_id == revision.extractor_profile_id,
                )
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

            chunk_conditions = [
                and_(
                    SourceDocument.collection == collection,
                    ExtractedDocument.extractor_profile_id == revision.extractor_profile_id,
                    ChunkedDocument.chunk_profile_id == revision.chunk_profile_id,
                )
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
                .filter(SourceDocument.status != "deleted", or_(*chunk_conditions))
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

            embedding_conditions = [
                and_(
                    SourceDocument.collection == collection,
                    ChunkEmbedding.embedding_profile_id == revision.embedding_profile_id,
                )
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
                    SourceDocument.collection, ChunkEmbedding.status, func.count(ChunkEmbedding.id)
                )
                .select_from(ChunkEmbedding)
                .join(Chunk, ChunkEmbedding.chunk_id == Chunk.id)
                .join(SourceDocument, Chunk.document_id == SourceDocument.id)
                .filter(SourceDocument.status != "deleted", or_(*embedding_conditions))
                .group_by(SourceDocument.collection, ChunkEmbedding.status)
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
            building_revision_label=getattr(building_by_collection.get(collection), "label", None),
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

    llama_daemon = "N/A"
    if embedding_provider == "llama-cpp":
        from cementic.embedding_runtime import llama_daemon_status

        llama_daemon = llama_daemon_status(config)

    embedding_healthy = False
    try:
        from cementic.embedding_runtime import create_provider, runtime_spec_from_config

        client = create_provider(runtime_spec_from_config(config), config, autostart=False)
        embedding_healthy = client.health_check()
    except Exception:
        embedding_healthy = False

    # llama_cpp.server serializes all requests behind a single model lock, so
    # /v1/models can legitimately block for the full duration of an in-flight
    # embedding batch (seconds to tens of seconds) -- no HTTP timeout/retry
    # budget can distinguish "busy" from "down" without either being too slow
    # or too eager to false-flag. The PID+start-token liveness check is
    # instant and process-level, so if the daemon process is confirmed alive,
    # a blocked HTTP probe means busy, not unhealthy.
    if embedding_provider == "llama-cpp" and not embedding_healthy:
        if llama_daemon.startswith("running"):
            embedding_healthy = True

    return HealthStatus(
        db_reachable=db_reachable,
        embedding_provider=embedding_provider,
        embedding_healthy=embedding_healthy,
        llama_daemon=llama_daemon,
    )


def load_file_progress(config: Config, collection: str) -> list[FileProgress]:
    """Load per-file pipeline progress for verbose status output."""
    engine = get_engine(config.database.url)
    session_factory = get_session_factory(engine)
    with session_factory() as session:
        building_revision = (
            session.query(PipelineRevision)
            .filter(
                PipelineRevision.collection == collection,
                PipelineRevision.status.in_(["building", "ready"]),
            )
            .order_by(PipelineRevision.id.desc())
            .first()
        )
        active_revision = (
            session.query(PipelineRevision)
            .filter_by(collection=collection, status="active")
            .order_by(PipelineRevision.id.desc())
            .first()
        )
        target_revision = _select_target_revision(building_revision, active_revision)

        if target_revision is None:
            return []

        rows = (
            session.query(SourceDocument, ExtractedDocument, ChunkedDocument)
            .outerjoin(
                ExtractedDocument,
                and_(
                    ExtractedDocument.document_id == SourceDocument.id,
                    ExtractedDocument.extractor_profile_id == target_revision.extractor_profile_id,
                ),
            )
            .outerjoin(
                ChunkedDocument,
                and_(
                    ChunkedDocument.extracted_document_id == ExtractedDocument.id,
                    ChunkedDocument.chunk_profile_id == target_revision.chunk_profile_id,
                ),
            )
            .filter(SourceDocument.collection == collection, SourceDocument.status != "deleted")
            .order_by(SourceDocument.source_path)
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
