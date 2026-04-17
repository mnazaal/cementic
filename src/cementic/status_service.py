"""Status query helpers for background workers and pipeline revisions."""

from __future__ import annotations

from dataclasses import dataclass

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
from cementic.supervisor import is_pid_running


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
    chunked_done: int
    pending_embeddings: int
    processing_embeddings: int
    done_embeddings: int
    failed_embeddings: int
    active_revision_label: str
    building_revision_label: str


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


def _process_pid(process: dict[str, object]) -> int:
    pid = process.get("pid", 0)
    return pid if isinstance(pid, int) else 0


def build_supervisor_status(supervisor_state: dict[str, object]) -> SupervisorStatus:
    """Build supervisor status from stored process metadata."""
    processes = supervisor_state.get("processes", [])
    if not isinstance(processes, list) or not processes:
        return SupervisorStatus(state="not started", collection="N/A", directories=[])

    process_rows = [proc for proc in processes if isinstance(proc, dict)]
    if not process_rows:
        return SupervisorStatus(state="not started", collection="N/A", directories=[])

    running_count = sum(1 for proc in process_rows if is_pid_running(_process_pid(proc)))
    directories = supervisor_state.get("directories", [])
    directory_list = directories if isinstance(directories, list) else []
    return SupervisorStatus(
        state=f"{running_count}/{len(process_rows)} running",
        collection=str(supervisor_state.get("collection", "N/A")),
        directories=[str(path) for path in directory_list],
    )


def load_pipeline_status(config: Config, collection: str) -> PipelineStatus:
    """Load DB-backed pipeline status for one collection."""
    engine = get_engine(config.database.url)
    session_factory = get_session_factory(engine)
    with session_factory() as session:
        documents = session.query(SourceDocument).filter_by(collection=collection).count()
        active_revision = (
            session.query(PipelineRevision)
            .filter_by(collection=collection, status="active")
            .order_by(PipelineRevision.id.desc())
            .first()
        )
        building_revision = (
            session.query(PipelineRevision)
            .filter(
                PipelineRevision.collection == collection,
                PipelineRevision.status.in_(["building", "ready"]),
            )
            .order_by(PipelineRevision.id.desc())
            .first()
        )
        target_revision = building_revision or active_revision

        extracted_done = 0
        chunked_done = 0
        pending_embeddings = 0
        processing_embeddings = 0
        done_embeddings = 0
        failed_embeddings = 0

        if target_revision is not None:
            extracted_done = (
                session.query(ExtractedDocument)
                .join(SourceDocument, ExtractedDocument.document_id == SourceDocument.id)
                .filter(
                    SourceDocument.collection == collection,
                    ExtractedDocument.extractor_profile_id == target_revision.extractor_profile_id,
                    ExtractedDocument.status == "done",
                )
                .count()
            )
            chunked_done = (
                session.query(ChunkedDocument)
                .join(
                    ExtractedDocument, ChunkedDocument.extracted_document_id == ExtractedDocument.id
                )
                .join(SourceDocument, ExtractedDocument.document_id == SourceDocument.id)
                .filter(
                    SourceDocument.collection == collection,
                    ExtractedDocument.extractor_profile_id == target_revision.extractor_profile_id,
                    ChunkedDocument.chunk_profile_id == target_revision.chunk_profile_id,
                    ChunkedDocument.status == "done",
                )
                .count()
            )
            pending_embeddings = (
                session.query(ChunkEmbedding)
                .join(Chunk, ChunkEmbedding.chunk_id == Chunk.id)
                .join(SourceDocument, Chunk.document_id == SourceDocument.id)
                .filter(
                    SourceDocument.collection == collection,
                    ChunkEmbedding.embedding_profile_id == target_revision.embedding_profile_id,
                    ChunkEmbedding.status == "pending",
                )
                .count()
            )
            processing_embeddings = (
                session.query(ChunkEmbedding)
                .join(Chunk, ChunkEmbedding.chunk_id == Chunk.id)
                .join(SourceDocument, Chunk.document_id == SourceDocument.id)
                .filter(
                    SourceDocument.collection == collection,
                    ChunkEmbedding.embedding_profile_id == target_revision.embedding_profile_id,
                    ChunkEmbedding.status == "processing",
                )
                .count()
            )
            done_embeddings = (
                session.query(ChunkEmbedding)
                .join(Chunk, ChunkEmbedding.chunk_id == Chunk.id)
                .join(SourceDocument, Chunk.document_id == SourceDocument.id)
                .filter(
                    SourceDocument.collection == collection,
                    ChunkEmbedding.embedding_profile_id == target_revision.embedding_profile_id,
                    ChunkEmbedding.status == "done",
                )
                .count()
            )
            failed_embeddings = (
                session.query(ChunkEmbedding)
                .join(Chunk, ChunkEmbedding.chunk_id == Chunk.id)
                .join(SourceDocument, Chunk.document_id == SourceDocument.id)
                .filter(
                    SourceDocument.collection == collection,
                    ChunkEmbedding.embedding_profile_id == target_revision.embedding_profile_id,
                    ChunkEmbedding.status == "failed",
                )
                .count()
            )

    return PipelineStatus(
        documents=documents,
        extracted_done=extracted_done,
        chunked_done=chunked_done,
        pending_embeddings=pending_embeddings,
        processing_embeddings=processing_embeddings,
        done_embeddings=done_embeddings,
        failed_embeddings=failed_embeddings,
        active_revision_label=getattr(active_revision, "label", "None"),
        building_revision_label=getattr(building_revision, "label", "None"),
    )
