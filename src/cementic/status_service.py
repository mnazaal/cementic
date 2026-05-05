"""Status query helpers for background workers and pipeline revisions."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text

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
    active_revision_label: str
    building_revision_label: str


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
        extracted_failed = 0
        chunked_done = 0
        chunked_failed = 0
        total_chunks = 0
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
            extracted_failed = (
                session.query(ExtractedDocument)
                .join(SourceDocument, ExtractedDocument.document_id == SourceDocument.id)
                .filter(
                    SourceDocument.collection == collection,
                    ExtractedDocument.extractor_profile_id == target_revision.extractor_profile_id,
                    ExtractedDocument.status == "failed",
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
            chunked_failed = (
                session.query(ChunkedDocument)
                .join(
                    ExtractedDocument, ChunkedDocument.extracted_document_id == ExtractedDocument.id
                )
                .join(SourceDocument, ExtractedDocument.document_id == SourceDocument.id)
                .filter(
                    SourceDocument.collection == collection,
                    ExtractedDocument.extractor_profile_id == target_revision.extractor_profile_id,
                    ChunkedDocument.chunk_profile_id == target_revision.chunk_profile_id,
                    ChunkedDocument.status == "failed",
                )
                .count()
            )
            total_chunks = (
                session.query(Chunk)
                .join(ChunkedDocument, Chunk.chunked_document_id == ChunkedDocument.id)
                .join(
                    ExtractedDocument,
                    ChunkedDocument.extracted_document_id == ExtractedDocument.id,
                )
                .join(SourceDocument, Chunk.document_id == SourceDocument.id)
                .filter(
                    SourceDocument.collection == collection,
                    ExtractedDocument.extractor_profile_id == target_revision.extractor_profile_id,
                    ChunkedDocument.chunk_profile_id == target_revision.chunk_profile_id,
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
        extracted_failed=extracted_failed,
        chunked_done=chunked_done,
        chunked_failed=chunked_failed,
        total_chunks=total_chunks,
        pending_embeddings=pending_embeddings,
        processing_embeddings=processing_embeddings,
        done_embeddings=done_embeddings,
        failed_embeddings=failed_embeddings,
        extraction_pct=_safe_pct(extracted_done, documents),
        chunking_pct=_safe_pct(chunked_done, extracted_done),
        embedding_pct=_safe_pct(done_embeddings, total_chunks),
        active_revision_label=getattr(active_revision, "label", "None"),
        building_revision_label=getattr(building_revision, "label", "None"),
    )


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
    embedding_healthy = False
    try:
        client: object  # EmbeddingProvider-like
        if embedding_provider == "llama-cpp":
            from cementic.embedding_runtime import get_llama_cpp_runtime_client

            client = get_llama_cpp_runtime_client(config)
            embedding_healthy = client.health_check()
        elif embedding_provider == "ollama":
            from cementic.embedding_providers import get_embedding_provider

            client = get_embedding_provider(
                "ollama",
                host=config.ollama.host,
                model=config.ollama.model,
                embedding_dim=config.ollama.embedding_dim,
            )
            embedding_healthy = client.health_check()
    except Exception:
        embedding_healthy = False

    llama_daemon = "N/A"
    if embedding_provider == "llama-cpp":
        pid_file = config.llama_cpp.daemon_pid_file
        if pid_file is not None and pid_file.exists():
            try:
                pid = int(pid_file.read_text(encoding="utf-8").strip())
                llama_daemon = f"running, pid={pid}" if is_pid_running(pid) else "stopped"
            except (ValueError, OSError):
                llama_daemon = "stopped"
        else:
            llama_daemon = "stopped"

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
        target_revision = (
            session.query(PipelineRevision)
            .filter(
                PipelineRevision.collection == collection,
                PipelineRevision.status.in_(["building", "ready"]),
            )
            .order_by(PipelineRevision.id.desc())
            .first()
        ) or (
            session.query(PipelineRevision)
            .filter_by(collection=collection, status="active")
            .order_by(PipelineRevision.id.desc())
            .first()
        )

        if target_revision is None:
            return []

        documents = (
            session.query(SourceDocument)
            .filter_by(collection=collection)
            .order_by(SourceDocument.source_path)
            .all()
        )

        result: list[FileProgress] = []
        for doc in documents:
            extraction = (
                session.query(ExtractedDocument)
                .filter_by(
                    document_id=doc.id,
                    extractor_profile_id=target_revision.extractor_profile_id,
                )
                .first()
            )
            extraction_status = extraction.status if extraction is not None else "pending"

            chunking = (
                session.query(ChunkedDocument)
                .filter_by(
                    extracted_document_id=extraction.id,
                    chunk_profile_id=target_revision.chunk_profile_id,
                )
                .first()
                if extraction is not None
                else None
            )
            chunking_status = chunking.status if chunking is not None else "pending"

            embeddings_done = 0
            embeddings_failed = 0
            embeddings_total = 0
            error_message = None

            if chunking is not None:
                embeddings_total = (
                    session.query(Chunk)
                    .filter_by(chunked_document_id=chunking.id)
                    .count()
                )
                embeddings_done = (
                    session.query(ChunkEmbedding)
                    .join(Chunk, ChunkEmbedding.chunk_id == Chunk.id)
                    .filter(
                        Chunk.chunked_document_id == chunking.id,
                        ChunkEmbedding.embedding_profile_id
                        == target_revision.embedding_profile_id,
                        ChunkEmbedding.status == "done",
                    )
                    .count()
                )
                embeddings_failed = (
                    session.query(ChunkEmbedding)
                    .join(Chunk, ChunkEmbedding.chunk_id == Chunk.id)
                    .filter(
                        Chunk.chunked_document_id == chunking.id,
                        ChunkEmbedding.embedding_profile_id
                        == target_revision.embedding_profile_id,
                        ChunkEmbedding.status == "failed",
                    )
                    .count()
                )

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
