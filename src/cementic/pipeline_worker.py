"""Pipeline worker that builds extraction, chunking, and embeddings for one revision."""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any

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
from cementic.embedding_providers import get_embedding_provider
from cementic.embedding_providers.base import EmbeddingProvider
from cementic.embedding_text import format_document_text
from cementic.extract import extract_pdf_markdown
from cementic.revisions import ensure_revision_ann_index, get_target_revision, mark_revision_ready
from cementic.state import DaemonState, StateManager
from cementic.storage import extracted_document_path, read_extracted_text, write_extracted_text


class PipelineWorker:
    """Builds the target pipeline revision for one collection."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or get_config()
        self.state_manager = StateManager(self.config.pipeline_worker.state_path)
        self._shutdown_event = threading.Event()
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
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _create_embedding_client(self) -> EmbeddingProvider:
        if self.config.pipeline.embedding_provider == "llama-cpp":
            return get_embedding_provider(
                "llama-cpp",
                model_path=self.config.llama_cpp.model_path,
                n_ctx=self.config.llama_cpp.n_ctx,
                n_gpu_layers=self.config.llama_cpp.n_gpu_layers,
                embedding_dim=self.config.llama_cpp.embedding_dim,
                verbose=self.config.llama_cpp.verbose,
            )
        if self.config.pipeline.embedding_provider == "ollama":
            return get_embedding_provider(
                "ollama",
                host=self.config.ollama.host,
                model=self.config.ollama.model,
                embedding_dim=self.config.ollama.embedding_dim,
            )
        raise ValueError(f"Unknown embedding provider: {self.config.pipeline.embedding_provider}")

    def start(self, collection: str = "default") -> None:
        self.collection = collection
        state = self.state_manager.load()
        if state.daemon_state == DaemonState.RUNNING and state.pid:
            try:
                os.kill(state.pid, 0)
                self._logger.error("Pipeline worker already running with PID %s", state.pid)
                return
            except (OSError, ProcessLookupError):
                pass

        engine = get_engine(self.config.database.url)
        create_tables(engine)
        self.Session = get_session_factory(engine)

        try:
            self.embedding_client = self._create_embedding_client()
            if not self.embedding_client.health_check():
                self._logger.error("Embedding provider health check failed")
                return
        except Exception as error:
            self._logger.error("Failed to initialize embedding provider: %s", error)
            return

        self.state_manager.update(daemon_state=DaemonState.RUNNING, pid=os.getpid())
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        self._logger.info("Pipeline worker started for collection=%s", self.collection)

        try:
            self._run_processing_loop()
        finally:
            self.stop()

    def _run_processing_loop(self) -> None:
        while not self._shutdown_event.is_set():
            revision_id = self._ensure_target_revision()
            if self._step_extract(revision_id):
                continue
            if self._step_chunk(revision_id):
                continue
            if self._step_embed(revision_id):
                continue
            self._mark_revision_ready_if_complete(revision_id)
            time.sleep(self.config.pipeline_worker.poll_interval)

    def _ensure_target_revision(self) -> int:
        with self.Session() as session:
            revision = get_target_revision(session, self.collection, self.config)
            session.commit()
            return revision.id

    def _step_extract(self, revision_id: int) -> bool:
        with self.Session() as session:
            revision = session.get(PipelineRevision, revision_id)
            if revision is None:
                return False

            profile_id = revision.extractor_profile_id
            document = None
            extracted = None
            for candidate in (
                session.query(SourceDocument)
                .filter_by(collection=self.collection)
                .order_by(SourceDocument.id)
                .all()
            ):
                current = (
                    session.query(ExtractedDocument)
                    .filter_by(document_id=candidate.id, extractor_profile_id=profile_id)
                    .first()
                )
                if (
                    current is None
                    or current.source_file_hash != candidate.file_hash
                    or current.status != "done"
                ):
                    document = candidate
                    extracted = current
                    break

            if document is None:
                return False

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
            content = extract_pdf_markdown(
                source_path,
                backend=self.config.extraction.backend,
                use_ocr=self.config.extraction.use_ocr,
            )
            content_hash = write_extracted_text(artifact_path, content)
            status = "done"
            error_message = None
        except Exception as error:
            content_hash = None
            status = "failed"
            error_message = str(error)

        with self.Session() as session:
            extracted = session.get(ExtractedDocument, extracted_id)
            document = (
                session.get(SourceDocument, extracted.document_id)
                if extracted is not None
                else None
            )
            if extracted is not None:
                extracted.source_file_hash = file_hash
                extracted.artifact_path = (
                    str(artifact_path) if content_hash is not None else extracted.artifact_path
                )
                extracted.content_hash = content_hash
                extracted.status = status
                extracted.error_message = error_message
            if document is not None:
                document.status = "indexed" if status == "done" else "failed"
                document.error_message = error_message
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
            extracted = None
            chunked = None
            for candidate in (
                session.query(ExtractedDocument)
                .join(SourceDocument, ExtractedDocument.document_id == SourceDocument.id)
                .filter(
                    SourceDocument.collection == self.collection,
                    ExtractedDocument.extractor_profile_id == extractor_profile_id,
                    ExtractedDocument.status == "done",
                )
                .order_by(ExtractedDocument.id)
                .all()
            ):
                current = (
                    session.query(ChunkedDocument)
                    .filter_by(
                        extracted_document_id=candidate.id, chunk_profile_id=chunk_profile_id
                    )
                    .first()
                )
                if (
                    current is None
                    or current.source_content_hash != candidate.content_hash
                    or current.status != "done"
                ):
                    extracted = candidate
                    chunked = current
                    break

            if extracted is None:
                return False

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
            if status == "done" and extracted is not None:
                for item in chunk_items:
                    session.add(
                        Chunk(
                            document_id=extracted.document_id,
                            chunked_document_id=chunked_id,
                            chunk_index=item.chunk_index,
                            content=item.content,
                            page_start=item.page_start,
                            page_end=item.page_end,
                        )
                    )
                chunked.source_content_hash = extracted.content_hash
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
                    ExtractedDocument.extractor_profile_id == revision.extractor_profile_id,
                    ChunkedDocument.chunk_profile_id == revision.chunk_profile_id,
                    ChunkedDocument.status == "done",
                    (ChunkEmbedding.id.is_(None))
                    | (ChunkEmbedding.status.in_(["pending", "failed"])),
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
                elif existing.status in {"pending", "failed"}:
                    existing.status = "processing"
                    existing.error_message = None
                    claimed.append((chunk.id, chunk.content))

            if not claimed:
                session.commit()
                return False

            session.commit()

        texts = [format_document_text(content, self.config) for _, content in claimed]
        try:
            embeddings = (
                self.embedding_client.embed_batch(texts)
                if self.embedding_client is not None
                else []
            )
        except Exception as error:
            embeddings = [None for _ in claimed]
            batch_error = str(error)
        else:
            batch_error = "Failed to generate embedding"

        with self.Session() as session:
            for (chunk_id, _), embedding in zip(claimed, embeddings):
                row = (
                    session.query(ChunkEmbedding)
                    .filter_by(
                        chunk_id=chunk_id, embedding_profile_id=revision.embedding_profile_id
                    )
                    .first()
                )
                if row is None:
                    continue
                if embedding is None:
                    row.status = "failed"
                    row.error_message = batch_error
                else:
                    row.embedding = embedding
                    row.status = "done"
                    row.error_message = None
            session.commit()
        return True

    def _mark_revision_ready_if_complete(self, revision_id: int) -> None:
        with self.Session() as session:
            revision = session.get(PipelineRevision, revision_id)
            if revision is None or revision.status != "building":
                return
            if not self._revision_complete(session, revision):
                return
            if hasattr(revision, "embedding_profile_id") and hasattr(revision, "embedding_profile"):
                ensure_revision_ann_index(session, revision)
            mark_revision_ready(session, revision)
            session.commit()

    def _revision_complete(self, session: Session, revision: PipelineRevision) -> bool:
        documents = session.query(SourceDocument).filter_by(collection=self.collection).count()
        extracted_done = (
            session.query(ExtractedDocument)
            .join(SourceDocument, ExtractedDocument.document_id == SourceDocument.id)
            .filter(
                SourceDocument.collection == self.collection,
                ExtractedDocument.extractor_profile_id == revision.extractor_profile_id,
                ExtractedDocument.status == "done",
                ExtractedDocument.source_file_hash == SourceDocument.file_hash,
            )
            .count()
        )
        if extracted_done != documents:
            return False

        chunked_done = (
            session.query(ChunkedDocument)
            .join(ExtractedDocument, ChunkedDocument.extracted_document_id == ExtractedDocument.id)
            .join(SourceDocument, ExtractedDocument.document_id == SourceDocument.id)
            .filter(
                SourceDocument.collection == self.collection,
                ExtractedDocument.extractor_profile_id == revision.extractor_profile_id,
                ChunkedDocument.chunk_profile_id == revision.chunk_profile_id,
                ChunkedDocument.status == "done",
                ChunkedDocument.source_content_hash == ExtractedDocument.content_hash,
            )
            .count()
        )
        if chunked_done != extracted_done:
            return False

        total_chunks = (
            session.query(Chunk)
            .join(ChunkedDocument, Chunk.chunked_document_id == ChunkedDocument.id)
            .join(ExtractedDocument, ChunkedDocument.extracted_document_id == ExtractedDocument.id)
            .join(SourceDocument, Chunk.document_id == SourceDocument.id)
            .filter(
                SourceDocument.collection == self.collection,
                ExtractedDocument.extractor_profile_id == revision.extractor_profile_id,
                ChunkedDocument.chunk_profile_id == revision.chunk_profile_id,
            )
            .count()
        )
        done_embeddings = (
            session.query(ChunkEmbedding)
            .join(Chunk, ChunkEmbedding.chunk_id == Chunk.id)
            .join(SourceDocument, Chunk.document_id == SourceDocument.id)
            .filter(
                SourceDocument.collection == self.collection,
                ChunkEmbedding.embedding_profile_id == revision.embedding_profile_id,
                ChunkEmbedding.status == "done",
            )
            .count()
        )
        return done_embeddings == total_chunks

    def _handle_shutdown(self, signum: int, frame: object) -> None:
        self._logger.info("Received signal %s, shutting down...", signum)
        self.stop()

    def stop(self) -> None:
        self._shutdown_event.set()
        self.state_manager.update(daemon_state=DaemonState.STOPPED, pid=None)
        self._logger.info("Pipeline worker stopped")


if __name__ == "__main__":
    worker = PipelineWorker()
    worker.start()
