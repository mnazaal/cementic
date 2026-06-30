"""Integration tests for PipelineWorker error paths and lifecycle.

Runs on SQLite with FakeEmbeddingClient — no PostgreSQL needed.
Exercises extraction, chunking, embedding error paths and daemon lifecycle.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cementic.config import Config
from cementic.db import (
    Base,
    Chunk,
    ChunkedDocument,
    ChunkEmbedding,
    ExtractedDocument,
    PipelineRevision,
    SourceDocument,
)
from cementic.embedding_provider import EmbeddingProvider
from cementic.pipeline_worker import PipelineWorker, compute_revision_counts
from cementic.revisions import requeue_interrupted_artifacts
from cementic.source_watcher import SourceWatcher
from tests.integration.test_pg_helpers import cleanup_pg_tables


@pytest.fixture(autouse=True)
def _skip_ann_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the ANN index build here (covered by test_db_pg / test_revisions_pg)."""
    monkeypatch.setattr(
        "cementic.pipeline_worker.ensure_revision_ann_index",
        lambda *args, **kwargs: None,
    )


class FakeEmbeddingClient(EmbeddingProvider):
    """Deterministic fake embedding client for integration tests."""

    TERMS = [
        "computer", "symbiosis", "man", "time", "machine", "learning",
        "vector", "semantic", "neural", "network",
    ]

    def health_check(self) -> bool:
        return True

    def embed(self, text: str) -> list[float]:
        lowered = text.lower().replace("search_document: ", "").replace("search_query: ", "")
        return [float(lowered.count(term)) for term in self.TERMS]

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        return [self.embed(text) for text in texts]

    @property
    def embedding_dim(self) -> int:
        return len(self.TERMS)


class FailingEmbeddingClient(FakeEmbeddingClient):
    """Embedding client that raises on every call for error path testing."""

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        raise RuntimeError("injected embedding failure")


def _config_for(temp_dir: Path) -> Config:
    config = Config()
    config.storage.artifacts_path = temp_dir / "artifacts"
    config.source_watcher.state_path = temp_dir / "source_watcher.json"
    config.pipeline_worker.state_path = temp_dir / "pipeline_worker.json"
    config.source_watcher.log_file = temp_dir / "source_watcher.log"
    config.pipeline_worker.log_file = temp_dir / "pipeline_worker.log"
    config.llama_cpp.model_path = str(temp_dir / "fake-model.gguf")
    config.llama_cpp.embedding_dim = len(FakeEmbeddingClient.TERMS)
    config.pipeline.embedding_provider = "llama-cpp"
    config.pipeline.chunk_size = 64
    config.pipeline.chunk_overlap = 16
    config.pipeline_worker.batch_size = 8
    config.pipeline_worker.poll_interval = 0.01
    return config


def _run_pipeline_until_idle(
    pipeline: PipelineWorker, revision_id: int, max_iterations: int = 200
) -> None:
    """Run pipeline steps until no more work to do."""
    for _ in range(max_iterations):
        worked = False
        if pipeline._step_extract(revision_id):
            worked = True
        if pipeline._step_chunk(revision_id):
            worked = True
        if pipeline._step_embed(revision_id):
            worked = True
        if not worked:
            pipeline._mark_revision_ready_if_complete(revision_id)
            break


@pytest.fixture
def sqlite_setup(temp_dir: Path):
    """Create fresh SQLite DB and return config + session_factory + PDF fixtures dir."""
    config = _config_for(temp_dir)
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    pdf_fixtures_dir = Path(__file__).resolve().parent.parent / "fixtures"
    return config, session_factory, pdf_fixtures_dir


@pytest.fixture
def pg_setup(pg_engine, temp_dir: Path):
    """Like sqlite_setup but backed by Postgres — required for the embed step."""
    config = _config_for(temp_dir)
    session_factory = sessionmaker(bind=pg_engine, expire_on_commit=False)
    pdf_fixtures_dir = Path(__file__).resolve().parent.parent / "fixtures"
    yield config, session_factory, pdf_fixtures_dir
    with sessionmaker(bind=pg_engine)() as session:
        cleanup_pg_tables(session)


def _setup_worker(
    config: Config, session_factory, collection: str, monkeypatch
) -> tuple[PipelineWorker, SourceWatcher]:
    """Set up PipelineWorker and SourceWatcher with SQLite session."""
    source_watcher = SourceWatcher(config)
    source_watcher.Session = session_factory
    source_watcher.collection = collection
    monkeypatch.setattr(source_watcher.state_manager, "update", lambda **kwargs: None)

    pipeline = PipelineWorker(config)
    pipeline.Session = session_factory
    pipeline.collection = collection
    monkeypatch.setattr(pipeline.state_manager, "update", lambda **kwargs: None)

    return pipeline, source_watcher


class TestPipelineWorkerErrorPaths:
    """Tests for pipeline worker error handling."""

    def test_create_embedding_client_unknown_provider(self, temp_dir: Path) -> None:
        config = _config_for(temp_dir)
        config.pipeline.embedding_provider = "unknown"
        worker = PipelineWorker(config)
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            worker._create_embedding_client()

    def test_start_health_check_fails(self, temp_dir: Path) -> None:
        config = _config_for(temp_dir)
        worker = PipelineWorker(config)
        worker.state_manager.update()

        with (
            patch("cementic.pipeline_worker.get_engine") as mock_engine,
            patch("cementic.pipeline_worker.create_tables"),
            patch.object(worker, "_create_embedding_client") as mock_create,
            patch.object(worker, "_run_processing_loop") as mock_loop,
        ):
            # engine must be connectable for create_tables to succeed
            engine = create_engine("sqlite:///:memory:")
            mock_engine.return_value = engine
            Base.metadata.create_all(engine)

            fake_client = FakeEmbeddingClient()
            type(fake_client).health_check = property(lambda self: False)
            mock_create.return_value = fake_client
            worker.start("test_health")
            mock_loop.assert_not_called()

    def test_start_embedding_init_fails(self, temp_dir: Path) -> None:
        config = _config_for(temp_dir)
        worker = PipelineWorker(config)
        worker.state_manager.update()

        with (
            patch("cementic.pipeline_worker.get_engine") as mock_engine,
            patch("cementic.pipeline_worker.create_tables"),
            patch.object(worker, "_create_embedding_client") as mock_create,
            patch.object(worker, "_run_processing_loop") as mock_loop,
        ):
            engine = create_engine("sqlite:///:memory:")
            mock_engine.return_value = engine
            Base.metadata.create_all(engine)
            mock_create.side_effect = RuntimeError("provider init failure")
            worker.start("test_init")
            mock_loop.assert_not_called()

    def test_step_extract_error_yields_failed_status(
        self, sqlite_setup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config, session_factory, pdf_fixtures_dir = sqlite_setup
        collection = "test_extract_error"

        pipeline, source_watcher = _setup_worker(config, session_factory, collection, monkeypatch)

        pdf_path = str(pdf_fixtures_dir / "test_doc_a.pdf")
        source_watcher._register_document(pdf_path)

        with session_factory() as session:
            doc = session.query(SourceDocument).filter_by(collection=collection).first()
            assert doc is not None
            # Corrupt the source path to force extraction failure
            doc.source_path = "/nonexistent/file.pdf"
            session.commit()

        revision = pipeline._ensure_target_revision()
        _run_pipeline_until_idle(pipeline, revision)

        with session_factory() as session:
            extracted = (
                session.query(ExtractedDocument)
                .join(SourceDocument, ExtractedDocument.document_id == SourceDocument.id)
                .filter(SourceDocument.collection == collection)
                .first()
            )
            assert extracted is not None
            assert extracted.status == "failed"
            assert extracted.error_message is not None

    def test_step_chunk_error_yields_failed_status(
        self, sqlite_setup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config, session_factory, pdf_fixtures_dir = sqlite_setup
        collection = "test_chunk_error"

        pipeline, source_watcher = _setup_worker(config, session_factory, collection, monkeypatch)

        pdf_path = str(pdf_fixtures_dir / "test_doc_a.pdf")
        source_watcher._register_document(pdf_path)

        revision = pipeline._ensure_target_revision()

        # Run extraction to get an ExtractedDocument
        pipeline._step_extract(revision)
        pipeline._step_extract(revision)  # second call: already done, no-op

        # Corrupt artifact_path to force chunking error
        with session_factory() as session:
            extracted = (
                session.query(ExtractedDocument).filter_by(status="done").first()
            )
            if extracted is not None:
                extracted.artifact_path = "/nonexistent/artifact.md.gz"
                session.commit()

        pipeline._step_chunk(revision)

        with session_factory() as session:
            chunked = (
                session.query(ChunkedDocument)
                .join(
                    ExtractedDocument,
                    ChunkedDocument.extracted_document_id == ExtractedDocument.id,
                )
                .join(SourceDocument, ExtractedDocument.document_id == SourceDocument.id)
                .filter(SourceDocument.collection == collection)
                .first()
            )
            assert chunked is not None
            assert chunked.status == "failed"

    def test_step_embed_no_candidates_returns_false(
        self, sqlite_setup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config, session_factory, pdf_fixtures_dir = sqlite_setup
        pipeline, source_watcher = _setup_worker(
            config, session_factory, "test_no_candidates", monkeypatch
        )

        revision = pipeline._ensure_target_revision()
        result = pipeline._step_embed(revision)
        assert result is False

    def test_step_embed_handles_embedding_error(
        self, pg_setup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config, session_factory, pdf_fixtures_dir = pg_setup
        collection = "test_embed_error"

        pipeline, source_watcher = _setup_worker(config, session_factory, collection, monkeypatch)

        pdf_path = str(pdf_fixtures_dir / "test_doc_a.pdf")
        source_watcher._register_document(pdf_path)

        pipeline.embedding_client = FakeEmbeddingClient()
        revision = pipeline._ensure_target_revision()
        _run_pipeline_until_idle(pipeline, revision)

        # Reset embeddings to pending
        with session_factory() as session:
            embeddings = session.query(ChunkEmbedding).filter_by(status="done").all()
            for emb in embeddings:
                emb.status = "pending"
            session.commit()

        # Use failing client
        pipeline.embedding_client = FailingEmbeddingClient()
        pipeline._step_embed(revision)

        with session_factory() as session:
            failed_count = session.query(ChunkEmbedding).filter_by(status="failed").count()
            assert failed_count > 0


class TestTerminalFailuresAndDeletedDocs:
    """Failed artifacts are terminal within a run; deleted docs are excluded."""

    def test_failed_extraction_is_terminal_and_reaches_ready(
        self, sqlite_setup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A permanently-failing document does not wedge the build.

        Regression: failed rows used to be re-selected every loop, so the worker
        spun forever and the revision never reached ``ready``.
        """
        config, session_factory, pdf_fixtures_dir = sqlite_setup
        collection = "test_failed_terminal"

        pipeline, source_watcher = _setup_worker(config, session_factory, collection, monkeypatch)
        pipeline.embedding_client = FakeEmbeddingClient()

        source_watcher._register_document(str(pdf_fixtures_dir / "test_doc_a.pdf"))
        with session_factory() as session:
            doc = session.query(SourceDocument).filter_by(collection=collection).first()
            assert doc is not None
            doc.source_path = "/nonexistent/file.pdf"  # force extraction failure
            session.commit()

        revision_id = pipeline._ensure_target_revision()
        _run_pipeline_until_idle(pipeline, revision_id)

        with session_factory() as session:
            revision = session.get(PipelineRevision, revision_id)
            assert revision is not None
            assert revision.status == "ready"  # the build finishes despite the failure
            extracted = session.query(ExtractedDocument).one()
            assert extracted.status == "failed"

    def test_failed_extraction_is_not_reselected_within_run(
        self, sqlite_setup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config, session_factory, pdf_fixtures_dir = sqlite_setup
        collection = "test_failed_once"

        pipeline, source_watcher = _setup_worker(config, session_factory, collection, monkeypatch)
        pipeline.embedding_client = FakeEmbeddingClient()

        source_watcher._register_document(str(pdf_fixtures_dir / "test_doc_a.pdf"))
        with session_factory() as session:
            doc = session.query(SourceDocument).filter_by(collection=collection).first()
            doc.source_path = "/nonexistent/file.pdf"
            session.commit()

        revision_id = pipeline._ensure_target_revision()
        assert pipeline._step_extract(revision_id) is True  # first attempt fails
        # Failed is terminal within the run: no candidate, so no more work.
        assert pipeline._step_extract(revision_id) is False

    def test_deleted_document_excluded_from_pipeline(
        self, sqlite_setup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config, session_factory, pdf_fixtures_dir = sqlite_setup
        collection = "test_deleted_excluded"

        pipeline, source_watcher = _setup_worker(config, session_factory, collection, monkeypatch)
        pipeline.embedding_client = FakeEmbeddingClient()

        source_watcher._register_document(str(pdf_fixtures_dir / "test_doc_a.pdf"))
        source_watcher._register_document(str(pdf_fixtures_dir / "test_doc_b.pdf"))
        with session_factory() as session:
            deleted = (
                session.query(SourceDocument)
                .filter_by(collection=collection)
                .order_by(SourceDocument.id)
                .first()
            )
            deleted.status = "deleted"
            deleted.file_hash = None
            session.commit()
            deleted_id = deleted.id

        # Drive only extraction (the embed step needs pgvector); enough to prove
        # the deleted document is skipped by the worker's candidate selection.
        revision_id = pipeline._ensure_target_revision()
        while pipeline._step_extract(revision_id):
            pass

        with session_factory() as session:
            # The deleted document is never extracted...
            assert (
                session.query(ExtractedDocument)
                .filter_by(document_id=deleted_id)
                .count()
                == 0
            )
            # ...and is excluded from the revision's document total.
            revision = session.get(PipelineRevision, revision_id)
            counts = compute_revision_counts(session, collection, revision)
            assert counts.documents == 1

    def test_requeue_interrupted_artifacts_resets_failed_to_pending(
        self, sqlite_setup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config, session_factory, pdf_fixtures_dir = sqlite_setup
        collection = "test_requeue"

        pipeline, source_watcher = _setup_worker(config, session_factory, collection, monkeypatch)
        pipeline.embedding_client = FakeEmbeddingClient()

        source_watcher._register_document(str(pdf_fixtures_dir / "test_doc_a.pdf"))
        with session_factory() as session:
            doc = session.query(SourceDocument).filter_by(collection=collection).first()
            doc.source_path = "/nonexistent/file.pdf"
            session.commit()

        revision_id = pipeline._ensure_target_revision()
        _run_pipeline_until_idle(pipeline, revision_id)
        with session_factory() as session:
            assert session.query(ExtractedDocument).filter_by(status="failed").count() == 1

        # A fresh `cementic start` re-queues the failure for another attempt.
        with session_factory() as session:
            revision = session.get(PipelineRevision, revision_id)
            requeue_interrupted_artifacts(session, collection, revision)
            session.commit()
            assert session.query(ExtractedDocument).filter_by(status="failed").count() == 0
            assert session.query(ExtractedDocument).filter_by(status="pending").count() == 1


class TestPipelineWorkerFullPipeline:
    """Tests for full pipeline flow on SQLite."""

    def test_full_pipeline_build_and_mark_ready(
        self, pg_setup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config, session_factory, pdf_fixtures_dir = pg_setup
        collection = "test_full"

        pipeline, source_watcher = _setup_worker(config, session_factory, collection, monkeypatch)
        pipeline.embedding_client = FakeEmbeddingClient()

        pdf_path = str(pdf_fixtures_dir / "test_doc_a.pdf")
        source_watcher._register_document(pdf_path)

        revision_id = pipeline._ensure_target_revision()
        _run_pipeline_until_idle(pipeline, revision_id)

        with session_factory() as session:
            revision = session.get(PipelineRevision, revision_id)
            assert revision is not None
            assert revision.status == "ready"

            chunks = session.query(Chunk).count()
            done_embeddings = session.query(ChunkEmbedding).filter_by(status="done").count()
            assert chunks > 0
            assert done_embeddings == chunks

    def test_ensure_target_revision_returns_same_on_repeat(
        self, sqlite_setup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config, session_factory, pdf_fixtures_dir = sqlite_setup
        pipeline, source_watcher = _setup_worker(
            config, session_factory, "test_repeat", monkeypatch
        )

        rev1 = pipeline._ensure_target_revision()
        rev2 = pipeline._ensure_target_revision()
        assert rev1 == rev2

        with session_factory() as session:
            rev = session.get(PipelineRevision, rev1)
            assert rev is not None
            assert rev.status == "building"

    def test_run_processing_loop_exits_on_shutdown(
        self, sqlite_setup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config, session_factory, pdf_fixtures_dir = sqlite_setup
        pipeline, source_watcher = _setup_worker(
            config, session_factory, "test_shutdown", monkeypatch
        )

        pipeline.embedding_client = FakeEmbeddingClient()
        pipeline._shutdown_event.set()  # set shutdown before entering loop
        # Should not raise, should exit immediately
        pipeline._run_processing_loop()

    def test_stop_writes_stopped_state(self, temp_dir: Path) -> None:
        config = _config_for(temp_dir)
        worker = PipelineWorker(config)
        worker.state_manager.update(daemon_state="running", pid=1234)
        worker.stop()

        from cementic.state import DaemonState
        state = worker.state_manager.load()
        assert state.daemon_state == DaemonState.STOPPED
        assert state.pid is None

    def test_handle_shutdown_calls_stop(self, temp_dir: Path) -> None:
        config = _config_for(temp_dir)
        worker = PipelineWorker(config)
        worker.embedding_client = FakeEmbeddingClient()

        with patch.object(worker, "stop") as mock_stop:
            worker._handle_shutdown(15, None)
            mock_stop.assert_called_once()

    def test_step_extract_nonexistent_revision_returns_false(
        self, sqlite_setup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config, session_factory, pdf_fixtures_dir = sqlite_setup
        pipeline, source_watcher = _setup_worker(config, session_factory, "test", monkeypatch)
        pipeline.embedding_client = FakeEmbeddingClient()
        result = pipeline._step_extract(99999)
        assert result is False

    def test_step_chunk_nonexistent_revision_returns_false(
        self, sqlite_setup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config, session_factory, pdf_fixtures_dir = sqlite_setup
        pipeline, source_watcher = _setup_worker(config, session_factory, "test", monkeypatch)
        pipeline.embedding_client = FakeEmbeddingClient()
        result = pipeline._step_chunk(99999)
        assert result is False

    def test_step_embed_nonexistent_revision_returns_false(
        self, sqlite_setup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config, session_factory, pdf_fixtures_dir = sqlite_setup
        pipeline, source_watcher = _setup_worker(config, session_factory, "test", monkeypatch)
        pipeline.embedding_client = FakeEmbeddingClient()
        result = pipeline._step_embed(99999)
        assert result is False
