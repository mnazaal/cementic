"""End-to-end integration tests for full CLI workflow.

Pipeline build and status tests run on SQLite.
Search tests require PostgreSQL (pgvector dependency).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cementic.config import Config
from cementic.db import (
    Base,
    Chunk,
    ChunkEmbedding,
    PipelineRevision,
    SourceDocument,
)
from cementic.embedding_providers.base import EmbeddingProvider
from cementic.pipeline_worker import PipelineWorker
from cementic.revisions import promote_revision
from cementic.source_watcher import SourceWatcher


class FakeEmbeddingClient(EmbeddingProvider):
    """Tiny deterministic embedding client for integration tests."""

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
        pipeline._mark_revision_ready_if_complete(revision_id)
        if not worked:
            break
    else:
        raise AssertionError("Pipeline did not settle within iteration budget")


class TestE2EPipeline:
    """Full pipeline build and status tests (SQLite-compatible)."""

    @pytest.fixture(autouse=True)
    def _skip_ann_index(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Skip ANN index creation — SQLite doesn't support pgvector HNSW."""
        monkeypatch.setattr(
            "cementic.pipeline_worker.ensure_revision_ann_index",
            lambda session, revision: None,
        )

    def test_full_pipeline_build_and_promote(
        self, temp_dir: Path, pdf_fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Register a real PDF, build full pipeline, promote revision, verify state."""
        config = _config_for(temp_dir)
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)

        collection = "e2e-docs"
        pdf_path = pdf_fixtures_dir / "test_doc_a.pdf"
        assert pdf_path.exists(), f"PDF fixture missing: {pdf_path}"

        source_watcher = SourceWatcher(config)
        source_watcher.Session = session_factory
        source_watcher.collection = collection
        monkeypatch.setattr(source_watcher.state_manager, "update", lambda **kwargs: None)

        pipeline = PipelineWorker(config)
        pipeline.Session = session_factory
        pipeline.collection = collection
        pipeline.embedding_client = FakeEmbeddingClient()
        monkeypatch.setattr(pipeline.state_manager, "update", lambda **kwargs: None)

        # Register PDF
        source_watcher._register_pdf(str(pdf_path))
        with session_factory() as session:
            docs = session.query(SourceDocument).filter_by(collection=collection).all()
            assert len(docs) == 1
            assert docs[0].status == "pending"

        # Build pipeline
        revision_id = pipeline._ensure_target_revision()
        _run_pipeline_until_idle(pipeline, revision_id)

        # Verify completion
        with session_factory() as session:
            revision = session.get(PipelineRevision, revision_id)
            assert revision is not None
            assert revision.status == "ready"
            chunks = session.query(Chunk).count()
            embeddings_done = session.query(ChunkEmbedding).filter_by(status="done").count()
            assert chunks > 0
            assert embeddings_done == chunks

        # Promote
        with session_factory() as session:
            revision = session.get(PipelineRevision, revision_id)
            assert revision is not None
            promote_revision(session, collection, revision)
            session.commit()

        # Verify active revision
        with session_factory() as session:
            active = (
                session.query(PipelineRevision)
                .filter_by(collection=collection, status="active")
                .first()
            )
            assert active is not None
            assert active.id == revision_id

    def test_two_collections_independent(
        self, temp_dir: Path, pdf_fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two collections build independently with separate revisions."""
        config = _config_for(temp_dir)
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)

        def _build_collection(name: str, pdf_file: str) -> int:
            source_watcher = SourceWatcher(config)
            source_watcher.Session = session_factory
            source_watcher.collection = name
            monkeypatch.setattr(source_watcher.state_manager, "update", lambda **kwargs: None)

            pipeline = PipelineWorker(config)
            pipeline.Session = session_factory
            pipeline.collection = name
            pipeline.embedding_client = FakeEmbeddingClient()
            monkeypatch.setattr(pipeline.state_manager, "update", lambda **kwargs: None)

            pdf_path = pdf_fixtures_dir / pdf_file
            source_watcher._register_pdf(str(pdf_path))
            revision_id = pipeline._ensure_target_revision()
            _run_pipeline_until_idle(pipeline, revision_id)

            with session_factory() as session:
                revision = session.get(PipelineRevision, revision_id)
                assert revision is not None
                promote_revision(session, name, revision)
                session.commit()
            return revision_id

        rev_a = _build_collection("docs-a", "test_doc_a.pdf")
        rev_b = _build_collection("docs-b", "test_doc_b.pdf")
        assert rev_a != rev_b

        # Verify both collections have active revisions
        with session_factory() as session:
            active_a = (
                session.query(PipelineRevision)
                .filter_by(collection="docs-a", status="active")
                .first()
            )
            active_b = (
                session.query(PipelineRevision)
                .filter_by(collection="docs-b", status="active")
                .first()
            )
            assert active_a is not None
            assert active_b is not None
            assert active_a.collection == "docs-a"
            assert active_b.collection == "docs-b"

    def test_model_change_creates_new_revision(
        self, temp_dir: Path, pdf_fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Changing embedding model creates new revision; old one superseded/retired."""
        config = _config_for(temp_dir)
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)

        collection = "model-test"
        pdf_path = pdf_fixtures_dir / "test_doc_a.pdf"

        source_watcher = SourceWatcher(config)
        source_watcher.Session = session_factory
        source_watcher.collection = collection
        monkeypatch.setattr(source_watcher.state_manager, "update", lambda **kwargs: None)

        pipeline = PipelineWorker(config)
        pipeline.Session = session_factory
        pipeline.collection = collection
        pipeline.embedding_client = FakeEmbeddingClient()
        monkeypatch.setattr(pipeline.state_manager, "update", lambda **kwargs: None)

        source_watcher._register_pdf(str(pdf_path))
        rev1_id = pipeline._ensure_target_revision()
        _run_pipeline_until_idle(pipeline, rev1_id)

        # Promote first revision
        with session_factory() as session:
            rev1 = session.get(PipelineRevision, rev1_id)
            assert rev1 is not None
            promote_revision(session, collection, rev1)
            session.commit()

        # Change model config
        config.pipeline.embedding_provider = "ollama"
        config.ollama.embedding_dim = 768
        rev2_id = pipeline._ensure_target_revision()

        with session_factory() as session:
            rev1 = session.get(PipelineRevision, rev1_id)
            rev2 = session.get(PipelineRevision, rev2_id)
            assert rev1 is not None
            assert rev2 is not None
            assert rev2.id != rev1.id
            assert rev1.status == "active"
            assert rev2.status == "building"

    def test_status_output_contains_new_fields(
        self, temp_dir: Path, pdf_fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify that load_pipeline_status includes all enhanced fields."""
        config = _config_for(temp_dir)
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)

        # Patch status_service.get_engine to use our SQLite engine
        monkeypatch.setattr(
            "cementic.status_service.get_engine", lambda url: engine,
        )

        collection = "status-test"
        pdf_path = pdf_fixtures_dir / "test_doc_b.pdf"

        source_watcher = SourceWatcher(config)
        source_watcher.Session = session_factory
        source_watcher.collection = collection
        monkeypatch.setattr(source_watcher.state_manager, "update", lambda **kwargs: None)

        pipeline = PipelineWorker(config)
        pipeline.Session = session_factory
        pipeline.collection = collection
        pipeline.embedding_client = FakeEmbeddingClient()
        monkeypatch.setattr(pipeline.state_manager, "update", lambda **kwargs: None)

        source_watcher._register_pdf(str(pdf_path))
        rev_id = pipeline._ensure_target_revision()
        _run_pipeline_until_idle(pipeline, rev_id)

        from cementic.status_service import load_pipeline_status

        status = load_pipeline_status(config, collection)

        assert status.documents == 1
        assert status.extracted_done == 1
        assert status.extracted_failed == 0
        assert status.chunked_done == 1
        assert status.chunked_failed == 0
        assert status.total_chunks > 0
        assert status.done_embeddings == status.total_chunks
        assert status.failed_embeddings == 0
        assert status.extraction_pct == 100.0
        assert status.chunking_pct == 100.0
        assert status.embedding_pct == 100.0
        assert status.active_revision_label != ""
        assert status.building_revision_label != ""

    def test_file_progress_for_verbose(
        self, temp_dir: Path, pdf_fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify that load_file_progress returns per-file breakdown."""
        config = _config_for(temp_dir)
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)

        # Patch status_service.get_engine to use our SQLite engine
        monkeypatch.setattr(
            "cementic.status_service.get_engine", lambda url: engine,
        )

        collection = "verbose-test"
        pdf_a = pdf_fixtures_dir / "test_doc_a.pdf"
        pdf_b = pdf_fixtures_dir / "test_doc_b.pdf"

        source_watcher = SourceWatcher(config)
        source_watcher.Session = session_factory
        source_watcher.collection = collection
        monkeypatch.setattr(source_watcher.state_manager, "update", lambda **kwargs: None)

        pipeline = PipelineWorker(config)
        pipeline.Session = session_factory
        pipeline.collection = collection
        pipeline.embedding_client = FakeEmbeddingClient()
        monkeypatch.setattr(pipeline.state_manager, "update", lambda **kwargs: None)

        # Register both PDFs
        source_watcher._register_pdf(str(pdf_a))
        source_watcher._register_pdf(str(pdf_b))

        rev_id = pipeline._ensure_target_revision()
        _run_pipeline_until_idle(pipeline, rev_id)

        from cementic.status_service import load_file_progress

        files = load_file_progress(config, collection)

        assert len(files) == 2
        paths = {f.source_path for f in files}
        assert str(pdf_a) in paths
        assert str(pdf_b) in paths

        for f in files:
            assert f.extraction_status == "done"
            assert f.chunking_status == "done"
            assert f.embeddings_total > 0
            assert f.embeddings_done == f.embeddings_total
            assert f.embeddings_failed == 0
            assert f.error_message is None
