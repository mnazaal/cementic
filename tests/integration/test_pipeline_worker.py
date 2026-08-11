"""Integration tests for PipelineWorker error paths and lifecycle.

Runs on SQLite with FakeEmbeddingClient — no PostgreSQL needed.
Exercises extraction, chunking, embedding error paths and daemon lifecycle.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from cementic import pipeline_worker as pipeline_worker_module
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
from cementic.pipeline_worker import (
    PipelineWorker,
    _pipeline_worker_lock_key,
    _release_pipeline_worker_lock,
    _try_acquire_pipeline_worker_lock,
    compute_revision_counts,
)
from cementic.revisions import promote_revision, requeue_interrupted_artifacts
from cementic.source_watcher import SourceWatcher
from tests.integration.test_pg_helpers import cleanup_pg_tables


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
    """Embedding client whose texts cannot be embedded at all.

    Fails single-text embedding as well as the batch: a batch failure alone is
    now retried one text at a time, so overriding only ``embed_batch`` would
    describe a *recoverable* batch, not unembeddable data.
    """

    def embed(self, text: str) -> list[float]:
        raise RuntimeError("injected embedding failure")

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        raise RuntimeError("injected embedding failure")


class BatchOnlyFailureClient(FakeEmbeddingClient):
    """Batch calls fail, individual calls succeed.

    Models an all-or-nothing batch endpoint rejecting one oversized input: the
    other chunks in the batch are perfectly embeddable.
    """

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        raise RuntimeError("batch rejected")


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

    def test_start_aborts_when_the_provider_cannot_embed(self, temp_dir: Path) -> None:
        """The startup gate checks capability, not just identity.

        It used to call health_check(), which only asks whether the daemon
        *lists* the expected model -- so one that answered /v1/models but failed
        every embed passed the gate and then failed every batch retryably,
        leaving the worker looping.
        """
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
            # Patch the instance, not the class: assigning to
            # type(fake_client) leaked into every later test using this fake.
            with patch.object(
                fake_client, "describe", side_effect=RuntimeError("embedding=True not set")
            ):
                mock_create.return_value = fake_client
                worker.start("test_health")

            mock_loop.assert_not_called()
        assert worker.fatal_reason is not None and "cannot embed" in worker.fatal_reason

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

    @pytest.mark.pg
    def test_batch_failure_falls_back_to_embedding_chunks_individually(
        self, pg_setup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One unembeddable chunk must not fail its whole batch.

        Regression: a batch call is all-or-nothing, so a single bad input marked
        all batch_size (default 32) chunks permanently failed -- which then
        blocks `collection promote`, and forcing past it silently drops them
        from the published index.
        """
        config, session_factory, pdf_fixtures_dir = pg_setup
        collection = "test_batch_isolation"

        pipeline, source_watcher = _setup_worker(config, session_factory, collection, monkeypatch)
        source_watcher._register_document(str(pdf_fixtures_dir / "test_doc_a.pdf"))
        pipeline.embedding_client = FakeEmbeddingClient()
        revision = pipeline._ensure_target_revision()
        pipeline._step_extract(revision)
        pipeline._step_chunk(revision)

        pipeline.embedding_client = BatchOnlyFailureClient()
        pipeline._step_embed(revision)

        with session_factory() as session:
            done = session.query(ChunkEmbedding).filter_by(status="done").count()
            failed = session.query(ChunkEmbedding).filter_by(status="failed").count()
        assert done > 0, "individually-embeddable chunks must still be saved"
        assert failed == 0


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

    @pytest.mark.pg
    def test_revision_with_zero_vectors_reaches_ready_on_postgres(
        self, pg_setup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A build that embeds nothing still reaches ``ready`` against real Postgres.

        Regression: the per-profile vector table was only created on the first
        successful embedding, but the ANN index was built unconditionally once the
        revision completed. With every document failing extraction there were no
        vectors, so `CREATE INDEX` raised UndefinedTable; the worker's retry loop
        swallowed it and the revision stayed `building` forever, with
        `cementic collection promote` reporting "no ready revision" indefinitely.
        This only reproduces on Postgres -- SQLite has no ANN index to build.
        """
        config, session_factory, pdf_fixtures_dir = pg_setup
        collection = "test_zero_vectors"

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
            assert revision.status == "ready"

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


class TestRevisionCompletionScoping:
    """A revision must settle even when it shares profiles with an older one.

    These are the wedge bugs: the completeness check compares counts drawn from
    differently-scoped queries, so a revision that has actually finished never
    satisfies the equality and the worker polls forever without ever reaching
    ``ready`` (so `collection promote` reports "no ready revision").
    """

    @pytest.mark.pg
    def test_failed_vector_writeback_does_not_strand_claimed_embeddings(
        self, pg_setup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A crash between claiming a batch and writing it back must not wedge.

        Regression: rows were committed as `processing` in one transaction and
        written back in another, and the embed step's candidate filter matched
        only `pending` -- unlike the extract and chunk steps, which re-pick their
        own `processing` rows. So one failed write-back stranded the batch for
        the life of the process: `done + failed` never reached `total_chunks`,
        the revision never completed, and the worker kept polling while looking
        perfectly healthy to `cementic status`.
        """
        config, session_factory, pdf_fixtures_dir = pg_setup
        collection = "test_writeback_failure"

        pipeline, source_watcher = _setup_worker(config, session_factory, collection, monkeypatch)
        pipeline.embedding_client = FakeEmbeddingClient()
        source_watcher._register_document(str(pdf_fixtures_dir / "test_doc_a.pdf"))
        revision_id = pipeline._ensure_target_revision()

        pipeline._step_extract(revision_id)
        pipeline._step_chunk(revision_id)

        # Fail the vector write-back exactly once, after the batch is claimed.
        calls = {"n": 0}
        real_upsert = pipeline_worker_module.upsert_vectors

        def flaky_upsert(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("injected write-back failure")
            return real_upsert(*args, **kwargs)

        monkeypatch.setattr(pipeline_worker_module, "upsert_vectors", flaky_upsert)
        with pytest.raises(RuntimeError, match="injected write-back failure"):
            pipeline._step_embed(revision_id)

        with session_factory() as session:
            stranded = (
                session.query(ChunkEmbedding).filter_by(status="processing").count()
            )
        assert stranded == 0, "claimed rows must be released, not left processing"

        monkeypatch.setattr(pipeline_worker_module, "upsert_vectors", real_upsert)
        _run_pipeline_until_idle(pipeline, revision_id)
        with session_factory() as session:
            revision = session.get(PipelineRevision, revision_id)
            assert revision is not None
            assert revision.status == "ready"

    def test_chunk_size_change_reaching_ready_with_shared_embedding_profile(
        self, pg_setup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Change chunking only: the new revision reuses the embedding profile.

        Regression: embedding counts were scoped by embedding profile alone, so
        the *old* revision's done embeddings were counted against the new
        revision's chunk total and `done == total_chunks` was never true.
        """
        config, session_factory, pdf_fixtures_dir = pg_setup
        collection = "test_chunk_change"

        pipeline, source_watcher = _setup_worker(config, session_factory, collection, monkeypatch)
        pipeline.embedding_client = FakeEmbeddingClient()
        source_watcher._register_document(str(pdf_fixtures_dir / "test_doc_a.pdf"))

        first_id = pipeline._ensure_target_revision()
        _run_pipeline_until_idle(pipeline, first_id)
        with session_factory() as session:
            first = session.get(PipelineRevision, first_id)
            assert first is not None and first.status == "ready"
            # Promote so the old revision stays active (its chunks/embeddings
            # are kept rather than pruned) -- the situation that triggers this.
            promote_revision(session, collection, first, config=config)
            session.commit()

        # Same embedding model, different chunking.
        config.pipeline.chunk_size = 32
        config.pipeline.chunk_overlap = 8
        second_id = pipeline._ensure_target_revision()
        assert second_id != first_id

        _run_pipeline_until_idle(pipeline, second_id)

        with session_factory() as session:
            second = session.get(PipelineRevision, second_id)
            assert second is not None
            assert second.embedding_profile_id == first.embedding_profile_id
            assert second.status == "ready"

    def test_chunk_failure_after_reextraction_still_reaches_ready(
        self, sqlite_setup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed chunking is counted against the extraction it belongs to.

        Regression: ``chunked_failed`` was not scoped by ``source_content_hash``
        while ``chunked_done`` was, so a failed chunking could never satisfy
        ``chunked_done + chunked_failed == extracted_done``.
        """
        config, session_factory, pdf_fixtures_dir = sqlite_setup
        collection = "test_chunk_failure"

        pipeline, source_watcher = _setup_worker(config, session_factory, collection, monkeypatch)
        pipeline.embedding_client = FakeEmbeddingClient()
        source_watcher._register_document(str(pdf_fixtures_dir / "test_doc_a.pdf"))

        revision_id = pipeline._ensure_target_revision()
        pipeline._step_extract(revision_id)

        # Break the artifact so chunking fails for this extraction.
        with session_factory() as session:
            extracted = session.query(ExtractedDocument).filter_by(status="done").one()
            extracted.artifact_path = "/nonexistent/artifact.md.gz"
            session.commit()

        _run_pipeline_until_idle(pipeline, revision_id)

        with session_factory() as session:
            revision = session.get(PipelineRevision, revision_id)
            assert revision is not None
            chunked = session.query(ChunkedDocument).one()
            assert chunked.status == "failed"
            # The failure is terminal, and the build still settles.
            assert revision.status == "ready"

    def test_revision_counts_ignore_other_revisions_embeddings(
        self, pg_setup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """compute_revision_counts must not count out-of-chain embeddings."""
        config, session_factory, pdf_fixtures_dir = pg_setup
        collection = "test_counts_scope"

        pipeline, source_watcher = _setup_worker(config, session_factory, collection, monkeypatch)
        pipeline.embedding_client = FakeEmbeddingClient()
        source_watcher._register_document(str(pdf_fixtures_dir / "test_doc_a.pdf"))

        first_id = pipeline._ensure_target_revision()
        _run_pipeline_until_idle(pipeline, first_id)
        with session_factory() as session:
            first = session.get(PipelineRevision, first_id)
            promote_revision(session, collection, first, config=config)
            session.commit()

        config.pipeline.chunk_size = 32
        config.pipeline.chunk_overlap = 8
        second_id = pipeline._ensure_target_revision()
        _run_pipeline_until_idle(pipeline, second_id)

        with session_factory() as session:
            second = session.get(PipelineRevision, second_id)
            counts = compute_revision_counts(session, collection, second)
            # Every count belongs to this revision's own chunk chain.
            assert counts.done_embeddings == counts.total_chunks
            assert counts.total_chunks > 0


class TestRevisionRollback:
    """Reverting config to a previously built revision must rebuild/republish it."""

    def test_reverting_config_resurrects_retired_revision(
        self, pg_setup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: a reverted-to retired revision stayed 'retired' forever.

        ``mark_revision_ready`` only promotes ``building`` revisions, so the
        pipeline had nothing to do and nothing to promote -- silently stuck on
        the newer revision the user had just configured away from.
        """
        config, session_factory, pdf_fixtures_dir = pg_setup
        collection = "test_rollback"

        pipeline, source_watcher = _setup_worker(config, session_factory, collection, monkeypatch)
        pipeline.embedding_client = FakeEmbeddingClient()
        source_watcher._register_document(str(pdf_fixtures_dir / "test_doc_a.pdf"))

        original_chunk_size = config.pipeline.chunk_size
        first_id = pipeline._ensure_target_revision()
        _run_pipeline_until_idle(pipeline, first_id)
        with session_factory() as session:
            first = session.get(PipelineRevision, first_id)
            promote_revision(session, collection, first, config=config)
            session.commit()

        # Move forward, then promote, retiring the first revision.
        config.pipeline.chunk_size = 32
        config.pipeline.chunk_overlap = 8
        second_id = pipeline._ensure_target_revision()
        _run_pipeline_until_idle(pipeline, second_id)
        with session_factory() as session:
            second = session.get(PipelineRevision, second_id)
            promote_revision(session, collection, second, config=config)
            session.commit()
            first = session.get(PipelineRevision, first_id)
            assert first is not None and first.status == "retired"

        # Roll the config back to what the first revision was built with.
        config.pipeline.chunk_size = original_chunk_size
        config.pipeline.chunk_overlap = 16
        reverted_id = pipeline._ensure_target_revision()
        assert reverted_id == first_id

        with session_factory() as session:
            reverted = session.get(PipelineRevision, reverted_id)
            assert reverted is not None
            assert reverted.status == "building"  # picked back up, not stuck retired

        _run_pipeline_until_idle(pipeline, reverted_id)

        with session_factory() as session:
            reverted = session.get(PipelineRevision, reverted_id)
            assert reverted is not None
            assert reverted.status == "ready"  # promotable again


class TestDeletedDuringExtraction:
    """A file deleted mid-extraction must not come back as searchable."""

    def test_delete_during_extraction_is_not_overwritten(
        self, sqlite_setup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: _step_extract stamped SourceDocument.status='indexed' after
        extraction, overwriting the watcher's 'deleted' and permanently
        resurrecting a removed file into search results."""
        config, session_factory, pdf_fixtures_dir = sqlite_setup
        collection = "test_delete_race"

        pipeline, source_watcher = _setup_worker(config, session_factory, collection, monkeypatch)
        pipeline.embedding_client = FakeEmbeddingClient()
        pdf_path = str(pdf_fixtures_dir / "test_doc_a.pdf")
        source_watcher._register_document(pdf_path)

        revision_id = pipeline._ensure_target_revision()

        # Simulate the watcher observing the deletion while extraction runs.
        real_extract_document = pipeline_worker_module.extract_document

        def extract_then_delete(path: str, cfg) -> str:
            content = real_extract_document(path, cfg)
            with session_factory() as session:
                doc = session.query(SourceDocument).filter_by(collection=collection).one()
                doc.status = "deleted"
                doc.file_hash = None
                session.commit()
            return content

        monkeypatch.setattr(
            "cementic.pipeline_worker.extract_document", extract_then_delete
        )
        pipeline._step_extract(revision_id)

        with session_factory() as session:
            doc = session.query(SourceDocument).filter_by(collection=collection).one()
            assert doc.status == "deleted"


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
        pipeline._run_processing_loop(pipeline._ensure_target_revision())

    def test_stop_writes_stopped_state(self, temp_dir: Path) -> None:
        config = _config_for(temp_dir)
        worker = PipelineWorker(config)
        worker.state_manager.update(daemon_state="running", pid=1234)
        worker.stop()

        from cementic.state import DaemonState
        state = worker.state_manager.load()
        assert state.daemon_state == DaemonState.STOPPED
        assert state.pid is None

    def test_handle_shutdown_only_sets_the_flag(self, temp_dir: Path) -> None:
        """The signal handler must not call stop().

        stop() takes StateManager's lock; a signal delivered while the main
        thread already held it deadlocked the process, and since the handler was
        the SIGTERM handler, only SIGKILL could recover. start()'s `finally`
        performs the cleanup instead.
        """
        config = _config_for(temp_dir)
        worker = PipelineWorker(config)
        worker.embedding_client = FakeEmbeddingClient()

        with patch.object(worker, "stop") as mock_stop:
            worker._handle_shutdown(15, None)
            mock_stop.assert_not_called()
        assert worker._shutdown_event.is_set()

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


@pytest.mark.pg
def test_advisory_lock_is_released_not_just_returned_to_the_pool(pg_engine) -> None:
    """Closing the connection does not release a session advisory lock.

    Connection.close() returns the connection to the pool; a pg_advisory_lock is
    bound to the backend session and survives it. The lock did go away when the
    worker process exited, so this was harmless in practice -- but the teardown
    read as if it released, and the engine is cached process-wide, so a second
    in-process start() could get the same pooled backend and see its own lock.
    """
    key = _pipeline_worker_lock_key("lockprobe")

    def locks_held() -> int:
        with pg_engine.connect() as conn:
            return conn.execute(
                text(
                    "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' "
                    "AND ((classid::bigint << 32) | objid::bigint) = :key"
                ),
                {"key": key},
            ).scalar()

    connection = _try_acquire_pipeline_worker_lock(pg_engine, "lockprobe")
    assert connection is not None
    assert locks_held() == 1

    _release_pipeline_worker_lock(connection, "lockprobe")
    assert locks_held() == 0
