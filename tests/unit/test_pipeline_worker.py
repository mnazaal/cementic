"""Tests for pipeline worker constructor, client creation, and lifecycle."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from cementic.config import Config
from cementic.db import (
    Base,
    Chunk,
    ChunkedDocument,
    ChunkEmbedding,
    ChunkProfile,
    EmbeddingProfile,
    ExtractedDocument,
    ExtractorProfile,
    PipelineRevision,
    SourceDocument,
)
from cementic.pipeline_worker import (
    PipelineCounts,
    PipelineWorker,
    _pipeline_worker_lock_key,
    _revision_is_complete,
    _try_acquire_pipeline_worker_lock,
)


def _count_queries(engine):
    """Context manager yielding a mutable counter of SELECT statements issued."""

    class _Counter:
        value = 0

    counter = _Counter()

    def _on_execute(conn, cursor, statement, *args, **kwargs):
        if statement.lstrip().upper().startswith("SELECT"):
            counter.value += 1

    event.listen(engine, "before_cursor_execute", _on_execute)
    return counter


class _ShortBatchClient:
    def format_document(self, text: str) -> str:
        return text

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        return [[1.0, 0.0]]


def _seed_embedding_batch(session) -> int:
    source = SourceDocument(collection="default", source_path="/tmp/a.txt", file_hash="h")
    extractor = ExtractorProfile(name="x", fingerprint="x", config_json="{}")
    chunk_profile = ChunkProfile(fingerprint="c", config_json="{}")
    embedding_profile = EmbeddingProfile(
        fingerprint="e",
        provider="test",
        model_identifier="test",
        embedding_dim=2,
        distance_metric="cosine",
        config_json="{}",
    )
    session.add_all([source, extractor, chunk_profile, embedding_profile])
    session.flush()
    extracted = ExtractedDocument(
        document_id=source.id,
        extractor_profile_id=extractor.id,
        status="done",
    )
    session.add(extracted)
    session.flush()
    chunked = ChunkedDocument(
        extracted_document_id=extracted.id,
        chunk_profile_id=chunk_profile.id,
        status="done",
    )
    session.add(chunked)
    session.flush()
    for index in range(2):
        session.add(
            Chunk(
                document_id=source.id,
                chunked_document_id=chunked.id,
                chunk_index=index,
                content=f"chunk {index}",
            )
        )
    session.flush()
    revision = PipelineRevision(
        collection="default",
        extractor_profile_id=extractor.id,
        chunk_profile_id=chunk_profile.id,
        embedding_profile_id=embedding_profile.id,
        status="building",
    )
    session.add(revision)
    session.commit()
    return revision.id


class TestPipelineWorkerConstructor:
    """Tests for PipelineWorker.__init__ and basic properties."""

    def test_default_constructs_with_config(self) -> None:
        config = Config()
        worker = PipelineWorker(config)
        assert worker.config is config
        assert worker.collection == "default"
        assert worker.embedding_client is None

    def test_log_file_required(self) -> None:
        config = Config()
        config.pipeline_worker.log_file = None
        with pytest.raises(RuntimeError, match="not configured"):
            PipelineWorker(config)

    def test_setup_logging_creates_handler(self, temp_dir: Path) -> None:
        config = Config()
        log_file = temp_dir / "worker.log"
        config.pipeline_worker.log_file = log_file
        worker = PipelineWorker(config)
        assert log_file.exists()
        assert worker._logger is not None

    def test_setup_logging_reuses_existing_file_handler(self, temp_dir: Path) -> None:
        """Repeated construction should not duplicate file handlers."""
        config = Config()
        log_file = temp_dir / "worker.log"
        config.pipeline_worker.log_file = log_file

        first = PipelineWorker(config)
        second = PipelineWorker(config)

        handlers = [
            handler for handler in second._logger.handlers
            if getattr(handler, "baseFilename", None) == str(log_file)
        ]
        assert first._logger is second._logger
        assert len(handlers) == 1


class TestCreateEmbeddingClient:
    """Tests for _create_embedding_client."""

    def test_llama_cpp_provider_uses_shared_runtime(self, temp_dir: Path) -> None:
        config = Config()
        config.pipeline_worker.log_file = temp_dir / "worker.log"
        config.pipeline.embedding_provider = "llama-cpp"
        worker = PipelineWorker(config)

        with patch("cementic.pipeline_worker.create_provider") as mock_create:
            sentinel = object()
            mock_create.return_value = sentinel

            client = worker._create_embedding_client()
            assert client is sentinel
            spec, passed_config = mock_create.call_args[0]
            assert passed_config is config
            assert spec.provider == "llama-cpp"
            assert spec.model_identifier == config.llama_cpp.model_path


    def test_unknown_provider_raises(self, temp_dir: Path) -> None:
        config = Config()
        log_file = temp_dir / "worker.log"
        config.pipeline_worker.log_file = log_file
        config.pipeline.embedding_provider = "invalid"
        worker = PipelineWorker(config)

        with pytest.raises(ValueError, match="Unknown embedding provider"):
            worker._create_embedding_client()


class TestPipelineWorkerStop:
    """Tests for PipelineWorker.stop()."""

    def test_stop_updates_state(self, temp_dir: Path) -> None:
        config = Config()
        log_file = temp_dir / "worker.log"
        state_file = temp_dir / "worker_state.json"
        config.pipeline_worker.log_file = log_file
        config.pipeline_worker.state_path = state_file
        worker = PipelineWorker(config)

        worker.stop()

        # Verify state file was updated to STOPPED
        assert state_file.exists()
        import json
        state = json.loads(state_file.read_text())
        assert state["daemon_state"] == "stopped"
        assert state["pid"] is None


class TestRevisionIsComplete:
    """Tests for the pure _revision_is_complete function."""

    def test_all_stages_match(self) -> None:
        counts = PipelineCounts(
            documents=3, extracted_done=3, chunked_done=3, total_chunks=9, done_embeddings=9
        )
        assert _revision_is_complete(counts) is True

    def test_extracted_fewer_than_documents(self) -> None:
        counts = PipelineCounts(
            documents=3, extracted_done=2, chunked_done=2, total_chunks=6, done_embeddings=6
        )
        assert _revision_is_complete(counts) is False

    def test_chunked_fewer_than_extracted(self) -> None:
        counts = PipelineCounts(
            documents=3, extracted_done=3, chunked_done=2, total_chunks=6, done_embeddings=6
        )
        assert _revision_is_complete(counts) is False

    def test_embeddings_fewer_than_total_chunks(self) -> None:
        counts = PipelineCounts(
            documents=3, extracted_done=3, chunked_done=3, total_chunks=9, done_embeddings=8
        )
        assert _revision_is_complete(counts) is False

    def test_zero_documents_complete(self) -> None:
        counts = PipelineCounts(
            documents=0, extracted_done=0, chunked_done=0, total_chunks=0, done_embeddings=0
        )
        assert _revision_is_complete(counts) is True

    def test_failed_extraction_counts_as_terminal(self) -> None:
        """A corrupt document should not block an otherwise terminal revision forever."""
        counts = PipelineCounts(
            documents=3,
            extracted_done=2,
            extracted_failed=1,
            chunked_done=2,
            total_chunks=6,
            done_embeddings=6,
        )
        assert _revision_is_complete(counts) is True

    def test_failed_embedding_counts_as_terminal(self) -> None:
        """A failed chunk embedding is terminal for readiness."""
        counts = PipelineCounts(
            documents=1,
            extracted_done=1,
            chunked_done=1,
            total_chunks=2,
            done_embeddings=1,
            failed_embeddings=1,
        )
        assert _revision_is_complete(counts) is True


class TestPipelineWorkerStart:
    """Tests for PipelineWorker.start() edge branches."""

    def test_advisory_lock_key_is_stable_per_collection(self) -> None:
        assert _pipeline_worker_lock_key("default") == _pipeline_worker_lock_key("default")
        assert _pipeline_worker_lock_key("default") != _pipeline_worker_lock_key("other")

    def test_postgres_lock_failure_closes_connection(self) -> None:
        engine = Mock()
        engine.dialect.name = "postgresql"
        connection = Mock()
        connection.execute.return_value.scalar.return_value = False
        engine.connect.return_value = connection

        assert _try_acquire_pipeline_worker_lock(engine, "default") is None
        connection.close.assert_called_once_with()

    def test_postgres_lock_success_returns_holding_connection(self) -> None:
        engine = Mock()
        engine.dialect.name = "postgresql"
        connection = Mock()
        connection.execute.return_value.scalar.return_value = True
        engine.connect.return_value = connection

        assert _try_acquire_pipeline_worker_lock(engine, "default") is connection
        connection.close.assert_not_called()

    def test_start_rejects_when_database_worker_lock_is_held(self, temp_dir: Path) -> None:
        config = Config()
        config.pipeline_worker.log_file = temp_dir / "worker.log"
        config.pipeline_worker.state_path = temp_dir / "state.json"
        worker = PipelineWorker(config)
        engine = Mock()
        engine.dialect.name = "postgresql"

        with (
            patch("cementic.pipeline_worker.get_engine", return_value=engine),
            patch("cementic.pipeline_worker.create_tables"),
            patch("cementic.pipeline_worker.get_session_factory"),
            patch("cementic.pipeline_worker._try_acquire_pipeline_worker_lock", return_value=None),
            patch.object(worker, "_create_embedding_client") as create_client,
        ):
            worker.start("default")

        create_client.assert_not_called()

    def test_already_running_rejects_duplicate(self, temp_dir: Path) -> None:
        """Worker that detects another running instance should log error and return early."""
        config = Config()
        state_file = temp_dir / "worker_state.json"
        config.pipeline_worker.log_file = temp_dir / "worker.log"
        config.pipeline_worker.state_path = state_file

        # Pre-populate state as RUNNING with a live PID
        import json
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(
            json.dumps({"daemon_state": "running", "pid": 1, "current_file": None})
        )

        worker = PipelineWorker(config)
        # os.kill(1, 0) will raise ProcessLookupError on most containers; patch it to succeed
        with patch("os.kill", return_value=None):
            worker.start("testcol")

        # Should have logged the error and NOT created an embedding client
        assert worker.embedding_client is None

    def test_health_check_failure_stops_start(self, temp_dir: Path) -> None:
        """Worker whose embedding provider fails health check should return early."""
        config = Config()
        config.pipeline_worker.log_file = temp_dir / "worker.log"
        config.pipeline_worker.state_path = temp_dir / "state.json"

        worker = PipelineWorker(config)
        mock_client = type("_", (), {})()
        mock_client.health_check = lambda: False

        with patch.object(worker, "_create_embedding_client", return_value=mock_client):
            with patch("cementic.pipeline_worker.get_engine"):
                with patch("cementic.pipeline_worker.create_tables"):
                    with patch("cementic.pipeline_worker.get_session_factory"):
                        worker.start("testcol")

        assert worker.embedding_client is mock_client

    def test_exception_during_client_init_stops_start(self, temp_dir: Path) -> None:
        """Worker whose embedding client init raises should log error and return early."""
        config = Config()
        config.pipeline_worker.log_file = temp_dir / "worker.log"
        config.pipeline_worker.state_path = temp_dir / "state.json"

        worker = PipelineWorker(config)

        with patch.object(
            worker, "_create_embedding_client", side_effect=RuntimeError("boom")
        ):
            with patch("cementic.pipeline_worker.get_engine"):
                with patch("cementic.pipeline_worker.create_tables"):
                    with patch("cementic.pipeline_worker.get_session_factory"):
                        worker.start("testcol")


class TestWorkerProcessingLoop:
    """Tests for PipelineWorker._run_processing_loop()."""

    def test_loop_iterates_and_exits_on_shutdown(self, temp_dir: Path) -> None:
        config = Config()
        config.pipeline_worker.log_file = temp_dir / "worker.log"
        worker = PipelineWorker(config)

        with (
            patch.object(worker, "_step_extract", return_value=False) as mock_extract,
            patch.object(worker, "_step_chunk", return_value=False) as mock_chunk,
            patch.object(worker, "_step_embed", return_value=False) as mock_embed,
            patch.object(worker, "_mark_revision_ready_if_complete") as mock_mark,
            patch.object(worker._shutdown_event, "wait") as mock_wait,
        ):
            # The idle wait ends the run, so the loop makes exactly one pass.
            mock_wait.side_effect = lambda _: worker._shutdown_event.set()
            worker._run_processing_loop(42)

        mock_extract.assert_called_once_with(42)
        mock_chunk.assert_called_once_with(42)
        mock_embed.assert_called_once_with(42)
        mock_mark.assert_called_once_with(42)
        mock_wait.assert_called_once()

    def test_loop_skips_when_extract_does_work(self, temp_dir: Path) -> None:
        """When _step_extract returns True the loop continues immediately."""
        config = Config()
        config.pipeline_worker.log_file = temp_dir / "worker.log"
        worker = PipelineWorker(config)

        def extract_once(_revision_id: int) -> bool:
            worker._shutdown_event.set()
            return True

        with (
            patch.object(worker, "_step_extract", side_effect=extract_once),
            patch.object(worker, "_step_chunk") as mock_chunk,
            patch.object(worker, "_step_embed") as mock_embed,
            patch.object(worker, "_mark_revision_ready_if_complete") as mock_mark,
            patch.object(worker._shutdown_event, "wait") as mock_wait,
        ):
            worker._run_processing_loop(1)

        # When extract does work, the loop continues immediately and skips the rest.
        mock_chunk.assert_not_called()
        mock_embed.assert_not_called()
        mock_mark.assert_not_called()
        mock_wait.assert_not_called()

    def test_loop_survives_a_failing_step(self, temp_dir: Path) -> None:
        """A transient error (e.g. Postgres restart) must not kill the worker."""
        config = Config()
        config.pipeline_worker.log_file = temp_dir / "worker.log"
        worker = PipelineWorker(config)
        calls: list[int] = []

        def flaky_extract(_revision_id: int) -> bool:
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("server closed the connection unexpectedly")
            worker._shutdown_event.set()
            return False

        with (
            patch.object(worker, "_step_extract", side_effect=flaky_extract),
            patch.object(worker, "_step_chunk", return_value=False),
            patch.object(worker, "_step_embed", return_value=False),
            patch.object(worker, "_mark_revision_ready_if_complete"),
            patch.object(worker._shutdown_event, "wait"),
        ):
            worker._run_processing_loop(7)

        # Retried after the failure rather than exiting on the first exception.
        assert len(calls) == 2


class TestPipelineWorkerEmbedStep:
    """Embedding step liveness regressions."""

    def test_short_embedding_batch_marks_every_claimed_row_failed(self, temp_dir: Path) -> None:
        """A short provider response must not leave trailing rows stuck processing."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        with session_factory() as session:
            revision_id = _seed_embedding_batch(session)

        config = Config()
        config.pipeline_worker.log_file = temp_dir / "worker.log"
        config.pipeline_worker.batch_size = 2
        worker = PipelineWorker(config)
        worker.Session = session_factory
        worker.embedding_client = _ShortBatchClient()

        assert worker._step_embed(revision_id) is True

        with session_factory() as session:
            rows = session.query(ChunkEmbedding).order_by(ChunkEmbedding.chunk_id).all()
            assert [row.status for row in rows] == ["failed", "failed"]
            assert all("returned 1 embeddings for 2 chunks" in row.error_message for row in rows)


def _seed_extract_selection_data(session, collection: str, n_already_done: int) -> tuple[int, int]:
    """Seed N already-extracted documents plus one pending; return (revision_id, pending_id)."""
    extractor = ExtractorProfile(name="x", fingerprint=f"ext-{collection}", config_json="{}")
    chunk_profile = ChunkProfile(fingerprint=f"chunk-{collection}", config_json="{}")
    embedding_profile = EmbeddingProfile(
        fingerprint=f"embed-{collection}",
        provider="test",
        model_identifier="test",
        embedding_dim=2,
        distance_metric="cosine",
        config_json="{}",
    )
    session.add_all([extractor, chunk_profile, embedding_profile])
    session.flush()

    for index in range(n_already_done):
        doc = SourceDocument(
            collection=collection,
            source_path=f"/tmp/done_{index}.txt",
            file_hash=f"hash_{index}",
            status="indexed",
        )
        session.add(doc)
        session.flush()
        session.add(
            ExtractedDocument(
                document_id=doc.id,
                extractor_profile_id=extractor.id,
                source_file_hash=f"hash_{index}",
                status="done",
            )
        )

    pending = SourceDocument(
        collection=collection, source_path="/tmp/pending.txt", file_hash="pending-hash"
    )
    session.add(pending)
    session.flush()

    revision = PipelineRevision(
        collection=collection,
        extractor_profile_id=extractor.id,
        chunk_profile_id=chunk_profile.id,
        embedding_profile_id=embedding_profile.id,
        status="building",
    )
    session.add(revision)
    session.commit()
    return revision.id, pending.id


class TestStepExtractSelectionIsBounded:
    """Regression guard: candidate selection is a single query, not O(n) in doc count."""

    def _run_and_count_queries(self, temp_dir: Path, n_already_done: int) -> int:
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        collection = "bounded"
        with session_factory() as session:
            revision_id, pending_id = _seed_extract_selection_data(
                session, collection, n_already_done
            )

        config = Config()
        config.pipeline_worker.log_file = temp_dir / "worker.log"
        worker = PipelineWorker(config)
        worker.Session = session_factory
        worker.collection = collection

        counter = _count_queries(engine)
        with (
            patch("cementic.pipeline_worker.extract_document", return_value="content"),
            patch("cementic.pipeline_worker.write_extracted_text", return_value="content-hash"),
            patch.object(worker.state_manager, "update"),
        ):
            worked = worker._step_extract(revision_id)

        assert worked is True
        with session_factory() as session:
            extracted = session.query(ExtractedDocument).filter_by(document_id=pending_id).first()
            assert extracted is not None
            assert extracted.status == "done"

        return counter.value

    def test_query_count_does_not_scale_with_document_count(self, temp_dir: Path) -> None:
        small = self._run_and_count_queries(temp_dir, n_already_done=2)
        large = self._run_and_count_queries(temp_dir, n_already_done=50)
        assert small == large


def _seed_chunk_selection_data(session, collection: str, n_already_done: int) -> tuple[int, int]:
    """Seed N already-chunked documents plus one pending; return (revision_id, pending_id)."""
    extractor = ExtractorProfile(name="x", fingerprint=f"ext-{collection}", config_json="{}")
    chunk_profile = ChunkProfile(fingerprint=f"chunk-{collection}", config_json="{}")
    embedding_profile = EmbeddingProfile(
        fingerprint=f"embed-{collection}",
        provider="test",
        model_identifier="test",
        embedding_dim=2,
        distance_metric="cosine",
        config_json="{}",
    )
    session.add_all([extractor, chunk_profile, embedding_profile])
    session.flush()

    def _add_extracted(index: int, status: str) -> ExtractedDocument:
        doc = SourceDocument(
            collection=collection, source_path=f"/tmp/doc_{index}.txt", file_hash=f"hash_{index}"
        )
        session.add(doc)
        session.flush()
        extracted = ExtractedDocument(
            document_id=doc.id,
            extractor_profile_id=extractor.id,
            source_file_hash=f"hash_{index}",
            content_hash=f"content_{index}",
            artifact_path=f"/tmp/artifact_{index}.md.gz",
            status=status,
        )
        session.add(extracted)
        session.flush()
        return extracted

    for index in range(n_already_done):
        extracted = _add_extracted(index, "done")
        session.add(
            ChunkedDocument(
                extracted_document_id=extracted.id,
                chunk_profile_id=chunk_profile.id,
                source_content_hash=f"content_{index}",
                status="done",
            )
        )

    pending_extracted = _add_extracted(n_already_done, "done")

    revision = PipelineRevision(
        collection=collection,
        extractor_profile_id=extractor.id,
        chunk_profile_id=chunk_profile.id,
        embedding_profile_id=embedding_profile.id,
        status="building",
    )
    session.add(revision)
    session.commit()
    return revision.id, pending_extracted.id


class TestStepChunkSelectionIsBounded:
    """Regression guard: candidate selection is a single query, not O(n) in doc count."""

    def _run_and_count_queries(self, temp_dir: Path, n_already_done: int) -> int:
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        collection = "bounded"
        with session_factory() as session:
            revision_id, pending_extracted_id = _seed_chunk_selection_data(
                session, collection, n_already_done
            )

        config = Config()
        config.pipeline_worker.log_file = temp_dir / "worker.log"
        config.pipeline.chunk_size = 64
        config.pipeline.chunk_overlap = 16
        worker = PipelineWorker(config)
        worker.Session = session_factory
        worker.collection = collection

        counter = _count_queries(engine)
        with (
            patch("cementic.pipeline_worker.read_extracted_text", return_value="some text"),
            patch.object(worker.state_manager, "update"),
        ):
            worked = worker._step_chunk(revision_id)

        assert worked is True
        with session_factory() as session:
            chunked = (
                session.query(ChunkedDocument)
                .filter_by(extracted_document_id=pending_extracted_id)
                .first()
            )
            assert chunked is not None
            assert chunked.status == "done"

        return counter.value

    def test_query_count_does_not_scale_with_document_count(self, temp_dir: Path) -> None:
        small = self._run_and_count_queries(temp_dir, n_already_done=2)
        large = self._run_and_count_queries(temp_dir, n_already_done=50)
        assert small == large
