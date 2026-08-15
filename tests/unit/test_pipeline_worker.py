"""Tests for pipeline worker constructor, client creation, and lifecycle."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests
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
    _purge_all_chunks,
    _purge_superseded_chunks,
    _try_acquire_pipeline_worker_lock,
    is_retryable_embed_error,
    revision_is_complete,
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


class TestIsRetryableEmbedError:
    """Provider-side failures must not be recorded against the documents.

    Regression: only ConnectionError and Timeout counted as retryable, but
    raise_for_status() raises HTTPError -- so a momentary 503 from llama.cpp
    stamped every chunk in the batch permanently failed, which blocks
    `collection promote` and silently omits them when forced.
    """

    def _http_error(self, status: int) -> requests.exceptions.HTTPError:
        response = requests.Response()
        response.status_code = status
        return requests.exceptions.HTTPError(response=response)

    @pytest.mark.parametrize("status", [408, 425, 429, 500, 502, 503, 504])
    def test_server_side_statuses_are_retryable(self, status: int) -> None:
        assert is_retryable_embed_error(self._http_error(status)) is True

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 422])
    def test_request_side_statuses_are_terminal(self, status: int) -> None:
        """A 4xx about the request itself will not fix itself on retry."""
        assert is_retryable_embed_error(self._http_error(status)) is False

    def test_connection_and_timeout_remain_retryable(self) -> None:
        assert is_retryable_embed_error(requests.exceptions.ConnectionError()) is True
        assert is_retryable_embed_error(requests.exceptions.Timeout()) is True
        assert is_retryable_embed_error(requests.exceptions.ChunkedEncodingError()) is True

    def test_unrelated_errors_are_terminal(self) -> None:
        assert is_retryable_embed_error(ValueError("bad data")) is False
        assert is_retryable_embed_error(KeyError("embedding")) is False

    def test_http_error_without_a_response_is_terminal(self) -> None:
        """Defensive: HTTPError can be constructed without a response."""
        assert is_retryable_embed_error(requests.exceptions.HTTPError()) is False


class TestRevisionIsComplete:
    """Tests for the pure revision_is_complete function."""

    def test_all_stages_match(self) -> None:
        counts = PipelineCounts(
            documents=3, extracted_done=3, chunked_done=3, total_chunks=9, done_embeddings=9
        )
        assert revision_is_complete(counts) is True

    def test_extracted_fewer_than_documents(self) -> None:
        counts = PipelineCounts(
            documents=3, extracted_done=2, chunked_done=2, total_chunks=6, done_embeddings=6
        )
        assert revision_is_complete(counts) is False

    def test_chunked_fewer_than_extracted(self) -> None:
        counts = PipelineCounts(
            documents=3, extracted_done=3, chunked_done=2, total_chunks=6, done_embeddings=6
        )
        assert revision_is_complete(counts) is False

    def test_embeddings_fewer_than_total_chunks(self) -> None:
        counts = PipelineCounts(
            documents=3, extracted_done=3, chunked_done=3, total_chunks=9, done_embeddings=8
        )
        assert revision_is_complete(counts) is False

    def test_zero_documents_is_not_complete(self) -> None:
        """A revision with nothing in it must not be promotable.

        All-zero counts satisfy every stage equality, so treating them as
        complete marked a revision `ready` before the watcher had registered
        its first document -- the normal race on a fresh `cementic start`,
        since both workers spawn together. Promoting that published an empty
        index and reported zero failures doing it.
        """
        counts = PipelineCounts(
            documents=0, extracted_done=0, chunked_done=0, total_chunks=0, done_embeddings=0
        )
        assert revision_is_complete(counts) is False

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
        assert revision_is_complete(counts) is True

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
        assert revision_is_complete(counts) is True


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


class TestFatalStartupReasonsReachTheBackgroundLog:
    """Startup failures must land in the log `cementic start` names.

    Regression: these paths logged to the module logger's own file
    (pipeline_worker.log) and exited, while `cementic start` told the user to
    look in pipeline-background.log, which captures stdout/stderr only and was
    therefore empty.
    """

    def test_fatal_writes_to_stderr(self, temp_dir: Path, capsys) -> None:
        config = Config()
        config.pipeline_worker.log_file = temp_dir / "worker.log"
        worker = PipelineWorker(config)

        worker._fatal("Embedding provider health check failed")

        assert "Embedding provider health check failed" in capsys.readouterr().err

    def test_fatal_interpolates_arguments(self, temp_dir: Path, capsys) -> None:
        config = Config()
        config.pipeline_worker.log_file = temp_dir / "worker.log"
        worker = PipelineWorker(config)

        worker._fatal("Pipeline worker already running with PID %s", 4321)

        assert "already running with PID 4321" in capsys.readouterr().err


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

    def test_loop_publishes_failures_to_the_state_file(self, temp_dir: Path) -> None:
        """A step failure must reach `cementic status`, not just the log file.

        Regression: the retry handler only logged, so a worker looping forever on
        a permanent failure was indistinguishable from a healthy idle one.
        """
        config = Config()
        config.pipeline_worker.log_file = temp_dir / "worker.log"
        config.pipeline_worker.state_path = temp_dir / "worker-state.json"
        worker = PipelineWorker(config)

        with (
            patch.object(worker, "_step_extract", side_effect=RuntimeError("boom")),
            patch.object(worker._shutdown_event, "wait") as mock_wait,
        ):
            mock_wait.side_effect = lambda _: worker._shutdown_event.set()
            worker._run_processing_loop(1)

        state = worker.state_manager.load()
        assert state.last_error == "RuntimeError: boom"
        assert state.last_error_at is not None

    def test_start_clears_an_error_left_by_a_previous_run(self, temp_dir: Path) -> None:
        """A fresh worker must not inherit the last run's error.

        Regression: the in-loop clear only reset errors recorded by the same
        process, so a failure from an earlier worker was reported by
        `cementic status` forever and made a healthy worker look broken.
        """
        config = Config()
        config.pipeline_worker.log_file = temp_dir / "worker.log"
        config.pipeline_worker.state_path = temp_dir / "worker-state.json"
        worker = PipelineWorker(config)
        worker.state_manager.update(last_error="stale from a previous run", last_error_at="then")

        with (
            patch("cementic.pipeline_worker.get_engine") as mock_engine,
            patch("cementic.pipeline_worker.create_tables"),
            patch.object(worker, "_create_embedding_client") as mock_create,
            patch.object(worker, "_ensure_target_revision", return_value=1),
            patch.object(worker, "_run_processing_loop"),
        ):
            engine = create_engine("sqlite:///:memory:")
            mock_engine.return_value = engine
            Base.metadata.create_all(engine)
            client = MagicMock()
            client.health_check.return_value = True
            mock_create.return_value = client
            worker.start("clears_stale")

        state = worker.state_manager.load()
        assert state.last_error is None
        assert state.last_error_at is None

    def test_loop_clears_a_recorded_failure_after_a_clean_pass(self, temp_dir: Path) -> None:
        config = Config()
        config.pipeline_worker.log_file = temp_dir / "worker.log"
        config.pipeline_worker.state_path = temp_dir / "worker-state.json"
        worker = PipelineWorker(config)
        worker.state_manager.update(last_error="stale", last_error_at="then")

        calls = {"n": 0}

        def fail_then_succeed(_revision_id: int) -> bool:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient")
            return False

        with (
            patch.object(worker, "_step_extract", side_effect=fail_then_succeed),
            patch.object(worker, "_step_chunk", return_value=False),
            patch.object(worker, "_step_embed", return_value=False),
            patch.object(worker, "_mark_revision_ready_if_complete"),
            patch.object(worker._shutdown_event, "wait") as mock_wait,
        ):
            # First wait is the error backoff; the second ends the run.
            waits = {"n": 0}

            def wait(_timeout):
                waits["n"] += 1
                if waits["n"] >= 2:
                    worker._shutdown_event.set()

            mock_wait.side_effect = wait
            worker._run_processing_loop(1)

        state = worker.state_manager.load()
        assert state.last_error is None
        assert state.last_error_at is None

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


class TestProviderFailuresAreVisible:
    """A provider that is down must not look like an idle worker.

    Returning False from the release path read as "no work this pass", so the
    loop never recorded anything: `cementic status` showed the workers running
    with no last error while progress had simply stopped, and the same batch was
    re-claimed every poll forever.
    """

    def _worker(self, temp_dir: Path) -> PipelineWorker:
        config = Config()
        config.pipeline_worker.log_file = temp_dir / "worker.log"
        config.pipeline_worker.state_path = temp_dir / "state.json"
        return PipelineWorker(config)

    def test_release_after_provider_failure_re_raises(self, temp_dir: Path) -> None:
        worker = self._worker(temp_dir)
        error = requests.exceptions.ConnectionError("daemon refused")

        with (
            patch.object(worker, "_release_claimed_embeddings") as mock_release,
            patch.object(worker, "_create_embedding_client"),
            pytest.raises(requests.exceptions.ConnectionError),
        ):
            worker._release_after_provider_failure(error, [(1, "a"), (2, "b")], profile_id=7)

        # The claim is returned to `pending` before the raise, so the chunks are
        # retried rather than stranded in `processing`.
        mock_release.assert_called_once_with([1, 2], 7)

    def test_release_re_raises_even_when_the_reconnect_fails(self, temp_dir: Path) -> None:
        worker = self._worker(temp_dir)
        error = requests.exceptions.Timeout("gone")

        with (
            patch.object(worker, "_release_claimed_embeddings"),
            patch.object(worker, "_create_embedding_client", side_effect=OSError("no daemon")),
            pytest.raises(requests.exceptions.Timeout),
        ):
            worker._release_after_provider_failure(error, [(1, "a")], profile_id=1)

    def test_a_down_provider_reaches_the_state_file(self, temp_dir: Path) -> None:
        """End of the chain: the raise propagates to the loop, which publishes it."""
        worker = self._worker(temp_dir)

        with (
            patch.object(worker, "_step_extract", return_value=False),
            patch.object(worker, "_step_chunk", return_value=False),
            patch.object(
                worker,
                "_step_embed",
                side_effect=requests.exceptions.ConnectionError("daemon refused"),
            ),
            patch.object(worker._shutdown_event, "wait") as mock_wait,
        ):
            mock_wait.side_effect = lambda _: worker._shutdown_event.set()
            worker._run_processing_loop(1)

        state = worker.state_manager.load()
        assert state.last_error is not None
        assert "daemon refused" in state.last_error
        assert state.last_error_at is not None


class TestStartupGateChecksEmbedding:
    def test_startup_requires_a_working_embed_not_just_a_listing(self, temp_dir: Path) -> None:
        """health_check() only asks whether the daemon lists the model, so one
        that answers /v1/models but fails every embed passed the gate and then
        failed every batch -- retryably, so the worker looped on it."""
        config = Config()
        config.pipeline_worker.log_file = temp_dir / "worker.log"
        config.pipeline_worker.state_path = temp_dir / "state.json"
        worker = PipelineWorker(config)

        client = MagicMock()
        client.health_check.return_value = True
        client.describe.side_effect = RuntimeError("model was not loaded with embedding=True")

        with (
            patch("cementic.pipeline_worker.get_engine"),
            patch("cementic.pipeline_worker.create_tables"),
            patch("cementic.pipeline_worker.get_session_factory"),
            patch.object(worker, "_create_embedding_client", return_value=client),
        ):
            worker.start(collection="c")

        assert worker.fatal_reason is not None
        assert "cannot embed" in worker.fatal_reason
        client.describe.assert_called_once()


class TestSupersededChunksArePurged:
    """Re-extraction must remove the chunks describing the old text.

    Search no longer carries the freshness join -- that filter lived on a joined
    table and was one reason the planner could never use the ANN index -- so
    stale rows have to be deleted when they go stale rather than filtered out at
    query time. `_step_chunk` would delete them when it re-chunks, but the gap
    between the two steps is unbounded: a stopped worker or a failed chunking
    leaves them indefinitely.
    """

    @staticmethod
    def _seed(session, *, content_hash):
        source = SourceDocument(
            collection="c", source_path="/a.pdf", file_hash="fh", status="done"
        )
        extractor = ExtractorProfile(name="x", fingerprint="ex", config_json="{}")
        chunk_profile = ChunkProfile(fingerprint="cp", config_json="{}")
        session.add_all([source, extractor, chunk_profile])
        session.flush()
        extracted = ExtractedDocument(
            document_id=source.id,
            extractor_profile_id=extractor.id,
            source_file_hash="fh",
            content_hash=content_hash,
            status="done",
        )
        session.add(extracted)
        session.flush()
        chunked = ChunkedDocument(
            extracted_document_id=extracted.id,
            chunk_profile_id=chunk_profile.id,
            source_content_hash=content_hash,
            status="done",
            total_chunks=1,
        )
        session.add(chunked)
        session.flush()
        session.add(
            Chunk(
                document_id=source.id,
                chunked_document_id=chunked.id,
                chunk_index=0,
                content="old text",
            )
        )
        session.commit()
        return extracted

    def test_chunks_from_the_previous_content_are_deleted(self):
        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        session = sessionmaker(bind=engine, expire_on_commit=False)()
        extracted = self._seed(session, content_hash="old-hash")

        _purge_superseded_chunks(session, extracted.id, "new-hash")
        session.commit()

        assert session.query(Chunk).count() == 0

    def test_chunks_matching_the_current_content_are_kept(self):
        """A re-extraction that produced identical bytes must not throw work away."""
        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        session = sessionmaker(bind=engine, expire_on_commit=False)()
        extracted = self._seed(session, content_hash="same-hash")

        _purge_superseded_chunks(session, extracted.id, "same-hash")
        session.commit()

        assert session.query(Chunk).count() == 1

    def test_a_chunking_that_never_recorded_a_hash_is_treated_as_stale(self):
        """NULL is not evidence the chunking matches; it is evidence it is unknown."""
        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        session = sessionmaker(bind=engine, expire_on_commit=False)()
        extracted = self._seed(session, content_hash=None)

        _purge_superseded_chunks(session, extracted.id, "new-hash")
        session.commit()

        assert session.query(Chunk).count() == 0

    def test_a_failed_re_extraction_drops_the_old_chunks_too(self):
        """A failed re-extraction must not leave the previous version searchable.

        On failure there is no new content_hash, so the superseded purge cannot
        fire -- and skipping it entirely meant replacing an indexed file with a
        corrupt or encrypted one left search serving the *old* text under the
        current path indefinitely: search applies no freshness filter, and
        `_step_chunk` only revisits a document once extraction succeeds.
        """
        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        session = sessionmaker(bind=engine, expire_on_commit=False)()
        extracted = self._seed(session, content_hash="old-hash")

        _purge_all_chunks(session, extracted.id)
        session.commit()

        assert session.query(Chunk).count() == 0


class TestIndexBuildIsVisible:
    """A build occupies the loop for minutes while writing nothing else.

    Without an announcement `cementic status` shows a running worker, a
    `building` revision and no current file -- indistinguishable from an idle
    one, for tens of minutes on a large corpus. Nothing can be published from
    inside the build, so it is published on both sides of it.
    """

    def _worker(self):
        worker = PipelineWorker()
        worker.collection = "research"
        worker.state_manager = MagicMock()
        revision = SimpleNamespace(status="building")
        session = MagicMock()
        session.__enter__.return_value = session
        session.__exit__.return_value = False
        session.get.return_value = revision
        worker.Session = MagicMock(return_value=session)
        return worker

    def test_activity_is_published_and_then_cleared(self):
        worker = self._worker()
        with patch.object(worker, "_revision_complete", return_value=True):
            with patch("cementic.pipeline_worker.ensure_revision_ann_index"):
                with patch("cementic.pipeline_worker.mark_revision_ready"):
                    worker._mark_revision_ready_if_complete(1)

        published = [
            call.kwargs["current_activity"]
            for call in worker.state_manager.update.call_args_list
            if "current_activity" in call.kwargs
        ]
        assert published == ["building hnsw index", None]

    def test_activity_is_cleared_even_when_the_build_fails(self):
        """Otherwise a failed build leaves status claiming it is still running."""
        worker = self._worker()
        with patch.object(worker, "_revision_complete", return_value=True):
            with patch(
                "cementic.pipeline_worker.ensure_revision_ann_index",
                side_effect=RuntimeError("no disk"),
            ):
                with pytest.raises(RuntimeError):
                    worker._mark_revision_ready_if_complete(1)

        published = [
            call.kwargs["current_activity"]
            for call in worker.state_manager.update.call_args_list
            if "current_activity" in call.kwargs
        ]
        assert published[-1] is None

    def test_nothing_is_published_when_the_revision_is_not_complete(self):
        worker = self._worker()
        with patch.object(worker, "_revision_complete", return_value=False):
            worker._mark_revision_ready_if_complete(1)

        assert not [
            call
            for call in worker.state_manager.update.call_args_list
            if "current_activity" in call.kwargs
        ]
