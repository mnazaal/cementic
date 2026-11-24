"""Tests for pipeline worker constructor, client creation, and lifecycle."""

from datetime import datetime, timedelta, timezone
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
    CollectionScan,
    EmbeddingProfile,
    ExtractedDocument,
    ExtractorProfile,
    PipelineRevision,
    SourceDocument,
)
from cementic.pipeline_worker import (
    PipelineCounts,
    PipelineWorker,
    _over_budget_reasons,
    _pipeline_worker_lock_key,
    _purge_superseded_chunks,
    _try_acquire_pipeline_worker_lock,
    compute_revision_counts,
    embed_submit_size,
    is_retryable_embed_error,
    plan_embed_outcome,
    purge_all_chunks,
    revision_is_complete,
    should_yield_to_search,
)
from cementic.state import DaemonState


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


class _RecordingBatchClient:
    """Succeeds for every text and records how many texts each call carried."""

    def __init__(self) -> None:
        self.call_sizes: list[int] = []

    def format_document(self, text: str) -> str:
        return text

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        self.call_sizes.append(len(texts))
        return [[1.0, 0.0] for _ in texts]


class _PoisonChunkClient:
    """Batch call raises on the poison text; individual embeds fail only it."""

    def __init__(self, poison: str) -> None:
        self.poison = poison
        self.individual_calls: list[str] = []
        self.batch_sizes: list[int] = []

    def format_document(self, text: str) -> str:
        return text

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        self.batch_sizes.append(len(texts))
        if self.poison in texts:
            raise ValueError("poison chunk")
        return [[1.0, 0.0] for _ in texts]

    def embed(self, text: str) -> list[float]:
        self.individual_calls.append(text)
        if text == self.poison:
            raise ValueError("poison chunk")
        return [1.0, 0.0]


def _seed_embedding_batch(session, n_chunks: int = 2) -> int:
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
    for index in range(n_chunks):
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


class TestWriteBackFailuresDoNotStallTheCollection:
    """A claim is committed as `processing` before the write-back runs.

    If the write-back raises, the row stays `processing`, the step re-claims the
    same lowest-id artifact every pass, and every other document in the
    collection waits behind it forever. Observed twice live: a chunk carrying a
    NUL byte, then a chunk PostgreSQL otherwise refused.
    """

    def _worker(self, temp_dir: Path) -> PipelineWorker:
        config = Config()
        config.pipeline_worker.log_file = temp_dir / "worker.log"
        config.pipeline_worker.state_path = temp_dir / "worker.json"
        return PipelineWorker(config)

    def test_a_failed_artifact_is_stamped_terminal(self, temp_dir: Path) -> None:
        worker = self._worker(temp_dir)
        artifact = MagicMock()
        session = MagicMock()
        session.__enter__.return_value = session
        session.__exit__.return_value = False
        session.get.return_value = artifact
        with patch.object(worker, "Session", return_value=session):
            worker._fail_artifact(
                ChunkedDocument,
                7,
                ValueError("NUL"),
                "chunk",
                scope_hash_field="source_content_hash",
                scope_hash="abc123",
            )
        assert artifact.status == "failed"
        assert "NUL" in artifact.error_message
        # The scoping hash must be stamped with the status: without it the
        # claim query re-selects the row (hash IS DISTINCT FROM source) and the
        # hash-scoped revision counts see neither done nor failed.
        assert artifact.source_content_hash == "abc123"
        session.commit.assert_called_once()

    def test_a_vanished_artifact_is_not_resurrected(self, temp_dir: Path) -> None:
        """The row may have been purged while the step ran; do not recreate it."""
        worker = self._worker(temp_dir)
        session = MagicMock()
        session.__enter__.return_value = session
        session.__exit__.return_value = False
        session.get.return_value = None
        with patch.object(worker, "Session", return_value=session):
            worker._fail_artifact(
                    ChunkedDocument,
                    7,
                    ValueError("gone"),
                    "chunk",
                    scope_hash_field="source_content_hash",
                    scope_hash="h",
                )
        session.commit.assert_not_called()

    def test_an_unreachable_database_still_raises(self, temp_dir: Path) -> None:
        """A failure to record the failure is not a per-document fact.

        The loop's back-off is the right handler for a database that is down,
        so this must not be swallowed into a silent skip.
        """
        worker = self._worker(temp_dir)
        with patch.object(worker, "Session", side_effect=OSError("db down")):
            with pytest.raises(OSError):
                worker._fail_artifact(
                    ChunkedDocument,
                    7,
                    ValueError("x"),
                    "chunk",
                    scope_hash_field="source_content_hash",
                    scope_hash="h",
                )


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

        # stop() has two jobs: signal the loop, and record that it stopped.
        # The signal half used to be asserted by test_embedder.py's
        # test_stop_sets_shutdown, which was dropped as a duplicate of this
        # test -- it was only a partial one.
        assert worker._shutdown_event.is_set()

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

    def _http_error_with_body(self, status: int, body: str) -> requests.exceptions.HTTPError:
        response = requests.Response()
        response.status_code = status
        response._content = body.encode("utf-8")
        return requests.exceptions.HTTPError(response=response)

    def test_an_over_long_input_is_terminal_despite_being_a_500(self) -> None:
        """llama.cpp reports an unembeddable input with a retryable status.

        Observed live: a 549-model-token chunk pinned a 22k-document run at 608
        embedded chunks. Released as transient, the same chunk is re-claimed
        every poll and the collection never finishes, so the status has to be
        read together with the body.
        """
        error = self._http_error_with_body(
            500,
            '{"error":{"code":500,"message":"input (547 tokens) is too large to '
            'process. increase the physical batch size (current batch size: 512)",'
            '"type":"server_error"}}',
        )
        assert is_retryable_embed_error(error) is False

    def test_a_plain_500_is_still_retryable(self) -> None:
        """The daemon restarting must not stamp a whole batch failed."""
        error = self._http_error_with_body(500, '{"error":{"message":"model is loading"}}')
        assert is_retryable_embed_error(error) is True

    def test_connection_and_timeout_remain_retryable(self) -> None:
        assert is_retryable_embed_error(requests.exceptions.ConnectionError()) is True
        assert is_retryable_embed_error(requests.exceptions.Timeout()) is True
        assert is_retryable_embed_error(requests.exceptions.ChunkedEncodingError()) is True

    def test_unrelated_errors_are_terminal(self) -> None:
        assert is_retryable_embed_error(ValueError("bad data")) is False
        assert is_retryable_embed_error(KeyError("embedding")) is False

    def test_http_error_without_a_response_is_retryable(self) -> None:
        """No status can be read, so the failure must not be blamed on the texts."""
        assert is_retryable_embed_error(requests.exceptions.HTTPError()) is True

    def test_json_decode_error_is_retryable(self) -> None:
        """requests.JSONDecodeError is not an HTTPError (its MRO runs through

        InvalidJSONError -> RequestException -> OSError), so it fell through
        every branch and was misclassified terminal. Raised by response.json()
        on a truncated llama.cpp body.
        """
        error = requests.exceptions.JSONDecodeError("Expecting value", "garbled", 0)
        assert is_retryable_embed_error(error) is True

    def test_content_decoding_error_without_response_is_retryable(self) -> None:
        """ContentDecodingError carries response=None, so the old status lookup

        read None, which is not in the retryable-status set, and misclassified
        terminal.
        """
        error = requests.exceptions.ContentDecodingError()
        assert is_retryable_embed_error(error) is True


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

    @patch("cementic.pipeline_worker.requeue_interrupted_artifacts")
    @patch("cementic.pipeline_worker.get_target_revision")
    @patch("cementic.pipeline_worker.create_tables")
    @patch("cementic.pipeline_worker.get_session_factory")
    @patch("cementic.pipeline_worker.get_engine")
    def test_start_sets_running_state(
        self,
        mock_get_engine,
        mock_session_factory,
        mock_create_tables,
        mock_get_target,
        mock_requeue,
    ):
        class StopLoopError(Exception):
            pass

        worker = PipelineWorker()
        worker.state_manager.load = MagicMock(
            return_value=SimpleNamespace(daemon_state=DaemonState.STOPPED, pid=None)
        )
        worker.state_manager.update = MagicMock()
        worker._run_processing_loop = MagicMock(side_effect=StopLoopError)

        mock_embedding_client = MagicMock()
        worker._create_embedding_client = MagicMock(return_value=mock_embedding_client)
        mock_session_factory.return_value = MagicMock()
        # A real dimension: startup now refuses one the configured ANN index
        # cannot handle, and a MagicMock is not comparable to the limit.
        mock_get_target.return_value.embedding_profile.embedding_dim = 768

        try:
            worker.start(collection="research")
        except StopLoopError:
            pass

        assert worker.collection == "research"
        running_updates = [
            call
            for call in worker.state_manager.update.call_args_list
            if call.kwargs.get("daemon_state") == DaemonState.RUNNING
        ]
        assert len(running_updates) == 1


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
        # The loop now verifies its revision still exists once per pass;
        # a MagicMock session answers that check (get() -> truthy).
        worker.Session = MagicMock()

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
        worker.Session = MagicMock()

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
        worker.Session = MagicMock()
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

    def test_loop_clears_a_recorded_failure_after_a_productive_pass(
        self, temp_dir: Path
    ) -> None:
        """A recovered worker that is busy must not keep reporting the old error.

        Regression (2026-10-22 audit): the clear ran only on an idle pass, and a
        productive step `continue`d past it, so one transient failure early in a
        multi-day import stayed in `cementic status` until the import ended.
        """
        config = Config()
        config.pipeline_worker.log_file = temp_dir / "worker.log"
        config.pipeline_worker.state_path = temp_dir / "worker-state.json"
        worker = PipelineWorker(config)
        worker.Session = MagicMock()

        calls = {"n": 0}
        observed: dict[str, object] = {}

        def extract(_revision_id: int) -> bool:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient")
            if calls["n"] == 3:
                # The pass after the first productive one, with no idle pass yet.
                observed["last_error"] = worker.state_manager.load().last_error
                worker._shutdown_event.set()
            return True

        with (
            patch.object(worker, "_step_extract", side_effect=extract),
            patch.object(worker, "_step_chunk", return_value=False),
            patch.object(worker, "_step_embed", return_value=False),
            patch.object(worker, "_mark_revision_ready_if_complete"),
            patch.object(worker._shutdown_event, "wait"),
        ):
            worker._run_processing_loop(1)

        assert observed["last_error"] is None

    def test_loop_skips_when_extract_does_work(self, temp_dir: Path) -> None:
        """When _step_extract returns True the loop continues immediately."""
        config = Config()
        config.pipeline_worker.log_file = temp_dir / "worker.log"
        worker = PipelineWorker(config)
        worker.Session = MagicMock()

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
        worker.Session = MagicMock()
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


class TestShouldYieldToSearch:
    """Pure yield policy for the embed step (query-first scheduling)."""

    NOW = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)

    def test_fresh_lease_yields(self) -> None:
        last = self.NOW - timedelta(seconds=3)
        assert should_yield_to_search(last, self.NOW, 10.0, None, 60.0) is True

    def test_stale_lease_does_not_yield(self) -> None:
        last = self.NOW - timedelta(seconds=11)
        assert should_yield_to_search(last, self.NOW, 10.0, None, 60.0) is False

    def test_missing_lease_does_not_yield(self) -> None:
        assert should_yield_to_search(None, self.NOW, 10.0, None, 60.0) is False

    def test_zero_ttl_disables_yielding(self) -> None:
        last = self.NOW - timedelta(seconds=1)
        assert should_yield_to_search(last, self.NOW, 0.0, None, 60.0) is False

    def test_yield_cap_forces_progress_under_continuous_searching(self) -> None:
        """A lease refreshed in a loop must not stall a build indefinitely."""
        last = self.NOW - timedelta(seconds=1)
        yielding_since = self.NOW - timedelta(seconds=60)
        assert should_yield_to_search(last, self.NOW, 10.0, yielding_since, 60.0) is False

    def test_within_cap_keeps_yielding(self) -> None:
        last = self.NOW - timedelta(seconds=1)
        yielding_since = self.NOW - timedelta(seconds=30)
        assert should_yield_to_search(last, self.NOW, 10.0, yielding_since, 60.0) is True


class TestEmbedSubmitSize:
    """Pure request-sizing policy for the embed step."""

    def test_idle_sends_the_whole_claim(self) -> None:
        """No query is waiting, so nothing is bought by splitting the claim."""
        assert embed_submit_size(32, 4, 32, search_is_active=False) == 32

    def test_active_search_shrinks_the_request(self) -> None:
        assert embed_submit_size(32, 4, 32, search_is_active=True) == 4

    def test_never_exceeds_what_is_left(self) -> None:
        """The last request of a claim carries the remainder, not a full size."""
        assert embed_submit_size(3, 4, 32, search_is_active=False) == 3
        assert embed_submit_size(3, 4, 32, search_is_active=True) == 3

    def test_always_makes_progress(self) -> None:
        """A zero would spin the caller's loop forever on a non-empty claim."""
        assert embed_submit_size(1, 0, 0, search_is_active=True) == 1
        assert embed_submit_size(1, 0, 0, search_is_active=False) == 1


class TestPipelineWorkerEmbedStep:
    """Embedding step liveness regressions."""

    def test_embed_step_sends_the_whole_claim_when_nobody_is_searching(
        self, temp_dir: Path, monkeypatch
    ) -> None:
        """With no search lease the claim goes to the provider in one call.

        Splitting it costs a fixed per-request overhead and drains the server's
        slots across each round trip -- ~2x throughput, measured -- and buys
        nothing when no query is waiting. The vector-table write is stubbed: it
        is Postgres-only and not the behavior under test.
        """
        import cementic.pipeline_worker as worker_module

        monkeypatch.setattr(worker_module, "ensure_vector_table_schema", lambda *a, **k: None)
        monkeypatch.setattr(worker_module, "upsert_vectors", lambda *a, **k: None)
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        with session_factory() as session:
            revision_id = _seed_embedding_batch(session, n_chunks=5)

        config = Config()
        config.pipeline_worker.log_file = temp_dir / "worker.log"
        config.pipeline_worker.batch_size = 6
        config.pipeline_worker.embed_submit_batch_size = 2
        worker = PipelineWorker(config)
        worker.Session = session_factory
        client = _RecordingBatchClient()
        worker.embedding_client = client

        assert worker._step_embed(revision_id) is True

        assert client.call_sizes == [5]
        with session_factory() as session:
            rows = session.query(ChunkEmbedding).order_by(ChunkEmbedding.chunk_id).all()
            assert [row.status for row in rows] == ["done"] * 5

    def test_embed_step_shrinks_requests_while_a_search_is_active(
        self, temp_dir: Path, monkeypatch
    ) -> None:
        """A lease that stays fresh caps each request at embed_submit_batch_size.

        This is the case the small request exists for: a query landing
        mid-request waits for a few texts rather than a whole claim. The yield
        cap is what lets the build proceed at all while the lease keeps being
        refreshed.
        """
        import cementic.pipeline_worker as worker_module
        from cementic.db import SearchActivity, utc_now

        monkeypatch.setattr(worker_module, "ensure_vector_table_schema", lambda *a, **k: None)
        monkeypatch.setattr(worker_module, "upsert_vectors", lambda *a, **k: None)
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        with session_factory() as session:
            revision_id = _seed_embedding_batch(session, n_chunks=5)
            session.merge(SearchActivity(id=1, last_search_at=utc_now()))
            session.commit()

        config = Config()
        config.pipeline_worker.log_file = temp_dir / "worker.log"
        config.pipeline_worker.batch_size = 6
        config.pipeline_worker.embed_submit_batch_size = 2
        # The lease never goes stale here, so the cap is the only thing that
        # releases a request: near-zero means every wait ends immediately with
        # the search still active, which is exactly the small-request state.
        config.pipeline_worker.search_yield_cap_seconds = 0.001
        config.pipeline_worker.poll_interval = 0.001
        worker = PipelineWorker(config)
        worker.Session = session_factory
        client = _RecordingBatchClient()
        worker.embedding_client = client

        assert worker._step_embed(revision_id) is True

        assert client.call_sizes == [2, 2, 1]
        with session_factory() as session:
            rows = session.query(ChunkEmbedding).order_by(ChunkEmbedding.chunk_id).all()
            assert [row.status for row in rows] == ["done"] * 5

    def test_embed_step_yields_between_sub_batches_while_search_lease_is_fresh(
        self, temp_dir: Path, monkeypatch
    ) -> None:
        """A fresh search lease delays provider calls until it goes stale."""
        import cementic.pipeline_worker as worker_module
        from cementic.db import SearchActivity, utc_now

        monkeypatch.setattr(worker_module, "ensure_vector_table_schema", lambda *a, **k: None)
        monkeypatch.setattr(worker_module, "upsert_vectors", lambda *a, **k: None)
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        with session_factory() as session:
            revision_id = _seed_embedding_batch(session, n_chunks=4)
            session.merge(SearchActivity(id=1, last_search_at=utc_now()))
            session.commit()

        config = Config()
        config.pipeline_worker.log_file = temp_dir / "worker.log"
        config.pipeline_worker.batch_size = 6
        config.pipeline_worker.embed_submit_batch_size = 2
        worker = PipelineWorker(config)
        worker.Session = session_factory
        client = _RecordingBatchClient()
        worker.embedding_client = client

        class _StaleOnWait:
            """Shutdown-event stand-in whose wait() ages the lease out."""

            def __init__(self) -> None:
                self.waits = 0

            def is_set(self) -> bool:
                return False

            def wait(self, timeout: float | None = None) -> bool:
                self.waits += 1
                with session_factory() as session:
                    row = session.get(SearchActivity, 1)
                    row.last_search_at = utc_now() - timedelta(seconds=60)
                    session.commit()
                return False

        event_stub = _StaleOnWait()
        worker._shutdown_event = event_stub

        assert worker._step_embed(revision_id) is True

        assert event_stub.waits >= 1
        # The wait ended because the lease aged out, so the request that
        # follows it is claim-sized rather than interactive-sized.
        assert client.call_sizes == [4]
        with session_factory() as session:
            rows = session.query(ChunkEmbedding).order_by(ChunkEmbedding.chunk_id).all()
            assert [row.status for row in rows] == ["done"] * 4

    def test_data_failure_falls_back_to_individual_embeds_within_its_request_only(
        self, temp_dir: Path, monkeypatch
    ) -> None:
        """One bad chunk costs its request the one-by-one retry, not the claim.

        The retry is scoped to the failed request, so how much it re-embeds
        follows the request size, and a claim-sized request is re-tried in
        groups before any text is embedded alone
        (test_data_failure_retries_in_groups_before_going_one_at_a_time).
        """
        import cementic.pipeline_worker as worker_module
        from cementic.db import SearchActivity, utc_now

        monkeypatch.setattr(worker_module, "ensure_vector_table_schema", lambda *a, **k: None)
        monkeypatch.setattr(worker_module, "upsert_vectors", lambda *a, **k: None)
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        with session_factory() as session:
            revision_id = _seed_embedding_batch(session, n_chunks=5)
            session.merge(SearchActivity(id=1, last_search_at=utc_now()))
            session.commit()

        config = Config()
        config.pipeline_worker.log_file = temp_dir / "worker.log"
        config.pipeline_worker.batch_size = 6
        config.pipeline_worker.embed_submit_batch_size = 2
        config.pipeline_worker.search_yield_cap_seconds = 0.001
        config.pipeline_worker.poll_interval = 0.001
        worker = PipelineWorker(config)
        worker.Session = session_factory
        client = _PoisonChunkClient(poison="chunk 2")
        worker.embedding_client = client

        assert worker._step_embed(revision_id) is True

        # Only the poisoned request ("chunk 2", "chunk 3") went one at a time.
        assert client.individual_calls == ["chunk 2", "chunk 3"]
        with session_factory() as session:
            rows = session.query(ChunkEmbedding).order_by(ChunkEmbedding.chunk_id).all()
            assert [row.status for row in rows] == ["done", "done", "failed", "done", "done"]
            assert "poison chunk" in rows[2].error_message

    def test_data_failure_retries_in_groups_before_going_one_at_a_time(
        self, temp_dir: Path, monkeypatch
    ) -> None:
        """A claim-sized request that fails is re-tried in groups, not per text.

        One text at a time uses a single server slot and costs a round trip per
        chunk, so isolating one bad chunk in a 32-text claim took 32 sequential
        calls. Only the group that fails again is walked individually; the rest
        of the claim stays batched.
        """
        import cementic.pipeline_worker as worker_module

        monkeypatch.setattr(worker_module, "ensure_vector_table_schema", lambda *a, **k: None)
        monkeypatch.setattr(worker_module, "upsert_vectors", lambda *a, **k: None)
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        with session_factory() as session:
            revision_id = _seed_embedding_batch(session, n_chunks=9)

        config = Config()
        config.pipeline_worker.log_file = temp_dir / "worker.log"
        config.pipeline_worker.batch_size = 16
        config.pipeline_worker.embed_submit_batch_size = 4
        worker = PipelineWorker(config)
        worker.Session = session_factory
        client = _PoisonChunkClient(poison="chunk 2")
        worker.embedding_client = client

        assert worker._step_embed(revision_id) is True

        # 9 in one request (no search active), then groups of 4, 4, 1.
        assert client.batch_sizes == [9, 4, 4, 1]
        # Only the group holding the poison went one at a time.
        assert client.individual_calls == ["chunk 0", "chunk 1", "chunk 2", "chunk 3"]
        with session_factory() as session:
            rows = session.query(ChunkEmbedding).order_by(ChunkEmbedding.chunk_id).all()
            statuses = [row.status for row in rows]
            assert statuses == ["done", "done", "failed", "done", "done", *["done"] * 4]

    def test_stop_during_the_retry_walk_releases_the_claim(
        self, temp_dir: Path, monkeypatch
    ) -> None:
        """SIGTERM mid-retry returns the claim to pending instead of failing it.

        The retry walk is one HTTP round trip per group and then per text, so
        without a shutdown check the worker ignored a stop for the whole walk
        -- 44 s measured, which outlasted `cementic stop`'s grace period and
        turned the documented restart into a stop that started nothing.
        """
        import cementic.pipeline_worker as worker_module

        monkeypatch.setattr(worker_module, "ensure_vector_table_schema", lambda *a, **k: None)
        monkeypatch.setattr(worker_module, "upsert_vectors", lambda *a, **k: None)
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        with session_factory() as session:
            revision_id = _seed_embedding_batch(session, n_chunks=9)

        config = Config()
        config.pipeline_worker.log_file = temp_dir / "worker.log"
        config.pipeline_worker.batch_size = 16
        config.pipeline_worker.embed_submit_batch_size = 4
        worker = PipelineWorker(config)
        worker.Session = session_factory
        client = _PoisonChunkClient(poison="chunk 2")
        worker.embedding_client = client

        class _StopOnceEmbeddingStarted:
            """Set only after the claim-sized request has been sent.

            Keyed on the client rather than a call count so the worker cannot
            satisfy this test by bailing out before it embeds anything -- which
            is a different code path and would leave the retry walk untested.
            """

            def __init__(self, recorder: _PoisonChunkClient) -> None:
                self.recorder = recorder

            def is_set(self) -> bool:
                return bool(self.recorder.batch_sizes)

            def wait(self, timeout: float | None = None) -> bool:
                return False

        worker._shutdown_event = _StopOnceEmbeddingStarted(client)

        assert worker._step_embed(revision_id) is False

        # The claim-sized request went out and failed; the retry walk then saw
        # the stop before issuing any group call.
        assert client.batch_sizes == [9]
        assert client.individual_calls == []
        with session_factory() as session:
            rows = session.query(ChunkEmbedding).order_by(ChunkEmbedding.chunk_id).all()
            # Back to pending, not stamped failed: nothing here is a property
            # of the texts.
            assert {row.status for row in rows} == {"pending"}

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


class TestFailArtifactStampsTheScopingHash:
    """`_fail_artifact` must stamp the claim-scoping hash with the status.

    Without it the failed row still matches the claim query's
    ``hash IS DISTINCT FROM source`` disjunct -- re-claimed (and re-extracted)
    every pass -- while the hash-scoped revision counts see neither done nor
    failed, so the revision never settles. The missing-artifact branch in
    `_step_chunk` stamps the hash for exactly this reason; this pins the
    write-back-failure path to the same invariant, through the real claim query.
    """

    def test_a_write_back_failure_is_not_reclaimed_forever(self, temp_dir: Path) -> None:
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        collection = "failstamp"
        with session_factory() as session:
            revision_id, pending_id = _seed_extract_selection_data(session, collection, 0)

        config = Config()
        config.pipeline_worker.log_file = temp_dir / "worker.log"
        worker = PipelineWorker(config)
        worker.Session = session_factory
        worker.collection = collection

        with (
            patch("cementic.pipeline_worker.extract_document", return_value="content"),
            patch("cementic.pipeline_worker.write_extracted_text", return_value="content-hash"),
            patch.object(
                worker, "_write_back_extraction", side_effect=RuntimeError("injected")
            ),
            patch.object(worker.state_manager, "update"),
        ):
            assert worker._step_extract(revision_id) is True

        with session_factory() as session:
            extracted = (
                session.query(ExtractedDocument).filter_by(document_id=pending_id).one()
            )
            assert extracted.status == "failed"
            # The stamp that keeps the row out of the claim query and inside
            # the failure counts.
            assert extracted.source_file_hash == "pending-hash"

        # The second pass must find nothing to claim: a re-claim here is the
        # stall this class exists to prevent.
        with (
            patch("cementic.pipeline_worker.extract_document", return_value="content"),
            patch("cementic.pipeline_worker.write_extracted_text", return_value="content-hash"),
            patch.object(worker.state_manager, "update"),
        ):
            assert worker._step_extract(revision_id) is False


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


class TestTokenizerOutageIsNotADocumentFailure:
    """An unfetchable tokenizer must not stamp every document failed (2026-10-22)."""

    def test_the_chunk_step_raises_and_leaves_the_row_claimable(self, temp_dir: Path) -> None:
        from cementic.chunk import TokenizerUnavailableError

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        with session_factory() as session:
            revision_id, pending_extracted_id = _seed_chunk_selection_data(
                session, "offline", n_already_done=0
            )
        config = Config()
        config.pipeline_worker.log_file = temp_dir / "worker.log"
        worker = PipelineWorker(config)
        worker.Session = session_factory
        worker.collection = "offline"

        with (
            patch("cementic.pipeline_worker.read_extracted_text", return_value="some text"),
            patch("cementic.pipeline_worker.tokenizer"),
            patch(
                "cementic.pipeline_worker.chunk_text",
                side_effect=TokenizerUnavailableError("offline"),
            ),
            patch.object(worker.state_manager, "update"),
            pytest.raises(TokenizerUnavailableError),
        ):
            worker._step_chunk(revision_id)

        with session_factory() as session:
            chunked = (
                session.query(ChunkedDocument)
                .filter_by(extracted_document_id=pending_extracted_id)
                .one()
            )
            # Handed back uncounted: a restart during the outage must not cap
            # a healthy document as a crash (review finding).
            assert (chunked.status, chunked.attempts) == ("pending", 0)

    def test_an_uncached_tokenizer_raises_before_any_claim(self, temp_dir: Path) -> None:
        from cementic.chunk import TokenizerUnavailableError

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        with session_factory() as session:
            revision_id, pending_extracted_id = _seed_chunk_selection_data(
                session, "offline-early", n_already_done=0
            )
        config = Config()
        config.pipeline_worker.log_file = temp_dir / "worker.log"
        worker = PipelineWorker(config)
        worker.Session = session_factory
        worker.collection = "offline-early"

        with (
            patch(
                "cementic.pipeline_worker.tokenizer",
                side_effect=TokenizerUnavailableError("offline"),
            ),
            pytest.raises(TokenizerUnavailableError),
        ):
            worker._step_chunk(revision_id)

        with session_factory() as session:
            assert (
                session.query(ChunkedDocument)
                .filter_by(extracted_document_id=pending_extracted_id)
                .count()
                == 0
            )

    def test_an_uncached_tokenizer_is_a_retryable_embed_error(self) -> None:
        """Review finding: stamping it failed every pending embedding offline."""
        from cementic.chunk import TokenizerUnavailableError
        from cementic.pipeline_worker import is_retryable_embed_error

        assert is_retryable_embed_error(TokenizerUnavailableError("offline")) is True


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
        # The loop verifies its revision still exists once per pass.
        worker.Session = MagicMock()

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


class TestFatalStartupReasonReachesTheStateFile:
    """A startup failure past `cementic start`'s 2 s grace only shows up in

    `cementic status` if it lands in `last_error` -- `_STARTUP_GRACE_SECONDS`
    (cli.py) is far shorter than `daemon_start_timeout_seconds`, so a slow
    embedding daemon load can keep failing long after `start` already reported
    success. Regression: `report_fatal` logged and echoed to stderr but never
    touched the state file, so neither `status` nor `doctor` could show it.
    """

    def test_embedding_startup_failure_lands_in_last_error(self, temp_dir: Path) -> None:
        config = Config()
        config.pipeline_worker.log_file = temp_dir / "worker.log"
        config.pipeline_worker.state_path = temp_dir / "state.json"
        worker = PipelineWorker(config)

        with (
            patch("cementic.pipeline_worker.get_engine"),
            patch("cementic.pipeline_worker.create_tables"),
            patch("cementic.pipeline_worker.get_session_factory"),
            patch.object(
                worker,
                "_create_embedding_client",
                side_effect=RuntimeError("connection refused"),
            ),
        ):
            worker.start(collection="c")

        state = worker.state_manager.load()
        assert state.last_error is not None
        assert "connection refused" in state.last_error
        assert state.last_error_at is not None

    def test_a_clean_start_after_a_failed_one_clears_last_error(self, temp_dir: Path) -> None:
        config = Config()
        config.pipeline_worker.log_file = temp_dir / "worker.log"
        config.pipeline_worker.state_path = temp_dir / "state.json"
        worker = PipelineWorker(config)

        with (
            patch("cementic.pipeline_worker.get_engine"),
            patch("cementic.pipeline_worker.create_tables"),
            patch("cementic.pipeline_worker.get_session_factory"),
            patch.object(
                worker, "_create_embedding_client", side_effect=RuntimeError("boom")
            ),
        ):
            worker.start(collection="c")
        assert worker.state_manager.load().last_error is not None

        class StopLoopError(Exception):
            pass

        client = MagicMock()
        with (
            patch("cementic.pipeline_worker.get_engine"),
            patch("cementic.pipeline_worker.create_tables"),
            patch("cementic.pipeline_worker.get_session_factory"),
            patch("cementic.pipeline_worker.requeue_interrupted_artifacts"),
            patch.object(worker, "_create_embedding_client", return_value=client),
            patch.object(worker, "_ensure_target_revision", return_value=1),
            patch.object(worker, "_run_processing_loop", side_effect=StopLoopError),
        ):
            try:
                worker.start(collection="c")
            except StopLoopError:
                pass

        state = worker.state_manager.load()
        assert state.last_error is None
        assert state.last_error_at is None


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

        purge_all_chunks(session, extracted.id)
        session.commit()

        assert session.query(Chunk).count() == 0


class TestEnsureTargetRevisionAndMarkReady:
    """Tests for `_ensure_target_revision` and `_mark_revision_ready_if_complete`."""

    def test_ensure_target_revision_commits(self):
        worker = PipelineWorker()
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = False
        mock_session.get.return_value = None
        worker.Session = MagicMock(return_value=mock_session)

        with patch(
            "cementic.pipeline_worker.get_target_revision", return_value=SimpleNamespace(
                id=7, embedding_profile=SimpleNamespace(embedding_dim=768)
            )
        ) as mock_get_target:
            revision_id = worker._ensure_target_revision()

        assert revision_id == 7
        mock_get_target.assert_called_once()
        mock_session.commit.assert_called_once()

    def test_mark_revision_ready_if_complete(self):
        worker = PipelineWorker()
        revision = SimpleNamespace(status="building")
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = False
        mock_session.get.return_value = revision
        worker.Session = MagicMock(return_value=mock_session)

        with patch.object(worker, "_revision_complete", return_value=True):
            with patch("cementic.pipeline_worker.ensure_revision_ann_index") as mock_ensure_index:
                with patch("cementic.pipeline_worker.mark_revision_ready") as mock_mark_ready:
                    worker._mark_revision_ready_if_complete(3)

        mock_ensure_index.assert_called_once_with(mock_session, revision, worker.config)
        mock_mark_ready.assert_called_once_with(mock_session, revision)
        # Two commits: one ends the counts' read transaction before the index
        # build (which runs DDL on its own connections while this session
        # would otherwise idle in transaction), one lands mark_revision_ready.
        assert mock_session.commit.call_count == 2


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


class TestOverBudgetReasonsAreCollectedOutsideTheTransaction:
    """Explaining a failed embedding must not hold a write transaction open."""

    def test_a_provider_that_raises_still_leaves_the_row_terminal(self):
        """Regression: `over_budget_reason` was called inside the session that
        had just upserted vectors. It can issue an exact-tokenize round trip, so
        a daemon that went away raised there -- and the handler released the
        claim, returning rows to `pending` that the old code stamped `failed`.
        The next poll re-embedded and re-failed them, indefinitely."""
        provider = MagicMock()
        provider.over_budget_reason.side_effect = ConnectionError("daemon went away")

        reasons = _over_budget_reasons(provider, ["a", "b"], [[0.1], None])

        assert reasons == [None, None]

    def test_reasons_are_only_requested_for_the_failed_texts(self):
        """One round trip per failure, not per row."""
        provider = MagicMock()
        provider.over_budget_reason.return_value = "too long"

        reasons = _over_budget_reasons(provider, ["a", "b", "c"], [[0.1], None, [0.3]])

        assert reasons == [None, "too long", None]
        provider.over_budget_reason.assert_called_once_with("b")

    def test_a_provider_without_the_method_yields_no_reasons(self):
        reasons = _over_budget_reasons(object(), ["a"], [None])
        assert reasons == [None]


class TestClaimedAttempts:
    def test_a_repeat_claim_counts_up(self) -> None:
        from cementic.revisions import claimed_attempts

        assert claimed_attempts(2, same_input=True) == 3

    def test_new_input_starts_again(self) -> None:
        from cementic.revisions import claimed_attempts

        assert claimed_attempts(7, same_input=False) == 1


class TestPlanEmbedOutcome:
    """The outcome policy of an embed batch, as data."""

    @pytest.mark.parametrize(
        ("results", "expected_done", "expected_failed"),
        [
            ([[1.0], [2.0]], [(10, [1.0]), (11, [2.0])], {}),
            ([None, None], [], {10: "e0", 11: "e1"}),
            ([[1.0], None], [(10, [1.0])], {11: "e1"}),
            ([None, [2.0]], [(11, [2.0])], {10: "e0"}),
            ([], [], {}),
        ],
    )
    def test_splits_claim_by_whether_a_vector_came_back(
        self, results, expected_done, expected_failed
    ) -> None:
        claimed = [(10, "a"), (11, "b")][: len(results)]
        outcome = plan_embed_outcome(claimed, results, ["e0", "e1"][: len(results)], {10, 11})
        assert outcome.done == expected_done
        assert outcome.failed == expected_failed

    @pytest.mark.parametrize(
        ("live", "expected_done", "expected_failed"),
        [
            ({10, 11}, [(10, [1.0])], {11: "e1"}),
            ({11}, [], {11: "e1"}),
            ({10}, [(10, [1.0])], {}),
            (set(), [], {}),
        ],
    )
    def test_drops_chunks_deleted_while_the_request_was_in_flight(
        self, live, expected_done, expected_failed
    ) -> None:
        outcome = plan_embed_outcome([(10, "a"), (11, "b")], [[1.0], None], ["e0", "e1"], live)
        assert outcome.done == expected_done
        assert outcome.failed == expected_failed


class _EveryOtherFailsClient:
    """Embeds the even-numbered texts and returns None for the odd ones."""

    def format_document(self, text: str) -> str:
        return text

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        return [[1.0, 0.0] if index % 2 == 0 else None for index in range(len(texts))]


class TestEmbedWriteBackStatementCount:
    def test_a_32_chunk_batch_is_written_back_in_a_constant_number_of_statements(
        self, temp_dir: Path, monkeypatch
    ) -> None:
        """Measured before the change at 64 write-back statements (32 chunks):
        a SELECT and an UPDATE per row.

        The count is taken from the vector upsert (stubbed; Postgres-only) to
        the end of the step, so it is the write-back alone.
        """
        import cementic.pipeline_worker as worker_module

        statements: list[str] = []
        upsert_at: list[int] = []

        def fake_upsert(*args, **kwargs) -> None:
            upsert_at.append(len(statements))

        monkeypatch.setattr(worker_module, "ensure_vector_table_schema", lambda *a, **k: None)
        monkeypatch.setattr(worker_module, "upsert_vectors", fake_upsert)
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        with session_factory() as session:
            revision_id = _seed_embedding_batch(session, n_chunks=32)
        event.listen(
            engine,
            "before_cursor_execute",
            lambda conn, cursor, statement, *a, **k: statements.append(statement),
        )

        config = Config()
        config.pipeline_worker.log_file = temp_dir / "worker.log"
        config.pipeline_worker.batch_size = 32
        worker = PipelineWorker(config)
        worker.Session = session_factory
        worker.embedding_client = _EveryOtherFailsClient()

        assert worker._step_embed(revision_id) is True

        write_back = len(statements) - upsert_at[0]
        with session_factory() as session:
            rows = session.query(ChunkEmbedding).order_by(ChunkEmbedding.chunk_id).all()
        assert [row.status for row in rows] == ["done", "failed"] * 16
        assert write_back == 2


class TestEveryPurgeReopensTheWork:
    """The invariant `purge_all_chunks` documents, checked at each door that empties a chunking.

    ``_step_chunk`` re-claims a *done* chunking only when its
    ``source_content_hash`` differs from the extraction's, so emptying a chunking
    without invalidating it strands the document: it owns no chunks, is counted
    as chunked, and `revision_is_complete` still returns True. `purge_all_chunks`
    states the rule; three neighbours applied it and two did not (2026-11-17
    audit, both scenarios executed).

    Each test drives the real write-back, not the helper, because the helper
    being right is what the audit already knew and the doors are where it broke.
    """

    @staticmethod
    def _worker(temp_dir: Path, session_factory):
        config = Config()
        config.pipeline_worker.log_file = temp_dir / "worker.log"
        config.pipeline_worker.state_path = temp_dir / "worker.json"
        config.pipeline.chunk_size = 64
        config.pipeline.chunk_overlap = 16
        worker = PipelineWorker(config)
        worker.Session = session_factory
        worker.collection = "revert"
        return worker

    @staticmethod
    def _seed(session_factory, *, chunks: int = 2):
        """An indexed document: extraction and chunking done at content hash A."""
        with session_factory() as session:
            source = SourceDocument(
                collection="revert", source_path="/papers/a.pdf", file_hash="file-A"
            )
            extractor = ExtractorProfile(name="x", fingerprint="x", config_json="{}")
            chunk_profile = ChunkProfile(fingerprint="c", config_json="{}")
            embedding_profile = EmbeddingProfile(
                fingerprint="e", provider="test", model_identifier="test",
                embedding_dim=2, distance_metric="cosine", config_json="{}",
            )
            session.add_all([source, extractor, chunk_profile, embedding_profile])
            session.flush()
            revision = PipelineRevision(
                collection="revert",
                extractor_profile_id=extractor.id,
                chunk_profile_id=chunk_profile.id,
                embedding_profile_id=embedding_profile.id,
                status="building",
            )
            session.add(revision)
            session.flush()
            extracted = ExtractedDocument(
                document_id=source.id, extractor_profile_id=extractor.id,
                source_file_hash="file-A", content_hash="A", status="done",
                artifact_path="/artifacts/a.md",
            )
            session.add(extracted)
            session.flush()
            chunked = ChunkedDocument(
                extracted_document_id=extracted.id, chunk_profile_id=chunk_profile.id,
                source_content_hash="A", status="done", total_chunks=chunks,
            )
            session.add(chunked)
            session.flush()
            for index in range(chunks):
                session.add(
                    Chunk(document_id=source.id, chunked_document_id=chunked.id,
                          chunk_index=index, content=f"text from A, chunk {index}")
                )
            session.commit()
            return revision.id, extracted.id, chunked.id, source.id

    def test_a_re_extension_that_reverts_does_not_strand_the_document(self, temp_dir: Path) -> None:
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        revision_id, extracted_id, chunked_id, _ = self._seed(session_factory)
        worker = self._worker(temp_dir, session_factory)

        # The file changes to B, then reverts to A before chunking ever ran.
        worker._write_back_extraction(
            extracted_id, "file-B", Path("/artifacts/b.md"), "B", "done", None
        )
        worker._write_back_extraction(
            extracted_id, "file-A", Path("/artifacts/a.md"), "A", "done", None
        )

        with session_factory() as session:
            chunked = session.get(ChunkedDocument, chunked_id)
            revision = session.get(PipelineRevision, revision_id)
            # The purge emptied the chunks, so the chunking must stop claiming to
            # be done -- NULL is what re-opens the claim.
            assert session.query(Chunk).count() == 0
            assert chunked.source_content_hash is None
            counts = compute_revision_counts(session, "revert", revision)
            assert counts.chunked_done == 0, "an empty chunking must not count as chunked"
            assert revision_is_complete(counts) is False

        # And the work actually re-opens, rather than merely being uncounted.
        with patch(
            "cementic.pipeline_worker.read_extracted_text", return_value="text from A " * 40
        ):
            assert worker._step_chunk(revision_id) is True
        with session_factory() as session:
            chunked = session.get(ChunkedDocument, chunked_id)
            assert chunked.status == "done"
            assert chunked.source_content_hash == "A"
            assert session.query(Chunk).count() > 0

    def test_a_failed_extraction_write_back_stops_serving_the_old_text(
        self, temp_dir: Path
    ) -> None:
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        revision_id, extracted_id, chunked_id, _ = self._seed(session_factory)
        worker = self._worker(temp_dir, session_factory)
        with session_factory() as session:
            session.get(SourceDocument, 1).file_hash = "file-B"
            session.get(ExtractedDocument, extracted_id).status = "pending"
            session.commit()

        with (
            patch("cementic.pipeline_worker.extract_document", return_value="text from B"),
            patch("cementic.pipeline_worker.write_extracted_text", return_value="B"),
            patch(
                "cementic.pipeline_worker._purge_superseded_chunks",
                side_effect=RuntimeError("deadlock detected"),
            ),
            patch.object(worker.state_manager, "update"),
        ):
            assert worker._step_extract(revision_id) is True

        with session_factory() as session:
            extracted = session.get(ExtractedDocument, extracted_id)
            chunked = session.get(ChunkedDocument, chunked_id)
            assert extracted.status == "failed"
            # Content is unknown now, so it must not name the old text.
            assert extracted.content_hash is None
            assert session.query(Chunk).count() == 0, "the superseded text is still served"
            assert chunked.source_content_hash is None

    def test_a_failed_chunk_write_back_leaves_a_settleable_revision(self, temp_dir: Path) -> None:
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        revision_id, extracted_id, chunked_id, _ = self._seed(session_factory)
        worker = self._worker(temp_dir, session_factory)
        # The text changed to B and extraction already wrote it back; the
        # chunking is still the one built from A, so the chunk step claims it.
        with session_factory() as session:
            session.get(ExtractedDocument, extracted_id).content_hash = "B"
            session.commit()

        with (
            patch("cementic.pipeline_worker.read_extracted_text", return_value="text from B " * 40),
            patch.object(
                PipelineWorker, "_write_back_chunks", side_effect=RuntimeError("NUL byte")
            ),
            patch.object(worker.state_manager, "update"),
        ):
            assert worker._step_chunk(revision_id) is True

        with session_factory() as session:
            chunked = session.get(ChunkedDocument, chunked_id)
            revision = session.get(PipelineRevision, revision_id)
            assert chunked.status == "failed"
            assert session.query(Chunk).count() == 0, "chunks of a failed chunking are still served"
            counts = compute_revision_counts(session, "revert", revision)
            # Both sides are zero here because the fixture seeds no embedding
            # rows: on SQLite the equality is 0 == 0 and proves nothing. The
            # half that matters -- that the removed chunks took their
            # embeddings and vectors with them, via chunks_v2's ON DELETE
            # CASCADE -- is a PostgreSQL fact and lives in D8 of the plan.
            assert counts.total_chunks == 0
            assert revision_is_complete(counts) is True

    def test_no_door_leaves_a_counted_chunking_with_no_chunks(self, temp_dir: Path) -> None:
        """The invariant as `chunked_scope` counts it, not as the row reads.

        A chunking is counted as chunked only while its hash matches the
        extraction's; that is the state the property is about, and the one
        `doctor` warns on (`total_chunks > 0` aside, which keeps a legitimately
        empty document out of it).
        """
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        revision_id, extracted_id, chunked_id, _ = self._seed(session_factory)
        worker = self._worker(temp_dir, session_factory)

        worker._write_back_extraction(
            extracted_id, "file-B", Path("/artifacts/b.md"), "B", "done", None
        )
        worker._write_back_extraction(
            extracted_id, "file-A", Path("/artifacts/a.md"), "A", "done", None
        )

        with session_factory() as session:
            stranded = (
                session.query(ChunkedDocument)
                .join(
                    ExtractedDocument,
                    ChunkedDocument.extracted_document_id == ExtractedDocument.id,
                )
                .filter(ChunkedDocument.status == "done")
                .filter(ChunkedDocument.source_content_hash == ExtractedDocument.content_hash)
                .filter(~ChunkedDocument.chunks.any())
                .count()
            )
        assert stranded == 0

    @pytest.mark.parametrize(
        "door", ["purge_all_chunks", "prune_history", "watcher_purge", "delete_collection"]
    )
    def test_every_other_delete_site_keeps_the_counted_rule(
        self, temp_dir: Path, door: str
    ) -> None:
        """B4's walk: the four delete sites besides the superseded purge.

        Each door runs for real against the seeded document, then the property is
        read with `chunked_scope` itself (so it counts exactly what status and
        completion count): no chunking that counts toward a revision may own zero
        chunks. SQLite enforces no foreign keys, so this asserts on what is
        counted, never on cascades (D8 covers those on PostgreSQL).
        """
        from sqlalchemy import select

        from cementic.collections import delete_collection_records
        from cementic.revisions import (
            chunked_scope,
            prune_collection_history,
            purge_all_chunks,
        )
        from cementic.source_watcher import _purge_document_chunks

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        revision_id, extracted_id, chunked_id, source_id = self._seed(session_factory)

        with session_factory() as session:
            active = session.get(PipelineRevision, revision_id)
            extractor_id = active.extractor_profile_id
            embedding_id = active.embedding_profile_id
            if door == "prune_history":
                # A second chunk profile built for a revision that is now
                # superseded: its chunking and chunks are what the prune removes.
                old_profile = ChunkProfile(fingerprint="old", config_json="{}")
                session.add(old_profile)
                session.flush()
                session.add(
                    PipelineRevision(
                        collection="revert",
                        extractor_profile_id=extractor_id,
                        chunk_profile_id=old_profile.id,
                        embedding_profile_id=embedding_id,
                        status="superseded",
                    )
                )
                old_chunked = ChunkedDocument(
                    extracted_document_id=extracted_id, chunk_profile_id=old_profile.id,
                    source_content_hash="A", status="done", total_chunks=1,
                )
                session.add(old_chunked)
                session.flush()
                session.add(
                    Chunk(document_id=source_id, chunked_document_id=old_chunked.id,
                          chunk_index=0, content="old profile")
                )
                active.status = "active"
            session.commit()
            profile_ids = {
                row.chunk_profile_id for row in session.query(PipelineRevision).all()
            }

        with session_factory() as session:
            if door == "purge_all_chunks":
                purge_all_chunks(session, extracted_id)
            elif door == "prune_history":
                prune_collection_history(session, "revert")
            elif door == "watcher_purge":
                _purge_document_chunks(session, [source_id])
            else:
                delete_collection_records(session, "revert")
            session.commit()

        with session_factory() as session:
            for profile_id in profile_ids:
                revision = PipelineRevision(
                    collection="revert", extractor_profile_id=extractor_id,
                    chunk_profile_id=profile_id, embedding_profile_id=embedding_id,
                )
                stranded = (
                    session.query(ChunkedDocument)
                    .join(
                        ExtractedDocument,
                        ChunkedDocument.extracted_document_id == ExtractedDocument.id,
                    )
                    .join(SourceDocument, ExtractedDocument.document_id == SourceDocument.id)
                    .filter(*chunked_scope(revision))
                    .filter(~select(Chunk.id).where(
                        Chunk.chunked_document_id == ChunkedDocument.id).exists())
                    .count()
                )
                assert stranded == 0, f"{door}: a counted chunking owns no chunks"
            if door == "prune_history":
                # The prune must not have emptied the kept revision's chunking.
                assert session.query(Chunk).filter_by(chunked_document_id=chunked_id).count() == 2


class TestNoReadyTransitionMidScan:
    """B5: draining the documents registered so far is not finishing the scan.

    `revision_is_complete` counts what the watcher has registered, and the
    watcher registers one file at a time, so a worker that drains the first N
    files saw "everything extracted" and marked the revision ready while the
    scan was still walking the tree.
    """

    @staticmethod
    def _setup(temp_dir: Path):
        from cementic.source_watcher import SourceWatcher

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        # chunks=0: one fully extracted and chunked document, so by the counts
        # alone this revision is complete.
        revision_id, _, _, _ = TestEveryPurgeReopensTheWork._seed(session_factory, chunks=0)
        worker = TestEveryPurgeReopensTheWork._worker(temp_dir, session_factory)
        config = Config()
        config.source_watcher.state_path = temp_dir / "watcher.json"
        config.source_watcher.log_file = temp_dir / "watcher.log"
        watcher = SourceWatcher(config)
        watcher.Session = session_factory
        watcher.collection = "revert"
        return session_factory, revision_id, worker, watcher

    @staticmethod
    def _status(session_factory, revision_id: int) -> str:
        with session_factory() as session:
            return session.get(PipelineRevision, revision_id).status

    def test_the_worker_does_not_mark_ready_while_the_scan_runs(self, temp_dir: Path) -> None:
        session_factory, revision_id, worker, watcher = self._setup(temp_dir)
        root = temp_dir / "papers"
        root.mkdir()
        seen: list[str] = []

        def mid_scan(directory: Path) -> None:
            # The worker has drained the documents registered so far and polls.
            with patch("cementic.pipeline_worker.ensure_revision_ann_index"):
                worker._mark_revision_ready_if_complete(revision_id)
            seen.append(self._status(session_factory, revision_id))

        with (
            patch("cementic.source_watcher.Observer"),
            patch.object(watcher, "_scan_existing", side_effect=mid_scan),
            patch.object(watcher, "_reconcile_deletions"),
        ):
            watcher._start_watcher([str(root)])

        assert seen == ["building"], "ready was reached before the scan finished"
        # Once the scan has finished, the same poll may mark it ready.
        with patch("cementic.pipeline_worker.ensure_revision_ann_index"):
            worker._mark_revision_ready_if_complete(revision_id)
        assert self._status(session_factory, revision_id) == "ready"

    def test_an_interrupted_scan_never_counts_as_finished(self, temp_dir: Path) -> None:
        session_factory, revision_id, worker, watcher = self._setup(temp_dir)
        root = temp_dir / "papers"
        root.mkdir()

        def stopped(directory: Path) -> None:
            watcher._shutdown_event.set()

        with (
            patch("cementic.source_watcher.Observer"),
            patch.object(watcher, "_scan_existing", side_effect=stopped),
            patch.object(watcher, "_reconcile_deletions"),
        ):
            watcher._start_watcher([str(root)])
        with patch("cementic.pipeline_worker.ensure_revision_ann_index"):
            worker._mark_revision_ready_if_complete(revision_id)
        assert self._status(session_factory, revision_id) == "building"

    def test_a_database_with_no_scan_record_is_not_held_back(self, temp_dir: Path) -> None:
        """An existing database has no marker row; that must not wedge it.

        No row means no watcher has recorded a scan, which is every database
        from before the marker existed: the old behaviour stands until the next
        watcher start writes one.
        """
        session_factory, revision_id, worker, _ = self._setup(temp_dir)
        with patch("cementic.pipeline_worker.ensure_revision_ann_index"):
            worker._mark_revision_ready_if_complete(revision_id)
        assert self._status(session_factory, revision_id) == "ready"

    def test_counts_report_an_open_scan_and_completion_refuses_it(self, temp_dir: Path) -> None:
        from dataclasses import replace

        session_factory, revision_id, _, watcher = self._setup(temp_dir)
        with session_factory() as session:
            revision = session.get(PipelineRevision, revision_id)
            counts = compute_revision_counts(session, "revert", revision)
        assert revision_is_complete(counts) is True
        assert revision_is_complete(replace(counts, scan_in_progress=True)) is False

    def _ready_with_open_scan(self, temp_dir: Path):
        """A complete, ready revision whose collection has an open scan row."""
        session_factory, revision_id, worker, watcher = self._setup(temp_dir)
        with patch("cementic.pipeline_worker.ensure_revision_ann_index"):
            worker._mark_revision_ready_if_complete(revision_id)
        assert self._status(session_factory, revision_id) == "ready"
        with session_factory() as session:
            session.add(CollectionScan(collection="revert"))
            session.commit()
        return session_factory, revision_id

    def test_an_open_scan_does_not_block_promoting_a_complete_ready_revision(
        self, temp_dir: Path
    ) -> None:
        """The marker gates only building -> ready; every `cementic start` reopens it."""
        from cementic.collections import promote_ready_revision

        session_factory, _ = self._ready_with_open_scan(temp_dir)
        config = Config()
        config.storage.artifacts_path = temp_dir
        with session_factory() as session:
            outcome = promote_ready_revision(session, "revert", config=config)
        assert outcome.status == "promoted"

    def test_promote_and_remove_work_before_the_scan_table_exists(self, temp_dir: Path) -> None:
        """An upgraded install has no `collection_scans` until the next start."""
        from cementic.collections import delete_collection_records, promote_ready_revision

        session_factory, revision_id, worker, _ = self._setup(temp_dir)
        with patch("cementic.pipeline_worker.ensure_revision_ann_index"):
            worker._mark_revision_ready_if_complete(revision_id)
        engine = session_factory.kw["bind"]
        CollectionScan.__table__.drop(engine)
        config = Config()
        config.storage.artifacts_path = temp_dir
        with session_factory() as session:
            assert promote_ready_revision(session, "revert", config=config).status == "promoted"
        with session_factory() as session:
            assert delete_collection_records(session, "revert") is not None


class _PgError(Exception):
    """Stands in for a psycopg error: the server's SQLSTATE rides on `.sqlstate`."""

    def __init__(self, sqlstate: str | None, message: str = "db error") -> None:
        super().__init__(message)
        self.sqlstate = sqlstate


def _db_error(sqlstate: str | None, kind: type = None):
    from sqlalchemy import exc

    kind = kind or exc.OperationalError
    return kind("UPDATE ...", {}, _PgError(sqlstate))


class TestWriteBackErrorClasses:
    """B2b: which database failures during write-back retry the document.

    A deadlock or statement timeout is a property of the moment, not of the
    document, so stamping the document terminal for the run (B2) misreports it.
    A data error is the document's own and stays terminal; a lost connection is
    the database's and belongs to the loop's back-off.
    """

    @pytest.mark.parametrize("sqlstate", ["40001", "40P01", "57014", "55P03"])
    def test_serialization_deadlock_and_timeouts_retry(self, sqlstate: str) -> None:
        from cementic.pipeline_worker import classify_write_back_error

        assert classify_write_back_error(_db_error(sqlstate)) == "transient"

    @pytest.mark.parametrize("sqlstate", ["22021", "22P05", "23505", "54000", "42703"])
    def test_data_and_schema_errors_are_the_documents(self, sqlstate: str) -> None:
        from sqlalchemy import exc

        from cementic.pipeline_worker import classify_write_back_error

        assert classify_write_back_error(_db_error(sqlstate, exc.DataError)) == "data"

    @pytest.mark.parametrize("sqlstate", ["08006", "08001", "57P01", "57P03", "53300", None])
    def test_lost_connections_belong_to_the_loop(self, sqlstate: str | None) -> None:
        from cementic.pipeline_worker import classify_write_back_error

        assert classify_write_back_error(_db_error(sqlstate)) == "connectivity"

    @pytest.mark.parametrize("name", ["DiskFull", "OutOfMemory", "ConfigurationLimitExceeded"])
    def test_server_resource_errors_are_the_servers_not_the_documents(self, name: str) -> None:
        """Class 53 (disk full, out of memory) is an outage; the document is not at fault."""
        import psycopg.errors
        from sqlalchemy import exc

        from cementic.pipeline_worker import classify_write_back_error

        cls = getattr(psycopg.errors, name)
        wrapped = exc.OperationalError("INSERT ...", {}, cls("server out of resources"))
        assert wrapped.orig.sqlstate.startswith("53")
        assert classify_write_back_error(wrapped) == "connectivity"

    def test_pool_exhaustion_and_invalidated_connections_are_connectivity(self) -> None:
        from sqlalchemy import exc

        from cementic.pipeline_worker import classify_write_back_error

        invalidated = _db_error("57P01")
        invalidated.connection_invalidated = True
        assert classify_write_back_error(invalidated) == "connectivity"
        assert classify_write_back_error(exc.TimeoutError("pool")) == "connectivity"

    def test_anything_that_is_not_a_database_error_is_the_documents(self) -> None:
        from cementic.pipeline_worker import classify_write_back_error

        assert classify_write_back_error(RuntimeError("deadlock detected")) == "data"
        assert classify_write_back_error(ValueError("NUL")) == "data"


class TestTransientWriteBackIsRetriedNotFailed:
    @staticmethod
    def _extract_ready(temp_dir: Path):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        revision_id, extracted_id, chunked_id, _ = TestEveryPurgeReopensTheWork._seed(
            session_factory
        )
        worker = TestEveryPurgeReopensTheWork._worker(temp_dir, session_factory)
        with session_factory() as session:
            session.get(SourceDocument, 1).file_hash = "file-B"
            session.get(ExtractedDocument, extracted_id).status = "pending"
            session.commit()
        return session_factory, revision_id, extracted_id, chunked_id, worker

    @staticmethod
    def _step(worker, revision_id: int, error: Exception) -> None:
        with (
            patch("cementic.pipeline_worker.extract_document", return_value="text from B"),
            patch("cementic.pipeline_worker.write_extracted_text", return_value="B"),
            patch.object(worker, "_write_back_extraction", side_effect=error),
        ):
            worker._step_extract(revision_id)

    def test_a_deadlock_returns_the_document_to_the_queue(self, temp_dir: Path) -> None:
        session_factory, revision_id, extracted_id, _, worker = self._extract_ready(temp_dir)

        self._step(worker, revision_id, _db_error("40P01"))

        with session_factory() as session:
            extracted = session.get(ExtractedDocument, extracted_id)
            assert extracted.status == "pending"
            assert extracted.attempts == 1
        state = worker.state_manager.load()
        assert state.retried_count == 1

        # The retry succeeds: the same claim goes through and nothing was failed.
        with (
            patch("cementic.pipeline_worker.extract_document", return_value="text from B"),
            patch("cementic.pipeline_worker.write_extracted_text", return_value="B"),
        ):
            worker._step_extract(revision_id)
        with session_factory() as session:
            assert session.get(ExtractedDocument, extracted_id).status == "done"

    def test_a_document_that_keeps_deadlocking_is_failed_after_the_claim_cap(
        self, temp_dir: Path
    ) -> None:
        from cementic.revisions import MAX_INTERRUPTED_CLAIMS

        session_factory, revision_id, extracted_id, _, worker = self._extract_ready(temp_dir)

        for _ in range(MAX_INTERRUPTED_CLAIMS):
            self._step(worker, revision_id, _db_error("40001"))

        with session_factory() as session:
            extracted = session.get(ExtractedDocument, extracted_id)
            assert extracted.status == "failed"
            assert "40001" in (extracted.error_message or "")
        assert worker.state_manager.load().retried_count == MAX_INTERRUPTED_CLAIMS - 1

    def test_a_data_error_is_still_terminal_at_once(self, temp_dir: Path) -> None:
        from sqlalchemy import exc

        session_factory, revision_id, extracted_id, _, worker = self._extract_ready(temp_dir)

        self._step(worker, revision_id, _db_error("22021", exc.DataError))

        with session_factory() as session:
            assert session.get(ExtractedDocument, extracted_id).status == "failed"
        assert worker.state_manager.load().retried_count == 0

    def test_a_lost_connection_reaches_the_loop_and_fails_nothing(self, temp_dir: Path) -> None:
        session_factory, revision_id, extracted_id, _, worker = self._extract_ready(temp_dir)

        with pytest.raises(Exception) as raised:
            self._step(worker, revision_id, _db_error("08006"))
        assert getattr(raised.value.orig, "sqlstate", None) == "08006"

        with session_factory() as session:
            assert session.get(ExtractedDocument, extracted_id).status != "failed"

    def test_a_chunk_write_back_deadlock_is_retried_too(self, temp_dir: Path) -> None:
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        revision_id, extracted_id, chunked_id, _ = TestEveryPurgeReopensTheWork._seed(
            session_factory
        )
        worker = TestEveryPurgeReopensTheWork._worker(temp_dir, session_factory)
        with session_factory() as session:
            session.get(ExtractedDocument, extracted_id).content_hash = "B"
            session.commit()

        with (
            patch("cementic.pipeline_worker.read_extracted_text", return_value="text from B " * 40),
            patch.object(PipelineWorker, "_write_back_chunks", side_effect=_db_error("40P01")),
        ):
            assert worker._step_chunk(revision_id) is True

        with session_factory() as session:
            chunked = session.get(ChunkedDocument, chunked_id)
            assert chunked.status == "pending"
        assert worker.state_manager.load().retried_count == 1
