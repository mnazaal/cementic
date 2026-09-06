"""Tests for status query helpers."""

from unittest.mock import MagicMock, patch

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
from cementic.embedding_runtime import RemoteEmbeddingClient
from cementic.pipeline_worker import compute_revision_counts
from cementic.state import DaemonState, WorkerState
from cementic.status_service import (
    HealthStatus,
    PipelineStatus,
    WorkerStatus,
    _safe_pct,
    _select_target_revision,
    build_supervisor_status,
    build_worker_status,
    check_health,
    daemon_state_text,
    load_file_progress,
    load_pipeline_status,
    load_pipeline_status_bulk,
    load_worker_statuses,
)


def _count_select_queries(engine):
    class _Counter:
        value = 0

    counter = _Counter()

    def _on_execute(conn, cursor, statement, *args, **kwargs):
        if statement.lstrip().upper().startswith("SELECT"):
            counter.value += 1

    event.listen(engine, "before_cursor_execute", _on_execute)
    return counter


def _seed_revision(session, collection: str) -> PipelineRevision:
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
    revision = PipelineRevision(
        collection=collection,
        extractor_profile_id=extractor.id,
        chunk_profile_id=chunk_profile.id,
        embedding_profile_id=embedding_profile.id,
        status="building",
    )
    session.add(revision)
    session.flush()
    return revision


def _seed_collection_with_documents(
    session, collection: str, n_documents: int, n_failed_embeddings: int = 0
) -> PipelineRevision:
    """Seed a collection with n_documents fully done through embedding."""
    revision = _seed_revision(session, collection)
    revision.status = "active"
    for i in range(n_documents):
        doc = SourceDocument(
            collection=collection, source_path=f"/{collection}/{i}.pdf", file_hash=f"h{i}"
        )
        session.add(doc)
        session.flush()
        # Hashes must mirror what the real pipeline writes: extraction records
        # the source hash it consumed, chunking records the content hash it
        # consumed. Progress is only "done" when those still match upstream.
        extracted = ExtractedDocument(
            document_id=doc.id,
            extractor_profile_id=revision.extractor_profile_id,
            source_file_hash=f"h{i}",
            content_hash=f"c{i}",
            status="done",
        )
        session.add(extracted)
        session.flush()
        chunked = ChunkedDocument(
            extracted_document_id=extracted.id,
            chunk_profile_id=revision.chunk_profile_id,
            source_content_hash=f"c{i}",
            status="done",
        )
        session.add(chunked)
        session.flush()
        chunk = Chunk(
            document_id=doc.id, chunked_document_id=chunked.id, chunk_index=0, content="c"
        )
        session.add(chunk)
        session.flush()
        status = "failed" if i < n_failed_embeddings else "done"
        session.add(
            ChunkEmbedding(
                chunk_id=chunk.id, embedding_profile_id=revision.embedding_profile_id, status=status
            )
        )
    return revision


class TestStatusAgreesWithWorkerCounts:
    """`cementic status` must report exactly what the worker counts.

    These are two separate query paths over the same data. They carried
    duplicate scope definitions and drifted: status omitted the source/content
    hash equalities and bucketed every non-"done" chunking as failed. The result
    was a status panel that reported queued work as failures and showed 100%
    extracted for files that had changed on disk and still owed re-extraction --
    while the revision quietly never reached `ready`.
    """

    def _session_factory(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        return engine, sessionmaker(bind=engine, expire_on_commit=False)

    def _status_for(self, engine, session_factory, collection: str):
        with (
            patch("cementic.status_service.get_engine", return_value=engine),
            patch("cementic.status_service.get_session_factory", return_value=session_factory),
        ):
            return load_pipeline_status_bulk(Config(), [collection])[collection]

    def test_queued_chunking_is_not_reported_as_failed(self) -> None:
        engine, session_factory = self._session_factory()
        with session_factory() as session:
            revision = _seed_revision(session, "queued")
            doc = SourceDocument(collection="queued", source_path="/a.pdf", file_hash="h")
            session.add(doc)
            session.flush()
            extracted = ExtractedDocument(
                document_id=doc.id,
                extractor_profile_id=revision.extractor_profile_id,
                source_file_hash="h",
                content_hash="c",
                status="done",
            )
            session.add(extracted)
            session.flush()
            session.add(
                ChunkedDocument(
                    extracted_document_id=extracted.id,
                    chunk_profile_id=revision.chunk_profile_id,
                    source_content_hash="c",
                    status="pending",  # merely queued, not failed
                )
            )
            session.commit()
            worker_counts = compute_revision_counts(session, "queued", revision)

        status = self._status_for(engine, session_factory, "queued")
        assert status.chunked_failed == 0
        assert status.chunked_failed == worker_counts.chunked_failed

    def test_chunks_under_an_unfinished_chunking_are_not_counted(self) -> None:
        """`total_chunks` must count only what the embed step will drain.

        _step_embed claims chunks whose ChunkedDocument is "done". Counting
        chunks under any other status put rows in the denominator that could
        never reach done or failed, so `done + failed == total_chunks` was
        unreachable and the revision stayed "building" forever.
        """
        engine, session_factory = self._session_factory()
        with session_factory() as session:
            revision = _seed_revision(session, "partial")
            doc = SourceDocument(collection="partial", source_path="/a.pdf", file_hash="h")
            session.add(doc)
            session.flush()
            extracted = ExtractedDocument(
                document_id=doc.id,
                extractor_profile_id=revision.extractor_profile_id,
                source_file_hash="h",
                content_hash="c",
                status="done",
            )
            session.add(extracted)
            session.flush()
            # Chunking is still in flight; its chunks are not embed candidates.
            chunked = ChunkedDocument(
                extracted_document_id=extracted.id,
                chunk_profile_id=revision.chunk_profile_id,
                source_content_hash="c",
                status="processing",
            )
            session.add(chunked)
            session.flush()
            session.add(
                Chunk(
                    document_id=doc.id,
                    chunked_document_id=chunked.id,
                    chunk_index=0,
                    content="c",
                )
            )
            session.commit()
            worker_counts = compute_revision_counts(session, "partial", revision)

        status = self._status_for(engine, session_factory, "partial")
        assert worker_counts.total_chunks == 0
        assert status.total_chunks == worker_counts.total_chunks

    def test_stale_extraction_is_not_counted_as_done(self) -> None:
        engine, session_factory = self._session_factory()
        with session_factory() as session:
            revision = _seed_revision(session, "stale")
            # The watched file changed on disk: its hash moved on, but the
            # existing extraction still carries the old one.
            doc = SourceDocument(collection="stale", source_path="/a.pdf", file_hash="new")
            session.add(doc)
            session.flush()
            session.add(
                ExtractedDocument(
                    document_id=doc.id,
                    extractor_profile_id=revision.extractor_profile_id,
                    source_file_hash="old",
                    content_hash="c",
                    status="done",
                )
            )
            session.commit()
            worker_counts = compute_revision_counts(session, "stale", revision)

        status = self._status_for(engine, session_factory, "stale")
        assert status.extracted_done == 0
        assert status.extraction_pct == 0.0
        assert status.extracted_done == worker_counts.extracted_done


class TestLoadPipelineStatusBulk:
    """Tests for load_pipeline_status_bulk."""

    def _session_factory(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        return engine, sessionmaker(bind=engine, expire_on_commit=False)

    def test_empty_collections_returns_empty_dict(self) -> None:
        assert load_pipeline_status_bulk(Config(), []) == {}

    def test_computes_correct_per_collection_counts(self) -> None:
        engine, session_factory = self._session_factory()
        with session_factory() as session:
            _seed_collection_with_documents(session, "research", n_documents=3)
            _seed_collection_with_documents(
                session, "math", n_documents=2, n_failed_embeddings=1
            )
            session.commit()

        with (
            patch("cementic.status_service.get_engine", return_value=engine),
            patch("cementic.status_service.get_session_factory", return_value=session_factory),
        ):
            result = load_pipeline_status_bulk(Config(), ["research", "math", "missing"])

        assert result["research"].documents == 3
        assert result["research"].extracted_done == 3
        assert result["research"].done_embeddings == 3
        assert result["research"].failed_embeddings == 0

        assert result["math"].documents == 2
        assert result["math"].done_embeddings == 1
        assert result["math"].failed_embeddings == 1

        assert result["missing"].documents == 0
        assert result["missing"].done_embeddings == 0
        assert result["missing"].active_revision_label is None

    def test_matches_single_collection_loader(self) -> None:
        engine, session_factory = self._session_factory()
        with session_factory() as session:
            _seed_collection_with_documents(session, "research", n_documents=3)
            session.commit()

        with (
            patch("cementic.status_service.get_engine", return_value=engine),
            patch("cementic.status_service.get_session_factory", return_value=session_factory),
        ):
            single = load_pipeline_status(Config(), "research")
            bulk = load_pipeline_status_bulk(Config(), ["research"])["research"]

        assert single == bulk

    def test_query_count_does_not_scale_with_collection_count(self) -> None:
        def run(n_collections: int) -> int:
            engine, session_factory = self._session_factory()
            names = [f"col-{i}" for i in range(n_collections)]
            with session_factory() as session:
                for name in names:
                    _seed_collection_with_documents(session, name, n_documents=2)
                session.commit()

            counter = _count_select_queries(engine)
            with (
                patch("cementic.status_service.get_engine", return_value=engine),
                patch("cementic.status_service.get_session_factory", return_value=session_factory),
            ):
                result = load_pipeline_status_bulk(Config(), names)
            assert len(result) == n_collections
            return counter.value

        small = run(3)
        large = run(30)
        assert small == large


class TestSafePct:
    """Tests for _safe_pct helper."""

    def test_normal(self) -> None:
        assert _safe_pct(3, 10) == 30.0

    def test_zero_total(self) -> None:
        assert _safe_pct(5, 0) == 0.0

    def test_zero_done(self) -> None:
        assert _safe_pct(0, 5) == 0.0

    def test_rounds_to_one_decimal(self) -> None:
        assert _safe_pct(1, 3) == 33.3

    def test_never_rounds_up_to_complete(self) -> None:
        # 2,161 chunks short of 2.16M rounds to 100.0 at one decimal place, and
        # a status line reading 100.0% is what a reader stops watching.
        assert _safe_pct(2_160_000 - 2_161, 2_160_000) == 99.9

    def test_reaches_complete_only_when_done(self) -> None:
        assert _safe_pct(2_160_000, 2_160_000) == 100.0


class TestSelectTargetRevision:
    """Tests for pure revision selection policy."""

    def test_prefers_building_revision(self) -> None:
        building = MagicMock()
        active = MagicMock()

        result = _select_target_revision(building, active)

        assert result is building

    def test_falls_back_to_active_revision(self) -> None:
        active = MagicMock()

        result = _select_target_revision(None, active)

        assert result is active

    def test_returns_none_when_no_revision_exists(self) -> None:
        assert _select_target_revision(None, None) is None


class TestDaemonStateText:
    """Tests for daemon_state_text."""

    def test_enum_value(self) -> None:
        assert daemon_state_text(DaemonState.RUNNING) == "running"

    def test_plain_string(self) -> None:
        assert daemon_state_text("running") == "running"

    def test_unknown_object(self) -> None:
        assert daemon_state_text(42) == "42"


class TestBuildWorkerStatus:
    """Tests for build_worker_status."""

    @patch("cementic.status_service.is_managed_process_alive", return_value=True)
    def test_running_worker(self, mock_running) -> None:
        state = WorkerState(
            daemon_state=DaemonState.RUNNING,
            pid=1234,
            watched_directories=["/a", "/b"],
            processed_count=10,
            current_file="/tmp/foo.pdf",
        )
        result = build_worker_status(state)
        assert result.state == "running"
        assert result.pid == "1234"
        assert result.process == "running"
        assert "/a" in str(result.watched_directories)
        assert result.processed_count == 10
        assert result.failed_count == 0

    @patch("cementic.status_service.is_managed_process_alive", return_value=False)
    def test_stopped_worker(self, mock_running) -> None:
        state = WorkerState(pid=None)
        result = build_worker_status(state)
        assert result.pid == "N/A"
        assert result.process == "stopped"

    def test_watched_directories_normalizes_non_list(self) -> None:
        """Line 106: non-list watched_directories (e.g. string) → wrapped in list."""
        state = WorkerState(watched_directories="/single/string/path")
        result = build_worker_status(state)
        assert "/single/string/path" in str(result.watched_directories)


class TestBuildSupervisorStatus:
    """Tests for build_supervisor_status."""

    def test_empty_process_list(self) -> None:
        result = build_supervisor_status({"processes": [], "collection": "c1"})
        assert result.state == "not started"

    def test_missing_processes_key(self) -> None:
        result = build_supervisor_status({})
        assert result.state == "not started"

    @patch("cementic.status_service.is_managed_process_alive", return_value=True)
    def test_all_running(self, mock_running) -> None:
        data = {
            "processes": [{"pid": 1}, {"pid": 2}],
            "collection": "mycoll",
            "directories": ["/watched"],
        }
        result = build_supervisor_status(data)
        assert result.state == "2/2 running"
        assert result.collection == "mycoll"
        assert result.directories == ["/watched"]

    @patch("cementic.status_service.is_managed_process_alive", side_effect=[True, False])
    def test_mixed_processes(self, mock_running) -> None:
        data = {"processes": [{"pid": 1}, {"pid": 2}]}
        result = build_supervisor_status(data)
        assert result.state == "1/2 running"

    def test_non_dict_process_items_returns_empty(self) -> None:
        """Line 140: all process items are non-dicts → process_rows empty."""
        result = build_supervisor_status({"processes": ["not-a-dict", 42, None]})
        assert result.state == "not started"


class TestLoadWorkerStatuses:
    """Tests for load_worker_statuses."""

    @patch("cementic.status_service.StateManager")
    def test_returns_statuses(self, mock_sm_cls, temp_dir) -> None:
        from cementic.config import Config

        config = Config()
        config.source_watcher.state_path = temp_dir / "sw.json"
        config.pipeline_worker.state_path = temp_dir / "pw.json"

        mock_sm = mock_sm_cls.return_value
        mock_sm.load.return_value = WorkerState()

        sw_status, pw_status = load_worker_statuses(config)
        assert isinstance(sw_status, WorkerStatus)
        assert isinstance(pw_status, WorkerStatus)


class TestDataClasses:
    """Tests for status dataclass structures."""

    def test_worker_status_creation(self) -> None:
        ws = WorkerStatus(
            state="running",
            pid="1234",
            process="running",
            current_file="f.pdf",
            watched_directories=["/a"],
            processed_count=5,
            failed_count=1,
        )
        assert ws.failed_count == 1

    def test_pipeline_status_creation(self) -> None:
        ps = PipelineStatus(
            documents=10,
            extracted_done=8,
            extracted_failed=2,
            chunked_done=7,
            chunked_failed=1,
            total_chunks=50,
            pending_embeddings=10,
            processing_embeddings=5,
            done_embeddings=32,
            failed_embeddings=3,
            extraction_pct=80.0,
            chunking_pct=87.5,
            embedding_pct=64.0,
            active_revision_label="v1",
            ready_revision_label=None,
            building_revision_label="v2",
        )
        assert ps.documents == 10
        assert ps.active_revision_label == "v1"

    def test_health_status_creation(self) -> None:
        hs = HealthStatus(
            db_reachable=True,
            embedding_provider="llama-cpp",
            embedding_healthy=True,
            llama_daemon="N/A",
        )
        assert hs.db_reachable is True
        assert hs.embedding_provider == "llama-cpp"


class TestCheckHealth:
    """Tests for check_health runtime health checks."""

    def test_db_reachable(self) -> None:
        config = Config()
        with patch("cementic.status_service.get_engine") as mock_engine:
            with patch("cementic.embedding_runtime.build_llama_cpp_client") as mock_client:
                mock_conn = MagicMock()
                mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
                mock_client.return_value.health_check.return_value = True
                mock_client.return_value.expected_fingerprint = "a" * 64
                result = check_health(config)
                assert result.db_reachable is True

    @patch("cementic.status_service.get_engine", side_effect=Exception("db down"))
    def test_db_unreachable(self, mock_engine) -> None:
        config = Config()
        result = check_health(config)
        assert result.db_reachable is False

    @patch("cementic.status_service.get_engine")
    @patch("cementic.embedding_runtime.build_llama_cpp_client")
    def test_health_llama_cpp_healthy(self, mock_client, mock_engine) -> None:
        config = Config()
        config.pipeline.embedding_provider = "llama-cpp"
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
        client = MagicMock(spec=RemoteEmbeddingClient)
        client.probe_served_runtime.return_value = True
        client.probe_embedding.return_value = True
        mock_client.return_value = client

        result = check_health(config)
        assert result.embedding_provider == "llama-cpp"
        assert result.embedding_healthy is True

    @patch("cementic.status_service.get_engine")
    @patch("cementic.embedding_runtime.build_llama_cpp_client")
    def test_health_llama_cpp_unhealthy(self, mock_client, mock_engine) -> None:
        config = Config()
        config.pipeline.embedding_provider = "llama-cpp"
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
        mock_client.return_value.health_check.return_value = False
        mock_client.return_value.expected_fingerprint = "a" * 64

        result = check_health(config)
        assert result.embedding_healthy is False

    @patch("cementic.status_service.get_engine")
    @patch("cementic.embedding_runtime.build_llama_cpp_client")
    def test_a_wedged_daemon_is_unhealthy_with_a_restart_hint(
        self, mock_client, mock_engine
    ) -> None:
        """Regression (2026-08-15): a daemon answered /v1/models but hung every
        embedding for 21 hours while `status` said `embedding healthy`. The
        model list is served without the model lock, so only an actual embedding
        round trip can see this state."""
        from cementic.embedding_runtime import RemoteEmbeddingClient

        config = Config()
        config.pipeline.embedding_provider = "llama-cpp"
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
        client = MagicMock(spec=RemoteEmbeddingClient)
        client.probe_served_runtime.return_value = True
        client.probe_embedding.return_value = False
        mock_client.return_value = client

        with patch(
            "cementic.embedding_runtime._worker_load_explains_slow_embeddings",
            return_value=False,
        ):
            result = check_health(config)

        assert result.embedding_healthy is False
        assert "not answering embeddings" in result.llama_daemon
        assert "embedding stop" in result.llama_daemon

    @patch("cementic.status_service.get_engine")
    @patch("cementic.embedding_runtime.build_llama_cpp_client")
    def test_a_daemon_saturated_by_a_live_worker_stays_healthy(
        self, mock_client, mock_engine
    ) -> None:
        """The same probe timeout during a worker batch is load, not a wedge."""
        from cementic.embedding_runtime import RemoteEmbeddingClient

        config = Config()
        config.pipeline.embedding_provider = "llama-cpp"
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
        client = MagicMock(spec=RemoteEmbeddingClient)
        client.probe_served_runtime.return_value = True
        client.probe_embedding.return_value = False
        mock_client.return_value = client

        with patch(
            "cementic.embedding_runtime._worker_load_explains_slow_embeddings",
            return_value=True,
        ):
            result = check_health(config)

        assert result.embedding_healthy is True

    def test_health_llama_cpp_busy_not_unhealthy(self, temp_dir) -> None:
        """A daemon mid-batch cannot answer /v1/models, because llama_cpp.server
        serializes every request behind one lock. With the process confirmed
        alive, that is busy, not unhealthy -- and the probe must reach that
        verdict without waiting the batch out."""
        config = Config()
        config.pipeline.embedding_provider = "llama-cpp"
        pid_file = temp_dir / "daemon.pid"
        pid_file.write_text("42")
        config.llama_cpp.daemon_pid_file = pid_file

        client = MagicMock(spec=RemoteEmbeddingClient)
        client.probe_served_runtime.return_value = None  # no answer

        with (
            patch("cementic.status_service.get_engine"),
            patch("cementic.embedding_runtime.is_managed_process_alive", return_value=True),
            patch("cementic.embedding_runtime.build_llama_cpp_client", return_value=client),
            patch("cementic.embedding_runtime._poll_until_ready") as mock_poll,
        ):
            result = check_health(config)

        assert "running" in result.llama_daemon
        assert result.embedding_healthy is True
        mock_poll.assert_not_called()

    def test_health_llama_cpp_wrong_model_is_not_healthy(self, temp_dir) -> None:
        """The daemon answered; what it serves is not what this config asks for.

        The old pid-file fallback flipped this back to healthy, so a stale
        daemon looked fine to `status` while search and the worker restarted it
        from under each other.
        """
        config = Config()
        config.pipeline.embedding_provider = "llama-cpp"
        pid_file = temp_dir / "daemon.pid"
        pid_file.write_text("42")
        config.llama_cpp.daemon_pid_file = pid_file

        client = MagicMock(spec=RemoteEmbeddingClient)
        client.probe_served_runtime.return_value = False  # answered, but not ours

        with (
            patch("cementic.status_service.get_engine"),
            patch("cementic.embedding_runtime.is_managed_process_alive", return_value=True),
            patch("cementic.embedding_runtime.build_llama_cpp_client", return_value=client),
        ):
            result = check_health(config)

        assert result.embedding_healthy is False
        assert "different model" in result.llama_daemon

    def test_health_llama_daemon_running(self, temp_dir) -> None:
        config = Config()
        config.pipeline.embedding_provider = "llama-cpp"
        pid_file = temp_dir / "daemon.pid"
        pid_file.write_text("42")
        config.llama_cpp.daemon_pid_file = pid_file

        with patch("cementic.status_service.get_engine"):
            with patch(
                "cementic.embedding_runtime.is_managed_process_alive", return_value=True
            ):
                with patch(
                    "cementic.embedding_runtime.build_llama_cpp_client"
                ) as mock_client:
                    mock_client.return_value.health_check.return_value = True
                    mock_client.return_value.expected_fingerprint = "a" * 64
                    result = check_health(config)
                    assert "running" in result.llama_daemon
                    assert "42" in result.llama_daemon

    def test_health_llama_daemon_stopped_no_pid_file(self) -> None:
        config = Config()
        config.pipeline.embedding_provider = "llama-cpp"
        config.llama_cpp.daemon_pid_file = None

        with patch("cementic.status_service.get_engine"):
            with patch(
                "cementic.embedding_runtime.build_llama_cpp_client"
            ) as mock_client:
                mock_client.return_value.health_check.return_value = True
                mock_client.return_value.expected_fingerprint = "a" * 64
                result = check_health(config)
                assert result.llama_daemon == "stopped"

    def test_health_llama_daemon_invalid_pid(self, temp_dir) -> None:
        config = Config()
        config.pipeline.embedding_provider = "llama-cpp"
        pid_file = temp_dir / "daemon.pid"
        pid_file.write_text("not-a-number")
        config.llama_cpp.daemon_pid_file = pid_file

        with patch("cementic.status_service.get_engine"):
            with patch(
                "cementic.embedding_runtime.build_llama_cpp_client"
            ) as mock_client:
                mock_client.return_value.health_check.return_value = True
                mock_client.return_value.expected_fingerprint = "a" * 64
                result = check_health(config)
                assert result.llama_daemon == "stopped"

    @patch("cementic.status_service.get_engine")
    @patch("cementic.embedding_runtime.build_llama_cpp_client")
    def test_a_failed_probe_keeps_its_reason(self, mock_client, mock_engine) -> None:
        """Regression: any probe exception was flattened to DOWN, so an
        ambiguous-PID refusal rendered as a benign "stopped (autostarts when
        needed)" -- for a state where autostart raises the same refusal."""
        from cementic.embedding_runtime import AmbiguousDaemonPidsError

        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
        client = MagicMock(spec=RemoteEmbeddingClient)
        client.probe_served_runtime.side_effect = AmbiguousDaemonPidsError([111, 222])
        mock_client.return_value = client

        result = check_health(Config())

        assert result.embedding_healthy is False
        assert result.llama_daemon_health is None
        assert "111, 222" in result.llama_daemon


class TestLoadFileProgress:
    """Tests for load_file_progress, against a real (in-memory) database."""

    def _session_factory(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        return engine, sessionmaker(bind=engine, expire_on_commit=False)

    def test_no_target_revision_returns_empty(self) -> None:
        engine, session_factory = self._session_factory()
        with (
            patch("cementic.status_service.get_engine", return_value=engine),
            patch("cementic.status_service.get_session_factory", return_value=session_factory),
        ):
            result = load_file_progress(Config(), "col")
        assert result == []

    def test_no_documents_returns_empty(self) -> None:
        engine, session_factory = self._session_factory()
        with session_factory() as session:
            _seed_revision(session, "col")
            session.commit()

        with (
            patch("cementic.status_service.get_engine", return_value=engine),
            patch("cementic.status_service.get_session_factory", return_value=session_factory),
        ):
            result = load_file_progress(Config(), "col")
        assert result == []

    def test_failed_extraction_sets_error_message(self) -> None:
        engine, session_factory = self._session_factory()
        with session_factory() as session:
            revision = _seed_revision(session, "col")
            doc = SourceDocument(collection="col", source_path="/test/a.pdf", file_hash="h1")
            session.add(doc)
            session.flush()
            session.add(
                ExtractedDocument(
                    document_id=doc.id,
                    extractor_profile_id=revision.extractor_profile_id,
                    status="failed",
                    source_file_hash="h1",
                    error_message="extraction broke",
                )
            )
            session.commit()

        with (
            patch("cementic.status_service.get_engine", return_value=engine),
            patch("cementic.status_service.get_session_factory", return_value=session_factory),
        ):
            result = load_file_progress(Config(), "col")
        assert len(result) == 1
        assert result[0].extraction_status == "failed"
        assert result[0].error_message == "extraction broke"

    def test_failed_chunking_sets_error_message(self) -> None:
        engine, session_factory = self._session_factory()
        with session_factory() as session:
            revision = _seed_revision(session, "col")
            doc = SourceDocument(collection="col", source_path="/test/a.pdf", file_hash="h1")
            session.add(doc)
            session.flush()
            # Both hashes are written by the worker on the failure path too, so
            # a realistic failed row still carries them.
            extracted = ExtractedDocument(
                document_id=doc.id,
                extractor_profile_id=revision.extractor_profile_id,
                source_file_hash="h1",
                content_hash="c1",
                status="done",
            )
            session.add(extracted)
            session.flush()
            session.add(
                ChunkedDocument(
                    extracted_document_id=extracted.id,
                    chunk_profile_id=revision.chunk_profile_id,
                    source_content_hash="c1",
                    status="failed",
                    error_message="chunker crashed",
                )
            )
            session.commit()

        with (
            patch("cementic.status_service.get_engine", return_value=engine),
            patch("cementic.status_service.get_session_factory", return_value=session_factory),
        ):
            result = load_file_progress(Config(), "col")
        assert len(result) == 1
        assert result[0].extraction_status == "done"
        assert result[0].chunking_status == "failed"
        assert result[0].error_message == "chunker crashed"

    def test_reports_embedding_counts_per_document(self) -> None:
        engine, session_factory = self._session_factory()
        with session_factory() as session:
            revision = _seed_revision(session, "col")
            doc = SourceDocument(collection="col", source_path="/test/a.pdf", file_hash="h1")
            session.add(doc)
            session.flush()
            extracted = ExtractedDocument(
                document_id=doc.id,
                extractor_profile_id=revision.extractor_profile_id,
                source_file_hash="h1",
                content_hash="c1",
                status="done",
            )
            session.add(extracted)
            session.flush()
            chunked = ChunkedDocument(
                extracted_document_id=extracted.id,
                chunk_profile_id=revision.chunk_profile_id,
                source_content_hash="c1",
                status="done",
            )
            session.add(chunked)
            session.flush()
            for index in range(3):
                chunk = Chunk(
                    document_id=doc.id,
                    chunked_document_id=chunked.id,
                    chunk_index=index,
                    content=f"chunk {index}",
                )
                session.add(chunk)
                session.flush()
                status = "done" if index < 2 else "failed"
                session.add(
                    ChunkEmbedding(
                        chunk_id=chunk.id,
                        embedding_profile_id=revision.embedding_profile_id,
                        status=status,
                    )
                )
            session.commit()

        with (
            patch("cementic.status_service.get_engine", return_value=engine),
            patch("cementic.status_service.get_session_factory", return_value=session_factory),
        ):
            result = load_file_progress(Config(), "col")
        assert len(result) == 1
        assert result[0].embeddings_total == 3
        assert result[0].embeddings_done == 2
        assert result[0].embeddings_failed == 1

    def test_query_count_does_not_scale_with_document_count(self) -> None:
        def run(n_documents: int) -> int:
            engine, session_factory = self._session_factory()
            with session_factory() as session:
                revision = _seed_revision(session, "col")
                for i in range(n_documents):
                    doc = SourceDocument(
                        collection="col", source_path=f"/test/{i}.pdf", file_hash=f"h{i}"
                    )
                    session.add(doc)
                    session.flush()
                    extracted = ExtractedDocument(
                        document_id=doc.id,
                        extractor_profile_id=revision.extractor_profile_id,
                        status="done",
                    )
                    session.add(extracted)
                    session.flush()
                    chunked = ChunkedDocument(
                        extracted_document_id=extracted.id,
                        chunk_profile_id=revision.chunk_profile_id,
                        status="done",
                    )
                    session.add(chunked)
                    session.flush()
                    chunk = Chunk(
                        document_id=doc.id,
                        chunked_document_id=chunked.id,
                        chunk_index=0,
                        content="content",
                    )
                    session.add(chunk)
                    session.flush()
                    session.add(
                        ChunkEmbedding(
                            chunk_id=chunk.id,
                            embedding_profile_id=revision.embedding_profile_id,
                            status="done",
                        )
                    )
                session.commit()

            counter = _count_select_queries(engine)
            with (
                patch("cementic.status_service.get_engine", return_value=engine),
                patch("cementic.status_service.get_session_factory", return_value=session_factory),
            ):
                result = load_file_progress(Config(), "col")
            assert len(result) == n_documents
            return counter.value

        small = run(2)
        large = run(50)
        assert small == large


class TestLoadPipelineStatusLabels:
    """Tests for load_pipeline_status revision-label defaults."""

    @patch("cementic.status_service.get_engine")
    @patch("cementic.status_service.get_session_factory")
    def test_no_revisions_yields_none_not_string_none(self, mock_factory, mock_engine) -> None:
        session = MagicMock()
        mock_factory.return_value.return_value.__enter__.return_value = session
        session.query().filter().count.return_value = 0
        session.query().filter_by().order_by().first.return_value = None
        session.query().filter().order_by().first.return_value = None

        config = Config()
        result = load_pipeline_status(config, "col")

        assert result.active_revision_label is None
        assert result.building_revision_label is None


class TestPerFileViewAgreesWithTheSummary:
    """`status -c X -v` prints the summary and the per-file list together.

    The list matched on profile ids only, so after a file changed on disk the
    summary said "extracted 9/10" while every row below it read "extract=done" --
    one command contradicting itself, with no way to tell which file was stuck.
    """

    def _session_factory(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        return engine, sessionmaker(bind=engine, expire_on_commit=False)

    def test_a_changed_file_is_pending_in_both_views(self) -> None:
        engine, session_factory = self._session_factory()
        with session_factory() as session:
            _seed_collection_with_documents(session, "col", 2)
            # The watcher saw /col/0.pdf change: its hash moves on, so the
            # existing extraction is work still owed, not work done.
            changed = (
                session.query(SourceDocument).filter_by(source_path="/col/0.pdf").one()
            )
            changed.file_hash = "h0-changed"
            session.commit()

        with (
            patch("cementic.status_service.get_engine", return_value=engine),
            patch("cementic.status_service.get_session_factory", return_value=session_factory),
        ):
            summary = load_pipeline_status_bulk(Config(), ["col"])["col"]
            files = load_file_progress(Config(), "col")

        by_path = {f.source_path: f for f in files}
        assert summary.extracted_done == 1
        assert by_path["/col/0.pdf"].extraction_status == "pending"
        assert by_path["/col/1.pdf"].extraction_status == "done"
        done_rows = sum(1 for f in files if f.extraction_status == "done")
        assert done_rows == summary.extracted_done


class TestFileProgressLimit:
    """The cap `status --verbose` uses, and what it is allowed to hide."""

    def _session_factory(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        return engine, sessionmaker(bind=engine, expire_on_commit=False)

    def test_limit_caps_rows_read(self) -> None:
        engine, session_factory = self._session_factory()
        with session_factory() as session:
            _seed_collection_with_documents(session, "col", 10)
            session.commit()

        with (
            patch("cementic.status_service.get_engine", return_value=engine),
            patch("cementic.status_service.get_session_factory", return_value=session_factory),
        ):
            assert len(load_file_progress(Config(), "col", 3)) == 3
            assert len(load_file_progress(Config(), "col", None)) == 10

    def test_failures_survive_the_cap(self) -> None:
        """The cap is only worth having if it keeps what a reader came for.

        Ten healthy documents sort before the broken one by path, so an
        unordered LIMIT 1 would return a `done` row and hide the failure.
        """
        engine, session_factory = self._session_factory()
        with session_factory() as session:
            revision = _seed_collection_with_documents(session, "col", 10)
            broken = SourceDocument(
                collection="col", source_path="/col/zzz-last-by-path.pdf", file_hash="hz"
            )
            session.add(broken)
            session.flush()
            session.add(
                ExtractedDocument(
                    document_id=broken.id,
                    extractor_profile_id=revision.extractor_profile_id,
                    status="failed",
                    source_file_hash="hz",
                    error_message="extraction broke",
                )
            )
            session.commit()

        with (
            patch("cementic.status_service.get_engine", return_value=engine),
            patch("cementic.status_service.get_session_factory", return_value=session_factory),
        ):
            files = load_file_progress(Config(), "col", 1)

        assert [f.source_path for f in files] == ["/col/zzz-last-by-path.pdf"]
        assert files[0].extraction_status == "failed"

    def test_unstarted_files_sort_ahead_of_finished_ones(self) -> None:
        engine, session_factory = self._session_factory()
        with session_factory() as session:
            _seed_collection_with_documents(session, "col", 5)
            session.add(
                SourceDocument(
                    collection="col", source_path="/col/zzz-untouched.pdf", file_hash="hu"
                )
            )
            session.commit()

        with (
            patch("cementic.status_service.get_engine", return_value=engine),
            patch("cementic.status_service.get_session_factory", return_value=session_factory),
        ):
            files = load_file_progress(Config(), "col", 1)

        assert files[0].source_path == "/col/zzz-untouched.pdf"
        assert files[0].extraction_status == "pending"
