"""Tests for status query helpers."""

from unittest.mock import MagicMock, patch

from cementic.config import Config
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
    load_worker_statuses,
)


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

    @patch("cementic.status_service.is_pid_running", return_value=True)
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

    @patch("cementic.status_service.is_pid_running", return_value=False)
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
            with patch("cementic.embedding_runtime.get_llama_cpp_runtime_client") as mock_client:
                mock_conn = MagicMock()
                mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
                mock_client.return_value.health_check.return_value = True
                result = check_health(config)
                assert result.db_reachable is True

    @patch("cementic.status_service.get_engine", side_effect=Exception("db down"))
    def test_db_unreachable(self, mock_engine) -> None:
        config = Config()
        result = check_health(config)
        assert result.db_reachable is False

    @patch("cementic.status_service.get_engine")
    @patch("cementic.embedding_runtime.get_llama_cpp_runtime_client")
    def test_health_llama_cpp_healthy(self, mock_client, mock_engine) -> None:
        config = Config()
        config.pipeline.embedding_provider = "llama-cpp"
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
        mock_client.return_value.health_check.return_value = True

        result = check_health(config)
        assert result.embedding_provider == "llama-cpp"
        assert result.embedding_healthy is True

    @patch("cementic.status_service.get_engine")
    @patch("cementic.embedding_runtime.get_llama_cpp_runtime_client")
    def test_health_llama_cpp_unhealthy(self, mock_client, mock_engine) -> None:
        config = Config()
        config.pipeline.embedding_provider = "llama-cpp"
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
        mock_client.return_value.health_check.return_value = False

        result = check_health(config)
        assert result.embedding_healthy is False

    def test_health_llama_cpp_busy_not_unhealthy(self, temp_dir) -> None:
        """HTTP health probe can fail while a large embedding batch holds
        llama_cpp.server's request lock; if the daemon process is confirmed
        alive via the PID file, that's busy, not unhealthy."""
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
                    "cementic.embedding_runtime.get_llama_cpp_runtime_client"
                ) as mock_client:
                    mock_client.return_value.health_check.return_value = False
                    result = check_health(config)
                    assert "running" in result.llama_daemon
                    assert result.embedding_healthy is True

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
                    "cementic.embedding_runtime.get_llama_cpp_runtime_client"
                ) as mock_client:
                    mock_client.return_value.health_check.return_value = True
                    result = check_health(config)
                    assert "running" in result.llama_daemon
                    assert "42" in result.llama_daemon

    def test_health_llama_daemon_stopped_no_pid_file(self) -> None:
        config = Config()
        config.pipeline.embedding_provider = "llama-cpp"
        config.llama_cpp.daemon_pid_file = None

        with patch("cementic.status_service.get_engine"):
            with patch(
                "cementic.embedding_runtime.get_llama_cpp_runtime_client"
            ) as mock_client:
                mock_client.return_value.health_check.return_value = True
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
                "cementic.embedding_runtime.get_llama_cpp_runtime_client"
            ) as mock_client:
                mock_client.return_value.health_check.return_value = True
                result = check_health(config)
                assert result.llama_daemon == "stopped"

    def test_health_llama_daemon_na_for_non_llama_provider(self) -> None:
        config = Config()
        config.pipeline.embedding_provider = "other"

        with patch("cementic.status_service.get_engine"):
            result = check_health(config)
            assert result.llama_daemon == "N/A"


class TestLoadFileProgress:
    """Tests for load_file_progress."""

    @patch("cementic.status_service.get_engine")
    @patch("cementic.status_service.get_session_factory")
    def test_no_target_revision_returns_empty(self, mock_factory, mock_engine) -> None:
        session = MagicMock()
        mock_factory.return_value.return_value.__enter__.return_value = session

        # Both building/ready and active queries return None
        session.query().filter().order_by().first.return_value = None
        session.query().filter_by().order_by().first.return_value = None

        config = Config()
        result = load_file_progress(config, "col")
        assert result == []

    @patch("cementic.status_service.get_engine")
    @patch("cementic.status_service.get_session_factory")
    def test_no_documents_returns_empty(self, mock_factory, mock_engine) -> None:
        session = MagicMock()
        mock_factory.return_value.return_value.__enter__.return_value = session

        # A revision exists but no documents
        mock_revision = MagicMock()
        mock_revision.id = 1
        mock_revision.extractor_profile_id = 1
        mock_revision.chunk_profile_id = 1
        mock_revision.embedding_profile_id = 1
        session.query().filter().order_by().first.return_value = mock_revision
        session.query().filter_by().order_by().all.return_value = []

        config = Config()
        result = load_file_progress(config, "col")
        assert result == []

    @patch("cementic.status_service.get_engine")
    @patch("cementic.status_service.get_session_factory")
    def test_failed_extraction_sets_error_message(self, mock_factory, mock_engine) -> None:
        """Line 455: extraction_status == 'failed' → error_message set."""
        session = MagicMock()
        mock_factory.return_value.return_value.__enter__.return_value = session

        mock_revision = MagicMock()
        mock_revision.id = 1
        mock_revision.extractor_profile_id = 1
        mock_revision.chunk_profile_id = 1
        mock_revision.embedding_profile_id = 1
        session.query().filter().order_by().first.return_value = mock_revision

        mock_doc = MagicMock()
        mock_doc.id = 1
        mock_doc.source_path = "/test/a.pdf"
        session.query().filter().order_by().all.return_value = [mock_doc]

        mock_extraction = MagicMock()
        mock_extraction.status = "failed"
        mock_extraction.error_message = "extraction broke"
        mock_extraction.id = 10
        session.query().filter_by().first.return_value = mock_extraction

        config = Config()
        result = load_file_progress(config, "col")
        assert len(result) == 1
        assert result[0].extraction_status == "failed"
        assert result[0].error_message == "extraction broke"

    @patch("cementic.status_service.get_engine")
    @patch("cementic.status_service.get_session_factory")
    def test_failed_chunking_sets_error_message(self, mock_factory, mock_engine) -> None:
        """Line 457: chunking_status == 'failed' → error_message set."""
        session = MagicMock()
        mock_factory.return_value.return_value.__enter__.return_value = session

        mock_revision = MagicMock()
        mock_revision.id = 1
        mock_revision.extractor_profile_id = 1
        mock_revision.chunk_profile_id = 1
        mock_revision.embedding_profile_id = 1
        session.query().filter().order_by().first.return_value = mock_revision

        mock_doc = MagicMock()
        mock_doc.id = 1
        mock_doc.source_path = "/test/a.pdf"
        session.query().filter().order_by().all.return_value = [mock_doc]

        mock_extraction = MagicMock()
        mock_extraction.status = "done"
        mock_extraction.id = 10
        mock_chunking = MagicMock()
        mock_chunking.status = "failed"
        mock_chunking.error_message = "chunker crashed"
        mock_chunking.id = 20
        session.query().filter_by().first.side_effect = [mock_extraction, mock_chunking]

        config = Config()
        result = load_file_progress(config, "col")
        assert len(result) == 1
        assert result[0].extraction_status == "done"
        assert result[0].chunking_status == "failed"
        assert result[0].error_message == "chunker crashed"
