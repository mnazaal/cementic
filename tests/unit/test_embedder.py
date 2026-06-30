"""Tests for the pipeline worker."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cementic.pipeline_worker import PipelineWorker
from cementic.state import DaemonState


class TestPipelineWorker:
    """Test pipeline worker functionality."""

    def test_create_embedding_client_llama_cpp(self):
        with patch("cementic.pipeline_worker.create_provider") as mock_create:
            daemon = PipelineWorker()
            daemon.config.pipeline.embedding_provider = "llama-cpp"
            daemon._create_embedding_client()
            spec, config = mock_create.call_args[0]
            assert config is daemon.config
            assert spec.provider == "llama-cpp"

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

        daemon = PipelineWorker()
        daemon.state_manager.load = MagicMock(
            return_value=SimpleNamespace(daemon_state=DaemonState.STOPPED, pid=None)
        )
        daemon.state_manager.update = MagicMock()
        daemon._run_processing_loop = MagicMock(side_effect=StopLoopError)

        mock_embedding_client = MagicMock()
        mock_embedding_client.health_check.return_value = True
        daemon._create_embedding_client = MagicMock(return_value=mock_embedding_client)
        mock_session_factory.return_value = MagicMock()

        try:
            daemon.start(collection="research")
        except StopLoopError:
            pass

        assert daemon.collection == "research"
        running_updates = [
            call
            for call in daemon.state_manager.update.call_args_list
            if call.kwargs.get("daemon_state") == DaemonState.RUNNING
        ]
        assert len(running_updates) == 1

    def test_ensure_target_revision_commits(self):
        daemon = PipelineWorker()
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = False
        mock_session.get.return_value = None
        daemon.Session = MagicMock(return_value=mock_session)

        with patch(
            "cementic.pipeline_worker.get_target_revision", return_value=SimpleNamespace(id=7)
        ) as mock_get_target:
            revision_id = daemon._ensure_target_revision()

        assert revision_id == 7
        mock_get_target.assert_called_once()
        mock_session.commit.assert_called_once()

    def test_mark_revision_ready_if_complete(self):
        daemon = PipelineWorker()
        revision = SimpleNamespace(status="building")
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = False
        mock_session.get.return_value = revision
        daemon.Session = MagicMock(return_value=mock_session)

        with patch.object(daemon, "_revision_complete", return_value=True):
            with patch("cementic.pipeline_worker.ensure_revision_ann_index") as mock_ensure_index:
                with patch("cementic.pipeline_worker.mark_revision_ready") as mock_mark_ready:
                    daemon._mark_revision_ready_if_complete(3)

        mock_ensure_index.assert_called_once_with(mock_session, revision, daemon.config)
        mock_mark_ready.assert_called_once_with(mock_session, revision)
        mock_session.commit.assert_called_once()


class TestPipelineWorkerStateManagement:
    """Test pipeline worker state transitions."""

    def test_stop_sets_shutdown(self):
        daemon = PipelineWorker()
        daemon._shutdown_event = MagicMock()

        with patch.object(daemon.state_manager, "update") as mock_update:
            daemon.stop()

        daemon._shutdown_event.set.assert_called_once()
        mock_update.assert_called_with(daemon_state=DaemonState.STOPPED, pid=None)
