"""Tests for embedder daemon."""

from unittest.mock import MagicMock, patch

from seman.embedder import ClaimedChunk, EmbedderDaemon
from seman.state import DaemonState


class TestEmbedderDaemon:
    """Test embedder daemon functionality."""

    @patch("seman.embedder.get_engine")
    @patch("seman.embedder.get_session_factory")
    @patch("seman.embedder.EmbedderDaemon._init_embedder")
    def test_get_pending_batch(self, mock_init_embedder, mock_session_factory, mock_get_engine):
        """Test retrieving pending chunks."""
        # Create mock session and chunks
        mock_session = MagicMock()
        # Make mock_session work as a context manager
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        # Make mock_session callable and return itself
        mock_session.return_value = mock_session
        mock_session_factory.return_value = mock_session

        # Setup query mock
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            MagicMock(id=1, content="chunk 1", document_id=1),
            MagicMock(id=2, content="chunk 2", document_id=1),
        ]
        mock_session.execute.return_value = mock_result

        # Create daemon and test
        daemon = EmbedderDaemon()
        daemon.Session = mock_session
        daemon.config.embedder.batch_size = 10

        # Get pending batch
        batch = daemon._get_pending_batch()

        # Verify batch was retrieved
        assert batch is not None
        assert len(batch) == 2
        assert batch[0].id == 1
        assert batch[1].id == 2
        assert mock_session.execute.called

    @patch("seman.embedder.get_engine")
    @patch("seman.embedder.get_session_factory")
    def test_process_batch(self, mock_session_factory, mock_get_engine):
        """Test processing a batch of chunks."""
        # Setup mock embedder
        mock_embedder = MagicMock()
        mock_embedder.embed_batch.return_value = [[0.1] * 768, [0.2] * 768]

        # Setup mock session
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.return_value = mock_session  # Make callable return itself
        mock_session_factory.return_value = mock_session

        # Create daemon
        daemon = EmbedderDaemon()
        daemon.Session = mock_session
        daemon.embedder = mock_embedder

        # Create mock chunks
        mock_chunks = [
            ClaimedChunk(id=1, content="chunk 1", document_id=1),
            ClaimedChunk(id=2, content="chunk 2", document_id=1),
        ]

        # Mock _update_document_status to avoid database calls
        with patch.object(daemon, "_update_document_status") as mock_update_status:
            daemon._process_batch(mock_chunks)

        # Verify embeddings were generated
        mock_embedder.embed_batch.assert_called_once_with(
            ["search_document: chunk 1", "search_document: chunk 2"]
        )

        # Verify database was updated
        assert mock_session.commit.called
        mock_update_status.assert_called_once_with({1})

    @patch("seman.embedder.get_engine")
    @patch("seman.embedder.get_session_factory")
    def test_process_batch_handles_failures(self, mock_session_factory, mock_get_engine):
        """Test batch processing handles embedding failures."""
        # Setup mock embedder with one failure
        mock_embedder = MagicMock()
        mock_embedder.embed_batch.return_value = [[0.1] * 768, None]  # Second one fails

        # Setup mock session
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.return_value = mock_session  # Make callable return itself
        mock_session_factory.return_value = mock_session

        # Create daemon
        daemon = EmbedderDaemon()
        daemon.Session = mock_session
        daemon.embedder = mock_embedder

        # Create mock chunks
        mock_chunks = [
            ClaimedChunk(id=1, content="chunk 1", document_id=1),
            ClaimedChunk(id=2, content="chunk 2", document_id=2),
        ]

        # Mock _update_document_status
        with patch.object(daemon, "_update_document_status"):
            daemon._process_batch(mock_chunks)

        # Verify database was updated for both chunks
        assert mock_session.commit.called

    @patch("seman.embedder.get_engine")
    @patch("seman.embedder.get_session_factory")
    def test_update_document_status(self, mock_session_factory, mock_get_engine):
        """Test document status update when all chunks are done."""
        # Setup mock session
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.return_value = mock_session  # Make callable return itself
        mock_session_factory.return_value = mock_session

        # Setup query mocks for chunk counts
        mock_session.query.return_value.filter_by.return_value.count.side_effect = [
            3,
            3,
        ]  # total, done

        # Create daemon
        daemon = EmbedderDaemon()
        daemon.Session = mock_session

        # Create mock chunks
        daemon._update_document_status({1})

        # Verify document status was updated to completed
        assert mock_session.commit.called

    @patch("seman.embedder.get_engine")
    @patch("seman.embedder.get_session_factory")
    def test_recover_stale_processing_chunks(self, mock_session_factory, mock_get_engine):
        """Test stale processing chunks are reset to pending."""
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.return_value = mock_session
        mock_session_factory.return_value = mock_session

        daemon = EmbedderDaemon()
        daemon.Session = mock_session

        daemon._recover_stale_processing_chunks()

        assert mock_session.query.called
        assert mock_session.commit.called

    def test_init_embedder_llama_cpp(self):
        """Test initialization of llama.cpp embedder."""
        with patch("seman.embedder.get_embedder") as mock_get_embedder:
            mock_embedder = MagicMock()
            mock_embedder.health_check.return_value = True
            mock_get_embedder.return_value = mock_embedder

            with patch("seman.embedder.get_engine"):
                with patch("seman.embedder.get_session_factory"):
                    daemon = EmbedderDaemon()
                    daemon.config.indexing.embedder = "llama-cpp"

                    daemon._init_embedder()

                    mock_get_embedder.assert_called_once()
                    assert mock_get_embedder.call_args[0][0] == "llama-cpp"

    def test_init_embedder_ollama(self):
        """Test initialization of Ollama embedder."""
        with patch("seman.embedder.get_embedder") as mock_get_embedder:
            mock_embedder = MagicMock()
            mock_embedder.health_check.return_value = True
            mock_get_embedder.return_value = mock_embedder

            with patch("seman.embedder.get_engine"):
                with patch("seman.embedder.get_session_factory"):
                    daemon = EmbedderDaemon()
                    daemon.config.indexing.embedder = "ollama"

                    daemon._init_embedder()

                    mock_get_embedder.assert_called_once()
                    assert mock_get_embedder.call_args[0][0] == "ollama"

    def test_init_embedder_health_check_failure(self):
        """Test that embedder initializes but fails health check."""
        with patch("seman.embedder.get_embedder") as mock_get_embedder:
            mock_embedder = MagicMock()
            mock_embedder.health_check.return_value = False
            mock_get_embedder.return_value = mock_embedder

            with patch("seman.embedder.get_engine"):
                with patch("seman.embedder.get_session_factory"):
                    daemon = EmbedderDaemon()
                    daemon.config.indexing.embedder = "ollama"

                    # _init_embedder returns the embedder; health check happens in start()
                    result = daemon._init_embedder()
                    assert result is mock_embedder
                    assert result.health_check() is False


class TestEmbedderStateManagement:
    """Test embedder state transitions."""

    @patch("seman.embedder.get_engine")
    @patch("seman.embedder.get_session_factory")
    def test_stop_sets_shutdown(self, mock_session_factory, mock_get_engine):
        """Test stop sets shutdown event."""
        daemon = EmbedderDaemon()
        daemon._shutdown_event = MagicMock()

        with patch.object(daemon.state_manager, "update") as mock_update:
            daemon.stop()

            daemon._shutdown_event.set.assert_called_once()
            mock_update.assert_called_with(daemon_state=DaemonState.STOPPED, pid=None)
