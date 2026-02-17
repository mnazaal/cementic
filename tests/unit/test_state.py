"""Tests for state management module."""

import json

from seman.state import DaemonState, IndexingState, StateManager


class TestIndexingState:
    """Test IndexingState dataclass."""

    def test_default_state(self):
        """Test default state values."""
        state = IndexingState()
        assert state.daemon_state == DaemonState.STOPPED
        assert state.watched_directories == []
        assert state.processed_count == 0
        assert state.failed_count == 0
        assert state.current_file is None
        assert state.pid is None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        state = IndexingState(
            daemon_state=DaemonState.RUNNING,
            watched_directories=["/path/1", "/path/2"],
            processed_count=10,
            pid=1234,
        )
        data = state.to_dict()

        assert data["daemon_state"] == "running"
        assert data["watched_directories"] == ["/path/1", "/path/2"]
        assert data["processed_count"] == 10
        assert data["pid"] == 1234

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "daemon_state": "paused",
            "watched_directories": ["/test"],
            "processed_count": 5,
            "failed_count": 1,
            "current_file": "/test/file.pdf",
            "last_updated": "2024-01-01T00:00:00",
            "pid": 5678,
        }
        state = IndexingState.from_dict(data)

        assert state.daemon_state == DaemonState.PAUSED
        assert state.watched_directories == ["/test"]
        assert state.processed_count == 5
        assert state.current_file == "/test/file.pdf"
        assert state.pid == 5678

    def test_from_dict_invalid_daemon_state_falls_back_to_stopped(self):
        """Unknown daemon states should not crash deserialization."""
        data = {
            "daemon_state": "definitely-not-valid",
            "watched_directories": [],
            "processed_count": 0,
            "failed_count": 0,
            "current_file": None,
            "last_updated": "2024-01-01T00:00:00",
            "pid": None,
        }

        state = IndexingState.from_dict(data)
        assert state.daemon_state == DaemonState.STOPPED


class TestStateManager:
    """Test StateManager functionality."""

    def test_load_nonexistent_file(self, temp_dir):
        """Test loading state when file doesn't exist."""
        state_path = temp_dir / "nonexistent_state.json"
        manager = StateManager(state_path)

        state = manager.load()
        assert isinstance(state, IndexingState)
        assert state.daemon_state == DaemonState.STOPPED

    def test_save_and_load(self, temp_dir):
        """Test saving and loading state."""
        state_path = temp_dir / "state.json"
        manager = StateManager(state_path)

        state = IndexingState(daemon_state=DaemonState.RUNNING, processed_count=42, pid=9999)
        manager.save(state)

        # Load it back
        loaded = manager.load()
        assert loaded.daemon_state == DaemonState.RUNNING
        assert loaded.processed_count == 42
        assert loaded.pid == 9999

    def test_update_single_field(self, temp_dir):
        """Test updating a single field."""
        state_path = temp_dir / "state.json"
        manager = StateManager(state_path)

        # Initial save
        initial = IndexingState(daemon_state=DaemonState.STOPPED)
        manager.save(initial)

        # Update one field
        updated = manager.update(daemon_state=DaemonState.RUNNING)
        assert updated.daemon_state == DaemonState.RUNNING

        # Verify it was saved
        loaded = manager.load()
        assert loaded.daemon_state == DaemonState.RUNNING

    def test_update_multiple_fields(self, temp_dir):
        """Test updating multiple fields at once."""
        state_path = temp_dir / "state.json"
        manager = StateManager(state_path)

        manager.update(
            daemon_state=DaemonState.RUNNING,
            watched_directories=["/path1", "/path2"],
            processed_count=100,
            pid=1234,
        )

        loaded = manager.load()
        assert loaded.daemon_state == DaemonState.RUNNING
        assert loaded.watched_directories == ["/path1", "/path2"]
        assert loaded.processed_count == 100
        assert loaded.pid == 1234

    def test_update_can_clear_optional_fields(self, temp_dir):
        """Test that update can explicitly clear fields using None."""
        state_path = temp_dir / "state.json"
        manager = StateManager(state_path)

        manager.update(current_file="/tmp/file.pdf", pid=1234)
        manager.update(current_file=None, pid=None)

        loaded = manager.load()
        assert loaded.current_file is None
        assert loaded.pid is None

    def test_reset(self, temp_dir):
        """Test resetting state."""
        state_path = temp_dir / "state.json"
        manager = StateManager(state_path)

        # Create some state
        state = IndexingState(daemon_state=DaemonState.RUNNING, processed_count=50)
        manager.save(state)
        assert state_path.exists()

        # Reset it
        manager.reset()
        assert not state_path.exists()

    def test_is_running(self, temp_dir):
        """Test is_running helper."""
        state_path = temp_dir / "state.json"
        manager = StateManager(state_path)

        assert not manager.is_running()

        manager.update(daemon_state=DaemonState.RUNNING)
        assert manager.is_running()

    def test_is_paused(self, temp_dir):
        """Test is_paused helper."""
        state_path = temp_dir / "state.json"
        manager = StateManager(state_path)

        assert not manager.is_paused()

        manager.update(daemon_state=DaemonState.PAUSED)
        assert manager.is_paused()

    def test_load_corrupted_file(self, temp_dir):
        """Test loading corrupted state file."""
        state_path = temp_dir / "state.json"
        state_path.write_text("not valid json")

        manager = StateManager(state_path)
        state = manager.load()

        # Should return default state
        assert isinstance(state, IndexingState)
        assert state.daemon_state == DaemonState.STOPPED

    def test_load_legacy_json_string_daemon_state(self, temp_dir):
        """Loading old JSON with string daemon_state should return enum type."""
        state_path = temp_dir / "state.json"
        state_path.write_text(
            json.dumps(
                {
                    "daemon_state": "running",
                    "watched_directories": ["/tmp"],
                    "processed_count": 1,
                    "failed_count": 0,
                    "current_file": None,
                    "last_updated": "2024-01-01T00:00:00",
                    "pid": 123,
                }
            )
        )

        manager = StateManager(state_path)
        state = manager.load()

        assert state.daemon_state == DaemonState.RUNNING
        assert isinstance(state.daemon_state, DaemonState)
