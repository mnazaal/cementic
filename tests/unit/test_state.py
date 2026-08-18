"""Tests for state management module."""

import json
import threading
from pathlib import Path

import pytest

from cementic.state import (
    MAX_RECORDED_SKIPPED_FILES,
    DaemonState,
    StateManager,
    WorkerState,
)


class TestWorkerState:
    """Test WorkerState dataclass."""

    def test_default_state(self):
        """Test default state values."""
        state = WorkerState()
        assert state.daemon_state == DaemonState.STOPPED
        assert state.watched_directories == []
        assert state.processed_count == 0
        assert state.failed_count == 0
        assert state.current_file is None
        assert state.pid is None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        state = WorkerState(
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
            "daemon_state": "running",
            "watched_directories": ["/test"],
            "processed_count": 5,
            "failed_count": 1,
            "current_file": "/test/file.pdf",
            "last_updated": "2024-01-01T00:00:00",
            "pid": 5678,
        }
        state = WorkerState.from_dict(data)

        assert state.daemon_state == DaemonState.RUNNING
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

        state = WorkerState.from_dict(data)
        assert state.daemon_state == DaemonState.STOPPED

    def test_from_dict_already_enum_type(self) -> None:
        """daemon_state already DaemonState enum passes through (line 48)."""
        data = {
            "daemon_state": DaemonState.RUNNING,
            "watched_directories": [],
            "processed_count": 0,
            "failed_count": 0,
            "current_file": None,
            "last_updated": "2024-01-01T00:00:00",
            "pid": None,
        }
        state = WorkerState.from_dict(data)
        assert state.daemon_state == DaemonState.RUNNING

    def test_from_dict_non_string_non_enum_falls_back(self) -> None:
        """Non-string, non-DaemonState daemon_state → STÖPPED fallback (line 50)."""
        data = {
            "daemon_state": 42,
            "watched_directories": [],
            "processed_count": 0,
            "failed_count": 0,
            "current_file": None,
            "last_updated": "2024-01-01T00:00:00",
            "pid": None,
        }
        state = WorkerState.from_dict(data)
        assert state.daemon_state == DaemonState.STOPPED


class TestAtomicIncrement:
    """Counters must not lose updates under concurrency.

    Regression: callers read the state and then wrote count + 1 in a separate
    update(). Only the write was inside the lock, so the watcher's scan thread
    and its debounce timers could read the same value and lose an increment.
    """

    def test_concurrent_increments_are_all_counted(self, temp_dir: Path) -> None:
        manager = StateManager(temp_dir / "state.json")
        manager.update(processed_count=0, failed_count=0)
        workers = 8
        per_worker = 25

        def bump() -> None:
            for _ in range(per_worker):
                manager.increment(processed=1)

        threads = [threading.Thread(target=bump) for _ in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert manager.load().processed_count == workers * per_worker

    def test_increment_can_set_current_file_at_the_same_time(self, temp_dir: Path) -> None:
        manager = StateManager(temp_dir / "state.json")
        manager.update(processed_count=3, current_file="/a.pdf")

        state = manager.increment(processed=1, current_file=None)

        assert state.processed_count == 4
        assert state.current_file is None

    def test_failed_counter_increments_independently(self, temp_dir: Path) -> None:
        manager = StateManager(temp_dir / "state.json")
        manager.increment(failed=1)
        manager.increment(failed=1)
        state = manager.load()
        assert state.failed_count == 2
        assert state.processed_count == 0

    def test_record_skipped_keeps_the_path_and_the_reason(self, temp_dir: Path) -> None:
        """The count alone could not be acted on -- which files were dropped?

        A skipped file never becomes a document, so it is absent from every
        pipeline count; the paths used to exist only in a log file the user is
        never pointed at.
        """
        manager = StateManager(temp_dir / "state.json")

        manager.record_skipped("/docs/link.md", "symlink")
        state = manager.record_skipped("/docs/huge.pdf", "too large (999 bytes, max 1)")

        assert state.failed_count == 2
        assert state.skipped_files == [
            "/docs/link.md: symlink",
            "/docs/huge.pdf: too large (999 bytes, max 1)",
        ]
        assert manager.load().skipped_files == state.skipped_files

    def test_skipped_files_reset_with_the_counters(self, temp_dir: Path) -> None:
        """Regression: restart reset processed/failed to zero but kept
        skipped_files, so a fresh run still showed last run's skips -- paths
        with no counter behind them."""
        manager = StateManager(temp_dir / "state.json")
        manager.record_skipped("/docs/old-run.md", "symlink")

        state = manager.update(processed_count=0, failed_count=0, skipped_files=[])

        assert state.failed_count == 0
        assert state.skipped_files == []
        assert manager.load().skipped_files == []

    def test_record_skipped_can_clear_current_file_in_the_same_write(
        self, temp_dir: Path
    ) -> None:
        """A registration failure clears the "now working on" display atomically,
        as increment() does, instead of leaving the failed file shown as current."""
        manager = StateManager(temp_dir / "state.json")
        manager.update(current_file="/docs/broken.md")

        state = manager.record_skipped(
            "/docs/broken.md", "registration failed: boom", current_file=None
        )

        assert state.current_file is None
        assert state.skipped_files == ["/docs/broken.md: registration failed: boom"]

    def test_recorded_skips_are_bounded_but_the_count_is_not(self, temp_dir: Path) -> None:
        """A directory of symlinks must not grow the state file without bound."""
        manager = StateManager(temp_dir / "state.json")
        total = MAX_RECORDED_SKIPPED_FILES + 10

        for i in range(total):
            manager.record_skipped(f"/docs/{i}.md", "symlink")

        state = manager.load()
        assert state.failed_count == total
        assert len(state.skipped_files) == MAX_RECORDED_SKIPPED_FILES
        # The most recent are the ones kept.
        assert state.skipped_files[-1] == f"/docs/{total - 1}.md: symlink"


class TestStateManager:
    """Test StateManager functionality."""

    def test_load_nonexistent_file(self, temp_dir):
        """Test loading state when file doesn't exist."""
        state_path = temp_dir / "nonexistent_state.json"
        manager = StateManager(state_path)

        state = manager.load()
        assert isinstance(state, WorkerState)
        assert state.daemon_state == DaemonState.STOPPED

    def test_save_and_load(self, temp_dir):
        """Test saving and loading state."""
        state_path = temp_dir / "state.json"
        manager = StateManager(state_path)

        state = WorkerState(daemon_state=DaemonState.RUNNING, processed_count=42, pid=9999)
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
        initial = WorkerState(daemon_state=DaemonState.STOPPED)
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

    def test_load_corrupted_file(self, temp_dir):
        """Test loading corrupted state file."""
        state_path = temp_dir / "state.json"
        state_path.write_text("not valid json")

        manager = StateManager(state_path)
        state = manager.load()

        # Should return default state
        assert isinstance(state, WorkerState)
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

    def test_init_raises_on_none_state_path(self) -> None:
        """StateManager(None) → ValueError (line 90)."""
        with pytest.raises(ValueError, match="state_path cannot be None"):
            StateManager(None)

    def test_update_sets_failed_count(self, temp_dir) -> None:
        """Update should set failed_count on the state (line 131)."""
        state_path = temp_dir / "state.json"
        manager = StateManager(state_path)
        manager.update(failed_count=7)
        loaded = manager.load()
        assert loaded.failed_count == 7
