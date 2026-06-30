"""Persistent worker state management."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

UNSET = object()


class DaemonState(str, Enum):
    """State of a background worker."""

    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"


@dataclass
class WorkerState:
    """State of a background worker process."""

    daemon_state: DaemonState = DaemonState.STOPPED
    watched_directories: list[str] = field(default_factory=list)
    processed_count: int = 0
    failed_count: int = 0
    current_file: str | None = None
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    pid: int | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "WorkerState":
        """Create from dictionary."""
        normalized = dict(data)
        raw_daemon_state = normalized.get("daemon_state", DaemonState.STOPPED)

        if isinstance(raw_daemon_state, DaemonState):
            daemon_state_value = raw_daemon_state
        elif isinstance(raw_daemon_state, str):
            try:
                daemon_state_value = DaemonState(raw_daemon_state)
            except ValueError:
                daemon_state_value = DaemonState.STOPPED
        else:
            daemon_state_value = DaemonState.STOPPED

        raw_watched_directories = normalized.get("watched_directories", [])
        watched_directories: list[str] = []
        if isinstance(raw_watched_directories, list):
            watched_directories = [str(path) for path in raw_watched_directories]

        raw_current_file = normalized.get("current_file")
        current_file = str(raw_current_file) if raw_current_file is not None else None

        raw_last_updated = normalized.get("last_updated", datetime.now(timezone.utc).isoformat())
        last_updated = str(raw_last_updated)

        raw_processed_count = normalized.get("processed_count", 0)
        processed_count = raw_processed_count if isinstance(raw_processed_count, int) else 0

        raw_failed_count = normalized.get("failed_count", 0)
        failed_count = raw_failed_count if isinstance(raw_failed_count, int) else 0

        raw_pid = normalized.get("pid")
        pid = raw_pid if isinstance(raw_pid, int) else None

        return cls(
            daemon_state=daemon_state_value,
            watched_directories=watched_directories,
            processed_count=processed_count,
            failed_count=failed_count,
            current_file=current_file,
            last_updated=last_updated,
            pid=pid,
        )


class StateManager:
    """Manages persistent state for pause/resume functionality."""

    def __init__(self, state_path: Path | None) -> None:
        """Initialize state manager."""
        if state_path is None:
            raise ValueError("state_path cannot be None")
        self.state_path = state_path
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> WorkerState:
        """Load state from file."""
        if not self.state_path.exists():
            return WorkerState()

        try:
            with open(self.state_path, "r") as f:
                data = json.load(f)
            return WorkerState.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError):
            return WorkerState()

    def save(self, state: WorkerState) -> None:
        """Save state to file."""
        state.last_updated = datetime.now(timezone.utc).isoformat()
        with open(self.state_path, "w") as f:
            json.dump(state.to_dict(), f, indent=2)

    def update(
        self,
        daemon_state: Any = UNSET,
        watched_directories: Any = UNSET,
        processed_count: Any = UNSET,
        failed_count: Any = UNSET,
        current_file: Any = UNSET,
        pid: Any = UNSET,
    ) -> WorkerState:
        """Update specific fields and save."""
        state = self.load()

        if daemon_state is not UNSET:
            state.daemon_state = daemon_state
        if watched_directories is not UNSET:
            state.watched_directories = watched_directories
        if processed_count is not UNSET:
            state.processed_count = processed_count
        if failed_count is not UNSET:
            state.failed_count = failed_count
        if current_file is not UNSET:
            state.current_file = current_file
        if pid is not UNSET:
            state.pid = pid

        self.save(state)
        return state

    def reset(self) -> None:
        """Reset all state."""
        if self.state_path.exists():
            self.state_path.unlink()

    def is_running(self) -> bool:
        """Check if daemon is marked as running."""
        state = self.load()
        return state.daemon_state == DaemonState.RUNNING

    def is_paused(self) -> bool:
        """Check if daemon is paused."""
        state = self.load()
        return state.daemon_state == DaemonState.PAUSED
