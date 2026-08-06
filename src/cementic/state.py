"""Persistent worker state management."""

import json
import os
import threading
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
    #: Kernel start-time token of ``pid`` (see supervisor.process_start_token);
    #: lets liveness checks reject a recycled PID. None = unknown (older state).
    start_token: str | None = None

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

        raw_start_token = normalized.get("start_token")
        start_token = raw_start_token if isinstance(raw_start_token, str) else None

        return cls(
            daemon_state=daemon_state_value,
            watched_directories=watched_directories,
            processed_count=processed_count,
            failed_count=failed_count,
            current_file=current_file,
            last_updated=last_updated,
            pid=pid,
            start_token=start_token,
        )


class StateManager:
    """Manages persistent on-disk worker state.

    Writes are atomic (tmp + rename) and read-modify-write ``update`` calls are
    serialized with a lock, since the watcher's debounce timers call it from
    concurrent threads while ``cementic status`` reads the same file.
    """

    def __init__(self, state_path: Path | None) -> None:
        """Initialize state manager. Does not touch the filesystem (read-only
        commands construct one just to ``load``); ``save`` creates the directory."""
        if state_path is None:
            raise ValueError("state_path cannot be None")
        self.state_path = state_path
        # Reentrant so a shutdown path that runs while this thread already holds
        # the lock cannot deadlock the process. Handlers are kept off this path
        # deliberately (see the workers' _handle_shutdown), but a plain Lock here
        # turns any future re-entry into an unkillable-by-SIGTERM hang.
        self._lock = threading.RLock()

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
        """Save state to file atomically."""
        state.last_updated = datetime.now(timezone.utc).isoformat()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.state_path.with_name(f".{self.state_path.name}.tmp")
        tmp_path.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
        os.replace(tmp_path, self.state_path)

    def update(
        self,
        daemon_state: Any = UNSET,
        watched_directories: Any = UNSET,
        processed_count: Any = UNSET,
        failed_count: Any = UNSET,
        current_file: Any = UNSET,
        pid: Any = UNSET,
        start_token: Any = UNSET,
    ) -> WorkerState:
        """Update specific fields and save."""
        with self._lock:
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
            if start_token is not UNSET:
                state.start_token = start_token

            self.save(state)
            return state
