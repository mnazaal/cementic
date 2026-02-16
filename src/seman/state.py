"""State management for pause/resume functionality."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional


class DaemonState(str, Enum):
    """State of the indexing daemon."""

    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"


@dataclass
class IndexingState:
    """State of the indexing process."""

    daemon_state: DaemonState = DaemonState.STOPPED
    watched_directories: List[str] = field(default_factory=list)
    processed_count: int = 0
    failed_count: int = 0
    current_file: Optional[str] = None
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    pid: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "IndexingState":
        """Create from dictionary."""
        return cls(**data)


class StateManager:
    """Manages persistent state for pause/resume functionality."""

    def __init__(self, state_path: Path) -> None:
        """Initialize state manager."""
        self.state_path = state_path
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> IndexingState:
        """Load state from file."""
        if not self.state_path.exists():
            return IndexingState()

        try:
            with open(self.state_path, "r") as f:
                data = json.load(f)
            return IndexingState.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError):
            return IndexingState()

    def save(self, state: IndexingState) -> None:
        """Save state to file."""
        state.last_updated = datetime.utcnow().isoformat()
        with open(self.state_path, "w") as f:
            json.dump(state.to_dict(), f, indent=2)

    def update(
        self,
        daemon_state: Optional[DaemonState] = None,
        watched_directories: Optional[List[str]] = None,
        processed_count: Optional[int] = None,
        failed_count: Optional[int] = None,
        current_file: Optional[str] = None,
        pid: Optional[int] = None,
    ) -> IndexingState:
        """Update specific fields and save."""
        state = self.load()

        if daemon_state is not None:
            state.daemon_state = daemon_state
        if watched_directories is not None:
            state.watched_directories = watched_directories
        if processed_count is not None:
            state.processed_count = processed_count
        if failed_count is not None:
            state.failed_count = failed_count
        if current_file is not None:
            state.current_file = current_file
        if pid is not None:
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
