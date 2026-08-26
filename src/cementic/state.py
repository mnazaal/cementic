"""Persistent worker state management."""

import dataclasses
import json
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

UNSET = object()

#: How many skipped-file paths to keep in the state file. Enough to act on
#: without letting a directory of symlinks grow the file without bound.
MAX_RECORDED_SKIPPED_FILES = 50


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
    #: Last unhandled error from the worker's processing loop, cleared on the
    #: next clean pass. Without this a worker stuck in a permanent retry loop is
    #: indistinguishable from a healthy idle one in `cementic status`.
    last_error: str | None = None
    last_error_at: str | None = None
    #: What the worker is doing when it is not working through files -- today
    #: only the ANN index build, which occupies the loop for minutes at a time
    #: while writing nothing else. Without it `cementic status` shows a running
    #: worker, a building revision and no current file: identical to an idle one.
    current_activity: str | None = None
    #: Recent files the watcher refused to register, as "path: reason". These
    #: never become documents, so they are absent from every pipeline count --
    #: a collection could report 100% complete having silently dropped a
    #: directory's worth of symlinks. ``failed_count`` said how many, but the
    #: paths existed only in a log file the user is never pointed at. Bounded so
    #: the state file cannot grow without limit; the count remains exact.
    skipped_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "WorkerState":
        """Create from a dict, keeping only known fields.

        The file is written only by this codebase, atomically, and ``load()``
        already answers any unusable file with a default ``WorkerState`` -- so
        per-field type coercion here amounted to ~70 lines defending against a
        hand-mangled file. Only the liveness-critical fields keep a type check
        (a non-int pid must not reach ``os.kill``); everything else is display.
        """
        names = {f.name for f in dataclasses.fields(cls)}
        kwargs: dict[str, Any] = {key: value for key, value in data.items() if key in names}
        try:
            kwargs["daemon_state"] = DaemonState(kwargs.get("daemon_state"))
        except ValueError:
            kwargs["daemon_state"] = DaemonState.STOPPED
        if not isinstance(kwargs.get("pid"), int):
            kwargs["pid"] = None
        if not isinstance(kwargs.get("start_token"), str):
            kwargs["start_token"] = None
        return cls(**kwargs)


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

        # Read as UTF-8 to match save(): under a non-UTF-8 locale a non-ASCII
        # current_file or watched directory would otherwise raise
        # UnicodeDecodeError, which is a ValueError and was not caught here.
        # OSError covers an unreadable file; every caller treats an unusable
        # state file as "nothing recorded" rather than an error.
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return WorkerState.from_dict(data)
        except (ValueError, OSError, KeyError, TypeError):
            return WorkerState()

    def save(self, state: WorkerState) -> None:
        """Save state to file atomically."""
        state.last_updated = datetime.now(timezone.utc).isoformat()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.state_path.with_name(f".{self.state_path.name}.tmp")
        tmp_path.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
        os.replace(tmp_path, self.state_path)

    def update(self, **changes: Any) -> WorkerState:
        """Update the given fields and save; an absent keyword means unchanged.

        Unknown names raise: with ``Any``-typed values the old 11-parameter
        UNSET chain bought no type safety, only ~40 lines -- but a typo'd field
        silently creating a dead attribute would be worse than either.
        """
        with self._lock:
            state = self.load()
            for name, value in changes.items():
                if not hasattr(state, name):
                    raise TypeError(f"WorkerState has no field {name!r}")
                setattr(state, name, value)
            self.save(state)
            return state

    def increment(
        self, *, processed: int = 0, failed: int = 0, current_file: Any = UNSET
    ) -> WorkerState:
        """Add to the counters atomically with respect to other threads.

        Callers used to read the state and then write ``count + 1`` in a separate
        ``update`` call. Only the write was inside the lock, so the watcher's
        scan thread and its debounce timers could read the same value and lose an
        increment -- under-reporting progress in ``cementic status --verbose``.
        """
        with self._lock:
            state = self.load()
            state.processed_count += processed
            state.failed_count += failed
            if current_file is not UNSET:
                state.current_file = current_file
            self.save(state)
            return state

    def record_skipped(
        self, path: str, reason: str, *, current_file: Any = UNSET
    ) -> WorkerState:
        """Count a file the watcher refused, and remember which one it was.

        Same lock as ``increment`` and the same reason for it: the initial scan
        and the debounce timers run on different threads. ``current_file`` lets
        a failure path clear the "now working on" display in the same atomic
        write, as ``increment`` does.
        """
        with self._lock:
            state = self.load()
            state.failed_count += 1
            state.skipped_files.append(f"{path}: {reason}")
            if current_file is not UNSET:
                state.current_file = current_file
            # Keep only the most recent, so a directory of symlinks cannot grow
            # the state file without bound. failed_count stays exact.
            if len(state.skipped_files) > MAX_RECORDED_SKIPPED_FILES:
                del state.skipped_files[:-MAX_RECORDED_SKIPPED_FILES]
            self.save(state)
            return state
