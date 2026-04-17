"""Background worker supervision helpers."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

SupervisorState = dict[str, object]


@dataclass
class ManagedProcess:
    """Metadata for one managed background process."""

    name: str
    pid: int
    log_file: str


def is_pid_running(pid: int) -> bool:
    """Check whether a PID is currently running."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, OSError):
        return False


def load_supervisor_state(state_path: Path) -> SupervisorState:
    """Load background supervisor state from disk."""
    if not state_path.exists():
        return {}

    try:
        payload = json.loads(state_path.read_text())
        return cast(SupervisorState, payload)
    except json.JSONDecodeError:
        return {}


def save_supervisor_state(state_path: Path, state: SupervisorState) -> None:
    """Persist background supervisor state."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2))


def spawn_detached(command: list[str], log_file: Path) -> int:
    """Spawn a detached background process and return its PID."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return process.pid


def wait_for_exit(pids: list[int], timeout_seconds: float = 20.0) -> list[int]:
    """Wait for PIDs to exit and return any still running."""
    if not pids:
        return []

    deadline = time.time() + timeout_seconds
    remaining = [pid for pid in pids if is_pid_running(pid)]
    while remaining and time.time() < deadline:
        time.sleep(0.2)
        remaining = [pid for pid in remaining if is_pid_running(pid)]
    return remaining


def force_kill(pids: list[int]) -> list[int]:
    """Send SIGKILL to PIDs and return any that couldn't be killed."""
    remaining = []
    for pid in pids:
        try:
            os.kill(pid, 9)
        except (ProcessLookupError, OSError):
            pass
        else:
            if is_pid_running(pid):
                remaining.append(pid)
    return remaining
