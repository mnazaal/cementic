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
    """Metadata for one managed background process.

    ``start_token`` is the process's kernel start time captured at spawn. It lets
    liveness checks distinguish *this* process from an unrelated one that later
    reuses the same PID (after a reboot or PID wraparound). ``None`` means the
    token is unknown (e.g. state written by an older version), in which case
    callers fall back to a PID-only check.
    """

    name: str
    pid: int
    log_file: str
    start_token: str | None = None


def is_pid_running(pid: int) -> bool:
    """Check whether a PID is currently running."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, OSError):
        return False


def process_start_token(pid: int) -> str | None:
    """Return a stable per-process start-time token, or None if unavailable.

    Reads field 22 (``starttime``) of ``/proc/<pid>/stat`` on Linux. The comm
    field (2) may contain spaces and parentheses, so we split after the final
    ``)`` before counting fields. Returns None on any platform without ``/proc``
    or if the process is gone, so callers degrade to a PID-only check.
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (OSError, ValueError):
        return None
    try:
        after_comm = stat[stat.rindex(")") + 2 :]
        # After the comm field, index 0 is field 3 (state); starttime is field 22.
        return after_comm.split()[19]
    except (ValueError, IndexError):
        return None


def is_managed_process_alive(pid: int, start_token: str | None) -> bool:
    """Check that PID is running *and* is the same process we recorded.

    When ``start_token`` is known it must match the live process's current start
    token; a mismatch means the PID was recycled by an unrelated process, so this
    returns False. When ``start_token`` is None (unknown), it falls back to a
    bare PID check for backward compatibility.
    """
    if pid <= 0:
        return False
    if not is_pid_running(pid):
        return False
    if start_token is None:
        return True
    return process_start_token(pid) == start_token


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
