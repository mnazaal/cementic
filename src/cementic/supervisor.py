"""Background worker supervision helpers."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, cast

SupervisorState = dict[str, object]

#: Root of the `/proc` filesystem `find_pids_by_cmdline` scans. A module-level
#: constant (rather than a hardcoded literal) so tests can point it at a fake
#: tree under `tmp_path` while still exercising the real scan logic, the same
#: way `Path.read_text` is patched to fake `/proc/<pid>/stat` above.
_PROC_ROOT = Path("/proc")


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


def managed_process_pid(record: dict[str, object]) -> int:
    """Read back the ``pid`` field of a persisted process record (dict form of
    ``ManagedProcess``), defensively re-checking its type after a JSON round-trip."""
    pid = record.get("pid", 0)
    return pid if isinstance(pid, int) else 0


def managed_process_start_token(record: dict[str, object]) -> str | None:
    """Read back the ``start_token`` field of a persisted process record."""
    token = record.get("start_token")
    return token if isinstance(token, str) else None


def _proc_stat_fields(pid: int) -> list[str] | None:
    """Fields of ``/proc/<pid>/stat`` from field 3 (state) onward, or None.

    The comm field (2) may contain spaces and parentheses, so we split after the
    final ``)``. Returns None on any platform without ``/proc`` or if the
    process is gone, so callers degrade to a PID-only check.
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (OSError, ValueError):
        return None
    try:
        return stat[stat.rindex(")") + 2 :].split()
    except ValueError:
        return None


def _proc_cmdline(pid_dir: Path) -> list[str] | None:
    """Argv of a `/proc/<pid>/cmdline` file as a list, or None.

    ``cmdline`` is NUL-separated (not space-separated like ``stat``), so a
    value containing a space -- a fingerprint hex digest never does, but a
    filesystem path might -- is not split apart. Returns None on any platform
    without `/proc`, an unreadable entry, or a process that's gone, mirroring
    ``_proc_stat_fields``'s degrade rule.
    """
    try:
        raw = (pid_dir / "cmdline").read_bytes()
    except OSError:
        return None
    if not raw:
        return None
    return raw.decode("utf-8", errors="replace").rstrip("\0").split("\0")


def find_pids_by_cmdline(
    match: Callable[[list[str]], bool], *, proc_root: Path | None = None
) -> list[int]:
    """PIDs of *this user's* processes whose `/proc/<pid>/cmdline` satisfies ``match``.

    Used to recover a daemon's identity from the OS when its pid-file record is
    missing or stale, by matching the exact command line it was spawned with --
    never by scanning for "whatever is on the port". A process owned by another
    uid is never returned, even if its command line matches.

    ``proc_root`` defaults to module-level ``_PROC_ROOT`` (looked up at call
    time, not baked in as a default value) so tests can patch that constant and
    still exercise this function through a real caller, without every caller
    needing to thread a ``proc_root`` parameter through.

    Degrades to ``[]`` on any platform without `/proc`, or when it can't be
    listed -- exactly as ``_proc_stat_fields`` degrades to None. Never raises.
    """
    root = proc_root if proc_root is not None else _PROC_ROOT
    try:
        entries = os.listdir(root)
    except OSError:
        return []
    my_uid = os.getuid()
    matches: list[int] = []
    for name in entries:
        if not name.isdigit():
            continue
        pid_dir = root / name
        cmdline = _proc_cmdline(pid_dir)
        if cmdline is None or not match(cmdline):
            continue
        try:
            owner_uid = pid_dir.stat().st_uid
        except OSError:
            continue
        if owner_uid != my_uid:
            continue
        matches.append(int(name))
    return matches


def is_pid_running(pid: int) -> bool:
    """Check whether a PID belongs to a live process.

    A zombie (exited but not yet reaped by its parent) still answers
    ``kill(pid, 0)``, so a plain signal probe reports a finished worker as
    running -- which would make ``cementic stop`` wait out its full timeout and
    then tell the user to ``--force`` a process that already exited. Detached
    workers are reparented on CLI exit and are reaped late (or never, under an
    init that does not reap), so this is the common case, not a corner one.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # EPERM proves the process exists -- we just may not signal it. Folding
        # this into "not running" made `cementic stop` report success while
        # workers started under another uid (sudo, a service account, a user
        # namespace) kept indexing. force_kill already reasons this way.
        return True
    except OSError:
        return False
    fields = _proc_stat_fields(pid)
    if fields is None:
        return True  # no /proc: fall back to the signal probe's answer
    return fields[0] != "Z" if fields else True


def process_start_token(pid: int) -> str | None:
    """Return a stable per-process start-time token, or None if unavailable.

    Reads field 22 (``starttime``) of ``/proc/<pid>/stat`` on Linux.
    """
    fields = _proc_stat_fields(pid)
    if fields is None:
        return None
    try:
        # After the comm field, index 0 is field 3 (state); starttime is field 22.
        return fields[19]
    except IndexError:
        return None


def is_managed_process_alive(pid: int, start_token: str | None) -> bool:
    """Check that PID is running *and* is the same process we recorded.

    When ``start_token`` is known it must match the live process's current start
    token; a mismatch means the PID was recycled by an unrelated process, so this
    returns False. When the token is unknown -- not recorded (state written by an
    older version) or not readable -- it falls back to a bare PID check, because
    "cannot tell" must not be reported as "not ours".
    """
    if pid <= 0:
        return False
    if not is_pid_running(pid):
        return False
    if start_token is None:
        return True
    live_token = process_start_token(pid)
    if live_token is None:
        # The token could not be *read* -- /proc mounted with hidepid, or a
        # process owned by another user in a namespace -- which is not evidence
        # of a mismatch. Treating it as one made `cementic stop` skip a live
        # worker, so it never joined the not-stopped set and the "nothing left
        # running" branch deleted its state files with "cleared stale state"
        # while it kept indexing: the exact outcome the EPERM handling in
        # is_pid_running exists to prevent, reached through a different door.
        return True
    return live_token == start_token


def load_supervisor_state(state_path: Path) -> SupervisorState:
    """Load background supervisor state from disk, or ``{}`` if unusable.

    Tolerates an unreadable or non-object file as well as malformed JSON: every
    caller treats "no state" as "nothing recorded", whereas an uncaught OSError
    or a JSON ``null`` reaching ``state.get(...)`` would abort `cementic
    start`/`status`/`stop` with a traceback.
    """
    if not state_path.exists():
        return {}

    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return cast(SupervisorState, payload)


def save_supervisor_state(state_path: Path, state: SupervisorState) -> None:
    """Persist background supervisor state atomically (a concurrent ``status``
    must never read a torn file)."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_path.with_name(f".{state_path.name}.tmp")
    tmp_path.write_text(json.dumps(state, indent=2))
    os.replace(tmp_path, state_path)


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
        except ProcessLookupError:
            continue  # already gone
        except OSError:
            remaining.append(pid)  # e.g. EPERM: alive but not ours to kill
            continue
        if is_pid_running(pid):
            remaining.append(pid)
    return remaining
