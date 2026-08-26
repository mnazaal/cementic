"""Advisory file locking for operations that must not run concurrently.

cementic's coordination is otherwise state-file based, which is fine for
recording *what* is running but cannot make a check-then-act sequence atomic.
Two of those sequences race destructively:

- starting the embedding daemon (check it is absent, then spawn), where both
  racers spawn and the loser's PID can end up in the pid file, orphaning a live
  daemon that holds the port and a multi-GB model;
- ``cementic start`` (check nothing is running, then spawn and record), where the
  second writer's supervisor record replaces the first's, leaving the first pair
  running and unfindable by ``cementic stop``.

``fcntl.flock`` is the right primitive here: it is released automatically when
the process exits or crashes, so a killed holder cannot wedge the lock the way a
lock *file* whose existence means "held" would.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import os
import time
from collections.abc import Iterator
from pathlib import Path


class LockUnavailableError(RuntimeError):
    """Raised when another process holds the lock and waiting was not requested."""


@contextlib.contextmanager
def file_lock(path: Path, *, timeout: float | None = None) -> Iterator[None]:
    """Hold an exclusive advisory lock on ``path`` for the duration of the block.

    ``timeout=None`` blocks until the lock is available; ``timeout=0`` fails
    immediately with :class:`LockUnavailableError`. POSIX-only by design: the
    old non-POSIX fallback was a silent no-op lock, a failure mode worse than
    the ImportError this now raises on such a platform.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        _acquire(handle, path, timeout)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        os.close(handle)


def _acquire(handle: int, path: Path, timeout: float | None) -> None:
    if timeout is None:
        fcntl.flock(handle, fcntl.LOCK_EX)
        return
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except OSError as error:
            if error.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            if time.monotonic() >= deadline:
                raise LockUnavailableError(f"another cementic process holds {path}") from None
            time.sleep(0.05)
