"""Tests for cross-process advisory locking."""

from __future__ import annotations

import multiprocessing
import time
from pathlib import Path

import pytest

from cementic.filelock import LockUnavailableError, file_lock


def _hold_lock(path: str, hold_seconds: float, ready, released) -> None:
    """Child process: take the lock, signal, hold, release."""
    with file_lock(Path(path)):
        ready.set()
        time.sleep(hold_seconds)
    released.set()


class TestFileLock:
    """The lock must actually exclude a *separate process*.

    A threading lock would not help here: the sequences being guarded --
    daemon stop-then-spawn, and start's check-then-record -- race between
    independent cementic invocations.
    """

    def test_lock_is_reentrant_across_sequential_uses(self, tmp_path: Path) -> None:
        lock = tmp_path / "a.lock"
        with file_lock(lock, timeout=0):
            pass
        with file_lock(lock, timeout=0):
            pass  # released cleanly, so a later acquire succeeds

    def test_second_process_cannot_take_a_held_lock(self, tmp_path: Path) -> None:
        lock = tmp_path / "b.lock"
        ctx = multiprocessing.get_context("spawn")
        ready, released = ctx.Event(), ctx.Event()
        child = ctx.Process(target=_hold_lock, args=(str(lock), 2.0, ready, released))
        child.start()
        try:
            assert ready.wait(timeout=10), "child never acquired the lock"
            with pytest.raises(LockUnavailableError):
                with file_lock(lock, timeout=0):
                    pass
        finally:
            child.join(timeout=15)

    def test_waiting_acquires_once_the_holder_releases(self, tmp_path: Path) -> None:
        lock = tmp_path / "c.lock"
        ctx = multiprocessing.get_context("spawn")
        ready, released = ctx.Event(), ctx.Event()
        child = ctx.Process(target=_hold_lock, args=(str(lock), 1.0, ready, released))
        child.start()
        try:
            assert ready.wait(timeout=10), "child never acquired the lock"
            with file_lock(lock, timeout=30):
                assert released.is_set() or not child.is_alive()
        finally:
            child.join(timeout=15)

    def test_lock_is_released_when_the_holder_dies(self, tmp_path: Path) -> None:
        """flock is released by the kernel on exit.

        This is why flock is used rather than a lock *file* whose existence
        means "held": a killed holder would wedge that one permanently.
        """
        lock = tmp_path / "d.lock"
        ctx = multiprocessing.get_context("spawn")
        ready, released = ctx.Event(), ctx.Event()
        child = ctx.Process(target=_hold_lock, args=(str(lock), 60.0, ready, released))
        child.start()
        try:
            assert ready.wait(timeout=10), "child never acquired the lock"
            child.kill()
            child.join(timeout=15)
            with file_lock(lock, timeout=10):
                pass  # must not hang or raise
        finally:
            if child.is_alive():  # pragma: no cover - defensive
                child.kill()
                child.join(timeout=5)
