"""Shared process plumbing for the background workers.

The source watcher and the pipeline worker carried identical copies of their
logger setup and fatal-reason reporting; this is the single implementation.
"""

from __future__ import annotations

import logging
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from cementic.state import StateManager


def setup_worker_logger(name: str, log_file: Path | None, owner: str) -> logging.Logger:
    """File logger for a background worker, idempotent per log file."""
    if log_file is None:
        raise RuntimeError(f"{owner} log file is not configured")
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    if not any(
        isinstance(handler, logging.FileHandler) and handler.baseFilename == str(log_file)
        for handler in logger.handlers
    ):
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    return logger


class _ShutdownAware(Protocol):
    """The pair of attributes a worker's signal handler reads and writes."""

    _shutdown_signal: int | None
    _shutdown_event: threading.Event


def handle_shutdown_signal(worker: _ShutdownAware, signum: int, frame: object) -> None:
    """Signal handler: set the shutdown flag and nothing else.

    Python runs handlers on the main thread between bytecodes, so anything
    that takes a lock the main thread may already hold deadlocks the process
    -- and since this *is* the SIGTERM handler, a deadlocked process can then
    only be killed with SIGKILL. `stop()` takes the state-file lock (and, for
    the source watcher, joins the observer thread too), so it runs from
    `start()`'s `finally` instead.
    """
    worker._shutdown_signal = signum
    worker._shutdown_event.set()


def report_fatal(
    logger: logging.Logger, state_manager: StateManager, message: str, *args: Any
) -> str:
    """Log a fatal startup reason, echo it to stderr, publish it, and return it.

    The module logger writes to its own file, but `cementic start` points the
    user at the spawned process's stdout/stderr log. Without the echo the user
    is sent to a file that cannot explain why the worker exited.

    `cementic start` only watches for two seconds before declaring success, so
    a startup failure past that grace period (e.g. the embedding daemon still
    loading its model) was previously reported nowhere `status` or `doctor`
    look -- only in the background log the user has to know to check. Writing
    it to `last_error` puts it in the one place `cementic status` already
    renders. A failure to write it must not mask the fatal reason itself, so
    it is logged and swallowed rather than raised.
    """
    logger.error(message, *args)
    rendered = message % args if args else message
    print(rendered, file=sys.stderr, flush=True)
    try:
        state_manager.update(
            last_error=rendered[:500],
            last_error_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception:
        logger.exception("Could not record fatal startup reason to the state file")
    return rendered
