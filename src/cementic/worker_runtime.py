"""Shared process plumbing for the background workers.

The source watcher and the pipeline worker carried identical copies of their
logger setup and fatal-reason reporting; this is the single implementation.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any


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


def report_fatal(logger: logging.Logger, message: str, *args: Any) -> str:
    """Log a fatal startup reason, echo it to stderr, and return it rendered.

    The module logger writes to its own file, but `cementic start` points the
    user at the spawned process's stdout/stderr log. Without the echo the user
    is sent to a file that cannot explain why the worker exited.
    """
    logger.error(message, *args)
    rendered = message % args if args else message
    print(rendered, file=sys.stderr, flush=True)
    return rendered
