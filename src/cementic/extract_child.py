"""Child-process server for isolated extraction (see `extract._run_isolated`).

Run as ``python -m cementic.extract_child <extraction config JSON> <parent pid>``. It imports
once, then answers one JSON object per stdin line (``{"target": "module:function",
"path": ...}``) with one JSON line on the real stdout: ``{"text": ...}`` or
``{"error": {"type": ..., "message": ...}}``. JSON escapes keep lone surrogates
intact. A Python exception is an answer, not a death: the child stays up. Stdin
EOF ends it; anything else (a signal, an interpreter abort) is the parent's to
describe from the return code.
"""

from __future__ import annotations

import importlib
import json
import os
import resource
import signal
import sys


def _tie_to_parent(parent_pid: int) -> None:
    """Die with the parent, and ignore the stop signals meant for the worker.

    A stop (systemd's control-group SIGTERM, Ctrl-C) reaches the child with the
    worker; ignoring it lets the current parse finish as in-process extraction
    did. An abrupt parent death (SIGKILL) sends no stop, so on Linux ask the
    kernel to kill the child with it, then check the parent did not die first.
    """
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if sys.platform == "linux":
        try:
            import ctypes

            ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
        except (OSError, AttributeError):
            pass
    if os.getppid() != parent_pid:
        os._exit(0)


def main(argv: list[str]) -> int:
    extraction_json, parent_pid = argv
    _tie_to_parent(int(parent_pid))
    # A parser may print to fd 1; keep the real stdout for the protocol alone.
    reply = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    # Limit before the heavy imports, so the parser runs under it.
    from cementic.config import Config, ExtractionConfig

    extraction = ExtractionConfig.model_validate_json(extraction_json)
    if extraction.memory_limit_mb:
        limit = extraction.memory_limit_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    config = Config(extraction=extraction)
    for line in sys.stdin.buffer:
        request = json.loads(line)
        try:
            module_name, _, function_name = request["target"].partition(":")
            function = getattr(importlib.import_module(module_name), function_name)
            answer: dict[str, object] = {"text": function(request["path"], config)}
        except Exception as error:
            answer = {"error": {"type": type(error).__name__, "message": str(error)}}
        reply.write(json.dumps(answer) + "\n")
        reply.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
