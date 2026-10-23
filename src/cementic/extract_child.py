"""Child-process server for isolated extraction (see `extract._run_isolated`).

Run as ``python -m cementic.extract_child <extraction config JSON>``. It imports
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
import sys


def main(argv: list[str]) -> int:
    (extraction_json,) = argv
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
