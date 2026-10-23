"""Child-process entry point for isolated extraction (see `extract._run_isolated`).

Run as ``python -m cementic.extract_child <module:function> <path> <extraction
config JSON>``. Text goes to stdout as UTF-8; a Python exception goes to stderr
as one final JSON line and exit code 3. Anything else (a signal, an interpreter
abort) is the parent's to describe from the return code.
"""

from __future__ import annotations

import importlib
import json
import os
import resource
import sys

EXCEPTION_EXIT_CODE = 3


def main(argv: list[str]) -> int:
    target, path, extraction_json = argv
    # A parser may print to fd 1; keep the real stdout for the text alone.
    result = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    try:
        # Limit before the heavy imports, so the parser runs under it.
        from cementic.config import Config, ExtractionConfig

        extraction = ExtractionConfig.model_validate_json(extraction_json)
        if extraction.memory_limit_mb:
            limit = extraction.memory_limit_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        module_name, _, function_name = target.partition(":")
        function = getattr(importlib.import_module(module_name), function_name)
        text = function(path, Config(extraction=extraction))
    except Exception as error:
        report = {"type": type(error).__name__, "message": str(error)}
        sys.stderr.write("\n" + json.dumps(report) + "\n")
        return EXCEPTION_EXIT_CODE
    result.write(text.encode("utf-8", errors="surrogatepass"))
    result.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
