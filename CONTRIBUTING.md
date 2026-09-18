# Contributing

## Development setup

```bash
uv sync --extra dev
```

Use the generated local Postgres setup from the [README](README.md) for
PostgreSQL-backed integration tests.

## Before proposing a change

```bash
./scripts/check.sh
```

The script runs project tools through `uv`, so it works after `uv sync` without
activating `.venv`. It checks the lockfile, Ruff, mypy, unit tests, and
non-PostgreSQL integration tests. It also runs PostgreSQL integration tests when a suitable database is
reachable. A missing PostgreSQL gate is reported as skipped, not passed.

For a fast loop, run the focused unit test you changed:

```bash
uv run pytest tests/unit/test_cli.py -q
```

## Code and tests

- Keep formatting, imports, and naming compliant with Ruff. Mypy runs in strict
  mode.
- Send command output to stdout and human-facing errors to stderr, so JSON output
  remains parseable.
- Test behavior through the public CLI or service boundary. Prefer structural
  assertions over timing limits.
- Add regression tests through the same path a user takes. Break the fix once to
  confirm the test would have failed before it.

## Runtime changes

After changing workers, daemon management, profiles, or persisted state, run a
small real corpus through indexing, promotion, and search. Unit and integration
tests do not prove that the local embedding runtime and background processes
behave correctly together.
