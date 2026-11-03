#!/usr/bin/env bash
# Run the lockfile, lint, type, unit and integration gates locally.
#
# Exists because "I ran the tests" meant different things locally and in CI:
# `pytest tests/unit` plus the non-pg integration job is green while the `pg`
# job is red, and that combination reached main. CI additionally installs a
# fresh wheel with unlocked dependencies and repeats the pg suite against
# newer SQLAlchemy; those release gates are not reproduced by this script.
#
# The pg job needs a reachable PostgreSQL. When there is none it is reported as
# SKIPPED rather than passed, because a silently absent check is what this
# script exists to prevent.

set -uo pipefail
cd "$(dirname "$0")/.."

# `--extra dev` on every `uv run`: ruff, mypy and pytest live in that extra, so
# a plain `uv run` in a fresh checkout syncs an environment without them. The
# tools then resolve from whatever is on PATH, or not at all, and the pg probe
# fails to import its conftest and reports PostgreSQL as missing (2026-10-22).

log_dir="${TMPDIR:-/tmp}/cementic-check.$$"
mkdir -p "$log_dir"
failures=()
skipped=()

run() {
    local name="$1"
    shift
    printf '%-28s' "$name"
    if "$@" >"$log_dir/$name.log" 2>&1; then
        echo "ok"
    else
        echo "FAILED  ($log_dir/$name.log)"
        failures+=("$name")
    fi
}

postgres_reachable() {
    uv run --extra dev python - <<'PY'
import sys
try:
    from tests.integration.conftest import _server_reachable
    sys.exit(0 if _server_reachable() else 1)
except Exception:
    sys.exit(1)
PY
}

# CI's locked jobs check this; the installed-wheel job deliberately resolves
# without the lock. A version bump that skips `uv lock` otherwise reaches local
# tests but fails the locked jobs -- which is how v0.2.0 shipped red.
run lockfile          uv lock --check
run ruff              uv run --extra dev ruff check src/ tests/
run mypy              uv run --extra dev mypy src/
run unit              uv run --extra dev pytest tests/unit -q
run integration       uv run --extra dev pytest tests/integration -m "not pg" -q

if postgres_reachable; then
    run integration-pg uv run --extra dev pytest tests/integration -m pg -q
else
    printf '%-28s%s\n' "integration-pg" "SKIPPED (no PostgreSQL reachable)"
    skipped+=("integration-pg")
fi

echo
if ((${#failures[@]})); then
    echo "FAILED: ${failures[*]}"
    exit 1
fi
if ((${#skipped[@]})); then
    echo "passed, but did not run: ${skipped[*]} — not a full check"
    exit 2
fi
echo "local checks passed (CI also tests an unlocked wheel and newer SQLAlchemy)"
