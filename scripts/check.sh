#!/usr/bin/env bash
# Run every gate CI runs, in one command.
#
# Exists because "I ran the tests" meant different things locally and in CI:
# `pytest tests/unit` plus the non-pg integration job is green while the `pg`
# job is red, and that combination reached main. The jobs mirror .github/
# workflows/ci.yml -- keep them in step.
#
# The pg job needs a reachable PostgreSQL. When there is none it is reported as
# SKIPPED rather than passed, because a silently absent check is what this
# script exists to prevent.

set -uo pipefail
cd "$(dirname "$0")/.."

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
    python - <<'PY'
import sys
try:
    from tests.integration.conftest import _server_reachable
    sys.exit(0 if _server_reachable() else 1)
except Exception:
    sys.exit(1)
PY
}

run ruff              ruff check src/ tests/
run mypy              mypy src/
run unit              pytest tests/unit -q
run integration       pytest tests/integration -m "not pg" -q

if postgres_reachable; then
    run integration-pg pytest tests/integration -m pg -q
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
echo "all checks passed"
