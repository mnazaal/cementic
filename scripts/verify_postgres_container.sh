#!/usr/bin/env bash
# Verifies the full Postgres container + Quadlet path end-to-end, on a real
# machine with working rootless Podman (this can't run in a sandboxed dev
# container -- rootless Podman needs a real /etc/subuid /etc/subgid mapping).
#
# Run this once before publishing, against a throwaway `cementic init postgres`
# dir, to confirm the Compose build/up path and the Quadlet unit both actually
# work as documented in templates/postgres/README.md.
#
# Fully isolated from any real cementic-postgres setup you already have
# running: container name, image tag, volume name, Quadlet unit filename, and
# port are all rewritten to a per-run "-verify-<suffix>" variant before
# anything starts, so this never needs your live instance stopped. The only
# guard needed is against a leftover verify run of its own (e.g. a prior
# invocation killed mid-way); that's checked and cleaned up automatically.
#
# Usage: ./scripts/verify_postgres_container.sh
set -euo pipefail

SUFFIX="$(date +%s)-$$"
SCRATCH="$(mktemp -d "/tmp/cementic-pg-verify.XXXXXX")"
PGDIR="$SCRATCH/pgdir"
CEMENTIC_BIN="${CEMENTIC_BIN:-cementic}"
CONTAINER_NAME="cementic-postgres-verify-${SUFFIX}"
IMAGE_TAG="cementic-postgres-vectorscale-verify-${SUFFIX}:latest"
VOLUME_NAME="cementic-postgres-verify-data-${SUFFIX}"
UNIT_NAME="cementic-postgres-verify-${SUFFIX}.container"
SERVICE_NAME="cementic-postgres-verify-${SUFFIX}.service"
DB_PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"

# cementic doctor resolves a *relative* model_path against the
# current working directory first (see config.py: resolve_llama_model_path),
# so this script must never `cd` away from the real project directory --
# doing so makes the model-file check fail for reasons unrelated to Postgres.
compose() { (cd "$PGDIR" && podman compose "$@"); }

cleanup() {
  set +e
  systemctl --user stop "$SERVICE_NAME" >/dev/null 2>&1
  rm -f "$HOME/.config/containers/systemd/$UNIT_NAME"
  systemctl --user daemon-reload >/dev/null 2>&1
  compose down -v >/dev/null 2>&1
  podman rm -f "$CONTAINER_NAME" >/dev/null 2>&1
  podman volume rm -f "$VOLUME_NAME" >/dev/null 2>&1
  podman rmi -f "$IMAGE_TAG" >/dev/null 2>&1
  echo "--- scratch dir left at $SCRATCH for inspection ---"
}
trap cleanup EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }
ok() { echo "OK: $*"; }

command -v podman >/dev/null || fail "podman not found on PATH"
command -v "$CEMENTIC_BIN" >/dev/null || fail "cementic not found on PATH (set CEMENTIC_BIN=...)"
command -v python3 >/dev/null || fail "python3 not found on PATH (needed for free-port detection)"

echo "== podman sanity (this is the step that fails in rootless-without-subuid sandboxes) =="
podman info >/dev/null || fail "podman info failed -- likely missing /etc/subuid or /etc/subgid entries for $(whoami)"
ok "podman info succeeded"

echo "== cementic init postgres =="
"$CEMENTIC_BIN" init postgres "$PGDIR"
[ -f "$PGDIR/compose.yml" ] || fail "compose.yml not generated"
[ -f "$PGDIR/Containerfile" ] || fail "Containerfile not generated"
[ -f "$PGDIR/quadlet/cementic-postgres.container" ] || fail "quadlet unit not generated"
ok "cementic init postgres generated the expected files"

echo "== isolating names: container=$CONTAINER_NAME image=$IMAGE_TAG volume=$VOLUME_NAME port=$DB_PORT =="
sed -i \
  -e "s/container_name: cementic-postgres/container_name: ${CONTAINER_NAME}/" \
  -e "s#image: cementic-postgres-vectorscale:latest#image: ${IMAGE_TAG}#" \
  "$PGDIR/compose.yml"
ok "compose.yml rewritten for isolation"

sed -i \
  -e "s/ContainerName=cementic-postgres/ContainerName=${CONTAINER_NAME}/" \
  -e "s#Image=localhost/cementic-postgres-vectorscale:latest#Image=localhost/${IMAGE_TAG}#" \
  -e "s/PublishPort=127.0.0.1:5432:5432/PublishPort=127.0.0.1:${DB_PORT}:5432/" \
  -e "s/Volume=cementic-postgres-data:/Volume=${VOLUME_NAME}:/" \
  "$PGDIR/quadlet/cementic-postgres.container"
ok "quadlet unit rewritten for isolation"

export CEMENTIC_DB_PORT="$DB_PORT"

echo "== podman compose build =="
# Guard: some podman-compose versions additionally tag the build result with
# the bare "cementic-postgres-vectorscale:latest" name (the one a real,
# non-verify deployment uses) regardless of this compose.yml's rewritten
# `image:` field. Record that tag's image ID beforehand so we can detect and
# reverse any such side effect rather than silently leaving your real image
# tag repointed.
REAL_TAG="cementic-postgres-vectorscale:latest"
PRE_BUILD_REAL_ID="$(podman image inspect --format '{{.Id}}' "$REAL_TAG" 2>/dev/null || true)"

compose build || fail "podman compose build failed"
ok "image built"

POST_BUILD_REAL_ID="$(podman image inspect --format '{{.Id}}' "$REAL_TAG" 2>/dev/null || true)"
if [ "$PRE_BUILD_REAL_ID" != "$POST_BUILD_REAL_ID" ]; then
  if [ -z "$PRE_BUILD_REAL_ID" ]; then
    echo "WARNING: build side-tagged '$REAL_TAG' (it didn't exist before) -- untagging it now, verify image is unaffected"
    podman rmi "$REAL_TAG" >/dev/null 2>&1 || true
  else
    echo "WARNING: build repointed '$REAL_TAG' from $PRE_BUILD_REAL_ID to $POST_BUILD_REAL_ID -- restoring it"
    podman tag "$PRE_BUILD_REAL_ID" "$REAL_TAG" || fail "could not restore '$REAL_TAG' to its pre-build image ID"
  fi
  ok "'$REAL_TAG' restored to its pre-verify state"
else
  ok "'$REAL_TAG' untouched by the verify build"
fi

echo "== podman compose up -d =="
compose up -d || fail "podman compose up -d failed"

wait_healthy() {
  local name="$1"
  local i
  for i in $(seq 1 30); do
    status="$(podman inspect --format '{{.State.Health.Status}}' "$name" 2>/dev/null || echo unknown)"
    [ "$status" = "healthy" ] && return 0
    sleep 2
  done
  return 1
}

echo "== waiting for healthy (up to 60s) =="
wait_healthy "$CONTAINER_NAME" && ok "container reported healthy" || {
  compose logs postgres || true
  fail "container never became healthy within 60s"
}

echo "== cementic doctor against the running container =="
"$CEMENTIC_BIN" doctor || fail "cementic doctor failed against the container"
ok "cementic doctor passed"

echo "== tearing down compose stack =="
compose down -v || fail "podman compose down -v failed"
ok "compose stack removed (including volume)"

echo "== Quadlet unit: install + start =="
mkdir -p "$HOME/.config/cementic"
touch "$HOME/.config/cementic/postgres.env"
mkdir -p "$HOME/.config/containers/systemd"
cp "$PGDIR/quadlet/cementic-postgres.container" "$HOME/.config/containers/systemd/$UNIT_NAME"
systemctl --user daemon-reload || fail "daemon-reload failed"
systemctl --user start "$SERVICE_NAME" || {
  journalctl --user -u "$SERVICE_NAME" --no-pager -n 50 || true
  fail "systemctl --user start $SERVICE_NAME failed"
}
ok "Quadlet service started"

echo "== waiting for Quadlet-managed container healthy (up to 60s) =="
wait_healthy "$CONTAINER_NAME" && ok "Quadlet-managed container reported healthy" || {
  systemctl --user status "$SERVICE_NAME" --no-pager || true
  fail "Quadlet-managed container never became healthy within 60s"
}

echo "== cementic doctor against the Quadlet-managed container =="
"$CEMENTIC_BIN" doctor || fail "cementic doctor failed against the Quadlet-managed container"
ok "cementic doctor passed against Quadlet-managed instance"

echo
echo "ALL CHECKS PASSED"
