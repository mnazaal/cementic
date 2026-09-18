# Setup

## Requirements

- Python 3.12 or newer and [`uv`](https://docs.astral.sh/uv/)
- PostgreSQL with the `pgvector` and `vectorscale` extensions
- `llama.cpp`'s `llama-server` on `PATH`
- 8–16 GB RAM for the default embedding backend, plus about 2 GB of disk for
  its model, database data, and extracted-text artifacts

cementic speaks HTTP to an OpenAI-compatible embedding server. It does not
install or link a Python `llama.cpp` binding. Install a CPU, Vulkan, or CUDA
build appropriate for your machine.

cementic is tested on Linux with Python 3.12 and 3.13. macOS and Windows are
best-effort until they are tested.

## Install from a checkout

```bash
git clone https://github.com/mnazaal/cementic.git
cd cementic
uv sync
uv run cementic --help
```

The repository is private. Clone it after obtaining access rather than trying to
install directly from its URL.

## Start local Postgres

Generate a self-contained Postgres setup directory, then start it once:

```bash
uv run cementic init postgres ./cementic-postgres
cd ./cementic-postgres
docker compose up -d             # or: podman compose up -d
cd ..
uv run cementic doctor
```

The generated Compose service binds only to `127.0.0.1:5432` and uses the
`CEMENTIC_DB_*` defaults. Its `restart: unless-stopped` policy means you do not
need to run Compose before every cementic command.

To use your own Postgres, set `CEMENTIC_DB_URL` or the individual
`CEMENTIC_DB_*` values. The database must provide both required extensions.

## First collection

```bash
# Optional: load the model before the first query.
uv run cementic embedding start

# Watch and index one or more directories into one collection.
uv run cementic start ~/research-papers -c research

# Inspect progress, then search the active or first building revision.
uv run cementic status
uv run cementic search "variational inference" -c research
```

A missing model downloads on first use when auto-download is enabled and its
resolved path is inside cementic's user-data directory. For a model path outside
that directory, place the model file there yourself; cementic will not download
to it. `doctor` reports the resolved model path without downloading it. A cold
model start can take 30 seconds or longer.

## Alternative service setup

For a host where a container engine is not always available, the generated
Postgres directory includes an optional Podman Quadlet user service at
`quadlet/cementic-postgres.container`. Read its header and generated README for
the one-time install steps. With `loginctl enable-linger $USER`, that service
can survive logout and reboot.

For long indexing jobs, use the user systemd units in
[`packaging/systemd/`](../packaging/systemd/). They manage the embedding server,
per-collection watcher and worker, and server-health restart timer. Do not run
those units alongside `cementic start`; each collection permits only one worker.
