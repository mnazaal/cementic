# cementic

Concrete semantic search for PDFs.

`cementic` watches one or more directories, registers PDFs, extracts text, chunks that text, embeds the chunks, and serves semantic search over the active revision for each collection.

## How It Works

The index is built as a versioned pipeline:

- extractor profile -> extracted document artifact
- chunk profile -> chunk artifacts
- embedding profile -> chunk embeddings
- pipeline revision -> the search-visible release that ties those three profiles together

This keeps old search available while a new extractor, chunking policy, or embedding model builds in the background.

## Features

- Watches directories for PDFs and records them as source documents
- Stores extracted text as compressed artifacts for cheap rechunking
- Separates extraction, chunking, and embedding so each stage can evolve independently
- Keeps one active searchable revision per collection while a replacement revision builds
- Uses explicit `cementic collection promote` to switch search to a new ready revision

## Installation

```bash
# From PyPI (once published):
uv tool install cementic
# or
pipx install cementic

# From source (development):
uv pip install -e ".[dev]"
```

## Prerequisites

- **Podman** or **Docker** — used to run PostgreSQL with pgvector + vectorscale
- **Python 3.10+**
- **8-16 GB RAM** recommended when using the default llama.cpp embedding backend (the model loads into memory)
- **Disk space**: ~2 GB for the llama.cpp model, plus PostgreSQL data and artifact storage

## Setup

When you run `cementic start`, cementic can automatically:

- start the PostgreSQL vectorscale container when using a local database host
- build a local PostgreSQL image with packaged `pgvector` + `vectorscale` when needed
- keep container build files under `containers/`, while runtime uses Podman commands
- wait for services to become healthy
- pull missing Ollama models only when the Ollama backend is selected
- validate or auto-download the configured `llama.cpp` model when using the default backend

## Usage

```bash
# Start watching a collection and building its target revision
cementic start /path/to/pdfs --collection research

# Inspect current worker state and active/building revisions
cementic status

# Show known collections
cementic collection list

# Search the active revision only
cementic search "vector database design" -c research

# Promote the newest ready revision to active
cementic collection promote research

# Inspect revision history for one collection
cementic collection revisions research

# Stop background workers and local infra
cementic stop

# Delete one collection and its stored artifacts
cementic collection remove research --force
```

## Revision behavior

- Model change:
  - reuses extracted text and chunks
  - builds only new embeddings
- Chunking change:
  - reuses extracted text
  - rebuilds chunks and embeddings
- Extractor change:
  - rebuilds extraction, chunks, and embeddings

Search keeps using the active revision until you explicitly run `cementic collection promote`.
By default, embeddings use `llama.cpp`, so Ollama is optional and only needed when you set
`CEMENTIC_PIPELINE_EMBEDDING_PROVIDER=ollama`.

For a brand-new collection with no active revision yet, search can use the in-progress build and return
partial results from chunks whose embeddings are already available.

## Configuration

Configuration is driven by environment variables.

```bash
# Database (required: set CEMENTIC_DB_PASSWORD before starting)
export CEMENTIC_DB_HOST=localhost
export CEMENTIC_DB_PORT=5432
export CEMENTIC_DB_NAME=cementic
export CEMENTIC_DB_USER=cementic
export CEMENTIC_DB_PASSWORD=your-secure-password

# Extraction
export CEMENTIC_EXTRACT_BACKEND=pymupdf4llm
export CEMENTIC_EXTRACT_USE_OCR=false

# Artifact storage
export CEMENTIC_STORAGE_ARTIFACTS_PATH=~/.local/share/cementic/artifacts

# Pipeline chunking
export CEMENTIC_PIPELINE_CHUNK_SIZE=512
export CEMENTIC_PIPELINE_CHUNK_OVERLAP=128

# Embedding provider selection (default)
export CEMENTIC_PIPELINE_EMBEDDING_PROVIDER=llama-cpp

# llama.cpp default path
export CEMENTIC_LLAMA_MODEL_PATH=./models/nomic-embed-text-v2-moe.Q8_0.gguf

# Optional Ollama backend
# export CEMENTIC_PIPELINE_EMBEDDING_PROVIDER=ollama
# export CEMENTIC_OLLAMA_HOST=http://localhost:11434
# export CEMENTIC_OLLAMA_MODEL=nomic-embed-text
# export CEMENTIC_OLLAMA_EMBEDDING_DIM=768

# Background worker
export CEMENTIC_PIPELINE_WORKER_MAX_WORKERS=1
export CEMENTIC_PIPELINE_WORKER_BATCH_SIZE=8
export CEMENTIC_PIPELINE_WORKER_POLL_INTERVAL=1.0

# Bootstrap
export CEMENTIC_BOOTSTRAP_AUTO_START_INFRA=true
export CEMENTIC_BOOTSTRAP_AUTO_PULL_OLLAMA_MODEL=true
export CEMENTIC_BOOTSTRAP_AUTO_DOWNLOAD_LLAMA_MODEL=true
export CEMENTIC_BOOTSTRAP_POSTGRES_BASE_IMAGE=docker.io/postgres:18.3-bookworm
export CEMENTIC_BOOTSTRAP_POSTGRES_IMAGE=localhost/cementic-postgres-vectorscale:pg18.3-v0.9.0
export CEMENTIC_BOOTSTRAP_PGVECTORSCALE_VERSION=0.9.0
export CEMENTIC_BOOTSTRAP_OLLAMA_IMAGE=docker.io/ollama/ollama:0.20.6
```

The default local setup is:

- Postgres with `pgvector` + `vectorscale`
- `llama.cpp` embeddings inside the cementic worker process for indexing
- a persistent local `llama.cpp` daemon for interactive search, so the model stays loaded

Switch to Ollama only when you specifically want the Ollama backend.

When using Nomic v2 models, cementic automatically applies task prefixes:

- document embeddings: `search_document: ...`
- query embeddings: `search_query: ...`

## Architecture

- `src/cementic/source_watcher.py`
  - watches directories and registers source PDFs
- `src/cementic/extract.py`
  - extracts markdown from PDFs for the extraction stage
- `src/cementic/pipeline_worker.py`
  - builds extraction, chunking, and embedding artifacts for the target revision
- `src/cementic/collections.py`
  - handles collection deletion and revision promotion/history queries
- `src/cementic/profiles.py`
  - resolves immutable extractor, chunk, and embedding profiles
- `src/cementic/revisions.py`
  - manages target, ready, active, retired, and superseded revisions
- `src/cementic/search.py`
  - searches only the active revision for the requested collection(s)
- `src/cementic/storage.py`
  - stores extracted text as compressed artifacts on disk

## Development

```bash
uv pip install -e ".[dev]"
pytest
pytest tests/integration/test_smoke.py -rs
ruff check src/ tests/
mypy src/
```
