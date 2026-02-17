# seman

Semantic search CLI tool for indexing and searching PDF documents using pgvectorscale.

## Features

- **PDF to Markdown**: Converts PDFs to markdown using pymupdf4llm
- **Background Conversion**: Daemon watches directories and auto-converts new PDFs
- **Background Indexing**: Daemon computes embeddings and stores in PostgreSQL
- **Pause/Resume**: Pause processing while keeping file watchers active
- **Semantic Search**: Search with pgvectorscale (DiskANN) and embeddings
- **No Raw SQL**: Pure SQLAlchemy ORM throughout

## Installation

```bash
# Using uv
uv tool install seman

# Using pipx
pipx install seman
```

## Setup

No manual setup is required for normal usage.

When you run `seman convert start` or `seman index start`, seman can automatically:
- Start PostgreSQL and Ollama containers (when using local hosts)
- Wait for services to become healthy
- Pull missing Ollama models

For `llama.cpp`, seman validates `SEMAN_LLAMA_MODEL_PATH` and can optionally auto-download
the model when `SEMAN_BOOTSTRAP_AUTO_DOWNLOAD_LLAMA_MODEL=true`.

## Usage

### Recommended (Background)

```bash
# Start converter + indexer in background
seman start /path/to/pdfs --collection test

# Check unified status
seman status

# Stop both background processes
seman stop

# Delete a collection and all indexed chunks/documents
seman delete-collection test --force
```

### Converting PDFs

```bash
# Start converter daemon watching directories
seman convert start /path/to/pdfs /another/path

# Assign documents to a collection
seman convert start /path/to/work-pdfs --collection work
seman convert start /path/to/personal-pdfs --collection personal

# Check status
seman status

# Control converter
seman convert pause
seman convert resume
seman convert stop
```

### Indexing (Computing Embeddings)

```bash
# Start indexer daemon
seman index start

# Check status
seman status

# Control indexer
seman index pause
seman index resume
seman index stop
```

### Search

```bash
seman search "your query here"
seman search "query" -n 20  # Top 20 results

# Search a specific collection (repeat flag for multiple)
seman search "query" --collection work
seman search "query" --collection work --collection personal
```

### Infrastructure

```bash
seman infra up      # Start containers
seman infra down    # Stop containers
seman infra status  # Check container status
```

## Configuration

Configuration is done via environment variables:

```bash
# Database
export SEMAN_DB_HOST=localhost
export SEMAN_DB_PORT=5432
export SEMAN_DB_NAME=seman
export SEMAN_DB_USER=seman
export SEMAN_DB_PASSWORD=seman

# Ollama
export SEMAN_OLLAMA_HOST=http://localhost:11434
export SEMAN_OLLAMA_MODEL=nomic-embed-text
export SEMAN_OLLAMA_EMBEDDING_DIM=768

# Indexing
export SEMAN_INDEX_CHUNK_SIZE=512
export SEMAN_INDEX_CHUNK_OVERLAP=128
export SEMAN_INDEX_EMBEDDER=llama-cpp  # or 'ollama'

# llama.cpp
export SEMAN_LLAMA_MODEL_PATH=./models/nomic-embed-text-v2-moe.Q8_0.gguf

# Converter daemon
export SEMAN_CONVERTER_PID_FILE=~/.local/share/seman/converter.pid
export SEMAN_CONVERTER_LOG_FILE=~/.local/share/seman/converter.log

# Indexer daemon (embedder config namespace)
export SEMAN_EMBEDDER_PID_FILE=~/.local/share/seman/indexer.pid
export SEMAN_EMBEDDER_LOG_FILE=~/.local/share/seman/indexer.log
export SEMAN_EMBEDDER_MAX_WORKERS=1
export SEMAN_EMBEDDER_BATCH_SIZE=32
export SEMAN_EMBEDDER_POLL_INTERVAL=1.0

# Runtime bootstrap
export SEMAN_BOOTSTRAP_AUTO_START_INFRA=true
export SEMAN_BOOTSTRAP_AUTO_PULL_OLLAMA_MODEL=true
export SEMAN_BOOTSTRAP_AUTO_DOWNLOAD_LLAMA_MODEL=true
export SEMAN_BOOTSTRAP_WAIT_TIMEOUT_SECONDS=90
export SEMAN_BOOTSTRAP_LLAMA_MODEL_URL="https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe-GGUF/resolve/main/nomic-embed-text-v2-moe.Q8_0.gguf"
```

When using Nomic v2 models, seman automatically applies task prefixes for better retrieval quality:
- document embeddings: `search_document: ...`
- query embeddings: `search_query: ...`

## Architecture

- **PostgreSQL + pgvectorscale**: Stores document chunks and embeddings with DiskANN indexing
- **Converter Daemon**: Watches directories, converts PDFs to markdown, creates chunks
- **Indexer Daemon**: Generates embeddings for pending chunks using Ollama or llama.cpp
- **Decoupled Design**: Converter and indexer run independently, communicating via database

## Development

```bash
# Install in development mode
uv pip install -e ".[dev]"

# Run tests
pytest

# Linting
ruff check src/
ruff format src/
mypy src/
```
