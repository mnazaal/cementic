# seman

Semantic search CLI tool for indexing and searching PDF documents using pgvectorscale.

## Features

- **PDF to Markdown**: Converts PDFs to markdown using pymupdf4llm
- **Background Conversion**: Daemon watches directories and auto-converts new PDFs
- **Background Indexing**: Daemon computes embeddings and stores in PostgreSQL
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

When you run `seman start`, seman can automatically:
- Start PostgreSQL and Ollama containers (when using local hosts)
- Wait for services to become healthy
- Pull missing Ollama models

For `llama.cpp`, seman validates `SEMAN_LLAMA_MODEL_PATH` and can optionally auto-download
the model when `SEMAN_BOOTSTRAP_AUTO_DOWNLOAD_LLAMA_MODEL=true`.

## Usage

### Commands

```bash
# Start converter + indexer in background
seman start /path/to/pdfs --collection test

# Detailed converter/indexer/queue status
seman status

# Search all collections
seman search "your query"

# Search specific collections
seman search "your query" --collection work --collection personal

# Stop both background processes and infrastructure
seman stop

# Or stop only processes, keep containers running
seman stop --no-infra

# Delete a collection and all indexed chunks/documents
seman delete-collection test --force
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
export SEMAN_EMBEDDER_PROCESSING_STALE_SECONDS=30

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
