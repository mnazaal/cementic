# seman

Semantic search CLI tool for indexing and searching PDF documents using pgvectorscale.

## Features

- **PDF to Markdown**: Converts PDFs to markdown using pymupdf4llm
- **Background Indexing**: Daemon watches directories and auto-indexes new PDFs
- **Pause/Resume**: Pause indexing while keeping file watcher active
- **Semantic Search**: Search with pgvectorscale and Ollama embeddings
- **No Raw SQL**: Pure SQLAlchemy ORM throughout

## Installation

```bash
# Using uv
uv tool install seman

# Using pipx
pipx install seman
```

## Setup

1. Start infrastructure (PostgreSQL + Ollama):
```bash
seman infra up
```

2. Pull the embedding model:
```bash
podman exec seman-ollama ollama pull nomic-embed-text
```

## Usage

### Start Indexing
```bash
# Start daemon watching directories
seman index start /path/to/pdfs /another/path

# Check status
seman index status
```

### Control Indexing
```bash
# Pause processing (keep watching)
seman index pause

# Resume processing
seman index resume

# Stop daemon completely
seman index stop
```

### Search
```bash
seman search "your query here"
seman search "query" -n 20  # Top 20 results
```

### Infrastructure
```bash
seman infra up      # Start containers
seman infra down    # Stop containers
seman infra status  # Check container status
```

## Configuration

Config file location: `~/.config/seman/config.yaml`

Example:
```yaml
database:
  host: localhost
  port: 5432
  name: seman
  user: seman

ollama:
  host: http://localhost:11434
  model: nomic-embed-text
  embedding_dim: 768

indexing:
  chunk_size: 512
  chunk_overlap: 128
  max_workers: 4
```

## Architecture

- **PostgreSQL + pgvector**: Stores document chunks and embeddings
- **Ollama**: Generates embeddings using nomic-embed-text
- **Native Daemon**: Python daemon watches directories and processes PDFs
- **SQLite Queue**: Persistent job queue for pause/resume support

## Development

```bash
# Install in development mode
uv pip install -e ".[dev]"

# Run tests
pytest

# Linting
ruff check src/
mypy src/
```
