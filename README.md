# cementic

Local semantic search for PDF, Markdown, and text collections.

`cementic` watches directories, extracts and chunks documents, embeds them with a
local `llama.cpp` server, and searches the resulting collection. It combines
semantic and exact-word search, so both concepts and rare names or acronyms are
findable.

## Quick start

You need Python 3.12+, `uv`, a `llama-server` binary on `PATH`, and Docker or
Podman. The local database uses PostgreSQL with pgvector and pgvectorscale.

```bash
git clone https://github.com/mnazaal/cementic.git
cd cementic
uv sync

# Generate and start a persistent local Postgres service.
uv run cementic init postgres ./cementic-postgres
cd ./cementic-postgres
docker compose up -d             # or: podman compose up -d
cd ..

# Check Postgres and llama-server, then index a directory.
uv run cementic doctor
uv run cementic start ~/research-papers -c research

# Search while indexing continues in the background.
uv run cementic search "vector database design" -c research
```

The first command that needs embeddings downloads the default model when it is
missing. It is about 2 GB. Start the embedding service in advance if you prefer
not to wait for the first search:

```bash
uv run cementic embedding start
```

## Daily workflow

```bash
# Inspect indexing progress and worker health.
uv run cementic status

# Promote a fully built replacement revision.
uv run cementic collection promote research

# Stop the watcher and pipeline worker. Postgres remains running.
uv run cementic stop
```

A collection has one search-visible **active revision**. Changing the extractor,
chunking, or model builds a replacement revision in the background. Search keeps
using the active revision until you promote the ready replacement.

`cementic start` runs one background session at a time. Pass multiple directories
to one command to index them into one collection. Stop the current session before
indexing another collection.

## What it supports

- PDF, Markdown, and plain-text indexing; custom extractors can call any command
  that writes text to stdout.
- Local embeddings through an OpenAI-compatible `llama.cpp` server.
- Versioned extraction, chunking, and embedding artifacts.
- Hybrid vector and PostgreSQL full-text search.
- HNSW and DiskANN vector indexes.

## Commands

```text
cementic start DIRECTORY... -c COLLECTION   # watch and index
cementic search QUERY -c COLLECTION         # search active revision
cementic status                              # progress and health
cementic doctor                              # diagnose local setup
cementic collection --help                   # revisions, promote, reindex, remove
cementic embedding --help                    # local embedding runtime
cementic config --help                       # config location and effective values
```

Run `cementic COMMAND --help` for flags and examples.

## Guides

- [Setup](docs/setup.md): installation, local Postgres, and prerequisites.
- [Configuration](docs/configuration.md): config files, models, extraction,
  search, and indexes.
- [Operations](docs/operations.md): revisions, diagnostics, services, and
  limitations.
- [Contributing](CONTRIBUTING.md): development setup, checks, and runtime-change guidance.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
