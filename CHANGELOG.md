# Changelog

## [0.1.0] — 2026-04-30

### Initial Release

- **Versioned pipeline architecture**: separate extractor, chunk, and embedding profiles so each stage can evolve independently
- **Background workers**: source watcher (registers PDFs) and pipeline worker (extract → chunk → embed) managed by process supervisor
- **Two embedding backends**: llama.cpp (default, in-process + persistent daemon for search) and Ollama
- **Bootstrap auto-setup**: automatically starts PostgreSQL + pgvector + vectorscale container, builds local image if needed, downloads llama.cpp model, pulls Ollama model
- **CLI commands**: `start`, `stop`, `status` (with `--verbose` and `--json`), `search`, `collection list|promote|revisions|remove`
- **Revision lifecycle**: promote ready revisions to active; old revisions auto-pruned; building revision fallback for search when no active exists
- **Status enhancements**: per-stage progress percentages, failure counts, health checks, per-file breakdown
- **Search**: semantic search over active revisions with cosine distance; model mismatch detection across collections
- **Automatic Nomic v2 task prefixes**: `search_document:` / `search_query:` applied when using Nomic models
- **Configuration**: environment-variable-driven via pydantic-settings with sensible defaults
- **Test suite**: 120 unit + integration tests; ruff linting, mypy strict type checking
