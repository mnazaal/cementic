# Changelog

## [0.1.0b1] — 2026-07-01

### Beta Release Prep

- Add GitHub-tag install target for `pipx` / `uv tool` beta users
- Add `cementic init postgres DIR` to generate packaged Docker/Podman Postgres setup files
- Add `cementic status --doctor` read-only runtime diagnostics with JSON support
- Enable required Postgres extensions during normal schema initialization with actionable failures
- Document Linux-first beta setup, persistent Postgres service setup, and model auto-download behavior

## [Unreleased pre-beta] — 2026-04-30

### Initial Development

Never published as a release; superseded by 0.1.0b1 above.

- **Versioned pipeline architecture**: separate extractor, chunk, and embedding profiles so each stage can evolve independently
- **Background workers**: source watcher (registers documents of any supported type) and pipeline worker (extract → chunk → embed) managed by a process supervisor
- **Local embedding backend**: a warm `llama.cpp` server (OpenAI-compatible) that stays loaded across indexing and interactive search
- **Document extractors**: `pymupdf4llm` for PDF and a plaintext extractor for `.txt` / `.md` / `.markdown`, dispatched through a content-type registry (one entry per type)
- **External Postgres**: cementic connects to a Postgres with `pgvector` + `vectorscale` (provisioned via the bundled `compose.yml` or pointed at by `CEMENTIC_DB_URL`); it does not start, build, or stop containers
- **Bootstrap auto-setup**: verifies Postgres is reachable and auto-downloads the `llama.cpp` model file when missing (confined to the cementic data directory)
- **CLI commands**: `start`, `stop`, `status` (with `--verbose` and `--json`), `search`, `extract`/`chunk`/`embed` stdin/stdout filters, `collection list|promote|revisions|remove`, `embedding start|stop|status`, `config init|path|show`
- **Revision lifecycle**: promote ready revisions to active; old revisions auto-pruned; building-revision fallback for search when no active revision exists yet
- **Selectable ANN index**: `index.method` chooses HNSW (pgvector, in-memory) or DiskANN (pgvectorscale, disk-resident); switching rebuilds the index, never re-embeds
- **Status enhancements**: per-stage progress percentages, failure counts, health checks, per-file breakdown
- **Search**: semantic search over active revisions with cosine distance; model-mismatch detection across collections
- **Automatic Nomic v2 task prefixes**: `search_document:` / `search_query:` applied when using Nomic models
- **Configuration**: TOML config file and/or `CEMENTIC_*` environment variables via pydantic-settings, with sensible local defaults
- **Quality gates**: unit + PostgreSQL integration test suites; ruff linting and mypy strict type checking
