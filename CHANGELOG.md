# Changelog

## [Unreleased]

### Fixed

- **Revisions no longer stall after a chunking or extraction change.** The build-completeness
  check counted embeddings by embedding profile alone while counting chunks by chunk profile,
  so a revision that reused an existing embedding model (the common case when only chunking
  changes) could never satisfy the check: it stayed `building` forever and
  `cementic collection promote` reported "no ready revision". Progress percentages could also
  exceed 100% for the same reason.
- **Reverting configuration back to a previously built revision works again.** A retired or
  superseded revision selected as the current target is now resumed as `building`, so it can
  finish and be promoted; previously it was adopted but could never become promotable, leaving
  search silently pinned to the newer revision.
- **A file deleted while it was being extracted stays deleted.** The extraction step no longer
  overwrites `SourceDocument.status`, which could resurrect a removed document into search
  results permanently.
- **`cementic stop` no longer times out on workers that already exited.** Liveness checks now
  treat an unreaped (zombie) process as stopped instead of waiting out the full grace period
  and advising `--force` for a process that had already shut down cleanly.
- **The chunker no longer emits a duplicate trailing chunk** when a chunk ends exactly at the
  end of the text; the final chunk was a pure suffix of its predecessor and was embedded and
  searchable as a near-duplicate. Chunk profile version bumped to `v2`.
- **Embedding-daemon outages no longer mark documents as permanently failed.** A connectivity
  error now releases the batch back to `pending` and retries, instead of stamping every chunk
  in the collection `failed`.
- **A transient database error no longer kills the pipeline worker**; the step is retried after
  a backoff.
- **Files created during the initial directory scan are no longer missed** — the watcher starts
  observing before scanning.
- Worker and supervisor state files are written atomically and guarded against concurrent
  updates, so `cementic status` can no longer read a torn file and report a running worker as
  stopped.
- Liveness checks for background workers now use the recorded process start-token, so a
  recycled PID can no longer block a fresh `cementic start` or show a dead worker as running.
- `force_kill` no longer reports a process it lacked permission to kill as killed.

### Changed

- `cementic start` verifies both workers survived startup and fails with the relevant log path
  instead of reporting success for a worker that exited immediately.
- Commands run against a reachable database with no cementic schema now print
  "nothing indexed yet" instead of a raw SQL error, and `cementic status` exits non-zero on
  failure.
- `cementic search` distinguishes "no matches" from an unknown or not-yet-indexed collection.
- `cementic chunk` reports a one-line error for non-text input instead of a traceback.
- `cementic status` reports a stopped-but-autostartable embedding runtime as a warning rather
  than "unhealthy", matching `--doctor`.
- `cementic collection remove` no longer reports failure when the collection was deleted but
  artifact cleanup failed.

### Removed

- Unused `pgvector` Python dependency (the extension is used through SQL, never imported).
- Never-populated `page_start` / `page_end` fields on chunks, and the write-only
  `SourceDocument.error_message` column.

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
