# Changelog

## [Unreleased]

### Fixed

- **`cementic stop` no longer deadlocks a worker against its own SIGTERM.** The signal
  handler wrote to the state file, whose lock the main thread was often already holding —
  most likely during a bulk initial scan. The process hung, `cementic stop` timed out and
  advised `--force`, and because the deadlocked handler *was* the SIGTERM handler, only
  SIGKILL could recover it. Handlers now only set the shutdown flag; cleanup runs on the
  main thread. `cementic stop` during a large directory scan is also responsive now
  instead of waiting out the whole walk.
- **A build that produces no vectors no longer wedges in `building` forever.** An empty
  watch directory, or one where every document failed to extract, reached the ANN index
  step with no vector table to index; the resulting error was swallowed by the worker's
  retry loop, so the revision never became promotable and the only evidence was a
  repeating traceback in a log file.
- **A failed embedding write-back no longer strands its batch.** Rows claimed as
  `processing` were never re-picked within a run — unlike the extract and chunk steps —
  so one failure between claiming and writing back meant the revision could never reach
  `ready`, with the worker still reporting itself healthy.
- **Reverting configuration to the currently active revision no longer leaves a phantom
  build.** The abandoned revision stayed `building` forever, was displayed as in-progress
  by `cementic status`, and pinned its artifacts on disk permanently.
- **`cementic status` no longer reports queued work as failures**, and no longer shows
  100% extracted for files that changed on disk and still owe a re-extraction. Status and
  the pipeline worker now share one definition of what counts, so they cannot drift.
- **Worker failures are visible in `cementic status`.** A worker looping on a permanent
  failure was previously indistinguishable from a healthy idle one; the reason now
  appears as `last error` and in `--json`.
- **`cementic status` and `cementic search --json` exit non-zero when they cannot
  answer.** `status` printed "database not reachable" and exited 0; `search --json`
  returned empty output and exited 0 for a collection that was never indexed.
- **A partly-failed `cementic start` cleans up after itself.** It reported failure while
  leaving the surviving worker running and the supervisor record written, so the next
  `cementic start` refused with "already running".
- **Worker startup failures appear in the log `cementic start` names**, rather than in a
  different file the user was never told about.
- **Files deleted while cementic was not running now drop out of search.** Deletion was
  only noticed through a live filesystem event, so such files kept matching queries with
  a path that no longer existed.
- **`cementic collection remove` can remove a collection that has revisions but no
  documents** — what `cementic start` on a directory with no supported files creates.
  Such collections were invisible to `collection list` and reported as "not found".
- Watched directories are resolved before being handed to the background workers, so a
  relative path cannot mean something different in the worker's working directory.
- A corrupt, unreadable, or non-object supervisor/worker state file no longer aborts
  `start`/`status`/`stop` with a traceback.

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
  "nothing indexed yet" instead of a raw SQL error.
- `cementic search` distinguishes "no matches" from an unknown or not-yet-indexed
  collection, in both human-readable and `--json` output.
- `cementic chunk` reports a one-line error for non-text input instead of a traceback.
- `cementic status` reports a stopped-but-autostartable embedding runtime as a warning rather
  than "unhealthy", matching `--doctor`.
- `cementic collection remove` no longer reports failure when the collection was deleted but
  artifact cleanup failed.

### Removed

- Unused `pgvector` Python dependency (the extension is used through SQL, never imported).
- Never-populated `page_start` / `page_end` fields on chunks, and the write-only
  `SourceDocument.error_message` column.
- A `python -m cementic.pipeline_worker` entry point that duplicated `cementic.runner`
  while skipping bootstrap and collection-name validation.

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
