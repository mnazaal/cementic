# Changelog

## [Unreleased]

### Added

- **`cementic collection reindex COLLECTION`** — rebuilds the active revision's ANN index
  in place, switching HNSW <-> DiskANN without re-embedding. `-f`/`--force` rebuilds even
  when `index.method` is unchanged, to pick up new `hnsw_m` / `hnsw_ef_construction`,
  which are fixed into the index at build time.
- **`index.build_memory`** (default `2GB`) — `maintenance_work_mem` for ANN index builds
  only. PostgreSQL's 64MB default spills the HNSW graph to disk: 1454s against 345s for
  100k 768-dimensional vectors.
- **`index.hnsw_iterative_scan`** (default `relaxed_order`) — search filters candidates
  during the index scan, so without it a collection holding a small share of a shared
  vector table could come back short or empty. Needs pgvector 0.8+; ignored on older
  servers.
- **`extraction.backends`** — per-file-type extractor choice, e.g. `pdf = "pymupdf4llm"`.
  Both the file type and the extractor name are validated against the registry at config
  load, with distinct messages for an unknown name and a wrong file type.
- **`source_watcher.ignore_directories`** — 16 default names (`.git`, `node_modules`,
  `build`, `dist`, `venv`, `target`, ...) never descended into. Replaces the defaults
  rather than adding to them; set `[]` to index everything.
- **`scripts/check.sh`** — runs all five CI gates in one command and reports a missing
  PostgreSQL as SKIPPED rather than passed.
- **`scripts/measure_chunk_context_fit.py`** — measures the tiktoken-to-model-token ratio
  over full-size chunks against the running daemon. Run it before changing
  `pipeline.chunk_size` or `llama_cpp.n_ctx`.

### Fixed

- **The embedding daemon no longer hangs for three minutes on any runtime-config
  change.** Restarting a mismatched daemon took the daemon lock and then called a
  stop that took the same lock again; `flock` is not reentrant even within one
  process, so `search`, `embedding start` and the pipeline worker each waited out
  the full 180s timeout and then failed with "another cementic process holds ..."
  — naming themselves.
- **`pipeline.chunk_size` values that merely cost a tokenize round trip are
  allowed again.** The new config check refused everything above 345 at the
  default `n_ctx = 512`, including 352 — a value measured to truncate nothing and
  documented as the way to keep an index already built at it — so every command,
  `config show` included, failed on such a config. Config load now refuses only
  pairings that *must* truncate; `cementic status --doctor` reports the
  round-trip cost as a warning, and truncation itself is still caught exactly,
  per chunk, at embed time.
- **Upgrading a database that already had two active revisions for one
  collection no longer bricks `cementic start`.** The new one-active-per-
  collection index could not be built on exactly the databases that hit the race
  it exists to prevent: both workers died at startup with an IntegrityError no
  cementic command could repair. The extra actives are now retired (newest kept,
  which is what every reader already resolved to) before the index is created.
- **The two workers no longer race each other creating that index.** `CREATE
  UNIQUE INDEX IF NOT EXISTS` checks the catalog before taking its lock, so the
  loser got a duplicate-key error on the first start after every upgrade.
- **Losing a concurrent `collection promote` reads as a sentence, not a psycopg2
  dump.** Two promotes of different ready revisions each lock only their own row;
  the loser now reports "another promote activated a revision first" and changes
  nothing.
- **`cementic status --json` reports `current_file: null` when a worker is idle**,
  not the literal string `"None"`, which every consumer testing truthiness read
  as a busy worker.
- **`... | cementic chunk "$UNSET"` reads the pipe again.** Rejecting an empty
  PATH outright broke this ordinary shell idiom with `Is a directory: '.'`,
  naming a path the user never typed. An empty PATH at a *terminal* is still a
  clear error rather than a hang.
- **A bad `CEMENTIC_CONFIG` is a message, not a traceback.** `~nosuchuser/...`
  made `Path.expanduser()` raise, so the guard whose whole job is to explain bad
  config paths crashed from every command.
- **Config errors caused by the environment name the variable.** `CEMENTIC_DB_URL`
  was never attributed (the section prefix was applied twice), and section-level
  validator errors blamed the config file even when the value came from the
  environment.
- **`cementic status` exits 1 when its health probe crashes**, and says the
  database and embedding rows are unknown instead of omitting them — a summary
  two rows short read as a complete report of a healthy system, and
  `cementic status 2>/dev/null && deploy` proceeded.
- **`cementic stop` keeps the supervisor record when a force-kill fails.** It
  holds the collection and watched directories, and was discarded while the
  workers were still indexing. A worker whose `/proc` entry cannot be read
  (hidepid, another uid) is also no longer cleared as stale while it keeps
  running.
- **Refusals print to stderr.** `promote`, `start`, `stop` and human-mode
  `search` printed theirs to stdout, so `... 2>errors.log || cat errors.log`
  showed nothing and a redirected result list ended with an error sentence in it.
- **`cementic search` distinguishes an unknown collection from one that is not
  indexed yet**, and no longer leads with "no results" when the name is the
  problem. Piped previews are no longer cut at 80 columns, which broke substring
  greps over redirected output.
- **`cementic embed` names the real stdin line** when a record cannot be
  embedded; blank lines used to shift the number.
- **`cementic status --doctor` says when it is ignoring `-c` and `-v`** instead of
  accepting both and using neither.
- **Partial model downloads are reaped.** A download killed outright leaked a
  multi-hundred-MB temp file per attempt, since each attempt used a new PID.
- **Explaining a failed embedding no longer holds a write transaction open**, and
  a daemon that goes away mid-explanation no longer converts a terminal `failed`
  stamp into a released claim the next poll re-fails.

- **The model auto-download is race-safe and reports failures in one line.** Two
  workers bootstrapping at once shared a single fixed temp file — interleaved writes
  could corrupt it, and with the SHA pin opted out the corrupt file was installed
  silently. The download now runs under a file lock (the loser reuses the winner's
  file), uses a per-process temp name whose leftovers are reaped on the next attempt,
  and a network failure raises one line instead of a raw requests traceback in the
  background log.
- **Config validation now bounds the numerics and enforces the chunk-size invariant for
  *your* values.** `n_ctx = 0` used to silently disable the token-budget guard (the fix
  for silent truncation), a non-positive `pipeline_worker.poll_interval` hot-spun the
  worker, and out-of-range ports surfaced only as connection errors. A
  `chunk_size`/`n_ctx` pairing that *must* embed truncated is refused at load — it was
  previously only pinned for the shipped defaults. (See above for the narrower bound
  this settled on.)
- **Config errors name the environment variable when the environment caused them.**
  `CEMENTIC_DB_PORT=bad` used to render `<config file>: [database] port: ...`, sending
  you to edit a file whose value was never read.
- **A chunk over the model's context window now says so.** The worker stamped the
  generic "Failed to generate embedding"; `status --verbose` now shows the token count,
  the window, and the `chunk_size` fix. A too-long *query* is no longer advised to
  "lower pipeline.chunk_size".
- **`status --doctor` reports a broken config instead of dying on it.** It exited with
  a one-line error and no report — less output than plain `status` from the one command
  that exists to diagnose the setup. A failure while inspecting extensions is also no
  longer misreported as an unreachable database.
- **`embedding stop` takes the daemon lock**, so it can no longer race a concurrent
  autostart into an orphaned daemon; and a freshly started daemon serving the wrong
  model fails fast with the model names instead of burning the whole 120 s startup
  budget to report "did not become ready".
- **`collection promote` with no ready revision exits 1.** It was the one
  promoted-nothing outcome that exited 0 (empty, incomplete and blocked all exit 1), so
  `promote && search` proceeded as if a revision had been published. The README documents
  the full exit-code convention.
- **Errors print to stderr, and output is no longer hard-wrapped at 80 columns when
  piped.** Config and database errors used to land on stdout, so `search --json | jq`
  choked on `config error: ...` as if it were data; and off a TTY every long path or hint
  was split mid-word by rich's 80-column fallback, breaking `grep` over the output.
- **Chunks are no longer embedded truncated.** `pipeline.chunk_size` counts tiktoken
  tokens while `llama_cpp.n_ctx` counts the embedding model's own, and for the default
  model one is up to 1.33 of the other — so at 512 against 512, **93% of full-size chunks
  overflowed the context window and the server silently dropped the overflow**, measured
  on a real corpus. The stored chunk text and the vector indexing it disagreed, and the
  tail of each chunk was unsearchable. `chunk_size` now defaults to 320, and the embedding
  client counts with the served model's own tokenizer and reports an over-budget chunk as
  a failure instead of letting it truncate. **This changes the chunk profile: existing
  collections re-index once on the next `cementic start`.**
- **`cementic stop` no longer reports success while workers keep running.** A liveness
  check treated `EPERM` — which proves a process exists — as "not running", so workers
  started under another user were reported stopped, their state cleared, while they went
  on indexing.
- **A typo'd collection name is no longer swallowed.** `search -c work -c persnal`
  reported nothing about the typo as long as `work` matched, at exit 0, in both output
  modes — half the query silently dropped. Naming a collection that does not exist is now
  an error in `search`, `status -c`, `collection revisions`, `collection promote` and
  `collection reindex` alike; `collection remove` still succeeds on a missing name so it
  stays safe to run twice.
- **A failed re-extraction no longer keeps serving the old text.** The stale-chunk purge
  only ran when re-extraction *succeeded*, so replacing an indexed file with a corrupt or
  unreadable one left the previous version searchable under the current path indefinitely.
- **`cementic collection reindex --force` can no longer leave a collection unindexed.**
  The index drop was committed in its own transaction, so any failure of the rebuild —
  interrupt, timeout, disk full — dropped the ANN index permanently and silently, since
  search still works by sequential scan.
- **`cementic search` no longer re-runs a schema migration on every query.** The
  vector-table migration was executed and then rolled back with the session, so on a
  database written by an older cementic every search repeated a full-table backfill and
  discarded it.
- **A config file that cannot be parsed is now an error.** It was discarded with one
  stderr line and every setting in it silently replaced by built-in defaults — a different
  database, a different chunk size. A non-UTF-8 config file, and a `CEMENTIC_*` variable
  for a list- or table-valued setting given in non-JSON form, both reached the user as
  tracebacks; they are one-line errors now.
- **`cementic search ""` is refused** instead of returning a confidently ranked top-k of
  the nearest neighbours of nothing.
- **`cementic embed` no longer invents vectors.** A record with no `content` was embedded
  as the empty string and a `null` one as the literal text `"None"`, each emitted as a
  normal-looking embedding at exit 0. Empty and whitespace-only strings are refused the
  same way — the pipeline worker never embeds such chunks either.
- **`cementic collection remove` reports artifacts it could not delete** rather than
  printing `status: deleted` with the files still on disk and the rows naming them gone.
- **Files the watcher refuses are visible in `cementic status`.** Symlinks, unreadable
  paths and oversized files never become documents, so they were absent from every
  progress percentage: a collection that dropped a directory of symlinks still reported
  100% and promoted cleanly. The count is now shown alongside worker errors, and the paths
  with their reasons under `--verbose` and in `--json`.
- **`cementic collection list` shows collections that own revisions but no documents.**
- Whitespace-only chunks are no longer stored and embedded, and `chunk_index` is
  contiguous again when a chunk boundary lands mid-character at small chunk sizes.

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
  Such collections were reported as "not found". (`collection list` no longer hides
  them either; see below.)
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
  searchable as a near-duplicate. (Chunk profile version was bumped for this; see the
  chunk-size change under Changed for the current `v3`.)
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

- **The HNSW index is created up front, on the still-empty vector table**, and maintained
  incrementally by every insert. This removes the unresumable build stall at the end of a
  revision, makes progress per-batch resumable, and leaves the index searchable during
  ingestion — at ~2.9x total indexing time, which is ~3.5ms of index maintenance per chunk
  against tens of milliseconds to embed one. DiskANN and resumed builds over existing rows
  keep the bulk build at the ready transition.
- Default `pipeline.chunk_size` 512 -> 320 and `chunk_overlap` 128 -> 80; chunk profile
  version bumped to `v3`, which also covers three earlier changes to chunk output that
  shipped without a bump.
- Every configuration key is now documented in the README, including the 21 that were live
  but unlisted — notably `source_watcher.ignore_directories`, which silently skips 16
  directory names and replaces rather than extends its defaults.
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

- `extractor_names()`, which had no callers, and `CURRENT_CONTENT_SQL`, whose freshness
  join search no longer uses (it now lives beside the test fixtures whose shape it
  describes).
- An unused required `config` argument on `prune_collection_history` and
  `promote_revision`, which six call sites were constructing a `Config` to satisfy.
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
