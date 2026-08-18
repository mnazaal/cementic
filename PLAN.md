# cementic — architecture & design

<!-- session-handoff:begin (2026-08-15) -->
## Where the work stands

The **fourth code review is done, fixed, and merged** — seven commits, `main` at
`14f9f2a`. Every finding, the commit that closed it, and what stays deliberately
open are in `notes/code-review-2026-08-14.html`; that note is the record and this
block does not repeat it.

The live corpus is **indexed and searchable for the first time**: collection
`test` (watching `~/bibs/papers`), 5 documents, 249/249 chunks embedded, 0
failed, revision 2 `default-03033c91-llama-cpp-bea283a3` active. `collection
promote` followed by `search` was exercised end to end against the live
database, which closes the previous handoff's one "not carried forward" item.

**Entry point — one decision, already measured, nothing blocking it.** The
active index was built at `chunk_size = 352`; the shipped default is now **320**.
Both are correct (0 chunks over the model's 512-token window, measured on this
corpus), but only 320 keeps the token-budget guard's cheap path — at 352 the
task prefix pushes every chunk 3 tokens past the skip threshold and each one
pays an extra tokenize round trip (the guard is
`RemoteEmbeddingClient.over_budget_tokens` in `embedding_runtime.py`).
Consequence: the next `cementic start ~/bibs/papers -c test` mints a new
revision and re-embeds all 274 chunks, roughly 7 minutes, with the current
revision serving throughout; then `cementic collection promote test`. Either let
that happen, or pin `chunk_size = 352` in a config file to keep the present
index. Nothing else was pending at handoff time. *(Correction 2026-08-18: the
fifth review has since run and its findings are open — see "Plan of record"
below.)*

**Branch:** `claude/session-handoff-2026-08-15`, branched from `main` at
`14f9f2a` and not yet merged — it carries this block, a correction to "Scale
context" below, and a staleness sweep of the standing docs. Merge it and no
`claude/*` branches remain. Nothing is running in the background.
*(Correction 2026-08-18: merged — `main` carries it.)*

**Verification:** `./scripts/check.sh`. At `14f9f2a`: 893 tests passing, ruff and
mypy clean.

### Environment facts that cost time to rediscover

Carried forward and still true: the venv is an **editable install of this working
tree** (`cementic` runs whatever branch is checked out); the user **merges
branches into `main` between turns** (re-check the branch before committing — it
happened twice more this session); integration tests need a separate `<name>_test`
database; `podman` and `psql` are unusable from the sandbox, so inspect Postgres
through SQLAlchemy. New this session:

- **The sandbox does not mount `~/bibs`.** Twice I read "I cannot see it" as "it
  does not exist" and said so — once concluding the user's whole PDF corpus had
  been deleted. Never infer a path's absence from inside the sandbox.
- **Timings measured in the sandbox are ~4x slower than the host.** In-sandbox
  embedding measured 5.9 s/chunk; the host does 1.4 s/chunk. A "the shipped
  defaults are incompatible" finding was built on the sandbox number and had to
  be retracted. Timings for user-facing advice must come from host-side evidence
  (e.g. polling `chunk_embeddings` while the real worker runs).
- **SIGKILL on a wedged llama.cpp daemon leaves its listening socket bound** —
  the port accepts connections and hangs forever, with no owning process, until
  the zombie is reaped. SIGTERM releases it cleanly. Use SIGTERM; if a port is
  stuck, `llama_cpp.daemon_port` is in neither the embedding-profile nor the
  runtime fingerprint, so moving it costs no re-embed.

### Two predicted failures actually occurred, live

Both are already in the review as accepted/known, but they are no longer
theoretical and are worth weighting accordingly:

- An **orphaned embedding daemon** ran 21 hours holding the port while cementic
  had lost its pid record, so `embedding stop` reported "already stopped" and
  nothing could reclaim it. It was wedged, and it is what stalled the re-index.
- **`check_health` called that wedged daemon "healthy"**, because it only asks
  whether `/v1/models` lists the model. `cementic status` said `embedding
  healthy` while every request hung.

### Numbers measured this session

- **50 chunks/PDF** at `chunk_size = 352` (the old figure of 34 was at 512), so
  20k PDFs is ~**1M chunks**. "Scale context" below has been corrected to match.
- **Embedding: 1.4 s/chunk** host-side, roughly linear in batch size. 1M chunks
  is therefore ~387 hours of continuous embedding — the real ceiling at 20k PDFs
  is indexing throughput, not query latency.
- **Search: 214 ms warm end to end, of which 197 ms is embedding the query** — a
  constant, independent of corpus size. Everything else is 17 ms; a bare KNN over
  10k vectors is 5.6 ms median / 9.3 ms p90. Search at 20k PDFs should stay
  ~0.2 s. What users perceive as a slow search is the daemon cold-starting.
- **RAM is the binding constraint before latency is:** 15 GB total, ~3 GB
  available, 6.2 GB swap already in use. A 1M-vector HNSW index needs ~3 GB
  resident, so switch `index.method` to `diskann` well before that (vectorscale
  0.9.0 and the `diskann` access method are installed and verified present).

### Dead ends — do not repeat

- **The ANN scaling benchmark was abandoned, and its design was the reason.** It
  built the HNSW index at 10k and then inserted 90k more rows *into the indexed
  table* (every row paying graph maintenance), while generating 768 Gaussians per
  row in pure Python — Postgres sat `idle in transaction` waiting on the client.
  A rewrite must bulk-load first, index once at the end, and generate vectors
  with numpy. Only the 10k point survived; it is recorded above. The script was
  discarded, not promoted.
- **Retracted, do not act on:** the `batch_size`/`llama_embed_timeout_seconds`
  defaults are fine (see the sandbox-timing note above), and the PDF corpus was
  never missing.
- A symbol inventory over the whole package found exactly **one** truly dead
  symbol, since removed. The package is clean; a future dead-code pass should
  expect a near-empty result rather than assume the tool is broken.

**Not carried forward:** nothing from the scratchpad was promoted — the fixes,
tests, review note and README changes are all committed, and the remaining
artifacts were run logs and throwaway probes whose cases are now covered by
regression tests.
<!-- session-handoff:end -->

Design rationale and roadmap for cementic: a CLI that watches directories of
documents, builds a versioned **extract → chunk → embed** pipeline in Postgres
(pgvector / vectorscale), and serves semantic search over the active revision of
each collection. User-facing docs live in `README.md`; this document is for
contributors.

## Plan of record — fifth-review fixes (2026-08-18)

**Executed same day** — twelve fix/refactor/test commits on
`claude/review-fixes-2026-08-17`, in the batch order below, every batch green
under all five `./scripts/check.sh` gates (PG included). The note's resolution
banner maps finding → commit; the deliberate exceptions are recorded there and
under "Deliberately not done". Kept as the record of the decisions.

Scope: close the fifth review, [`notes/code-review-2026-08-17.html`](notes/code-review-2026-08-17.html).
All file:line evidence lives in the note; this section holds only execution
order, the decisions, and the exit criteria. Mechanics: one branch
(`claude/review-fixes-2026-08-17`), conventional commits, one commit per
finding-cluster with its regression test, each commit green under
`./scripts/check.sh` — all five gates, PG included (that lesson is paid for).
When done, the note gets a resolution banner mapping finding → commit, same
shape as the 2026-08-14 note.

### Order of attack

`§` references are the note's sections.

1. **Finish the five half-landed fixes (§1).** Each currently contradicts a
   commit message or docstring that claims it done, so they go first:
   - `status --json` on an unknown collection: validate before the JSON early
     return; error to stderr, non-zero exit. Correct the README/CHANGELOG
     "exits non-zero" claims in the same commit.
   - The initial scan records skips: symlinks, walk errors, and registration
     failures go through `record_skipped` exactly as live inotify events do.
   - `stop`'s kill loop: `PermissionError` means alive-but-not-ours (mirror
     `force_kill`'s reasoning); never clear supervisor/worker state while such
     a pid remains; report it and exit non-zero.
   - `CEMENTIC_CONFIG`: `expanduser` in `resolve_config_path`; `config path`
     calls the existing `config_path_error` guard so an unusable value is
     reported instead of silently masked by the fallback.
   - `promote` prints the artifact-removal failure list (as `remove` already
     does); `embed` rejects empty/whitespace `content`.
2. **Worker/DB correctness (§2)**, in severity order:
   - §2.1 `collection remove` vs a running worker. **Decision — recommended
     mechanism:** the worker re-validates its cached revision id once per poll
     cycle (one cheap SELECT) and exits cleanly when it is gone; `collection
     remove` additionally warns when workers for that collection are running.
     Rejected alternative: refusing removal while workers run — heavier UX,
     and the delete itself is already cascade-safe; the defect is only the
     zombie worker and the watcher resurrecting the collection.
   - §2.7 search's `-c` fallback ranks revisions via `_searchable_revisions`
     so there is one source of revision choice. Twice-derived carry; the
     regression test pins building-vs-ready.
   - §2.5 `IS DISTINCT FROM` semantics for `source_content_hash` in the claim
     query, and write the hash on the failure path too, so a NULL row cannot
     wedge a revision in `building`.
   - §2.2 `SET LOCAL maintenance_work_mem`; delete the false "connection is
     discarded" comment.
   - §2.4 reset `skipped_files` on watcher restart; §2.6 drop whitespace-only
     chunks before `chunk_index` assignment so indexes stay contiguous and
     `total_chunks` honest; §2.3 a post-commit cleanup failure after a durable
     promote is a warning, not "promote failed" exit 1.
   - §2.8 partial unique index `ON pipeline_revisions (collection) WHERE
     status = 'active'`, applied through the ensure-schema path (same
     mechanism as `ensure_vector_table_schema` — `create_all` won't retrofit
     it), plus promote re-reading status under `FOR UPDATE`.
   - §2.9's smaller items ride along wherever their file is already open;
     the directory-move blindness (unverified) gets a repro test first and a
     fix only if it reproduces.
3. **Error-stream and wrapping discipline (§4.2–4.3).** Mechanical, wide
   blast radius, kept in its own commits: all human error text to stderr via
   an `err_console`; `soft_wrap=True` so off-TTY output stops hard-wrapping
   paths at 80 columns. Tests pipe the output and assert stream and absence
   of mid-path wraps. This is what unblocks `--json | jq` composability.
4. **Exit-code normalization (§4.1).** Adopt the convention most commands
   already follow — 0 ok, 1 operation failed, 2 usage error / unknown name —
   and move the stragglers to it (`collection remove <unknown>`, `promote`
   with no ready revision). Document the table in README.
5. **Config/runtime hardening (§3.1–3.3, §3.5–3.6).** Bounds on numerics
   (`n_ctx >= 1` closes the guard-disable hole; positive intervals and
   timeouts; port ranges), the chunk_size↔n_ctx invariant enforced at config
   validation against the *configured* values (today it is only tested at the
   shipped defaults), env-vs-file attribution in config error messages,
   `over_budget_reason` wired into worker failure rows and the query-side
   message (which currently blames `pipeline.chunk_size` for a long query),
   `runner.py` parity with the CLI's error handling (`RuntimeError`,
   `SettingsError`), `embedding stop` under the daemon lock, autostart
   failing fast on a definitive model mismatch instead of waiting 120 s, and
   doctor's unreachable/false branches.
6. **Bootstrap download (§3.4).** Third-time carry — **decision: fix now.**
   Unique temp name, download performed under the daemon file lock (taken
   before the download, not after), `requests` exceptions wrapped into the
   normal error format. If overruled, the deferral gets written into
   "Deliberately not done" with reasons, so it stops being re-derived.
7. **CLI paper cuts (§4.4)**, batched by file: binary-stdin decode error in
   `embed`, `chunk ""` falsy-check reading stdin, embed JSONL error line
   attribution, `strict=True` on the zips, the wrong-noun messages
   (`start <file>`, `extract <dir>`), `status -v` file-listing errors
   surfaced, `check_health` crash no longer silently deleting the health
   section, `current_file` included in `--json`.
8. **Trimming (§5).** Last, so cleanup diffs never mix with behavior fixes:
   the 17-key status dict ×2, engine/session boilerplate ×7, the
   `validate_collection_name` wrapper ×9, the ~70 worker/watcher duplicated
   lines, the bucketing loop ×2, the dead `session.commit()`, TOML parsed
   once per `get_config()`, and the stale docstrings/comments — except those
   an earlier batch already touches, which get fixed there.
9. **Test and doc debt (§6–§7).** Regression tests for the three untested
   fourth-review fixes (search-migration commit, `reindex --force`
   failure-atomicity, embed input validation — the last largely produced by
   batch 1), plus whatever README/CHANGELOG claims batches 1 and 4 have not
   already corrected.

### Out of scope here, tracked elsewhere

- The chunk_size 320-vs-352 re-embed decision — handoff block above.
- Environment, not code: the orphaned daemon observed on port 11555 (SIGTERM
  it, per the handoff block's socket note), and the indexed corpus directory
  no longer existing on disk (every current search result carries a dead
  `source_path` — re-point or remove the collection).
- The deferred-features list at the bottom of this document.

### Exit criteria

- Every §1–§7 finding is either fixed with a regression test or explicitly
  moved to "Deliberately not done" with a reason.
- Resolution banner in the 2026-08-17 note, finding → commit.
- `./scripts/check.sh` green, PG gate included.
- No unmerged `claude/*` branch left behind.

Rough sizing: batches 1–4 are one focused session; 5–9 one to two more.

## Design principles

- **Functional core, imperative shell.** Pure functions (extraction, chunking,
  SQL builders, resolvers, fingerprints) are kept separate from the
  side-effecting shells (database, embedding daemon, file watcher, CLI).
- **Select by data, not by `if` chains.** Each pluggable thing — embedding
  provider, extractor, ANN index method — owns everything about itself and is
  chosen through a registry keyed by data. Adding one is a single registry
  entry; call sites never branch on a type/provider name. (Open for extension,
  closed for modification.)
- **Mechanism vs policy.** Versioning, the warm daemon, and artifact storage are
  mechanism; *which* model / extractor / index to use is policy expressed in
  config.
- **Unix composability.** The pipeline stages are also stdin/stdout filters
  (`extract | chunk | embed`), so a single document can be run end-to-end with no
  database.

## Scale context

The target is a large personal corpus (~40k papers and textbooks). At the
measured **50 chunks per paper** that is ~2M vectors. The machinery is justified
rather than over-engineered:

- "Indexed" and "fast" are not in tension — indexing is the amortized one-time
  cost; an ANN index keeps *queries* sub-second over millions of vectors.
  Measured 2026-08-15: a warm search is **214 ms end to end, 197 ms of which is
  embedding the query string** — a constant that does not grow with the corpus.
  A bare KNN over 10k vectors is 5.6 ms. Query latency is not the scaling risk.
- The scaling risks are the other two axes, and both bind earlier. **Embedding
  throughput** measured 1.4 s/chunk on CPU, so 2M vectors is ~780 hours of
  continuous embedding — "amortized one-time" is measured in weeks, not hours,
  without a GPU. **Memory** binds before latency does: an HNSW index over 1M
  768-dim vectors needs ~3 GB resident, against a dev machine with ~3 GB free,
  which is the point of the `diskann` seam below.
- Postgres + `pgvector` + `vectorscale`, with a per-embedding-profile vector
  table and ANN index.
- Versioned pipeline revisions: swap a model / chunking policy / extractor, build
  a new revision in the background, and promote atomically — live search keeps
  serving the old revision until then.
- Batch embedding against a warm local llama.cpp server, so the model stays
  loaded across both indexing and interactive search.

## Key seams (the pluggable registries)

### Embedding provider — `embedding_runtime.py`

- `EmbeddingRuntimeSpec` carries model + runtime identity. Providers are
  self-describing: `embedding_dim`, `distance_metric`, `format_document` /
  `format_query`, `embed_batch`, `health_check`.
- `_PROVIDER_FACTORIES` maps provider name → factory; `create_provider(spec,
  config)` is the single resolver every caller uses.
- The runtime spec flows into the embedding profile, so changing the model
  re-versions the revision automatically.
- Implemented: `llama-cpp` (a warm local llama.cpp server). Adding a provider
  (remote API, sentence-transformers, a multimodal model) is one
  `_PROVIDER_FACTORIES` entry plus one branch in `runtime_spec_from_config`.

### Extractor — `extract.py`

- `ExtractorSpec` (name, version, handled extensions) + the `_EXTRACTORS`
  registry keyed by name.
- Pure resolvers: `supported_extensions()`, `extractor_for(path, config)`, and
  `extract_document(path, config)` — the single dispatch the worker and the
  `extract` CLI both call.
- Per-file-type backend choice via the `[extraction.backends]` config map (e.g.
  `pdf = "docling"`); an unset type falls back to the registry default, and a
  backend that can't handle the type raises a clear error.
- Implemented: `pymupdf4llm` (`.pdf`), `plaintext` (`.txt` / `.md` /
  `.markdown`). Adding a document type or an alternative tool for an existing
  type is one registry entry.
- The watcher (`source_watcher.py`) is content-type-driven: `_should_process` and
  `_scan_existing` filter on `supported_extensions()` rather than a hardcoded
  `*.pdf`.

### ANN index — `index_strategies.py` + `vector_store.py`

- `_INDEX_STRATEGIES` (hnsw / diskann) builds the index DDL via
  `build_index_ddl`; the `index.method` config selects it
  (`supported_index_methods()` lists the choices).
- Each embedding profile gets its own fixed-dimension `embedding_vectors_p{id}`
  table, which is what lets DiskANN work.
- Rebuilding a profile's index reconciles it to `index.method` (a stale-method
  index is dropped first, since `CREATE INDEX IF NOT EXISTS` alone would keep the
  old one); it never re-embeds. Search tunes for the index that actually exists,
  not the configured method, so a config change is never silently mis-applied.

### Versioned revisions — `revisions.py` + `profiles.py`

- Immutable, content-fingerprinted extractor / chunk / embedding profiles. A
  pipeline revision ties one of each together.
- Change propagation: model change → reuse text + chunks, re-embed only;
  chunking change → reuse text, rechunk + re-embed; extractor change → rebuild
  all three.
- Revision states: `target → building → ready → active → retired/superseded`.
  `cementic collection promote` flips `active` atomically.
- Failure handling: a failed extract/chunk/embed is terminal *within* a worker run,
  so the build still reaches `ready` (it never spins re-trying the same failure).
  `requeue_interrupted_artifacts` re-queues failed/interrupted rows on the next
  `cementic start`, giving transient failures another attempt. Documents marked
  `deleted` by the watcher are excluded from selection, revision counts, and search.

## Pipeline as composable filters — `cli.py`

`extract` / `chunk` / `embed` are thin stdin/stdout wrappers over the *same* pure
functions the worker calls — no per-document shell-out, so there are two entry
points to one core. They let you debug a single document end-to-end without a
database, test stages in isolation, or pipe a stage to an external tool.

### Markdown as the text intermediate representation

Every document type is normalized to Markdown/text, then chunked, then embedded.
This is the right default and matches mature RAG tooling: one normalized
representation ⇒ one chunker + one embedder, fully composable, and Markdown keeps
lightweight structure (headings, lists, tables, code) that aids chunking and
retrieval while staying plain text — which a *text* embedding model needs anyway.

Caveat: it assumes a text embedder. Genuinely multimodal embedding (a vision
model over a figure or page image) can't pass through Markdown without losing
signal, so images get two options: (a) image → OCR/caption → Markdown (lossy, but
reuses the whole pipeline — cheap) or (b) a parallel path: image → multimodal
embedder (no Markdown, different chunk semantics). The current seams support (a)
today and can grow (b) as a separate extractor + provider pair. So "Markdown IR"
is the contract for the *text* extraction family, not a universal law; per-type
backend choice is what raises text-extraction quality (e.g. docling/marker) where
it matters.

## CLI surface

- Typer with plain, case-consistent help: `USAGE` / `OPTIONS` / `COMMANDS` /
  `ARGUMENTS` uppercased to match the usage metavars, `EXAMPLES:` rendered at the
  base indent; `-h`/`--help` on every command and subcommand; `-V`/`--version`.
- Concise, aligned output for `status`, `collection list`, `collection
  revisions`, and `search`; internals (PIDs, per-worker state, per-file progress)
  live behind `--verbose`; `status --json` for machine consumption.
- Configuration by TOML file, environment variables, or both. Precedence (low →
  high): built-in defaults < config file < `CEMENTIC_*` env vars < command-line
  flags. `cementic config init | path | show` manage it.
- HNSW vs DiskANN is a serving choice exposed as `index.method` — HNSW (pgvector,
  in-memory, lowest latency) vs DiskANN (pgvectorscale, disk-resident, low RAM at
  scale). The method is applied when the index is built and takes effect on the
  next rebuild; it never re-embeds.

## Review history and what is still open

**Everything found by the first four reviews is fixed and merged**, except the
items under "Deliberately not done" below. The fifth pass (2026-08-17) is the
current open findings list — nothing from it is fixed yet. The notes are the
record of what each found; this section keeps only the engineering *lessons and
measurements* that have no other home, in the order they were learned.

- [`notes/code-review-2026-08-07.html`](notes/code-review-2026-08-07.html) — first full pass.
- [`notes/code-review-2026-08-11.html`](notes/code-review-2026-08-11.html) — third
  pass, five unprimed reviewers, so its overlaps are independent re-derivations.
- [`notes/code-review-2026-08-14.html`](notes/code-review-2026-08-14.html) — fourth
  pass. Carries a resolution banner mapping every finding to the commit that
  closed it, and a reconciliation of the two earlier notes, so a fifth review
  starts from that rather than re-deriving.
- [`notes/code-review-2026-08-17.html`](notes/code-review-2026-08-17.html) — fifth
  pass, reviewing the fourth pass's fixes plus fresh eyes per subsystem. Headline
  pattern: several fixes are correct on the path they touched and absent on an
  adjacent path the same defect reaches (`status --json`, the initial scan,
  `stop`'s kill loop, `config path`). **Closed 2026-08-18** on
  `claude/review-fixes-2026-08-17`; the note carries a resolution banner mapping
  finding → commit. Only §2.9's directory-move blindness (unverified, needs a
  repro) stays open — see "Deliberately not done".

**Read the 2026-08-17 and 2026-08-14 notes before opening a new review.** Its most useful section
is not the findings but the ledger of what the earlier passes found and never
fixed — roughly fifteen items were re-derived independently three times before
anyone acted on them.

The 2026-08-11 pass re-confirmed ~20 findings independently (both blockers among
them) and added: an unguarded `shutil.rmtree` in `init postgres --force`; a
revision reaching `ready` with zero documents; failure counts laundered past the
promote gate by a worker restart; `index.method` unreachable on a built system;
`--n_batch` never passed, so raising `n_ctx` is a no-op; raw tracebacks on a
malformed `CEMENTIC_DB_URL` in the three commands meant to explain it; and two
CI marker holes that make a green run meaningless.

### Fixed on `claude/review-fixes-2026-08-11`

Seven commits, each with regression tests; 635 unit tests, ruff and mypy strict
all clean, and `uv lock --check` passes.

1. **CI honesty** — B1 (`uv lock`) and N17 (`pg` markers). The "not pg" job no
   longer drags a 30-60min from-source Postgres build into a database-free job,
   and the only end-to-end smoke test now runs somewhere.
2. **Both data-destruction paths** — B2 (named volume in both `compose.yml`,
   plus the generated README naming `down -v` as the destructive one) and N1
   (`init postgres --force` overwrites template files in place instead of
   `rmtree`-ing whatever it was pointed at).
3. **What gets published** — N2 + N3 + H2 closed together: promotion re-checks
   completeness against current counts, `revision_is_complete` requires
   `documents > 0`, and nothing publishes an empty revision even under
   `--force` (promotion retires the active one, so that removes coverage).
4. **Silently wrong search answers** — C1.4 (chunk boundaries now align to whole
   characters; concatenation of non-overlapping chunks became lossless),
   C1.2 (freshness predicates, with the definition centralised beside the scope
   builders as `CURRENT_CONTENT_SQL` — *since removed from `src/`; the freshness
   join was replaced by deleting stale rows eagerly, and the constant now lives
   in `tests/integration/test_pg_helpers.py` as a fixture invariant*),
   C1.1 + N5 (query bounded in tokens
   against the window; `--n_batch`/`--n_ubatch` now follow `n_ctx`), C1.3
   (`ef_search` never below the requested `top_k`).
5. **H4** — a `ready` revision reports as ready, and `status` names the promote
   command for it.
6. **Failure reporting** — N8 (`stop` exit codes), H3 (runner exit code on fatal
   worker startup failure), H8 (`extract` catches `OSError`), plus the
   undeclared `click` and `pymupdf` dependencies.

### Round-two review (2026-08-11, four reviewers)

One reviewer re-read the batch above adversarially; three re-verified the open
findings against the changed code. The batch had **two real defects and CI was
red** — fixed on `claude/review-round-two`, see that commit. The lesson is
recorded under "Environment facts" above: the unit suite is not the CI gate.

Re-verification changed three findings materially, so the old descriptions
should not be trusted:

- `Path.exists()` does **not** swallow `PermissionError` on 3.12. The watcher's
  mass-deletion trigger is an *unmounted* subdirectory (ENOENT); a permission
  error instead kills the watcher with a traceback. Same fix, different symptom.
- Mixed-model search does not break during any rebuild — a collection with an
  active revision keeps working. It needs ≥2 collections with at least one never
  promoted, and a `ready` revision poisons it indefinitely, not just mid-build.
- The `pymupdf._get_layout` guard is **retracted**: the attribute is declared in
  pymupdf 1.27.1, so the current code is correct.

### Fixed on `claude/review-round-three`

**All of items 1-9 below, plus the config leftovers and the opportunistic
group**, each with regression tests. In order: empty extractions; the config
diagnosability cluster (including the value-leak in error messages); provider
failures reaching `status` with a capability startup gate; `~`/relative path
handling; the daemon-probe consolidation; the watcher bundle; the per-file
view's freshness predicates; the unindexable-dimension pre-flight; the
task-prefix policy in the profile; doctor's model-path confinement; and
`embedding stop`, OCR and page-chunk handling.

**One of these forces a rebuild.** Recording the task-prefix policy changes
every existing embedding-profile fingerprint, so the next `cementic start`
builds a fresh revision and re-embeds once. That is the price of not silently
mixing prefixed and unprefixed vectors in one table, and it is on its own commit.

Two notes for whoever picks this up:

- **Test fixtures were unrealistic, not the code.** Three `load_file_progress`
  tests seeded artifact rows without the hashes the worker writes. Both step
  functions set those on the *failure* path too (in `_step_extract`'s and
  `_step_chunk`'s write-back blocks — line numbers have drifted since), so the
  fixtures, not the new predicates, were wrong. Check that before assuming a
  similar failure means a regression.
- **A relative `artifacts_path` is still relative** — to where cementic was
  started. Anchoring at load time removes the silent-no-op deletion, but it
  cannot make a relative path mean the same thing from two directories. A real
  mismatch is now refused rather than quietly succeeding.

### Fixed on `claude/review-round-four`

All four remaining items, plus one found while sizing them.

1. **Pruning leaks** — the embedding delete was scoped through the *chunk*
   profiles being removed, so a swap that changed only the model matched
   nothing: the retired model kept every `chunk_embeddings` row and its whole
   `embedding_vectors_p{id}` table, one full copy of the corpus per swap. Both
   anticipated footguns were real and are handled — the droppable set is
   re-queried globally, and the `DROP TABLE` is deferred past `commit()`
   alongside the artifact removals. A third turned up: a profile another
   collection still uses keeps its table, but this collection's vectors in it
   have to go explicitly, because the `chunks_v2` cascade does not reach them
   when the chunks themselves are untouched.
2. **Every command paid ~0.5s warm / ~1.3s cold for a PDF layout model.** Not
   on the list; found while checking whether a config validator could afford to
   import the extractor registry. `cli` imports `extract`, which imported
   `pymupdf.layout` (ONNX analyser + networkx) at module scope. Importing
   `cementic.cli`: 1.9–3.4s before, 0.36s after. The existing runtime budgets
   could not see it — they time dispatch, after the test module has already
   imported the CLI — so the guard is structural instead.
3. **`[extraction.backends]` validation**, mirroring `index.method`. Unknown
   name and wrong file type are now separate messages.
4. **Mixed-model search message** names each collection, its model and its
   status. Kept as a refusal for both paths: scores from different models are
   not comparable, so dropping the odd collection would produce a silently
   meaningless ranking rather than a visible error.
5. **`cementic collection reindex`**, with `--force` for the build-time knobs
   (`hnsw_m`, `ef_construction`) that `CREATE INDEX IF NOT EXISTS` would
   otherwise leave at their old values while reporting success.

Two notes for whoever picks this up:

- **Deferring the pymupdf import moved it inside test patch contexts.** The
  first `import pymupdf.layout` runs an `activate()` that rebinds
  `pymupdf4llm.to_markdown`, so a patch applied beforehand was silently
  replaced mid-test — and only when no earlier test had already triggered the
  import, making it look like flakiness. The extraction tests patch
  `_get_pymupdf` now; do not go back to patching the real modules.
- **Three PG search tests fail on `main` too** (`test_search_pg.py` ×2,
  `test_cli_pg.py` ×1) — diagnosed and fixed since; see below.

### Fixed on `claude/pg-correctness`

**The three red PG tests were a regression of ours, not inherited.** `74942b4`
added `CURRENT_CONTENT_SQL` to search's WHERE clause;
`seed_active_vector_collection` had never set `source_file_hash`,
`content_hash` or `source_content_hash`, so every seeded chunk failed
`ed.source_file_hash = sd.file_hash` — `NULL = NULL` is not true — and search
returned nothing. It merged because the unit and non-PG jobs were green and the
PG job was not run locally.

The production code was never implicated: `write_extracted_text` returns a hash
or raises, so a `done` extraction always has one, and the live database has
zero NULL hashes on `done` rows across 172 chunks.

`scripts/check.sh` now runs all five gates in one command and reports a missing
PostgreSQL as SKIPPED rather than passed.

### Fixed on `claude/ann-filterable-vectors`

**The ANN index is now reachable.** The filter columns (`collection`,
`extractor_profile_id`, `chunk_profile_id`) live on `embedding_vectors_p*`, so
the `WHERE` clause applies to the vector row and the planner can drive from the
index scan. Same data, same index, 100k rows at 768 dimensions:

| query shape | ANN used | results | time |
|---|---|---|---|
| filters on joined tables (before) | no | 10/10 | 407.6 ms |
| filters on the vector row | yes | 10/10 | 1.0 ms |
| filters on the vector row, 2% slice, `iterative_scan=off` | yes | **0/10** | 1.3 ms |
| filters on the vector row, 2% slice, `relaxed_order` | yes | 10/10 | 15.3 ms |

`hnsw.iterative_scan` landed with it, defaulting to `relaxed_order` and gated on
pgvector ≥ 0.8 — row 3 is why. The gate is not optional: PostgreSQL accepts an
unknown *qualified* setting as a placeholder until the defining module loads on
that connection and rejects it with `InvalidName` afterwards, so behind a
connection pool an ungated `SET` fails only on connections that had already run
a vector query. (`SET LOCAL diskann.query_rescore` being accepted where
pgvectorscale is absent is the same effect — not evidence it took effect.)

**What the three columns cost.** They replace query-time freshness filtering
with an invariant: stale vectors are deleted when they go stale. Re-chunking
already did this via the `chunks_v2` cascade; two paths did not and now do —
document deletion (`_purge_document_chunks`) and re-extraction that changes the
content (`_purge_superseded_chunks`). The visible behaviour change: a document
whose re-extraction succeeded but whose re-chunking has not run returns nothing
rather than its previous contents.

Existing vector tables are migrated in place by `ensure_vector_table_schema` —
`ADD COLUMN`, backfill from the joins, then `SET NOT NULL`. No re-embedding.

**No test reproduces the thin-slice recall failure**, deliberately. It needs
~100k rows: below that the planner picks an exact sequential scan for a
selective filter, which returns the right answer and would make the test pass
for the wrong reason. Verified that trap directly — at 4k rows the `slice`
query plans as a seq scan even with `enable_seqscan = off`. What CI does pin is
that the ANN index *is* in the plan, which is the thing that regressed.

### Fixed on `claude/index-build-stage-1`

**The index build is faster, visible, and explicable.** `ensure_revision_ann_index`
runs synchronously in `_mark_revision_ready_if_complete`, at the
`building → ready` transition — *not* at promotion, as an earlier note here
said.

- `index.build_memory` (default 2GB) raises `maintenance_work_mem` on the
  build's own connection. 100k × 768 is 293 MiB of graph against Postgres's
  64MB default, so the build spilled: **1454 s at 64MB, 345 s at 2GB**.
- The worker publishes `current_activity` around the build, because nothing can
  be published from inside it. `cementic status` previously showed a running
  worker, a `building` revision and no current file — identical to idle.
- `cementic stop` waits 10 s and then *refuses* (it does not force-kill; the
  5-second SIGKILL is `_terminate_managed`, used only on start-up rollback). The
  worker cannot answer SIGTERM from inside `CREATE INDEX`, so that timeout was
  guaranteed and read as a hang. Both the timeout and `--force` now say what is
  running and that forcing discards it.

Verified that a killed build loses everything: on a 40k table whose build takes
135 s, killing the builder at ~34 s leaves only the primary-key index.

### Decided — build the ANN index up front (2026-08-13)

**Taken.** `ensure_revision_ann_index_up_front` (`revisions.py`) creates the
HNSW index beside the vector table in `_ensure_target_revision`, guarded to the
empty-table case (rows without an index mean a resumed build — its bulk build
stays at the ready transition) and to HNSW (DiskANN is unmeasured and keeps
build-at-ready). The ready-transition and `collection reindex` calls stay for
resumes and method changes. The measurements that justified it:

**Should the ANN index be created up front, on the empty table?** HNSW has no
training step, so it can be. Every insert then maintains the graph and the
build stall disappears entirely. Measured at 100k × 768, `maintenance_work_mem`
2GB for both, clustered vectors (uniform random ones sit at near-identical
distances, which makes recall meaningless):

| | insert then build (today) | build then insert |
|---|---|---|
| insert | 61.8 s | 413.1 s |
| build | 81.5 s — worker blocked | 0.0 s |
| **total** | **143.3 s** | **413.1 s** |
| longest unresumable step | 81.5 s | none |
| recall@10 mean / worst | 0.996 / 0.900 | 1.000 / 1.000 |
| query latency | ~0.74 ms | ~0.74 ms |
| index size (30k) | 117 MB | 117 MB |

So it costs ~2.9× total indexing time and buys: no stall, full resumability
(inserts commit per batch, so a kill loses one batch instead of the whole
build), a searchable index *during* ingestion rather than after, and recall and
latency that are equal or better. pgvector's guidance that building after
loading is faster holds; that it yields a better graph did not, here.

The 2.9× is on the insert step, and cementic's pipeline is embedding-bound:
+351 s per 100k vectors is ~3.5 ms of index maintenance per chunk, against tens
of milliseconds to embed one. Proportionally small, and spread out instead of
concentrated.

**Two earlier numbers here were wrong and are corrected above.** A first A/B run
reported a 100× query-latency gap and a recall difference between the two build
orders. Both were artifacts: the run never checked which plan each query got,
and the `enable_indexscan = off` used to force an exact baseline leaked onto
pooled connections, so some "ANN" queries were sequential scans. Re-measured
with both plans asserted via `EXPLAIN` and a separate pool for the baseline, the
two indexes are the same size and the same speed.

### Fourth review (2026-08-14) — the lesson worth keeping

What it found and what closed each is in the note. One lesson has no other home,
because it is about how the fix itself went wrong:

**A guard is only as good as the thing its test measures.** The headline finding
was that `chunk_size` (tiktoken tokens) was pitted against `n_ctx` (the model's
own tokens), silently truncating ~30% of every full chunk. The fix lowered
`chunk_size` and added a runtime guard with a cheap pre-filter, pinned by a new
invariant test. Both the chosen constant and the test were wrong in the same way:
they measured the bare chunk, while the guard measures the *formatted* text, and
the `search_document: ` prefix adds 3 tokens. That pushed every chunk 3 tokens
past the skip threshold, so the "rare" exact-count path ran on every single
chunk — which, against a daemon that serialises requests, stalled a live
re-index for hours. The test now builds a real chunk and formats it exactly as
the client does; set back to the old value, it fails.

The same trap produced the original bug: `measure_chunk_context_fit.py` embedded
`chunk[:len*0.75]` while printing the *full* chunk's token count beside an "ok"
verdict, so the config comment citing it as proof of safety was citing a
measurement that never tested the case it claimed.

### Superseded

**The ANN index is never used by cementic's search query.** *(Fixed above; kept
because the measurements are the justification for the schema change.)* Measured
with `EXPLAIN (ANALYZE)` against a real corpus:

| query | plan | time |
|---|---|---|
| bare KNN, 768-dim, 20k rows | `Index Scan using ...ann` | 2–6 ms |
| cementic's search query, same data | top-N heapsort over a full nested loop | 25 ms |
| cementic's search query, 8-dim, 60k rows | same, 60k per-row PK lookups | 83–90 ms |

Confirmed across 5k/20k/60k/100k rows, 8 and 768 dimensions, filters matching
40 rows or every row, and with `enable_seqscan` both on and off. The planner
always drives from `chunked_documents` and probes `embedding_vectors_p*` by
primary key, because every filter (`collection`, the profile ids, the freshness
predicates) lives on *joined* tables rather than on the vector table.

Consequences, in order of importance:

- Search is exact — so this is a scaling defect, not a correctness one. Results
  are right, and were right before the freshness predicates too.
- Search costs O(rows in the profile's vector table) per query, growing
  linearly. (Superseded: the filter columns below made the ANN index reachable,
  so search no longer scales with table size.)
- `index.method`, `hnsw_m`, `ef_construction`, `hnsw_ef_search`, the DiskANN
  knobs and `collection reindex` all maintain an index nothing reads.

A no-schema-change alternative was considered and rejected: a materialised CTE
doing the bare KNN first, then filtering. It takes the global top-N and filters
afterwards, which is arithmetically the same as `iterative_scan=off` — it
degrades to zero results in exactly the case that motivates it.

### Deliberately not done

Findings from the **first** review (2026-08-06) judged not worth fixing — not
oversights, and distinct from the feature roadmap in `TODO.md`. Reasoning is in
the relevant commit messages: ANN pre-filter recall on shared vector tables
(inherent to ANN + post-filter; the actionable slice is purging vectors for
deleted documents), `check_health` treating a live-but-broken daemon as healthy
(documented tradeoff), DiskANN + inner-product `storage_layout` (unreachable
while only cosine exists), and reading model identity from GGUF metadata instead
of the filename (the filename heuristic is now at least *reported* by
`cementic embedding start`).

Of those, the `status` half of the `check_health` tradeoff **was** reopened and
fixed — `status` no longer blocks 120 s to return an answer the pid file already
had. Two remain open and are worth reopening rather than re-closing:

- **`check_health` still calls a live-but-broken daemon healthy**, and this is no
  longer theoretical. On 2026-08-15 a wedged daemon held the port for 21 hours
  while `cementic status` reported `embedding healthy` and every request hung.
  It only probes whether `/v1/models` lists the model.
- **The filename heuristic silently disables Nomic task prefixes** while
  `models/nomic-embed-text-v1.5.f16.gguf` sits in the repo — v1/v1.5 need the
  same prefixes as v2 but do not match `_NOMIC_V2_MARKER`. Note the wrong
  behaviour is currently *pinned by a test* (`test_embedding_text.py`), which
  must be retired with the fix or it reads as intentional.

From the **fifth** review (2026-08-18), deferred or kept with reasons:

- **The watcher is blind to directory moves until restart** (fifth review §2.9,
  tagged unverified). Needs a repro against watchdog's actual event stream
  before any fix — a speculative fix here would be code for a defect nobody has
  observed.
- **TOML parsed once per pydantic-settings section source** (~9× per `Config`
  construction). Caching keyed on path+mtime risks stale reads on
  coarse-mtime filesystems — test flakiness — for a millisecond-scale win.
  Revisit only if config load ever shows up in a profile.
- Three §5 trim candidates stay: `_llama_daemon_runtime_status` (a seam its
  tests pin), `_state` (turned out multi-use), and the `state_path is None`
  guards (they defend the field's declared type).

Closed again by the round-two review, with reasons — each of these looks like a
bug and is one, but the fix costs more than the defect:

- **`verbose` in the embedding profile fingerprint.** Removing it re-fingerprints
  every existing profile and re-embeds every corpus — exactly the harm it is
  accused of causing — to protect against a debug flag almost nobody toggles.
  Only worth doing batched with a model-identity change, so users pay once. It
  correctly stays in the *runtime* fingerprint, where it is a launch argument.
- **Naive `TIMESTAMP` columns.** Zero readers today (one write, no comparison,
  no display). The fix is a column-type change in a project whose only schema
  mechanism is `create_all`, so old and new databases would diverge with nothing
  to reconcile them. Revisit if anything ever reads these columns.
- **`connect_args` / `gssencmode` on non-psycopg URLs.** `DatabaseConfig` can
  only produce `postgresql://`, and sqlite never reaches `get_engine` outside
  tests that patch around it. Worth the `make_url` cleanup only if `db.py` is
  open for another reason — *and it has been twice since* (`build_memory`, then
  the atomic `force_rebuild` drop), so the stated precondition is now met.
  Note the gap is wider than first recorded: the check is
  `startswith("postgresql://")`, so it also misses driver-qualified URLs like
  `postgresql+psycopg2://`, which a user setting `CEMENTIC_DB_URL` may well write.
- **`ignore_directories` replacing the defaults.** Working as documented,
  including the "empty list indexes everything" escape hatch a union would
  break. If discoverability is the concern, add a separate
  `additional_ignore_directories` — a feature decision, not a defect fix.
- **The advisory-lock leak and the `RUNNING` state-write ordering.** Every path
  that leaks the lock is immediately followed by process exit, which releases
  it; the stale state file is corrected by `start`'s liveness check and cleared
  by `stop`. Two-line fixes if those functions are open anyway, not worth a slot.
- **`index.method` being unreachable on a built system.** Closed by
  `cementic collection reindex`, which reconciles the active revision's index
  with the current `[index]` config. *(The instruction that used to sit here —
  "correct the sentence in `TODO.md` claiming the build path already reconciles
  a changed method" — was itself stale: no such sentence exists in `TODO.md`.)*

**Also still open, from the fourth review** — full detail and file:line in
`notes/code-review-2026-08-14.html`, listed here so they are not buried:

- `cementic start` reports success after a 2 s grace period, while the startup
  path can fail up to `daemon_start_timeout_seconds` (120 s) later. The failure
  reason goes only to a background log whose path is printed in the *other*
  branch and appears nowhere in `status` or `doctor`.
- An orphaned embedding daemon cannot be reclaimed once its pid record is lost:
  `embedding stop` reports "already stopped" while the process holds the port.
  Observed live on 2026-08-15 — this is what stalled a re-index for hours.
- Model identity is the *path string*, so `resolve_llama_model_path` trying cwd
  first means indexing from two directories can silently mean two different
  GGUF files under one fingerprint. Batch with the `verbose` change above, since
  both force a re-embed.
- `EXTRACTION_VERSION` is a hand-maintained integer, not the pymupdf/pymupdf4llm
  versions that actually produce the Markdown, so a dependency bump changes
  extraction output without moving the fingerprint.

## Deferred

- **Images / multimodal** — one new extractor (OCR/caption, or raw-image
  passthrough) + one multimodal embedding provider. Both self-register; the
  registries above mean nothing else changes. (A true vision embedder bypasses
  the Markdown IR — the parallel path noted above.)
- **More document types** (`.docx`, `.pptx`, `.html`, `.epub`) — one extractor
  entry each.
- `cementic add <path>` for direct one-off ingestion.
- Optional Markdown artifact mirrors alongside the compressed pipeline artifacts.
- Richer search-result metadata (document id, collection, artifact path).
- Hybrid lexical + vector search (exact author names, acronyms, equation labels).
- A multi-profile embedding daemon pool, if old-model search and new-model
  indexing must run concurrently.
- Lighter embedding providers if llama.cpp memory use is too high on small
  machines.
