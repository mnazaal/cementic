# cementic — architecture & design

<!-- session-handoff:begin (2026-09-02) -->
## Where the work stands

**Entry point: the `papers` import is RUNNING and needs no babysitting — read
"Decided — query-first embed scheduling" below before touching embed
throughput, because this session's headline finding there is now in doubt.**

**Repo state.** Working tree clean. `main` is at `90d66ad`, **one commit ahead
of `origin/main`**; this handoff commit sits on **`claude/session-handoff-2026-09-02`,
unmerged**. Merging and pushing are the user's calls (a hook rejects agents
touching `main`):
```bash
git checkout main && git merge --ff-only claude/session-handoff-2026-09-02
```
Three code commits landed today, all six gates green on the final state
(`./scripts/check.sh`, exit 0, single clean run):
- `9681259` embed request size follows the search lease instead of being constant.
- `8254f77` `cementic stop` grace 10 s → 30 s.
- `90d66ad` retry walk bounded to groups of 4 and made shutdown-responsive.
Branch `claude/embed-retry-bounds` is merged into `main` and can be deleted.

**Live state.** Collection `papers`, revision 16, still `building`. At 15:44 on
2026-09-02: 2,265,313 done / 5,177 failed / 21,033 pending of 2,291,555. Worker
started 14:33 and is progressing. When it drains: `cementic collection promote
papers`. Only `papers` exists now — `soak` and `test` are gone, and the vector
table is `embedding_vectors_p6`.

If it needs restarting (`stop` then `start` is safe again as of `90d66ad`):
```bash
cementic stop
cementic start /u/71/ibrahin1/data/Documents/Papers -c papers
```
Every `start` requeues the ~5,000 over-budget failures, so `failed` drops and
`pending` jumps at each restart. That is the requeue, not new breakage.

**Correction — distrust this session's throughput story, not just its numbers.**
Three commits were justified by "constant sub-batching at 4 cost ~2× indexing
throughput". Re-measured after the fix shipped, that attribution does not hold
up: requests are confirmed claim-sized again (40 groups of exactly 32 launches
between client round trips in the daemon log) with a 3-hour-stale search lease,
and throughput is **2.24 chunk/s — inside the 1.7–2.9 band the sub-batched
build ran at**, not the ~4.0 that preceded it. So restoring big requests did
*not* restore the old rate, and the 4.0 → 2.0 step at the 2026-09-01 17:23
restart has an unidentified cause. Ruled out this session: chunk length (flat at
317–323 model tokens across the entire run, measured from the daemon log's
`n_tokens`) and a missing GPU (`--list-devices` shows the Vulkan iGPU; the
server has never restarted). Still open, in rough order of promise: HNSW insert
cost growing with the graph (fits the slow 5.5 → 4.0 decay better than the
step), host/GPU contention, and pgvector index maintenance.

The changes are still worth keeping on their own merits — the per-request
overhead is real (~0.36 s), and the `stop`/retry fixes are correctness fixes —
but the "~2×" figure in `9681259`'s message, in PLAN's scheduling amendment, and
in `embed_submit_size`'s docstring is **unconfirmed** and should not be repeated
as established.

**A hard ceiling worth knowing before optimising anything here.** The server
runs `--ubatch-size {n_ctx}` = 512 and chunks average ~318 model tokens, so
roughly one chunk fits per forward pass: slot 0 took 300 of the last 400
launches with 32 tasks queued. Extra slots and bigger requests cannot buy
concurrency the physical batch has no room for. Raising `n_ctx`/ubatch is the
untested lever, and it changes the runtime fingerprint (`--alias`), so cementic
would refuse the running server until it restarts.

**Load-bearing numbers a cold reader should not re-derive.**
- One `/v1/embeddings` request costs ~0.36 s fixed plus ~0.38 s per input
  (0.72 s for 1 input, 12.2 s for 32), measured against the live server.
- ~5,000 chunks fail as over-budget. `embed_batch` pre-filters on the *cheap*
  tiktoken estimate, so 513–549-model-token chunks still reach the server and
  return a 500 that `_server_rejected_the_input` treats as terminal. Each
  `cementic start` requeues them, so failures re-run and re-fail every restart.
- 113 documents fail extraction as scanned PDFs with no text layer. Expected.

**Reading the daemon log** (`~/.local/share/cementic/llama_cpp_daemon.log`, 566
MB, never rotated): timestamps are **uptime**, not wall clock, formatted
`min.sec.ms.us`, and the file concatenates every server run — find the current
run with `grep -an 'load_model: loading model'` first. Two artifacts: a requeue
shows as an hour *above* the GPU rate (over-budget chunks are rejected at zero
tokens), and startup banners with `couldn't bind` are failed autostarts, not
restarts. Probe scripts for both measurements are in `notes/`
(`probe_server_idle.py`, `probe_embed_batch_scaling.py`) — note `notes/` is
gitignored, so they exist on this machine only and not in a fresh clone.

**Environment quirks.**
- The agent sandbox has a **separate PID namespace**: `/proc/<pid>` and `ps` are
  useless for liveness. Check the database and file mtimes instead.
- `cementic` on PATH is the uv-tool snapshot; the project `.venv` is an editable
  install of the working tree, so `.venv/bin/cementic` runs uncommitted code.
  The running worker was started from `.venv` and therefore has all three fixes.
- Only one `pytest -m pg` run at a time — two overlapping `./scripts/check.sh`
  runs corrupted this session's evidence and had to be discarded.

**Deviation from plan, agent-decided:** the throughput work was not on any
roadmap. It started from "what's next?" and displaced the queued items below.

**Exit criteria — commands whose output confirms the above.**
```bash
git status --short                                   # empty
git log --oneline origin/main..main                  # 90d66ad (+ handoff once merged)
git branch --show-current                            # main, after merging the handoff
cementic status -c papers                            # counters climbing
./scripts/check.sh                                   # six gates, exit 0 (~4 min)
```
<!-- session-handoff:end -->

## Decision log

**Migrated 2026-08-23; this section is now a pointer.** The eight entries it held
were three different kinds of thing, and keeping them together made a register
that duplicated the code:

- **Code decisions** — context manager over decorator, the narrowed
  `EXTRACTION_VERSION`, the deferred imports — already live against the code they
  explain (`_reporting_db_errors`'s docstring, `profiles.py`, `config.py`). A
  comment beside the thing it explains cannot drift from it; a separate register
  can, and these had already become copies.
- **Project policy** — `notes/` gitignored, no backwards-compatibility
  obligation, the 3.12 floor — moved to README's "Decisions worth knowing".
- **Scheduling** — why Batch C ran before Batch A, why a characterization test
  preceded the refactor — deliberately dropped. They were about sequencing one
  piece of work and answer nothing six months out.

The two parked items that had revisit conditions moved to `TODO.md`'s "Parked,
with a trigger", because nothing here was watching for them.

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

The target is no longer an estimate: the corpus was surveyed on 2026-08-24 at
22,246 PDFs / 592,248 pages, which at 3.65 chunks/page is **~2.16M vectors**.
The numbers behind that (query latency, embedding and extraction throughput,
GPU rates, HNSW memory) are in README's "Measurements behind the defaults";
the design consequence is everything under Key seams below:
per-embedding-profile vector tables, a `diskann` seam for when memory binds
before latency does, versioned revisions so a model or chunking change builds in
the background and promotes atomically, and a warm llama.cpp server shared by
indexing and search.

## Key seams (the pluggable registries)

### Embedding provider — `embedding_runtime.py`

- `EmbeddingRuntimeSpec` carries model + runtime identity. Providers are
  self-describing: `embedding_dim`, `distance_metric`, `format_document` /
  `format_query`, `embed_batch`, `describe()`.
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

## Design decisions and open items

### Decided — query-first embed scheduling (2026-09-01)

**Taken; implementation next.** A search query landing mid-import waits on the
shared llama-server: the worker POSTs 32 texts per request, b10605 schedules
per-input FIFO with no HTTP priority hook, so a query's one task queues behind
up to 32 — measured 0.4–9.6 s of queue delay (17.2 s worst-case wall) against
~0.1 s of actual ANN+SQL. Design, prior-art survey (storage engines, vector
DBs, inference servers, desktop indexers), and verified b10605 scheduler facts:
`notes/design-embed-scheduling.html`.

Three layers, first two to build:

1. **Sub-batch submission** — keep the DB claim at 32, submit to llama-server
   in sub-batches of ~slot count (4). Bounds query wait ~10 s → ~1 s with no
   coordination code (queue delay is proportional to per-request input count).
2. **Search lease** — `search` upserts a timestamp (`utc_now()`, house style) before embedding;
   the worker checks between sub-batches and yields while fresh (TTL ~10 s,
   refreshed per search), with a starvation cap so continuous searching cannot
   stall a build. Precedent: Windows `IRowsetPrioritization`, Postgres
   autovacuum cancellation, OpenAI flex tier.
3. **Abort in-flight on lease activity** — possible (b10605 cancels queued
   tasks ≤ ~1 s on client disconnect) but not built: Layer 1 shrinks the
   quantum until aborting saves ≤ ~1 s. Seam left open.

Parked with triggers: a second query-side llama-server (Vespa's documented
practice; the strong fix if Layers 1–2 measure insufficient — costs runtime-
fingerprint work and resident memory) and an upstream llama.cpp patch exposing
the internal front-of-queue path as a `priority` field on embeddings.

**Amended 2026-09-02 — Layer 1 was not free, and is now conditional.** The
design claimed sub-batching left indexing throughput untouched when nobody
searches. It did not: measured on the papers import, a constant request size of
4 cost **~2× throughput** (≈4.0 → ≈1.7–2.9 emb/s), a step visible at the exact
worker restart that put it into service. Two causes, both measured against the
live llama-server: a fixed ~0.36 s per request (0.72 s for one input against
12.2 s for 32, so ~0.36 s of that is size-independent), and ~25% of wall clock
with no slot busy, because a request sized to the slot count empties all four
slots across every client round trip.

The request size now follows the lease instead of being constant
(`embed_submit_size`): the whole claim when no search is recent,
`embed_submit_batch_size` while one is. Layer 2's yield is unchanged and still
does the work during an interactive session. The residual is the honest cost of
not building Layer 3 — the *first* query after a quiet stretch can land
mid-request and wait out one full claim. It is bounded, and it exists only
while a build runs: once a collection is promoted the worker embeds nothing and
queries are uncontended.

This also promotes Layer 3 from "value evaporated" to the thing that would
remove the residual, since the premise that retired it — that Layer 1 had
already shrunk the quantum for free — is what the measurement falsified.

**Correction, same day, after the fix shipped: the ~2× attribution above does
not hold.** With the adaptive sizing running, requests are confirmed
claim-sized (40 groups of exactly 32 launches between client round trips) and
the search lease is hours stale, yet throughput is 2.24 chunk/s — inside the
1.7–2.9 band the constant-4 build ran at, not the ~4.0 that preceded it.
Restoring big requests did not restore the old rate, so the 4.0 → 2.0 step at
the 2026-09-01 17:23 restart is **unexplained**; it correlated with the
sub-batch change and was attributed to it on that correlation alone.

Ruled out since: chunk length (flat at 317–323 model tokens across the whole
run) and GPU availability (the iGPU is visible and the server never restarted).
Still open: HNSW insert cost growing with the graph — which fits the slow
5.5 → 4.0 decay better than it fits a step — host contention, and pgvector
index maintenance.

What survives: the fixed ~0.36 s per request is measured and real, so
sub-batching does cost *something*; and a hard ceiling was found that bounds
any work here — `--ubatch-size` is 512 while chunks average ~318 model tokens,
so about one chunk fits per forward pass and one slot took 300 of 400 launches
with 32 tasks queued. Request size cannot buy concurrency the physical batch
has no room for. Raising `n_ctx`/ubatch is the untested lever and changes the
runtime fingerprint, so the server must restart for it.

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

### Deliberately not done

Findings from the **first** review (2026-08-06) judged not worth fixing — not
oversights, and distinct from the feature roadmap in `TODO.md`. Reasoning is in
the relevant commit messages: ANN pre-filter recall on shared vector tables
(inherent to ANN + post-filter; the actionable slice is purging vectors for
deleted documents), `check_health` treating a live-but-broken daemon as healthy
(documented tradeoff), DiskANN + inner-product `storage_layout` (unreachable
while only cosine exists), and reading model identity from GGUF metadata instead
of the filename.

That last item bundled two concerns the code has since separated, so read it as
half-closed. *Which model produced these vectors* no longer comes from a
filename: Batch C (`77e0e3f`) made it a content digest (`profiles.py:169`).
*Which text policy applies* is still a filename match (`embedding_text.py:33`),
deliberately so — that half is the fifth-review residual below, and it is the
only live part of this item.

Of those, the `status` half of the `check_health` tradeoff **was** reopened and
fixed — `status` no longer blocks 120 s to return an answer the pid file already
had. Two more were reopened and are now also closed; both entries below were
still written as open when the sixth review (2026-08-23) checked them against
the code, which is why they carry their resolution inline:

- ~~**`check_health` still calls a live-but-broken daemon healthy.**~~
  **RESOLVED 2026-08-18** by the release plan's batch 1a (`04e974a`):
  `DaemonHealth.WEDGED` (`embedding_runtime.py:553-567`) plus the two-stage
  probe, with the busy/wedged disambiguation observed working live during the
  batch-3 build. The motivating incident stands as the record: on 2026-08-15 a
  wedged daemon held the port for 21 hours while `cementic status` reported
  `embedding healthy`.
- ~~**The filename heuristic silently disables Nomic task prefixes.**~~
  **RESOLVED 2026-08-18** by batch 1b (`9f1d148`): `_NOMIC_V2_MARKER` no longer
  exists; `embedding_text.py:33` is `_NOMIC_FAMILY_MARKER = "nomic-embed-text"`,
  matching v1/v1.5/v2, and the pinning test was retired. The *residual* is
  unchanged and still deferred: matching on filename at all still mis-selects
  for a renamed GGUF, and reading GGUF metadata remains the real fix.

From the **fifth** review (2026-08-18), deferred or kept with reasons:

- ~~**The watcher is blind to directory moves until restart** (fifth review
  §2.9).~~ **RESOLVED 2026-08-18** (`b1999da`) — the repro was run and the
  measured event table is in README's "Measurements behind the defaults":
  move-out (case C) was real and is fixed; root-move (case D) is unfixable from
  inside the watch and is documented in README's Known limitations.
- **TOML parsed once per pydantic-settings section source** (~9× per `Config`
  construction). Caching keyed on path+mtime risks stale reads on
  coarse-mtime filesystems — test flakiness — for a millisecond-scale win.
  Revisit only if config load ever shows up in a profile.
- Three §5 trim candidates stay: `_llama_daemon_runtime_status` (a seam its
  tests pin), `_state` (turned out multi-use), and the `state_path is None`
  guards (they defend the field's declared type).

Closed again by the round-two review, with reasons — each of these looks like a
bug and is one, but the fix costs more than the defect:

- ~~**`verbose` in the embedding profile fingerprint.**~~ **RESOLVED 2026-08-23**
  by Batch C (`77e0e3f`). The entry's own condition is what closed it: removal
  was only worth doing batched with a model-identity change so the corpus is
  re-embedded once, and Batch C was that change. `profiles.py:174` now records
  the deliberate absence. It correctly stays in the *runtime* fingerprint, where
  it is a launch argument.
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
  **Audit 2026-08-23:** still live at `db.py:316`, and the entry now fails its
  own test — a register of things deliberately not done cannot hold an item whose
  stated trigger has already fired. Queued in `TODO.md` instead.
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

**Closed 2026-08-23 — the four fourth-review carry-overs.** All four are done
and were verified live; the commits cited below carry the reasoning.

- `cementic start` reporting success after 2 s → Batch B (`ba3ac8a`): the fatal
  reason now reaches `cementic status`, and the message no longer claims a
  success two seconds of watching cannot establish.
- The unreclaimable orphaned daemon → Batch A (`e5ba0b9`): recovery from `/proc`
  by the spawned command line, which closes the general case rather than the one
  cause Batch 1 fixed.
- Model identity being the path string → Batch C (`77e0e3f`): identity is a
  content digest.
- `EXTRACTION_VERSION` hand-maintained → Batch C (`77e0e3f`): derived from the
  installed extraction libraries.

### Audit findings not fixed (2026-08-24)

Three read-only audits ran before the bulk import — hostile document content,
failure handling, scale. Four findings were fixed (`<|endoftext|>` chunking, the
unescaped LIKE prefix, both write-back stalls, and separately the NUL bytes and
the retryable-500). These are the rest, kept because they are real and because a
finding that lives only in a chat log is a finding nobody will act on. None
blocks the import.

Checked against the real 22,246-file corpus and **not present**, so reachable in
principle but not here: filenames containing `[` (would break
`status --verbose`, `render.py:260`), non-UTF-8 filenames (`source_watcher.py:566`
then `render.py:157`), and paths over ~2,600 bytes (btree limit on
`ix_source_documents_collection_source`, `db.py:70`).

Live, in rough order of how likely they are to bite:

- **`cementic status --verbose` has no limit** (`status_service.py:477`,
  `render.py:253`). One line per document; at 22k it loads ~50 MB of ORM objects
  and floods the terminal. Add `--limit`, defaulting to failed/pending first.
- **`_step_embed` publishes no worker activity** (`pipeline_worker.py`, the
  embed step writes neither `current_file` nor `current_activity`). The
  busy-vs-wedged guard reads exactly that state, so during the phase that
  saturates the daemon a healthy server can be reported WEDGED — and the
  suggested remedy, restarting it, discards an in-flight batch.
- **Whole-corpus deletes run as one unbounded transaction** with a 22k-element
  `IN` list (`source_watcher.py:338`, `collections.py:136`), blocking autovacuum
  for its duration and leaving millions of dead index entries.
- **Progress rounds to `100.0%` while thousands of chunks are outstanding**
  (`status_service.py:115`); at 2.16M the display resolution is ~2,162 chunks.
- **A revision can reach `ready` mid-initial-scan** (`pipeline_worker.py:977`):
  the watcher registers documents one at a time with no scan-complete marker, so
  a drain of the first N looks complete. Promotion re-checks and refuses, so this
  misleads rather than corrupts.
- **`chunk_text` is quadratic in one whitespace-free run** (`chunk.py`): 160k
  characters took 12.2 s, inside a Rust call that ignores the shutdown event.
  Reachable from a minified line or a PDF text layer with no spaces.
- **`extractor_registry_payload` hashes the whole registry** (`extract.py:282`),
  so adding a `.docx` extractor re-versions every existing PDF corpus. Matters
  more now that new extractors are near-term work.
- **`extract_pdf_markdown`'s signature defaults `use_ocr=True`** while the config
  defaults `False` (`extract.py:47`). No effect today — `_pdf_extractor` always
  passes the config value — but the two disagree.

Resolved by the systemd units rather than by code: the worker had no respawn
after an OOM or reboot, so a multi-day run ended silently.

## Road to v1

All four must hold before a 1.0 tag; none is scheduled work yet:

1. **Soak time under real use** — weeks of daily driving on the live corpus
   without surprises. Confidence comes from use, not review passes.
2. **More features first** — some of `TODO.md` belongs in a v1: more
   extractors (`.docx`/`.html`/`.epub`), `cementic add`, richer search
   output. Which subset is a decision for when v1 planning starts.
3. **Config/CLI stability confidence** — the config schema and CLI surface
   should stop moving; recent review cycles changed both repeatedly. A signal:
   several consecutive releases with no breaking config/CLI change.
4. **Multi-platform verification** — macOS (and possibly Windows) actually
   tested rather than "best-effort", since the README ships install
   instructions for them.

