# cementic — architecture & design

<!-- session-handoff:begin (2026-08-24b) -->
## Where the work stands

**Repo state.** On `claude/igpu-findings`, branched from `main`. `main` carries
everything from earlier today (soak measurements, doc de-staling, the
`pymupdf-raw` extractor) and is **6 commits ahead of `origin/main`, unpushed**.
v0.2.0 is still the last tag. Pushing `main` and merging this branch are the
user's calls: a hook rejects agents touching `main` at all, so commit on a
`claude/*` branch (`AGENT_BRANCH_PREFIX=claude`) and hand over the merge.

**Entry point: the bulk import is designed and unblocked; what remains is
running it.** The corpus is real and surveyed: `~/OneDrive/Material/Papers`,
22,246 PDFs, 43.8 GiB, 592,248 pages, **99.9% PDF** so no new extractor is
needed. Projected ~2.16M chunks. The plan is in README's "Measurements behind
the defaults" (scale-target table) and the GPU recipe is
`~/.cache/cementic-igpu/RUNBOOK.md`. Next concrete action: a ~200-document smoke
test with the Vulkan server, to confirm the measured 6.5× survives cementic's
serving path, then the full import.

**Set these before indexing — each re-versions the corpus if changed after.**
```
extraction.backends.pdf     = "pymupdf-raw"     # 128 h -> 0.5 h on this corpus
llama_cpp.n_gpu_layers      = 99
llama_cpp.daemon_autostart  = false             # an external llama-server serves instead
index.method                = "hnsw"            # keep; see below
```
Keep HNSW for the initial build even though ~2.16M × 768 needs ~6.5 GB against
~5 GB free: HNSW is maintained per insert and so resumable across a multi-day
job, where DiskANN builds in one unresumable pass at the ready transition.
`collection reindex` switches method later without re-embedding.

**The load-bearing numbers a cold reader should not re-derive.**
- Embedding on this hardware: pure CPU 382 tok/s, iGPU 2,468 tok/s (6.5×), but
  cementic's HTTP path runs ~26% under raw, so derate. Projected ~100–160 h.
- Extraction scales with **pages, not documents**: 2.8 ms/page raw,
  0.78 s/page with pymupdf4llm.
- `vectorscale 0.9.0` and `vector 0.8.3` are installed; both index methods work.

**Corrections — three things measured wrong earlier today, all now fixed in the
docs. Distrust the reasoning style that produced them, not just the numbers.**
- *"The iGPU is only 1.22×, not worth it."* Wrong: `-ngl 0` is not a CPU
  baseline, because llama.cpp offloads big matmuls to any visible GPU by default.
  Only `-dev none` measures CPU. The real figure is 6.5×.
- *"Thread pinning is worth ~41%."* Noise. Re-running an identical config moved
  53%. This machine's spread swamps differences under ~1.5×; interleave arms in
  one invocation and never compare across runs.
- *FLOP arithmetic assuming a 137M model.* It is `nomic-bert-moe`, 475M.
- Also: the agent sandbox has **no `/dev/dri`**, so any GPU work must be run by
  the user. Its CPU numbers are trustworthy (382 pure-CPU matched the user's).

**Live state.** Nothing running; workers stopped, no external server, port 11555
free. Collections `soak` (153 docs, revision 6 active) and `test` (5 docs,
revision 5) both searchable and sharing vector table `embedding_vectors_p5`.
`test` is a testbed — remove and rebuild freely. Note the `pymupdf-raw` commit
re-versioned every extractor profile (`05e4c8a0` → `d18c0a0e`), so the next
`cementic start` on either collection rebuilds it.

**Not in git:** `~/.cache/cementic-igpu/` (upstream llama.cpp b10605 Vulkan
build, benchmark scripts, RUNBOOK.md), `~/.cache/cementic-corpus/survey.py` and
its `survey.json`, `~/.cache/cementic-ab/` (the extractor A/B harness).

**Deviations, attributed.** *Agent-decided:* used a prebuilt upstream llama.cpp
release to measure the GPU rather than building llama-cpp-python with Vulkan —
`glslc` is unavailable and measuring first was the cheaper order. *Blocked:* the
branch-prefix hook rejects `git clone` (it creates a `main` ref); not worked
around, since routing past a guard needs the user's say-so.

**Environment quirks — still true.**
- The pg fixture short-circuits when Postgres is reachable, so no local run
  exercises `compose build`; CI's `integration-pg` job proves that path.
- On a fresh connection `SHOW hnsw.ef_search` errors and `SET hnsw.ef_search` is
  accepted as an inert placeholder until a vector query loads pgvector's module.
- Green gates are not sufficient evidence: end changes touching workers, the
  daemon, or profiles with a live run, not just `check.sh`.
- **The real corpus has filenames containing shell metacharacters** — quotes at
  minimum; a plain `find ... | xargs` over it dies with "unmatched single
  quote". Any shell tooling written against
  `/u/71/ibrahin1/data/Documents/Papers` needs `find -print0 | xargs -0`.
  cementic itself is unaffected and this was checked, not assumed: there is no
  `shell=True`, `os.system` or `os.popen` anywhere in `src/`, and
  `spawn_detached` (`supervisor.py:229`) hands `subprocess.Popen` a `list[str]`
  with watched directories passed after `--` (`cli.py:475`), so no path is ever
  parsed by a shell.
- The corpus path is two symlinks deep: `~/OneDrive/Material/Papers` →
  `~/Documents/Papers` → `/u/71/ibrahin1/data/Documents/Papers`. It is local
  ext4 on the same filesystem as `$HOME` (44 GiB, 88 GiB free), **not** a
  network or cloud-placeholder mount — 50 PDFs read in 0.245 s. `du` without
  `-L` reports 36 bytes and looks empty, which is the symlink, not the corpus.
  Index the resolved path so a moved symlink cannot orphan every `source_path`.

**Exit criteria — commands whose output confirms the above.**
```bash
git status --short                        # empty
cementic status -c soak                   # workers stopped, 153 docs, 100%
cementic search "multiple kernel learning" -c soak -n 3   # 3 topical hits
./scripts/check.sh                        # six gates, exit 0 (~4 min)
```
<!-- session-handoff:end -->

## Plan of record — index-driven embedding claims (2026-08-24)

**Why.** `_step_embed` finds work with an anti-join (`ce.id IS NULL`) that
cannot be indexed, and pays for it once per 32-chunk batch. Measured on this
database:

| case | buffers to claim one batch |
|---|---|
| `smoke`, 19,400 candidates, real query | 178,681 (165 ms) |
| same, with the profile-id subselects removed | 2,400 |
| `soak`, 7,306 chunks, all embedded — returns 0 rows | 64,536 |
| proposed shape, index scan on `(embedding_profile_id, status)` | **5 (0.070 ms)** |

Two separate faults. The profile-id subselects make the planner estimate 29
rows where 19,400 match, so it chooses seq-scan-plus-sort and `LIMIT 32` cannot
terminate early. Underneath that, the scan walks the completed prefix to reach
the remaining tail, so cost grows with *corpus size* rather than remaining work
— the `soak` row is the shape of the end of a run, ~8.8 buffers per corpus
chunk per batch. Extrapolated to 2.16M chunks that is ~19M buffers per batch
late in the run, against 128 MB `shared_buffers`. The 300× extrapolation is
arithmetic, not measurement; the scale test below is what turns it into one.

**Design.** Materialise instead of anti-join.

1. Every in-scope chunk gets a `pending` `ChunkEmbedding` row from one
   idempotent set-based statement — `INSERT ... SELECT ... ON CONFLICT
   (chunk_id, embedding_profile_id) DO NOTHING`, riding the existing unique
   index — rather than being created one at a time inside the claim.
2. `_step_embed` then claims with `WHERE embedding_profile_id = :p AND status
   IN ('pending','processing') ORDER BY id LIMIT :n`, which is an index scan on
   `ix_chunk_embeddings_profile_status`. No join, no anti-join, flat cost.
3. The materialise step runs when work can have appeared: at worker start
   (covers a resumed run and a new embedding profile, whose chunks all predate
   it) and after any chunk write-back. An in-process flag gates it so an idle
   worker does not re-run an O(n) statement every poll.

**Revised 2026-08-24 after the scale test falsified the first design.** The
original plan claimed no schema change was needed, on the strength of a 5-buffer
measurement at 19,400 chunks. At 300,000 chunks the planner flips: with the
scope filters on joined tables it drives from `source_documents`, checks
embedding status last, and walks the finished prefix — 1,895,124 buffers at 90%
embedded, growing 12.5x from the 10% mark. Materialising the rows was necessary
but not sufficient.

The fix is the one README already documents for the vector tables: put the
filters on the row being scanned. `chunk_embeddings` gains `collection`,
`extractor_profile_id` and `chunk_profile_id` plus a covering index, and the
claim becomes single-table. Re-measured at 300,000 chunks: 17 / 806 / 1,997
buffers at 10 / 50 / 90% embedded, with the driving scan reading exactly
`batch_size` rows every time. Worst case ~18 ms, or about 20 minutes of claim
time across the whole 22k-document import against ~170 hours before.

Safe to denormalise because none of the three ever changes for a row: a chunk
cannot move collection, and a deleted document's chunks — and these rows, by
cascade — are removed outright rather than filtered at claim time, which is the
same reasoning `_purge_document_chunks` already records for vectors.

**So there is a migration**, though not a new mechanism:
`ensure_chunk_embedding_filter_columns` is idempotent, adds the columns,
backfills them with the join it exists to remove, and mirrors
`ensure_vector_table_schema`. Rollback is still a plain revert — the columns are
nullable and the old claim query ignores them.

**Rejected alternatives.** Fixing only the planner misestimate (a query-shape
change, much safer) halves the early-run cost and leaves the quadratic tail
untouched — it treats the symptom that is cheap to see and not the one that
ends the run. A partial index cannot express "has no row for this profile".
A high-water-mark on chunk id skips rows that later return to `pending`, which
is precisely how a retried chunk gets silently dropped.

**Work items, in order.**

1. `materialize_pending_embeddings(session, revision) -> int` in `revisions.py`,
   scoped by `chunk_scope(revision)` + the revision's embedding profile.
   Verified by a pg test: run twice, assert the second inserts 0 and the total
   equals `total_chunks`.
2. Replace the claim in `_step_embed`, preserving the deliberate re-claim of
   `processing` rows and the batch semantics around it.
3. Wire the trigger: call after `requeue_interrupted_artifacts` at start, set a
   flag on a successful chunk write-back, consume it in `_step_embed`.
4. **Scale test** — the check that would actually catch a regression here, and
   which this codebase has never had: build a synthetic collection of a few
   hundred thousand chunk rows, measure claim cost at 10%, 50% and 90%
   complete, and assert it does not grow with the completed fraction.
5. Record the measured before/after in README's "Measurements behind the
   defaults".

**Risks.** The claim path is where two stalls came from today, so item 2 is the
one to review hardest. Correctness rests on one fact worth stating plainly:
one worker per collection holds the advisory lock, so a `processing` row can be
re-claimed without a second worker racing it — the same assumption the current
code already documents. Chunk deletion and re-chunking need no new cleanup;
`chunk_id` carries `ondelete="CASCADE"`, so purged chunks take their pending
rows with them.

**Falsification.** If the scale test shows claim cost rising with the completed
fraction, the design is wrong and the work stops there rather than shipping on
the strength of the 5-buffer measurement above.

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

## Design decisions and open items

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

