# cementic — architecture & design

<!-- session-handoff:begin (2026-08-24) -->
## Where the work stands

**Repo state.** On `claude/soak-measurements`, branched from `main`. `main`
itself is 1 commit ahead of `origin/main` and unpushed — the previous session's
doc-drain work, which the user merged. v0.2.0 is still the last tag, 35 commits
behind. Pushing `main` and merging this branch are both the user's calls: a hook
rejects agents touching `main` at all.

*Committing here:* an agent must be on a `claude/*` branch (`AGENT_BRANCH_PREFIX=claude`).
Branch first, then hand the merge to the user.

**Entry point: the soak is DONE and its numbers are in README. The open thread
is what to do about indexing speed.** Collection `soak` — 153 papers, 178 MB,
7,306 chunks — indexed clean (zero failures), was promoted, and searches
correctly. Everything measured is written up in README's "Measurements behind
the defaults"; the follow-on work is `TODO.md`'s "Cut indexing wall clock at
corpus scale" under Later. Nothing is running and nothing is half-finished.

**What the soak established.** Full tables are in README; the load-bearing
numbers, because a cold reader should not have to re-derive which constraint
binds:

- Indexing: 10.2 s/doc extract, 814 ms/chunk embed, 47.8 chunks/paper, 2.08 h
  total. Projects to ~113 h extract + ~432 h embed at the 40k-paper target.
- **The stages do not overlap** — all extraction finishes before the first
  chunk, all chunking before the first embedding — so those two projections add
  rather than hide under each other. This is the finding with design
  consequences, and it is why the throughput item moved off the parked list.
- Search: 38.7 ms warm in-process, of which 35.3 ms is embedding the query and
  2.6 ms is pgvector. Embedding-bound, not index-bound. The CLI's ~650 ms is
  almost entirely Python import startup (`cementic --help` alone is 0.38 s,
  SQLAlchemy pulled in via `cementic.render` → `cementic.collections`).
- Recall@10 vs exact: mean 0.990, worst 0.900, 18/20 identical. `promote` took
  0.7 s with no bulk index build, which is the up-front-index decision paying
  off at corpus scale.

**Corrections this session.**
- README's `1.4 s/chunk`, `780 h`, `50 chunks/paper` and `197 ms` query-embedding
  figures were all superseded by the run above. The 197 → 35 ms embedding gap is
  **unexplained** — the old measurement's conditions are unrecorded, so README
  marks it superseded rather than reconciled. Distrust other single-number
  claims of that vintage.
- The previous handoff's "extraction is 15.3 s/doc → ~170 h" came from an
  18-document sample and was pessimistic; the full run gives 10.2 s/doc → ~113 h.
- `TODO.md` listed `collection reindex --force` as untested through the CLI.
  It was closed on 2026-08-23 by `2a2c856`, whose test docstring says so
  outright (`tests/unit/test_cli.py:1366`). Removed.
- The sibling coverage gap in that same list — no test connecting a pipeline
  failure to what `cementic status` prints — was re-checked and **is still
  real**. Left in place.

**Live state.** No workers running. The embedding daemon is **up and warm**
(autostarted by the first search, ~1.1 GB resident); `cementic embedding stop`
if you want the memory back. Collections: `soak` 153 docs, revision 6 active;
`test` 5 docs, revision 5 active — `test` is a **testbed** (user, 2026-08-23),
remove and rebuild it freely. Both share vector table `embedding_vectors_p5`
(7,580 rows) because they share an embedding profile.

**Deviations from plan, attributed.** *Agent-decided:* the README write-up
covers the whole soak, including the indexing-throughput table, though the user
had approved only "promote and run the search measurements" — same write-up, and
flagged to them at the time rather than done quietly.

**Environment quirks — carried forward, still true.**
- The pg integration fixture short-circuits when Postgres is already reachable,
  so **no local run exercises `compose build`** — CI's `integration-pg` job is
  what proves that path.
- A zombie process keeps its `/proc/<pid>` directory, so `test -d /proc/<pid>`
  is not a liveness check; read the state field.
- `notes/review-codebase.html` (the sixth-review audit) was written after
  `notes/` was gitignored, so it is in no git history — it exists only on disk.
- **Green gates are not sufficient evidence here.** Five defects in the
  2026-08-23 session were found by running the CLI against the live corpus while
  all six gates passed. End changes that touch workers, the daemon, or profiles
  with a live run, not just `check.sh`.

**Found this session, worth knowing before you touch pgvector tuning:** on a
fresh connection `SHOW hnsw.ef_search` errors as an unrecognized parameter and
`SET hnsw.ef_search` is accepted as an inert placeholder, until some vector
query loads pgvector's module on that connection. Run a throwaway vector query
first or your tuning silently does nothing. This is the same trap
`_iterative_scan_mode`'s docstring documents, met from the other side.

**Exit criteria — commands whose output confirms the above.**
```bash
git status --short                            # empty
git branch --show-current                     # claude/soak-measurements
cementic status -c soak                       # workers stopped, 153 docs, 100% all stages
cementic collection list                      # soak + test, both with an active revision
cementic search "multiple kernel learning" -c soak -n 3   # 3 topically correct hits
./scripts/check.sh                            # all six gates, exit 0 (~4 min)
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

The target is a large personal corpus (~40k papers and textbooks) at a measured
47.8 chunks per paper — roughly 1.9M vectors. The numbers behind that (query
latency, embedding and extraction throughput, HNSW memory) are in README's
"Measurements behind the defaults", re-measured 2026-08-24 on a 153-paper run;
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

