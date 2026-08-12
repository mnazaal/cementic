# cementic — architecture & design

<!-- session-handoff:begin (2026-08-07) -->
## Where the work stands

A second full code review of the CLI and every path it reaches (2026-08-07).
**Nothing was fixed** — the review is read-only and complete. All findings, with
file:line, user-visible symptom, and repro commands, are in
[`notes/code-review-2026-08-07.html`](notes/code-review-2026-08-07.html).
"Open review findings" below carries only the priority order.

**Entry point:** ~~run `uv lock`~~ — done on 2026-08-11 along with the named
volume in both `compose.yml` files, on `claude/review-fixes-2026-08-11`
(unmerged). `uv lock --check` now passes. The next entry point is the config
diagnosability cluster; see "Still open" below.

**Branch:** `claude/code-review-2026-08-07`, **two commits ahead of `main`, not
merged and not pushed** — `b39f6f6` (the review note) and the commit carrying
this block and the section below. Working tree clean; `git merge-tree` reports
no conflicts against `main`, so it fast-forwards.

`claude/hygiene-fixes` still exists but is **fully merged** — the previous
handoff block claimed 5 unmerged commits (`27e5ab0`, `6d56d9a`, `e93d79c`,
`9d50d48`, `cb5cace`); `git merge-base --is-ancestor` confirms all five are on
`main`. That branch can be deleted.

**Verification state:** ruff clean, mypy clean, **610 unit tests pass in 16s**
locally. Every finding in the note is something the suite does not cover. CI has
never been green on this state (see entry point).

**Live database state:** the real `cementic` DB holds collection `test`, 5
documents, 172/172 chunks embedded, revision 1 in `ready` with **no active
revision**. Both worker state files read `process=stopped, state=running`; that
is the known state-file bug (the pipeline worker writes `RUNNING` outside the
`try/finally` that would clear it), not a live process. No cleanup needed.

### Environment facts that cost time to rediscover

- **The venv is an editable install of this working tree.** Running `cementic`
  executes whatever branch is checked out — not `main`. Check
  `git branch --show-current` before interpreting CLI behaviour.
- **Integration tests use a separate `<name>_test` database** and refuse to run
  otherwise (`tests/integration/conftest.py`). Before that guard existed they
  dropped every table in the user's real database; `pytest tests/unit` never
  touches Postgres.
- Postgres answers on `localhost:5432`; the container engine is *not* reachable
  from an agent sandbox, and the sandbox is in its own PID namespace, so `ps`
  cannot see the user's worker processes. Ask the user to run process checks.
- The embedding daemon was up and healthy during the review, so DB-backed and
  embedding-backed commands are exercisable directly.
- Commits must be on a `claude/*` branch (`AGENT_BRANCH_PREFIX`), and commit
  messages containing dependency-directory names can trip a path guard — write
  the message to a file and use `git commit -F`.
- Shell cwd resets to the repo root between tool calls; redirect probe output to
  an absolute scratch path or it lands in the repo.

**Not carried forward:** the previous entry point (`cementic collection promote
test` then `search -c test`) is still unexercised. Running it now would only
demonstrate two known bugs rather than confirm health — `collection list` labels
that `ready` revision `building`, and promoting a `ready` revision that has since
absorbed new work publishes it without re-checking completeness. Do it once both
have landed.
<!-- session-handoff:end -->

Design rationale and roadmap for cementic: a CLI that watches directories of
documents, builds a versioned **extract → chunk → embed** pipeline in Postgres
(pgvector / vectorscale), and serves semantic search over the active revision of
each collection. User-facing docs live in `README.md`; this document is for
contributors.

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

The target is a large personal corpus (~40k papers and textbooks). The machinery
is justified rather than over-engineered:

- "Indexed" and "fast" are not in tension — indexing is the amortized one-time
  cost; an ANN index keeps *queries* sub-second over millions of vectors.
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

## Open review findings

Two read-only reviews, both still open against an unchanged `src/`:
[`notes/code-review-2026-08-07.html`](notes/code-review-2026-08-07.html) (first
full pass) and
[`notes/code-review-2026-08-11.html`](notes/code-review-2026-08-11.html) (third
pass — five unprimed reviewers, so its overlaps are independent re-derivations,
and it carries only what is *new* plus the re-confirmation map). File:line,
symptom and repro live in the notes; this section carries only the fix order.

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
   builders as `CURRENT_CONTENT_SQL`), C1.1 + N5 (query bounded in tokens
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
  functions set those on the *failure* path too (`pipeline_worker.py:494`,
  `:606`), so the fixtures — not the new predicates — were wrong. Check that
  before assuming a similar failure means a regression.
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

### Still open

**The ANN index is never used by cementic's search query.** Measured, not
inferred, with `EXPLAIN (ANALYZE)` against a real corpus:

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
  linearly. Fine at the current 172 chunks; ~25 ms at 20k, ~90 ms at 60k.
- `index.method`, `hnsw_m`, `ef_construction`, `hnsw_ef_search`, the DiskANN
  knobs and `collection reindex` all maintain an index nothing reads.

**`hnsw.iterative_scan` was investigated and deliberately not landed.** The
premise — that post-filtering an ANN scan under-returns — is true in principle
and measurable on a single-table query (0 of 10 results with the index forced),
but it cannot occur here while the planner never chooses the index. It becomes
required the moment the query is restructured, and should land with that change
rather than as a knob that does nothing. Two findings from that work are worth
keeping:

- pgvector here is 0.8.3, which supports it; the parameter needs a version gate
  regardless. PostgreSQL accepts an unknown *qualified* setting as a
  placeholder until the defining module loads on that connection and rejects it
  with `InvalidName` afterwards — so behind a connection pool an ungated `SET`
  fails only on connections that had already run a vector query.
- `SET LOCAL diskann.query_rescore` is accepted even where pgvectorscale is
  absent, for the same placeholder reason. It is not evidence the setting took
  effect.

The fix, if taken: denormalise the filter columns onto the vector table
(collection, chunk/extractor profile ids, a currency flag) so the WHERE clause
applies to `embedding_vectors_p*` directly and the planner can drive from the
ANN index — landing `hnsw.iterative_scan` at the same time, since post-filtering
then becomes real. That is a schema change and a design decision, not a bug fix.

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

Two of those were re-derived independently by the second review with new
evidence, and are worth reopening rather than re-closing: the `check_health`
tradeoff also makes `cementic status` block for 120s to return an answer it
already had (C2.4), and the filename heuristic silently disables Nomic task
prefixes while `models/nomic-embed-text-v1.5.f16.gguf` sits in the repo.

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
  open for another reason.
- **`ignore_directories` replacing the defaults.** Working as documented,
  including the "empty list indexes everything" escape hatch a union would
  break. If discoverability is the concern, add a separate
  `additional_ignore_directories` — a feature decision, not a defect fix.
- **The advisory-lock leak and the `RUNNING` state-write ordering.** Every path
  that leaks the lock is immediately followed by process exit, which releases
  it; the stale state file is corrected by `start`'s liveness check and cleared
  by `stop`. Two-line fixes if those functions are open anyway, not worth a slot.
- **`index.method` being unreachable on a built system.** A dead knob with no
  correctness consequence, because search deliberately tunes for the index that
  exists. The actionable part is that `TODO.md` asserts "the build path already
  reconciles a changed method", which is false — correct that sentence and leave
  `collection reindex` on the roadmap.

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
- CI: unit tests always; PG integration tests when a container engine is
  available (the integration suite already brings compose up/down itself).
- Hybrid lexical + vector search (exact author names, acronyms, equation labels).
- A multi-profile embedding daemon pool, if old-model search and new-model
  indexing must run concurrently.
- Lighter embedding providers if llama.cpp memory use is too high on small
  machines.
