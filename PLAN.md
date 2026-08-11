# cementic — architecture & design

<!-- session-handoff:begin (2026-08-07) -->
## Where the work stands

A second full code review of the CLI and every path it reaches (2026-08-07).
**Nothing was fixed** — the review is read-only and complete. All findings, with
file:line, user-visible symptom, and repro commands, are in
[`notes/code-review-2026-08-07.html`](notes/code-review-2026-08-07.html).
"Open review findings" below carries only the priority order.

**Entry point:** run `uv lock` and commit it. CI is red on `main` right now —
all three jobs install with `uv sync --locked`, and `uv lock --check` fails
because `uv.lock` still lists the removed `pgvector` dependency. One command,
and nothing else can be validated in CI until it lands. Then add a named volume
to both `compose.yml` files (finding B2) before touching anything else.

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

Fix order, highest first:

1. **CI honesty** — `uv lock` (B1), then `pytestmark = pytest.mark.pg` on
   `test_db_isolation`'s fixture test and on `test_smoke` (N17). Until both land,
   a green CI run does not mean the suite ran: the "not pg" job would launch a
   30-60min from-source Postgres build, and the only end-to-end smoke test runs
   in no job at all.
2. **The two data-destruction paths** — a named volume in both `compose.yml`
   files (B2; the Quadlet template already does this correctly), and a guard on
   `init postgres --force` so it cannot `rmtree` an arbitrary directory (N1).
3. **What gets published** — `promote_ready_revision` must require
   `_revision_is_complete`, and `_revision_is_complete` must require
   `documents > 0`. One fix closes three findings: promoting an empty index
   (N2), a restart laundering failures past the gate (N3), and the `ready`
   one-way latch (H2). Then C2.1 (empty extractions counted as success).
4. **Silently wrong search answers** — query truncation (C1.1) and the missing
   `--n_batch` (N5) are one mechanism, fix together; the search SQL's missing
   freshness predicates (C1.2 — reuse the scope builders rather than
   hand-writing them a third time); chunk-boundary U+FFFD corruption (C1.4);
   `ef_search` below `MAX_SEARCH_RESULTS` (C1.3).
5. **Config diagnosability** — C2.2 (misspelled section), C2.3 (missing
   `CEMENTIC_CONFIG`), H9 (misspelled key → raw traceback), N7 (malformed DB
   URL), N9 (doctor passes configs `start` rejects). Together these are why no
   diagnostic can currently diagnose a broken config.
6. **Supply chain / first run** — H1 (`config init` disables the checksum pin via
   a `./models/…` string compare), the undeclared `click` and `pymupdf` imports,
   N10 (download failures as raw tracebacks).
7. **Failure visibility** — C2.5, H3, C2.7, C2.8, N8 (`stop` exits 0 on failure),
   N13 (healthy-looking wrong-model daemon); then C2.4's 120s block, via the
   "busy vs dead" consolidation that is *why* `status`, `--doctor` and `search`
   disagree about what healthy means.
8. **Reporting the lifecycle** — H4 (`ready` shown as `building`) and N4
   (`index.method` silently dead; `collection reindex` from TODO.md is the
   natural trigger).

Dead code remains branch/parameter level — no whole function in `src/` is
unreachable. Both notes list the items; 2026-08-11 adds an unreachable branch in
`extract_pdf_markdown` that would join dict reprs into "extracted text" if it
ever ran, a `distance_metric` parsed but never applied, and two dead DB columns.

### Implementation order for the current pass

Landing 1-3 first (CI honesty, data destruction, publish correctness), each with
tests, then 4 onward. Steps 1 and 2 are mechanical; step 3 is the first one that
changes behaviour users depend on, so it needs regression tests for both the
empty-revision and requeue-laundering paths before the change.

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
