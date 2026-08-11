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

### Still open, in execution order

Ordering is by (user impact × likelihood), with dependencies noted. Each item
names the reviewer's minimal fix surface; detail is in the two notes.

1. **Empty extractions are a silent success** (C2.1). Scanned PDFs with
   `use_ocr` off extract to `""`, which becomes `done` with zero chunks; the
   revision reaches `ready` and promotes with **zero failures reported** and no
   vectors. The `documents > 0` guard does not catch it — the documents exist,
   the chunks do not. Fix: a pure emptiness predicate plus a `raise` inside the
   existing `try` in `_step_extract`, so the existing failure machinery reports
   it and `promote` blocks. Name `extraction.use_ocr` in the message.
2. **Config diagnosability, six steps** (C2.2, C2.3, H9, N7, N9). Land in the
   reviewer's order: `config.py` foundations first (behaviour-neutral), then
   catch-and-format plus the `url_override` validator *together* (the validator
   turns an `ArgumentError` into a `ValidationError`, so it is only survivable
   once the catch exists), then the provider validator, then the file/env
   problem scan, then the three residual traceback sites, then doctor's
   model-path confinement check last (largest test churn).
   **Security constraint:** build messages from pydantic's `loc`/`msg`/`type`
   only. `str(ValidationError)` and `err["input"]` both carry the offending
   value, so a mistyped `[database] passwrd` currently prints the password —
   into terminal scrollback *and* the worker log files the CLI points at. Ship a
   redaction regression test with it.
3. **Worker failures never reach `status`**, paired with the capability gate.
   A retryable provider failure logs only and returns "no work", so the worker
   re-claims the same chunks forever while status reads healthy — the opposite
   of what README promises. Fix by releasing and **re-raising** into the existing
   loop handler. Do *not* write `last_error` directly: `reported_error` would
   stay unset and the recovery path would never clear it, which an existing test
   was written to prevent. Then swap the worker's startup gate from
   `health_check()` (identity only) to `describe()` (a real embed round-trip),
   so a daemon that 500s on every embed fails at startup instead of looping.
4. **Path handling** (H5). No `expanduser` anywhere in `src/`, and a relative
   `artifacts_path` makes `safe_remove_artifact` a silent no-op from any other
   CWD. Single normalisation point already exists in `Config.__init__`; use
   `expanduser` + `Path.cwd() / p`, not `.resolve()`, or the log-path tests
   break on symlinked tmp dirs. **Do this before pruning** — pruning cannot
   reclaim disk while artifact deletion silently fails.
5. **The "busy vs dead" triplicate** (C2.4, C2.8, N13). Three probes with three
   budgets is why `status`, `--doctor` and `search` disagree. `status` can block
   ~129s and then discard the answer; a wrong-model daemon is reported healthy.
   Fix: expose a tri-state probe (`healthy` / `wrong model` / `no response`),
   one helper with an explicit time budget, and let each call site pick. Deletes
   `client_is_healthy_or_busy` and `doctor._daemon_reachable`, and takes the
   redundant per-search probe with it for free.
6. **Watcher bundle** (H7, N11, plus uncounted skips and `os.walk`). Ancestor
   matching is the damaging one: watch a directory under any ignored name and
   the initial scan indexes everything, then every live event is dropped
   forever. Needs the event handler constructed *after* `_watched_roots`.
7. **`load_file_progress` contradicts its own summary** (H11). Reuse the ORM
   scope builders — but take only the two `ChunkedDocument` predicates, not
   `chunked_scope` wholesale, or failed extractions vanish from the outer join.
8. **Vector-dimension pre-flight** (N14). pgvector's HNSW caps at 2000
   dimensions while we allow 8192, so a 2560/4096-dim model embeds the whole
   corpus and *then* loops forever failing index creation. Add
   `max_indexable_dim(method)` beside the index registry and check it before
   embedding starts.
9. **Record the task-prefix policy in the embedding profile** (README:284).
   Renaming the GGUF silently switches to `plain` — and because the policy is
   not in the payload, prefixed and unprefixed corpora share one profile and one
   vector table, mixing incompatible vector spaces. Own commit: it invalidates
   stored fingerprints and forces a rebuild.
10. **Pruning leaks** (N16) — orphaned vector tables and stale
    `chunk_embeddings` after a model swap. Two footguns: embedding profiles are
    shared across collections (re-query globally, as `collection remove` does),
    and `DROP TABLE` must run after `session.commit()`.
11. Opportunistic, small: mixed-model search message naming the culprit
    collections; `embedding stop` reporting success after a failed kill;
    redirecting pymupdf4llm's OCR notice off stdout (it lands in the
    `extract | chunk` pipe); `[extraction.backends]` key normalisation and a
    "no such extractor" message.

**`hnsw.iterative_scan` is now more urgent, not less.** The freshness predicates
added in the last batch post-filter *more* rows, so a scan yielding `ef_search`
candidates discards more of them and can return fewer than `top_k`. It still
needs verification against the live database first — a pgvector older than 0.8
rejects the parameter and would break all search.

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
