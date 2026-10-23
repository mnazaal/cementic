# cementic — architecture & design

<!-- session-handoff:begin (2026-10-23) -->
## Where the work stands

**Entry point: "Execution order (2026-10-22)" → "Waves (2026-10-23)" → Wave
B.** Wave A and Wave R passed the user's gate on 2026-10-23 (`claude/wave-a`,
fast-forwarded to main once the user runs it); Wave D is empty because the
riders failed their rule. Wave B is on `claude/wave-b`, rebased onto
`claude/wave-a`, pending review and its gate.

**Phase 2a closed 2026-10-23: stay on v2-moe.** Qwen3 and Harrier pass the
title rule and fail the phrase rule; see "Retrieval Phase 2" → 2a → "RUN
2026-10-23". 2b-2e are now judged alone. The noise-floor anomaly is resolved
as a scoring defect (2a → "Noise-floor anomaly").

**Open from review, accepted.** A file renamed while its extraction is in
flight fails with FileNotFoundError and stays failed until the next `cementic
start`. The window is one document per worker.

**Environment quirks.**
- Worktrees work from the sandbox (2026-10-23): `git worktree add
  "$TMPDIR/wt-<topic>"`, then `uv sync --extra dev --offline` there (PyPI is
  blocked; the uv cache covers the lock). An earlier session could not write
  `.git/worktrees`; if that recurs, ship a patch under `~/.cache/patches/`.
- Claude Code kills its own background jobs under memory pressure. Long runs
  belong in the user's terminal.
- `git diff`/`git status` with no pathspec fail in the sandbox on the masked
  `.env`. Pass a pathspec. Untracked dotfiles (`.bashrc`, `.claude/`, ...) in
  sandbox `git status` are bind-mount stubs; never stage them.
- The sandbox cannot see systemd (`Failed to connect to bus`) or the real
  daemon; `cementic status` there reports everything stopped. Ask the user
  for `status`/`doctor` output.
- Bash hooks block `git config`, branch deletion (the pattern matches even
  inside a heredoc) and `ls` of the session scratchpad.
- In the sandbox, run the suite with
  `TIKTOKEN_CACHE_DIR=$HOME/.local/share/cementic/tiktoken`. The non-pg suite
  then fails exactly 3 tests, `test_embedding_runtime.py`
  `test_get_runtime_client_*`, on the blocked HuggingFace download (88 s,
  2026-10-23). Without the variable about 60 fail on tiktoken. pg tests cannot
  reach 5432; the user's gate is the authority.
- `llama-bench -p 2048` aborts on encoder models whose window is at or below
  2048; bench them at 512.

**Artifacts outside the repo.** Audit: `notes/review-codebase.html`
(gitignored). The six subagent audit reports behind it and the branch
review: `~/.cache/cementic-audit/reviews-2026-10-22/`. Gate logs and
baselines: `~/.cache/cementic-audit/`. Model comparison:
`~/.cache/cementic-model-compare/` and `~/.cache/cementic-model-compare-all.log`.

**Exit criteria.**
```bash
git -C ~/projects/cementic log --oneline -1 main      # eae6f16 or later
git -C ~/projects/cementic worktree list              # only ~/projects/cementic
cementic status                                        # workers running, 100.0% embedded
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
  obligation, the 3.12 floor — moved to README's "Decisions worth knowing", which `6df8ab9` later removed
  (`git show 6df8ab9^:README.md`, line 907).
- **Scheduling** — why Batch C ran before Batch A, why a characterization test
  preceded the refactor — deliberately dropped. They were about sequencing one
  piece of work and answer nothing six months out.

The two parked items that had revisit conditions moved to `TODO.md`'s "Parked,
with a trigger", because nothing here was watching for them.

### 2026-10-22 — decisions from the audit interview

Each entry is a question a later reader would ask. All are **live**. Each names
the execution-order step that carries it out.

- **Why does one execution order hold the audit, the CLI audit and retrieval?**
  Three live orders gave no single answer to "what next", and they interact.
  The worker fixes should land before the Phase 2 rebuild re-extracts the
  corpus. The lexical-arm change has to wait for 2a so it doesn't confound the
  model comparison. And the systemd copy wants the `cementic run` group from the
  CLI audit. *Rejected:* two cross-linked tracks, which is two live plans; and
  parking CLI steps 2-6, which drops `cementic run` and the search document id.
- **Why both an attempt cap and out-of-process extraction for poison PDFs?** A
  cap stops the crash loop cheaply but cannot see a hang, since a hang never
  crashes. Only a child process with a timeout and an rlimit catches hangs, and
  it also closes the hostile-PDF exposure. *Rejected:* cap only, which leaves
  hangs and the security item; isolation only, which still loops on a crash
  outside extraction.
- **Why does the repo ship a copy of the live systemd unit instead of owning
  it?** User's call: dotfiles stay untouched. *Accepted cost:* two copies that
  can drift. *Rejected:* the repo owns the unit and dotfiles symlink to it
  (couples dotfiles to the checkout); deleting `packaging/systemd` (users write
  their own). *Revisit when:* the two copies differ in anything other than
  paths.
- **Why only a docs warning about the default Postgres password?** It is a
  single-user tool, and generating a password per install couples
  `init postgres` to the user config and needs a migration note. *Revisit
  when:* cementic is documented for shared machines.
- **Why pin tiktoken's cache directory rather than vendor the encoding or count
  with the server?** Pinning is small and survives reboots. Vendoring adds about
  1.7 MB of an OpenAI-hosted file with unclear redistribution terms. Counting
  with the server would make `chunk` need a running server and would move every
  chunk boundary, forcing a full re-chunk. *Accepted cost:* the first run after
  install still needs the network once.
- **Why do `status` and `collection list` stay separate after the audit flagged
  the overlap?** The 2026-09-10 measurement still holds: `collection list` is
  about 2x cheaper because it skips the worker and daemon probes. Reaffirmed,
  not reopened.
- **Why does `llama_cpp.verbose` leave the runtime fingerprint?** This reverses
  the 2026-08-23 line under "Deliberately not done" that said it "correctly
  stays in the runtime fingerprint". The audit found the cost that line missed.
  Two processes whose configs differ only in verbosity compute different aliases
  and kill each other's daemon on every search. Verbosity stays a launch
  argument, so a reused daemon just keeps the log level it started with.
  Agent-decided during the plan write-up, not asked in the interview.
- **Why tag 0.3.0?** HEAD is 144 commits past `v0.2.0` and `uv tool` installs
  from HEAD, so the version string currently means nothing. CHANGELOG keeps its
  tag-time policy.

### 2026-10-23 — decisions from the next-steps interview

All **live**. Each names the plan section that carries it out.

- **Why is the Phase 2 rebuild gated on a measurement now?** With 2a closed on
  v2-moe, no model change rides the rebuild, so its ~108 h buys only the
  riders 2b, 2c and 2e, whose effect nobody has measured. The riders are scored
  together as one harness arm first ("Retrieval Phase 2" → "Rider
  measurement"). *Rejected:* rebuilding unmeasured (108 h on an unknown gain);
  dropping the rebuild (de-hyphenation and boundaries would never reach the
  existing corpus); per-rider arms (about 3x the harness time, and the riders
  ship together anyway).
- **Why is the rider arm's control fresh v2-moe, not the stored vectors?**
  *Reason superseded the same day by the entry below; the choice stands.*
  Agent correction to the interview question, which said "stored". On sample
  `8d3c95f989f5`, fresh v2-moe at 320 beat its own stored vectors on 99 phrase
  queries and lost 9 (p = 2.6e-20), so any freshly embedded arm beats stored on
  re-embedding drift alone. Fresh v2-moe at 320 without riders isolates the
  riders.
- **Correction: the 99/9 was a scoring defect, not drift.** `score_arm`
  dropped the 21 chunks fresh v2-moe refused (dot-leader tables of contents,
  hub vectors that outrank the gold) from the fresh arm only. Scored on the
  pool every arm embedded, fresh against stored is 20/15, p = 0.50, and every
  embedded chunk has cosine >= 0.9997 between the two. The fresh control still
  stands, for a different reason: the riders arm is a fresh embedding of
  re-chunked text, and the fresh arm is the same pipeline with the riders
  switched off.
- **Why does 0.3.0 wait for everything?** User's call: one release after the
  whole execution order, including the rebuild if the riders pass. *Rejected:*
  releasing after the CLI track, which would ship the output-contract changes
  sooner. *Accepted cost:* the breaking output changes sit unreleased on main
  for weeks.
- **Why is fuzzy matching scheduled?** A one-character typo returns zero
  lexical rows and recall@10 0.233, and that is the case a person searching
  their own library from memory hits most. It gets a measurement, then a
  choice or a recorded floor. *Rejected:* closing at 0.233 unmeasured; leaving
  it unowned.
- **Why does the user run the gate after every wave?** The sandbox cannot
  reach PostgreSQL, and the schema retrofits and the embed planner are pg-only
  behaviour. One `check.sh` per wave catches a pg regression inside the wave
  that caused it. *Rejected:* one gate at the end, which would mean bisecting
  across all waves.

### 2026-10-23 — decisions from the Wave A gate

All **live**.

- **Why does extraction isolation ship on a missed wall-clock rule?** User's
  call. The 200-PDF ratio was 1.0535 against a pre-registered 1.05; the
  rule's own fallback (one long-lived child) was already in. The overhead is
  ~4.5 ms per document, against seconds of embedding per document in a real
  run, and a colder pass measured 1.019. *Rejected:* best of four repeats (the
  bar already sat between two runs); isolation off by default (gives up the
  memory limit and timeout on untrusted PDFs that the rule was pricing).
  *Recorded as a miss*, not a pass ("Extraction isolation").
- **Why no full fuzzy run?** The trigram arm is an unindexed scan at ~879 s
  per query, unshippable whatever its recall, and the run would take ~14.6 h.
  Agent-decided: measure relaxed OR alone, and keep trigram open only as an
  indexed candidate ("Fuzzy matching").

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
- **Pipeline stages are also filters.** `extract | chunk | embed` are
  stdin/stdout filters, so a single document can be run end-to-end with no
  database. **Narrowed 2026-09-10** from "Unix composability": the chain has no
  exit — nothing ingests embedded JSONL and `search` always derives its own
  vector — and it cannot get one, because a cementic vector is only meaningful
  inside a revision that pins three profiles and their fingerprints. These are
  debugging and export tools, not a composable loop. See "Decided — the CLI surface after the `llm`
  audit".

## Scale context

The target is no longer an estimate: the corpus was surveyed on 2026-08-24 at
22,246 PDFs / 592,248 pages, which at 3.65 chunks/page is **~2.16M vectors**.
The numbers behind that (query latency, embedding and extraction throughput,
GPU rates, HNSW memory) were in README's "Measurements behind the defaults" until `6df8ab9`
removed it (`git show 6df8ab9^:README.md`);
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

Their JSONL is not the stored shape, so do not read it as a picture of the
database: the filter emits `index` where `chunks_v2` stores `chunk_index`, it
carries no document identity, and it keeps the whitespace chunks the worker
drops and renumbers. The CLI step "Rename the filter's chunk index key" closes
the key-name half of that.

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

## Execution order (2026-10-22)

The one live plan. It merges the 2026-10-22 audit (`notes/review-codebase.html`,
local only), steps 2-6 of the 2026-09-10 CLI-surface audit, and
retrieval-quality Phase 2. Why one order is in the decision log. Each step names
what ends it. Line numbers are as of `73b5d4c`.

**Ordering is load-bearing:**
- **Correctness comes first, and the search projection fix comes first within it.**
  It is a one-line fix for a defect every multi-hit search shows.
- **Worker changes land before the Phase 2 rebuild.** These are the attempt cap,
  the `last_error` clear, rename as re-path, the schema retrofits, set-based
  embed write-back, and extraction isolation. The rebuild re-extracts all 23k
  PDFs, which is when a poison PDF is likeliest to surface, and set-based
  write-back shortens it.
- **Worker changes need a quiet moment.** The unit runs the working tree, so
  each one takes effect at `systemctl --user restart cementic@papers`. Restart
  only while nothing is mid-batch.
- **The import diet precedes `cementic run`**, and `cementic run` precedes the
  systemd copy. This is the CLI audit's own constraint, plus the packaging step
  that now depends on it.
- **The lexical-arm change waits until 2a closes**, so the model comparison is
  never scored across two search behaviours.
- **The schema retrofits go in one step**, because `create_all` never alters an
  existing table. The attempt cap adds the first retrofit, and the efficiency
  step adds the rest through the same helper.
- **The functional-core refactors are behaviour-preserving, one commit each.**
  The embed planner lands before the rebuild. The rest can land any time after
  correctness.
- **The release comes last.** It waits for everything, the rebuild included
  if the riders pass (decision log, 2026-10-23).
- **Rider code stays out of the live pipeline until the rebuild is
  authorized.** Wiring 2b, 2c or 2e into extraction or chunking re-versions the
  profile, and the next restart of `cementic@papers` would start the 108 h
  rebuild by itself. Until the verdict, the riders exist as pure functions the
  harness calls.

### Waves (2026-10-23)

The open items below, grouped so each wave's members edit disjoint functions
and run concurrently. Each member works on its own `claude/` branch in a
worktree under `$TMPDIR` (handoff → environment quirks), lands on one wave
branch, and the user runs `./scripts/check.sh` before the next wave starts.
Test files are shared; their hunks touch different tests.

**Wave A — five members, concurrent.**
- *CLI output contracts* (`cli_collection.py`, the `embedding status` and
  `chunk` commands in `cli.py`, `embedding_runtime.py`, `vector_store.py`,
  `search.py`), one member, in this order: "Machine-readable output for the
  three query commands" → "Rename the filter's chunk index key" → "A document id
  in search results".
- *Watcher* (`source_watcher.py`, `db.py`, `revisions.py:303`), one member:
  "Functional core" 3 (watcher identity choice) → "Schema retrofits and
  efficiency" 1 (one retrofit pass) → 2 (restart without re-hashing). Identity
  goes first because 2 edits the same function.
- *Extraction isolation* (`extract.py`, the extract step of
  `pipeline_worker.py`). The 200-PDF byte-identity and wall-clock numbers run
  in the user's terminal on an idle machine; the member hands over the command.
- *Embed planner* ("Functional core" 1; `_step_embed` in
  `pipeline_worker.py`).
- *Small cores* (`revisions.py`, `collections.py`, `config.py`, `stop` in
  `cli.py`): "Functional core" 4, 6 and 2, and "Schema retrofits and
  efficiency" 4, one commit each.

**Wave R — the measurements, concurrent with Wave A.** Harness work only
(`scripts/measure_retrieval_quality.py`, new pure functions); nothing reaches
the live pipeline. "Rider measurement" and "Fuzzy matching" under "Retrieval
Phase 2". The scoring runs go to the user's terminal.

**Wave B — after Wave A's gate.**
- *`cementic run`*, then the systemd copy and the shared-machine warning
  ("Packaging, docs and deletions").
- *Lexical arm* ("Retrieval"), after "A document id in search results", since
  both edit `search.py` and `vector_store.py`.
- *One table printer* ("Functional core" 8).

**Wave C — after Wave B's gate, alone.** "Functional core" 7, public names. It
renames across every module, so it runs when nothing else is in flight.

**Wave D — the rebuild, if the riders pass. EMPTY (2026-10-23):** the riders
failed their rule ("Rider measurement"), so there is no rebuild, and "Schema
retrofits and efficiency" 3 waits for the next one.

**Then "Release 0.3.0".**

### Correctness

**1. Search shows each document's best chunk.** **DONE, on main since 2026-10-23 (`b756181`).** `_merge_arms` (`search.py:479-491`)
fills `by_path` last-write-wins over score-ordered rows, so a multi-chunk paper
carries its *worst* vector chunk while ranked by its best, and
`scores_explain_order` then hides the score column. Use first-write-wins, key on
`(collection, source_path)`, and lift the merge into a module function taking
`lead` as a parameter so it is testable without a session.
*Ends when:* a unit test with rows A 0.9, B 0.8, A 0.3 returns A's 0.9 chunk; it
fails on `73b5d4c`; harness `score` numbers are unchanged, since ranking does
not move.
*Anti-scope:* no ranking change, no `document_id` yet (the CLI step "A document
id in search results" re-keys on it later).

**2. Attempt cap for work that kills the worker.** **DONE, on main since 2026-10-23 (`350aeca`).** The pg retrofit test has not run; the crash loop is tested on SQLite. Today a document whose
extraction segfaults MuPDF or OOM-kills the worker is re-claimed first on every
restart, because the claim orders by lowest id and `requeue_interrupted_artifacts`
(`revisions.py:364-420`) resets every `processing` row. Add `attempts` to
`extracted_documents` and `chunked_documents`, bump it in the claim commit, and
have requeue stamp rows at the cap `failed` with a reason naming the count. The
schema change is an `ADD COLUMN IF NOT EXISTS ... DEFAULT 0` retrofit in
`create_tables`, beside the existing ones.
*Ends when:* a pg test drives claim → simulated death → requeue to the cap and
the next document is claimed; `status --verbose` shows the reason.
*Anti-scope:* no timeout here, since hangs are the isolation step's job; leave
embeddings alone, because an embed crash is the server's, not the worker's.

**3. Tokenizer offline.** **DONE, on main since 2026-10-23 (`696dd55`).** Proven here: cold and blocked. Not provable in the sandbox: warm and blocked. tiktoken caches `cl100k_base` in
`$TMPDIR/data-gym-cache`, which is wiped at reboot, and re-downloads it on
first use. Offline, `cementic chunk` prints a 26 KB traceback, the worker
stamps chunks failed, and 57 tests fail. Route the three `get_encoding` sites
(`chunk.py:87`, `search.py:79`, `embedding_runtime.py:425`) through one helper
that defaults `TIKTOKEN_CACHE_DIR` to the cementic data dir unless the user set
it. Add a `doctor` check for the cached encoding, and make `chunk` print one line
on a fetch failure. The decision log says why this beats vendoring or
server-side counting.
*Ends when:* with the cache warm and the network blocked, `chunk` works; cold
and blocked, `chunk` exits 1 in one line and `doctor` names the fix.

**4. A rename re-paths instead of re-embedding.** **DONE, on main since 2026-10-23 (`655f4c7`, `1974aa6`).** The directory case uses watchdog's synthesised per-file events, not a live observer. `on_moved`
(`source_watcher.py:439-449`) runs delete-then-create, and the delete purges
vectors at once. When the destination is unregistered and the source row is
live, update `source_path` in place: the same outcome
`_document_reached_by_another_path` already gives a moved tree found at scan.
*Ends when:* an integration test moves a file and a directory under a live
watcher and the chunk ids survive with no re-embed.
*Anti-scope:* do not defer deletes in general. The sync-client
delete-then-create trade at `source_watcher.py:285-298` is documented and
separate.

**5. Clear `last_error` after any productive pass.** **DONE, on main since 2026-10-23 (`fde1f1c`).** The clear at
`pipeline_worker.py:598-600` runs only on an idle pass, so a 10-second server
blip stays red in `status` for a whole import.
*Ends when:* a unit test of error, then productive pass, sees it cleared.

**6. Drop `verbose` from the runtime fingerprint.** **DONE, on main since 2026-10-23 (`fe6d057`).** It reverses a 2026-08-23
line, and the reason is in the decision log. It stays in the launch command.
*Ends when:* two configs differing only in `verbose` fingerprint equally. Expect
one daemon restart on deploy, because the alias changes once.

**7. Two CLI lies.** **DONE, on main since 2026-10-23 (`3fd2dd7`).** The `search` synopsis shows `[TRAILING_COLLECTIONS]...`,
which only works after `-c` (`cli.py:387-402`). Make usage and help show the
working form, `search "q" -c a b`, which the `tool-cementic` skill uses.
`config init` always writes the user-dir path (`cli.py:166`). Write the path
`config path` resolves to, or refuse and name the file that shadows it.
*Ends when:* `--help` shows only accepted forms, and `config init` with
`./cementic.toml` present does not write a file nothing reads.

**8. The small ones, one commit.** **DONE, on main since 2026-10-23 (`ebb2672`).** Usage errors exit 2 through `typer.Exit(2)`, not `BadParameter`, which would have changed every message's wording.
- The extractor profile display name exceeds `String(100)` with about eight
  backends (`profiles.py:250-255`), so truncate it, since the fingerprint
  carries identity.
- Plaintext extraction rejects non-UTF-8 files (`extract.py:239`), so decode with
  replacement. Only files that fail today change output.
- cementic's own argument errors exit 1 where Click's exit 2 (`cli_shared.py:431-441`),
  so raise `BadParameter`.
- `doctor` prints an empty `extensions:` heading and raw driver text when the
  database is down, so reuse the one-line unreachable message the other
  commands print.
*Ends when:* each has a test, and `check.sh` is green.

### User-facing CLI

The CLI-surface audit's remaining steps, carried over verbatim from 2026-09-10
except for renamed headings. Its rationale, and the directions it closed, are
under "Decided — the CLI surface after the `llm` audit". **Ordering:** the
import diet precedes `cementic run`, because `cli.py` does not import
`source_watcher` and `cementic run` adds a path to it. Done naively, `cementic run`
undoes the import diet. Machine-readable output precedes the document id,
because both change output contracts and the convention is fixed in the first.
The key rename is independent. Then four audit additions.

#### Import diet — stop the CLI importing the database and HTTP layers

**DONE, on main since 2026-10-23 (`1b5d7f6`..`b6a6632`); the user's gate,
including pg, passed.** `import cementic.cli` loads none of the six guarded modules
(`test_cli_import_does_not_pull_in_the_database_or_http_layers`). Interleaved,
best of 10, measured under load average ~20 so absolute numbers run high:
`import cementic.cli` 477 ms median against 1060 ms on `791c406`, `cementic
--version` 502 ms against 1264 ms. An independent review found no behaviour
change across 26 `status` scenarios diffed against `791c406`. The pg tests
whose patches were repointed (`test_cli_pg.py:114,134`,
`test_pipeline_worker.py:660,681`) passed in the user's gate.

The measured defect. Import time for `cementic.cli`, and for the set a light
command actually needs, six runs each (import only, excluding interpreter
startup). **Measure on an idle machine**: the first numbers taken for this plan
were recorded while `check.sh` was running and were about 2× too high, which is
corrected here.

| | idle | under a concurrent test run |
| --- | --- | --- |
| `import cementic.cli` today | 375–456 ms | 893–1079 ms |
| light floor (typer, rich, pydantic-settings, the leaf cementic modules) | 146–177 ms | 365–388 ms |

The light floor was verified to have `sqlalchemy` and `requests` absent from
`sys.modules`. The stable figure is the **ratio, about 2.5×**; the absolute
saving is roughly 230 ms idle and roughly 600 ms on a busy machine, on every
invocation of `--version`, `config path`, `extract` and `chunk`, none of which
touch a database. For scale, `cementic --version` end to end is 0.50–0.68 s
idle, so this is most of what that command spends.

**Refined 2026-10-23 against `791c406`.** Re-probed: all six guarded modules
are still in `sys.modules` after `import cementic.cli`. The counts below
replace the 2026-09-10 ones, which undercounted the test cost about 45×.

**What changes: 18 heavy module-level import statements move inside the
functions that use them**: `config.py` 2, `render.py` 3 (1 after the status document split), `cli_shared.py` 3,
`cli_collection.py` 2, `cli.py` 8. "Heavy" means importing the module alone
loads `sqlalchemy`, `requests` or `tiktoken`, probed per module; `supervisor`,
`state`, `hybrid`, `filelock`, `validation`, `cli_format`, `embedding_text` and
`index_strategies` are light and stay. `extract` and `chunk` are heavy only
through `config.py`, so fixing `config.py` also lightens `cementic extract` and
`cementic chunk` — but `chunk` still loads `tiktoken`, so `cli.py`'s
`chunk_text` and `TokenizerUnavailableError` imports move too. Two of those are
the crux: `config.py:27`
(`from sqlalchemy.engine import URL, make_url`) and `config.py:30`
(`from cementic.vector_store import HNSW_ITERATIVE_SCAN_MODES`). Verified
2026-09-10: `import cementic.config` alone pulls SQLAlchemy, and every light
command reaches `_get_config` or `resolve_config_path`, so **leaving `config.py`
alone caps the payoff at about 250 ms and fails the new guard test.** It is
all-or-nothing per dependency: one surviving edge re-imports the whole layer.

Three edges are not plain moves, each decided 2026-10-23:
- `render.py:25`'s `DaemonHealth` is a runtime enum comparison at
  `render.py:106`, so a `TYPE_CHECKING` guard does not work; it gets a real
  function-local import.
- `search.MAX_SEARCH_RESULTS` is consumed at `cli.py:977` inside a
  `typer.Option(max=…)` default, evaluated at import time. **Inline `max=50`**
  and add a test pinning it equal to `search.MAX_SEARCH_RESULTS`; no new
  module.
- `HNSW_ITERATIVE_SCAN_MODES` is read by the `hnsw_iterative_scan` validator
  (`config.py:814`), which runs whenever a config sets that field, so a
  function-local import there would load SQLAlchemy on every command for those
  users. **Relocate the tuple to `index_strategies.py`**, already a light module
  `config.py` imports; `vector_store.py` imports it from there.

**Hidden cost, and the largest one in this plan: 225 patch sites, 30 names,
5 test files.** Tests patch names bound by these imports at the importing
module (`cementic.cli_shared.get_engine` 42, `get_session_factory` 34,
`cementic.cli.Searcher` 17, `check_health` 17, ...). Moving an import into a
function deletes that module attribute and `unittest.mock.patch` raises
`AttributeError`. **Repoint each to its source module** (`cementic.db.get_engine`);
the function-local import reads the patched attribute at call time. Decided
2026-10-23 over thin wrapper seams, which would keep ~30 production functions
whose only reason is the tests.

**Patches cannot be repointed ahead of the import move.** While `cli.py` still
does `from cementic.search import Searcher` at module level, it holds its own
binding, so patching `cementic.search.Searcher` does not reach it. Each import
move and its repoints land together.

**Commits, one per file, each green, in two parallel waves.** Each runs on its
own branch and worktree off `claude/import-diet`, and merges back. The waves are
cut along the source files each commit edits; the test files are shared, but
their repoint hunks touch different patch strings.
- **Wave 1, concurrent:** `config.py` (with the `index_strategies`
  relocation); "Functional core" step 5, the status document split (edits
  `render.py`, `status_service.py` and the two `cli.py` call sites);
  `cli_shared.py`; `cli_collection.py`.
- **Wave 2, after the split merges, concurrent:** `render.py` (`DaemonHealth`
  only); `cli.py` (with the `max=50` pin). `cli.py` waits because the split
  edits its status call sites.
- **Last:** the guard test below, which can only pass once both waves are in.

**Why the split goes first (2026-10-23):** `render.py`'s heavy imports exist
because it runs queries, which breaks functional core / imperative shell.
Deferring them would hide the dependency; the split removes it. After the
split, `render.py`'s diet commit moves only `DaemonHealth`. Without it,
`render.py` would be edited twice, once to defer the imports and once to delete
them.

*Exit condition:* a new structural guard beside the existing PDF one, asserting
that after `import cementic.cli` in a subprocess, none of `sqlalchemy`,
`requests`, `tiktoken`, `cementic.pipeline_worker`, `cementic.collections` or
`cementic.db` is in `sys.modules`. Structural, not clocked, so it does not flake
under machine load. Probe for `sqlalchemy`, not `sqlalchemy.orm`: the cost is in
`sqlalchemy/__init__.py`. Naming the cementic modules too makes a regression
point at the module that reintroduced it. Plus `./scripts/check.sh` green,
including the seven other budgets in `test_cli_performance.py`.

*Anti-scope:* move imports, relocate the one tuple, and repoint the patch targets. Do not restructure
the module graph, do not defer `pydantic_settings` or `importlib.resources`
(that is the next ~400 ms and a separate, larger change), and do not make
`cli_collection` lazily loaded — `collection_app` is bound at `cli.py:79`, and
deferring it needs a lazy Typer group.

*Risk:* `config.py` already carries four function-local imports to dodge cycles
with `embedding_runtime` and `extract`. Moving an import *into* a function
cannot create a cycle, only break one, so the hazard is exposure rather than
creation. `index_strategies.py` has no cementic imports, so relocating
`HNSW_ITERATIVE_SCAN_MODES` there does not reshuffle the knot.

#### Machine-readable output for the three query commands

`collection list`, `collection revisions`, `embedding status`. Skip `stop`: it
is an action, not a query.

**`collection list` is not made redundant by `status --json`, despite the
overlap.** Verified 2026-09-10: `status --json` already emits a `collections`
block carrying documents and the active, ready and building labels, a strict
superset of what `collection list` prints. It costs more, because it also probes
workers and the daemon:

Measured idle, after the 2026-09-10 vacuum described below:

| | range |
| --- | --- |
| `cementic status --json` | 1.37–1.95 s |
| `cementic collection list` | 0.65–0.96 s |

So the cheap path earns its place, though by about 2× rather than the 3× an
earlier measurement suggested. That earlier pair (4.4–6.2 s against 1.6–1.8 s)
was taken while `check.sh` ran *and* before the vacuum, and is superseded.

**The `status` regression this step first surfaced is fixed.** It was planner
statistics: `chunks_v2` and `chunk_embeddings` were last analyzed 2026-08-24,
before the corpus doubled and before the 2026-09-09 repair. `VACUUM (ANALYZE)`
on 2026-09-10 took `load_pipeline_status_bulk` from 2.8–2.9 s to 0.62 s and
`status` end to end from ~4.1 s to ~1.6 s. Details and the operational lesson
are in `TODO.md`'s `cementic status` entry.

*What changes.* `collection list --json` emits JSONL from `CollectionSummary`
(`collections.py:52`), one object per collection, with `ready_revision_label`
and `building_revision_label` as **separate keys**. Do not reuse
`render._in_flight_revision_text` (`render.py:181`): it collapses the two into
one string and embeds rich markup. `collection revisions --json` emits JSONL
built from the `PipelineRevision` ORM rows, and must be built **inside** the
open session, because the profile relationships lazy-load; it can expose the
full fingerprints the table truncates to eight characters. `embedding status
--json` emits one document with the atoms latent in today's string:
`state` in `running`/`stopped`/`ambiguous`, `pid`, and `recovered`.

*Anti-scope, and it matters.* `embedding status` performs **no health probe**
today — `llama_daemon_status` (`embedding_runtime.py:1055`) returns a string
built from a pid and a recovered-flag. Do not add `probe_daemon` to it: that
changes the command's cost and meaning, and `status`/`doctor` already own the
health question. Add a structured helper alongside `llama_daemon_status` rather
than changing its return type — four tests
(`tests/unit/test_cli.py:1825–1864`) pin the literal string `"stopped"`.

*Exit condition:* each of the three commands' `--json` output parses under `jq`,
and `test_database_hint_is_one_unwrapped_stderr_line`
(`tests/unit/test_cli.py:2533`) still passes — it asserts stdout is empty when
`collection list` fails, and a `--json` error path must not break that stream
purity.

*Convention to settle here, once:* a command returning a list emits JSONL with a
bare `json.dumps` per line, as `search` does (`cli.py:1004`); a command returning
one object emits `json.dumps(doc, indent=2, default=str)`, as `status` does
(`render.py:393`). `doctor` uses `sort_keys=True` instead; leave it alone rather
than churn a third command to match.

#### `cementic run` — promote the runner subcommands

`cementic.runner` already has both commands with foreground semantics and a
deliberate non-zero exit on fatal startup (`runner.py:93, 141`), specifically so
a service manager can see it. This is surfacing existing code.

*What changes.* Expose them as a `cementic run watcher` / `cementic run worker`
group, then point the packaged unit at it (2026-10-22: the five old units are
replaced by a copy of the live one, under "Packaging, docs and deletions"). A group rather than two top-level commands,
because the surface is already the audit's weakest point and this adds one entry
rather than two.

*Risk, and the reason for the ordering.* `cli.py` does not import
`source_watcher` today. Mounting the runner eagerly would pull it in and undo
the import diet, so the group must load its implementation inside the command body.

*Anti-scope:* do not change what the workers do, do not add a `--foreground`
flag to `start`, and do not make `start` spawn the new commands — the supervisor
matches processes by command line (`supervisor.find_pids_by_cmdline`), so
changing the spawn string is a separate change with its own failure mode. Leave
the user's own `~/dotfiles` unit for last, and keep it pointing at the
checkout's interpreter: that pins the PyMuPDF version in the extractor
fingerprint, and switching to the installed snapshot re-versions the corpus.

*Exit condition:* `cementic run worker --collection papers` starts and stops
cleanly in the foreground, the packaged units start under `systemctl --user`,
and the import-diet guard test still passes.

#### Rename the filter's chunk index key

One line at `cli.py:1155`, from `{"index": …}` to `{"chunk_index": …}`, plus the
six test sites asserting or feeding the key (`tests/unit/test_cli.py:481, 2687,
2712, 2718, 2795, 2806` — two of those are `chunk` stdin fixtures, not
assertions, so a grep for the assertion alone under-counts the work) and the
README pipe example. A breaking change to an output format with no known
consumers, which is why it only gets more expensive to defer.

*Anti-scope:* the key name only. Do not add document identity to the filter and
do not make it drop whitespace chunks to match the worker — those divergences
are documented under "Pipeline as composable filters" and are the honest
description of a debugging tool.

*Exit condition:* `cementic extract X | cementic chunk | cementic embed` still
runs end to end, and no test or doc still says `index`.

#### A document id in search results

*What changes, and it is a query change rather than a dict change.* Neither
`vector_store.knn_sql` (`vector_store.py:147`) nor `lexical_sql`
(`vector_store.py:182`) projects an id today. Both already join
`source_documents`, so this is adding `sd.id AS document_id` to two SELECT
lists, two constructor edits in `search.py` (:337 and :380), one key on the
`SearchResult` TypedDict (`search.py:80`), and the README field list.

*Decided: `document_id` only, not `chunk_id`.* `_merge_arms` dedupes by
`source_path` and lets the vector arm win the projection (`search.py:407–412`),
so a chunk id would be whichever chunk the winning arm happened to match —
arbitrary, and unstable across a query the other arm leads. A document id is
stable because path and document are one-to-one. *Added 2026-10-22:* once the
id exists, re-key the search projection (Correctness step 1) on it instead of
`(collection, source_path)`.

*Exit condition:* `cementic search Q -c papers --json | jq -r .document_id`
returns ids, and the README field count matches the emitted keys — the defect
the 2026-09-10 docs fix corrected was exactly that count drifting.

*Anti-scope:* one id field. Not the artifact path, not the chunk index, not the
revision label; each is a separate argument about what a result is for.

*Anti-scope for the CLI steps, so nobody re-derives it:* no ingest command
for embedded JSONL and no search-by-vector (the composability decision above
says why — and note it does not reach `cementic add`); no deletion of
`start`/`stop` or the supervisor; no dropping of PostgreSQL or the daemon to
match `llm`'s zero-dependency shape, which is the engine and not the interface;
and no unifying of the `--force` short forms.

#### Audit additions

- **Drop `search --limit`.** **DONE, on main since 2026-10-23 (`239e565`).** It is a third spelling of `-n` and clashes with
  `status --limit` (`cli.py:966`). *Ends when:* `search --limit` is a usage
  error, and no doc or test names it.
- **README lists the filters.** **DONE, on main since 2026-10-23 (`239e565`).** `extract`, `chunk` and `embed` are the only way
  to test an extractor without a database. *Ends when:* README's command block
  names them.
- **Document the JSON shapes.** **DONE, on main since 2026-10-23 (`239e565`).** Cover `status --json` and `doctor --json` in
  `docs/operations.md`, including the `error` key and which keys vanish when the
  database is down (`render.py:355-386`). Do it alongside the JSONL convention
  above. *Ends when:* the documented keys match a test that asserts them.
- **Delete `collection_callback`** **DONE, on main since 2026-10-23 (`a4eeb5b`).** (`cli_collection.py:33-38`). It never runs,
  because `no_args_is_help` fires first, and its unit test calls it directly.
  *Ends when:* the function and that test are gone and `cementic collection`
  still prints help.

`collection list` stays a separate command. The decision log has the
reaffirmation.

### Schema retrofits and efficiency

**1. One retrofit pass.** Use the helper the attempt cap introduced, in
`create_tables`:
- add `size_bytes` and `mtime_ns` to `source_documents`;
- drop the two redundant prefix indexes, `ix_source_documents_collection`
  (`db.py:71`) and `ix_chunk_embeddings_profile_status` (`db.py:307`). Both have
  been deferred since 2026-08-26;
- set `NOT NULL` on the three `chunk_embeddings` scope columns, which stayed
  nullable from a deleted migration. Refuse with a count if any NULL row exists.
Fix the stale index credit at `revisions.py:303`.
*Ends when:* a pg test builds the old shape, runs `create_tables`, and matches a
fresh database. `EXPLAIN` of the embed claim and of
`_document_reached_by_another_path` still uses the wider indexes.

**2. Watcher restart without re-hashing.** Every start SHA-256s the whole
corpus (`source_watcher.py:897-901`). Compare `(size_bytes, mtime_ns)` first,
and hash only on change.
*Ends when:* a test counts zero hashes on an unchanged second start, and the
`papers` watcher restart time is recorded here, before and after.
**DONE 2026-10-23 (Wave A).** `systemctl --user restart cementic@papers` to
"scan complete": 683 s on the old code, 219 s on the second start of the new
one (the first, 630 s, fills the new stat columns and hashes once). Item 1's
`EXPLAIN`: the embed claim uses `ix_chunk_embeddings_claim`, the move lookup
`ix_source_documents_collection_file_hash`. Logs:
`~/.cache/cementic-waves/watcher_restart_*.log`, `explain_indexes.log`.

**3. Measure the extract and chunk claims.** They order by id over outer-joined
`IS DISTINCT FROM` predicates (`pipeline_worker.py:641-669, 807-835`), which
likely walks the done prefix. The embed claim had this exact pattern, measured
and fixed. Run `EXPLAIN (ANALYZE, BUFFERS)` at a mostly-done revision. The
Phase 2 rebuild at about 90% is the natural moment.
*Ends when:* the plan is recorded here with a fix, or a verdict of "flat in the
done count".

**4. `collection remove` by subselect.** `collections.py:143-175` loads every
document as an ORM object and ships three literal `IN` lists.
*Ends when:* a pg test passes and the statements carry no id literals.

*Anti-scope:* no TOML parse cache, for the reason under "Deliberately not
done", and no throttling of the worker's state-file writes.

### Functional core

Behaviour-preserving, one commit each. A commit that has to change an existing
test has changed behaviour and is out of scope. *Anti-scope for the phase:* no
new layers (no state-machine framework, no command bus), and no module-graph
change beyond the one split named below.

1. **Embed planner and set-based write-back, before the rebuild.**
   `_step_embed` (`pipeline_worker.py:963-1170`, about 210 lines) mixes the
   claim, publishing, retry policy, provider calls and write-back. Extract a
   pure `plan_embed_outcome(claimed, results, errors)` that returns done,
   failed-with-reason and release sets. Write back with two set UPDATEs in place
   of about 96 statements per 32-chunk batch, and drop chunk ids that vanished
   mid-request before the vector upsert, which closes the foreign-key race at
   `:1134-1168`.
   *Ends when:* the outcome policy is tested as data, the statement count per
   batch is measured, and the pg suite is green.
2. **`stop` decides as data.** `stop_background` (`cli.py:751-897`) picks one of
   nine messages inline with `os.kill`. Make the decision a pure function of
   processes, liveness and kill results, returning messages, the next state and
   the exit code.
   *Ends when:* each of the nine branches is a table-driven test.
3. **Watcher identity choice.** Inside `_register_document`
   (`source_watcher.py:865-966`), "which existing row is this file" becomes a
   pure function over the three lookups. *Ends when:* tested without a session.
4. **Target-revision rule.** The resurrect, supersede or create decision in
   `get_target_revision` (`revisions.py:222-279`) becomes a pure function over
   existing revisions and profile ids. *Ends when:* tested without a session.
5. **Status document split.** **DONE, on main since 2026-10-23 (`4989219`).** **Pulled forward into the import diet
   (2026-10-23, user's call): it runs before the diet's `render.py` commit.**
   `render.build_status_document` becomes a query function in `status_service`
   and a pure document builder, dropping the two smuggled callables and the
   hint parameter. `_print_collection_detail` stops loading file progress
   mid-print (`render.py:259`); its caller loads `files` and passes them in.
   *Ends when:* the builder is tested from plain data, and `render.py` imports
   nothing from `collections` or `status_service`. Exception to the phase's
   no-test-changes rule: `tests/unit/test_json_shapes.py` imports
   `build_status_document` and passes the callables this step drops, so it is
   rewritten to the new signature. The JSON shapes it asserts must not change.
6. **`Config` as a value.** Move the path defaulting out of `Config.__init__`
   (`config.py:1160-1191`) into an after-validator. Split the diagnostics half
   (`config.py:129-412`) into `config_diagnostics.py`, the one module-graph
   change this phase allows. *Ends when:* two `Config()` built from different
   working directories differ only in the fields that name a path.
7. **Public names for cross-module use.** Drop the underscore from the five
   `render._print_*` helpers, the 17 `cli_shared` names `cli.py` imports, and
   `_TASK_PREFIX_TOKEN_ALLOWANCE`. Fix the "pure bookkeeping" label on
   `_over_budget_reasons`, which may make an HTTP call. Capture
   `extracted.content_hash` into a local in `_step_chunk`, not off a detached
   instance. *Ends when:* no `_`-prefixed name is imported across modules.
8. **One table printer.** Replace the three hand-rolled column-width printers
   (`cli.py:674-691`, `cli_collection.py:114-123`, `:329-339`) with a single
   pure rows-to-lines helper. *Ends when:* the CLI tests are unchanged and the
   helper has its own tests.

### Extraction isolation, before the rebuild

Extract PDFs in a child process with a wall-clock timeout and `RLIMIT_AS`. A
crash or timeout stamps the document failed with the reason. Today MuPDF parses
in the worker, which holds the database credentials, with no limit, and the
900 s timeout covers only command extractors.
*Pre-registered:* on a 200-PDF sample, the extracted text must be byte-identical
to in-process extraction, so the extractor fingerprint must not move.
Extraction wall clock may rise at most 5%. Past that, use one long-lived child
restarted on death instead of one process per document.
*Ends when:* both numbers are recorded here, and a test PDF that kills its
child is stamped failed while the next document proceeds.
*Anti-scope:* no seccomp or namespace sandbox; command extractors are
unchanged.
**DONE 2026-10-23 (Wave A), wall-clock rule missed and accepted.** One
long-lived child (one process per document measured 4.6x). On 200 PDFs: 200 of
200 byte-identical; isolated 19.0 s against 18.1 s in-process, ratio 1.0535
against the 1.05 bar. An earlier pass over 161 of the same PDFs, with a colder
cache, measured 1.019. The cost is a fixed ~4.5 ms per document of IPC, so the
ratio rises as extraction gets faster. Accepted by the user (decision log,
2026-10-23 gate). Log: `~/.cache/cementic-waves/isolation_measure.log`.

### Retrieval

**Lexical arm, after 2a closes.** The inner `LIMIT :pool` in `lexical_sql`
(`vector_store.py:197-205`) has no ORDER BY, so for a common term the 2000-row
pool is roughly the oldest rows. Ordering it is not the fix, because ranking
every match is the recorded hang under "Two performance traps". Measure whether
the sample bias moves the harness at all by scoring pools of 2000 and 10000, and
either raise the pool or document it as an unranked sample. Separately, scope
`lexical_should_lead`'s rarity probe (`search.py:503-515`) to the searched
revisions, the way `lexical_sql` scopes its arm. Make the non-hybrid and
one-arm-empty paths (`search.py:295, 476-477`) return one row per document,
like the fused path.
*Ends when:* harness `score` before and after shows no set losing more than one
query, 0.017, on recall@1 or recall@10, and the verdict is recorded here.

#### Retrieval Phase 2 — the single rebuild, batched

Everything here re-versions a profile. Decide all of it, then run once — the
cost of missing one is another full rebuild.

**2a. Reconsider the embedding model.** What the rebuild authorisation actually
buys, and it dominates every ranking change. Current
`nomic-embed-text-v2-moe` at `n_ctx = 512` forces `chunk_size = 320`
(`cementic doctor`, 2026-09-18). The v1.5
analysis (below, under "Rejected by measurement") already costed the prize: a 2048-token window supports
`chunk_size` near 1,300, cutting the corpus from ~2.3M chunks to ~575k and the
vector table from 18 GB to ~4.5 GB. v1.5 lost on retrieval quality; the reason
to want the window did not. Candidate GGUF models need a fresh check — no model
is recommended here, and any name predating this line is stale. Method is Phase
0's, on a 1,000-paper sample *(run at 500, an agent-decided deviation: about
5M tokens per candidate at two chunk sizes)* with the stored index as control, before
committing ~108 h.

**CLOSED 2026-10-23 — stay on v2-moe; no candidate passes the pre-registered rules.**
2b-2d are now judged alone. History: instrument built 2026-09-24, Gemma run
2026-09-29, Qwen3 and Harrier run 2026-10-23.
`measure_retrieval_quality.py compare-models` replaces the v1.5 script in
`~/.cache/cementic-igpu/calibration/` (not yet deleted; delete it once a full
run has completed). **Before deleting, move out `calibrate.sh` and its `*-cpu`
artifacts:** that directory also holds the only canary recalibration script
(see "Live residuals of closed records" → canary). Each of 500 seeded papers contributes one 2,000-token body
span (stored chunks 2-9, skipping the title page), embedded at the stored
320-token chunking and at 1,300 with overlap 325. That fixes the v1.5 run's
stated flaw, which compared a long-window model only at 320. Queries are the
paper's title and one verbatim phrase from the span; a document ranks by its
best chunk. Arms: stored v2-moe vectors (the control, paired with fresh v2-moe
queries, as production search does), fresh v2-moe at 320 (the noise floor),
and each candidate at both sizes. Each model is served by the live daemon's
own build (10605) on port 11557, and the harness caches after every finished
step under `~/.cache/cementic-model-compare/`.

*Candidates, checked 2026-09-24:* Qwen3-Embedding-0.6B (official Q8_0,
Apache-2.0, 32k window, 1024 dims, last-token pooling), EmbeddingGemma-300M
(ggml-org Q8_0, Gemma licence, 2k window, 768 dims), Harrier-oss-v1-0.6b
(MIT, April 2026, Qwen3 architecture; only community GGUFs, mradermacher's
used). All three GGUFs append their own end-of-text token (`/tokenize`), so
nothing adds one. All three are in `models/`, gitignored.

*Throughput through llama-server, under load average 16-30 from other work:*

| model | chunks | tokens/s, 8-paper smoke | tokens/s, 500-paper run |
| --- | --- | --- | --- |
| v2-moe | 320 | 568 | 838 |
| EmbeddingGemma | 320 | 1,036 | 902 |
| EmbeddingGemma | 1,300 | 1,164 | 609 (stopped at 40%, not saved) |
| Qwen3-0.6B | 320 / 1,300 | 124 / 138 | 277 / 253 |
| Harrier-0.6B | 320 / 1,300 | 217 / 209 | 393 / 290 |

The absolute numbers are load-bound (v2-moe benched 754 under load against
2,468 recorded idle), so read ratios. Qwen3 and Harrier run 2.6-4.6x slower
than v2-moe, which puts their projected rebuild above the pre-registered
threshold of about twice today's ~108 h. They still get scored, but a win
comes back as a speed-versus-quality decision, not a go. EmbeddingGemma is the
only candidate at or above v2-moe's speed.

*Quality so far:* the 8-paper smoke saturated recall and decides nothing; it
verified the pipeline (fresh v2-moe margin 0.1286 against stored 0.1285, zero
misaligned chunks, zero refusals).

**RUN 2026-09-29 — EmbeddingGemma fails.** 500
papers (sample `8d3c95f989f5`), run in the user's terminal, log
`~/.cache/cementic-model-compare-gemma.log`. Recall@1, with the paired sign
test against stored v2-moe:

| arm | title r@1 | title better/worse, p | phrase r@1 | phrase better/worse, p |
| --- | --- | --- | --- | --- |
| v2-moe stored@320 | 0.850 | control | 0.301 | control |
| v2-moe fresh@320 | 0.848 | 2/2, 1 | 0.299 | 99/9, 2.6e-20 |
| Gemma@320 | 0.890 | 54/31, 0.017 | 0.234 | 171/224, 0.0088 |
| Gemma@1300 | 0.874 | 52/34, 0.066 | 0.216 | 170/238, 0.00089 |

Verdict: Gemma@1300 fails two of the three rules. Its title gain misses
p < 0.01, and it loses on phrases (recall@1 0.216 against 0.301, MRR 0.302
against 0.381). Gemma@320 fails the same two. The query and document prefixes
are the model card's (`task: search result | query: `, `title: none | text: `),
so a prompt mismatch does not explain the phrase loss. Throughput is not
comparable across the two runs: Gemma ran at 1,246 / 1,025 tok/s under load
6-9, and v2-moe ran at 838 under load 16-30.

Noise-floor anomaly, **RESOLVED 2026-10-23 (`0d0aae6`): a scoring defect.**
The 21 chunks fresh v2-moe refused are dot-leader tables of contents, which sit
in the stored index as hub vectors outranking the gold; `score_arm` dropped
them from the fresh arm only. `compare-models` now scores every arm of a
granularity on the rows all arms embedded (3,979 of 4,000 at 320). On that
pool fresh against stored is 20/15 on phrases, p = 0.50, so the noise floor is
null. Rescoring the cached arms moves phrase recall@1 by at most 0.003 and
changes no 2a verdict.

**RUN 2026-10-23 — Qwen3 and Harrier fail; 2a closes on v2-moe.** Same
sample `8d3c95f989f5`, run in the user's terminal, log
`~/.cache/cementic-model-compare-all.log`. Sign test against stored v2-moe:

| arm | title r@1 | title better/worse, p | phrase r@1 | phrase r@10 | phrase better/worse, p | tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| v2-moe stored@320 | 0.850 | control | 0.301 | 0.545 | control | 838 |
| Qwen3@320 | 0.876 | 55/27, 0.0026 | 0.236 | 0.533 | 169/223, 0.0074 | 277 |
| Qwen3@1300 | 0.894 | 61/25, 0.00013 | 0.238 | 0.473 | 161/235, 0.00023 | 253 |
| Harrier@320 | 0.906 | 59/24, 0.00015 | 0.230 | 0.513 | 167/220, 0.0081 | 393 |
| Harrier@1300 | 0.920 | 63/16, 9.4e-08 | 0.246 | 0.481 | 163/234, 0.00043 | 290 |

Verdict: both pass the title rule at 1,300 and both fail "not lose on
phrases", so no candidate wins. Harrier@1300 is the strongest title arm
measured (+0.070 recall@1). Its phrase loss reaches recall@10 too (0.481
against 0.545), so routing to the lexical arm cannot be assumed to cover it.
By throughput ratio, a Harrier@1300 rebuild projects to about 270 h and a
Qwen3@1300 rebuild to about 310 h, against ~108 h today. Both are above the 2x
threshold. The verdict does not depend on the noise-floor anomaly below.

*Pre-registered rules (2026-09-24):* a candidate at 1,300 must beat stored
v2-moe on title queries by a paired sign test at p < 0.01, not lose on
phrases, and gain more margin than fresh-versus-stored v2-moe does. No winner
closes 2a as "stay on v2-moe", and 2b-2d are then judged alone. A winner whose
projected rebuild exceeds ~2x today's goes to the user with the number.

*Expected ceiling of the payoff:* a better embedding can move title recall@1,
where the vector arm leads, and recall@10 / MRR everywhere. It cannot move
phrase recall@1, which routing owns (Phase 1 step 2).

**2b. The three riders the v1.5 section parked.** Unchanged and specified
under "Rejected by measurement — v1.5" below: drop punctuation-dominated chunks at chunk time, re-derive
`chunk_size` against the chosen model's `/tokenize`, remove the `use_ocr` knob.
Their *ends when* conditions stand as written.

**2c. Structure-aware chunk boundaries.** From txtai, verified in source: the
extractor writes a separator at page breaks and headings (`htmltomd.py:120`,
`:224`) and the chunker splits on it when present, falling back to blank-line
runs when absent (`segmentation.py:148-149`). One character is the entire
protocol between the two stages — no import, no shared type, and the chunker
stays a pure function. `pymupdf4llm` already emits Markdown headings to hang it
on. Keep the token window as fallback and as the hard cap: this is a boundary
*preference*, not a new chunker, and it adds no dependency.

**2d. Narrow the extractor-registry fingerprint.** `TODO.md`'s parked item,
whose stated trigger is "anything else forces a rebuild". This is that day; it
rides along free.

**2e. De-hyphenate at extract time.** Restored from Phase 0's known artefact:
*Known artefact:* one accepted phrase is `augmentations by max- imizing the
agreement` — PDF line-break hyphenation surviving into stored chunk text. It
splits words across both arms' term space. A de-hyphenation pass at extract
time re-versions the extractor profile, so it is nearly free as a Phase 2
rider and expensive at any other time.

*Anti-scope, so nobody re-derives it:* no log-odds or convex score fusion —
Phase 1 step 2 buys the same thing more cheaply *(stale reason: that step was
closed on latency 2026-09-24; the anti-scope stands because the routing ceiling
equals the Phase 0 bar, so no fusion rule can beat it)*, and txtai's implementation
cites a paper absent from its own tree (`hybrid.py:232`); no chonkie or NLTK
chunker (2c does it with a separator); no second embedding provider added
merely to exercise a one-entry registry; no Postgres search extension.

**Rider measurement (scheduled 2026-10-23, Wave R).** Implement 2b
(punctuation-dominated chunk drop), 2c (structure-aware boundaries) and 2e
(de-hyphenation) as pure functions, not wired into the pipeline. Add one
harness arm that applies all three to the spans of sample `8d3c95f989f5` and
embeds with v2-moe at 320. 2d moves no vector and is not scored.
*Built 2026-10-23 (`0d0aae6`, `measure_retrieval_quality.py riders`).* What
the sample's 500 spans give the riders to act on: 6,228 line-break hyphen
shapes (5,765 joined; about 1 false join in a hand check of 40), 75 of 4,000
chunks punctuation-dominated (all 21 refused chunks among them), and almost
nothing for 2c: 27 Markdown heading lines in 9 spans, mostly Python comments,
because the sample is raw text-layer extraction. The verdict therefore rests
on 2b and 2e. A diagnostic row with only 2b's chunks removed separates them.
*Pre-registered rule:* the riders arm, paired against fresh v2-moe at 320
without riders, must beat it on title or phrase recall@1 by sign test at
p < 0.01, and must not lose on the other. Pass: Wave D runs. Fail: park 2b, 2c
and 2e with the numbers, and Wave D is empty.
*Ends when:* the verdict and its table are recorded here.
**RUN 2026-10-23: FAIL — 2b, 2c and 2e PARKED, Wave D empty.** Riders against
the fresh control on recall@1: title 11 better / 10 worse (p = 1), phrase
16 / 21 (p = 0.51). Embedding: 3,955 chunks, 1,331,920 tokens in 1,377 s.

| Arm (sample `8d3c95f989f5`) | title R@1 | title R@10 | phrase R@1 | phrase R@10 |
|---|---|---|---|---|
| stored@320 | 0.850 | 0.974 | 0.299 | 0.545 |
| control: fresh v2-moe@320 | 0.848 | 0.974 | 0.299 | 0.545 |
| control without 2b chunks | 0.846 | 0.970 | 0.291 | 0.539 |
| riders (2b+2c+2e) | 0.850 | 0.966 | 0.289 | 0.535 |

Dropping 2b's 75 chunks alone moves phrase rankings on 148 queries (141 better
at any rank, 7 worse) yet lowers recall@1, so the ranks it improves are below
the top. *Revisit when:* a rebuild is authorized for another reason; the
riders are free to ride it, since they did not hurt. Log:
`~/.cache/cementic-waves/rider-arm.log`.
*Anti-scope:* no per-rider attribution and no new chunk size; 2b's re-derived
`chunk_size` rides along only if the rebuild runs.

**The rebuild runs only after** the rider measurement passes, and after correctness steps 2 and 4, the schema
retrofits, the embed planner, and extraction isolation are in.

**Fuzzy matching (opened 2026-09-19 as Phase 1 step 3) — scheduled 2026-10-23, Wave R.**
No rebuild, no re-versioning, independent of everything above; ordered last
within Phase 1 because it is the only item here with no measured target yet.

The gap, measured: a query one character off its document retrieves nothing
lexically (60/60 typo'd phrases returned zero rows) and lands at fused recall@10
0.233 on the vector arm alone. Every number this project has ever reported is on
queries spelled exactly as the corpus spells them, so the whole eval is blind to
the case — which is the one a person searching their own library from memory is
most likely to produce.

*Not yet a decision.* Three candidates, none costed: `pg_trgm` similarity as a
fallback only when the lexical arm returns zero rows (narrow, adds an extension
and an index); relaxing `plainto_tsquery`'s conjunction to an OR-with-threshold
(no extension, but it changes recall on every query, not just typo'd ones); or
accepting the vector arm as the only typo path (Phase 2a kept v2-moe, so no
model change moves 0.233).

*Ends when:* one of the three is chosen with a measured recall@10 on
`phrase-typo` to justify it, or the item is closed as "not worth it" with the
0.233 recorded as the accepted floor.
**Trigram without an index: not viable (2026-10-23).** A probe of 20,000
chunks took 7.65 s, so `similarity`/`word_similarity` over the ~2.3M chunks is
about 879 s per query on one core, and the full measurement about 14.6 h. Not
run. The relaxed-OR candidate needs no scan and is measured alone in Wave B's
gate (`fuzzy --no-trigram`). The trigram candidate stays open only as an
indexed one. *Revisit when:* relaxed OR fails on `phrase-typo` and a GIN
trigram index on `chunks_v2.content` is costed (build time and size).
~~**Do not start this before Phase 1 step 2**~~ — moot since 2026-09-24:
step 2 is closed and nothing is about to be deleted, so this item is no
longer blocked, only unscheduled.

### Packaging, docs and deletions

Independent of the code phases, except where noted.

- **systemd: ship a copy of the live unit.** Replace the five units in
  `packaging/systemd/` with a templated copy of
  `~/dotfiles/.config/systemd/user/cementic@.service`. Template the paths and
  state that it needs `daemon_autostart = true`. Use `cementic run` once that
  step has landed. Point `docs/setup.md:77` and `docs/operations.md:35` at it.
  Dotfiles stay untouched. The decision log records the accepted drift.
  *Ends when:* the copy starts under `systemctl --user` from a checkout at a
  path other than `~/projects/cementic`.
- **Shared-machine warning.** Add it to `docs/setup.md` and the template README:
  the default Postgres role is superuser `cementic`/`cementic`. Give the
  `ALTER ROLE` command. *Ends when:* both docs carry it.
- **Delete dead and duplicate code.** **DONE, on main since 2026-10-23 (`a4eeb5b`).**
  - `vector_store.backfill_filter_columns_sql` and `enforce_filter_columns_sql`,
    which have zero callers.
  - `tests/integration/test_runner.py`, after folding its two unique cases into
    the unit file.
  - `scripts/measure_embedding_claim_scale.py` and
    `scripts/measure_index_build_order.py`. No doc names either, and the second
    imports a private test helper.

  *Ends when:* every check passes.
- **Structural debounce tests.** **DONE, on main since 2026-10-23 (`a4eeb5b`).** Replace the four wall-clock tests
  (`tests/integration/test_source_watcher.py:336-389`) with an injected timer.
  *Ends when:* no `time.sleep` remains in them.
- **Doc anchors.** **DONE, on main since 2026-10-23 (`a4eeb5b`).** Both docs that count gates say six. Fix TODO.md's three dead anchors: `tests/unit/conftest.py`
  and two README headings removed in `6df8ab9`. State one gate count everywhere.
  *Ends when:* every path TODO.md cites exists.
- **`tool-cementic` skill.** It lives outside the repo and is not writable by
  agents. If a surface it names changes, hand the user the edit. Today it uses
  only `-n` and `-c`, so likely nothing.

### Release 0.3.0

Bump `pyproject.toml`, `uv lock`, write the CHANGELOG entry per its tag-time
policy, tag, push, then `uv tool upgrade cementic`. The push is outward-facing,
so it needs the user's go at the time.
*Ends when:* `cementic --version` from PATH prints `0.3.0`.

### Not doing (so nobody re-derives it)

- **Merging `status` into `collection list`:** reaffirmed 2026-10-22.
- **A per-install Postgres password:** see the decision log.
- **`llama-server --api-key`:** a local-user port squat is out of scope for a
  single-user tool.
- **The `RUNNING`-before-startup ordering:** the 2026-08 verdict under
  "Deliberately not done" stands. It is a two-line fix only if
  `PipelineWorker.start` is open anyway.
- **A TOML parse cache:** same section.
- **Pinning CI actions to SHAs:** major tags are acceptable for a personal repo.

## Closed records

One line each: verdict, the numbers later steps cite, and a pointer. The full
text of every record is `git show 73b5d4c:PLAN.md`.

- **Hybrid retrieval execution order (2026-09-07), CLOSED 2026-09-10.** The
  leading arm owns rank 1 and fusion owns the rest (`hybrid.py`). Lexical-lead
  routing fires on a single identifier-shaped token in at most 20 documents,
  the threshold set D's document-frequency bands gave. The GIN index is built
  up front, because it costs 0.1% of import wall clock. Commits `36c73ff`,
  `45ed214`, `03f443d`, `64b24d4`. Risks that remain are under "Decided — hybrid
  lexical + vector retrieval".
- **CLI-surface audit step 1 (2026-09-10), DONE.** README's `search --json`
  field count is fixed, and `verify_postgres_container.sh` no longer calls the
  removed `status --doctor`. Steps 2-6 moved into the execution order.
- **v1.5 migration (2026-09-05), CLOSED, the gate failed.** On 1,000 papers,
  v1.5 had recall@1 0.6590 against v2-moe's 0.7170, with sign test
  p = 8.9e-09. The stored-index control arm scored 0.7200 / 0.8860 / 0.7812
  (recall@1 / recall@10 / MRR). The canary shipped in `06f7d62`.
- **Retrieval Phase 0 (2026-09-18), DONE.** The harness is
  `scripts/measure_retrieval_quality.py`, with instrument `eval/queries.json`,
  60 queries per set. Baseline recall@1 / recall@10 / MRR: rare-token
  0.633 / 0.817 / 0.689, title 0.783 / 0.883 / 0.817, phrase 0.117 / 1.000 / 0.348.
  **The bar any rank-1 change must clear is recall@1 ≥ 0.650 rare-token,
  ≥ 0.783 title, ≥ 0.967 phrase.** The perturbed phrase sets are diagnostics,
  not gates. The typo set's lexical arm returns 0/60 rows, and fused recall@10
  is 0.233.
- **Retrieval Phase 1 step 1, over-fetch (2026-09-19), CLOSED.** The per-arm
  fetch multiple is 5, capped at `MAX_SEARCH_RESULTS`, and costs no measurable
  latency. Nothing was measured above 50 per arm, so if the multiple is ever
  raised, re-measure rather than extrapolate. The step exposed chunk-count RRF scoring, fixed in `1e0bd7d` by
  fusing documents, not chunks. Rare-token recall@10 rose to 0.983, and
  recall@1 was unchanged everywhere.
- **Retrieval Phase 1 step 2, reranker (2026-09-24), CLOSED on latency.**
  Router mode passes cementic's alias check but needs two fixes: `/tokenize`
  must send a `model` field, and the supervisor's `--alias` command-line match
  does not see a router launch. 50 chunks take 16.9 s best-of-5 against a
  500 ms bar. Benchmarking `ts_rank` against BM25 stays withdrawn, because
  BM25 inherits the same `plainto_tsquery` conjunction and cannot fix typo
  recall. The `lead` ablation showed per-query oracle routing
  equals the Phase 0 bar exactly, and that deleting `lead` bare loses 36
  rare-token and 13 title queries. A reranker an order of magnitude smaller
  over the top 10 only was never measured. Logs: `~/.cache/cementic-router/`,
  `~/.cache/cementic-eval-lead-ablate.log`.
- **Full-text index reports its own absence (2026-09-12), DONE.**
  `collection reindex` builds it, `doctor`'s `lexical_index` check names its
  state, and an invalid leftover is dropped before the rebuild.
- **Document identity (2026-09-09), DONE.** The invariant to keep: **any code
  path that removes chunks clears the chunking's `source_content_hash` in the
  same statement.** A file is the same document when its stored path still
  resolves to it. Fixed by `38b0bdd`, `e755474`, `cae580c`, `7645182`.
- **External command extractor (2026-09-03), DONE** in `935bf45`. The residual
  risk is that a version command printing a constant cannot be detected, so
  `doctor` shows what was recorded for a human to check. The contract and the
  fallback rule are under "Live residuals of closed records".
- **Query-first embed scheduling (2026-09-01), DONE.** It shipped as
  `embed_submit_batch_size` and `db.SearchActivity`. Still open: HNSW insert
  cost growing with the graph, host contention, and pgvector index maintenance
  as explanations for the embed-rate decay from 5.5 to 4.0.
- **Build the ANN index up front (2026-08-13), TAKEN.**
  `ensure_revision_ann_index_up_front`, HNSW only. DiskANN keeps build-at-ready
  because it is unmeasured.
- **Server drift detected by a canary (2026-09-05), IMPLEMENTED** in
  `canary.py` and `doctor`'s `embedding_canary`. Left undone deliberately: a
  `doctor --restamp` front door, and canaries for collections that never index
  again.

## Design decisions and open items

### Decided — the CLI surface after the `llm` audit (2026-09-10)

Evidence, the full command inventory and the `llm` comparison are in
`notes/design-cli-surface.html` (gitignored). The remaining steps are under "Execution
order (2026-10-22)" → "User-facing CLI". This block holds only what a later reader would otherwise
re-derive: why the audit cut nothing, and which directions are closed.

**The comparison is the front door, not the engine.** `llm` needs no daemon, no
PostgreSQL and no systemd, which is the shape this document's own principles
ask for, so the pull to copy it is real. It buys that shape by not doing
cementic's job: no chunking, no PDF extraction, no watching, and a search that
registers a Python function and linear-scans every row — minutes on 2.3M chunks
where cementic answers in 1–2 s. So the audit compared four things only: command
count, output contract, what a script can do with the output, and what must be
running first. *Rejected:* dropping PostgreSQL or the daemon. That ports `llm`'s
premise (a small SQLite corpus) along with its mechanism.

**Nothing was cut, and the criterion is worth stating because it was lenient.**
All 21 leaf commands appear in README and in at least one test. The test applied
was "does deleting this save anything", and for a thin wrapper the answer is
almost always no. A stricter test — "does this cost a reader attention" — would
have cut more, and the honest answer to that cost is better help output, not
fewer capabilities: a user meets 8 leaf commands and 4 groups entered
deliberately. *Reopen if:* you decide the surface is too large to hold in your
head, in which case the target is the groups, not the leaves.

**The composability principle overstated the design, so the principle changed.**
`extract | chunk | embed` runs, but nothing consumes its output and nothing can.
A vector is only searchable inside a revision pinning an extractor profile, a
chunk profile and an embedding profile plus fingerprints, which is exactly what
lets a new model build while the old one stays searchable. A hand-injected
vector carries none of that, so an ingest command must either record a false
provenance or take every profile id as an argument, at which point it is not a
filter. *Rejected:* an ingest command **for embedded JSONL**, and
`search --vector`. The Design principles entry was narrowed instead
(2026-09-10). **Scope, spelled out 2026-09-12 because the line above read
wider than it is:** what is rejected is ingesting *vectors*. `TODO.md`'s
`cementic add <path>` ingests a *file* through the ordinary pipeline, which
creates its own provenance, and is untouched by this.

**The supervisor stays; the inversion it created does not.** Six commands and
~630 lines (`supervisor.py` 268, `filelock.py` 69, `start`/`stop` handlers 290)
reimplement supervision systemd provides, and this machine's deployment uses
none of it — `cementic@.service` runs `python -m cementic.runner` with
`Restart=always` and health-checks by `curl`. *Rejected:* deleting `start` and
`stop`. The evidence says the author does not use them, not that nobody needs
them; README leads with `start`, macOS has no systemd, and keeping working
tested code costs nothing. The asymmetry decided it: deletion is irreversible
with a hypothetical beneficiary, promotion is additive with a beneficiary
running now. What remains is that the used entry point is an internal module the
user's own unit names directly, and an internal module carries no stability
promise — an ordinary refactor could break a live service silently.

**Rejected — unifying the `--force` short forms.** `-f/--force` on `collection
promote` and `reindex`, bare `--force` on `collection remove`, `stop`, `config
init` and `init postgres`. A breaking change to muscle memory for a consistency
nobody has tripped over. *Reopen when:* someone reports it.

### Decided — hybrid lexical + vector retrieval (2026-09-07) — DONE

Shipped 2026-09-09. A dense embedding cannot recover a rare exact token, so
vector-only search found the right paper for 7 of 150 rare-token queries, and
hybrid found it every time (recall@10 0.047 to 1.000). Plain RRF loses rank 1
on both query kinds, so the leading arm owns rank 1 and fusion owns the rest.
Full measurements: `git show 73b5d4c:PLAN.md`. What stays live:

**Resolved 2026-09-12 — maintaining this index per insert costs 82% of its
size.** The live index read 1,121 MB against 617 MB for a fresh rebuild of the
same rows, and `REINDEX INDEX CONCURRENTLY` recovered it in 6.0 min with search
up throughout. Build memory, corpus growth and dead tuples were each ruled out.

**What this changes.** The up-front decision stands — its argument was wall
clock and correctness-at-every-moment, and those hold — but it now reads: +82%
index size until rebuilt, recoverable in 6 min whenever a bulk import ends.
That step is in README's hybrid section as `REINDEX INDEX CONCURRENTLY`,
deliberately manual: size is not a correctness fault, `doctor` cannot judge it
without building the alternative, and `collection reindex` only rebuilds an
index that is missing or invalid. *Reopen if:* a later measurement shows the
maintained index costs latency as well as space. It did not here — 291 ms
against 342 ms for the non-embedding half of a search, inside one arm's own
80 ms spread, so no speedup is claimed.

**Risk — the precondition is unmeasured.** All of this assumes real queries are
identifier-shaped. Nobody has looked at one; sets A and C are synthetic by
construction. If such queries are a small fraction of real use then this is a
well-measured solution to a problem the corpus does not have, and set A's 0.047
does not change that. **This gated step 5 of the closed hybrid execution order, which shipped anyway**, so the gate
was passed by decision rather than by measurement — the strongest reason to
distrust this section is that its own precondition was never checked. Cheap
counter: log query strings and timestamps locally and read the mix.
*Revisit when:* a week of real query logs exists, or hybrid is suspected of
hurting an ordinary semantic search.

**Resolved — the probe index has a declared home.** The hand-made
`chunks_v2_fts_probe` is gone and `ix_chunks_v2_fts` replaced it, created by
`ensure_lexical_index` (`db.py:461`) from `create_tables` (`db.py:569`), so a
fresh database gets one. Verified 2026-09-10 against the live database, which
carries `ix_chunks_v2_fts` and no probe index.

**Risk — the harness is not version controlled.** Every number in this section
came from `notes/probe_hybrid_retrieval.py`, and `notes/` is gitignored by the
project's own decision. The results survive here; the code that produced them
would not survive a `git clean`. Either exempt that one file or accept that the
measurements are reproducible only from their description. *Narrowed
2026-09-18:* Phase 0 moved on-demand measurement into the tracked
`scripts/measure_retrieval_quality.py`. Only the original probe stays untracked.

**Two performance traps, both measured rather than reasoned**, kept because
each cost an hour: `select distinct document_id … limit n` makes the planner
walk `ix_chunks_v2_document` and apply the text match as a filter, never
touching the GIN index — a token matching *nothing* then costs 8 s+ to prove
absence, and `enable_seqscan = off` does not fix it because the bad plan is an
index scan too. And `ts_rank` over every match of a common term recomputes a
tsvector per row and hangs for minutes; ranking must happen over a capped
candidate pool.

### Rejected by measurement — v1.5, and what would reopen a model swap

Verdict in "Closed records". Kept for the reopen condition, which Phase 2a is
now exercising:

**What would reopen this.** A candidate that beats v2-moe on
`compare-models.sh`, run the same way. Two facts worth carrying into that
search: a longer context is not itself worth a rebuild -- v1.5's 2048 against
v2-moe's 512 did not compensate for worse retrieval -- and v1.5 embedded ~4x
faster in the same run (3.4 against 0.8 embeds/s, partly because v2-moe's
512-token window refused five batches to v1.5's one, forcing per-text retries).
If indexing throughput ever binds harder than retrieval quality, that trade is
now measured rather than guessed.

Three changes were parked waiting for a rebuild to happen for some independent
reason, each costing a full re-embed on its own. A model swap would have been
that reason, so they were to ride along — doing them afterwards costs a second
~5-day re-embed for nothing.

What a model swap actually costs, read out of the code and the live database
rather than estimated:

- **Extraction is untouched.** The extractor profile does not change, so all
  23,064 documents are reused as they are.
- **Chunks are reused unless chunk settings change.** `build_chunk_profile_payload`
  is `chunk_size`, `chunk_overlap`, the tiktoken tokenizer and a version — the
  model is not in it. In practice the swap changes them anyway, because
  `chunk_size = 320` was chosen for this model's 512-token window. Re-chunking
  needs no GPU and no server.
- **Every chunk is re-embedded.** 2,298,558 at the measured 5.56 chunk/s is about
  five days. A longer context trades chunk count against per-chunk cost rather
  than removing the work, so treat any speedup as unmeasured until it is measured.
- **No downtime, and no mixing.** Vectors live in a table per embedding profile
  (`embedding_vectors_p6` today). The new revision builds into its own table
  while the old one stays active and searchable; `promote_revision` flips at the
  end. Cross-collection search already refuses to merge two models' scores
  (`search.py:_mixed_model_message`).
- **Disk roughly doubles until the old revision is pruned.** The database is
  21 GB, of which `embedding_vectors_p6` — vectors plus HNSW index — is 18 GB.
- **One daemon serves one model.** The server's `--alias` fingerprints what it
  loaded and the readiness check refuses a server not serving it, so a
  wrong-model answer is impossible — but indexing the new model and searching the
  old contend for port 11555. This is the trigger for `TODO.md`'s parked
  "multi-profile embedding daemon pool"; the no-code workaround is a second
  `llama-server` on another port, which is already how the iGPU setup runs.

**Chosen 2026-09-05 — `nomic-embed-text-v1.5`.**
768 dimensions (unchanged, so the vector table shape is unchanged), a longer
context than the current 512 (measured below, and it is not the 8192 the model
card advertises), Matryoshka truncation, a published
GGUF, and an embedding space shared with `nomic-embed-vision-v1.5` — which would
let figures and page images be searched alongside text without a second vector
space, the thing `Markdown as the text intermediate representation` above calls
option (b). The cost is giving up v2-moe's multilingual MoE, and v1.5 is an
older generation (MTEB 62.28 at 768 dims, per its model card).

Rejected: staying on v2-moe, which keeps the 512-token window and leaves all
three parked items parked; and going straight to a multimodal model
(`nomic-embed-multimodal-3b/7b`, `colnomic-embed-multimodal-3b/7b`), which is
3-7B parameters against v1.5's ~0.1B, needs the parallel non-Markdown path, and
has unverified llama.cpp support. v1.5's shared vision space is what makes that
a cheap follow-on rather than a competing direction.

**Load check done 2026-09-05, and it corrects the headline number.** The GGUF
already in `models/` (`nomic-embed-text-v1.5.f16.gguf`, 274 MB) loads under
b10818 and serves 768-dim vectors. But its context is **2048, not 8192**:
`n_ctx_train` in the GGUF is 2048, and llama.cpp caps the slot to it -- "the
slot context (8192) exceeds the training context of the model (2048) -
capping", after which a 3,002-token input is refused outright. The card's 8192
comes from RoPE scaling that this conversion does not carry.

8192 *can* be forced -- `--rope-scaling yarn --rope-scale 4` loads at
`n_ctx_slot = 8192` and embeds a 3,000-word input fine -- but that is untested
for quality, and the flags do not match what the card specifies (dynamic NTK
scaling, factor 2.0). Treat scaled RoPE as its own experiment, not as a
config line.

So the honest gain is **4x the current window, not 16x**: 2048 tokens supports
a `chunk_size` near 1,300 against today's 320, which cuts the corpus from
~2.3M chunks to roughly 575k and the vector table from 18 GB to ~4.5 GB. Still
worth the migration; just not the number the plan was written around.

**What rides along, in the order the pipeline runs them.**

*Drop chunks that are mostly punctuation, at chunk time.* Measured on the live
corpus: of 5,025 chunks that failed the token budget, ~3,830 are dot-leader
tables of contents (`. . . . . 117 C.5.4 Proof of Claim 19`). They tokenize at
~1.59 model tokens per tiktoken token against a 1.45 bound, which is why they
overflow. They are worthless for semantic search either way, and 735 shorter
ones *did* embed and sit in the index as junk vectors. The length limit is
currently acting as an accidental filter for the rest. *Ends when:* a re-chunk
of the corpus produces no dot-leader chunks and the junk vectors are gone.

*Re-derive `chunk_size` against the model's own tokenizer, not tiktoken.* The
mismatch is the root cause of the item above. Do not simply lower it: on the
current model, clearing the observed p95 (637 model tokens) needs 249 and the
observed max (1958) needs 75, which is too small to be a useful chunk. v1.5's
measured 2048-token window changes that arithmetic entirely: the observed max
fits with room to spare, so the filter above stops being load-bearing.
*Ends when:* `chunk_size` is set from a measured token-ratio distribution
against the chosen model's `/tokenize`, recorded here.

*Remove OCR from the codebase.* See the section below; it is blocked on this
same rebuild and on the command extractor landing first.

### Parked again — remove OCR from the codebase

**Not blocked on design; it now has a rebuild to ride on.** `use_ocr` reaches
extraction only through the pymupdf4llm backend, so it is a knob that silently
does nothing under `pymupdf-raw` — `doctor` grew a warning for exactly that,
which is a guard where the fix is removing the knob. Deleting it removes
rapidocr from the picture entirely (dependency, extra, and `--with` all moot).

The cost is that `use_ocr` is a key in the extractor payload, so removing the
field changes every extraction fingerprint: `papers` re-extracts, re-chunks and
re-embeds all 2,291,555 chunks. The model migration would have paid for that
rebuild anyway, and it is not happening, so this waits for the next real reason
to rebuild. Its *other* constraint is now satisfied: the command extractor
shipped in `935bf45`, so a configured `ocrmypdf` wrapper remains an OCR route
once the knob is gone.
Pre-processing with `ocrmypdf` into `pymupdf-raw` works with no cementic code at
all and is the documented fallback — untested here, no OCR engine is installed
on this host.

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

### Live residuals of closed records

Verbatim from `73b5d4c`, because each still constrains work in the execution
order or carries a revisit trigger.

**Document identity: the retirement rules.** Both "A rename re-paths" and
"Watcher identity choice" edit this code.

**Retiring a stale row may not delete chunks at all.** The reconcile retires a
row whose path is no longer its own real path — previously invisible to it,
since `_is_under_watched_roots` compares literally and a pre-move path is under
no watched root. The first guard written for it retired a row whose twin *held
chunks*, on the reasoning that the twin covered the text. An independent review
killed that: holding chunks is a membership test, so a twin one chunk into a
five-hundred-chunk document satisfies it while covering almost none of it. The
rule now is that the row must hold no chunks and a live document must already
hold its real path — retiring deletes nothing, and nothing unreplaced is
retired. Ungated, a restart on the damaged corpus would have deleted 2.3M
chunks and bought hours of re-embedding to reach the state it was already in.

**Which then required folding the twin back in.** With retirement that strict,
a duplicate pair would be permanent: the repath only fires when nothing is
registered at the scanned path, so a twin already sitting there short-circuits
every later scan. So an empty twin gives way to the row that holds the work —
the same resolution the manual repair applied to the live corpus — and a pair
that both hold chunks is left for a human, because choosing between them means
discarding one. Both halves came out of the review; the branch shipped the
first guard and the permanent duplicate it implied.

**External command extractor: the contract.**

**Contract, kept narrow on purpose:** argv in, text on stdout, nothing else. A
tool that writes files instead (`ocrmypdf`) gets a two-line wrapper — that is
the composition boundary, not cementic's problem. A wide contract is what turns
an early abstraction into debt.
The command extractor is never a fallback: it must be named in `backends`.

**Canary: the invariants.** The 0.99 cosine threshold sits about 100x above the
worst observed cross-build noise. The calibration table is in `73b5d4c`.

**The comparison is strict** — doctor flags any difference from the stored
vectors, and reports the cosine only to say how bad it is (~1e-9 is arithmetic
noise, below 0.999 is a changed function). Accepting a drifted build must be an
explicit act that re-stamps the canary, not a threshold that silently absorbs
it; that is the difference between a check that stays trustworthy and one that
is tuned until it never fires.

The canary must live *outside* `build_embedding_profile_payload`. Anything in
that payload moves the fingerprint and forks the corpus, which is the outcome
being avoided. The natural home is a sibling table keyed by
`embedding_profile_id` holding the texts, their vectors, and the `build_info`
string the server reported when they were taken (`/props`, today
`b10605-a130532ae`). Note that `EmbeddingProfile.config_json` is *not* the
fingerprinted payload — it carries the runtime payload `search` rebuilds a spec
from (`profiles.py:301`) — so it is not a shortcut home for this either.

Rejected:

- **Leave it and document the hole.** Free, and forgotten by exactly the person
  it would have saved.
- **Put the build in the embedding profile.** Fires on every upgrade including
  the arithmetic-class majority, each costing a full re-embed (~5 days at the
  measured 5.56 chunk/s), and makes profile construction depend on a live
  daemon, which today it does not.
- **Compare build strings in doctor.** The same false-alarm rate without the
  rebuild. A check that fires on every upgrade is a check that gets ignored.

Reproduce with `~/.cache/cementic-igpu/calibration/calibrate.sh` (it reuses
`sample.json`, so inputs stay identical); the CPU-run artifacts are preserved
alongside as `*-cpu`. Note for whoever repeats this: absence of Vulkan lines in
the server log does **not** mean CPU — the known-GPU run from 2026-08-24 has
none either. Check `llama-server --list-devices`, and remember an agent shell
sees no GPU at all.

**Left undone deliberately.** Re-stamping is a row delete plus one indexing
run, which capture's store-if-absent rule already makes correct; a
`cementic doctor --restamp` would be a nicer front door for it, and is not
worth a command until a real build upgrade makes someone want it. Capture is
also indexing-only: a collection that never indexes again keeps no canary, so
the check reports honestly that it has nothing to replay rather than inventing
a reference from today's server.

**Query-first scheduling: what is parked, and the ceiling.**

3. **Abort in-flight on lease activity** — possible (b10605 cancels queued
   tasks ≤ ~1 s on client disconnect) but not built: Layer 1 shrinks the
   quantum until aborting saves ≤ ~1 s. Seam left open.

Parked with triggers: a second query-side llama-server (Vespa's documented
practice; the strong fix if Layers 1–2 measure insufficient — costs runtime-
fingerprint work and resident memory) and an upstream llama.cpp patch exposing
the internal front-of-queue path as a `priority` field on embeddings.

What survives: the fixed ~0.36 s per request is measured and real, so
sub-batching does cost *something*; and a hard ceiling was found that bounds
any work here — `--ubatch-size` is 512 while chunks average ~318 model tokens,
so about one chunk fits per forward pass and one slot took 300 of 400 launches
with 32 tasks queued. Request size cannot buy concurrency the physical batch
has no room for. Raising `n_ctx`/ubatch is the untested lever and changes the
runtime fingerprint, so the server must restart for it.

### Deliberately not done

Judged not worth fixing, with the reason. Full history:
`git show 73b5d4c:PLAN.md`.

- **ANN pre-filter recall on shared vector tables:** inherent to ANN plus
  post-filter. The actionable slice was purging vectors of deleted documents,
  and that is done.
- **DiskANN with inner-product `storage_layout`:** unreachable while only cosine
  exists.
- **Text policy by filename** (`embedding_text.py:33`): a renamed GGUF gets the
  wrong prefixes. Reading GGUF metadata is the real fix, and it is deferred.
  Model *identity* is already a content digest.
- **TOML parsed once per settings source,** about 9x per `Config`. A cache keyed
  on path and mtime risks stale reads on coarse-mtime filesystems for a
  millisecond win. *Revisit when:* config load shows up in a profile.
- **Three trim candidates stay:** `_llama_daemon_runtime_status`, `_state`, and
  the `state_path is None` guards.
- **Naive `TIMESTAMP` columns:** nothing reads them, and `create_all` cannot
  migrate a column type. *Revisit when:* anything reads them.
- **`ignore_directories` replacing the defaults:** works as documented. A
  separate `additional_ignore_directories` would be a feature, not a fix.
- **The advisory-lock leak and the `RUNNING` state-write ordering:** every path
  that leaks is followed by process exit, and `start`'s liveness check corrects
  the state file. A two-line fix if those functions are open anyway.
- **The `verbose` line from 2026-08-23 is superseded.** It said `verbose`
  "correctly stays in the runtime fingerprint", and the 2026-10-22 decision log
  reverses that.

### Audit findings not fixed (2026-08-24)

Checked against the real 22,246-file corpus and **not present**, so reachable in
principle but not here: filenames containing `[` (would break
`status --verbose`), non-UTF-8 filenames (the bare `.decode()` calls in
`source_watcher.py`), and paths over about 2,600 bytes (btree limit on
`ix_source_documents_collection_source`).

Still live:

- **A revision can reach `ready` mid-initial-scan** (`revision_is_complete`):
  the watcher registers documents one at a time with no scan-complete marker,
  so a drain of the first N looks complete. Promotion re-checks and refuses, so
  this misleads rather than corrupts. The all-zero case is refused outright;
  what remains is the first-N drain of a scan already under way.

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
