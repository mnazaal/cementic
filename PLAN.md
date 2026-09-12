# cementic — architecture & design

<!-- session-handoff:begin (2026-09-10, second session that day) -->
## Where the work stands

**Entry point: "Execution order — CLI surface audit" below, step 2.** Step 1 is
done. Steps 2–6 are specified in full — what changes, the exit condition, the
anti-scope, the hidden cost — so the next session implements rather than
re-derives. Read the "Decided — the CLI surface after the `llm` audit" block
first if you want to know *why* the audit cut nothing; skip it if you only want
to build.

**Repo state.** `main` = `origin/main` = `bccd23a`, everything merged and
pushed, no branches outstanding. *(Corrected 2026-09-12: this described the
pre-merge state and gave the merge commands. Both were carried out.)*

**Running, and safe to ignore.** `cementic@papers.service` is up; corpus is
settled at 23,075 documents, 2,299,762 vectors, 100% embedded. No background
jobs, no watch loops, nothing on a cluster.

**What this session did.** Ran the CLI-surface audit `TODO.md` had gated on the
hybrid thread closing, then planned its follow-up work in detail, then fixed a
`status` performance regression found on the way. Three commits, all on `main`.
Nothing is half-finished.

**Corrections — distrust these sections' history, not their current text.**
- **Two figures I published were wrong, and are the reason to be careful here.**
  The step-2 import timings and the step-3 cost comparison were first measured
  while `check.sh` was running, which inflated them about 2×. Both are corrected
  in place and now say what conditions they were taken under. The lesson is in
  the step-2 text: measure on an idle machine, and cite the ratio rather than
  the absolute when you cannot.
- The hybrid section was titled "In progress" for a thread that closed on
  2026-09-10, and two of its risks referenced execution-order steps that had
  shipped. Retitled, one risk resolved against the live database, one restated.
  The remaining open risk there is real: **the precondition for hybrid retrieval
  was never measured** — nobody has looked at whether real queries are
  identifier-shaped — and the feature shipped anyway.
- README claimed `search --json` emits six fields; it emits seven.
  `scripts/verify_postgres_container.sh` called `cementic status --doctor`, a
  flag deleted months ago. Both fixed.

**Deviations from the plan, attributed.**
- *User-directed:* running the vacuum, and every merge and push.
- *Agent-decided:* cancelling the vacuum of `embedding_vectors_p6` after 78
  minutes. It was not stale, it was not needed, and its 9,150 MB HNSW index
  makes it an hours-long job. Recorded under `TODO.md`'s `cementic status`
  entry so nobody includes it again.
- *Agent-decided:* fixing the two documentation defects during the audit rather
  than leaving them as step 1 of its own plan, because writing a document that
  called them false while leaving them in place was worse.
- *Agent-decided:* the audit cut nothing. That is the plan's weakest point and
  it is argued explicitly in the Decided block, with the stricter criterion that
  would have cut more and why it was not used.

**Environment quirks that cost time.**
- **The `agent-checkpoint` hook blocks every Bash command when the tree is
  clean.** Its snapshot legitimately holds nothing new, but it cannot tell that
  from a failed index, so it exits fatal and the gating hook refuses the turn.
  Any edit to a tracked file clears it; the `Read`/`Edit`/`Write` tools are not
  gated. Worth fixing in the script rather than working around.
- The Bash sandbox blocks 127.0.0.1:5432 and the systemd user bus, so anything
  touching the database or `systemctl --user` needs the sandbox disabled.
- The masked device-node files at the repo root (`.bashrc`, `.zshrc`, …) are
  sandbox artifacts, not real files. Ignore them in `git status`.
- `zsh` does not word-split an unquoted variable. A loop of the form
  `for c in "status -c x"; do cementic $c; done` passes the whole string as one
  argument and every command looks like a usage error. This produced a false
  "README's exit codes are wrong" finding before it was caught.

**Artifacts.** `~/.cache/cementic-vacuum.py` is worth keeping — it vacuums the
five tables `status` reads and prints before/after `pg_stat_user_tables` rows.
`notes/design-cli-surface.html` holds the audit evidence and the `llm`
comparison (gitignored, so it exists only on this machine). The last full green
gate is `~/.cache/cementic-check-vacuum.log`. Everything else this session
produced — the help-tree dump and a sample Markdown file in the scratchpad — is
discarded deliberately; the help tree regenerates from `--help` in one command.

**Exit criteria — commands whose output confirms the above.**
```bash
git status --short                       # empty apart from masked dotfiles
git log --oneline -1                     # bccd23a, equal to origin/main
./scripts/check.sh                       # six gates, all ok
./.venv/bin/cementic doctor              # every line ok
./.venv/bin/cementic status              # ~1.6 s, papers 100% embedded
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
- **Pipeline stages are also filters.** `extract | chunk | embed` are
  stdin/stdout filters, so a single document can be run end-to-end with no
  database. **Narrowed 2026-09-10** from "Unix composability": the chain has no
  exit — nothing ingests embedded JSONL and `search` always derives its own
  vector — and it cannot get one, because a cementic vector is only meaningful
  inside a revision that pins three profiles and their fingerprints. These are
  debugging and export tools, not a composable loop. See "Execution order — CLI
  surface audit".

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

Their JSONL is not the stored shape, so do not read it as a picture of the
database: the filter emits `index` where `chunks_v2` stores `chunk_index`, it
carries no document identity, and it keeps the whitespace chunks the worker
drops and renumbers. Step 5 of "Execution order — CLI surface audit" closes the
key-name half of that.

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

## Execution order — hybrid retrieval (2026-09-07)

**CLOSED 2026-09-10.** Steps 1–5b are done: the mechanism is measured, the
shape is settled (*the leading arm owns rank 1, fusion owns the rest*), the
pure core ships in `hybrid.py` verified against both query sets, and the shell
shipped a session later than this section claimed — `search.py` branches on
`config.search.hybrid`, `ensure_lexical_index` runs inside `create_tables`, and
`ix_chunks_v2_fts` is present on the live database. Evidence and predictions
are in **Decided — hybrid lexical + vector retrieval** below and
`notes/design-hybrid-retrieval.html`. Kept for the measurements; nothing here
is outstanding except the one risk that section still carries, which is that
the precondition for the whole feature was never measured.

1. ~~**Confirm the rank-1 collapse is a tie-break artifact.**~~ **DONE
   2026-09-07.** Set A, n=150: 68.0% lost an *exact* score tie (winner was a
   vector-only document in 101 of 102), 28.7% genuinely outscored, 3.3% already
   rank 1. The pre-registered bar was ≥90%, so the diagnosis is **not**
   confirmed as stated — but the decomposition is exact: 68.0 + 3.3 = 71.3,
   which is lexical-only's recall@1 to the digit, and the 28.7% complement is
   precisely the queries where gold is not lexical rank 1.

2. ~~**Fix the tie-break and re-measure.**~~ **DONE 2026-09-07 — the rule
   failed, so step 3 stands.** Step 1 reframed this: the tie is not a bug.
   Both arms' rank-1 documents score `1/(k+1)` because RRF is symmetric, so
   there is no neutral tie-break — choosing one *is* a bet on an arm. Measured
   with the bet placed on lexical:

   | set | vector | lexical | hybrid RRF | hybrid + lexical ties |
   | --- | --- | --- | --- | --- |
   | A recall@1 | 0.027 | 0.713 | 0.047 | **0.713** PASS |
   | B recall@1 | 0.713 | 0.173 | 0.540 | **0.567** FAIL (−0.147) |
   | A recall@10 | 0.047 | 1.000 | 1.000 | 1.000 |
   | B recall@10 | 0.787 | 0.207 | 0.840 | 0.840 |

   The rule required both sets within 0.05 of the best single arm. A lands on
   it exactly; B misses by 0.147. **The finding worth keeping is the split:**
   fusion is unambiguously right for the result *list* (recall@10 improves on
   both sets, 0.047→1.000 and 0.787→0.840) and cannot work for the *top slot*,
   because the tie-break that maximises A is the one that costs B. So the
   design is *fuse the list, route the ordering* — narrower than a routing
   subsystem.

3. ~~**Find where lexical stops beating vector, as a function of document
   frequency.**~~ **DONE 2026-09-07 — shape found, exact threshold not
   identified.** Set D, single tokens on a fresh sampling stream, 40 per band,
   gold = the document the token was drawn from:

   | document frequency | lexical | vector | lexical − vector |
   | --- | --- | --- | --- |
   | 1–5 | 0.500 | 0.050 | **+0.450** |
   | 6–20 | 0.200 | 0.050 | **+0.150** |
   | 21–100 | 0.050 | 0.000 | +0.050 |
   | 101–500 | 0.000 | 0.025 | −0.025 |
   | 501–2000 | 0.000 | 0.000 | 0.000 |

   The pre-registered rule put \(T\) at the top of the last band winning by
   ≥ 0.05, which reads as \(T = 100\) — but that band's margin is **2 queries
   out of 40** and the design was never powered to resolve it. Honest reading:
   the *shape* is unambiguous and the exact threshold is not. Lexical's edge
   decays monotonically and is gone by 100, so any \(T\) in 20–100 captures
   almost all of the benefit. **Use \(T = 20\)** — the conservative end, routing
   only where lexical wins decisively.

   Two things this also settled. Set A was **optimistically constructed**: it
   required the gold document to appear in the lexical top 50, which is why its
   lexical recall@1 is 0.713 where set D's unfiltered 1–5 band gives 0.500. The
   vector arm reads 0.050 against set A's 0.027 — so the core finding (vector
   fails on rare tokens, lexical does not) survives both constructions, but
   set A's absolute numbers are the flattering ones. And the falsifier did not
   fire: vector stays at or below 0.050 in every band, including on freshly
   sampled tokens.

4. ~~**Repair the null control.**~~ **CLOSED 2026-09-07 — set D already is
   one, and a better one.** Set C was confounded (its gold document came from
   the same unordered sample the lexical arm ranks within, so both favoured the
   same physically-early rows). It does not need repairing, because set D
   supplies the null condition properly: the 501–2000 band *is* "common tokens,
   where exact matching cannot identify a document", and lexical's advantage
   there is **+0.000**. Better still, set D gives a dose-response — the
   advantage runs +0.450, +0.150, +0.050, −0.025, 0.000 as frequency rises,
   while the vector arm stays flat at 0.000–0.050 throughout. A mechanism that
   scales with the quantity it is supposed to depend on, and vanishes where it
   should, is stronger evidence than a single null cell. No set C claim is
   made and none is needed.

5a. ~~**The pure core.**~~ **DONE 2026-09-07 — `hybrid.py`, verified against
   both sets.** `combine(vector, lexical, lead=...)`: the leading arm owns rank
   1, reciprocal rank fusion owns everything below. One position is routed, not
   the whole ordering — taking more of the leading arm's list gives back the
   recall@10 that fusion earns. Measured end to end through the production
   functions, n=150 per set:

   | set | arm | recall@1 | recall@10 | routing |
   | --- | --- | --- | --- | --- |
   | A rare tokens | best single (lexical) | 0.713 | 1.000 | |
   | | **routed** | **0.713** (+0.000) | **1.000** (+0.000) | 150/150 lexical |
   | B semantic | best single (vector) | 0.713 | 0.787 | |
   | | fused only | 0.540 | 0.840 | |
   | | **routed** | **0.713** (+0.000) | **0.853** (+0.013) | 150/150 vector |

   Both optima at once, which is what the split was designed for: rank 1 and
   recall@10 are decided in different places, so they stop trading against each
   other. Set B's routed recall@10 beats even pure fusion, because leading with
   the vector arm's first result costs nothing there and the fused tail still
   contributes the documents only the lexical arm found.

5b. **The shell — still to do.** *Correction 2026-09-07:* this step previously
   said the lexical index becomes per-revision, reconciled by `collection
   reindex` like the ANN index. That was wrong and made the step look three
   times its real size. Vectors live in per-profile tables
   (`embedding_vectors_p6`) because different models produce incomparable
   vectors; `chunks_v2` is a single global table and text is
   model-independent, so **one GIN index serves every collection and every
   revision**. It belongs beside `ix_chunks_v2_document` as a table-level
   index, and touches the revision machinery not at all. It stays out of every
   profile payload for the reason already established — a text-derived index
   cannot change a vector.

   i. ~~**Measure the insert cost before making it default-on.**~~ **DONE
      2026-09-07 — and the pre-registered rule asked the wrong question.**
      Interleaved arms, real chunk text, first batch discarded
      (`notes/probe_gin_insert_cost.py`):

      | arm | median per 2,000 rows | rows/s |
      | --- | --- | --- |
      | with GIN index | 0.480 s | 4,166 |
      | without | 0.104 s | 19,184 |

      **+360% write overhead**, far past the 10% bar, and the no-index arm's
      own spread is 0.041 s so the difference is not noise. By the rule as
      written: build after import.

      The rule is wrong, because relative insert throughput is not the
      decision-relevant quantity — total import wall clock is, and inserts are
      not remotely the bottleneck. Over the whole corpus: 2,298,558 chunks at
      4,166 rows/s is **9.2 minutes** of insert time against **2.0 minutes**
      without, a difference of ~7 minutes. Embedding the same corpus runs at
      the measured 5.56 chunk/s — about **115 hours**. The index costs **0.1%
      of import wall clock**, and the arithmetic that makes it look expensive
      is a ratio between two quantities that are both ~750× faster than the
      step that actually gates the pipeline.

      **Decision: build up front**, which is also the `ensure_revision_ann_index_up_front`
      precedent — no post-import build step to forget, no window where the
      index is silently absent, and the index is correct at every moment. This
      is a deliberate departure from a rule fixed in advance, recorded as one
      because the alternative is pretending the rule was never written.

   ii. ~~**Creation path.**~~ **DONE 2026-09-07 (`36c73ff`), with a defect
      found and fixed in `45ed214`.** The first version ran the CONCURRENTLY
      build inside `create_tables`, which runs at worker startup — so on an
      existing corpus `cementic start` would have blocked for minutes, or
      indefinitely, because CONCURRENTLY waits for every open transaction on
      the table before it begins. A hanging test found it. Startup now builds
      only on an empty table, where it is instant; an existing corpus upgrades
      via `build_lexical_index`, which a person runs and watches.
      `ensure_lexical_index(engine)`, PostgreSQL-only and a
      no-op elsewhere — the pattern `ensure_vector_extensions` already uses, and
      the reason the sqlite unit suite keeps working. An empty table indexes
      instantly; an existing corpus needs `CREATE INDEX CONCURRENTLY`, which
      cannot run inside a transaction and so needs its own connection.
      *Exit:* met 2026-09-12, and the exit criterion had to be built before it
      could be met. `doctor` had no lexical-index check, so a missing GIN index
      was reported by nothing — the same silence the document-identity section
      calls out as costing more than the bugs. It now has one; see "Decided —
      the full-text index reports its own absence" below. The sqlite-backed
      unit tests are untouched.

   iii. ~~**The lexical query, scoped to the active revision.**~~ **DONE
      2026-09-07 (`45ed214`).** `lexical_sql` joins the *same* per-profile
      vector table `knn_sql` joins and filters on the *same* three denormalised
      columns, so scope parity is structural rather than remembered. The test
      was mutation-checked: with the chunk-profile filter neutralised it fails,
      so it tests the filter and not the join. The correctness
      risk of the whole step: the vector arm is implicitly scoped because each
      profile owns its own vector table, but `chunks_v2` holds chunks from
      *every* revision including superseded ones. The lexical query must filter
      to the active revision's chunk scope or search will return documents the
      vector arm structurally cannot.
      *Exit:* a test that seeds a superseded revision's chunks and proves they
      are never returned.

   iv. ~~**Wire in `hybrid.combine`.**~~ **DONE 2026-09-07 (`03f443d`).**
      Verified on the live corpus: a semantic query stays vector-led and
      unchanged, a rare token is lexical-led, `--json` parses.
      `SearchResult.distance` became `float | None` — a lexical hit has no
      distance from the query vector, and NaN would have serialised to invalid
      JSON, breaking the documented `search --json | jq`. Routing decided by
      `looks_like_identifier`
      plus one document-frequency probe at threshold 20; config to set the
      threshold and to turn hybrid off.
      *Exit:* `cementic search` returns routed results on `papers`, and
      `./scripts/check.sh` is green.

   v. ~~**Stop the score column contradicting the order.**~~ **DONE — shipped
      in `64b24d4`**, which is where the four-survey prior-art review landed
      (`notes/design-hybrid-retrieval.html`): human-facing tools withhold a
      numeric score by default, machine-facing ones publish the fused value.
      `rank` on `SearchResult`, the terminal column shown only when
      `scores_explain_order` says it still explains the order, `--scores` to
      force it (Recoll's off-by-default `%R`), and `score_kind` naming each
      arm's own score in `--json`. README documents the behaviour.
      *Anti-scope, still live:* do not write the RRF sum into `score`. It is
      the Elasticsearch/LlamaIndex convention and it is monotonic, but it
      replaces a meaningful cosine 0.675 with a meaningless 0.0164 *including
      for pure-vector queries where nothing was fused* — a regression in the
      common case to fix a problem that exists only in the fused one.

   vi. ~~**README.**~~ **DONE 2026-09-07.** The search section gains what hybrid does, when the lexical
      arm leads, and that a single common word is not an identifier query —
      "Hochreiter" appears in 1,697 of 23,064 documents, so it routes to the
      vector arm and searching famous names will not behave like searching rare
      ones.

   *Anti-scope for all of 5b:* one index, one fusion rule, one threshold. No
   per-collection tuning surface, no query classifier beyond the two-line
   predicate, and no re-opening of RRF `k`.

## Execution order — CLI surface audit (2026-09-10)

Ran the audit `TODO.md` had gated on the hybrid thread closing. Full evidence,
including the `llm` comparison and the per-command usage matrix, is in
`notes/design-cli-surface.html` (gitignored). Surface as audited: **21 leaf
commands** against `llm`'s 9.

The rationale — why the audit cut nothing, what the boundary was, and which
directions are closed — is in "Decided — the CLI surface after the `llm` audit"
under Design decisions. This section is the order of work only.

**Ordering is not arbitrary.** Step 2 must precede step 4: `source_watcher` is
the *one* cementic module `cli.py` does not import today, and step 4 adds a path
to it. Done naively, step 4 undoes step 2. Step 3 must precede step 6, because
both change output contracts and step 3 is where the convention gets fixed.
Steps 5 and 6 are independent of everything else.

### Step 1 — fix the two documentation defects — DONE 2026-09-10

Shipped in the same commit as this section. README claimed `search --json` emits
six fields; it emits seven — `rank` was added by `64b24d4` (`search.py:202`) and
never reached the docs. And `scripts/verify_postgres_container.sh` called
`cementic status --doctor`, a flag removed when `doctor` was split out
(README:880 records the removal with no alias kept), so the script could not
have run.

### Step 2 — stop the CLI importing the database and HTTP layers

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

**What changes: 17 module-level import statements move inside the functions that
use them**, across `config.py`, `render.py`, `cli_shared.py`, `cli_collection.py`
and `cli.py`. Two of those are the crux: `config.py:27`
(`from sqlalchemy.engine import URL, make_url`) and `config.py:30`
(`from cementic.vector_store import HNSW_ITERATIVE_SCAN_MODES`). Verified
2026-09-10: `import cementic.config` alone pulls SQLAlchemy, and every light
command reaches `_get_config` or `resolve_config_path`, so **leaving `config.py`
alone caps the payoff at about 250 ms and fails the new guard test.** It is
all-or-nothing per dependency: one surviving edge re-imports the whole layer.

Two edges are not plain moves. `render.py:25`'s `DaemonHealth` is a runtime enum
comparison at `render.py:101`, so a `TYPE_CHECKING` guard does not work and it
needs a real function-local import. And `search.MAX_SEARCH_RESULTS` is consumed
at `cli.py:955` inside a `typer.Option(max=…)` default, evaluated at import
time — so it must move to a module with no cementic imports, or be inlined as
`max=50` with a test pinning the invariant.

**Hidden cost, and the largest one in this plan.** Five tests patch by module
attribute: `cementic.cli.Searcher`, `cementic.cli.list_collections`,
`cementic.cli.check_health`, `cementic.cli_shared.get_engine`,
`cementic.cli_shared.get_session_factory`
(`tests/unit/test_cli_performance.py:96, 112–116, 180–181`). Moving an import
into a function deletes the module attribute and `unittest.mock.patch` then
raises `AttributeError`. All five must be repointed at their source modules in
the same change.

*Exit condition:* a new structural guard beside the existing PDF one, asserting
that after `import cementic.cli` in a subprocess, none of `sqlalchemy`,
`requests`, `tiktoken`, `cementic.pipeline_worker`, `cementic.collections` or
`cementic.db` is in `sys.modules`. Structural, not clocked, so it does not flake
under machine load. Probe for `sqlalchemy`, not `sqlalchemy.orm`: the cost is in
`sqlalchemy/__init__.py`. Naming the cementic modules too makes a regression
point at the module that reintroduced it. Plus `./scripts/check.sh` green,
including the seven other budgets in `test_cli_performance.py`.

*Anti-scope:* move imports and repoint the five patch targets. Do not restructure
the module graph, do not defer `pydantic_settings` or `importlib.resources`
(that is the next ~400 ms and a separate, larger change), and do not make
`cli_collection` lazily loaded — `collection_app` is bound at `cli.py:79`, and
deferring it needs a lazy Typer group.

*Risk:* `config.py` already carries four function-local imports to dodge cycles
with `embedding_runtime` and `extract`. Moving an import *into* a function
cannot create a cycle, only break one, so the hazard is exposure rather than
creation. If `HNSW_ITERATIVE_SCAN_MODES` is relocated rather than
function-imported, it must land somewhere with no cementic imports, or it just
reshuffles the knot.

### Step 3 — machine-readable output for the three query commands

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

### Step 4 — promote the runner subcommands

`cementic.runner` already has both commands with foreground semantics and a
deliberate non-zero exit on fatal startup (`runner.py:93, 141`), specifically so
a service manager can see it. This is surfacing existing code.

*What changes.* Expose them as a `cementic run watcher` / `cementic run worker`
group, then repoint `packaging/systemd/cementic-worker@.service:19` and
`cementic-watcher@.service:26`. A group rather than two top-level commands,
because the surface is already the audit's weakest point and this adds one entry
rather than two.

*Risk, and the reason for the ordering.* `cli.py` does not import
`source_watcher` today. Mounting the runner eagerly would pull it in and undo
step 2, so the group must load its implementation inside the command body.

*Anti-scope:* do not change what the workers do, do not add a `--foreground`
flag to `start`, and do not make `start` spawn the new commands — the supervisor
matches processes by command line (`supervisor.find_pids_by_cmdline`), so
changing the spawn string is a separate change with its own failure mode. Leave
the user's own `~/dotfiles` unit for last, and keep it pointing at the
checkout's interpreter: that pins the PyMuPDF version in the extractor
fingerprint, and switching to the installed snapshot re-versions the corpus.

*Exit condition:* `cementic run worker --collection papers` starts and stops
cleanly in the foreground, the packaged units start under `systemctl --user`,
and the step-2 guard test still passes.

### Step 5 — rename the filter's chunk index key

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

### Step 6 — a document id in search results

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
stable because path and document are one-to-one.

*Exit condition:* `cementic search Q -c papers --json | jq -r .document_id`
returns ids, and the README field count matches the emitted keys — the defect
step 1 fixed was exactly that count drifting.

*Anti-scope:* one id field. Not the artifact path, not the chunk index, not the
revision label; each is a separate argument about what a result is for.

*Anti-scope for the whole thread, so nobody re-derives it:* no ingest command
for embedded JSONL and no search-by-vector (the composability decision above
says why — and note it does not reach `cementic add`); no deletion of
`start`/`stop` or the supervisor; no dropping of PostgreSQL or the daemon to
match `llm`'s zero-dependency shape, which is the engine and not the interface;
and no unifying of the `--force` short forms.

## Execution order — v1.5 migration: CLOSED, the gate failed (2026-09-05)

The migration this sequenced is **not happening**: v1.5 retrieves measurably
worse than v2-moe on this corpus. Steps 1 and 2 ran; 3 to 6 are void. Kept
short as the record of how it closed; the reasoning is in the migration section
below, and `TODO.md` holds the feature backlog as before.

1. **Merge the canary.** DONE -- `06f7d62` on `main`.
2. **Compare v1.5 against v2-moe.** DONE, and it is the reason this list is
   closed. 1,000 papers, 2,000 body chunks, title-to-body retrieval, both
   models embedded fresh in one session with the stored index vectors as a
   control arm (`~/.cache/cementic-igpu/calibration/compare-models.sh`):

   | arm | recall@1 | recall@10 | MRR | mean margin |
   | --- | --- | --- | --- | --- |
   | v2-moe (current) | 0.7170 | 0.8870 | 0.7797 | 0.0546 |
   | v1.5 (candidate) | 0.6590 | 0.8450 | 0.7267 | 0.0279 |
   | v2-moe from the stored index (control) | 0.7200 | 0.8860 | 0.7812 | 0.0545 |

   v1.5 is worse on every measure: -8.1% recall@1, -6.8% MRR, and it separates
   the right paper from the best wrong one **half** as well (-48.9%). Paired
   over 1,000 queries it ranks the paper higher on 133 and lower on 245 (exact
   sign test p = 8.9e-09), and wins the margin on 299 against the control arm's
   own 523 -- ten standard deviations below the null the control measures. The
   control arm also lands on top of fresh v2-moe, which is what says the
   comparison itself is sound.

3-6. **Void.** `chunk_size` re-derivation, the punctuation filter and OCR
   removal go back to parked: each costs a full rebuild and there is no longer
   a rebuild to ride on. The five-day re-embed and its five days without search
   are not being spent.

## Design decisions and open items

### Decided — the CLI surface after the `llm` audit (2026-09-10)

Evidence, the full command inventory and the `llm` comparison are in
`notes/design-cli-surface.html` (gitignored). The steps are in "Execution order
— CLI surface audit". This block holds only what a later reader would otherwise
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

### Decided — the full-text index reports its own absence (2026-09-12) — DONE

Found by sweeping this document against the code: step 5b.ii above claimed
`cementic doctor` reported the lexical index, and nothing did. Three gaps, one
per layer, and they compounded — the index could be absent, unreported, and
unbuildable at the same time.

**Nothing could build it.** `build_lexical_index` had zero callers.
`ensure_lexical_index` deliberately skips a populated `chunks_v2` (building
there blocks worker startup for minutes), so a corpus indexed before hybrid
landed could not get the index from any command, only from hand-written Python.
It is now reached by `cementic collection reindex`, which is where a user
already goes when an index is wrong. *Rejected:* a new leaf command. The CLI
audit named the surface the weakest point in the design, and this is a
once-per-corpus repair. *Also rejected:* making the index per-revision and
reconciled like the ANN index — 5b already settled that a text-derived index
cannot depend on a model, so it stays a table-level index that this command
merely ensures.

**Nothing reported it.** `doctor`'s `lexical_index` check names the state
rather than a boolean, because the three cases need different actions: missing
on a corpus that holds chunks is a warning with the repair command; missing on
an empty table is fine, since the next startup builds it for free; and
`search.hybrid = false` makes it moot, so warning would be noise. A warning
rather than a failure, like `stranded_chunkings`: the installation is fit to
run and a corpus inside it needs one command.

**Presence was the wrong question, which is the part worth keeping.** An
interrupted `CREATE INDEX CONCURRENTLY` leaves an index that `pg_indexes`
lists and the planner refuses to read, so every exact-word query silently
falls back to scanning 2.3M rows — a check that asked only "is it there"
would have passed on it. Worse, `CREATE INDEX CONCURRENTLY IF NOT EXISTS`
matches on the *name*, so the retry no-opped against the leftover and the old
code needed two runs to recover: one to drop, one to build. It now drops an
invalid index before rebuilding. Pinned by a PostgreSQL test that marks a real
index invalid in the catalog, mutation-checked by removing the drop — without
it the test fails with the two-run `RuntimeError`.

### Decided — hybrid lexical + vector retrieval (2026-09-07) — DONE

**The mechanism is confirmed and large, and the fusion rule that settles it
shipped 2026-09-09.** A dense
embedding compresses a chunk into one vector, and a rare exact token — an
author surname, an acronym, an equation label — is not recoverable from it.
Measured on `papers`, n=150 per set, predictions registered before the run
(`notes/design-hybrid-retrieval.html`):

| set | arm | recall@1 | recall@10 |
| --- | --- | --- | --- |
| A rare exact tokens | vector | 0.020 | 0.047 |
| | lexical (ceiling) | 0.713 | 1.000 |
| | hybrid RRF | 0.047 | 1.000 |
| B semantic title→body | vector | 0.713 | 0.787 |
| | lexical | 0.173 | 0.207 |
| | hybrid RRF | 0.540 | 0.840 |

Vector-only found the right paper 7 times in 150. Hybrid found it every time:
+0.953 recall@10 against a pre-registered build threshold of +0.15, with the
falsifier (vector ≥ 0.85, which would have killed the feature outright) never
in sight.

**Cost is not a constraint.** A GIN index on `to_tsvector('english', content)`
built concurrently in 8.3 min and occupies 617 MB; lexical queries run in
1–173 ms. The 0.98 GB projected from a 100k-chunk sample was 60% high — GIN
posting lists compress better at corpus scale.

**Rank 1 was the hard part, and it is settled.** Plain RRF at k=60 drops set B
from 0.713 to 0.540 and reaches only 0.047 on set A where the lexical arm alone
gets 0.713 — each arm alone beats their combination at the top slot, which is
the slot a search tool is read from. Steps 1 and 2 established why: the tie is
not a bug, because RRF is symmetric and both arms' rank-1 documents score
`1/(k+1)`, so there is no neutral tie-break and choosing one *is* a bet on an
arm. The rule that shipped bets per query rather than globally.
`_lexical_should_lead` (`search.py:422`) gives rank 1 to the lexical arm only
for a single token that is syntactically identifier-shaped and matches at most
`search.lexical_lead_max_documents` documents; fusion owns the rest of the list.

**Risk — the precondition is unmeasured.** All of this assumes real queries are
identifier-shaped. Nobody has looked at one; sets A and C are synthetic by
construction. If such queries are a small fraction of real use then this is a
well-measured solution to a problem the corpus does not have, and set A's 0.047
does not change that. **This gated step 5, which shipped anyway**, so the gate
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
measurements are reproducible only from their description.

**Two performance traps, both measured rather than reasoned**, kept because
each cost an hour: `select distinct document_id … limit n` makes the planner
walk `ix_chunks_v2_document` and apply the text match as a filter, never
touching the GIN index — a token matching *nothing* then costs 8 s+ to prove
absence, and `enable_seqscan = off` does not fix it because the bad plan is an
index scan too. And `ts_rank` over every match of a common term recomputes a
tsvector per row and hangs for minutes; ranking must happen over a capped
candidate pool.

### Decided — what makes two rows the same document (2026-09-09) — DONE

**Implemented on `claude/watcher-document-identity` (four commits).** Written
after the live corpus held 46,139 documents where 23,077 belong, and 18,103 of
them matched nothing in search. Both halves of that came from the same place:
identity and the work attached to it are keyed on the path, and nothing checked
that the key still meant what it meant when it was written.

**Chunks may not be deleted without invalidating the chunking.** `_step_chunk`
re-claims a `done` chunking only when its `source_content_hash` differs from
the extraction's, so a chunking left with a current hash and no chunks is never
revisited. `0deeb88` fixed that for the failed-re-extraction purge; the
watcher's delete purge had the same hole, and a delete-then-create of an
unchanged file — a sync client, an editor renaming a temp file over the
original — is an ordinary event, so it fired 18,101 times. The rule is now: any
code path that removes chunks clears the hash in the same statement.

**A document is the same document when it is the same file, not the same
path.** A watched root given as a symlink is stored resolved, so moving a tree
and leaving a link behind re-registers every file under a path cementic has
never seen, and the pre-move rows keep resolving through the link so nothing
retires them. Registration now repaths instead of inserting a twin.

*Matching the moved row by content hash was tried first and is not enough.*
It assumes the corpus only moved. On this corpus 3,338 papers had also been
rewritten between the two scans, so their recorded hash no longer identified
the file their row still pointed at, and a rescan created 7,512 duplicates
before it was stopped. The watcher resolved the root itself, so it can name the
pre-move path outright — an exact lookup on the unique index, indifferent to
how much the file has changed. The hash probe stays as the fallback for a move
no root alias explains.

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

**The silence cost more than the bugs.** Both instances of the lost-chunk class
were invisible for weeks because `cementic status` read 100% chunked
throughout. `cementic doctor` now counts chunkings that are done and own no
chunks (0.17s against 46k documents), so a third path lands in a report rather
than nowhere.

### Decided — external command extractor (2026-09-03) — DONE

**Shipped in `935bf45`** (registry entry, `[extraction.commands]` with its
paired `command_versions`, the `extraction_commands` doctor check, tests).
Retained for the rationale; nothing below is outstanding. Extraction was the
last subsystem that hardcoded
Python libraries and grows a boolean per library feature. Embeddings already
work the other way: `daemon_command` names a binary, cementic spawns it and
consumes its output. A `command` extractor makes extraction symmetric — text
comes from an operator-named argv, so OCR, `pdftotext`, `docling` and anything
else are config, not dependencies. Built ahead of demand deliberately: the
user's call, on the grounds that the seam is cheaper to add now than to retrofit.

**Contract, kept narrow on purpose:** argv in, text on stdout, nothing else. A
tool that writes files instead (`ocrmypdf`) gets a two-line wrapper — that is
the composition boundary, not cementic's problem. A wide contract is what turns
an early abstraction into debt.

```toml
[extraction.backends]
pdf = "command"
[extraction.commands]
pdf = ["pdftotext", "-layout", "{path}", "-"]
[extraction.command_versions]
pdf = ["pdftotext", "-v"]
```

**`version_command` is required, because version flags cannot be guessed.**
Probed on this host: `pdftotext --version` treats the flag as a filename,
prints an I/O error and **exits 0**, while `pdftotext -v` works; `gs --version`
and `mutool -v` disagree again. So neither the flag nor the exit code is a
reliable signal, and a guessed flag would record a constant error string as
"the version" — never changing on upgrade, silently disabling the guarantee.
Its output enters the extraction payload so a tool upgrade re-versions
revisions, the same property `extraction_libraries` gives pymupdf. Residual
risk: a wrong flag that prints a constant cannot be detected automatically, so
`doctor` prints what was recorded for a human to check.

**Probe cost is a non-issue.** `build_extractor_profile_payload` is reached only
via `get_or_create_extractor_profile` ← `revisions.py:229` ← revision creation.
Once per revision, not per document; no cache needed.

**Fingerprint-neutral by gating.** Both new payload keys appear only when a
command backend is configured, so existing fingerprints do not move and no
corpus rebuilds. Same trick as the conditional rapidocr entry in `profiles.py`.

**`supported_extensions()` gains a `Config`.** The command extractor's
extensions come from the `[extraction.commands]` keys, otherwise it could never
add a file type cementic does not already know — its best use. Ripples to
`source_watcher.py:350`, `:596` and `:700`; `DocumentEventHandler`
(`source_watcher.py:328`) takes the resolved
set as plain data, matching how `ignore_directories` is already passed. The
command extractor is never a fallback: it must be named in `backends`.

### Rejected by measurement — migrating `papers` to v1.5 (2026-09-05)

**Decided, then reversed the same day by the gate that was built to test it.**
The candidate was `nomic-embed-text-v1.5`; it retrieves worse than the v2-moe
already in place (numbers in the closed execution order above), so `papers`
stays on v2-moe and the three rebuild-blocked changes stay parked. Everything
below is kept because it is what a *future* model swap costs, and because the
reasoning that pointed at v1.5 was sound right up to the measurement -- which
is the argument for keeping the gate cheap and running it first.

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

Already done, and it needed no rebuild: **splitting over-budget chunks at embed
time** (`c99d84b`, 2026-09-04). Chunking and both profiles were untouched, so
requeueing the failed rows was enough — chunks that had no vector gained one,
which is additive. The splitter halves on whitespace, then on characters,
because dot-leader pages are frequently one whitespace-free run. Roughly 1,200
of the affected chunks are genuinely dense content (code, maths, long structured
titles); the rest are contents pages that the punctuation filter above should
stop creating at all.

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

### Decided — detect server drift with a canary, not a fingerprint (2026-09-05)

**Implemented 2026-09-05 (`canary.py`, `doctor.py`'s `embedding_canary` check,
`embedding_profile_canaries`). Supersedes the open item that asked whether to
record the llama.cpp build in the embedding profile, and corrects its central
claim.** That note said a server upgrade is "usually numerically a no-op".
Nothing had measured it, and as stated it is false: new kernels and different
reduction orders move the last bits by construction.

Measured 2026-09-05 against the live server (`b10605-a130532ae`,
nomic-embed-text-v2-moe Q8_0, Vulkan iGPU), embedding one text three ways over
`/v1/embeddings`:

| Same text, same model, same build | Bitwise identical | max abs diff | cosine |
| --- | --- | --- | --- |
| the same request, twice | yes | 0 | 1.000000000000 |
| alone vs. first of a 2-text request | no | 7.5e-9 | 1.000000000000 |
| alone vs. last of a 32-text request | no | 2.3e-3 | 0.999796 |

**The corpus is already not bitwise reproducible, and no fingerprint can make it
so.** The worker submits `pipeline_worker.batch_size = 32` texts per request, so
a chunk's exact vector depends on which 31 others rode with it — on arrival
order, and on where a restart landed. What the fingerprint guarantees is
therefore not "the same bits" but "the same vector space, comparable distances",
and 2.3e-3 of wobble does not threaten that.

That reframes the question. It is not *does an upgrade change the numbers*
(everything does, including doing nothing) but *does it change them by more than
the noise the corpus already carries*. Two classes of llama.cpp change, which a
build string cannot tell apart:

- **arithmetic** — new kernels, different reduction order, quantization
  rewrites. The same magnitude as the batching wobble above. A rebuild buys
  nothing.
- **semantic** — a tokenizer fix, a pooling-type or normalization default, an
  attention-mask fix for embedding models. These change the function rather than
  its rounding, and would move cosine far below 0.999. A rebuild is mandatory.

**Decision: `cementic doctor` grows a behavioral canary, and the server build is
recorded beside it as evidence rather than as identity.** A revision stores a
handful of fixed reference texts with their vectors; doctor re-embeds them **as
one fixed request — same texts, same order, same count** — and reports the
cosine against the stored ones. The first row of the table is what makes this
work: a fixed request is bitwise reproducible within a build, so any difference
at all is the server's doing, and the size of it says which class the change is.
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

**Threshold: cosine 0.99 — and the "no threshold at all" plan above was
wrong.** Cross-build calibration ran b10605 against b10818 (213 builds and ~12
days later), same model, same flags, the same 200 real `papers` chunks in the
production 32-per-request shape, on both backends:

| Backend | Result |
| --- | --- |
| CPU (agent sandbox, no `/dev/dri`) | 200/200 bitwise identical, max abs diff 0 |
| Vulkan iGPU (`Intel(R) Graphics (MTL)`) | 200/200 bitwise identical, max abs diff 0 |

That reads as "zero drift, so compare exactly", and it is the wrong conclusion.
Both runs replayed the *same traffic sequence* against a fresh server, holding
fixed the one variable that actually moves vectors. Running the implemented
canary against the live server exposed it on first contact: replaying the
canary after an unrelated 3-text request gives cosine 0.999908 instead of 1.0,
and after an 18-text request 0.999887, while back-to-back replays with nothing
in between are exact. llama-server packs concurrent slot work into unified
batches (`kv_unified = true` in its own startup log), so reproducibility is a
property of the request *and its predecessors* — which no client can fix, and
which `doctor` cannot avoid, since its own daemon probe embeds first.

The floor is therefore measured rather than absent: scheduling noise reaches
2.0e-4 in (1 - cosine) and request composition 2.3e-3, while a tokenizer,
pooling or normalization change lands two orders of magnitude below. `0.99`
sits in that gap, ~100x above the worst observed noise, deliberately
conservative because both bounds come from a handful of sampled shapes rather
than a derivation. What the cross-build table still establishes is that no
*semantic* change happened across those 213 builds; that conclusion never
depended on the traffic sequence.

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

### Decided — query-first embed scheduling (2026-09-01) — DONE

**Both layers shipped**: sub-batching as `pipeline_worker.embed_submit_batch_size`
and the search lease as `db.SearchActivity` / `search.record_search_activity`,
consumed by `pipeline_worker`'s `_wait_while_search_is_active`. Retained for the
measurements and the parked triggers. A search query landing mid-import waits on the
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
filename: Batch C (`77e0e3f`) made it a content digest (`profiles.py:170`).
*Which text policy applies* is still a filename match (`embedding_text.py:33`),
deliberately so — that half is the fifth-review residual below, and it is the
only live part of this item.

Of those, the `status` half of the `check_health` tradeoff **was** reopened and
fixed — `status` no longer blocks 120 s to return an answer the pid file already
had. Two more were reopened and are now also closed; both entries below were
still written as open when the sixth review (2026-08-23) checked them against
the code, which is why they carry their resolution inline:

**Correction 2026-09-06:** the claim above that `status` no longer blocks is
not true of the code today. Seven timed runs of `cementic status -c papers`
took 12–128 s at 0–2% CPU, on this build and on an install predating it. What
was fixed was one path; something else in the command still waits. Queued in
`TODO.md` as a root-cause item.

**Root-caused and fixed 2026-09-07.** The remaining wait was never a timeout:
it was one query. The embedding counts reached the collection and the two
profiles by joining `chunk_embeddings` -> `chunks_v2` -> `chunked_documents` ->
`extracted_documents` -> `source_documents`, so counting them scanned all of
`chunks_v2` — 2.6 GB of chunk text read to fetch three integers a row, with the
hash join spilling to 32 disk batches. Per-statement timing of one run: 7.96 s
of a 9.08 s total in that single statement, the rest under 0.1 s each; the CPU
sat at 0–2% because the work was in the Postgres backend, not the CLI. The
count now reads the denormalised columns `chunk_embeddings` already carries for
`_step_embed`'s claim (`embedding_scope_denormalised`, `revisions.py`).
Interleaved on the live corpus, same connection, both forms returning the same
row: 1.15 s joined against 0.20 s flat warm, 7.96 s against 0.21 s cold.
`cementic status -c papers` end to end: 1.97-2.27 s across five warm runs,
against 12-128 s before. A cold cache still costs -- one run just after the pg
suite took 10.1 s -- so the tail is smaller, not gone.

- ~~**`check_health` still calls a live-but-broken daemon healthy.**~~
  **RESOLVED 2026-08-18** by the release plan's batch 1a (`04e974a`):
  `DaemonHealth.WEDGED` (`embedding_runtime.py:758-772`) plus the two-stage
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
  re-embedded once, and Batch C was that change. `profiles.py:230` now records
  the deliberate absence. It correctly stays in the *runtime* fingerprint, where
  it is a launch argument.
- **Naive `TIMESTAMP` columns.** Zero readers today (one write, no comparison,
  no display). The fix is a column-type change in a project whose only schema
  mechanism is `create_all`, so old and new databases would diverge with nothing
  to reconcile them. Revisit if anything ever reads these columns.
- ~~**`connect_args` / `gssencmode` on non-psycopg URLs.**~~ **RESOLVED, and
  this entry was doubly stale when checked on 2026-09-12.** The
  `startswith("postgresql://")` prefix match it described is gone: `db.py`
  parses the URL instead, and says so in the comment beside it, so
  driver-qualified URLs like `postgresql+psycopg2://` are handled. The entry
  also claimed it had been "queued in `TODO.md`", and no such item was ever
  there.
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
`status --verbose`, `render.py:260`), non-UTF-8 filenames (the four bare
`.decode()` calls at `source_watcher.py:411`, `:420`, `:442`, `:444`, then
`render.py:157`), and paths over ~2,600 bytes (btree limit on
`ix_source_documents_collection_source`, `db.py:72`).

Live, in rough order of how likely they are to bite:

**Closed 2026-09-06.** Six of the eight are closed. Five went in one pass: the unlimited
`status --verbose` (now `--limit`, default 20, failures first), the
rounding that reported 100.0% with work outstanding, the unbounded
whole-corpus delete (batched at 500 documents), `chunk_text`'s unkillable
quadratic (refused above a 100,000-character unbroken run), and
`extract_pdf_markdown`'s `use_ocr` default. `_step_embed` needed nothing:
it had already been fixed, and this entry had gone on describing a defect
the code closed -- which is why the two that remain say what is still true
of the code rather than what was once observed.

`extractor_registry_payload` moved to `TODO.md`'s parked list. It is real,
but narrowing the payload moves the fingerprint exactly as adding an
extractor does, so the fix costs the 2.3M-vector rebuild it exists to
avoid; it is only free on a day something else is already rebuilding.

Still live:

- **A revision can reach `ready` mid-initial-scan** (`revision_is_complete`,
  `pipeline_worker.py:263`): the watcher registers documents one at a time with
  no scan-complete marker, so a drain of the first N looks complete. Promotion
  re-checks and refuses, so this misleads rather than corrupts. **Narrowed
  since it was written:** the all-zero case — the fresh-`cementic start` race
  where the worker's first pass beats the watcher's first document — is now
  refused outright and documented in the function. What remains is the first-N
  drain of a scan already under way.

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

