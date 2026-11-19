# cementic — audit report (2026-11-17/18)

A self-contained record of what is wrong with this codebase, how to see it, and
what to do about it. Written so that a reader who opens only this file can
reproduce, judge and repair every item. `PLAN.md` owns *what order to work in*
and the design rationale; this file owns *what is broken and how to fix it*.

**Status at the time of writing.** Three sequential passes: a six-lane audit, a
sweep of everything the lanes produced, and a fresh-eyes review of the repairs.
The branch `pi/audit-handoff` carries the repairs marked **Fixed** below; the
rest are open. Nothing was pushed, merged, or applied to the live corpus.

**Read this first if you are picking the work up:** §1 (how the pipeline works,
so the file:line references mean something), §2 (the grades — do not trust an
item beyond its grade), §11 (what was never verified).

---

## 1. The pipeline, in the terms this report uses

Six tables, one row per stage, each keyed to the stage before it:

```
source_documents      one row per file on disk (path, file_hash, status)
  extracted_documents one row per (document, extractor profile): text → content_hash
    chunked_documents one row per (extracted, chunk profile): owns N chunks,
                      and carries source_content_hash — the content hash its
                      chunks were built from
        chunks_v2     the chunk text
          chunk_embeddings + embedding_vectors_p<id>   one vector per chunk
pipeline_revisions    a collection's build; status building/ready/active
```

Four consequences this report keeps referring to:

- **A revision is "complete" when the counts add up**: every document extracted,
  every extraction chunked, every chunk embedded. The equalities are in
  `revision_is_complete` (`pipeline_worker.py:271-291`).
- **The `*_scope` predicates** (`revisions.py:95-183`) are the single definition
  of *what counts*: the worker, `cementic status` and `promote` all use them.
  `chunked_scope` requires `ChunkedDocument.source_content_hash ==
  ExtractedDocument.content_hash`, so a chunking whose hash does not match its
  extraction is counted as *neither* done nor failed.
- **The claim re-opens on a hash mismatch.** `_step_chunk` re-selects a chunking
  when its `source_content_hash` `IS DISTINCT FROM` the extraction's — so NULLing
  the hash is what re-opens work, and *emptying* a chunking without NULLing its
  hash strands it forever.
- **Search applies no freshness filter.** It reads vector rows by
  (collection, profile) only (`vector_store.py:126-131`, `:165-178`), so a stale
  row that was never deleted is *served*. Purging at the right moment is the only
  thing standing between a re-extraction and stale results.

The invariant this audit found violated in three places, and fixed in all three:

> **A chunking that is counted as chunked must own chunks. Emptying one must
> invalidate it (`source_content_hash := NULL`) in the same transaction as the
> delete, or the work never re-opens.**

---

## 2. Grades, and how to check anything in this file

Every item carries one:

| Grade | Means |
| --- | --- |
| **executed** | The code was called on an isolated input: in-memory SQLite, a scratch `*_test` PostgreSQL database, an injected exception, `CliRunner`, or stubbed dependencies. Reproducible from what this file prints. |
| **source-traced** | The failure path was read and shown reachable, but not run. |
| **asserted** | Neither. Treat as a hypothesis. |

Neither grade means *repaired*, and **executed on SQLite is not executed on
PostgreSQL** — see §10.

The gates, from the repo root:

```bash
./scripts/check.sh                 # lockfile, ruff, mypy, unit, integration, integration-pg
                                   # (reports a missing Postgres as SKIPPED, exit 2, not as pass)
```

Individual pieces, and the two things `check.sh` does not do:

```bash
.venv/bin/pytest tests/unit tests/integration -m "not pg" -q
.venv/bin/ruff check src/ tests/ scripts/measure_retrieval_quality.py   # scripts/ is NOT in check.sh
.venv/bin/mypy src/                                                     # nor is scripts/
```

PostgreSQL tests need a server with pgvector + pgvectorscale. They refuse to run
against a database whose name does not end in `_test`, and each lane in this
audit used its own to avoid collisions:

```bash
CEMENTIC_DB_URL='postgresql+psycopg://cementic:cementic@127.0.0.1:5432/audit_test' \
  .venv/bin/pytest tests/integration -m pg -q
```

---

## 3. Fixed in this branch (verified; do not re-fix)

Each has a regression that was **red against the unpatched source** — proved by
stashing the source change and re-running — and green after.

**F1 · Edit-and-revert stranded a document while the revision reported complete.**
`_purge_superseded_chunks` (`pipeline_worker.py:293-334`) deleted chunks without
invalidating the chunking: after A→B→A between chunk passes, the chunking stayed
`done` with hash A, owned 0 chunks, `_step_chunk` refused it, and
`revision_is_complete` returned True. **Fixed** by NULLing the hash of exactly the
chunkings whose chunks were deleted, in the same transaction. A *failed* chunking
keeps its hash — see §5 open item on that exemption. Test:
`TestEveryPurgeReopensTheWork::test_a_re_extension_that_reverts_does_not_strand_the_document`.

**F2 · A failed extraction write-back kept serving the superseded text.**
`_fail_artifact` (`pipeline_worker.py:788-843`) stamped the row terminal without
purging, because the purge lived in the transaction that had just rolled back: the
old content hash, old chunking and old chunks stayed live and searchable, and
`chunked_done + chunked_failed == extracted_done` was satisfied as `0 == 0`.
**Fixed**: for an `ExtractedDocument` it now clears `content_hash` and calls
`purge_all_chunks` in the same session that stamps it failed. Test:
`test_a_failed_extraction_write_back_stops_serving_the_old_text`.

**F3 · A chunking write-back failure wedged the revision.** `_fail_artifact` on a
`ChunkedDocument` set status failed, which dropped `total_chunks` to 0 while
`done_embeddings` stayed 2 — so `done + failed == total_chunks` was unreachable,
the row was never re-claimed, and no `last_error` reached `status`.
**Fixed**: the failed chunking's chunks are deleted with it, so both sides settle
to 0 and the failure reports as `chunked_failed`; `requeue_interrupted_artifacts`
re-opens it on the next `cementic start`. Test:
`test_a_failed_chunk_write_back_leaves_a_settleable_revision`.

**F4 · The retrieval harness paid for misses.** `gold_rank_and_margin` returned
`len(pool)+1` for an absent gold, so a one-document pool scored positive
recall@10 for a query no arm answered; a refused query did the same in
`score_arm`; `shared_pool([])` raised `IndexError`; `score_arm(keep=[])` raised
`numpy.AxisError`; `summarise([])` divided by zero. **Fixed** with one absence
marker (`UNRANKED`), zero credit for it in `summarise`, and clean empty
behaviour. `gold_rank` now returns the same marker instead of `inf`, because
`1e6 < inf` would have let a caller mixing the two score a miss as a win.

**F5 · The ablation had no baseline.** Every policy in `POLICIES`
(`measure_retrieval_quality.py`) fused both arms, so *"hybrid beats plain vector
top-k"* — the project's central claim — had no row that could test it. **Fixed**
by adding `vector_only` and `lexical_only`. First measured result, over the arms
cached 2026-09-23 (pre-fallback snapshot; all rows scored on the same arms):

| set (n=60) | shipped | vector_only | lexical_only | shipped vs vector_only | shipped vs lexical_only |
| --- | --- | --- | --- | --- | --- |
| rare-token | 0.633 r@1 | 0.017 | 0.650 | 56 better, 1 worse, p=8e-16 | 19 worse, p=4e-06 |
| title | 0.783 | 0.783 | 0.050 | tie, p=0.25 | 48 better, p=7e-15 |
| phrase | 0.117 | 0.117 | 0.967 | 50 better, p=2e-15 | 53 worse, p=2e-16 |

Read it twice. Hybrid beats vector-only decisively; **the bare lexical arm beats
it on phrases**, which is either a routing defect or the phrase set's
construction — open item **O12**.

**F6 · The fuzzy table could not report the shipped rule.** `changed` counts
queries whose *ranking moved*; a typo query's relaxed lexemes usually hit
documents the vector arm already ranked, so the row read "fires on 0" while the
rule fired on most of them. **Fixed** with a separate `fired` column. (The first
version of this fix repeated the same defect — it is recorded here because the
next reader should expect this class of mistake in this file.)

---

## 4. Triage table

| # | Severity | What | Where | Grade |
| --- | --- | --- | --- | --- |
| O1 | high | A deadlock turns one document terminal where a retry would have worked | `pipeline_worker.py:738-748` | executed (state) |
| O2 | high | A revision can reach `ready` before the initial scan finishes | `pipeline_worker.py:271-291` | source-traced |
| O3 | high | Doctor reports `ok` while a required check failed | `doctor.py:170-177,263-272` | executed |
| O4 | high | `collection remove <typo> --force` exits 0 | `cli_collection.py:50-55` | executed |
| O5 | high | `status --json` exits 0 where text `status` exits 1 | `cli.py:629-665` vs `692-704` | executed |
| O6 | high | `cementic doctor` crashes on a config it accepted | `doctor.py:256`, `config.py:655-689` | executed |
| O6b | low | The documented external-server contract exceeds what is implemented | `docs/configuration.md:72-76` | source-traced |
| O7 | high | The rare-token/phrase eval sets are answerable by construction | `measure_retrieval_quality.py:1011-1020` | source-traced |
| O8 | high | Model bytes are not part of daemon identity | `embedding_runtime.py:160-174` | source-traced |
| O9 | med | Deferred cleanup can drop a live vector table | `collections.py:183-196,224-234` | source-traced |
| O9b | note | The failed-chunking hash exemption is deliberate — do not "fix" it | `pipeline_worker.py:322-340` | executed |
| O10 | med | Query-string passwords are not masked | `bootstrap.py:119`, `doctor.py:143` | executed |
| O11 | med | Extractors inherit the worker's database URL | `extract.py:529-545` | executed |
| O12 | med | The shipped routing rule loses rank 1 to the bare lexical arm | `search.py:642-676` | executed (F5 table) |
| O13 | med | The typo fallback cannot fire for a 1–2 word query | `search.py:110-135` | executed |
| O14 | med | `combine` resolves RRF ties toward the vector arm, against its docstring | `hybrid.py:78-95` | executed |
| O15 | med | The lexical candidate pool can be starved or over-saturated | `vector_store.py:165-178` | source-traced |
| O16 | med | `ts_rank` ties resolve arbitrarily, and fusion is rank-based | `vector_store.py:178` | source-traced |
| O17 | med | `stop` takes no lock; `start` does | `cli.py:797-880` vs `:452` | source-traced |
| O18 | med | `embedding status` exits 0 for states `doctor` fails | `cli.py:957-966` | source-traced |
| O19 | med | A config error prescribes a value the config rejects | `config.py:918-930` | executed |
| O20 | high | No chunk-delete path has a PostgreSQL cascade test | `test_revisions_pg.py:601-611` | source-traced |
| O20b | med | The invalidation invariant is tested at one door of six | `pipeline_worker.py:318,1003` etc. | executed |
| O21 | low | Index maintenance and recall are unmeasured | `db.py:855-913` | source-traced |
| O22 | low | Extraction limits: 1 h timeout, unbounded stdout, whole-file reads | `extract.py:247,263-268,291-293` | source-traced |
| O23 | low | The generated config is not the complete reference | `templates/config.toml` | executed |
| O24 | low | The pooled-vector formula has no oracle | `embedding_runtime.py:582` | source-traced |
| O25 | low | `status` and `doctor` describe the daemon differently, always | `status_service.py:436-482` | executed |
| O26 | low | The service story is half-documented, half-wired | `docs/operations.md:25`, systemd unit | source-traced |
| O27 | low | `scripts/` is in no lint or type gate | `check.sh`, `ci.yml` | executed |
| O28 | low | Five small contracts (version, classifiers, score, session.info, ABC) | see O28 | source-traced |
| O28b | low | Two doc contradictions (private repo; "about 2 GB" twice) | `docs/setup.md:27,8-9` | source-traced |
| O29 | ? | Eleven items carried from the previous audit, unverified here | see O29 | source-traced by that audit |

---

## 5. Data-loss and repair risk

### O1 — A database error becomes a permanent document failure

**Symptom.** A write-back that fails (deadlock, statement timeout, connection
reset, disk full) is stamped terminal by `_fail_artifact` instead of retried, so
one document is lost for the run where a second attempt would have worked.

**Why it matters.** F2 makes this honest — the old text stops being served — but
it does not make it rare. On a long import the difference is "retried and
succeeded" versus "reported failed until someone notices".

**Fix.** Classify the failure: serialization/deadlock and connection errors →
leave the row `pending`/`processing` for the loop's existing back-off;
data errors (a NUL byte, a value Postgres refuses) → terminal, as now. The seam
is `_fail_artifact`'s call sites in `_step_extract` and `_step_chunk`.

**Proves it.** One test per class: inject each exception type and assert the row's
status and that the lineage retries (or does not). **Anti-scope:** no generic
back-off framework, no new state machine.

### O2 — A revision can reach `ready` before the initial scan finishes

**Symptom.** `revision_is_complete` counts the documents the watcher has
registered *so far*; the watcher registers them one at a time with no
scan-complete marker, so draining the first N looks identical to finishing.

**Why it matters.** Promotion re-checks and refuses, and the all-zero case is
refused outright, so this misleads (status says ready) rather than corrupts. It
has been live since 2026-08-24.

**Fix.** A scan-complete marker the counts can see — a row, a timestamp, or a
revision field set when the initial walk ends. **Not** a timeout.

**Proves it.** A regression that starts a scan, drains the first N, and asserts no
`ready` transition; then finishes the scan and asserts one.

### O9 — Deferred cleanup can drop a live vector table

**Symptom.** `delete_collection_records`/`drop_orphan_vector_tables`
(`collections.py:183-196,224-234`), pruning (`revisions.py:691-745`) and
`drop_vector_table` (`vector_store.py:332-335`) compute droppability *before*
commit, then drop unconditionally. Another collection can adopt the same retained
embedding profile in between and lose its vector table while its work rows say
`done`.

**Grade.** source-traced; the interleaving is unexecuted.

**First check.** In an isolated `*_test` database: delete the last collection
using a profile and retain its cleanup result; create and populate another
collection on that profile; run the retained cleanup; assert the live vectors
survive. Cover pruning and promotion too.

**Fix (candidate, not decided).** Serialize reuse and drop under one
per-profile lock and recheck references inside it. A recheck without the lock
keeps the race. **Do not** build a lock framework before finding the smallest
shared boundary.

### O20 — No chunk-delete path has a PostgreSQL cascade test

**Symptom.** Deleting `chunks_v2` rows is supposed to remove their
`chunk_embeddings` and vector rows through `ON DELETE CASCADE`. SQLite enforces
no foreign keys unless `PRAGMA foreign_keys=ON` is issued and nothing issues it,
so the 1,508-test non-PG suite **cannot see** a missing or mis-scoped purge.
One cascade is checked against a real database — the watcher's
(`test_revisions_pg.py:601-611`) — and the three paths F1–F3 changed have none.

**Why it matters.** This is the residual on the three repairs: the SQLite
regressions prove the chunk delete and the settleability, not that the embeddings
and vectors went with them. It is the only check I would run before the next
`cementic start` applies these repairs to the live corpus.

**Fix.** For each path that deletes `Chunk` rows
(`pipeline_worker.py:318,1003`, `revisions.py:437,801`,
`source_watcher.py:358`, `collections.py:168`), a `-m pg` test asserting
`chunk_embeddings` and the per-profile vector rows are gone. Copy the shape of
`test_deleting_a_document_takes_its_embedding_rows_with_it`.

### O20b — The invariant is checked at one door, not all six

`test_no_door_leaves_a_counted_chunking_with_no_chunks` asserts the property for
the superseded purge only. The other four `Chunk` delete sites
(`revisions.py:437,801`, `source_watcher.py:358`, `collections.py:168`) were
*read* on 2026-11-17 and each invalidates its chunking, but no test walks them.
Fix: parameterise one test over the six sites, or a helper per site that takes
the door as an argument. **Proves it.** Deleting the invalidation from any one
door turns a test red.

### O-adjacent — the failed-chunking hash exemption (documentation only)

`_purge_superseded_chunks` deliberately leaves a *failed* chunking's hash alone.
Measured per state on 2026-11-18: load-bearing when the hash **matches** the
extraction (clearing drops `chunked_failed` to 0, and completion becomes
unreachable), inert when the hash is stale (uncounted either way). Keep the
exemption; the comment in the source says both cases. Listed so nobody "fixes"
it by clearing.

---

## 6. The surface can report something untrue

Each of these is a command telling you something false. They are cheap and
independent; none needs a corpus.

### O3 — `doctor` reports `ok` while a required check failed (executed)

`_active_extractor_profiles(conn)` (`doctor.py:170-177`) runs *after*
`extension_ok` is computed. If it throws, the `except` replaces
`checks["extensions"]` with a `(inspection)` failure and leaves `extension_ok`
True, so the verdict at `:263-272` never sees it. Reproduce:

```python
from unittest.mock import patch
from cementic.config import Config
from cementic import doctor
with patch.object(doctor, "_daemon_state", return_value=(True, True, None)), \
     patch.object(doctor, "_active_extractor_profiles", side_effect=RuntimeError("x")), \
     patch.object(doctor, "_embedding_server_check", return_value={"status": "ok"}):
    r = doctor.collect_doctor_report(Config())
print(r["ok"], r["checks"]["extensions"])   # True {'(inspection)': {'status': 'fail', ...}}
```

**Fix.** Compute the verdict from the checks after they exist, counting nested
statuses; make the exit code follow the verdict. **Proves it.** A test with a
raising inspection asserting `ok is False` and exit 1, plus a warning-only case
asserting `ok is True`.

### O4 — `collection remove <typo> --force` exits 0 (executed)

`cli_collection.py:50-55` prints `status: not found` and returns. Every sibling
(`promote`, `reindex`, `revisions`, `status`) exits 1 via
`require_known_collection` (`cli_shared.py:180-196`). Reproduce against any
seeded database:

```bash
cementic collection remove no-such-collection --force; echo $?   # 0
cementic collection promote no-such-collection;         echo $?   # 1
```

**Fix.** `raise typer.Exit(1)` in the not-found branch. **Proves it.** A CLI test
pinning the exit code for that branch (none exists today).

### O5 — `status --json` exits 0 where text `status` exits 1 (executed)

The human path exits 1 when the health probe raised (`cli.py:692-704`, with a
comment explaining exactly why). The JSON path only fails when the *data load*
failed (`render.py:414-417`), never when `health is None` or `db_reachable` is
false. Reproduce:

```python
from unittest.mock import patch
from typer.testing import CliRunner
from cementic.cli import app
with patch("cementic.status_service.check_health", side_effect=RuntimeError("boom")):
    print(CliRunner().invoke(app, ["status"]).exit_code)          # 1
    print(CliRunner().invoke(app, ["status", "--json"]).exit_code) # 0
```

**Fix.** Share one failure predicate between the two modes' exit decisions, keep
the JSON body parseable with `health.error` intact. **Proves it.** The two-mode
matrix in one test.

### O6 — `cementic doctor` crashes on a config it accepted (executed)

`ExtractionConfig(backends={"epub": "command"})` validates, then
`_extraction_commands_check` (`doctor.py:256`, body at `:361-379`) raises
`KeyError: 'epub'` outside any `try`, so the user gets a rich traceback.

```bash
printf '[extraction]\nbackends = { epub = "command" }\n' > /tmp/bad.toml
CEMENTIC_CONFIG=/tmp/bad.toml cementic doctor      # traceback, exit 1
```

**Fix.** Require both tables (`commands`, `command_versions`) at *config load*
whenever a backend is `"command"`, with an error naming the file type; fix the
template's example so it actually dispatches (`templates/config.toml`, dispatch
in `extract.py`). **Proves it.** `ExtractionConfig(backends={"epub":"command"})`
raises at load, and the template-derived example reaches the command extractor.

### O25 — `status` and `doctor` describe the daemon differently (executed)

`status --json` reports `embedding_healthy: true` beside `llama_daemon:
stopped`; `doctor` fails the same daemon ("running but not answering
embeddings"). Reproduced at a fully idle, healthy corpus (0 pending, 100%
embedded, no worker error), so it is steady state, not an incident artefact.

**Why it matters.** A user cannot tell from `status` whether their daemon is
usable, and the two commands disagree by design without saying so.

**Fix.** One field per question: *does something answer the configured
endpoint*, and *is it the daemon cementic manages*. Name the difference in the
message when they diverge. **Proves it.** A test pinning both fields for
(managed+answering), (unmanaged+answering), (managed+silent).

### O18 — `embedding status` cannot say "not ok" (source-traced)

`cli.py:957-966` exits 0 for `stopped` and `ambiguous`, while `doctor.py:225-233`
calls the same state `fail` and exits 1. A script has no exit-code way to ask
"is my embedding runtime usable?".

**Fix.** A flag on the existing command that makes the exit code follow the
state. **Not** a new command.

### O19 — A config error prescribes a value the config rejects (executed)

`CEMENTIC_LLAMA_N_CTX=8 cementic config show` exits 1 with *"Use chunk_size <=
0, or raise n_ctx"*, but `chunk_size` is `ge=1` (`config.py:450-455`). The
message is caught at load, names both fields, and is unusable.

**Fix.** Name an achievable value (`chunk_size >= 1`, or state that no valid
`chunk_size` fits and `n_ctx` must rise). **Proves it.** A test asserting the
message's suggested value loads.

### O26 — The service story is half-documented and half-wired (source-traced)

- `stop` deliberately leaves the embedding daemon running (`cli.py:784-800`), but
  the operations table names only watcher, worker and Postgres
  (`docs/operations.md:25`).
- `packaging/systemd/cementic@.service:82-84` hardcodes `127.0.0.1:11555`, so its
  wedge check probes nothing when `daemon_port` differs — the unit's own comment
  at `:36` admits it, the docs do not.
- `docs/setup.md:99` says the unit needs `llama_cpp.daemon_autostart = true`;
  the unit header (`:62`) says set it **false** when the server has its own unit.

**Fix.** Name what is left running in the operations table; read the port from
the config in the unit's check (or say it must be edited); qualify the setup
sentence to the bundled-server case.

---

## 7. Retrieval: what is evidenced, and what is not

### O7 — The evaluation sets are answerable by construction (source-traced)

`is_known_item` (`measure_retrieval_quality.py:1011-1020`) requires the gold
document to be an exact `plainto_tsquery` match **and** the match set to be ≤5
documents (`MAX_MATCHING_DOCUMENTS = 5`). The lexical arm matches with the same
predicate (`vector_store.py:165-178`). So 120 of the 360 checked-in queries
(`eval/queries.json`: 60 rare-token, 60 phrase) cannot produce a counterexample,
and *"rare tokens 0.047 → 1.000"* is a ceiling, not a rate over user queries.

**Fix.** Add kinds whose gold is **not** an exact FTS match: case/hyphen-joined
identifiers (`InstantNGP` → `instant ngp`), a paraphrase of a title, a
single-word typo, and a verbatim multi-word phrase whose tokens the text
configuration cannot match (which separates O12's two readings directly).
**Store them in a new file** — `eval/queries.json` is the control for every
historical number. **Proves it.** `build` emits the kinds with a builder test on a
scratch database, and `score` reports shipped vs `vector_only`/`lexical_only`
paired on them. **Anti-scope:** no `pg_trgm` without a GIN index — an unindexed
trigram scan was costed at ~879 s/query over 2.3M chunks.

### O12 — The shipped routing rule loses rank 1 to the bare lexical arm (executed)

From §3's F5 table: on the phrase set the lexical arm alone scores recall@1 0.967
against shipped's 0.117, and shipped loses 53 of 60 queries to it. Shipped gave
rank 1 to the lexical arm on only 59 of 360 queries.

**Two readings the data cannot separate**, which is why O7 is the next step:
either `lexical_should_lead` (`search.py:642-676`) is wrong for multi-word
queries a person types verbatim, or the phrase set is lexical-favourable by
construction and the rule is right for real use.

**Do not "fix" the routing before O7 lands** — a routing change measured on a set
that favours one arm is the failure this file exists to prevent.

### O13 — The typo fallback cannot fire for a 1–2 word query (executed)

`relaxed_tsquery` (`search.py:110-135`) with share 0.6 yields the input unchanged
at one lexeme and a conjunction at two — i.e. the same match set as
`plainto_tsquery` — so only ≥3-lexeme queries relax, and there is no
edit-distance path in `src/`:

```python
relaxed_tsquery(["kalmna"], 0.6)             # 'kalmna'          (no help)
relaxed_tsquery(["kalman", "filtre"], 0.6)   # 'kalman & filtre' (no help)
relaxed_tsquery(["kalman", "filtre", "desgin"], 0.6)   # OR-of-pairs (helps)
```

The measured 0.233 → 0.650 for typo'd phrases is real for 3+ word phrases. The
single misspelled word — the case a person searching from memory actually
produces — is unmeasured and unhelped.

**Fix.** Decide from a measurement, not taste: score a 1-word and a 2-word typo
set (O7's kinds), then either extend the fallback or document the limit as
accepted. **Anti-scope:** do not change exact-query or phrase behaviour.

### O14 — `combine` resolves RRF ties toward the vector arm, against its docstring (executed)

`hybrid.py:78-81` says ties are left to the caller; the `sorted` over a
vector-first dict silently picks one:

```python
combine([100,101,102], [200,201,202], lead="lexical")
# [200, 100, 101, 201, 102, 202]  -- lexical #2 loses position 3 on an exact 1/62 tie
```

Nothing pins the tie either way: `test_hybrid.py:41-52` asserts
`1/(RRF_K+1) == 1/(RRF_K+1)`, a tautology.

**Fix.** Either implement the promise (an explicit tie order, with a test) or
correct the docstring. Decide against O12's paired ablation for lexical-lead
queries, not by preference.

### O15 — The lexical candidate pool can be starved or over-saturated (source-traced)

`vector_store.py:165-178` takes a global unordered `LIMIT :pool`
(`search.lexical_candidate_pool = 2000`) *before* filtering to the collection and
revision, so another collection or an old revision can fill the pool with
out-of-scope rows. Separately, `probe_documents`/`lexical_should_lead`
(`search.py:142-145,645-670`) can saturate on one long document and read "rare"
when it is not.

**First check.** In an isolated database, populate more than a pool of
out-of-scope matches plus one known in-scope hit, and see whether the arm keeps
the hit. **Anti-scope:** raising the cap postpones starvation; naively moving the
filters can lose GIN index access (`EXPLAIN (ANALYZE, BUFFERS)` decides).

### O16 — `ts_rank` ties resolve arbitrarily, and fusion is rank-based (source-traced)

`vector_store.py:178` orders by rank with no secondary key. Ties are routine for
short chunks and single-term queries; fused scores are positions, so an arbitrary
tie order becomes the final order, and dedup keeps whichever tied chunk came
first — which becomes the shown `content`.

**Settle it first:** `SELECT count(*), count(DISTINCT rank) FROM (… @@
plainto_tsquery(…))` for a few queries; a ratio well below 1 confirms the ties.
**Then** add a deterministic secondary key (document id then chunk index) so the
order is at least reproducible — and note it changes ranking, so it lands after
the O7/O12 measurement.

### O21 — Index maintenance and recall are unmeasured (source-traced)

No `VACUUM`, `REINDEX` or `ANALYZE` anywhere in `src/`. The HNSW index is built
once (`db.py:855-913`) and rebuilt only on a method change or `--force`
(rebuild path in `collections.py`), so tombstones left by deletion and content
supersession accumulate unmeasured. DiskANN has no filter-aware scan:
`query_tuning_statements` (in `vector_store.py`) sets only `query_rescore` where
HNSW gained `iterative_scan`, so a selective collection filter on a shared vector
table can return short lists.

**Settle it:** measure recall of a churned index against a freshly built one on
the same rows, and measure the DiskANN short-list case. Then decide whether a
`REINDEX` step belongs in the documented maintenance path. (The measured
"index grows 82% under maintenance, recoverable in 6 min" note is in PLAN.md.)

---

## 8. Security and runtime

### O8 — Model bytes are not part of daemon identity (source-traced)

The daemon alias hashes the model **path** plus launch arguments
(`embedding_runtime.py:160-174`): `{model_path, n_ctx, n_batch, n_gpu_layers}`.
`model_content_digest` is called only when building a profile
(`profiles.py:195`). So replacing a GGUF in place reuses the running daemon while
the new profile claims the new bytes — vectors labelled as model B, produced by
model A. Related: the canary witnesses whatever server answered at capture time
(`canary.py:174-206`), so a pre-existing mismatch is baked into the reference and
later replayed as "reproduces exactly".

**Fix.** Carry the content digest into the runtime fingerprint payload so the
alias moves with the bytes, and keep profile identity as it is. Then re-check the
relocation case: identical bytes at a new path must still resolve without a new
vector identity, while a *different* file offered at the old path must not be
accepted under the old alias. **Proves it.** Replace bytes at the same path, index
one document, and assert a second profile/daemon rather than a silent reuse.

### O10 — Query-string passwords are not masked (executed)

SQLAlchemy masks the *authority* password and nothing else.

```python
from sqlalchemy.engine import make_url
u = make_url("postgresql+psycopg://u:SENTINEL_A@h:5432/d?password=SENTINEL_B&sslpassword=SENTINEL_C")
print(u.render_as_string(hide_password=True))
# postgresql+psycopg://u:***@h:5432/d?password=SENTINEL_B&sslpassword=SENTINEL_C
```

Reachable from any `CEMENTIC_DB_URL` using libpq keyword parameters; rendered to
stderr by `runner.py:87,139` into the background log, and in
`doctor.py:143`/`--json`.

**Fix.** Sanitize displayed endpoints (drop or mask credential-bearing query
values) in every site that renders a URL — `bootstrap.py:119`,
`doctor.py:143`, `runner.py`, and the config diagnostics. **Proves it.** Sentinels
absent from stdout, stderr, JSON and logs, including driver exception text. Use
synthetic values only; a real credential must never be printed to prove redaction.

### O11 — Extractors inherit the worker's environment (executed)

The isolated extraction child is spawned with `Popen` and no `env=`
(`extract.py:529-545`), so it inherits `CEMENTIC_DB_URL` and
`CEMENTIC_DB_PASSWORD` — the very credentials the module's own comment
(`:616-635`) says the worker holds. Confirmed: a sentinel in the parent's
environment is visible to a child spawned the same way. Isolation buys crash and
OOM containment only.

**Fix.** Pass a minimal environment to the child (`PATH`, locale, and the model
cache variables it actually needs). Also give the child its own process group
(`start_new_session=True`) and kill the group on timeout — the current timeout
kills one PID (`:532-545,654-658`), so a parser's own descendants survive.

### O17 — `stop` takes no lock while `start` does (source-traced)

`cli.py:797-880` reads supervisor state, signals, then writes `clear` — with no
`file_lock`, while `start` correctly takes one (`cli.py:452`). If a concurrent
`start` saves its records between the read and the clear, the clear deletes a
live session's state file and worker state files, leaving running workers visible
only to `ps`.

**Fix.** Take the same lock for the whole read/signal/clear sequence.
**Proves it.** An interleaving test that cannot lose a live session's records.

### O22 — Extraction limits and the direct-file path (source-traced)

- The isolated extractor allows **3600 s** per document (`extract.py:247`): a
  hostile file stalls the single child for an hour rather than hanging forever.
  `RLIMIT_AS` (8 GiB) is the real OOM guard.
- The **command extractor buffers all of stdout in-process** with no rlimit
  (`:291-293`), inside the credential-holding worker.
- `_plaintext_extractor` (`:263-268`) reads a whole file into memory; the 512 MiB
  cap lives only in the watcher, not on `cementic extract`'s direct path.

**Fix.** Apply the size cap and an output cap wherever a document is read, not
only where the watcher read it.

---

## 9. Contracts, docs, packaging, tests

### O23 — The generated config is not the complete reference (executed)

`docs/configuration.md:12-13` and `cli.py:155` call the file `config init` writes
"the complete reference… the source of truth". It omits
`extraction.timeout_seconds`, `extraction.memory_limit_mb` and
`search.lexical_candidate_pool` entirely. Verify:

```python
import tomllib
from cementic.config import Config
tpl = tomllib.load(open("src/cementic/templates/config.toml", "rb"))
cfg = Config().model_dump()
print([f"{s}.{k}" for s, v in cfg.items() if isinstance(v, dict)
       for k in v if k not in tpl.get(s, {})])
```

**Fix.** Add the three annotated keys. **Proves it.** A test that parses the
template and asserts every `Config` field appears, counting commented entries —
red first on exactly those three.

### O24 — The pooled-vector formula has no oracle (source-traced)

`_embed_by_splitting` ends with `pooled = [sum(values)/len(vectors) for values in
zip(*vectors)]` (`embedding_runtime.py:582`). The tests
(`test_embedding_runtime.py:1647,1664`) assert only `is not None` and `‖v‖ = 1`,
which first-piece, max-pool and mean all satisfy.

**Fix.** One assertion on the value: inputs `[[3,0],[0,4]]` must pool to
`[0.6, 0.8]`.

### O27 — `scripts/` is in no lint or type gate (executed)

`scripts/check.sh` and `.github/workflows/ci.yml` lint and type only `src/` and
`tests/`. `scripts/measure_retrieval_quality.py` is 86 KB and produces every
retrieval verdict in this project; `ruff` and `mypy` never see it. (This audit
linted it by hand.) Its usage lines also say `python scripts/…` where
`CONTRIBUTING.md` routes everything through `uv run`.

**Fix.** Add the paths to both gates and align the usage lines. Expect real
findings on the first run.

### O28 — Five small contracts (source-traced)

1. `__version__` (`src/cementic/__init__.py:3`) has **zero importers**; the CLI
   reads `importlib.metadata` (`cli.py:371`), and nothing ties the git tag to
   `pyproject` to `__init__`. Fix: delete one, or assert they agree in a test.
2. `pyproject.toml:8-12` lists only the 3.12 classifier while the CI matrix runs
   3.13. Fix: add it.
3. `search`'s `score` mixes cosine similarity with `ts_rank`, and is hidden
   whenever the order cannot explain it (`cli.py:1082`). No `fused_score` field
   exists to close what `scores_explain_order` was written about.
4. `prune_collection_history` records artifact removals in `session.info`
   (`revisions.py:636-656`), which **survives a `lost_race` rollback**
   (`collections.py:328`), so a second promote in one session could unlink
   artifacts that are still referenced. Latent: the CLI opens one session per
   invocation.
5. `EmbeddingProvider` (`embedding_provider.py:14-107`) has exactly one concrete
   implementation, and `test_cli.py:2090,2179,2341` are duplicate edge-case
   classes. Both are deletion candidates, not defects. (66 `MagicMock()` sites in
   `test_cli.py` also mean an internal refactor is punished without a behaviour
   change — the coupling is the cost, not the line count.)

### O28b — Two documentation contradictions worth one commit each

- `docs/setup.md:27` says "The repository is private" while `README.md` and
  `pyproject.toml` publish a plain public URL.
- README says the model download "is about 2 GB"; `docs/setup.md:8-9` assigns the
  same figure to the model **plus** database data and artifacts. One budget, one
  place.
- Both `scripts/measure_*.py` docstrings say `python scripts/…` (see O27).

### O6-adjacent — the external-server promise (source-traced)

`docs/configuration.md:72-76` suggests disabling autostart is enough to use an
externally supervised `llama-server`; bootstrap and doctor still require local
model/binary state, and runtime resolution requires cementic's exact alias.
Fix: document the contract that actually works, and exercise it with a fake
server. Remote-only support is a separate product decision.

### O29 — Carried from the previous audit, not re-verified here

Eleven items arrived from the 2026-11-13 audit and this pass neither confirmed
nor refuted them. Each is `source-traced` **by that audit**, which means
*someone read the path*; treat them as leads, and re-grade before acting. They
are listed in one place because none is individually urgent.

| Item | Where |
| --- | --- |
| The digest cache can become a correctness dependency: invalid UTF-8 cache bytes raise, and lock/directory failures escape the no-cache fallback | `model_digest.py:44-49,82-95` |
| Valid JSON with a non-object root (`null`, `[]`, a scalar) crashes worker-state recovery at `.items()`, and can reach PID signalling | `state.py:74,121-125` |
| The index capability registry keeps builders and dimension limits separately, so an absent entry means *unbounded* | `index_strategies.py:54-89` |
| File hashing is duplicated instead of using the shared helper | `source_watcher.py:157-163` vs `hashing.py:15-21` |
| The context-fit probe can label another model's output as the current config (trusts the first served alias), and its tail-detection uses exact equality against a no-noise assumption | `scripts/measure_chunk_context_fit.py:62-80,110-127` |
| "A failure prints nothing on stdout" is false for partial JSONL streams and for diagnostics that fail by design | `docs/operations.md:48-54` vs `cli.py:1058-1067,1281-1319` |
| `files_truncated` reports truncation when the listing fits exactly (equality used as the predicate) | `render.py:371-374`, `status_service.py:593-595` |
| Dev tools live in an optional extra, so a bare `uv sync` **removes** ruff/mypy/pytest | `pyproject.toml:74-83` vs `README.md:18` |
| CI never runs `config init` or `init postgres` from the built wheel, so a wheel missing its templates reports green | `ci.yml` `installed-wheel` job |
| Only the unit job runs on the 3.12 floor; the integration and pg jobs are 3.13-only | `ci.yml:22-52` vs `:60,79,99` |
| `pymupdf-layout` pulls ONNX Runtime, NumPy, NetworkX and PyYAML into every install | `uv.lock` |

Each needs a fix of the shape the table implies (validate the shape before
`.items()`, co-locate the capability with the builder, share one hasher, add the
missing CI step). **Proves it.** The item's own line turns into a failing test
first.

---

## 10. Checked, and NOT a defect

Recorded so nobody re-chases them. Each was verified on 2026-11-17/18.

- **The live corpus's 392 done-chunkings with zero chunks are correct.** All 392
  are sources the user deleted: the file is gone, `source_documents.status =
  'deleted'`, and `deleted sources that still own chunks: 0`. `doctor`'s
  predicate excludes them by design. **It does catch a live-source stranding** —
  run against that seeded state its own SQL returns 1 (and 0 for a deleted twin).
  An earlier draft of this audit claimed the opposite; the check refuted it.
- **DiskANN `num_neighbors > search_list_size` is not a config-validation
  defect.** pgvectorscale accepted both orders on a scratch database.
- **No unused or undeclared runtime dependency.** Every module-scope import in
  `src/` resolves to a declared dependency; `numpy`/`onnxruntime` are never
  imported.
- **No `shell=True` in extraction.** All three `subprocess` calls take argv
  lists, and `{path}` is substituted per element.
- **No 3.13-only syntax under the 3.12 floor.** No `typing.TypeIs`, no PEP 696
  defaults, no `Path.full_match`.
- **The file lock behaves.** Contention blocks, a SIGKILLed holder releases
  immediately, fds are non-inheritable.
- **`config show` masks both secret fields**, and the CLI prints no URL or
  password for an invalid `CEMENTIC_DB_URL`.
- **The PG CI job cannot silently skip**: it asserts `skipped == 0` from the
  junit XML, and tests are auto-marked by fixture.

---

## 11. What this audit never verified

The limits below are load-bearing: they are why some items are graded
`source-traced` and why F1–F3 are not yet a production claim.

1. **PostgreSQL-semantics anything.** The three repairs were proven on SQLite,
   which enforces no foreign keys — see O20. Run the `-m pg` suite and add the
   cascade tests before applying the repairs to a live corpus.
2. **No retrieval number was re-measured.** The embedding server was started by
   neither pass, and the F5 table comes from arms cached 2026-09-23 (pre-fallback,
   so it describes that snapshot, not `HEAD`).
3. **The live corpus was read, never written.** Queries only: 24,533 documents,
   2,423,430 chunks, 26 GB, one active revision, all embeddings `done`.
4. **Corpus-scale throughput, multi-collection latency, and the ANN recall
   arguments in O21** were not measured.
5. **A fresh-install path was not exercised** (no `uv build` + wheel install +
   `init postgres` from the wheel; CI does not run that either — O28/O27 adjacent).
6. **No service was started or restarted**, and no production database had a row
   written to it.

---

## 12. Relationship to PLAN.md, and when to delete this file

`PLAN.md`'s **"Execution order — open items"** is the live order; the ids there
(A3, B2b, C1…D13) are the same items as O1–O28 here, and its graded entries
carry the fix's end condition. This file is the point-in-time explanation: what
is wrong, how to see it, what to change, and what proves it.

Keep it while any item above is open. Delete it when the last one closes — a
report that outlives its findings becomes a second, drifting account of the
project, and `PLAN.md` plus the code comments are the durable record.
