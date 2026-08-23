# cementic — architecture & design

<!-- session-handoff:begin (2026-08-23) -->
## Where the work stands

**v0.2.0 is shipped** — tag `v0.2.0` = merge commit `9c9273f` on `main`, pushed
2026-08-18. The two plan-of-record sections that recorded it were dropped in the
2026-08-23 compaction; `git log` and `CHANGELOG.md` carry that history, and the
measurements worth keeping were harvested into "Measurements that justify
current defaults" below.

**A sixth review ran 2026-08-23** — a full-codebase audit (CLI surface, call
graph, error handling, coverage, FP/UNIX discipline, stale weight, docs drift),
recorded in `notes/review-codebase.html`. `notes/` is gitignored as of
`c7069e1`, so that note is local-only; its findings are carried into the plan
of record directly below, which is the live plan.

The audit's baseline: all six `./scripts/check.sh` gates green, mypy strict
clean on 27 files, ruff clean, 93% line / **88% branch** coverage. One
correctness bug, one dominant readability problem (`cli.py` at 2138 lines), and
substantial doc drift including a 404ing install URL. All three are now
addressed; `cli.py` is 1413 lines.

**Entry point:** Batch 4's remaining weight removal, then Batch 6. Batches 1,
2, 3 and 5 are done (see Exit criteria). Two decisions are still open and are
recorded under Open risks: promoting `status --doctor` to `cementic doctor`, and
whether Python 3.10/3.11 get a CI matrix or get dropped from `requires-python`.
Repo visibility turned out not to gate anything — the install docs were fixed
without it.

The live corpus is indexed and searchable at the shipped defaults: collection
`test` (watching `~/projects/bibs/papers`), 5 documents, 274/274 chunks embedded
at `chunk_size = 320`, 0 failed, revision `default-68e212bc-llama-cpp-12f77de0`
active. Workers are stopped; nothing runs in the background.

**Verification:** `./scripts/check.sh` — all six gates (the sixth, `lockfile`,
was added in `e5812cd`; several docs still say five, which is Batch 5).
<!-- session-handoff:end -->

## Plan of record — sixth-review fixes (2026-08-23)

**Status: LIVE.** This is the only live execution order in this file; everything
below it is closed record or deferral register.

**Scope.** The 17-item ordered action list in `notes/review-codebase.html`,
batched by dependency. Every file:line and every piece of evidence lives in that
note; this section holds only execution order, decisions, anti-scope, and exit
criteria. File:line references throughout are as-audited (pre-refactor) and will
not match `cli.py` after `c69936d`.

**What this plan does not cover.** The four items under "Also still open, from
the fourth review" near the bottom of this file stay open and unscheduled: the
2 s start grace vs the 120 s startup timeout, the orphaned-daemon reclaim in
general (Batch 1 item 5 closes only one of its causes), model identity being the
path string, and `EXTRACTION_VERSION` being hand-maintained. The sixth review
re-confirmed the first, third and fourth are still real; none is in scope here,
and none is silently absorbed into it. Likewise everything in `TODO.md`, which
is features, and everything under "Deliberately not done", which is kept
deliberately.

**Mechanics.** One branch, `claude/audit-followup`. Conventional commits, one
commit per item or tight item-cluster, each carrying its regression test, each
green under all six `./scripts/check.sh` gates (PG included — that lesson is
paid for). `Assisted-by:` trailers, never `Co-Authored-By:`.

**Step 0 — novelty lit-gate: skip.** This is defect and structure work on an
existing tool; there is no novelty claim to gate. Recorded so the omission reads
as a decision rather than a miss.

### Batch map

| # | What | Items | Kind | Needs user? |
|---|---|---|---|---|
| 1 | Correctness | 1, 3, 4, 5 | Code + tests | No |
| 2 | Make the suite trustworthy | 8, 9, 10 | Tests | No |
| 3 | Structural refactor | 6, 7, 17 | Pure moves | No |
| 4 | Weight removal | 11, 12 | Deletion | No |
| 5 | Doc truth | 2, 14, 13 | Docs | **Yes** (item 2 decision; 13 confirmation) |
| 6 | Surface changes | 15, 16 | CLI + CI | **Yes** (both are decisions) |

Dependencies that actually bind: **2 before 3** — you cannot safely restructure
a 2138-line file behind 80% branch coverage and a test that fails under
coverage instrumentation. **3 before 4**, so cleanup diffs never mix with
refactor diffs. Batches 1, 5 and 6 are independent of the rest and of each
other.

### Batch 1 — correctness

Four independent defects. Each lands as its own commit with a regression test
that is verified red before the fix.

**1. `is_retryable_embed_error` mis-classifies two ordinary failures.**
`pipeline_worker.py:131-150` enumerates retryable exception types.
`requests.JSONDecodeError` is not an `HTTPError` (MRO verified), and
`ContentDecodingError` is one but carries `response = None`, so both return
`False`. A garbled llama.cpp body therefore stamps every chunk in the batch
`status="failed"` permanently; `promote` then blocks, and `--force` publishes an
index silently missing them. This is the only path in the system that both loses
work and reports success.

Change: invert the test. Any `requests.RequestException` from which no HTTP
status can be read is retryable; keep the explicit status allow-list only for
the case where a status *is* readable. Tests: synthesised `JSONDecodeError` and
`ContentDecodingError` both classify retryable; a real 400 still classifies
terminal.

*Anti-scope: do not add a retry/backoff policy layer. This is one predicate
function; the backoff around it already exists.*

**3. Three failure paths print to stdout.** `cli.py:356`, `:1460`, `:1778` use
`console` then `raise typer.Exit(1)`, against the contract `README.md:178`
states. Two independent passes agree the set is exactly these three, so no
survey is needed — change `console` to `err_console` at each. While in
`cli.py:1114-1126`, make the whole failed-start report use one stream.

*Anti-scope: do not audit stdout usage generally. The sweep is done; three
sites.*

**4. Deletion failures are logged and forgotten.** `source_watcher.py:423-433`:
`_on_file_deleted` and `_on_directory_deleted` catch, log, and continue, while
`_on_file_detected` three lines above routes its failure through
`record_skipped`. Consequence: search keeps returning a file the user deleted,
with nothing in `status`. Route both handlers through `record_skipped` the same
way.

Second half, same commit: the source watcher never writes `last_error` at all
(verified — zero occurrences in the module), so the "last error" row
`cli.py:748-755` renders for it is dead. Give the watcher's loop the same
`last_error` / `last_error_at` write the pipeline worker has at
`pipeline_worker.py:476-482`. Correct `README.md:214`, which attributes the row
to "a background worker".

**5. A failed pid-file write orphans a live daemon.**
`embedding_runtime.py:909-911` calls `_write_daemon_pid_file` unguarded straight
after `spawn_detached`. If the write fails the server is running with a
multi-GB model resident and nothing records its PID; `embedding stop` then
reports "already stopped" forever. Wrap it: on failure, kill the child, then
re-raise.

Note this is a **new mechanism for a defect already recorded as open** — see
"Also still open, from the fourth review" below, second bullet, observed live
2026-08-15. Closing this does not close that item in general (a lost pid record
has other causes); update that bullet to name the one cause now fixed.

### Batch 2 — make the suite trustworthy

Nothing here changes product behaviour. It exists so Batch 3 is safe.

**9. Six wall-clock budget assertions.** `tests/integration/test_cli_pg.py:64,116`
and `tests/unit/test_cli_performance.py:24,57,67,78`. The first already fails
under coverage instrumentation (`assert 2.0099… < 2.0`), which means the suite
currently punishes measuring coverage. Replace with structural assertions,
following the precedent already set in that same file by
`test_cli_import_does_not_pull_in_the_pdf_stack`, which chose a structural check
explicitly "so it does not depend on machine load".

**10. 35 `create_engine` calls with zero `dispose()`** across 7 unit-test files.
Harmless on 3.12; on 3.14 the `ResourceWarning` plus `filterwarnings = ["error"]`
plus pytest 9's unraisable hook turns them into a GC-timing-dependent failing
set (mechanism reproduced 3.12 vs 3.14 during the audit). Add a fixture that
disposes. This unblocks any 3.14 move and is a plain resource leak regardless.

**8. Characterization test: a pipeline failure must reach what `status` prints.**
The audit confirmed the two halves of the suite never meet — the pipeline tests
assert on DB rows, the status tests run only clean pipelines, and
`cli.py:891,893` (the lines printing `(failed: N)` and `| error: …`) are
uncovered by the entire suite. Write it **now, before Batch 3**, against the
current code: it is the characterization test that protects the render split.

Then close the two adjacent gaps the audit named: `collection reindex --force`
through the CLI (currently only `force is False` is asserted, at
`tests/unit/test_cli.py:1322`), and `runner.py:154-155`, the pipeline worker's
`fatal_reason` exit-1 guard, which is unreached because both runner tests named
`test_success_path` set `start.side_effect = KeyboardInterrupt`.

Also in this batch, because it is one line and makes every later number honest:
add `--cov-branch` to the coverage invocation.

*Anti-scope: do not chase the coverage percentage. Four named gaps, then stop.
`tests/unit/test_embedder.py` is a stale duplicate of `test_pipeline_worker.py`
and should be deleted rather than extended, but that is Batch 4.*

### Batch 3 — structural refactor

Pure moves. No behaviour change, so the diff is reviewable by reading names, and
Batch 2's tests are the proof.

**6. Collapse the sevenfold error ladder.** `cli.py:1302, 1536, 1607, 1653,
1768, 1827, 1926` each end with the identical `except typer.Exit: raise / except
Exception: _report_db_error(...); raise typer.Exit(1)` block, five of them
carrying a verbatim four-line comment explaining that `typer.Exit` subclasses
`RuntimeError`. One context manager, e.g.
`_reporting_db_errors(action: str)`, replaces all seven and states the reason
once. ~44 lines, one place for the policy.

**7. Split `cli.py`.** 2138 lines holding four unrelated concerns:

| Lines | Content | Destination |
|---|---|---|
| 84–141 | Six Click/Typer subclasses (uppercased headings, un-indented epilogs) | `cli_format.py` |
| 282–332 | `_DEFAULT_CONFIG_TOML`, a data file in a Python string | `templates/config.toml`, read via `resources.files()` |
| 673–995 | Seven `_print_*` / `_state` renderers | `render.py` |
| 1509–1834 | The five `collection` subcommands | `cli_collection.py` |

The renderer move is the one with substance: `_print_status_summary`,
`_print_collection_detail` and `_print_status_json` all call `_get_config()`
from inside a render function, and `_print_status_json` opens a database session
and runs queries — it is a command implementation wearing a printer's name.
Moving it forces the split into a pure `build_status_document(...) -> dict` plus
a one-line emitter, which is both the FP fix and what makes item 8's test
straightforward.

*Anti-scope: this is a move, not a rewrite. Do not rename public behaviour, do
not change output text, do not "improve" the renderers while relocating them.
Any behaviour change found necessary stops the batch and becomes its own item.*

**17. Two unnecessary function-local imports in `config.py`.** Transitive-closure
analysis during the audit showed that of the five deferred imports in
`config.py`'s validators, three genuinely break cycles
(`embedding_runtime` ×2, `extract` ×1) and two do not: neither `vector_store`
(`:665`) nor `index_strategies` (`:678`) imports `config`, directly or
transitively. Hoist those two to module scope.

For the three real cycles, record the direction rather than fixing it now:
`config` has the highest fan-in in the project (14) and belongs at the bottom of
the graph, so the durable fix is a dependency-free module holding registry
*names* that `config` can validate against without importing implementations.
One of the three (`:858`) also reaches for a private symbol,
`_TASK_PREFIX_TOKEN_ALLOWANCE`. See decision log entry 2026-08-23-c.

*Anti-scope: hoist two imports. Do not build the registry-names module in this
batch — it touches four modules and wants its own commit and its own review.*

### Batch 4 — weight removal

**11. Delete the duplicated container build files. DONE (`fdefdfa`) — and the
item as first written was wrong.**

What was true: `containers/postgres-vectorscale/Containerfile` was
byte-identical to `src/cementic/templates/postgres/Containerfile`, and
`containers/quadlet/cementic-postgres.container` had no consumer at all
(`scripts/verify_postgres_container.sh` reads the quadlet unit out of a
*generated* `cementic init postgres` directory, not that one). Both deleted;
`compose.yml` now builds from the packaged Containerfile, so the image the pg
gate tests is the image a user gets.

What was wrong: this item also called for deleting root `compose.yml` as a
near-duplicate of the template's. It is not one. It is the stack the pg
integration fixture brings up (`tests/integration/conftest.py:20`), and it
differs from the template deliberately — no `restart: unless-stopped`, because
the fixture tears it down. Deleting it would have broken the pg gate. Kept.

Residual, not closed: the repoint is unverified locally, because
`_compose_postgres` short-circuits when Postgres is already reachable, so no
local run exercises `compose build`. CI's `integration-pg` job has no service
container and will prove it.

**12. Delete three orphan `PDF_SPECS` entries** (~103 lines) in
`tests/fixtures/generate_pdfs.py`: `cs_neural_nets`, `bio_cell`, `hist_rome`. A
repo-wide grep returns hits only inside the generator itself, while
`test_doc_a` returns 25 — the search works. This is the only true dead code in
the repo.

Also here, from Batch 2's note: delete `tests/unit/test_embedder.py` (114 lines,
docstring "Tests for the pipeline worker", class `TestPipelineWorker`, filename
matching nothing in `src/`; both its non-trivial tests duplicate
`test_pipeline_worker.py:156` and `:188`).

**Duplication collapses, in payoff order, as budget allows.** Each is verified
byte-identical or near-identical; none changes behaviour:

1. `_step_extract`/`_step_chunk`/`_step_embed` preamble
   (`pipeline_worker.py:508, 610, 748`) plus the repeated live-documents filter
   (`:522, :626, :771`) — ~20 lines out of three functions that are 100–157
   lines each. Highest readability payoff in that file.
2. `runner.py:61` and `:111` — identical skeleton, three differing points,
   ~35 lines. *Care: their distinct comments record distinct bug histories and
   must survive the merge.*
3. Config-error rendering, `cli.py:259-276` and `runner.py:45-58`. The latter's
   comment already says "Parity with the CLI's `_get_config`". Move to
   `config.render_config_error(error) -> str | None`. This also closes a
   coverage gap, since the `runner` copy is entirely untested.
4. `_handle_shutdown` — byte-identical bodies plus an eight-line docstring in
   `pipeline_worker.py:998` and `source_watcher.py:603`. `worker_runtime.py`
   exists for exactly this and says so in its own module docstring; this handler
   was missed.
5. `on_created` / `on_modified` (`source_watcher.py:146,153`) — byte-identical.
   `on_created = on_modified`.
6. The unremoved-artifact warning block (`cli.py:1578-1583`, `:1727-1732`).
7. ~~Deferred-cleanup recorders; revision-by-status query; the join chain.~~
   **DROPPED 2026-08-23, on inspection during execution.** All three are
   parallel type-specific instances rather than real duplication, and
   collapsing each costs about what it saves:
   - `_defer_*`/`drain_*` (`revisions.py:400,414` + partners): the pairs differ
     in element type (`list[str]` vs `list[int]`). A generic `_defer(session,
     key, items)` plus the typed wrappers mypy strict would still need comes to
     the same line count and reads worse than four named functions.
   - `find_ready_revision` (`collections.py:219`) vs `get_active_revision`
     (`revisions.py:144`): identical bar the status string, but collapsing saves
     ~4 lines and moves a function across a module boundary.
   - The Chunk→Extracted→Source join chain (`pipeline_worker.py:762`,
     `status_service.py:332`, `:358`): three sites, cross-module, ~8 lines.

   Recorded rather than silently skipped: the readability goal is the point, and
   a line count is not a proxy for it. Do not re-derive these.

*Anti-scope: the three §5 trim candidates deliberately kept in "Deliberately not
done" below — `_llama_daemon_runtime_status`, `_state`, and the
`state_path is None` guards — stay kept. Do not re-derive them. Likewise the
dense "why" comments throughout: they are the codebase's best feature and are
not bloat.*

### Batch 5 — doc truth

**2. The install command 404s. DONE — and it never needed the decision it was
filed behind.** `README.md` gave `pipx install "git+https://github.com/...@v0.2.0"`,
which 404s anonymously because the repo is private. This was recorded as blocked
on a repo-visibility decision. It was not: the repo being private blocks nobody
who already installs from source, and the defect was simply that the docs
described a command that cannot work. README now documents clone-then-install and
says why. Repo visibility remains a real question, but it gates nothing here.

**14. Correct the counts and the wrong details. DONE (`dde9413`).** All verified
during the audit. Note the `AGENTS.md` items below were corrected there and then
superseded: that file was folded into README's Development section and deleted,
so README is now the single source of truth for contributors and agents both.


- "five gates" → six, in 8 places: `TODO.md:3`, `CHANGELOG.md:24`,
  `PLAN.md:24,31,40,166,345,742`. `lockfile` was added in `e5812cd`.
- `README.md:158` lists five doctor checks; `doctor.py` emits six
  (`chunk_budget`, `:216`).
- `AGENTS.md` Project Structure omits `worker_runtime.py` and `__init__.py`.
- `AGENTS.md:34,178-183` prescribes `pytest -v && ruff && mypy` as "all checks"
  and never mentions `./scripts/check.sh` — while `TODO.md:3-6` records that
  precisely this unit-only habit is why three PG tests once reached `main` red.
  The agent-facing doc prescribes the failure mode the roadmap doc warns about.
  Fix this one first; it is the one that causes further defects.
- `AGENTS.md:125` claims JSON-file "pause/resume"; no pause or resume exists.
- `README.md:414` says task prefixes apply to "Nomic v2"; the code matches the
  whole `nomic-embed-text` family. `README.md:318` gives the chunk arithmetic as
  `320 × 1.45`; the code computes `(320 + 8) × 1.45`. `README.md:290` says
  "bundled" model; nothing is bundled. `README.md:300` implies `rapidocr` is
  optional; `pyproject.toml:49` makes it required.
- `TODO.md:37-38` says the pg CI job uses a service container;
  `ci.yml:55-76` has no `services:` block.
- `CHANGELOG.md` has no `[Unreleased]` section for the four post-tag commits.

Then add what is implemented and undocumented but a user needs: the
collection-name grammar (`validation.py:5-46`), multi-collection search
(`-c a b`, documented only in the epilog), the mixed-model refusal
(`search.py:203-205`), the query limits, the six `search --json` field names,
and the cold-start ordering (`create_tables` runs only inside the workers
`cementic start` spawns, so `status`/`search`/`collection list` all exit 1 on a
fresh database — verified live).

**13. Compact `PLAN.md` and `README.md`. DONE.**

`PLAN.md` 1498 -> 865 lines. Dropped: the two closed plans of record, the
per-branch "Fixed on `claude/*`" logs, the `Deferred` list (which duplicated
`TODO.md` -- two roadmaps, one of which nobody updates), `CLI surface` (which
restated README), and the narrative around the superseded ANN analysis.

Harvested first, into "Measurements that justify current defaults": the
`1454 s at 64MB vs 345 s at 2GB` build-memory figure, the `EXPLAIN` plan-shape
table behind the filter-column schema, the watchdog directory-move event table,
the tokenizer-ratio derivation moved out of README, and the adversarial-review
lesson. Deleting the closed sections wholesale would have destroyed the
justification for two live defaults -- the numbers sat inside the branch logs.

`README.md` 506 -> 458 lines. Removed the `Architecture` module list (duplicating
`AGENTS.md`'s tree and Key seams above) and the `Development` section
(contributor instructions belonging in `AGENTS.md`, and stale -- it listed five
gates). Compressed the chunk-size and ANN-index sections to what a user acts on,
leaving the derivations here.

### Batch 6 — surface changes

Both items are decisions before they are work; neither is scheduled until the
user calls them. See Open risks.

**15. Promote `status --doctor` to `cementic doctor`.** The doctor path
(`cli.py:1170-1211`) is 42 lines of a 162-line function sharing no data, no
output shape and no control flow with `status`; it returns before touching
anything else. The tell is `cli.py:1174-1186`, which exists only to warn that
`-c` and `-v` are ignored — a warning needed only because one command does two
jobs. README already treats it as a setup-time command (lines 67, 83, 87, before
any indexing). Promoting it deletes that warning.

**16. Python 3.10 and 3.11 are claimed but never tested.**
`requires-python = ">=3.10"` and the classifiers claim three versions; CI tests
only 3.12. No 3.11+ syntax is in use (checked: no `datetime.UTC`, `Self`,
`ExceptionGroup`, `except*`, `StrEnum`, `TaskGroup`), and mypy targets 3.10, so
the claim is plausible — but unverified is unverified.

### Exit criteria

- [x] Batch 1: four fixes, each with a regression test verified red before the
      fix (`7ed71f7`).
- [x] Batch 2 (`2a2c856`): the six wall-clock budgets are structural; branch
      coverage is on by default; a real pipeline failure is asserted against
      `status --verbose` output (red-verified independently in the parent
      session by disabling the render guard); `reindex --force` and
      `runner.py:154-155` covered. Three timing checks survive elsewhere and
      are deliberate: they are liveness bounds with ~6x headroom, not budgets.
- [x] Batch 3 (`c69936d`): `cli.py` 2138 -> 1413; six of seven ladder sites
      share one context manager (`search` keeps its own — it branches on
      `json_output` through three paths that never call `_report_db_error`, so
      sharing would need a parameter that distorts the shape); no output text
      changed, proven by `git diff --stat -- tests/` empty and 1006 tests
      passing unmodified.
- [x] Batch 4 (part): `containers/` gone, `compose.yml` repointed at the
      packaged Containerfile and kept (`fdefdfa`).
- [ ] Batch 4 (rest): the three PDF specs gone, `test_embedder.py` gone, and at
      least collapses 1–5 landed.
- [x] Batch 5: README no longer claims an install path that cannot work
      (`dde9413` for the counts, plus the compaction commit for the install
      text); every count in the list above corrected; `PLAN.md` 1498 -> 865 and
      `README.md` 506 -> 458, with the load-bearing measurements harvested
      rather than deleted.
- [ ] Batch 6: executed or explicitly deferred with a reason recorded here.
- [ ] All six `./scripts/check.sh` gates green at every commit.
- [ ] No unmerged `claude/*` branch left behind.

**Sizing.** Batch 1: one focused session. Batch 2: one session. Batch 3: one to
two, mostly mechanical but wide. Batch 4: one. Batch 5: one, plus the user's
decision on item 2. Batch 6: short once decided.

### Open risks

1. **Repo visibility is unresolved and blocks item 2.**
   `https://github.com/mnazaal/cementic` 404s anonymously, so the documented
   install path cannot work for anyone. `PLAN.md:386-392` records that this
   exact command could not be run at release time and it shipped anyway.
   *Recommendation:* if the repo is meant to be public, publish it and re-verify
   the install; if it is meant to stay private, delete the pipx/uv-tool
   instructions from `README.md` and the URL from `cli.py:213`'s epilog, and
   document install-from-source as the only path. Either is cheap; shipping a
   404 as the first instruction is not.
2. **`cementic doctor` is a CLI surface change** (item 15). Removing
   `status --doctor` outright breaks anyone scripting it; keeping both doubles
   the surface. *Recommendation:* add `cementic doctor`, keep `status --doctor`
   as a hidden alias for one release, remove it at the next minor. Since the
   project has one user and `Road to v1` names "config/CLI stability" as a v1
   bar, doing it now is cheaper than doing it after v1.
3. **Python 3.10/3.11 support is claimed, not tested** (item 16).
   *Recommendation:* add both to the CI matrix — it is three lines of YAML and
   the code already looks compatible — rather than narrowing
   `requires-python`, which would be a real capability loss for a
   plausible-but-unverified reason.
4. **Batch 3 is a wide diff in the file with the lowest branch coverage.**
   `cli.py` is at 80%. Batch 2 is the mitigation and is a hard prerequisite, not
   a nicety. If Batch 2 slips, Batch 3 must slip with it.
5. **The audit was a review, not a proof.** Everything above was verified by
   reading the cited lines or executing a check, but three findings rest on
   reasoning about paths that no test exercises today (the pid-file orphan, the
   deletion-handler consequence, the 3.14 warning set). Batch 1 and Batch 2 turn
   each into a test; until then they are well-founded, not demonstrated.

## Decision log

Append-only. One entry per hard-to-reverse decision: the question a future
reader would ask, the choice, the rejected alternatives, and a status.

**2026-08-23-a — Why is `notes/` gitignored rather than tracked?**
Choice: untracked, kept on disk, history preserved (`c7069e1`). Code-review
notes are working artifacts with one reader; tracking them meant a commit per
review and 2034 lines of HTML in the tree. Rejected: keeping them tracked (they
are not part of the shipped project) and deleting them outright (the resolution
banners are the record of what each review closed). Consequence handled in Batch
5: four `PLAN.md` links to them are now dead for a fresh clone. Status: live.

**2026-08-23-b — Why collapse the `except typer.Exit` ladder into a context
manager rather than a decorator?**
Choice: context manager. The seven sites wrap a *region* inside the command
body, not the whole command — several do work before and after the guarded
region (`promote` reads outcome fields inside the session, then renders after
it). A decorator would force that work into the guarded scope and change which
exceptions the handler sees. Rejected: a decorator (wrong granularity), and
leaving it alone (five verbatim copies of a four-line comment is the maintenance
cost the collapse removes). Status: live.

**2026-08-23-c — Why not fix the `config.py` import cycles now?**
Choice: hoist only the two imports that break no cycle (`:665`, `:678`); leave
the three real ones deferred. The durable fix is a dependency-free module of
registry names, which touches `config`, `embedding_runtime`, `extract` and
whatever imports them — a four-module change with its own risk, landing in the
same batch as a 2138-line file split. Rejected: doing both in Batch 3 (two wide
structural diffs at once is how a bisect stops being useful), and doing neither
(the two gratuitous deferred imports each cost a reader a "why is this here?").
Status: parked. *Revisit when:* Batch 3 has landed and `config.py` is next
opened for any other reason — the same precondition the round-two review used
for `connect_args`, which has since been met twice.

**2026-08-23-d — Why is a characterization test (item 8) scheduled before the
refactor rather than after?**
Choice: before. The audit found the two halves of the suite never meet — a
pipeline failure is never asserted against what `status` prints, and
`cli.py:891,893` are uncovered by the entire suite. That is exactly the
behaviour Batch 3's render split moves. Writing the test after the move would
pin the new behaviour, not the old, which is the one thing a refactor's test
must not do. Rejected: writing it after (pins the wrong thing) and skipping it
(the "reports success, dropped the work" defect class is the project's recurring
failure mode). Status: live.

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
50 chunks per paper — roughly 2M vectors. The numbers behind that (query
latency, embedding throughput, HNSW memory) are in README's "Measurements behind
the defaults"; the design consequence is everything under Key seams below:
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
of the filename (the filename heuristic is now at least *reported* by
`cementic embedding start`).

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
  measured event table is under "Batch 2" above: move-out (case C) was real and
  is fixed; root-move (case D) is unfixable from inside the watch and is
  documented in README's Known limitations. This entry contradicted its own
  file's Batch 2 heading until the sixth review caught it.
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
  *Sixth review found one concrete cause:* `_write_daemon_pid_file` runs
  unguarded straight after `spawn_detached` (`embedding_runtime.py:909-911`), so
  a failed write orphans a daemon that is already serving. Batch 1 item 5 closes
  that cause; the general item stays open, since a lost pid record has others
  (crash between spawn and write, a wiped data dir).
- Model identity is the *path string*, so `resolve_llama_model_path` trying cwd
  first means indexing from two directories can silently mean two different
  GGUF files under one fingerprint. Batch with the `verbose` change above, since
  both force a re-embed.
- `EXTRACTION_VERSION` is a hand-maintained integer, not the pymupdf/pymupdf4llm
  versions that actually produce the Markdown, so a dependency bump changes
  extraction output without moving the fingerprint.

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

**Sizing.** Batch 1: one focused session. Batch 2: ~1 h timebox plus fix time
if it reproduces. Batch 3: ~30 min wall clock, mostly waiting. Batches 4–6:
one short session combined.

