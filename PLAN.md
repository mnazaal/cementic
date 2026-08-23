# cementic — architecture & design

<!-- session-handoff:begin (2026-08-23) -->
## Where the work stands

**v0.2.0 is shipped** — tag `v0.2.0` = merge commit `9c9273f` on `main`, pushed
2026-08-18. The two plan-of-record sections below it are closed records.

**A sixth review ran 2026-08-23** — a full-codebase audit (CLI surface, call
graph, error handling, coverage, FP/UNIX discipline, stale weight, docs drift),
recorded in `notes/review-codebase.html`. `notes/` is gitignored as of
`c7069e1`, so that note is local-only; its findings are carried into the plan
of record directly below, which is the live plan.

The audit's baseline: all six `./scripts/check.sh` gates green, mypy strict
clean on 27 files, ruff clean, 93% line / **88% branch** coverage. One
correctness bug, one dominant readability problem (`cli.py` at 2138 lines), and
substantial doc drift including a 404ing install URL.

**Entry point:** Batch 1, item 1 — widen `is_retryable_embed_error`. Three
decisions need the user before their batches can run (repo visibility, `doctor`
promotion, Python 3.10/3.11 support); all three are recorded under Open risks
with a recommendation, and none blocks Batch 1.

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
criteria — same division as the two closed plans below.

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
refactor diffs (the same rule the fifth-review plan's batch 8 followed). Batches
1, 5 and 6 are independent of the rest and of each other.

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

**11. Delete `containers/` and root `compose.yml`.**
`containers/postgres-vectorscale/Containerfile` is **byte-identical** to
`src/cementic/templates/postgres/Containerfile`; the quadlet unit and root
`compose.yml` are near-duplicates of their `templates/postgres/` counterparts,
differing only in comments. Three infra files maintained in two places, one
already guaranteed to drift silently. Point the development workflow at
`src/cementic/templates/postgres/` and delete the duplicates. Update any
`compose.yml` reference in `README.md`, `TODO.md`, `AGENTS.md` and
`scripts/check.sh`'s docstring in the same commit.

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
7. Deferred-cleanup recorders (`revisions.py:400,414` and their `drain_*`
   partners); revision-by-status query (`collections.py:219`,
   `revisions.py:144`); the Chunk→Extracted→Source join chain
   (`pipeline_worker.py:762`, `status_service.py:332`, `:358`).

*Anti-scope: the three §5 trim candidates deliberately kept in "Deliberately not
done" below — `_llama_daemon_runtime_status`, `_state`, and the
`state_path is None` guards — stay kept. Do not re-derive them. Likewise the
dense "why" comments throughout: they are the codebase's best feature and are
not bloat.*

### Batch 5 — doc truth

**2. The install command 404s.** `README.md:31,33` give
`pipx install "git+https://github.com/mnazaal/cementic.git@v0.2.0"`; `curl`
returns HTTP 404 anonymously. The same dead URL is in `cli.py:213`'s root epilog
— so it prints in `--help` — and in `pyproject.toml:79-81`. It is the first
instruction a new user follows. **Needs the user's decision** (see Open risks).

**14. Correct the counts and the wrong details.** All verified during the audit:

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

**13. Compact `PLAN.md`.** ~763 of its 1013 lines are closed record. Live design
content is Design principles (414-430), Scale context (431-455), Key seams
(456-514), Pipeline as composable filters (515-539), plus the handoff block and
this section — about 250 lines plus this plan. Candidates for removal, in order:
the two closed plans of record (27-412, 386 lines, both fully ticked and
recorded in commit history and `CHANGELOG.md`); the closed-branch review logs
(607-872, 266 lines); the Deferred list (998-1013), which duplicates `TODO.md`
in substance — two roadmaps, one of which nobody will update. Keep the 6-line
measurement table from Superseded (873-905) and drop its surrounding 27 lines.
Also fix `CLI surface` (540-556), which restates README with no added design
content, and the four `notes/` links (35, 565-572, 981) that went dead when
`notes/` was untracked in `c7069e1`.

**Needs the user's confirmation before executing** — deleting 763 lines of
record is not something to do on inference, and `research-plan` treats
compaction as a separate confirmed pass.

Separately, `README.md` (475 lines) carries developer content against the
standing rule that README is user-facing: move the Architecture module list
(445-464, duplicating `AGENTS.md`'s tree), the tokenizer-ratio derivation
(312-327), and the `1454s vs 345s` index benchmark (349-352) into this file;
merge `Development` (465-475) into `AGENTS.md`, where those instructions
currently live and are wrong.

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

- [ ] Batch 1: four fixes, each with a regression test verified red before the
      fix.
- [ ] Batch 2: no wall-clock assertion remains in the suite; `--cov-branch` is
      the default; a pipeline failure is asserted end-to-end against `status`
      output; `reindex --force` and `runner.py:154-155` covered.
- [ ] Batch 3: `cli.py` under 1500 lines, `cli.py` has one copy of the
      db-error ladder, and **no output text changed** — proven by the Batch 2
      tests passing unmodified.
- [ ] Batch 4: `containers/` gone, the three PDF specs gone,
      `test_embedder.py` gone, and at least collapses 1–5 landed.
- [ ] Batch 5: the install path works as written from a clean environment, or
      the README no longer claims it does; every count in the list above
      corrected.
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

## Plan of record — fifth-review fixes (2026-08-18)

**Executed same day** — twelve fix/refactor/test commits on
`claude/review-fixes-2026-08-17`, in the batch order below, every batch green
under all five `./scripts/check.sh` gates (PG included). The note's resolution
banner maps finding → commit; the deliberate exceptions are recorded there and
under "Deliberately not done". Kept as the record of the decisions.

Scope: close the fifth review, `notes/code-review-2026-08-17.html`.
All file:line evidence lives in the note; this section holds only execution
order, the decisions, and the exit criteria. Mechanics: one branch
(`claude/review-fixes-2026-08-17`), conventional commits, one commit per
finding-cluster with its regression test, each commit green under
`./scripts/check.sh` — all five gates, PG included (that lesson is paid for).
When done, the note gets a resolution banner mapping finding → commit, same
shape as the 2026-08-14 note.

### Order of attack

`§` references are the note's sections.

1. **Finish the five half-landed fixes (§1).** Each currently contradicts a
   commit message or docstring that claims it done, so they go first:
   - `status --json` on an unknown collection: validate before the JSON early
     return; error to stderr, non-zero exit. Correct the README/CHANGELOG
     "exits non-zero" claims in the same commit.
   - The initial scan records skips: symlinks, walk errors, and registration
     failures go through `record_skipped` exactly as live inotify events do.
   - `stop`'s kill loop: `PermissionError` means alive-but-not-ours (mirror
     `force_kill`'s reasoning); never clear supervisor/worker state while such
     a pid remains; report it and exit non-zero.
   - `CEMENTIC_CONFIG`: `expanduser` in `resolve_config_path`; `config path`
     calls the existing `config_path_error` guard so an unusable value is
     reported instead of silently masked by the fallback.
   - `promote` prints the artifact-removal failure list (as `remove` already
     does); `embed` rejects empty/whitespace `content`.
2. **Worker/DB correctness (§2)**, in severity order:
   - §2.1 `collection remove` vs a running worker. **Decision — recommended
     mechanism:** the worker re-validates its cached revision id once per poll
     cycle (one cheap SELECT) and exits cleanly when it is gone; `collection
     remove` additionally warns when workers for that collection are running.
     Rejected alternative: refusing removal while workers run — heavier UX,
     and the delete itself is already cascade-safe; the defect is only the
     zombie worker and the watcher resurrecting the collection.
   - §2.7 search's `-c` fallback ranks revisions via `_searchable_revisions`
     so there is one source of revision choice. Twice-derived carry; the
     regression test pins building-vs-ready.
   - §2.5 `IS DISTINCT FROM` semantics for `source_content_hash` in the claim
     query, and write the hash on the failure path too, so a NULL row cannot
     wedge a revision in `building`.
   - §2.2 `SET LOCAL maintenance_work_mem`; delete the false "connection is
     discarded" comment.
   - §2.4 reset `skipped_files` on watcher restart; §2.6 drop whitespace-only
     chunks before `chunk_index` assignment so indexes stay contiguous and
     `total_chunks` honest; §2.3 a post-commit cleanup failure after a durable
     promote is a warning, not "promote failed" exit 1.
   - §2.8 partial unique index `ON pipeline_revisions (collection) WHERE
     status = 'active'`, applied through the ensure-schema path (same
     mechanism as `ensure_vector_table_schema` — `create_all` won't retrofit
     it), plus promote re-reading status under `FOR UPDATE`.
   - §2.9's smaller items ride along wherever their file is already open;
     the directory-move blindness (unverified) gets a repro test first and a
     fix only if it reproduces.
3. **Error-stream and wrapping discipline (§4.2–4.3).** Mechanical, wide
   blast radius, kept in its own commits: all human error text to stderr via
   an `err_console`; `soft_wrap=True` so off-TTY output stops hard-wrapping
   paths at 80 columns. Tests pipe the output and assert stream and absence
   of mid-path wraps. This is what unblocks `--json | jq` composability.
4. **Exit-code normalization (§4.1).** Adopt the convention most commands
   already follow — 0 ok, 1 operation failed, 2 usage error (Click's own
   parser errors) — and move the stragglers to it (`collection remove
   <unknown>`, `promote` with no ready revision). An unknown *name* is an
   operation failure, so it exits 1, not 2. Document the table in README.
5. **Config/runtime hardening (§3.1–3.3, §3.5–3.6).** Bounds on numerics
   (`n_ctx >= 1` closes the guard-disable hole; positive intervals and
   timeouts; port ranges), the chunk_size↔n_ctx invariant enforced at config
   validation against the *configured* values (today it is only tested at the
   shipped defaults), env-vs-file attribution in config error messages,
   `over_budget_reason` wired into worker failure rows and the query-side
   message (which currently blames `pipeline.chunk_size` for a long query),
   `runner.py` parity with the CLI's error handling (`RuntimeError`,
   `SettingsError`), `embedding stop` under the daemon lock, autostart
   failing fast on a definitive model mismatch instead of waiting 120 s, and
   doctor's unreachable/false branches.
6. **Bootstrap download (§3.4).** Third-time carry — **decision: fix now.**
   Unique temp name, download performed under the daemon file lock (taken
   before the download, not after), `requests` exceptions wrapped into the
   normal error format. If overruled, the deferral gets written into
   "Deliberately not done" with reasons, so it stops being re-derived.
7. **CLI paper cuts (§4.4)**, batched by file: binary-stdin decode error in
   `embed`, `chunk ""` falsy-check reading stdin, embed JSONL error line
   attribution, `strict=True` on the zips, the wrong-noun messages
   (`start <file>`, `extract <dir>`), `status -v` file-listing errors
   surfaced, `check_health` crash no longer silently deleting the health
   section, `current_file` included in `--json`.
8. **Trimming (§5).** Last, so cleanup diffs never mix with behavior fixes:
   the 17-key status dict ×2, engine/session boilerplate ×7, the
   `validate_collection_name` wrapper ×9, the ~70 worker/watcher duplicated
   lines, the bucketing loop ×2, the dead `session.commit()`, TOML parsed
   once per `get_config()`, and the stale docstrings/comments — except those
   an earlier batch already touches, which get fixed there.
9. **Test and doc debt (§6–§7).** Regression tests for the three untested
   fourth-review fixes (search-migration commit, `reindex --force`
   failure-atomicity, embed input validation — the last largely produced by
   batch 1), plus whatever README/CHANGELOG claims batches 1 and 4 have not
   already corrected.

### Out of scope here, tracked elsewhere

- The chunk_size 320-vs-352 re-embed decision — handoff block above.
- Environment, not code: the orphaned daemon observed on port 11555 (SIGTERM
  it, per the handoff block's socket note), and the indexed corpus directory
  no longer existing on disk (every current search result carries a dead
  `source_path` — re-point or remove the collection).
- The deferred-features list at the bottom of this document.

### Exit criteria

- Every §1–§7 finding is either fixed with a regression test or explicitly
  moved to "Deliberately not done" with a reason.
- Resolution banner in the 2026-08-17 note, finding → commit.
- `./scripts/check.sh` green, PG gate included.
- No unmerged `claude/*` branch left behind.

Rough sizing: batches 1–4 are one focused session; 5–9 one to two more.

## Plan of record — zero-rough-edges release (2026-08-19; shipped as v0.2.0)

**Goal.** Ship a release with zero rough edges: every known defect either fixed
with a regression test or documented as a limitation with recorded evidence,
the one code path no test evidence covers exercised against the live system,
and the release mechanics done. "Rough edge" is defined by the user's
criterion: anything a real user would hit and be surprised by.

**Re-versioned 2026-08-18: this shipped as `v0.2.0`, not `v1.0.0`.** The work
is unchanged; the user judged the 1.0 stability promise premature after days of
single-user use. The bar for a future v1 is recorded below ("Road to v1").

**Where this starts from.** `main` at `c34b57f` — fifth review closed,
adversarial re-review closed, all five `./scripts/check.sh` gates green.
Working branch: `claude/v1-release`. Current version: `0.1.0b1`.

**Decisions already settled** (user, 2026-08-18 — do not re-litigate):

| Decision | Choice | Rejected alternative and why |
|---|---|---|
| chunk_size | Re-embed at the shipped **320** | Pinning 352 kept the old index but paid a tokenize round trip per chunk forever and left config diverged from the default |
| Release form | **Annotated git tag** (`v0.2.0` after the re-version), no PyPI | PyPI needs an account, a free name, and a publish pipeline nobody has asked for |
| Piped previews | **Full text kept** | Capping reintroduces the substring-grep breakage the review fixed |
| Reopened defects | **Both fixed in v1** | Shipping known silent-failure modes contradicts the zero-rough-edges goal |
| Corpus location | `~/projects/bibs/papers` (5 PDFs, verified readable) | Old `~/bibs/papers` exists but is outside the agent's reach |

### Batch map

Six batches. 1 and 2 are independent of each other and of the corpus; 3 needs
the user present; 4–6 are cheap and sequential. Dependency: 5 must follow 1–4
(it documents their outcomes); 6 is last by definition.

| # | What | Kind | Needs user? |
|---|---|---|---|
| 1 | Two silent-degradation fixes | Code + tests | No |
| 2 | §2.9 directory-move repro | Investigation, then code or docs | No |
| 3 | Live-fire: remove → start at 320 → promote → search → stop | Operation | **Yes** (writes live state) |
| 4 | Cold-start UX line in quickstart | Docs | No |
| 5 | README walked cold + Known limitations section | Docs + verification | No |
| 6 | Version bump, CHANGELOG cut, tag, install-from-tag check | Release | Tag push is user's |

### Batch 1a — `check_health` must detect a wedged daemon

**Symptom.** On 2026-08-15 a daemon answered `/v1/models` but hung every
embedding request for 21 hours; `cementic status` said `embedding healthy` the
whole time.

**Why.** `check_health` (`status_service.py:399`) classifies via
`probe_daemon(..., wait_seconds=0.0)` (`embedding_runtime.py:547`), which only
does a `/v1/models` round trip. A wedged daemon still answers listings — the
probe cannot see that the *embedding* path is dead.

**Design constraint.** The daemon serializes requests behind one model lock, so
a real embedding probe against a daemon mid-batch (30 s+ is normal) times out
too — naive probing misreports *busy* as *wedged*. `status` must also never
block long (that defect was already fixed once; do not reintroduce it).

**Change.**
- `probe_daemon` gains an opt-in second stage: after the model list answers
  HEALTHY, issue a one-token `/v1/embeddings` request with a ~5 s budget.
- On timeout/error, consult the pipeline worker's state file
  (`current_activity` / `current_file`, `state.py:34–49`): a worker mid-embed
  means the daemon is legitimately saturated → `BUSY` (healthy, as today).
  No active worker and no answer → new `DaemonHealth.WEDGED` → unhealthy,
  `llama_daemon` message "running but not answering embeddings".
- `check_health` and `status --doctor` use the two-stage probe; `--doctor` may
  spend a slightly larger budget. Total worst-case `status` latency stays
  under ~7 s and only when the first stage said healthy.

**Tests.** Fake daemon (local HTTP server) that answers `/v1/models` and hangs
`/v1/embeddings`: with no worker activity → `status` reports unhealthy, exits
per the health rules, within the budget. Same fake with a worker state file
showing mid-embed activity → healthy/busy. Genuine fast fake → healthy.
Falsify: revert the second stage, the wedged test must go red.

**Files.** `embedding_runtime.py` (probe), `status_service.py` (wiring),
`doctor.py` (budget), `tests/unit/test_status_service.py`,
`tests/unit/test_embedding_runtime.py`.

### Batch 1b — Nomic v1/v1.5 models must get task prefixes

**Symptom.** Pointing `llama_cpp.model_path` at
`models/nomic-embed-text-v1.5.f16.gguf` (which sits in this repo) silently
disables the asymmetric `search_document:`/`search_query:` prefixes that v1 and
v1.5 need exactly as v2 does. Retrieval degrades measurably; nothing errors.

**Why.** `_NOMIC_V2_MARKER = "nomic-embed-text-v2"` (`embedding_text.py`)
matches only v2 filenames. The wrong behaviour is *pinned by a test*
(`test_embedding_text.py:24–26` asserts `"nomic-embed-text"` gets no prefix),
so it reads as intentional.

**Change.**
- Widen the marker to the family: any model filename containing
  `nomic-embed-text` selects the task-prefix policy. Rename the policy
  constant accordingly (`NOMIC_V2_POLICY` → family name);
  `describe_text_policy` keeps reporting the selection.
- Bump `EMBEDDING_TEXT_FORMAT_VERSION` `"v1"` → `"v2"` (`profiles.py:22`):
  v1/v1.5 vectors embedded under the old rule are unprefixed, and the version
  exists precisely so old and new vectors never mix in one profile. **Impact
  on the live index: none in effect** — the bump changes every profile
  fingerprint, so the next `start` mints a new revision, but batch 3 rebuilds
  from scratch anyway; do batch 1b before batch 3 so the rebuild happens once.
- Retire the pinning test; replace with three: v1.5 filename gets prefixes,
  v2 unchanged, non-Nomic unchanged.

**Files.** `embedding_text.py`, `profiles.py`,
`tests/unit/test_embedding_text.py`.

**Known residual (documented, not fixed).** Matching on *filename* still
mis-selects for a renamed GGUF; reading GGUF metadata is the real fix and
stays deferred ("Deliberately not done", first review). v1 documents the
filename convention in the README.

### Batch 2 — §2.9 directory-move blindness: RESOLVED 2026-08-18

**Measured** (watchdog inotify backend, scripted repro, no cementic):

| Case | Events delivered | Handler coverage |
|---|---|---|
| A: `mv watch/sub watch/sub2` (within) | `DirMovedEvent` + per-file `FileMovedEvent`s | already covered (`on_moved` per file) |
| B: `mv outside/new watch/new` (move in) | `DirCreatedEvent` + per-file `FileCreatedEvent`s | already covered (`on_created` per file) |
| C: `mv watch/sub outside/` (move out) | **one `DirDeletedEvent`, no per-file deletions** | **was uncovered — fixed** |
| D: `mv watch watch2` (root itself) | **nothing at all** | unfixable from inside the watch — documented |

The review's "blind until restart" claim was therefore true for C and D, in the
*stale-results* direction (documents under a moved-out directory stayed
"present"; searches matched dead paths). The feared *silent non-indexing*
direction (files entering unseen) does not occur — case B synthesizes per-file
created events.

**Fix (case C).** `on_deleted` no longer drops directory events: a
`delete_directory_callback` marks every document under the vanished prefix
deleted (trailing-separator match, so `/a/docs` never claims
`/a/docs-archive`), reusing the same purge path as single-file deletion.
Three tests: prefix delete, prefix-sibling safety, and end-to-end through a
real observer (that one verified red with the wiring removed).

**Documented limitation (case D).** inotify delivers nothing when the watched
root itself is moved; the watcher cannot see it. Startup reconciliation repairs
it on the next `cementic start`. Goes in README Known limitations (batch 5).

### Batch 3 — live-fire validation (user present)

The only remaining path with no evidence on the real system, and the execution
of the 320 decision, in one pass. Also live-exercises this week's fixes on the
start path: the index retrofit in `create_tables`, supervisor liveness, the
worker's restructured embed transaction, promote.

**Steps** (each with its expected outcome; stop and diagnose on any mismatch):
1. `cementic status --doctor` → all ok/warning, no fail.
2. `cementic collection remove test --force` → deleted; reports docs/chunks
   removed and vector tables dropped.
3. `cementic start ~/projects/bibs/papers -c test` → workers up;
   `status` shows building revision, documents appearing.
4. Wait ~7 min (5 PDFs, ~250 chunks at 320). `status -c test` → extraction,
   chunking, embedding all complete, 0 failed.
5. `cementic collection promote test` → promoted; revision label reported.
6. `cementic search "language models" -n 3` → hits with **live** paths under
   `~/projects/bibs/papers`, sensible scores.
7. `cementic stop` → both workers stop; `status` shows stopped; exit 0.

**Note.** Search is empty between steps 2 and 5 (~7 min) — accepted when the
fresh-start route was chosen (no old revision to serve).

**Failure rule.** Any step failing reopens code work before release; the fix
gets a regression test and batch 3 restarts from step 1.

### Batch 4 — cold-start UX (docs only)

`search` autostarts the daemon; a cold model load blocks 30 s+ with a stderr
notice. By design. Two doc changes: quickstart gains "run
`cementic embedding start` once after install to pay the model load up front";
the README documents autostart where `search` is introduced.

### Batch 5 — README walked cold + Known limitations

1. In a clean environment (fresh venv; container if Postgres setup is part of
   the walk), follow README top to bottom **exactly as written** — install,
   `init postgres`, config, start, promote, search. Every text/behaviour
   mismatch is a defect: fix the text or the behaviour, nothing else.
2. Add a **Known limitations** section (user-facing wording; reasons stay
   here): one background session at a time (by design); ANN pre-filter recall
   on shared vector tables; model identity matched by filename (batch 1b
   residual); §2.9's verdict from batch 2; anything batch 3 surfaced and
   deliberately did not fix.

### Batch 6 — release mechanics

1. `pyproject.toml` version `0.1.0b1` → `1.0.0`.
2. CHANGELOG: cut `[Unreleased]` → `[1.0.0] — <date>`. (CHANGELOG.md stays:
   the no-changelog default is for repos without external consumers, which
   stops applying at a tagged release.)
3. All five gates green at the release commit.
4. Merge `claude/v1-release` → `main` (user), then annotated tag `v1.0.0` on
   main, message = release highlights (user pushes tag).
5. In a clean environment: `uv tool install git+<repo-url>@v1.0.0`, run the
   quickstart's first commands. This is the last gate — it catches packaging
   problems (missing files in the sdist, entry-point breakage) that no test
   in the repo can.

### Out of scope, tracked elsewhere

- Everything in `TODO.md` (extractors, `cementic add`, multimodal, search
  enrichment) — features, not edges; post-v1 by definition.
- PyPI publication — revisit if anyone outside this machine wants
  `pip install cementic`.
- GGUF-metadata model identity (batch 1b residual), the kept §5 trims, the
  TOML re-parse — reasons under "Deliberately not done".

### Exit criteria (all must hold at the tagged commit)

- [x] Five `./scripts/check.sh` gates green (last run: at the release commit).
- [x] Batch 1a: wedged-daemon fake test red-green verified (04e974a).
- [x] Batch 1b: v1.5 prefix test in place, pinning test retired, format
      version bumped v1→v2 (9f1d148).
- [x] Batch 2: §2.9 resolved — move-out case fixed with three tests (e2e one
      verified red), root-move case documented with the measured evidence
      table (b1999da).
- [x] Batch 3: all seven steps passed against the live database 2026-08-18 —
      remove (5 docs/249 chunks), rebuild 274/274 embedded 0 failed at 320,
      revision `default-68e212bc-llama-cpp-12f77de0` promoted, search returns
      live `~/projects/bibs/papers` paths, clean stop. The probe reported
      `embedding healthy` *during* the build — the busy/wedged disambiguation
      working live.
- [x] README walked cold in a fresh venv (install, init postgres, doctor,
      extract|chunk|embed): one mismatch found and fixed (missing C/C++
      toolchain note); Known limitations section present (2a5428f).
- [x] Version cut (re-versioned to 0.2.0), CHANGELOG cut. Remaining, in
      order (user steps marked): merge `claude/v1-release` → `main` (user),
      tag `v0.2.0` on main (user), `uv tool install` from the pushed tag in a
      clean environment (agent can verify once the tag exists; install from
      local source already verified at the release version).
- [x] Tag pushed by the user 2026-08-18 (`v0.2.0` = merge commit `9c9273f`).
      Install verified in a clean venv from the exact tagged tree
      (`git archive v0.2.0`): reports 0.2.0, doctor ok. The literal
      `uv tool install git+https://...@v0.2.0` could not run from the agent
      sandbox (private repo, no agent credentials; the git+file:// route is
      blocked by the ref-transaction guard) — the tagged *tree* installing
      cleanly is the same evidence minus network transport. **Plan closed:
      v0.2.0 shipped.**

### Road to v1 (the user's bar, recorded 2026-08-18)

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

The target is a large personal corpus (~40k papers and textbooks). At the
measured **50 chunks per paper** that is ~2M vectors. The machinery is justified
rather than over-engineered:

- "Indexed" and "fast" are not in tension — indexing is the amortized one-time
  cost; an ANN index keeps *queries* sub-second over millions of vectors.
  Measured 2026-08-15: a warm search is **214 ms end to end, 197 ms of which is
  embedding the query string** — a constant that does not grow with the corpus.
  A bare KNN over 10k vectors is 5.6 ms. Query latency is not the scaling risk.
- The scaling risks are the other two axes, and both bind earlier. **Embedding
  throughput** measured 1.4 s/chunk on CPU, so 2M vectors is ~780 hours of
  continuous embedding — "amortized one-time" is measured in weeks, not hours,
  without a GPU. **Memory** binds before latency does: an HNSW index over 1M
  768-dim vectors needs ~3 GB resident, against a dev machine with ~3 GB free,
  which is the point of the `diskann` seam below.
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

## Review history and what is still open

**Everything found by the first five reviews is fixed and merged**, except the
items under "Deliberately not done" below. The fifth pass (2026-08-17) was closed
on 2026-08-18, and an adversarial re-review of those fixes followed the same day
(see below). The notes are the record of what each found; this section keeps only
the engineering *lessons and measurements* that have no other home, in the order
they were learned.

`notes/` is gitignored as of `c7069e1`, so the files below are local to the
working copy and are not in a fresh clone. They remain in history up to that
commit; `git show <rev>:notes/<file>` retrieves any of them.

- `notes/code-review-2026-08-07.html` — first full pass.
- `notes/code-review-2026-08-11.html` — third
  pass, five unprimed reviewers, so its overlaps are independent re-derivations.
- `notes/code-review-2026-08-14.html` — fourth
  pass. Carries a resolution banner mapping every finding to the commit that
  closed it, and a reconciliation of the two earlier notes, so a fifth review
  starts from that rather than re-deriving.
- `notes/code-review-2026-08-17.html` — fifth
  pass, reviewing the fourth pass's fixes plus fresh eyes per subsystem. Headline
  pattern: several fixes are correct on the path they touched and absent on an
  adjacent path the same defect reaches (`status --json`, the initial scan,
  `stop`'s kill loop, `config path`). **Closed 2026-08-18** on
  `claude/review-fixes-2026-08-17`; the note carries a resolution banner mapping
  finding → commit. Still open from it: §2.9's directory-move blindness
  (unverified, needs a repro), the TOML re-parse, and three §5 trim candidates —
  all under "Deliberately not done".
- **Adversarial re-review of those fixes (2026-08-18)**, six reviewers primed to
  refute, disjoint scopes, every claim re-verified in-parent before acting. It
  found that the fixes had introduced six new defects of their own — a daemon
  lock taken twice in one process (a deterministic 180 s hang on every
  runtime-config change), a `chunk_size` validator that refused the value this
  plan documents as the way to keep the live index, a partial unique index that
  could not be built on the databases it existed to protect, an unguarded DDL
  race between the two workers, `chunk ""` rejecting piped stdin, and a `"None"`
  string in the machine-readable `status --json`. **Lesson: a fix reviewed only
  by the pass that wrote it is unfinished.** The adversarial pass cost about as
  much as the fifth review and found defects of the same severity, in code that
  had just been written to close defects.

**Read the 2026-08-17 and 2026-08-14 notes before opening a new review.** Its most useful section
is not the findings but the ledger of what the earlier passes found and never
fixed — roughly fifteen items were re-derived independently three times before
anyone acted on them.

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
   builders as `CURRENT_CONTENT_SQL` — *since removed from `src/`; the freshness
   join was replaced by deleting stale rows eagerly, and the constant now lives
   in `tests/integration/test_pg_helpers.py` as a fixture invariant*),
   C1.1 + N5 (query bounded in tokens
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
  functions set those on the *failure* path too (in `_step_extract`'s and
  `_step_chunk`'s write-back blocks — line numbers have drifted since), so the
  fixtures, not the new predicates, were wrong. Check that before assuming a
  similar failure means a regression.
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

### Fixed on `claude/ann-filterable-vectors`

**The ANN index is now reachable.** The filter columns (`collection`,
`extractor_profile_id`, `chunk_profile_id`) live on `embedding_vectors_p*`, so
the `WHERE` clause applies to the vector row and the planner can drive from the
index scan. Same data, same index, 100k rows at 768 dimensions:

| query shape | ANN used | results | time |
|---|---|---|---|
| filters on joined tables (before) | no | 10/10 | 407.6 ms |
| filters on the vector row | yes | 10/10 | 1.0 ms |
| filters on the vector row, 2% slice, `iterative_scan=off` | yes | **0/10** | 1.3 ms |
| filters on the vector row, 2% slice, `relaxed_order` | yes | 10/10 | 15.3 ms |

`hnsw.iterative_scan` landed with it, defaulting to `relaxed_order` and gated on
pgvector ≥ 0.8 — row 3 is why. The gate is not optional: PostgreSQL accepts an
unknown *qualified* setting as a placeholder until the defining module loads on
that connection and rejects it with `InvalidName` afterwards, so behind a
connection pool an ungated `SET` fails only on connections that had already run
a vector query. (`SET LOCAL diskann.query_rescore` being accepted where
pgvectorscale is absent is the same effect — not evidence it took effect.)

**What the three columns cost.** They replace query-time freshness filtering
with an invariant: stale vectors are deleted when they go stale. Re-chunking
already did this via the `chunks_v2` cascade; two paths did not and now do —
document deletion (`_purge_document_chunks`) and re-extraction that changes the
content (`_purge_superseded_chunks`). The visible behaviour change: a document
whose re-extraction succeeded but whose re-chunking has not run returns nothing
rather than its previous contents.

Existing vector tables are migrated in place by `ensure_vector_table_schema` —
`ADD COLUMN`, backfill from the joins, then `SET NOT NULL`. No re-embedding.

**No test reproduces the thin-slice recall failure**, deliberately. It needs
~100k rows: below that the planner picks an exact sequential scan for a
selective filter, which returns the right answer and would make the test pass
for the wrong reason. Verified that trap directly — at 4k rows the `slice`
query plans as a seq scan even with `enable_seqscan = off`. What CI does pin is
that the ANN index *is* in the plan, which is the thing that regressed.

### Fixed on `claude/index-build-stage-1`

**The index build is faster, visible, and explicable.** `ensure_revision_ann_index`
runs synchronously in `_mark_revision_ready_if_complete`, at the
`building → ready` transition — *not* at promotion, as an earlier note here
said.

- `index.build_memory` (default 2GB) raises `maintenance_work_mem` on the
  build's own connection. 100k × 768 is 293 MiB of graph against Postgres's
  64MB default, so the build spilled: **1454 s at 64MB, 345 s at 2GB**.
- The worker publishes `current_activity` around the build, because nothing can
  be published from inside it. `cementic status` previously showed a running
  worker, a `building` revision and no current file — identical to idle.
- `cementic stop` waits 10 s and then *refuses* (it does not force-kill; the
  5-second SIGKILL is `_terminate_managed`, used only on start-up rollback). The
  worker cannot answer SIGTERM from inside `CREATE INDEX`, so that timeout was
  guaranteed and read as a hang. Both the timeout and `--force` now say what is
  running and that forcing discards it.

Verified that a killed build loses everything: on a 40k table whose build takes
135 s, killing the builder at ~34 s leaves only the primary-key index.

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

### Superseded

**The ANN index is never used by cementic's search query.** *(Fixed above; kept
because the measurements are the justification for the schema change.)* Measured
with `EXPLAIN (ANALYZE)` against a real corpus:

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
  linearly. (Superseded: the filter columns below made the ANN index reachable,
  so search no longer scales with table size.)
- `index.method`, `hnsw_m`, `ef_construction`, `hnsw_ef_search`, the DiskANN
  knobs and `collection reindex` all maintain an index nothing reads.

A no-schema-change alternative was considered and rejected: a materialised CTE
doing the bare KNN first, then filtering. It takes the global top-N and filters
afterwards, which is arithmetically the same as `iterative_scan=off` — it
degrades to zero results in exactly the case that motivates it.

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
- Hybrid lexical + vector search (exact author names, acronyms, equation labels).
- A multi-profile embedding daemon pool, if old-model search and new-model
  indexing must run concurrently.
- Lighter embedding providers if llama.cpp memory use is too high on small
  machines.
