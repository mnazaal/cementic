# Roadmap

Baseline: `./scripts/check.sh` runs all six gates — lockfile, ruff, mypy, unit,
integration, integration-pg — and reports a missing PostgreSQL as SKIPPED rather
than passed. Use it rather than running the pieces by hand; the reason three PG
tests once reached `main` red is that "I ran the tests" meant unit-only.

The PG gate needs a Postgres with pgvector + vectorscale (generate one with
`cementic init postgres ./cementic-postgres`, then `docker compose up -d` or
`podman compose up -d`); the integration suite can bring the compose stack
up/down itself when a container engine is available.

Design rationale and the pluggable seams (embedding provider, extractor, ANN
index, versioned revisions) are documented in [PLAN.md](PLAN.md).

## Near term

- Add more document-type extractors (`.docx`, `.pptx`, `.html`, `.epub`) — one
  `_EXTRACTORS` entry each now that the content-type registry and watcher exist.
- Add a direct ingestion command: `cementic add <path> -c <collection>`.
- Add optional Markdown mirrors for extracted artifacts, alongside the existing
  compressed pipeline artifacts.
- Enrich search results with document id and optional artifact path. (`collection`
  already ships — a result carries collection, source_path, content, score,
  distance and score_kind.)
- Match database URLs with `make_url` rather than `startswith("postgresql://")`
  (`db.py:316`). The current check skips the `gssencmode` connect-arg for
  driver-qualified URLs like `postgresql+psycopg2://`, which a user setting
  `CEMENTIC_DB_URL` may well write. Carried over from PLAN.md's "Deliberately
  not done", where it was parked behind "only if `db.py` is open for another
  reason" — a condition that has since been met twice.

## Images / multimodal

- Add an image extractor (`.png` / `.jpg`) — OCR/caption into the Markdown text
  path, or a raw-image passthrough for a vision embedder.
- Add a multimodal embedding provider (one `_PROVIDER_FACTORIES` entry + one
  `runtime_spec_from_config` branch). A vision embedder uses a parallel path that
  bypasses the Markdown IR (see PLAN.md).

## Test coverage / CI

CI is done — `.github/workflows/ci.yml` runs unit, `integration -m "not pg"` and
`integration -m pg` as separate jobs; the last has no `services:` block, since
the `pg_engine` fixture brings up `compose.yml`'s pgvector+vectorscale
container itself. Keep it green with `./scripts/check.sh` before pushing.
Remaining gaps, from the 2026-08-14 review:

- Under Python 3.14, several unit tests leak sqlite3 connections (unclosed
  engines in test fixtures) and pytest 9's unraisable-exception hook escalates
  the ResourceWarnings to failures with a GC-timing-dependent failing set.
  Invisible on CI's pinned 3.12; found by a CI-simulation audit 2026-08-18.
  Close the engines (`engine.dispose()` in fixtures) before any 3.14 upgrade.
- No test connects a pipeline failure to what `cementic status` prints. The
  DB-row assertions and the status-rendering assertions never meet, which is how
  a whole class of "reports success, dropped the work" defects stayed invisible
  to a green suite.
- `collection reindex --force` is never exercised through the CLI; only the
  negative (`force is False`) is asserted, so the flag-to-kwarg wiring is
  untested for the one flag README calls the only way to pick up changed
  `hnsw_m` / `hnsw_ef_construction`.

## Later

- Hybrid lexical + vector search for exact author names, acronyms, citations, and
  equation labels.
- A multi-profile embedding daemon pool if old-model search and new-model
  indexing need to run concurrently.
- Evaluate lighter embedding providers if llama.cpp memory use is too high on
  small machines.

## Parked, with a trigger

Deliberately not done, each with the observable that should reopen it. Parked
without a trigger is indistinguishable from forgotten, so if you add one here,
give it a condition someone could actually notice.

- **Collapse `config.py`'s remaining import cycles** into a dependency-free
  module holding registry *names*, so `config` can validate against them without
  importing the implementations. Three validators still use function-local
  imports for this, and one reaches for a private symbol
  (`_TASK_PREFIX_TOKEN_ALLOWANCE`). *Reopen when:* `config.py` is next open for
  another reason — the change touches four modules and is not worth a slot on
  its own.
- **Publish the repo.** Nothing gates on it today; the install docs no longer
  describe a command that cannot work. Publishing only changes whether a
  `pipx install git+https://...` one-liner works for someone who is not the
  author. *Reopen when:* you want to hand cementic to another person — at that
  point README's install section goes back to the one-liner in the same commit.
- **Restore support for Python older than 3.12.** *Reopen when:* cementic needs
  to run somewhere that cannot get 3.12. Unlikely while `uv` installs one in a
  single command, and the compatibility branch was deleted precisely because
  nothing exercised it.
- **The embedding throughput ceiling.** At the measured 1.4 s/chunk on CPU, the
  ~2M-vector target corpus in README's measurements is roughly 780 hours of
  embedding. *Reopen when:* a real indexing run makes that concrete — it is a
  project-shaping constraint, not a defect, and it is what "evaluate lighter
  embedding providers" above is actually for.
