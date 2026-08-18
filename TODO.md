# Roadmap

Baseline: `./scripts/check.sh` runs all five gates — ruff, mypy, unit,
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

## Images / multimodal

- Add an image extractor (`.png` / `.jpg`) — OCR/caption into the Markdown text
  path, or a raw-image passthrough for a vision embedder.
- Add a multimodal embedding provider (one `_PROVIDER_FACTORIES` entry + one
  `runtime_spec_from_config` branch). A vision embedder uses a parallel path that
  bypasses the Markdown IR (see PLAN.md).

## Test coverage / CI

CI is done — `.github/workflows/ci.yml` runs unit, `integration -m "not pg"` and
`integration -m pg` as separate jobs, the last against a service container. Keep
it green with `./scripts/check.sh` before pushing. Remaining gaps, from the
2026-08-14 review:

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
