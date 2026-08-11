# Roadmap

Baseline: unit tests, ruff, and mypy pass locally. PostgreSQL integration tests
require a running Postgres with pgvector + vectorscale (generate one with
`cementic init postgres ./cementic-postgres`, then `docker compose up -d` or
`podman compose up -d`); the integration suite can bring the compose stack
up/down itself when a container engine is available.

Design rationale and the pluggable seams (embedding provider, extractor, ANN
index, versioned revisions) are documented in [PLAN.md](PLAN.md).

## Near term

- Add more document-type extractors (`.docx`, `.pptx`, `.html`, `.epub`) — one
  `_EXTRACTORS` entry each now that the content-type registry and watcher exist.
- Add a direct ingestion command: `cementic add <path> -c <collection>`.
- Add `cementic collection reindex <collection>` to rebuild the active revision's
  ANN index under the current `index.method`. This is the *only* trigger there
  would be: the reconciling code runs solely while a revision is still building,
  and changing `index.method` creates no new revision, so on a built collection
  the setting currently does nothing. Search is unaffected — it tunes for the
  index that actually exists — so this is a dead knob, not wrong results.
- Add optional Markdown mirrors for extracted artifacts, alongside the existing
  compressed pipeline artifacts.
- Enrich search results with document id, collection, and optional artifact path.

## Images / multimodal

- Add an image extractor (`.png` / `.jpg`) — OCR/caption into the Markdown text
  path, or a raw-image passthrough for a vision embedder.
- Add a multimodal embedding provider (one `_PROVIDER_FACTORIES` entry + one
  `runtime_spec_from_config` branch). A vision embedder uses a parallel path that
  bypasses the Markdown IR (see PLAN.md).

## Test coverage / CI

- Keep `pytest tests/unit`, `ruff check src/ tests/`, and `mypy src/` clean.
- Run the PostgreSQL integration tests before release checks.
- Add CI that runs unit tests by default and the PG tests when a service
  container is available.

## Later

- Hybrid lexical + vector search for exact author names, acronyms, citations, and
  equation labels.
- A multi-profile embedding daemon pool if old-model search and new-model
  indexing need to run concurrently.
- Evaluate lighter embedding providers if llama.cpp memory use is too high on
  small machines.
