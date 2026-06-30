# Roadmap

Baseline: unit tests, ruff, and mypy pass locally. PostgreSQL integration tests
require a running Postgres with pgvector + vectorscale (e.g. `docker compose up
-d` or `podman compose up -d`); the integration suite brings the bundled compose
stack up/down itself when a container engine is available.

Design rationale and the pluggable seams (embedding provider, extractor, ANN
index, versioned revisions) are documented in [PLAN.md](PLAN.md).

## Near term

- Add more document-type extractors (`.docx`, `.pptx`, `.html`, `.epub`) — one
  `_EXTRACTORS` entry each now that the content-type registry and watcher exist.
- Add a direct ingestion command: `cementic add <path> -c <collection>`.
- Add `cementic collection reindex <collection>` to rebuild the active revision's
  ANN index under the current `index.method` (the build path already reconciles a
  changed method; this just gives it a one-command trigger without a re-embed).
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
