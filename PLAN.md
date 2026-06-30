# cementic — architecture & design

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
