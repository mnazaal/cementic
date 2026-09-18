# Configuration

You can index and search with the defaults. Create a config file only when you
need to change how cementic connects, extracts, embeds, or indexes documents.

```bash
cementic config init      # write the annotated defaults
cementic config path      # show the file cementic would use
cementic config show      # show the effective merged configuration as JSON
```

The file written by `config init` is the complete reference for setting names,
defaults, and comments. It is the source of truth; this guide explains the
choices that change behavior.

## How settings are chosen

Higher-precedence sources override lower-precedence sources:

```text
built-in defaults < config file < CEMENTIC_* environment variables < command flags
```

cementic checks `CEMENTIC_CONFIG`, then `./cementic.toml`, then
`~/.config/cementic/config.toml`.

Use the generated local Postgres defaults unless you already manage a database.
For an existing database, set `CEMENTIC_DB_URL`.

## Choose an extraction path

The default PDF extractor preserves document structure. For a large collection
of born-digital PDFs, choose `pymupdf-raw` in the generated config: it reads the
text layer only and is much faster.

A scan without a text layer fails rather than adding an empty document. To use
OCR, install the optional `rapidocr` dependency, enable `use_ocr`, and keep the
layout-aware PDF extractor selected. The generated config shows those settings.

To support another file type, configure the command extractor. It receives a
path and must write only extracted text to stdout. Provide its version command
too, so tool upgrades create a new extraction profile instead of silently
changing indexed text.

Changing an extractor setting creates a replacement revision. See
[Operations](operations.md#revisions).

## Choose a search index

Use the default HNSW index when low query latency and sufficient RAM matter.
Use DiskANN when the collection is large enough that a disk-resident index is a
better trade-off.

Changing the index method or its build settings does not re-embed documents.
Rebuild the index afterward:

```bash
cementic collection reindex research
```

Use `--force` when rebuilding after changing an HNSW build setting that keeps
the same index method.

## Choose an embedding runtime

cementic starts `llama-server` on demand by default. Point the generated config
at a GPU-capable build when you have one, or disable autostart to use an
externally managed OpenAI-compatible embedding server.

Auto-download only writes a missing model below cementic's user-data directory.
If you configure another location, place the model file there before running
cementic. Changing the embedding model or runtime profile creates a replacement
revision. Choose the intended runtime before a bulk import.

## Inspect search behavior

Hybrid search combines semantic retrieval with PostgreSQL full-text search. It
helps rare names and acronyms without making broad queries literal-only. Disable
it in the generated config only when vector-only search is intentional.

`cementic search --json` emits JSON Lines for scripts. Use `--scores` when you
need raw relevance scores for a human-readable search result.
