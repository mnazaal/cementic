# cementic

Concrete semantic search for your documents.

`cementic` watches one or more directories, registers documents (PDF, Markdown, and plain text), extracts text, chunks that text, embeds the chunks, and serves semantic search over the active revision for each collection.

## How It Works

The index is built as a versioned pipeline:

- extractor profile -> extracted document artifact
- chunk profile -> chunk artifacts
- embedding profile -> chunk embeddings
- pipeline revision -> the search-visible release that ties those three profiles together

This keeps old search available while a new extractor, chunking policy, or embedding model builds in the background.

## Features

- Watches directories for documents (PDF, Markdown, plain text) and records them as source documents
- Adds new document types through a small extractor registry — one entry per type
- Stores extracted text as compressed artifacts for cheap rechunking
- Separates extraction, chunking, and embedding so each stage can evolve independently
- Keeps one active searchable revision per collection while a replacement revision builds
- Uses explicit `cementic collection promote` to switch search to a new ready revision
- Searches meaning and exact words together, so a rare surname or acronym is findable

## Installation

```bash
git clone https://github.com/mnazaal/cementic.git
cd cementic
uv pip install -e ".[dev]"
```

The repository is private, so installing straight from the URL
(`pipx install git+https://...`) only works once you have access to it; clone
first.

cementic is verified for Linux with Python 3.12. macOS and Windows are
best-effort until tested. No llama.cpp Python binding is installed: cementic
drives an OpenAI-compatible embedding server over HTTP and never links
llama.cpp, so `llama-server` is a system prerequisite alongside Postgres —
see "Prerequisites". The build you install is the build that serves, GPU
support included, and there is no second copy of ggml in the Python
environment to drift from it. The embedding model itself is downloaded on
first use when auto-download is enabled.

## Prerequisites

- **PostgreSQL with pgvector + vectorscale** — provision it however you like. The
  easiest path is generating a local setup via `cementic init postgres`, then
  starting it with **Docker** or **Podman** (see Setup). cementic itself does
  not manage containers.
- **llama.cpp's `llama-server`** on PATH — cementic spawns and supervises it but
  ships no embedding server of its own, so install a build (a GPU one if you
  want a GPU; see "Running embeddings on a GPU") or point
  `llama_cpp.daemon_command` at one you already have. `cementic doctor` checks
  that it resolves *and* runs.
- **Python 3.12+**
- **8-16 GB RAM** recommended for the llama.cpp embedding backend (the model loads into memory)
- **Disk space**: ~2 GB for the llama.cpp model, plus PostgreSQL data and artifact storage

## Setup

Generate a local Postgres setup directory, then start it once as a persistent
local service. This does not make cementic manage containers; it only writes
copy-pasteable setup files.

```bash
cementic init postgres ./cementic-postgres
cd ./cementic-postgres
docker compose up -d      # or: podman compose up -d
cementic doctor
```

You do **not** run Compose every time you use cementic. The generated Compose
file uses `restart: unless-stopped`; optional Podman Quadlet/user-systemd files
are generated for users who prefer a user service.

The generated setup binds PostgreSQL to `127.0.0.1:5432` with the default
`CEMENTIC_DB_*` values.
To use a Postgres you manage yourself, just point `CEMENTIC_DB_URL` (or the
`CEMENTIC_DB_*` variables) at it — it must have the `pgvector` and `vectorscale`
extensions available.

cementic connects to that database and, on first indexing/search run, validates
or auto-downloads the configured `llama.cpp` model into the cementic user data
directory when it is missing. Auto-download is the recommended path; use
`cementic doctor` to check the resolved model path without downloading.
If Postgres is not reachable, doctor suggests generating the local setup or
pointing cementic at your own Postgres. It also reports faults in an existing
corpus without changing anything — a document counted as chunked that holds no
chunks, for instance, which matches nothing in search until it is re-chunked.

After the doctor check, run `cementic embedding start` once. It is optional —
any command that needs embeddings starts the daemon on demand — but a cold
start loads a ~2 GB model and can take 30 seconds or more, and it is nicer to
pay that now than inside your first `cementic search`.

### Running Postgres as a persistent service (optional, Podman + systemd)

If you'd rather not start/stop the container by hand, or you're running cementic
from an environment (CI runner, sandboxed agent, etc.) that can't reach your
host's container engine, you can run Postgres as a persistent
user-level `systemd` service via [Podman Quadlet](https://docs.podman.io/en/latest/markdown/podman-systemd.unit.5.html)
using the `quadlet/cementic-postgres.container` file included in the setup
directory generated by `cementic init postgres`. See the comments at the top
of that file (and the "Optional Podman Quadlet service" section of its
generated README) for the one-time build/install/start steps.
Once started, Postgres starts with your login session (or survives
logout/reboot if you also run `loginctl enable-linger $USER`) and any
`cementic` command — or any agent running one — just needs network access to
it; nothing needs to start the container itself anymore.

## Usage

The commands below are shown in the order a fresh database needs them:
`cementic start` creates the database schema as a side effect of the workers
it spawns, so on a database with nothing indexed yet, `status`, `search`, and
`collection list` all exit 1 (message: "nothing indexed yet — run
`cementic start DIRECTORY -c COLLECTION` first") until `start` has run at
least once.

```bash
# Start watching a collection and building its target revision
cementic start /path/to/pdfs --collection research

# Inspect current worker state and active/building revisions
cementic status

# Show known collections
cementic collection list

# Search the active revision, falling back to an in-progress build if a
# collection has never been promoted (1-50 results; default 10).
# Starts the embedding daemon on demand: the first search after a reboot
# can block 30s+ on the model load (a notice is printed to stderr) —
# `cementic embedding start` ahead of time avoids that.
cementic search "vector database design" -c research

# Run one document through the pipeline with no database — stdin/stdout filters,
# handy for debugging a single file end-to-end
cementic extract paper.pdf | cementic chunk | cementic embed

# Manage the embedding runtime explicitly when needed
cementic embedding status
cementic embedding start
cementic embedding stop

# Promote the newest ready revision to active
cementic collection promote research

# Inspect revision history for one collection
cementic collection revisions research

# Rebuild the active revision's ANN index after changing [index] settings
cementic collection reindex research

# Stop cementic's background workers (Postgres is left running)
cementic stop

# Delete one collection and its stored artifacts
cementic collection remove research --force
```

Useful flags beyond the above:

| Flag | Applies to | What it does |
| --- | --- | --- |
| `-V`, `--version` | root | Print the version and exit |
| `-v`, `--verbose` | `status` | Per-file pipeline progress, watched directories, worker PIDs |
| `--json` | `status`, `doctor`, `search` | Machine-readable output (`search` emits JSONL, one object per line) |
| `-c`, `--collection` | `start`, `status`, `search` | Which collection to act on. `search` accepts more than one — `-c work personal` or repeated `-c` — and searches all of them together |
| `-n`, `--top-k`, `--limit` | `search` | Number of results, 1–50 |
| `--force` | `stop` | SIGKILL workers that ignored the graceful stop, discarding in-progress work |
| `-f`, `--force` | `collection promote` | Promote despite failed documents or chunks |
| `-f`, `--force` | `collection reindex` | Rebuild even when the method is unchanged, to pick up `hnsw_m` / `hnsw_ef_construction` |
| `--force` | `collection remove` | Skip the confirmation prompt |
| `--force` | `config init`, `init postgres` | Overwrite existing files |
| `--chunk-size`, `--chunk-overlap` | `chunk` | Override the configured chunking for this run |

Naming a collection that does not exist is an error, not an empty result:
`status -c`, `collection revisions`, `collection promote`, `collection reindex`
and `search -c` each say so and exit non-zero. `collection remove` is the
deliberate exception — removing something already gone reports `not found` and
succeeds, so it stays safe to run twice.

A collection name must start and end with a letter or digit, contain only
letters, digits, hyphens, and underscores, and be 100 characters or fewer.
Every command that takes `-c`/`--collection` enforces this and rejects the
name up front rather than at the database.

`search --json` emits one JSON object per line, each with the same seven
fields: `collection`, `source_path`, `content`, `score`, `distance`,
`score_kind`, and `rank` (the 1-based position in the final order, stamped
once the order is settled so it cannot disagree with the list it describes).

A query is rejected before it reaches the embedding model if it is longer
than 8,000 characters, or if its estimated token count exceeds the indexed
model's context window; both surface as a "query too long" error.

Searching collections indexed by different embedding models in one `-c a b`
call is a hard error, not a merged ranking — see Known limitations.

Exit codes follow one convention: **0** — the operation happened (including a
no-op documented as safe, like removing an already-absent collection); **1** —
it could not happen (database unreachable, a refused or empty promote, workers
that would not stop, an unknown collection); **2** — a usage error from the CLI
parser (unknown flag, missing argument). Errors and hints print to stderr;
stdout carries only the command's output, so `--json` streams stay parseable
under `jq` even when something fails.

`cementic start` runs a single background session (one source watcher + one pipeline worker)
at a time, tracked in one supervisor state file. Running `cementic start` again for a different
collection while one is already active refuses with "Background cementic processes already
running" — run `cementic stop` first to switch collections. To index multiple directories into
one collection concurrently, pass them all to a single `cementic start` call (it accepts more
than one directory); indexing two different collections at the same time is not supported.

## Revision behavior

- Model change:
  - reuses extracted text and chunks
  - builds only new embeddings
- Chunking change:
  - reuses extracted text
  - rebuilds chunks and embeddings
- Extractor change:
  - rebuilds extraction, chunks, and embeddings

Search keeps using the active revision until you explicitly run `cementic collection promote`.
Embeddings are produced by a local `llama.cpp` server.

If a revision finished building with failed documents or chunks, `cementic collection promote`
refuses it and prints the failure counts, so you never silently publish a partial index. Re-run
with `--force` to promote it anyway.

A document that fails to extract, chunk, or embed is recorded as failed and does not block the
rest of the build from finishing; failed documents are retried automatically the next time you
run `cementic start`. Deleting a file from a watched directory drops it from search — while
cementic is running, including mid-index, and on the next `cementic start` for files removed
while it was stopped. If the embedding runtime becomes unreachable mid-build, the affected
chunks stay pending and are retried rather than recorded as failed.

If a background worker hits an error it cannot recover from immediately, it retries with a
backoff and reports the reason as `last error` in `cementic status` (and in `status --json`),
so a worker stuck on a persistent problem is visible without reading log files.

Reverting your configuration back to a revision you previously built (for example, rolling back
a model or chunking change) resumes that revision from its existing artifacts and makes it
promotable again.

For a brand-new collection with no active revision yet, search can use the in-progress build and return
partial results from chunks whose embeddings are already available.

## Configuration

cementic can be configured by a TOML config file, by environment variables, or both.
Precedence, lowest to highest:

```
built-in defaults  <  config file  <  CEMENTIC_* env vars  <  command-line flags
```

The defaults target local development; override `CEMENTIC_DB_PASSWORD` on shared
machines or any non-local deployment. The generated Postgres setup uses the same
`CEMENTIC_DB_*` values and binds PostgreSQL to `127.0.0.1` by default.

### Config file

```bash
cementic config init      # write an annotated config.toml to ~/.config/cementic/
cementic config path      # print the active config file (or where it would live)
cementic config show      # print the effective merged config as JSON (pipe to jq)
```

cementic looks for a config file at `CEMENTIC_CONFIG` (if set), then `./cementic.toml`,
then `~/.config/cementic/config.toml`. Everything is optional — set only what you
want to change:

```toml
# ~/.config/cementic/config.toml
[database]
host = "localhost"
port = 5432

[pipeline]
embedding_provider = "llama-cpp"
chunk_size = 320

[index]
method = "hnsw"   # or "diskann"

[search]
# Blend exact-word matching with vector search (default true).
hybrid = true
# A one-word query matching at most this many documents is led by exact matching.
lexical_lead_max_documents = 20

[llama_cpp]
model_path = "models/nomic-embed-text-v2-moe.Q8_0.gguf"
```

Model identity is the file's content digest, so any spelling of `model_path`
that resolves to the same file is the same profile. Pointing it at a
*different* file re-embeds the corpus.

#### Every setting

Anything below can go in the config file under its section, or be set as the
matching `CEMENTIC_*` variable (see [Environment variables](#environment-variables)).

| Section | Key | Default | What it does |
| --- | --- | --- | --- |
| `database` | `host`, `port`, `name`, `user`, `password` | `localhost`, `5432`, `cementic`, `cementic`, `cementic` | Connection parts |
| | `url_override` | unset | Whole connection URL, bypassing the parts above (`CEMENTIC_DB_URL`) |
| `pipeline` | `embedding_provider` | `llama-cpp` | Only provider currently registered |
| | `chunk_size` | `320` | Tokens per chunk, counted with tiktoken — **not** the model's tokenizer. Must stay well under `llama_cpp.n_ctx`; see below |
| | `chunk_overlap` | `80` | Token overlap between neighbouring chunks |
| `index` | `method` | `hnsw` | `hnsw` or `diskann` |
| | `hnsw_m`, `hnsw_ef_construction` | `16`, `64` | Build-time graph knobs; fixed into the index, so changing them needs `collection reindex --force` |
| | `hnsw_ef_search` | `40` | Query-time candidate list; raised automatically to at least `top_k` |
| | `hnsw_iterative_scan` | `relaxed_order` | `off`, `relaxed_order`, or `strict_order`; needs pgvector 0.8+ |
| | `diskann_num_neighbors`, `diskann_search_list_size` | `50`, `100` | DiskANN build knobs |
| | `diskann_query_rescore` | `50` | DiskANN query-time rescoring depth |
| | `build_memory` | `2GB` | `maintenance_work_mem` for index builds only |
| `llama_cpp` | `model_path` | Nomic model (downloaded on first use) | GGUF to load |
| | `n_ctx` | `512` | Model context window. The default model's architecture caps at 512; raising it past what the model supports has no effect |
| | `n_gpu_layers` | `0` | Layers offloaded to GPU |
| | `embedding_dim` | `768` | Fallback only; the live model is probed |
| | `daemon_autostart` | `true` | Start the embedding server on demand |
| | `daemon_host`, `daemon_port` | `127.0.0.1`, `11555` | Where the embedding server listens |
| | `daemon_start_timeout_seconds` | `120` | How long to wait for a cold start |
| | `llama_embed_timeout_seconds` | `120` | Per-request embedding timeout |
| | `daemon_pid_file`, `daemon_log_file` | under the data dir | Daemon bookkeeping |
| | `daemon_command` | `llama-server ...` | Embedding-server command cementic spawns and supervises; defaults to upstream `llama-server` on PATH. Placeholders `{model}` `{alias}` `{host}` `{port}` `{n_ctx}` `{n_gpu_layers}` `{verbosity}` (see "Running embeddings on a GPU") |
| | `verbose` | `false` | Verbose llama.cpp logging |
| `extraction` | `use_ocr` | `false` | OCR pages with no text layer. Needs the opt-in `rapidocr` package and the `pymupdf4llm` backend — see "OCR for scanned PDFs" |
| | `backends` | registry default | Per-file-type extractor choice, e.g. `pdf = "pymupdf-raw"`. PDFs: `pymupdf4llm` (default, Markdown structure via an ONNX layout model) or `pymupdf-raw` (text layer only, ~275× faster — see "Choosing a PDF extractor") |
| | `commands` | `{}` | Per-file-type argv for the `command` extractor, e.g. `pdf = ["pdftotext", "-layout", "{path}", "-"]`. Must print the document's text to stdout and nothing else (see "Extracting with an external command") |
| | `command_versions` | `{}` | Per-file-type argv that prints the tool's version, e.g. `pdf = ["pdftotext", "-v"]`. **Required** for every entry in `commands` |
| `source_watcher` | `ignore_directories` | 16 names incl. `.git`, `node_modules`, `build`, `dist`, `venv`, `target` | Directory names skipped anywhere under a watched root. **Replaces** the defaults rather than adding to them; set `[]` to index everything |
| | `state_path`, `log_file` | under the data dir | Watcher bookkeeping |
| `pipeline_worker` | `batch_size` | `32` | Chunks claimed from the database per embedding pass (1–128) |
| | `embed_submit_batch_size` | `4` | Chunks per embedding-server request **while a search is active** (1–128). llama-server schedules per input, strict FIFO, so this bounds how long a concurrent search query queues behind indexing (~1 s at 4 vs ~10 s at 32 on the measured iGPU). With no recent search the worker sends `batch_size` in one request instead: splitting it costs a fixed ~0.36 s per request and leaves the server's slots idle across each round trip, measured at ~2× throughput on the papers import |
| | `search_lease_ttl_seconds` | `10.0` | `search` records when it last ran; the worker pauses between embedding-server requests while a search happened this recently, so interactive queries do not queue behind bulk work. `0` disables |
| | `search_yield_cap_seconds` | `60.0` | Bound on continuous yielding: after this long the worker runs one request anyway, so nonstop searching slows a build instead of stalling it |
| | `poll_interval` | `1.0` | Seconds between polls when idle |
| | `state_path`, `log_file` | under the data dir | Worker bookkeeping |
| `bootstrap` | `auto_download_llama_model` | `true` | Fetch the model when missing |
| | `llama_model_url` | Nomic GGUF on Hugging Face | Where to fetch it from |
| | `llama_model_sha256` | pinned digest | Integrity check; set `""` to disable for a custom model |
| `storage` | `artifacts_path` | under the data dir | Where compressed extracted text lives |

##### Chunk size and the context window

`chunk_size` is counted with tiktoken; `n_ctx` is counted with the embedding
model's own tokenizer, and the two disagree — so `chunk_size` must stay well
under `n_ctx`. The shipped 320/512 pair has margin for the worst ratio measured
plus the task prefix. A chunk that would exceed the window is refused rather
than embedded truncated, and shows up as a failed chunk in `cementic status`.
Raising either value without re-measuring risks silently truncated embeddings;
the derivation is under "Measurements behind the defaults" below, and the
script is `scripts/measure_chunk_context_fit.py`.

### How search combines meaning and exact words

Search runs two arms over the same chunks and merges them.

- The **vector arm** finds text that *means* something similar to the query. It is
  what you want for `variational inference` or `how do transformers handle long
  context`.
- The **lexical arm** matches the words literally, through a PostgreSQL full-text
  index. It is what you want for a surname, an acronym, an equation label — the
  things a dense embedding cannot recover, because a single vector for a whole
  chunk does not record that the chunk literally contains the string `BLEU`.

Measured on a 23,064-paper corpus: for queries that are a rare exact token, vector
search alone put the right paper in the top 10 **7 times out of 150**. With the
lexical arm, every time.

**One thing to know before you judge it: a common word is not an identifier
query.** The lexical arm leads only when the query is a single word appearing in
at most `search.lexical_lead_max_documents` documents (default 20). `Hochreiter`
appears in 1,697 of 23,064 papers in one test corpus — it is the LSTM citation, so
it is everywhere — and a word in 1,697 papers does not identify one. That query is
led by the vector arm, and the top result may look unrelated. Author search works
best for authors your corpus cites *rarely*, which is the opposite of the famous
ones.

Turn the whole thing off with `hybrid = false` under `[search]`.

#### Why results sometimes show no score

When both arms contribute, the list is ordered by *rank fusion*, not by either
arm's score — and a cosine similarity and a text-rank score are not on the same
scale. Printing them side by side produces a column that goes up and down the page
and looks mis-sorted, so cementic prints the score only when it still explains the
order. Pass `--scores` to see the numbers anyway, or use `--json`, which always
carries `rank` alongside each result's own `score` and a `score_kind` naming which
arm produced it:

```bash
cementic search "Hochreiter" -c papers --json | jq '{rank, score, score_kind}'
```

### Choosing a PDF extractor

`[extraction.backends] pdf = "..."` picks how PDFs become text. Both produce the
input to the same chunker.

- **`pymupdf4llm`** (default) — Markdown with headings, tables, and
  header/footer stripping, using an ONNX page-layout model.
- **`pymupdf-raw`** — the PDF's text layer, nothing else.

Measured over 153 born-digital academic papers:

| | pymupdf4llm | pymupdf-raw |
|---|---|---|
| whole corpus | 25.9 min wall, ~4.3 core-hours | **5.6 s wall, 5.0 s CPU** |
| per document | 10.2 s | **37 ms** |
| worst single document | 42 s (a 4-page paper) | **0.54 s** (44 pages) |
| chunks produced | 2,588 (60-doc sample) | 2,792 — 7.9% more, mostly boilerplate the layout model strips |
| known-item retrieval, 38 title queries | R@1 0.895, R@5 0.921 | R@1 0.868, R@5 0.921 |

The retrieval difference is one document out of 38, which this sample cannot
distinguish from noise — it rules out a large quality gap, not a small one.
That is the reason `pymupdf4llm` remains the default despite costing ~3,000×
the CPU: the structure it produces is real, even though `chunk_text` slices a
fixed token window and reads none of it.

Prefer `pymupdf-raw` for bulk-importing a large born-digital corpus, where the
difference is hours against days. Keep `pymupdf4llm` for scanned or
table-heavy documents, which the measurement above does not cover.

Extraction cost scales with pages, not documents: use **2.8 ms/page** for
`pymupdf-raw` and **0.78 s/page** for `pymupdf4llm` when projecting. A per-document
rate taken from short papers understates a corpus of longer ones by the ratio of
their page counts.

### OCR for scanned PDFs

A PDF with no text layer extracts empty, and cementic fails the document rather
than storing nothing (`extracted no text: the PDF has no text layer`). OCR is
how you get text out of one, and it takes three things that must all agree:

```bash
# 1. install the OCR backend -- it is not a declared dependency
uv tool install --with 'rapidocr>=3.6.0' git+ssh://git@github.com/mnazaal/cementic
```

```toml
# 2. and 3. -- OCR only reaches extraction through the pymupdf4llm backend
[extraction]
backends = { pdf = "pymupdf4llm" }
use_ocr = true
```

`cementic doctor` reports all three together, including the silent no-op where
`use_ocr` is on but the pdf backend never calls OCR.

`rapidocr` is deliberately not installed by default: it pulls opencv and an
ONNX runtime, about 252 MB, for a path most corpora never take. cementic never
imports it — pymupdf4llm owns the adapter and cementic hands its OCR callback
over — so nothing breaks by its absence except OCR itself. Note that `uv tool
install` *replaces* the requirement set rather than merging into it, so re-list
any other `--with` pins in the same command.

Both `use_ocr` and the backend choice are part of the extraction fingerprint,
so turning OCR on re-versions the revision and re-extracts the collection.
Point scanned documents at their own collection rather than flipping the flag
under a corpus that does not need it.

The installed `pymupdf`, `pymupdf-layout` and `pymupdf4llm` versions are in that
fingerprint too, which is why `pyproject.toml` pins all three exactly rather
than by floor: a resolver picking a newer one at install time would re-extract
every corpus built by an older install. Bumping them is a deliberate commit that
costs a full rebuild. Pins hold only within one cementic version, so where a
checkout and a released snapshot both exist, `cementic doctor` compares this
install's extractor profile against each collection's active revision and warns
which ones indexing from here would rebuild.

Two alternatives need no Python package at all: run `ocrmypdf` over the file
first and index the result with any backend, or use PyMuPDF's own Tesseract
binding. Both want the `tesseract` system package instead.

### Extracting with an external command

Any file type can be extracted by a command cementic runs, instead of a built-in
extractor. The contract is the whole design: **argv in, text on stdout.**

```toml
[extraction.backends]
pdf = "command"

[extraction.commands]
pdf = ["pdftotext", "-layout", "{path}", "-"]

[extraction.command_versions]
pdf = ["pdftotext", "-v"]
```

This is the same shape as `llama_cpp.daemon_command`: the tool is yours,
cementic runs it and consumes its output. A tool that writes files rather than
printing text — `ocrmypdf`, say — needs a two-line wrapper script that prints;
that is the composition boundary, not something cementic tries to absorb.

It is also how to index a file type cementic has no extractor for. Naming a
type in `[extraction.commands]` adds it to the watcher's set, so `epub = [...]`
makes `.epub` indexable. The command extractor is never a fallback — it only
ever runs where `[extraction.backends]` names it.

**`command_versions` is required, and cannot be guessed.** Its output goes into
the extraction fingerprint, so upgrading the tool re-versions revisions instead
of silently rewriting extracted text under them — the same protection a
`pymupdf` bump already gets. It has to be configured because version flags
disagree: `pdftotext -v` works while `pdftotext --version` reads the flag as a
filename, prints an I/O error, and *exits 0*. A guessed flag would record that
constant error as the version and never notice an upgrade.

Nothing can detect a wrong flag automatically, so `cementic doctor` prints what
each tool reported. If it says `I/O Error` rather than a version, fix the flag.

A failing command is a failed document, not a silent empty one: a non-zero exit
is reported with the tool's own stderr, and empty output is refused the same way
it is for every other extractor.

### Running embeddings on a GPU

cementic identifies its embedding server only by the model id reported at
`/v1/models`, and it spawns that server from `llama_cpp.daemon_command` —
by default `llama-server` on PATH. Nothing about GPU support lives in cementic:
point the command at a Vulkan/CUDA build (the template config ships a Vulkan
example) and that is the build that serves. `cementic embedding
start/stop/status`, on-demand autostart, and crashed-daemon recovery apply to
whatever the command names, and the `{alias}` placeholder keeps the served
fingerprint in lockstep with the config so a `n_ctx`/`n_gpu_layers` change can
never leave a stale alias behind.

Recovering a running daemon from `/proc` matches both `{alias}` and `{port}` in
its argv, so a command that never names its port cannot be re-adopted after the
pid file is lost. Every documented invocation carries both.

Alternatively, run any OpenAI-compatible server entirely outside cementic with
`llama_cpp.daemon_autostart = false` and `--alias <fingerprint>`; print the
fingerprint with `llama_cpp_runtime_fingerprint` (see `embedding_runtime.py`).

Measured with upstream `llama-bench`, pp512, nomic-embed-text-v2-moe Q8_0, on an
Intel Core Ultra 5 125U with its integrated Arc GPU via Vulkan:

| arm | tok/s | vs pure CPU |
|---|---|---|
| `-dev none` (pure CPU) | 382 | 1.0× |
| `-ngl 0` | 2,112 | 5.5× |
| `-ngl 99` (full offload) | 2,468 | 6.5× |

**`-ngl 0` is not a CPU baseline.** llama.cpp offloads large matmuls to any
visible GPU backend by default (`--no-op-offload` defaults to `0`), so with a GPU
present `ngl 0` is already GPU-assisted. Only `-dev none` measures the CPU.
Reading `ngl 0` as the baseline made a 6.5× speedup look like 1.17×.

Expect less end to end: through cementic's HTTP path the CPU arm reached ~281
tok/s against a raw 382, so derate by roughly a quarter.

`n_gpu_layers` is part of both the runtime fingerprint and the embedding
profile, so changing it re-versions the corpus. Set it before a bulk import,
not after.

### Detecting an embedding server that changed under you

`n_gpu_layers` is in the fingerprint, but the llama.cpp *build* is not — it is
knowable only by asking a running server, and putting it in the profile would
re-version the whole corpus on every upgrade. So cementic detects instead.

Each embedding profile stores a few reference texts with the vectors one server
produced for them, taken once on the first indexing run that uses the profile.
`cementic doctor` replays that exact request against the running server and
compares by cosine. It warns only when the server computes *different*
embeddings from the indexed ones — the collection then needs rebuilding, or the
previous server restoring.

The comparison cannot be bit-for-bit, and the reason is worth knowing before
you tune anything: llama-server packs concurrent slot work into unified
batches, so the same request returns slightly different vectors depending on
what the server handled just before it. Measured here, replaying a canary after
an unrelated 3-text request moved it to cosine 0.999908; request composition
moves it up to 2.3e-3 (a text embedded alone versus as the last of 32). Bit
equality would report that scheduling as corruption.

What noise cannot do is cross the gap to a changed function. A tokenizer,
pooling or normalisation change lands two orders of magnitude below any of the
above, which is where the threshold sits. Upgrades themselves are usually
harmless: b10605 against b10818 — 213 builds apart — returned bitwise identical
vectors for 200 real chunks on both CPU and Vulkan.

Only a server serving that profile's own model is asked, so a deliberate model
swap and collections left on a legacy profile are skipped rather than reported
as corruption. To get these numbers for your own hardware, run two builds in
turn on a spare port with identical flags and embed the same sample through
each — and keep the traffic sequence identical between the two, or you measure
the scheduling noise above instead of the build difference.

### Running indexing as a background service

`cementic start` spawns the watcher and worker as detached children of your
shell, which is fine for a short build and wrong for one that spans days: a
suspend that kills the embedding server leaves the worker waiting for it
indefinitely, and nothing restarts either after a reboot.

`packaging/systemd/` ships user units for that case — the embedding server, a
per-collection watcher and worker, and a timer that restarts the server if
`/v1/embeddings` stops answering. Installation is in each unit's header.
Restarting is safe at any point: the worker re-queues anything left
`processing` at startup, every batch is committed, and the vector upsert is
idempotent, so an interrupted run loses at most one batch.

Two things to know when running this way. Use `cementic start` **or** the units,
never both — one worker per collection holds an advisory lock and the second
will simply idle. And `cementic status` reports `workers stopped` under systemd,
because that line reads state written only by `cementic start`; the progress
counters beside it are still accurate, and
`systemctl --user is-active cementic-worker@<collection>` is the real check.

### Choosing an ANN index (HNSW vs DiskANN)

`index.method` selects how vectors are indexed for similarity search:

- **`hnsw`** (default, pgvector) — graph index held in memory. Lowest query
  latency; wants enough RAM to hold the index.
- **`diskann`** (pgvectorscale) — disk-resident, compressed. Much lower RAM use
  at scale, for some added latency.

At a few-million-vector scale HNSW usually wins latency and DiskANN wins memory.
Both ship in the generated Postgres image.

The method is a *serving* choice: it applies when a collection's vector index is
built and never re-embeds. To change it on a collection that is already built:

```bash
cementic collection reindex research
```

That swaps the index in place without touching the embeddings. `hnsw_m` and
`hnsw_ef_construction` are fixed into the index when it is created, so changing
those needs `cementic collection reindex research --force`. Either way it can
take several minutes on a large corpus, and `cementic status` reports it under
`activity:`.

`index.hnsw_iterative_scan` (`relaxed_order` by default) keeps the index scan
going until it has a full page of results, rather than stopping after
`hnsw_ef_search` candidates. Without it a collection holding a small share of a
shared vector table can come back short, or empty. It needs pgvector 0.8+ and is
ignored on older servers.

### Environment variables

Every setting also has a `CEMENTIC_*` environment variable, which overrides the
config file (handy for one-off overrides and CI):

```bash
# Database (defaults shown)
export CEMENTIC_DB_HOST=localhost
export CEMENTIC_DB_PORT=5432
export CEMENTIC_DB_NAME=cementic
export CEMENTIC_DB_USER=cementic
export CEMENTIC_DB_PASSWORD=cementic
# Or point at an existing Postgres with a full URL (wins over the parts above):
# export CEMENTIC_DB_URL=postgresql://user:pass@host:5432/dbname

# Extraction (per-file-type extractor choice lives in the
# [extraction.backends] config table; this toggles OCR, which also needs the
# opt-in rapidocr package and the pymupdf4llm backend)
export CEMENTIC_EXTRACT_USE_OCR=false

# Artifact storage
export CEMENTIC_STORAGE_ARTIFACTS_PATH=~/.local/share/cementic/artifacts

# Pipeline chunking
export CEMENTIC_PIPELINE_CHUNK_SIZE=320
export CEMENTIC_PIPELINE_CHUNK_OVERLAP=80

# Embedding provider selection (default)
export CEMENTIC_PIPELINE_EMBEDDING_PROVIDER=llama-cpp

# llama.cpp defaults
export CEMENTIC_LLAMA_MODEL_PATH=models/nomic-embed-text-v2-moe.Q8_0.gguf
export CEMENTIC_LLAMA_DAEMON_AUTOSTART=true

# Background worker
export CEMENTIC_PIPELINE_WORKER_BATCH_SIZE=32
export CEMENTIC_PIPELINE_WORKER_POLL_INTERVAL=1.0

# Bootstrap (model download only; cementic does not manage containers)
export CEMENTIC_BOOTSTRAP_AUTO_DOWNLOAD_LLAMA_MODEL=true
```

The default local setup is:

- Postgres with `pgvector` + `vectorscale`, provisioned via the generated setup or your own Postgres
- a shared persistent local `llama.cpp` server for indexing and interactive search, so the model stays loaded once

When the configured model is a member of the `nomic-embed-text` family (v1,
v1.5, or v2 — matched on the model filename), cementic automatically applies
task prefixes:

- document embeddings: `search_document: ...`
- query embeddings: `search_query: ...`

## Known limitations

- **One background session at a time**, by design — see the note under Usage.
  Multiple directories can share one session; multiple collections cannot run
  concurrently.
- **Moving the watched directory itself is invisible until restart.** The
  filesystem watch delivers no event when the watched root is renamed or moved
  away, so cementic keeps watching the old location. The next
  `cementic start` reconciles: a document whose file is gone drops out of
  search, and one still reachable under a new real path — a tree moved with a
  symlink left at the old location, say — is repathed rather than indexed
  again, keeping its extraction, chunks and vectors. (Moving or deleting
  *subdirectories* inside the watched tree is handled live.)
- **Model identity is matched by filename.** The task-prefix policy for Nomic
  models is selected from the model file's name; renaming the GGUF (or
  mirroring it under another name) silently switches to plain, unprefixed
  embedding. Keep the upstream filename. `cementic embedding start` reports
  which text policy was selected.
- **Filtered search recall is approximate.** Collections sharing an embedding
  profile share one ANN index, and a search restricted to one collection
  filters candidates during the index scan; a collection holding a very small
  share of a large shared table can return slightly fewer results than exist.
  Dedicated per-collection profiles avoid this entirely.
- **A daemon wedged mid-batch reads as busy until the batch clears.** The
  health probe treats an unanswered embedding as legitimate load while a live
  worker is mid-batch; a daemon that wedges at exactly that moment is reported
  `busy` until the worker's claim times out, then `wedged`.
- **Searching collections indexed by different embedding models refuses,
  rather than merging results.** Distances from different models are not
  comparable, so `cementic search "q" -c a b` errors out and names which
  collection uses which model if `a` and `b` were built with different
  models. Search each separately instead.

## Development

Everything below is for working *on* cementic rather than with it. It is the
single source of truth for contributors and coding agents alike — `AGENTS.md`
used to hold a second copy and drifted from this one.

### Commands

```bash
uv pip install -e ".[dev]"

./scripts/check.sh          # the gate to trust — all six CI gates
pytest tests/unit           # fast inner loop
pytest tests/unit/test_extractors.py::test_extract_document_reads_plain_text -v
pytest --cov=cementic       # branch coverage is on by default
ruff check --fix src/ tests/
ruff format src/ tests/
mypy src/
```

`./scripts/check.sh` runs lockfile (`uv lock --check`), ruff, mypy, unit,
integration, and integration-pg, and reports a missing PostgreSQL as SKIPPED
rather than passed. `pytest -v && ruff check && mypy` looks like "all checks"
but silently skips the PG-marked integration tests — precisely the gap that once
let three PG tests reach `main` red. Use `check.sh` before pushing.

The PG gate needs a Postgres with pgvector + vectorscale. The integration suite
brings `compose.yml` up itself when a container engine is available, and leaves
an already-running one alone.

### Conventions

Style is enforced mechanically — ruff (100 columns, import order, naming) and
mypy `strict` — so only what the tools cannot check is written down here:

- **Docstrings** are Google-style and short; type information lives in
  annotations, not prose. Comments explain *why*, and several record a specific
  past defect — those are load-bearing, not clutter.
- **Errors.** Raise specific exceptions and let them bubble where a caller can
  act. Human-facing error text goes to stderr via `err_console`; stdout carries
  only the command's output, so `--json` stays parseable under `jq` even when
  something fails. Every failure path exits non-zero.
- **Database.** SQLAlchemy 2.0 ORM with `Mapped[]`, `select()` over raw SQL,
  relationships with `back_populates`, indexes in `__table_args__`. Raw `text()`
  is confined to what the ORM cannot model: per-profile vector DDL and KNN in
  `vector_store.py`, ANN index DDL, full-text index DDL and extension setup in
  `db.py`, the KNN/lexical execution and the document-frequency probe in
  `search.py`, advisory locks in `pipeline_worker.py`, the per-collection vector
  delete in `revisions.py`, and the probes in `doctor.py` / `status_service.py`.
  Values are always bound; the only interpolated fragments are identifiers
  computed from an int profile id, the `LEXICAL_TEXT_CONFIG` module constant,
  and settings a config validator has already closed (`index.method`,
  `build_memory`).
- **Config.** Pydantic `BaseSettings`, one env prefix per section
  (`CEMENTIC_DB_`, …), sensible defaults, every field documented.
- **CLI.** Help text and error messages are part of the product. A namespace
  invoked without its subcommand prints focused help rather than a parser error.
  Infrastructure failures name a next step. Commands taking a collection offer
  both `--collection` and `-c`. Renaming a command means updating root help,
  namespace help, and the partial-invocation tests.
- **Tests.** `test_*.py` / `test_*`, fixtures from `conftest.py`, external
  dependencies mocked. Prefer structural assertions over wall-clock budgets —
  timing assertions are load-sensitive and one of them used to fail under
  coverage instrumentation.
- **Red-verify every regression test by breaking the fix.** A test that enters
  below the real entry point can pass against the broken code it was written
  for: one regression test called the fingerprint payload builder directly and
  was green before the fix existed. Enter through the same path a user takes.
- **Green gates are not sufficient evidence for worker, daemon or profile
  changes.** One batch passed all six gates and still shipped three defects,
  every one of them found by running the CLI against a real corpus — including
  search broken after promote, and unit tests writing into
  `~/.local/share/cementic/`. End such changes with a live run.
- **Changing a shared blob means checking every reader.** Making the embedding
  fingerprint path-independent silently made the daemon launch spec unusable,
  because `config_json` was serving as both. Enumerate the callers first.

### Daemon architecture

The source watcher registers documents of any type the extractor registry
handles. The pipeline worker builds extracted text, chunks, and embeddings for a
revision. The embedding runtime is a warm local llama.cpp server shared by
indexing and search. Both workers use SQLAlchemy sessions as context managers
and publish state — PIDs, health, `last error` — through JSON files in
`state.py`.

### Module map

```
src/cementic/
├── cli.py                # Typer CLI: commands + the extract|chunk|embed filters
├── cli_collection.py     # `cementic collection` subcommands
├── cli_format.py         # Click/Typer subclasses for plain, uppercase help
├── cli_shared.py         # helpers shared by the CLI modules
├── render.py             # status/doctor/collection output rendering
├── source_watcher.py     # content-type-driven document watcher
├── extract.py            # extractor registry (content type -> Markdown/text)
├── chunk.py              # token-based text chunking
├── embedding_runtime.py  # embedding provider registry + warm llama.cpp client
├── embedding_provider.py # abstract base class for embedding providers
├── embedding_text.py     # per-model query/document prompt formatting
├── pipeline_worker.py    # builds extract/chunk/embed artifacts for a revision
├── profiles.py           # immutable extractor/chunk/embedding profiles
├── revisions.py          # target/building/ready/active/retired lifecycle
├── index_strategies.py   # ANN index registry (hnsw/diskann) -> DDL
├── vector_store.py       # per-profile vector tables + KNN SQL
├── search.py             # semantic search over the active revision
├── hybrid.py             # lexical + vector rank fusion (pure)
├── collections.py        # collection delete + revision promote/history
├── config.py             # Pydantic settings (TOML + env + flags)
├── db.py                 # SQLAlchemy models + engine/session
├── storage.py            # compressed artifact storage
├── hashing.py            # shared file/content hashing
├── model_digest.py       # cached content digest of the embedding model file
├── status_service.py     # worker / health / pipeline status
├── supervisor.py         # background process management
├── runner.py             # internal background runner for both workers
├── worker_runtime.py     # shared logging/shutdown plumbing for both workers
├── state.py              # daemon state files
├── bootstrap.py          # DB-reachable check + llama model download
├── doctor.py             # read-only runtime diagnostics
├── canary.py             # reference vectors that catch a changed embedding server
├── filelock.py           # advisory file locks (start, daemon autostart)
├── validation.py         # collection-name validation
└── templates/            # files `cementic init postgres` and `config init` copy out

tests/
├── unit/                 # mocked / SQLite
├── integration/          # PostgreSQL + smoke tests
└── fixtures/             # generated PDF fixtures
```

### Decisions worth knowing

Choices that are not obvious from the code, and that someone is otherwise
likely to reverse by accident.

- **`notes/` is gitignored.** Code-review notes are working artifacts with one
  reader; they stay on disk and in history up to `c7069e1`, and new ones need no
  commit. `git show <rev>:notes/<file>` retrieves an old one.
- **There is no backwards-compatibility obligation.** cementic has one user, so
  a breaking CLI change is made outright rather than behind a deprecation alias
  — `status --doctor` became `cementic doctor` with no alias kept. Weigh CLI
  changes on whether they are right, not on who might be scripting them.
- **Python 3.12+ is required, deliberately narrowed from 3.10.** The older
  floor was an untested claim, and `config.py` carried a `tomli` fallback for it
  that no tested interpreter ever executed. Deleting the branch, the dependency
  and the claim beat adding CI jobs to exercise code nobody runs. Reopen only if
  cementic must run somewhere that cannot get 3.12.

Rationale for *code* decisions lives against the code instead — see
`_reporting_db_errors`'s docstring, `EXTRACTION_VERSION` in `profiles.py`, and
the deferred imports in `config.py`. A comment beside the thing it explains
cannot drift from it; a separate register can.

### Measurements behind the defaults

These are the evidence for values that are live today. Re-measure before
changing any of them.

**Chunk size against the context window** (`pipeline.chunk_size = 320`,
`llama_cpp.n_ctx = 512`). `chunk_size` counts tiktoken tokens; `n_ctx` counts the
model's own. For the default model one tiktoken token is a median of 1.14 model
tokens, p95 1.24, up to 1.33 on English and source code. The runtime guard
assumes an upper bound of 1.45, and `(320 + 8) × 1.45 = 475.6` fits inside 512 —
the `+ 8` being the task-prefix allowance. A chunk that would exceed the window
is refused, not truncated. Re-measure with
`scripts/measure_chunk_context_fit.py`.

**ANN index build memory** (`index.build_memory = 2GB`). 100k × 768 is 293 MiB of
graph against PostgreSQL's 64MB default, so the build spills to disk: **1454 s at
64MB against 345 s at 2GB**.

**Why vector rows carry their own filter columns.** Measured with
`EXPLAIN (ANALYZE)` on a real corpus, back when every filter lived on a joined
table:

| query | plan | time |
|---|---|---|
| bare KNN, 768-dim, 20k rows | `Index Scan using ...ann` | 2–6 ms |
| the search query, same data | top-N heapsort over a full nested loop | 25 ms |
| the search query, 8-dim, 60k rows | same, 60k per-row PK lookups | 83–90 ms |

The planner drove from `chunked_documents` and probed the vector table by
primary key, so the ANN index was never used. Search was exact but scaled
linearly with the table.

Moving those filters onto `embedding_vectors_p*` made the index reachable, and
is also the measurement behind `index.hnsw_iterative_scan`. Same data, same
index, 100k rows at 768 dimensions:

| query shape | ANN used | results | time |
|---|---|---|---|
| filters on joined tables (before) | no | 10/10 | 407.6 ms |
| filters on the vector row | yes | 10/10 | 1.0 ms |
| filters on the vector row, 2% slice, `iterative_scan=off` | yes | **0/10** | 1.3 ms |
| filters on the vector row, 2% slice, `relaxed_order` | yes | 10/10 | 15.3 ms |

The 0/10 row is why `relaxed_order` is the default: once the planner really
drives from the ANN index, a collection holding a thin slice of a shared vector
table post-filters its way to an empty result. Reproducing it needs ~100k rows,
which is why `tests/integration/test_search_ann_pg.py` asserts the planner's
choice rather than this recall failure.

**Indexing throughput on a real corpus** (2026-08-24, 153 papers / 178 MB into
one collection, 14-core CPU, no GPU). This is the first run at a size worth
projecting from; the per-stage numbers replace earlier estimates taken from
5-document samples.

| stage | wall clock | rate |
|---|---|---|
| extract (pymupdf4llm, 153 PDFs) | 25.9 min | 10.2 s/doc, ~11 of 14 cores busy |
| chunk (7,306 chunks) | < 1 min | 47.8 chunks/paper |
| embed (7,306 chunks) | 1.65 h | 814 ms/chunk |
| **total to 100%** | **2.08 h** | zero extract or embed failures |

Peak resident memory: pipeline worker 3.5 GB, llama.cpp daemon 1.1 GB, watcher
75 MB.

**The stages do not overlap.** All extraction finishes before the first chunk is
written, and all chunking before the first embedding, so wall clock is the sum
of the stages rather than the longest one. Extraction is therefore its own
budget line, not time hidden under embedding.

**Scale target, measured against the real corpus** (22,246 PDFs, 43.8 GiB,
592,248 pages — median 20 pages, p99 178, max 1,083). At 3.65 chunks/page that
is **~2.16M vectors**. Projected from the rates above, and additive because the
stages are staged:

| | pymupdf4llm + CPU | pymupdf-raw + iGPU |
|---|---|---|
| extraction | 128 h | **0.5 h** |
| embedding | ~490 h | **~108 h** |
| **total** | **~26 days** | **~4.5 days** |

The embedding figure is measured end to end, not projected: a 200-document run
through the real pipeline — GPU server, HTTP, 32-chunk batches, index-driven
claims — sustained **5.56 chunk/s**, which is 108 h for 2.16M chunks. The
earlier 4–7 day range came from 4.27 chunk/s on the pre-denormalisation claim
path.

Two constraints bind at this size and neither is query latency. **Memory:** an
HNSW index over 2.16M × 768 vectors is ~6.5 GB resident, against 15 GB of RAM
with ~5 GB free — which is what the `diskann` seam exists for. Prefer HNSW for
the initial build anyway: it is maintained per insert and therefore resumable
across a multi-day job, where DiskANN builds at the ready transition in one
unresumable pass. `cementic collection reindex` switches method afterwards
without re-embedding, so the decision is cheap to revisit. **Disk:** the
database lands near 30 GB.

**Search latency** (warm daemon, 7,580 vectors, `hnsw.ef_search = 40`):

| step | time |
|---|---|
| embed the query string | 35.3 ms |
| pgvector kNN, top-10 | 2.6 ms (0.3 ms in-engine) |
| **`Searcher.search()` total** | **38.7 ms** |
| CLI end to end | ~650 ms — the rest is interpreter and import startup |
| first query after a cold start | 6.2 s, loading the model |

Search is embedding-bound, not index-bound. An earlier reading of this section
put the embedding step at 197 ms; re-measured on the corpus above it is 35 ms,
so treat the old figure as superseded rather than reconciled.

**ANN recall against exact search** (same corpus, 20 queries, top-10): mean
0.990, worst 0.900, 18 of 20 identical to the exact ranking. The ANN query is
8.9× faster than the exact one (2.7 ms against 24.1 ms), and the HNSW index is
29 MB over 7,580 × 768.

**Promotion does no index build.** Because the HNSW index is created up front
and maintained per insert, `cementic collection promote` on the finished
153-paper revision took 0.7 s — the payoff the build-order measurement above
predicted, now observed on a real corpus.

**Claiming embedding work at scale.** The embed step used to find its next
batch with an anti-join — chunks with no row for this profile — which cannot be
indexed. Cost then tracked corpus size rather than remaining work, because the
scan walks the finished prefix to reach the unfinished tail. Measured on a
synthetic 300,000-chunk collection with `scripts/measure_embedding_claim_scale.py`:

| % embedded | joined claim | single-table claim |
|---|---|---|
| 10% | 151,392 buffers | **17** |
| 50% | 903,930 buffers | **806** |
| 90% | 1,895,124 buffers | **1,997** |

`chunk_embeddings` carries `collection`, `extractor_profile_id` and
`chunk_profile_id` for the same reason the vector rows carry their filters: on a
joined table the planner drives from the collection and checks embedding status
last. The driving scan now reads exactly `batch_size` rows at any completion
level. Across a 22k-document import that is roughly 20 minutes of claim time
rather than ~170 hours.

Two things this depends on, both easy to lose and both pinned by a test: no
`ORDER BY` (it makes PostgreSQL read every pending row and top-N sort before
honouring the `LIMIT`), and no scope filter on a joined table.

**Directory moves under the watcher**, measured against watchdog's inotify
backend:

| case | events delivered | handled |
|---|---|---|
| rename inside the watched tree | `DirMovedEvent` + per-file events | yes |
| move a directory in | `DirCreatedEvent` + per-file events | yes |
| move a directory out | one `DirDeletedEvent`, no per-file events | yes, via prefix delete |
| move the watched root itself | nothing at all | no — see Known limitations |
