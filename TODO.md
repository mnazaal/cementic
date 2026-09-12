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
- Enrich search results with document id and optional artifact path. A result
  today carries seven fields: collection, source_path, content, score, distance,
  score_kind and rank. Specified as step 6 of PLAN.md's "Execution order — CLI
  surface audit", including why `chunk_id` is deliberately not among them.

- Pre-filter over-budget chunks against the model's own tokenizer, not just
  the cheap tiktoken estimate. `count_model_tokens` learned upstream
  `llama-server`'s `/tokenize` in `1ff5269`, so the *exact* check works, while
  `embed_batch`'s pre-filter still uses the estimate: a chunk at 513–549 model
  tokens is sent anyway, fails with a 500, and drags its whole request into the
  isolate-and-retry path. **Not urgent, measured 2026-09-06:** `chunk_embeddings`
  is 2,298,558 rows, all `done`, none failed or pending — `c99d84b` splits
  over-budget chunks instead of dropping them, so they already succeed. What
  remains is a wasted round trip during indexing only, which a settled corpus
  never pays. *Do it when:* a bulk import or rebuild is next on the cards.

- ~~**Find out why `cementic status` blocks for tens of seconds.**~~ **Done
  2026-09-07.** One query, not a timeout: the embedding counts joined
  `chunk_embeddings` through `chunks_v2` to reach three columns
  `chunk_embeddings` already carries, so every run scanned 2.6 GB of chunk text
  and spilled the hash join to disk. 7.96 s of a 9.08 s run; the 0-2% CPU was
  the CLI blocked on a socket while Postgres worked. Reading the denormalised
  columns instead (`embedding_scope_denormalised`) takes it to 1.93-2.20 s end
  to end warm. The `total_chunks` count was rewritten the same day to count
  from `ix_chunks_v2_chunked_document_chunk` rather than the heap: 90,515
  buffers against 338,793, though its wall-clock effect is inside the noise
  (0.6-2.3 s for both forms) and 342,820 heap fetches say part of it waits on a
  vacuum. `compute_revision_counts` moved onto the same predicate, so the
  worker and status no longer describe one set two ways.

  **A cold cache still costs.** Five warm runs land at 1.97-2.27 s, but a run
  straight after the pg suite churned the page cache took 10.1 s. The
  reproducible win is the embedding count (8.0 s to 0.2 s for that statement);
  the cold tail is smaller than it was and not gone.

  **Reopened and closed 2026-09-10: it was stale planner statistics, and a
  `VACUUM (ANALYZE)` fixed it.** Measured before and after, idle machine:

  | | before | after |
  | --- | --- | --- |
  | `load_pipeline_status_bulk` | 2.8-2.9 s | 0.62 s |
  | `check_health` | 449 ms | 142 ms |
  | `cementic status` end to end | ~4.1-5.5 s | 1.59-1.81 s |

  **The operational lesson, which is the part worth keeping.** A bulk delete on
  this corpus does not get an autoanalyze on the big tables. `chunks_v2` and
  `chunk_embeddings` were still carrying statistics from 2026-08-24 -- before
  the corpus doubled and before the 2026-09-09 repair deleted ~23k documents
  worth of rows -- while autovacuum had run on them as recently as 09-09. So
  **any future corpus repair should end with `VACUUM (ANALYZE)` on the tables it
  touched**, not just leave it to autovacuum.

  **Do not vacuum `embedding_vectors_p6` as part of that.** It was included in
  the first attempt and had to be cancelled after 78 minutes without finishing.
  Its heap is only 171 MB (the vectors are TOASTed, 19 GB total), but it carries
  a **9,150 MB HNSW index**, and vacuuming an HNSW graph means traversing it to
  repair links for every dead tuple. It also did not need it: autoanalyze had
  covered it on 09-09, and its 323,392 dead tuples affect search, which measures
  1-2 s and has no reported problem. Cancelling a `VACUUM` is safe and leaves
  the table consistent -- `pg_cancel_backend` on the leader pid.

  The five tables `status` reads (`chunks_v2`, `chunk_embeddings`,
  `chunked_documents`, `extracted_documents`, `source_documents`) all vacuum in
  under 2.5 minutes combined, `chunks_v2` being 134 s of that. Reusable script:
  `~/.cache/cementic-vacuum.py`, which records before/after `pg_stat_user_tables`
  rows so the effect is provable rather than assumed.

  **What remains is startup, not the database.** At 1.6 s, `status` now spends
  roughly 0.4-0.5 s importing `cementic.cli` before it touches anything, and
  0.62 s on queries. That floor is PLAN.md's "Execution order -- CLI surface
  audit" step 2, and it caps how fast any command can be.

- ~~**Two papers were silently unsearchable.**~~ **Fixed 2026-09-07**, found
  while chasing a 103-chunk discrepancy in `chunked_documents.total_chunks`.
  When re-extraction fails, `_purge_all_chunks` drops the document's chunks but
  left the chunking `done` with its old `source_content_hash`; a file that then
  extracted to the *same* text matched its own stale hash, so `_step_chunk`
  never re-claimed it. Zero chunks, zero embeddings, counted as 100% chunked,
  reported by nothing. The purge now clears the hash, which re-opens the work
  and keeps the row out of `chunked_done` until it is genuinely re-chunked.
  Reproduced first as a failing test
  (`test_a_document_is_rechunked_after_its_failed_re_extraction`).

  **Nothing is stranded any more, so the repair this entry carried is gone.**
  Checked 2026-09-12: `cementic doctor` reports `stranded_chunkings: ok — every
  chunking counted as done owns chunks`. The 3 rows outstanding on 2026-09-09
  went with that day's cleanup. The repair was a single self-finding `UPDATE`
  that cleared `source_content_hash` on every `done` chunking owning no chunks;
  it is in this file's history if the state ever recurs, and `doctor` is what
  detects it. Deleted rather than left standing because it was dangerous to run
  unread — on 2026-09-08 the same selector would have matched 18,103 rows and
  queued 1.8M chunks of re-embedding, most of them duplicates.

- ~~**The corpus doubled, and 18,101 papers went unsearchable.**~~ **Fixed
  2026-09-09** on `claude/watcher-document-identity`; the live corpus was
  repaired the same day. `~/data/Documents/Papers` became a symlink to
  `~/sync/Material/Papers`, so the watcher resolved every file to a path it had
  never seen and indexed the whole corpus a second time -- and the watcher's
  chunk purge left each chunking `done` with a current hash and no chunks, the
  same class of bug as the entry above through a different purge. Root causes,
  the design that replaced path-and-hash identity, and what was rejected on the
  way are in PLAN.md, "Decided -- what makes two rows the same document".
  `cementic doctor` now reports the lost-chunk state rather than leaving it to
  be noticed.

- ~~**Audit the CLI surface against `llm`'s embeddings commands.**~~ **Done
  2026-09-10.** Raised 2026-09-07 after reading that tool
  (https://llm.datasette.io/en/stable/embeddings/cli.html): it is JSON-first,
  pipeable, has a small obvious surface, and needs no daemon, no PostgreSQL and
  no systemd. The audit's decisions and its six-step execution order are in
  PLAN.md, "Execution order — CLI surface audit"; the evidence, the `llm`
  comparison and the per-command usage matrix are in
  `notes/design-cli-surface.html`. Headline: 21 leaf commands against `llm`'s 9,
  nothing dead, and the real defects are elsewhere than the surface — a false
  README claim about `search --json`, a container script calling a flag removed
  months ago, 1.2 s of eager imports on every invocation, and a composability
  principle the design will never support. Nothing was cut: the weakest
  commands are thin wrappers whose deletion saves nothing.

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

Both gaps the 2026-08-14 review recorded here are closed: the 3.14 sqlite
engine leak (autouse dispose fixture, `tests/unit/conftest.py`) and the
missing pipeline-failure-to-status connection
(`test_status_verbose_reports_a_real_extraction_and_embedding_failure`).

## Later

- ~~Hybrid lexical + vector search for exact author names, acronyms, citations
  and equation labels.~~ **Shipped; the thread closed 2026-09-10.** The fusion
  rule is *the leading arm owns rank 1, RRF owns the rest* — vector-only
  recall@10 is 0.047 on rare exact tokens against hybrid's 1.000. The one open
  risk is that the precondition was never measured: nobody has checked whether
  real queries are identifier-shaped. Measurements and that risk are in PLAN.md,
  "Decided — hybrid lexical + vector retrieval".
- A multi-profile embedding daemon pool if old-model search and new-model
  indexing need to run concurrently. *Trigger receded:* the migration that
  would have put both on one port was rejected (`PLAN.md`), so nothing needs
  this today. It becomes real again the moment a model swap is back on.
- Evaluate lighter embedding providers if llama.cpp memory use is too high on
  small machines.
- **Build llama-cpp-python with Vulkan** so the iGPU works under
  `daemon_autostart` instead of an externally-run `llama-server`. Less pressing
  than it looked: the external server needs no code change, is documented in
  README ("Running embeddings on a GPU"), and is now supervised by the systemd
  units, so the practical gap this would close is small. Blocked on `glslc`,
  which is neither packaged here nor on PyPI — it needs shaderc from source.
- ~~**Cut indexing wall clock at corpus scale.**~~ **Done 2026-08-24.** 26 days
  → 4.5, by three changes: the `pymupdf-raw` extractor (128 h → 0.5 h), the
  iGPU via an external llama-server (~490 h → ~108 h measured end to end), and
  denormalising the embedding claim so it stays flat instead of degrading with
  the completed fraction. Numbers in README's "Measurements behind the
  defaults". Three levers that did **not** pay, so nobody re-tries them: thread
  pinning (this machine's run-to-run spread reached 53%, which swamps any
  thread-count effect), `batch_size = 128` (3.12 chunk/s against 32's 4.27,
  though one sample each cannot separate that from noise — the point is there
  was no gain), and overlapping the pipeline stages (worth ~0.5 h once
  extraction is raw text, against a worker rearchitecture).

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
- **Stop the extractor registry re-versioning corpora that cannot be affected.**
  `extractor_registry_payload` (`extract.py:518`) hashes every built-in
  extractor, so adding a `.docx` entry re-versions an all-PDF corpus and forces
  a full rebuild — 2.3M vectors, ~108 hours. The gating that already excludes
  config-driven extractors cannot simply be widened: narrowing the payload
  moves the fingerprint by itself, so the fix costs exactly the rebuild it
  prevents. *Reopen when:* you want a non-PDF extractor, or anything else
  forces a rebuild — on that day the narrowing rides along for free.
- **Publish the repo.** Nothing gates on it today; the install docs no longer
  describe a command that cannot work. Publishing only changes whether a
  `pipx install git+https://...` one-liner works for someone who is not the
  author. *Reopen when:* you want to hand cementic to another person — at that
  point README's install section goes back to the one-liner in the same commit.
- **Restore support for Python older than 3.12.** *Reopen when:* cementic needs
  to run somewhere that cannot get 3.12. Unlikely while `uv` installs one in a
  single command, and the compatibility branch was deleted precisely because
  nothing exercised it.
