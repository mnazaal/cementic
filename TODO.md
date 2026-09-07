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

- Find out why `cementic status` blocks for tens of seconds. Measured
  2026-09-06 on the `papers` collection: 12 s, 18 s, 44 s, 50 s, 60 s, 83 s and
  128 s across seven runs, at 0–2% CPU — it is waiting on something, not
  computing. Reproduces on the pre-change uv-tool install too, so it is not
  from this pass, and `check_health` timed at 0.15 s when called directly, so
  the two-minute daemon probe is not obviously the answer either. PLAN.md
  records this as fixed ("`status` no longer blocks 120 s to return an answer
  the pid file already had"), which the measurements contradict. Worth a
  root-cause pass before anything else in this file: it is the command run most.

- Audit the CLI surface against `llm`'s embeddings commands
  (https://llm.datasette.io/en/stable/embeddings/cli.html), and cut what does not
  earn its place. Raised 2026-09-07 after reading that tool: it is JSON-first,
  pipeable, has a small obvious surface, and needs no daemon, no PostgreSQL and
  no systemd — which is the shape PLAN.md's own "Unix composability" principle
  asks for and cementic's front door does not have. It could not replace
  cementic (verified from its source: `similar_by_vector` registers a Python
  UDF and linear-scans every row, so 2.3M chunks would take minutes against
  cementic's 1-2 s; and it has neither chunking nor PDF extraction) — the point
  is the interface, not the engine. cementic already has the composable layer:
  `extract | chunk | embed` are stdin/stdout filters. It is buried under the
  daemon and the database. *Do it when:* the hybrid thread closes; this is a
  separate piece of work and conflating the two would hide both.

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

- Hybrid lexical + vector search for exact author names, acronyms, citations and
  equation labels. **Now an active thread, not a someday item** — the mechanism is
  measured (vector-only recall@10 is 0.047 on rare exact tokens; hybrid is 1.000)
  and the remaining question is the fusion rule. See PLAN.md's
  "Execution order — hybrid retrieval".
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
  `extractor_registry_payload` (`extract.py:516`) hashes every built-in
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
