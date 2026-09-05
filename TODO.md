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

- Pre-filter over-budget chunks against the model's own tokenizer, not just
  the cheap tiktoken estimate. `count_model_tokens` learned upstream
  `llama-server`'s `/tokenize` in `1ff5269`, so the *exact* check works — but
  `embed_batch`'s pre-filter still uses the estimate, so a chunk at 513–549
  model tokens is sent anyway, fails with a 500, and drags its whole request
  into the isolate-and-retry path. ~5,000 chunks per corpus scan, re-run on
  every `cementic start` because the requeue resets them. Sequence this with
  `PLAN.md`'s model migration, which re-derives `chunk_size` against the chosen
  model's tokenizer and changes this item's arithmetic.

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

- Hybrid lexical + vector search for exact author names, acronyms, citations, and
  equation labels.
- A multi-profile embedding daemon pool if old-model search and new-model
  indexing need to run concurrently. *Trigger reached in principle:* the model
  migration in `PLAN.md` puts exactly these two on one port. The no-code
  workaround is a second `llama-server` on another port, so build this only if
  running two servers by hand proves annoying during that migration.
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
- **Publish the repo.** Nothing gates on it today; the install docs no longer
  describe a command that cannot work. Publishing only changes whether a
  `pipx install git+https://...` one-liner works for someone who is not the
  author. *Reopen when:* you want to hand cementic to another person — at that
  point README's install section goes back to the one-liner in the same commit.
- **Restore support for Python older than 3.12.** *Reopen when:* cementic needs
  to run somewhere that cannot get 3.12. Unlikely while `uv` installs one in a
  single command, and the compatibility branch was deleted precisely because
  nothing exercised it.
