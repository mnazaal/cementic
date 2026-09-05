# cementic — architecture & design

<!-- session-handoff:begin (2026-09-04) -->
## Where the work stands

**No thread is mid-flight; pick up from `TODO.md`'s roadmap.** The 2026-09-05
session closed three things and opened none. The parked fingerprint item is
settled: the embedding canary is implemented and merged (`06f7d62`). The
migration to `nomic-embed-text-v1.5` was decided and then **rejected by its own
gate** -- v1.5 retrieves measurably worse than v2-moe here, so `papers` stays as
it is and the three rebuild-blocked changes stay parked (see the closed
execution order above). And two sections that read as pending work were marked
DONE: the command extractor and query-first embed scheduling had both already
shipped.

**Repo state (corrected 2026-09-05 — the lines below described the previous
session's branches, which have since merged).** `main` carries `06f7d62`, the
canary implementation. Unmerged: `claude/migration-plan`, this plan revision.
A hook rejects agents touching `main`, so merging is yours:
```bash
git checkout main && git merge --ff-only claude/migration-plan
```
`uv tool upgrade cementic` has run: the PATH install reports pymupdf 1.27.1 /
pymupdf4llm 0.3.4 and `extractor_profile: ok`, so the checkout and the snapshot
agree.

**Live state.** `papers` is whole and served: 23,064 documents, 2,298,558
embeddings, revision 16 active, no building revision. Indexing now runs as the
systemd user unit `cementic@papers.service` (one templated unit at
`~/dotfiles/.config/systemd/user/cementic@.service`, stow-deployed, `loginctl
enable-linger` on). It has NOT survived a reboot
yet — that is the only unproven claim about it.

**Correction — PLAN said nothing about this, and it cost a near-rebuild.**
`profiles.py` records the installed pymupdf/pymupdf-layout/pymupdf4llm versions
in the extractor fingerprint, and `pyproject.toml` declared them by floor, so the
checkout (1.27.1/0.3.4) and the uv-tool install (1.28.2) indexed the same corpus
under *different* extractor profiles. A systemd unit pointed at the uv-tool
interpreter opened extractor profile 9 and began re-extracting all 23,064
documents; caught at 747 extractions, before any chunk or vector row was
written. Revision 17, its extractions and its artifact directory were deleted.
`fa03c95` pins all three exactly and adds a `cementic doctor` check that
compares this install's extractor payload against each collection's active
revision. Both environments now resolve the same
versions, so either interpreter would extract identically today — but the unit
stays on `~/projects/cementic/.venv/bin/python`, and **anything that repoints a
`cementic.runner` invocation should check `extractor_profiles.config_json`
against the install it is moving to first.** `cementic doctor` answers that in
one line now.

**Deviation from plan, user-directed:** none of this was on the roadmap. The
session started from "workers didn't start after a reboot" and stayed there.

**Scratchpad — nothing promoted beyond the unit.** `~/.cache/cementic-autostart/`
holds `cementic@.service` (already installed and committed to `~/dotfiles`),
the `install.sh`/`fix.sh` that ran once, and `split/` — a five-unit alternative
(separate embedding server + health timer) kept for the day a second collection
makes the shared-server churn matter. The agent scratchpad held only throwaway
HTTP stubs and is discarded.

**Environment quirks that cost time this session.**
- The agent sandbox has a **separate PID namespace and no systemd bus**:
  `systemctl`, `ps` and `/proc/<pid>` are all useless for liveness. Read the
  database, state-file mtimes, and HTTP probes instead.
- `~/.config`, `~/dotfiles` and `~/.local/bin` are read-only binds; writes there
  silently no-op. Hand the user a script under `~/.cache/` to run.
- Watcher/worker state files are **event-driven**, so a stale `last_updated` on
  an idle collection is normal, not a dead process.
- `cementic status` always reads `workers stopped` under systemd; the real check
  is `systemctl --user is-active cementic@papers`.

**Exit criteria — commands whose output confirms the above.**
```bash
git status --short                              # empty
git log --oneline main..claude/session-handoff-2026-09-04         # this commit only
cementic status -c papers                       # 2,298,558/2,298,558, building=-
cementic doctor                                 # extractor_profile: ok (PATH install)
systemctl --user is-active cementic@papers      # active  (user runs this; no bus in the sandbox)
pytest -m "not pg" -q && pytest -m pg -q        # 1125 passed / 56 passed
```
<!-- session-handoff:end -->

## Decision log

**Migrated 2026-08-23; this section is now a pointer.** The eight entries it held
were three different kinds of thing, and keeping them together made a register
that duplicated the code:

- **Code decisions** — context manager over decorator, the narrowed
  `EXTRACTION_VERSION`, the deferred imports — already live against the code they
  explain (`_reporting_db_errors`'s docstring, `profiles.py`, `config.py`). A
  comment beside the thing it explains cannot drift from it; a separate register
  can, and these had already become copies.
- **Project policy** — `notes/` gitignored, no backwards-compatibility
  obligation, the 3.12 floor — moved to README's "Decisions worth knowing".
- **Scheduling** — why Batch C ran before Batch A, why a characterization test
  preceded the refactor — deliberately dropped. They were about sequencing one
  piece of work and answer nothing six months out.

The two parked items that had revisit conditions moved to `TODO.md`'s "Parked,
with a trigger", because nothing here was watching for them.

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

The target is no longer an estimate: the corpus was surveyed on 2026-08-24 at
22,246 PDFs / 592,248 pages, which at 3.65 chunks/page is **~2.16M vectors**.
The numbers behind that (query latency, embedding and extraction throughput,
GPU rates, HNSW memory) are in README's "Measurements behind the defaults";
the design consequence is everything under Key seams below:
per-embedding-profile vector tables, a `diskann` seam for when memory binds
before latency does, versioned revisions so a model or chunking change builds in
the background and promotes atomically, and a warm llama.cpp server shared by
indexing and search.

## Key seams (the pluggable registries)

### Embedding provider — `embedding_runtime.py`

- `EmbeddingRuntimeSpec` carries model + runtime identity. Providers are
  self-describing: `embedding_dim`, `distance_metric`, `format_document` /
  `format_query`, `embed_batch`, `describe()`.
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

## Execution order — CLOSED, the gate failed (2026-09-05)

The migration this sequenced is **not happening**: v1.5 retrieves measurably
worse than v2-moe on this corpus. Steps 1 and 2 ran; 3 to 6 are void. Kept
short as the record of how it closed; the reasoning is in the migration section
below, and `TODO.md` holds the feature backlog as before.

1. **Merge the canary.** DONE -- `06f7d62` on `main`.
2. **Compare v1.5 against v2-moe.** DONE, and it is the reason this list is
   closed. 1,000 papers, 2,000 body chunks, title-to-body retrieval, both
   models embedded fresh in one session with the stored index vectors as a
   control arm (`~/.cache/cementic-igpu/calibration/compare-models.sh`):

   | arm | recall@1 | recall@10 | MRR | mean margin |
   | --- | --- | --- | --- | --- |
   | v2-moe (current) | 0.7170 | 0.8870 | 0.7797 | 0.0546 |
   | v1.5 (candidate) | 0.6590 | 0.8450 | 0.7267 | 0.0279 |
   | v2-moe from the stored index (control) | 0.7200 | 0.8860 | 0.7812 | 0.0545 |

   v1.5 is worse on every measure: -8.1% recall@1, -6.8% MRR, and it separates
   the right paper from the best wrong one **half** as well (-48.9%). Paired
   over 1,000 queries it ranks the paper higher on 133 and lower on 245 (exact
   sign test p = 8.9e-09), and wins the margin on 299 against the control arm's
   own 523 -- ten standard deviations below the null the control measures. The
   control arm also lands on top of fresh v2-moe, which is what says the
   comparison itself is sound.

3-6. **Void.** `chunk_size` re-derivation, the punctuation filter and OCR
   removal go back to parked: each costs a full rebuild and there is no longer
   a rebuild to ride on. The five-day re-embed and its five days without search
   are not being spent.

## Design decisions and open items

### Decided — external command extractor (2026-09-03) — DONE

**Shipped in `935bf45`** (registry entry, `[extraction.commands]` with its
paired `command_versions`, the `extraction_commands` doctor check, tests).
Retained for the rationale; nothing below is outstanding. Extraction was the
last subsystem that hardcoded
Python libraries and grows a boolean per library feature. Embeddings already
work the other way: `daemon_command` names a binary, cementic spawns it and
consumes its output. A `command` extractor makes extraction symmetric — text
comes from an operator-named argv, so OCR, `pdftotext`, `docling` and anything
else are config, not dependencies. Built ahead of demand deliberately: the
user's call, on the grounds that the seam is cheaper to add now than to retrofit.

**Contract, kept narrow on purpose:** argv in, text on stdout, nothing else. A
tool that writes files instead (`ocrmypdf`) gets a two-line wrapper — that is
the composition boundary, not cementic's problem. A wide contract is what turns
an early abstraction into debt.

```toml
[extraction.backends]
pdf = "command"
[extraction.commands]
pdf = ["pdftotext", "-layout", "{path}", "-"]
[extraction.command_versions]
pdf = ["pdftotext", "-v"]
```

**`version_command` is required, because version flags cannot be guessed.**
Probed on this host: `pdftotext --version` treats the flag as a filename,
prints an I/O error and **exits 0**, while `pdftotext -v` works; `gs --version`
and `mutool -v` disagree again. So neither the flag nor the exit code is a
reliable signal, and a guessed flag would record a constant error string as
"the version" — never changing on upgrade, silently disabling the guarantee.
Its output enters the extraction payload so a tool upgrade re-versions
revisions, the same property `extraction_libraries` gives pymupdf. Residual
risk: a wrong flag that prints a constant cannot be detected automatically, so
`doctor` prints what was recorded for a human to check.

**Probe cost is a non-issue.** `build_extractor_profile_payload` is reached only
via `get_or_create_extractor_profile` ← `revisions.py:184` ← revision creation.
Once per revision, not per document; no cache needed.

**Fingerprint-neutral by gating.** Both new payload keys appear only when a
command backend is configured, so existing fingerprints do not move and no
corpus rebuilds. Same trick as the conditional rapidocr entry in `profiles.py`.

**`supported_extensions()` gains a `Config`.** The command extractor's
extensions come from the `[extraction.commands]` keys, otherwise it could never
add a file type cementic does not already know — its best use. Ripples to
`source_watcher.py:125` and `:385`; `DocumentEventHandler` takes the resolved
set as plain data, matching how `ignore_directories` is already passed. The
command extractor is never a fallback: it must be named in `backends`.

### Rejected by measurement — migrating `papers` to v1.5 (2026-09-05)

**Decided, then reversed the same day by the gate that was built to test it.**
The candidate was `nomic-embed-text-v1.5`; it retrieves worse than the v2-moe
already in place (numbers in the closed execution order above), so `papers`
stays on v2-moe and the three rebuild-blocked changes stay parked. Everything
below is kept because it is what a *future* model swap costs, and because the
reasoning that pointed at v1.5 was sound right up to the measurement -- which
is the argument for keeping the gate cheap and running it first.

**What would reopen this.** A candidate that beats v2-moe on
`compare-models.sh`, run the same way. Two facts worth carrying into that
search: a longer context is not itself worth a rebuild -- v1.5's 2048 against
v2-moe's 512 did not compensate for worse retrieval -- and v1.5 embedded ~4x
faster in the same run (3.4 against 0.8 embeds/s, partly because v2-moe's
512-token window refused five batches to v1.5's one, forcing per-text retries).
If indexing throughput ever binds harder than retrieval quality, that trade is
now measured rather than guessed.

Three changes were parked waiting for a rebuild to happen for some independent
reason, each costing a full re-embed on its own. A model swap would have been
that reason, so they were to ride along — doing them afterwards costs a second
~5-day re-embed for nothing.

What a model swap actually costs, read out of the code and the live database
rather than estimated:

- **Extraction is untouched.** The extractor profile does not change, so all
  23,064 documents are reused as they are.
- **Chunks are reused unless chunk settings change.** `build_chunk_profile_payload`
  is `chunk_size`, `chunk_overlap`, the tiktoken tokenizer and a version — the
  model is not in it. In practice the swap changes them anyway, because
  `chunk_size = 320` was chosen for this model's 512-token window. Re-chunking
  needs no GPU and no server.
- **Every chunk is re-embedded.** 2,298,558 at the measured 5.56 chunk/s is about
  five days. A longer context trades chunk count against per-chunk cost rather
  than removing the work, so treat any speedup as unmeasured until it is measured.
- **No downtime, and no mixing.** Vectors live in a table per embedding profile
  (`embedding_vectors_p6` today). The new revision builds into its own table
  while the old one stays active and searchable; `promote_revision` flips at the
  end. Cross-collection search already refuses to merge two models' scores
  (`search.py:_mixed_model_message`).
- **Disk roughly doubles until the old revision is pruned.** The database is
  21 GB, of which `embedding_vectors_p6` — vectors plus HNSW index — is 18 GB.
- **One daemon serves one model.** The server's `--alias` fingerprints what it
  loaded and the readiness check refuses a server not serving it, so a
  wrong-model answer is impossible — but indexing the new model and searching the
  old contend for port 11555. This is the trigger for `TODO.md`'s parked
  "multi-profile embedding daemon pool"; the no-code workaround is a second
  `llama-server` on another port, which is already how the iGPU setup runs.

**Chosen 2026-09-05 — `nomic-embed-text-v1.5`.**
768 dimensions (unchanged, so the vector table shape is unchanged), a longer
context than the current 512 (measured below, and it is not the 8192 the model
card advertises), Matryoshka truncation, a published
GGUF, and an embedding space shared with `nomic-embed-vision-v1.5` — which would
let figures and page images be searched alongside text without a second vector
space, the thing `Markdown as the text intermediate representation` above calls
option (b). The cost is giving up v2-moe's multilingual MoE, and v1.5 is an
older generation (MTEB 62.28 at 768 dims, per its model card).

Rejected: staying on v2-moe, which keeps the 512-token window and leaves all
three parked items parked; and going straight to a multimodal model
(`nomic-embed-multimodal-3b/7b`, `colnomic-embed-multimodal-3b/7b`), which is
3-7B parameters against v1.5's ~0.1B, needs the parallel non-Markdown path, and
has unverified llama.cpp support. v1.5's shared vision space is what makes that
a cheap follow-on rather than a competing direction.

**Load check done 2026-09-05, and it corrects the headline number.** The GGUF
already in `models/` (`nomic-embed-text-v1.5.f16.gguf`, 274 MB) loads under
b10818 and serves 768-dim vectors. But its context is **2048, not 8192**:
`n_ctx_train` in the GGUF is 2048, and llama.cpp caps the slot to it -- "the
slot context (8192) exceeds the training context of the model (2048) -
capping", after which a 3,002-token input is refused outright. The card's 8192
comes from RoPE scaling that this conversion does not carry.

8192 *can* be forced -- `--rope-scaling yarn --rope-scale 4` loads at
`n_ctx_slot = 8192` and embeds a 3,000-word input fine -- but that is untested
for quality, and the flags do not match what the card specifies (dynamic NTK
scaling, factor 2.0). Treat scaled RoPE as its own experiment, not as a
config line.

So the honest gain is **4x the current window, not 16x**: 2048 tokens supports
a `chunk_size` near 1,300 against today's 320, which cuts the corpus from
~2.3M chunks to roughly 575k and the vector table from 18 GB to ~4.5 GB. Still
worth the migration; just not the number the plan was written around.

One scaffold-time check remains: how v1.5 and v2-moe actually compare on
retrieval over these papers. It precedes scheduling the five-day re-embed,
because it is the only claim here still resting on a model card.

**What rides along, in the order the pipeline runs them.**

*Drop chunks that are mostly punctuation, at chunk time.* Measured on the live
corpus: of 5,025 chunks that failed the token budget, ~3,830 are dot-leader
tables of contents (`. . . . . 117 C.5.4 Proof of Claim 19`). They tokenize at
~1.59 model tokens per tiktoken token against a 1.45 bound, which is why they
overflow. They are worthless for semantic search either way, and 735 shorter
ones *did* embed and sit in the index as junk vectors. The length limit is
currently acting as an accidental filter for the rest. *Ends when:* a re-chunk
of the corpus produces no dot-leader chunks and the junk vectors are gone.

*Re-derive `chunk_size` against the model's own tokenizer, not tiktoken.* The
mismatch is the root cause of the item above. Do not simply lower it: on the
current model, clearing the observed p95 (637 model tokens) needs 249 and the
observed max (1958) needs 75, which is too small to be a useful chunk. v1.5's
measured 2048-token window changes that arithmetic entirely: the observed max
fits with room to spare, so the filter above stops being load-bearing.
*Ends when:* `chunk_size` is set from a measured token-ratio distribution
against the chosen model's `/tokenize`, recorded here.

*Remove OCR from the codebase.* See the section below; it is blocked on this
same rebuild and on the command extractor landing first.

Already done, and it needed no rebuild: **splitting over-budget chunks at embed
time** (`c99d84b`, 2026-09-04). Chunking and both profiles were untouched, so
requeueing the failed rows was enough — chunks that had no vector gained one,
which is additive. The splitter halves on whitespace, then on characters,
because dot-leader pages are frequently one whitespace-free run. Roughly 1,200
of the affected chunks are genuinely dense content (code, maths, long structured
titles); the rest are contents pages that the punctuation filter above should
stop creating at all.

### Parked again — remove OCR from the codebase

**Not blocked on design; it now has a rebuild to ride on.** `use_ocr` reaches
extraction only through the pymupdf4llm backend, so it is a knob that silently
does nothing under `pymupdf-raw` — `doctor` grew a warning for exactly that,
which is a guard where the fix is removing the knob. Deleting it removes
rapidocr from the picture entirely (dependency, extra, and `--with` all moot).

The cost is that `use_ocr` is a key in the extractor payload, so removing the
field changes every extraction fingerprint: `papers` re-extracts, re-chunks and
re-embeds all 2,291,555 chunks. The model migration would have paid for that
rebuild anyway, and it is not happening, so this waits for the next real reason
to rebuild. Its *other* constraint is now satisfied: the command extractor
shipped in `935bf45`, so a configured `ocrmypdf` wrapper remains an OCR route
once the knob is gone.
Pre-processing with `ocrmypdf` into `pymupdf-raw` works with no cementic code at
all and is the documented fallback — untested here, no OCR engine is installed
on this host.

### Decided — detect server drift with a canary, not a fingerprint (2026-09-05)

**Implemented 2026-09-05 (`canary.py`, `doctor.py`'s `embedding_canary` check,
`embedding_profile_canaries`). Supersedes the open item that asked whether to
record the llama.cpp build in the embedding profile, and corrects its central
claim.** That note said a server upgrade is "usually numerically a no-op".
Nothing had measured it, and as stated it is false: new kernels and different
reduction orders move the last bits by construction.

Measured 2026-09-05 against the live server (`b10605-a130532ae`,
nomic-embed-text-v2-moe Q8_0, Vulkan iGPU), embedding one text three ways over
`/v1/embeddings`:

| Same text, same model, same build | Bitwise identical | max abs diff | cosine |
| --- | --- | --- | --- |
| the same request, twice | yes | 0 | 1.000000000000 |
| alone vs. first of a 2-text request | no | 7.5e-9 | 1.000000000000 |
| alone vs. last of a 32-text request | no | 2.3e-3 | 0.999796 |

**The corpus is already not bitwise reproducible, and no fingerprint can make it
so.** The worker submits `pipeline_worker.batch_size = 32` texts per request, so
a chunk's exact vector depends on which 31 others rode with it — on arrival
order, and on where a restart landed. What the fingerprint guarantees is
therefore not "the same bits" but "the same vector space, comparable distances",
and 2.3e-3 of wobble does not threaten that.

That reframes the question. It is not *does an upgrade change the numbers*
(everything does, including doing nothing) but *does it change them by more than
the noise the corpus already carries*. Two classes of llama.cpp change, which a
build string cannot tell apart:

- **arithmetic** — new kernels, different reduction order, quantization
  rewrites. The same magnitude as the batching wobble above. A rebuild buys
  nothing.
- **semantic** — a tokenizer fix, a pooling-type or normalization default, an
  attention-mask fix for embedding models. These change the function rather than
  its rounding, and would move cosine far below 0.999. A rebuild is mandatory.

**Decision: `cementic doctor` grows a behavioral canary, and the server build is
recorded beside it as evidence rather than as identity.** A revision stores a
handful of fixed reference texts with their vectors; doctor re-embeds them **as
one fixed request — same texts, same order, same count** — and reports the
cosine against the stored ones. The first row of the table is what makes this
work: a fixed request is bitwise reproducible within a build, so any difference
at all is the server's doing, and the size of it says which class the change is.
**The comparison is strict** — doctor flags any difference from the stored
vectors, and reports the cosine only to say how bad it is (~1e-9 is arithmetic
noise, below 0.999 is a changed function). Accepting a drifted build must be an
explicit act that re-stamps the canary, not a threshold that silently absorbs
it; that is the difference between a check that stays trustworthy and one that
is tuned until it never fires.

The canary must live *outside* `build_embedding_profile_payload`. Anything in
that payload moves the fingerprint and forks the corpus, which is the outcome
being avoided. The natural home is a sibling table keyed by
`embedding_profile_id` holding the texts, their vectors, and the `build_info`
string the server reported when they were taken (`/props`, today
`b10605-a130532ae`). Note that `EmbeddingProfile.config_json` is *not* the
fingerprinted payload — it carries the runtime payload `search` rebuilds a spec
from (`profiles.py:307`) — so it is not a shortcut home for this either.

Rejected:

- **Leave it and document the hole.** Free, and forgotten by exactly the person
  it would have saved.
- **Put the build in the embedding profile.** Fires on every upgrade including
  the arithmetic-class majority, each costing a full re-embed (~5 days at the
  measured 5.56 chunk/s), and makes profile construction depend on a live
  daemon, which today it does not.
- **Compare build strings in doctor.** The same false-alarm rate without the
  rebuild. A check that fires on every upgrade is a check that gets ignored.

**Threshold: cosine 0.99 — and the "no threshold at all" plan above was
wrong.** Cross-build calibration ran b10605 against b10818 (213 builds and ~12
days later), same model, same flags, the same 200 real `papers` chunks in the
production 32-per-request shape, on both backends:

| Backend | Result |
| --- | --- |
| CPU (agent sandbox, no `/dev/dri`) | 200/200 bitwise identical, max abs diff 0 |
| Vulkan iGPU (`Intel(R) Graphics (MTL)`) | 200/200 bitwise identical, max abs diff 0 |

That reads as "zero drift, so compare exactly", and it is the wrong conclusion.
Both runs replayed the *same traffic sequence* against a fresh server, holding
fixed the one variable that actually moves vectors. Running the implemented
canary against the live server exposed it on first contact: replaying the
canary after an unrelated 3-text request gives cosine 0.999908 instead of 1.0,
and after an 18-text request 0.999887, while back-to-back replays with nothing
in between are exact. llama-server packs concurrent slot work into unified
batches (`kv_unified = true` in its own startup log), so reproducibility is a
property of the request *and its predecessors* — which no client can fix, and
which `doctor` cannot avoid, since its own daemon probe embeds first.

The floor is therefore measured rather than absent: scheduling noise reaches
2.0e-4 in (1 - cosine) and request composition 2.3e-3, while a tokenizer,
pooling or normalization change lands two orders of magnitude below. `0.99`
sits in that gap, ~100x above the worst observed noise, deliberately
conservative because both bounds come from a handful of sampled shapes rather
than a derivation. What the cross-build table still establishes is that no
*semantic* change happened across those 213 builds; that conclusion never
depended on the traffic sequence.

Reproduce with `~/.cache/cementic-igpu/calibration/calibrate.sh` (it reuses
`sample.json`, so inputs stay identical); the CPU-run artifacts are preserved
alongside as `*-cpu`. Note for whoever repeats this: absence of Vulkan lines in
the server log does **not** mean CPU — the known-GPU run from 2026-08-24 has
none either. Check `llama-server --list-devices`, and remember an agent shell
sees no GPU at all.

**Left undone deliberately.** Re-stamping is a row delete plus one indexing
run, which capture's store-if-absent rule already makes correct; a
`cementic doctor --restamp` would be a nicer front door for it, and is not
worth a command until a real build upgrade makes someone want it. Capture is
also indexing-only: a collection that never indexes again keeps no canary, so
the check reports honestly that it has nothing to replay rather than inventing
a reference from today's server.

### Decided — query-first embed scheduling (2026-09-01) — DONE

**Both layers shipped**: sub-batching as `pipeline_worker.embed_submit_batch_size`
and the search lease as `db.SearchActivity` / `search.record_search_activity`,
consumed by `pipeline_worker`'s `_wait_while_search_is_active`. Retained for the
measurements and the parked triggers. A search query landing mid-import waits on the
shared llama-server: the worker POSTs 32 texts per request, b10605 schedules
per-input FIFO with no HTTP priority hook, so a query's one task queues behind
up to 32 — measured 0.4–9.6 s of queue delay (17.2 s worst-case wall) against
~0.1 s of actual ANN+SQL. Design, prior-art survey (storage engines, vector
DBs, inference servers, desktop indexers), and verified b10605 scheduler facts:
`notes/design-embed-scheduling.html`.

Three layers, first two to build:

1. **Sub-batch submission** — keep the DB claim at 32, submit to llama-server
   in sub-batches of ~slot count (4). Bounds query wait ~10 s → ~1 s with no
   coordination code (queue delay is proportional to per-request input count).
2. **Search lease** — `search` upserts a timestamp (`utc_now()`, house style) before embedding;
   the worker checks between sub-batches and yields while fresh (TTL ~10 s,
   refreshed per search), with a starvation cap so continuous searching cannot
   stall a build. Precedent: Windows `IRowsetPrioritization`, Postgres
   autovacuum cancellation, OpenAI flex tier.
3. **Abort in-flight on lease activity** — possible (b10605 cancels queued
   tasks ≤ ~1 s on client disconnect) but not built: Layer 1 shrinks the
   quantum until aborting saves ≤ ~1 s. Seam left open.

Parked with triggers: a second query-side llama-server (Vespa's documented
practice; the strong fix if Layers 1–2 measure insufficient — costs runtime-
fingerprint work and resident memory) and an upstream llama.cpp patch exposing
the internal front-of-queue path as a `priority` field on embeddings.

**Amended 2026-09-02 — Layer 1 was not free, and is now conditional.** The
design claimed sub-batching left indexing throughput untouched when nobody
searches. It did not: measured on the papers import, a constant request size of
4 cost **~2× throughput** (≈4.0 → ≈1.7–2.9 emb/s), a step visible at the exact
worker restart that put it into service. Two causes, both measured against the
live llama-server: a fixed ~0.36 s per request (0.72 s for one input against
12.2 s for 32, so ~0.36 s of that is size-independent), and ~25% of wall clock
with no slot busy, because a request sized to the slot count empties all four
slots across every client round trip.

The request size now follows the lease instead of being constant
(`embed_submit_size`): the whole claim when no search is recent,
`embed_submit_batch_size` while one is. Layer 2's yield is unchanged and still
does the work during an interactive session. The residual is the honest cost of
not building Layer 3 — the *first* query after a quiet stretch can land
mid-request and wait out one full claim. It is bounded, and it exists only
while a build runs: once a collection is promoted the worker embeds nothing and
queries are uncontended.

This also promotes Layer 3 from "value evaporated" to the thing that would
remove the residual, since the premise that retired it — that Layer 1 had
already shrunk the quantum for free — is what the measurement falsified.

**Correction, same day, after the fix shipped: the ~2× attribution above does
not hold.** With the adaptive sizing running, requests are confirmed
claim-sized (40 groups of exactly 32 launches between client round trips) and
the search lease is hours stale, yet throughput is 2.24 chunk/s — inside the
1.7–2.9 band the constant-4 build ran at, not the ~4.0 that preceded it.
Restoring big requests did not restore the old rate, so the 4.0 → 2.0 step at
the 2026-09-01 17:23 restart is **unexplained**; it correlated with the
sub-batch change and was attributed to it on that correlation alone.

Ruled out since: chunk length (flat at 317–323 model tokens across the whole
run) and GPU availability (the iGPU is visible and the server never restarted).
Still open: HNSW insert cost growing with the graph — which fits the slow
5.5 → 4.0 decay better than it fits a step — host contention, and pgvector
index maintenance.

What survives: the fixed ~0.36 s per request is measured and real, so
sub-batching does cost *something*; and a hard ceiling was found that bounds
any work here — `--ubatch-size` is 512 while chunks average ~318 model tokens,
so about one chunk fits per forward pass and one slot took 300 of 400 launches
with 32 tasks queued. Request size cannot buy concurrency the physical batch
has no room for. Raising `n_ctx`/ubatch is the untested lever and changes the
runtime fingerprint, so the server must restart for it.

### Decided — build the ANN index up front (2026-08-13)

**Taken.** `ensure_revision_ann_index_up_front` (`revisions.py`) creates the
HNSW index beside the vector table in `_ensure_target_revision`, guarded to the
empty-table case (rows without an index mean a resumed build — its bulk build
stays at the ready transition) and to HNSW (DiskANN is unmeasured and keeps
build-at-ready). The ready-transition and `collection reindex` calls stay for
resumes and method changes. The measurements that justified it:

**Should the ANN index be created up front, on the empty table?** HNSW has no
training step, so it can be. Every insert then maintains the graph and the
build stall disappears entirely. Measured at 100k × 768, `maintenance_work_mem`
2GB for both, clustered vectors (uniform random ones sit at near-identical
distances, which makes recall meaningless):

| | insert then build (today) | build then insert |
|---|---|---|
| insert | 61.8 s | 413.1 s |
| build | 81.5 s — worker blocked | 0.0 s |
| **total** | **143.3 s** | **413.1 s** |
| longest unresumable step | 81.5 s | none |
| recall@10 mean / worst | 0.996 / 0.900 | 1.000 / 1.000 |
| query latency | ~0.74 ms | ~0.74 ms |
| index size (30k) | 117 MB | 117 MB |

So it costs ~2.9× total indexing time and buys: no stall, full resumability
(inserts commit per batch, so a kill loses one batch instead of the whole
build), a searchable index *during* ingestion rather than after, and recall and
latency that are equal or better. pgvector's guidance that building after
loading is faster holds; that it yields a better graph did not, here.

The 2.9× is on the insert step, and cementic's pipeline is embedding-bound:
+351 s per 100k vectors is ~3.5 ms of index maintenance per chunk, against tens
of milliseconds to embed one. Proportionally small, and spread out instead of
concentrated.

**Two earlier numbers here were wrong and are corrected above.** A first A/B run
reported a 100× query-latency gap and a recall difference between the two build
orders. Both were artifacts: the run never checked which plan each query got,
and the `enable_indexscan = off` used to force an exact baseline leaked onto
pooled connections, so some "ANN" queries were sequential scans. Re-measured
with both plans asserted via `EXPLAIN` and a separate pool for the baseline, the
two indexes are the same size and the same speed.

### Fourth review (2026-08-14) — the lesson worth keeping

What it found and what closed each is in the note. One lesson has no other home,
because it is about how the fix itself went wrong:

**A guard is only as good as the thing its test measures.** The headline finding
was that `chunk_size` (tiktoken tokens) was pitted against `n_ctx` (the model's
own tokens), silently truncating ~30% of every full chunk. The fix lowered
`chunk_size` and added a runtime guard with a cheap pre-filter, pinned by a new
invariant test. Both the chosen constant and the test were wrong in the same way:
they measured the bare chunk, while the guard measures the *formatted* text, and
the `search_document: ` prefix adds 3 tokens. That pushed every chunk 3 tokens
past the skip threshold, so the "rare" exact-count path ran on every single
chunk — which, against a daemon that serialises requests, stalled a live
re-index for hours. The test now builds a real chunk and formats it exactly as
the client does; set back to the old value, it fails.

The same trap produced the original bug: `measure_chunk_context_fit.py` embedded
`chunk[:len*0.75]` while printing the *full* chunk's token count beside an "ok"
verdict, so the config comment citing it as proof of safety was citing a
measurement that never tested the case it claimed.

### Deliberately not done

Findings from the **first** review (2026-08-06) judged not worth fixing — not
oversights, and distinct from the feature roadmap in `TODO.md`. Reasoning is in
the relevant commit messages: ANN pre-filter recall on shared vector tables
(inherent to ANN + post-filter; the actionable slice is purging vectors for
deleted documents), `check_health` treating a live-but-broken daemon as healthy
(documented tradeoff), DiskANN + inner-product `storage_layout` (unreachable
while only cosine exists), and reading model identity from GGUF metadata instead
of the filename.

That last item bundled two concerns the code has since separated, so read it as
half-closed. *Which model produced these vectors* no longer comes from a
filename: Batch C (`77e0e3f`) made it a content digest (`profiles.py:169`).
*Which text policy applies* is still a filename match (`embedding_text.py:33`),
deliberately so — that half is the fifth-review residual below, and it is the
only live part of this item.

Of those, the `status` half of the `check_health` tradeoff **was** reopened and
fixed — `status` no longer blocks 120 s to return an answer the pid file already
had. Two more were reopened and are now also closed; both entries below were
still written as open when the sixth review (2026-08-23) checked them against
the code, which is why they carry their resolution inline:

- ~~**`check_health` still calls a live-but-broken daemon healthy.**~~
  **RESOLVED 2026-08-18** by the release plan's batch 1a (`04e974a`):
  `DaemonHealth.WEDGED` (`embedding_runtime.py:553-567`) plus the two-stage
  probe, with the busy/wedged disambiguation observed working live during the
  batch-3 build. The motivating incident stands as the record: on 2026-08-15 a
  wedged daemon held the port for 21 hours while `cementic status` reported
  `embedding healthy`.
- ~~**The filename heuristic silently disables Nomic task prefixes.**~~
  **RESOLVED 2026-08-18** by batch 1b (`9f1d148`): `_NOMIC_V2_MARKER` no longer
  exists; `embedding_text.py:33` is `_NOMIC_FAMILY_MARKER = "nomic-embed-text"`,
  matching v1/v1.5/v2, and the pinning test was retired. The *residual* is
  unchanged and still deferred: matching on filename at all still mis-selects
  for a renamed GGUF, and reading GGUF metadata remains the real fix.

From the **fifth** review (2026-08-18), deferred or kept with reasons:

- ~~**The watcher is blind to directory moves until restart** (fifth review
  §2.9).~~ **RESOLVED 2026-08-18** (`b1999da`) — the repro was run and the
  measured event table is in README's "Measurements behind the defaults":
  move-out (case C) was real and is fixed; root-move (case D) is unfixable from
  inside the watch and is documented in README's Known limitations.
- **TOML parsed once per pydantic-settings section source** (~9× per `Config`
  construction). Caching keyed on path+mtime risks stale reads on
  coarse-mtime filesystems — test flakiness — for a millisecond-scale win.
  Revisit only if config load ever shows up in a profile.
- Three §5 trim candidates stay: `_llama_daemon_runtime_status` (a seam its
  tests pin), `_state` (turned out multi-use), and the `state_path is None`
  guards (they defend the field's declared type).

Closed again by the round-two review, with reasons — each of these looks like a
bug and is one, but the fix costs more than the defect:

- ~~**`verbose` in the embedding profile fingerprint.**~~ **RESOLVED 2026-08-23**
  by Batch C (`77e0e3f`). The entry's own condition is what closed it: removal
  was only worth doing batched with a model-identity change so the corpus is
  re-embedded once, and Batch C was that change. `profiles.py:174` now records
  the deliberate absence. It correctly stays in the *runtime* fingerprint, where
  it is a launch argument.
- **Naive `TIMESTAMP` columns.** Zero readers today (one write, no comparison,
  no display). The fix is a column-type change in a project whose only schema
  mechanism is `create_all`, so old and new databases would diverge with nothing
  to reconcile them. Revisit if anything ever reads these columns.
- **`connect_args` / `gssencmode` on non-psycopg URLs.** `DatabaseConfig` can
  only produce `postgresql://`, and sqlite never reaches `get_engine` outside
  tests that patch around it. Worth the `make_url` cleanup only if `db.py` is
  open for another reason — *and it has been twice since* (`build_memory`, then
  the atomic `force_rebuild` drop), so the stated precondition is now met.
  Note the gap is wider than first recorded: the check is
  `startswith("postgresql://")`, so it also misses driver-qualified URLs like
  `postgresql+psycopg2://`, which a user setting `CEMENTIC_DB_URL` may well write.
  **Audit 2026-08-23:** still live at `db.py:316`, and the entry now fails its
  own test — a register of things deliberately not done cannot hold an item whose
  stated trigger has already fired. Queued in `TODO.md` instead.
- **`ignore_directories` replacing the defaults.** Working as documented,
  including the "empty list indexes everything" escape hatch a union would
  break. If discoverability is the concern, add a separate
  `additional_ignore_directories` — a feature decision, not a defect fix.
- **The advisory-lock leak and the `RUNNING` state-write ordering.** Every path
  that leaks the lock is immediately followed by process exit, which releases
  it; the stale state file is corrected by `start`'s liveness check and cleared
  by `stop`. Two-line fixes if those functions are open anyway, not worth a slot.
- **`index.method` being unreachable on a built system.** Closed by
  `cementic collection reindex`, which reconciles the active revision's index
  with the current `[index]` config. *(The instruction that used to sit here —
  "correct the sentence in `TODO.md` claiming the build path already reconciles
  a changed method" — was itself stale: no such sentence exists in `TODO.md`.)*

**Closed 2026-08-23 — the four fourth-review carry-overs.** All four are done
and were verified live; the commits cited below carry the reasoning.

- `cementic start` reporting success after 2 s → Batch B (`ba3ac8a`): the fatal
  reason now reaches `cementic status`, and the message no longer claims a
  success two seconds of watching cannot establish.
- The unreclaimable orphaned daemon → Batch A (`e5ba0b9`): recovery from `/proc`
  by the spawned command line, which closes the general case rather than the one
  cause Batch 1 fixed.
- Model identity being the path string → Batch C (`77e0e3f`): identity is a
  content digest.
- `EXTRACTION_VERSION` hand-maintained → Batch C (`77e0e3f`): derived from the
  installed extraction libraries.

### Audit findings not fixed (2026-08-24)

Three read-only audits ran before the bulk import — hostile document content,
failure handling, scale. Four findings were fixed (`<|endoftext|>` chunking, the
unescaped LIKE prefix, both write-back stalls, and separately the NUL bytes and
the retryable-500). These are the rest, kept because they are real and because a
finding that lives only in a chat log is a finding nobody will act on. None
blocks the import.

Checked against the real 22,246-file corpus and **not present**, so reachable in
principle but not here: filenames containing `[` (would break
`status --verbose`, `render.py:260`), non-UTF-8 filenames (`source_watcher.py:566`
then `render.py:157`), and paths over ~2,600 bytes (btree limit on
`ix_source_documents_collection_source`, `db.py:70`).

Live, in rough order of how likely they are to bite:

- **`cementic status --verbose` has no limit** (`status_service.py:477`,
  `render.py:253`). One line per document; at 22k it loads ~50 MB of ORM objects
  and floods the terminal. Add `--limit`, defaulting to failed/pending first.
- **`_step_embed` publishes no worker activity** (`pipeline_worker.py`, the
  embed step writes neither `current_file` nor `current_activity`). The
  busy-vs-wedged guard reads exactly that state, so during the phase that
  saturates the daemon a healthy server can be reported WEDGED — and the
  suggested remedy, restarting it, discards an in-flight batch.
- **Whole-corpus deletes run as one unbounded transaction** with a 22k-element
  `IN` list (`source_watcher.py:338`, `collections.py:136`), blocking autovacuum
  for its duration and leaving millions of dead index entries.
- **Progress rounds to `100.0%` while thousands of chunks are outstanding**
  (`status_service.py:115`); at 2.16M the display resolution is ~2,162 chunks.
- **A revision can reach `ready` mid-initial-scan** (`pipeline_worker.py:977`):
  the watcher registers documents one at a time with no scan-complete marker, so
  a drain of the first N looks complete. Promotion re-checks and refuses, so this
  misleads rather than corrupts.
- **`chunk_text` is quadratic in one whitespace-free run** (`chunk.py`): 160k
  characters took 12.2 s, inside a Rust call that ignores the shutdown event.
  Reachable from a minified line or a PDF text layer with no spaces.
- **`extractor_registry_payload` hashes the whole registry** (`extract.py:282`),
  so adding a `.docx` extractor re-versions every existing PDF corpus. Matters
  more now that new extractors are near-term work.
- **`extract_pdf_markdown`'s signature defaults `use_ocr=True`** while the config
  defaults `False` (`extract.py:47`). No effect today — `_pdf_extractor` always
  passes the config value — but the two disagree.

Resolved by the systemd units rather than by code: the worker had no respawn
after an OOM or reboot, so a multi-day run ended silently.

## Road to v1

All four must hold before a 1.0 tag; none is scheduled work yet:

1. **Soak time under real use** — weeks of daily driving on the live corpus
   without surprises. Confidence comes from use, not review passes.
2. **More features first** — some of `TODO.md` belongs in a v1: more
   extractors (`.docx`/`.html`/`.epub`), `cementic add`, richer search
   output. Which subset is a decision for when v1 planning starts.
3. **Config/CLI stability confidence** — the config schema and CLI surface
   should stop moving; recent review cycles changed both repeatedly. A signal:
   several consecutive releases with no breaking config/CLI change.
4. **Multi-platform verification** — macOS (and possibly Windows) actually
   tested rather than "best-effort", since the README ships install
   instructions for them.

