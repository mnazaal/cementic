#!/usr/bin/env python
"""Retrieval quality of the active revision, scored on synthetic known-item queries.

The harness PLAN.md's "Execution order — retrieval quality and the rebuild it
rides" calls Phase 0. It exists so that a change to the ranker, the chunker or
the embedding model can be scored rather than argued about, and so the score
survives the rebuild that produced it.

Three commands, because they cost different things and only the two that write
are allowed to move the query set:

    build     Draw query sets from the corpus and write them to a JSON file.
              Thousands of document-frequency queries; minutes. Run once.
    perturb   Derive imperfectly-recalled variants of the phrase set and add
              them to that file. Pure rewrite, no corpus, no model; instant.
    score     Read that file, run each query through the real search path, and
              print recall@1 / recall@10 / MRR per set. Needs the embedding
              server; seconds per query.
    arms      Fetch both arms' pre-fusion candidates for every query, plus the
              shipped rank-1 decision, and cache them outside the repo. Same
              database cost as `score`; run once per revision.
    ablate    Score rank-1 policies against that cache: shipped routing, plain
              fusion, each arm always leading, and the per-query oracle.
              Pure; no database, seconds.
    compare-models
              Embedding models head to head on one body-text span per paper,
              each at 320- and 1300-token chunks, with the stored index as the
              control. Starts one llama-server per model; hours, resumable.
    riders    The chunking riders 2b, 2c and 2e applied to the same spans and
              embedded with v2-moe, scored per document against fresh v2-moe.
              Needs `compare-models` to have run; one llama-server, minutes.
    fuzzy     Two lexical fallbacks for typo'd queries (pg_trgm, a relaxed
              full-text match) scored against the live database on the phrase
              sets. Creates the pg_trgm extension; the trigram scan is slow.

Splitting them is the point. A judgment re-derived on every run is not a
judgment, it is a coin flip with a seed -- so `build`'s output is checked into
git and `score` never writes to it. Comparing two systems means running `score`
twice against the *same* file, including across a re-index. `perturb` writes,
but only ever appends its own derived kinds and regenerates them in place, so
the sets `build` drew stay byte-identical.

Judgments are mechanical. Nobody hand-labels anything: each query is drawn from
a document that is by construction the answer, which is what makes the gold
label free and the query set re-buildable on a different corpus.

Usage:
    python scripts/measure_retrieval_quality.py build -o eval/queries.json
    python scripts/measure_retrieval_quality.py perturb -q eval/queries.json
    python scripts/measure_retrieval_quality.py score -q eval/queries.json
    python scripts/measure_retrieval_quality.py arms -q eval/queries.json
    python scripts/measure_retrieval_quality.py ablate
    python scripts/measure_retrieval_quality.py compare-models
    python scripts/measure_retrieval_quality.py riders
    python scripts/measure_retrieval_quality.py fuzzy

The database is not reachable from an agent sandbox; run these unsandboxed.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import random
import re
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

from sqlalchemy import text

from cementic.chunk import chunk_text, tokenizer
from cementic.config import get_config
from cementic.db import LEXICAL_TEXT_CONFIG, get_engine
from cementic.embedding_text import (
    format_document_text_for_model,
    format_query_text_for_model,
)
from cementic.hybrid import combine, reciprocal_rank_fusion
from cementic.search import Searcher, arm_fetch_limit, split_arms
from cementic.storage import read_extracted_text
from cementic.vector_store import LEXICAL_CANDIDATE_POOL, vector_table_name

#: Default ranking depth requested from search. Matches
#: `search.MAX_SEARCH_RESULTS`, so recall@10 and MRR are measured over the
#: deepest list the CLI can return.
#:
#: **This is not what `cementic search` does by default, and the gap is not
#: cosmetic (found 2026-09-19).** `search()` passes `top_k` straight through as
#: the per-arm fetch limit for both arms *and* into `query_tuning_statements`,
#: so it sets ANN effort as well as list length. The CLI defaults to 10
#: (`cli.py:965`). Scoring at 50 therefore measures a system that over-fetches
#: 5x relative to the shipped default -- which is exactly the change Phase 1
#: step 1 proposes, already switched on. Use `--depth 10` for the shipped
#: configuration and `--depth 50` for the over-fetched one; the number is
#: printed with every table because the two are not comparable.
DEFAULT_DEPTH = 50

#: Depth `cementic search` uses unless asked otherwise (`cli.py:965`).
PRODUCTION_DEPTH = 10

#: Fixed so `build` is reproducible. Changing it draws a different query set,
#: which makes the scores incomparable with every earlier run -- that is what
#: the checked-in JSON exists to prevent.
SEED = 20260918

#: Identifier shape for the rare-token set: capitalised, >= 3 characters.
#: Deliberately loose; the document-frequency filter is what makes a query
#: well-posed, not this regex.
TOKEN_RE = re.compile(r"\b[A-Z][A-Za-z]{2,}\b")

#: Sentence-initial capitals make ordinary words look like identifiers. Carried
#: over from notes/probe_hybrid_retrieval.py, where they were the ones that
#: survived the shape filter often enough to matter.
STOPWORDS = {
    "The", "This", "That", "These", "Those", "There", "Then", "They", "Their",
    "We", "Our", "It", "Its", "In", "For", "And", "But", "Not", "All", "One",
    "Two", "Table", "Figure", "Section", "Appendix", "However", "Since", "Thus",
    "Here", "When", "While", "With", "From", "Given", "Let", "Note", "See",
}

#: A query matching more *documents* than this is not a known-item query: the
#: gold document is one right answer among many, so recall@1 punishes a system
#: for returning a different legitimate match and measures noise.
#:
#: Measured 2026-09-18, which is why it is 5 and not 20. At a gate of 20 the
#: accepted rare-token queries had a median of 7 matching documents and 34 of
#: 60 matched more than 5. Carried over from
#: notes/probe_hybrid_retrieval.py, whose rare set used the same bound.
MAX_MATCHING_DOCUMENTS = 5

#: Chunk rows read when counting distinct matching documents. Generous enough
#: that the distinct count is exact for anything near the gate, capped so a
#: common term does not walk its whole posting list.
MATCH_PROBE_ROWS = 100

#: Word count bounds for a phrase query. Below three words it is a token query
#: (set A already covers those); above six it is effectively a quotation and
#: any lexical index finds it.
PHRASE_MIN_WORDS = 3
PHRASE_MAX_WORDS = 6

#: Words a phrase query may contain but must not begin or end with. A window
#: that starts on "the" or ends on "and" is a fragment, not something anyone
#: types. Only the edges are checked: dropping these from the middle is what
#: produced "function enable planning emerge while training" -- a real sentence
#: with its short words deleted, which measures nothing a user would ask for.
FUNCTION_WORDS = frozenset(
    """a an the and or but of in on at to for with from by as is are was were be been being
    this that these those it its we our their they he she his her which who whom whose what
    have has had do does did not no nor so if then than there here between among over under
    above below into onto up down out off again further more most other some such only own
    same too very can will just should now""".split()
)

#: Seed for the perturbed sets. Separate from SEED because `perturb` derives
#: from an existing query file rather than from the corpus, so it must stay
#: reproducible even when the corpus has moved under it.
PERTURB_SEED = 20260918

#: Shortest word `transpose_characters` will damage. Below four characters a
#: transposition is as likely to produce another real word as a typo.
TYPO_MIN_WORD = 4

#: The kinds `perturb` writes, derived from the `phrase` set. Listed so the
#: command can drop and regenerate its own output: running it twice must not
#: perturb the perturbations.
PERTURBED_KINDS = ("phrase-reordered", "phrase-dropped", "phrase-typo")


#: Where `arms` caches candidate lists. Outside the repo on purpose: it holds
#: filenames from a private library, and it is regenerable from the query file
#: in minutes.
DEFAULT_ARMS_CACHE = "~/.cache/cementic-lead-arms.json"

#: Rank-1 policies `ablate` scores, in the order printed. Every one fuses the
#: same two arm lists; they differ only in who owns position one.
#:
#:   shipped  `Searcher.lexical_should_lead`, recorded per query by `arms`.
#:   none     plain reciprocal rank fusion, nobody routed -- the configuration
#:            Phase 1 step 2 proposes deleting `lead` down to.
#:   vector   the vector arm always leads.
#:   lexical  the lexical arm always leads.
#:   oracle   whichever of vector-lead and lexical-lead ranks gold higher, per
#:            query. Not a system: the ceiling any rank-1 routing rule over these
#:            two arms can reach, and so the most that tuning `lead` could buy.
POLICIES = ("shipped", "none", "vector", "lexical", "oracle")

#: Where `compare-models` keeps its sample, per-model vectors and server logs.
#: Outside the repo for the same reason as the arms cache, and so a run that
#: dies after three models resumes at the fourth.
COMPARE_DIR = "~/.cache/cementic-model-compare"

#: Papers in the comparison. 500 rather than the v1.5 run's 1,000 because each
#: now contributes a whole span at two granularities, about 5M model tokens per
#: candidate; a paired sign test over 500 still resolves the v1.5 run's effect
#: size at p < 1e-4.
COMPARE_DOCUMENTS = 500

#: The stored chunking, which the span is laid out in. Must match the active
#: revision's chunk profile, or the stored vectors describe different text;
#: `compare_sample` checks the first span chunk of every paper against a
#: re-chunk of its artifact and refuses the run if many disagree.
STORED_CHUNK_SIZE = 320
STORED_CHUNK_OVERLAP = 80

#: Stored chunks skipped at the front of each paper, then taken as the span.
#: The title is one of the queries and sits verbatim on the title page, which
#: turned title-to-body into string matching in the v1.5 run (both models
#: scored a perfect recall@1 with those chunks in, measured).
SPAN_SKIP_CHUNKS = 2
SPAN_CHUNKS = 8

#: The long granularity: about what a 2,048-token window holds after the
#: model-token / tiktoken ratio, per the v1.5 analysis in PLAN.md. Overlap
#: keeps the stored ratio of one quarter.
LONG_CHUNK_SIZE = 1300
LONG_CHUNK_OVERLAP = 325

#: Instruction for the instruction-tuned models' queries. One fixed string for
#: every query kind, because production would ship one; tuning it per kind
#: would score a configuration nobody runs. It is the models' own default.
QWEN_INSTRUCTION = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
    "Query: "
)


@dataclass(frozen=True)
class ModelSpec:
    """How to serve and prompt one embedding model."""

    name: str
    gguf: str
    ctx: int
    pooling: str
    query_prefix: str = ""
    document_prefix: str = ""
    #: Nomic models take their prefixes from cementic's own policy, so this
    #: harness scores exactly the strings production embeds.
    nomic: bool = False


#: The candidates, keyed by the name results are reported under. The GGUFs
#: append their own end-of-text token (checked 2026-09-24 via /tokenize), so
#: no prefix here adds one.
MODELS: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in (
        ModelSpec("v2-moe", "nomic-embed-text-v2-moe.Q8_0.gguf", 512, "mean", nomic=True),
        ModelSpec(
            "qwen3-0.6b", "Qwen3-Embedding-0.6B-Q8_0.gguf", 4096, "last",
            query_prefix=QWEN_INSTRUCTION,
        ),
        ModelSpec(
            "harrier-0.6b", "harrier-oss-v1-0.6b.Q8_0.gguf", 4096, "last",
            query_prefix=QWEN_INSTRUCTION,
        ),
        ModelSpec(
            "gemma-300m", "embeddinggemma-300M-Q8_0.gguf", 2048, "mean",
            query_prefix="task: search result | query: ",
            document_prefix="title: none | text: ",
        ),
    )
}

#: The model the stored index was built with. Its fresh 320 arm is the noise
#: floor: same model, same text, only the batch composition differs.
BASELINE_MODEL = "v2-moe"


@dataclass(frozen=True)
class Query:
    """One scored query and the filename of the document that answers it."""

    query: str
    gold: str
    kind: str


# --------------------------------------------------------------------------
# Pure functions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmRecord:
    """One query's pre-fusion candidates, as document keys in arm order."""

    query: str
    gold: str
    kind: str
    vector: list[str]
    lexical: list[str]
    shipped_lead: str


def document_key(source_path: str) -> str:
    """Filename identifying a document, for storage and comparison (pure).

    Deliberately not the stored `source_path`. That is an absolute path inside
    a home directory, and this file is checked into git: it would publish a
    username, a machine's directory layout and the contents of a private
    library. The filename identifies the document just as well for scoring and
    keeps the query set readable -- which is what caught two defects in the
    phrase set on 2026-09-18, and a hash would not have.

    Assumes filenames are unique within a collection. `build` checks that and
    refuses to write a set where they are not, rather than scoring a gold label
    that matches two documents.
    """
    return source_path.rsplit("/", 1)[-1]


def metrics(ranked: list[str], gold: str) -> tuple[int, int, float]:
    """(recall@1, recall@10, reciprocal rank) for one query (pure)."""
    try:
        rank = ranked.index(gold) + 1
    except ValueError:
        return 0, 0, 0.0
    return int(rank == 1), int(rank <= 10), 1.0 / rank


def deduplicate(paths: list[str]) -> list[str]:
    """Drop repeats, keeping first position (pure).

    Search returns one row per chunk; a document is represented by its
    best-ranked chunk. Mirrors `hybrid.deduplicate`, not imported from it,
    because this harness must keep scoring a revision whose ranking code has
    since changed.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered


def title_from_path(source_path: str) -> str:
    """Filename stem as a natural-language title (pure).

    Strips a leading year prefix and turns separators into spaces, so
    ``2019-attention-is-all-you-need.pdf`` becomes ``attention is all you
    need``. Derived from the path rather than the text because a PDF's own
    title metadata is missing or wrong often enough to need judging, and
    judging is what this harness refuses to do.
    """
    stem = source_path.rsplit("/", 1)[-1].removesuffix(".pdf")
    return re.sub(r"^\d{4}-", "", stem).replace("-", " ").replace("_", " ").strip()


def phrase_candidates(chunk: str) -> list[str]:
    """Contiguous word runs from a chunk, longest first (pure).

    Drawn from within one sentence so the phrase reads as something a person
    might type, and kept verbatim -- short words included.

    Filtering short words out is the obvious implementation and it is wrong:
    it turns "enable planning to emerge while training" into "enable planning
    emerge while training", a string no user would type, so the set would
    measure retrieval of mangled pseudo-phrases. Function words are instead
    barred only at the window's edges, where they make a fragment.
    """
    out: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", chunk):
        words = re.findall(r"[A-Za-z][A-Za-z-]*", sentence)
        if len(words) < PHRASE_MAX_WORDS:
            continue
        for size in range(PHRASE_MAX_WORDS, PHRASE_MIN_WORDS - 1, -1):
            for start in range(0, len(words) - size + 1):
                window = words[start : start + size]
                if window[0].lower() in FUNCTION_WORDS or window[-1].lower() in FUNCTION_WORDS:
                    continue
                out.append(" ".join(window))
    return out


def has_function_word_edge(words: list[str]) -> bool:
    """Whether a word list begins or ends on a function word (pure).

    The rule `phrase_candidates` applies when drawing a window, factored out so
    the perturbations apply the same one to their output. A perturbation that
    leaves "the" at the edge has turned a phrase into a fragment, which is the
    defect the original rule exists to prevent.
    """
    return bool(words) and (
        words[0].lower() in FUNCTION_WORDS or words[-1].lower() in FUNCTION_WORDS
    )


def swap_adjacent_words(phrase: str, rng: random.Random) -> str | None:
    """Swap one adjacent pair of words (pure given `rng`); None if impossible.

    Models a user who remembers the words but not the order. Predicted to be a
    no-op for the lexical arm: `plainto_tsquery` builds an order-free AND of
    stemmed lexemes, so the set of matching documents is unchanged and only
    `ts_rank`'s ordering within it can move. That prediction is the reason to
    measure it -- if the lexical arm's phrase advantage survives reordering,
    the advantage is bag-of-words matching rather than phrase matching.
    """
    words = phrase.split()
    positions = list(range(len(words) - 1))
    rng.shuffle(positions)
    for i in positions:
        swapped = list(words)
        swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
        if swapped != words and not has_function_word_edge(swapped):
            return " ".join(swapped)
    return None


def drop_interior_word(phrase: str, rng: random.Random) -> str | None:
    """Delete one non-edge word (pure given `rng`); None if impossible.

    Models a user who half-remembers the phrase. Only interior words are
    eligible, so the result keeps the edges that made the original read as
    something a person types. Predicted to *grow* the lexical match set: one
    fewer term in the AND matches more documents, so the gold document stops
    being the only exact match and ranking has to do the work.
    """
    words = phrase.split()
    if len(words) < 3:
        return None
    positions = list(range(1, len(words) - 1))
    rng.shuffle(positions)
    for i in positions:
        kept = words[:i] + words[i + 1 :]
        if not has_function_word_edge(kept):
            return " ".join(kept)
    return None


def transpose_characters(phrase: str, rng: random.Random) -> str | None:
    """Transpose two adjacent characters in the longest word (pure given `rng`).

    Models a typo or a misremembered spelling. The longest word is chosen
    because it carries the most information: damaging "the" changes nothing a
    stemmer would not absorb. Predicted to be the harshest of the three --
    Postgres has no fuzzy matching here, so the damaged lexeme fails its half
    of the AND and the gold document can leave the candidate pool entirely,
    which would show up as recall@10 falling rather than recall@1.
    """
    words = phrase.split()
    candidates = sorted(
        (i for i, w in enumerate(words) if len(w) >= TYPO_MIN_WORD and w.isalpha()),
        key=lambda i: len(words[i]),
        reverse=True,
    )
    for i in candidates:
        word = words[i]
        offsets = [j for j in range(len(word) - 1) if word[j] != word[j + 1]]
        if not offsets:
            continue
        j = rng.choice(offsets)
        typo = word[:j] + word[j + 1] + word[j] + word[j + 2 :]
        damaged = list(words)
        damaged[i] = typo
        if not has_function_word_edge(damaged):
            return " ".join(damaged)
    return None


#: Perturbation per derived kind. A mapping rather than a chain of ifs so that
#: `perturb` reports coverage per kind without knowing what any of them do.
PERTURBATIONS = {
    "phrase-reordered": swap_adjacent_words,
    "phrase-dropped": drop_interior_word,
    "phrase-typo": transpose_characters,
}


def perturbed_queries(queries: list[Query]) -> list[Query]:
    """Derive the perturbed sets from the `phrase` set (pure).

    Paired by construction: every derived query keeps the gold label of the
    verbatim phrase it came from, so a drop in recall is attributable to the
    perturbation and not to having drawn a different sample of documents. That
    pairing is the whole design -- an independently drawn "imperfect phrase"
    set would confound the perturbation with the draw.

    Seeded per source phrase rather than per run, so the output does not depend
    on the order the sets are visited in.
    """
    out: list[Query] = []
    for kind, perturb in PERTURBATIONS.items():
        for query in queries:
            if query.kind != "phrase":
                continue
            rng = random.Random(f"{PERTURB_SEED}:{kind}:{query.query}")
            perturbed = perturb(query.query, rng)
            if perturbed is not None and perturbed != query.query:
                out.append(Query(query=perturbed, gold=query.gold, kind=kind))
    return out


def format_table(
    name: str, rows: dict[str, tuple[float, float, float]], n: int, depth: int
) -> str:
    """Render one set's scores (pure).

    The depth is in the heading because scores taken at different depths are
    not comparable, and a table that omits it invites exactly that comparison.
    """
    lines = [
        f"\n{name}  (n={n}, depth={depth})",
        f"  {'arm':<10} {'recall@1':>9} {'recall@10':>10} {'MRR':>7}",
    ]
    for arm, (r1, r10, mrr) in rows.items():
        lines.append(f"  {arm:<10} {r1:>9.3f} {r10:>10.3f} {mrr:>7.3f}")
    return "\n".join(lines)


def fused_ranking(
    vector: list[str], lexical: list[str], lead: str | None, depth: int
) -> list[str]:
    """The document ranking `search` returns for these arms and this lead (pure).

    ``lead=None`` is plain reciprocal rank fusion with nobody routed. Mirrors
    `Searcher._merge_arms` rather than calling it: with one arm empty that
    method returns the other arm's chunk rows cut to `depth` *before* collapsing
    to documents, so the fallback slices first here too. `ablate` checks this
    mirror against the live system by requiring the ``shipped`` row to
    reproduce `score` exactly.
    """
    if not vector or not lexical:
        return deduplicate((vector or lexical)[:depth])
    if lead is None:
        ordering = reciprocal_rank_fusion(deduplicate(vector), deduplicate(lexical))
    else:
        ordering = combine(vector, lexical, lead=lead)
    return ordering[:depth]


def gold_rank(ranking: list[str], gold: str) -> float:
    """1-based position of `gold`, or infinity when absent (pure)."""
    try:
        return float(ranking.index(gold) + 1)
    except ValueError:
        return float("inf")


def policy_ranking(record: ArmRecord, policy: str, depth: int) -> list[str]:
    """The ranking one rank-1 policy produces for one query (pure).

    ``oracle`` reads the gold label, which is what makes it a ceiling and not a
    system. On a tie it keeps vector-lead, the shipped default.
    """
    if policy == "oracle":
        options = [
            fused_ranking(record.vector, record.lexical, lead, depth)
            for lead in ("vector", "lexical")
        ]
        return min(options, key=lambda ranking: gold_rank(ranking, record.gold))
    leads: dict[str, str | None] = {
        "shipped": record.shipped_lead,
        "none": None,
        "vector": "vector",
        "lexical": "lexical",
    }
    if policy not in leads:
        raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")
    return fused_ranking(record.vector, record.lexical, leads[policy], depth)


def ablation(
    records: list[ArmRecord], depth: int
) -> dict[str, dict[str, tuple[float, float, float, int, int]]]:
    """Per set, per policy: (recall@1, recall@10, MRR, better, worse) (pure).

    ``better`` and ``worse`` count queries whose gold rank beats or trails the
    shipped policy's on the same query -- the paired comparison, which a
    difference of means over n=60 hides. Both are zero for ``shipped`` itself.
    """
    by_kind: dict[str, list[ArmRecord]] = {}
    for record in records:
        by_kind.setdefault(record.kind, []).append(record)

    out: dict[str, dict[str, tuple[float, float, float, int, int]]] = {}
    for kind, group in by_kind.items():
        shipped = [gold_rank(policy_ranking(r, "shipped", depth), r.gold) for r in group]
        rows: dict[str, tuple[float, float, float, int, int]] = {}
        for policy in POLICIES:
            totals = [0, 0, 0.0]
            better = worse = 0
            for record, base in zip(group, shipped):
                ranking = policy_ranking(record, policy, depth)
                r1, r10, rr = metrics(ranking, record.gold)
                totals[0] += r1
                totals[1] += r10
                totals[2] += rr
                rank = gold_rank(ranking, record.gold)
                better += rank < base
                worse += rank > base
            n = len(group)
            rows[policy] = (totals[0] / n, totals[1] / n, totals[2] / n, better, worse)
        out[kind] = rows
    return out


def format_ablation(
    kind: str, rows: dict[str, tuple[float, float, float, int, int]], n: int, depth: int
) -> str:
    """Render one set's ablation (pure)."""
    lines = [
        f"\n{kind}  (n={n}, depth={depth})",
        f"  {'policy':<8} {'recall@1':>9} {'recall@10':>10} {'MRR':>7} "
        f"{'better':>7} {'worse':>6}",
    ]
    for policy, (r1, r10, mrr, better, worse) in rows.items():
        lines.append(
            f"  {policy:<8} {r1:>9.3f} {r10:>10.3f} {mrr:>7.3f} {better:>7} {worse:>6}"
        )
    return "\n".join(lines)


def span_token_range(chunk_size: int, overlap: int, skip: int, count: int) -> tuple[int, int]:
    """Token interval covered by chunks ``skip .. skip+count-1`` (pure).

    `chunk_text` slides a `chunk_size` window by ``chunk_size - overlap``, so
    chunk ``i`` starts at ``i * step``.
    """
    step = chunk_size - overlap
    return skip * step, (skip + count - 1) * step + chunk_size


def format_for(spec: ModelSpec, text: str, *, query: bool) -> str:
    """The exact string a model embeds for a query or a document (pure)."""
    if spec.nomic:
        formatter = format_query_text_for_model if query else format_document_text_for_model
        return formatter(text, spec.gguf)
    return (spec.query_prefix if query else spec.document_prefix) + text


def gold_rank_and_margin(
    similarities: list[float], pool_documents: list[int], gold: int
) -> tuple[int, float]:
    """Rank of the gold document by its best chunk, and its margin (pure).

    A document is represented by its best chunk, as search collapses chunks to
    documents. Rank 1 is best; a gold document with no chunk in the pool ranks
    after every document that has one. The margin is gold's best similarity
    minus the best other document's -- it separates two models that both rank
    gold first, where recall cannot.
    """
    best: dict[int, float] = {}
    for similarity, document in zip(similarities, pool_documents, strict=True):
        if similarity > best.get(document, -math.inf):
            best[document] = similarity
    if gold not in best:
        return len(best) + 1, -math.inf
    mine = best.pop(gold)
    rank = 1 + sum(1 for other in best.values() if other > mine)
    margin = mine - max(best.values()) if best else 0.0
    return rank, margin


def sign_test(better: int, worse: int) -> float:
    """Two-sided exact binomial p-value for a paired win/loss count (pure).

    Ties are dropped, as in the v1.5 comparison this replaces.
    """
    n = better + worse
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(better, worse) + 1)) / 2**n
    return min(1.0, 2 * tail)


def summarise(ranks: list[int], margins: list[float]) -> dict[str, float]:
    """recall@1, recall@10, MRR and mean finite margin over one arm (pure)."""
    n = len(ranks)
    finite = [m for m in margins if math.isfinite(m)]
    return {
        "recall@1": sum(r == 1 for r in ranks) / n,
        "recall@10": sum(r <= 10 for r in ranks) / n,
        "MRR": sum(1 / r for r in ranks) / n,
        "margin": sum(finite) / len(finite) if finite else float("nan"),
    }


def paired(ranks: list[int], baseline: list[int]) -> tuple[int, int, float]:
    """(better, worse, sign-test p) of one arm against the baseline (pure)."""
    better = sum(a < b for a, b in zip(ranks, baseline, strict=True))
    worse = sum(a > b for a, b in zip(ranks, baseline, strict=True))
    return better, worse, sign_test(better, worse)


def paired_at_1(ranks: list[int], baseline: list[int]) -> tuple[int, int, float]:
    """(better, worse, sign-test p) on recall@1 alone (pure).

    A query counts only when exactly one arm ranks its gold first. `paired`
    counts any rank movement, so 8 -> 3 there is a win; the rider rule is
    pre-registered on recall@1, where that query is a tie.
    """
    better = sum(a == 1 and b != 1 for a, b in zip(ranks, baseline, strict=True))
    worse = sum(a != 1 and b == 1 for a, b in zip(ranks, baseline, strict=True))
    return better, worse, sign_test(better, worse)


def shared_pool(pools: list[list]) -> list[int]:
    """Indices every pool holds a vector for (pure).

    The pools are the arms of one granularity, row-aligned, with None where the
    model refused a chunk. Scoring each arm on its own surviving rows lets a
    refusal delete a distractor from one arm and not from the control; every
    arm has to be scored on the rows they all have.
    """
    return [i for i in range(len(pools[0])) if all(pool[i] is not None for pool in pools)]


#: Rider thresholds and tables. Policy, kept out of the functions.
#:
#: A chunk is punctuation-dominated when at least this share of its
#: whitespace-separated words contain no letter or digit (the dots of a
#: dot-leader table of contents). Measured on the 4,000 sample chunks: the
#: share is below 0.45 for 98% of them and above 0.73 for every chunk the
#: server refused, with 75 chunks at or above this value.
PUNCTUATION_WORD_SHARE = 0.5

#: Significance level of the rider rule, pre-registered 2026-10-23.
RIDER_P = 0.01

#: Name and cache file of the rider arm, next to the other models' caches.
RIDERS_ARM = "v2-moe+riders"

#: Where a structure-aware chunker may cut: a Markdown heading line, and a line
#: holding only form feeds (the page break PyMuPDF text output would write).
#: Form feeds inside a line are glyph noise from math fonts in this corpus and
#: are deliberately not a boundary.
HEADING_RE = re.compile(r"(?m)^#{1,6} \S")
PAGE_BREAK_RE = re.compile(r"(?m)^\f+[ \t]*\n")

#: A line-break hyphen: letters, the hyphen, then a break (newline or spaces)
#: and a continuation. `max- imizing` and `func-\ntion` are both this shape.
HYPHEN_BREAK_RE = re.compile(
    r"(?P<left>[^\W\d_]+)-(?:[ \t]*\n[ \t]*|[ \t]+)(?P<right>[^\W\d_]+)"
)
WORD_RE = re.compile(r"[^\W\d_]+")

#: Left fragments that start a real compound, so `well-\nknown` keeps its
#: hyphen. The continuation being a function word (`fixed-\nand`) keeps it too.
COMPOUND_PREFIXES = frozenset(
    "well self non cross high low long short large small real state end first "
    "second multi inter intra semi anti pre post sub".split()
)
#: ... except these, which join: `multi-\nvariate` is one word.
JOINING_PREFIXES = frozenset("multi inter intra semi anti pre post sub non".split())


def is_punctuation_dominated(chunk: str) -> bool:
    """Whether most of a chunk's words are bare punctuation (pure). 2b."""
    words = chunk.split()
    if not words:
        return True
    bare = sum(not any(ch.isalnum() for ch in word) for word in words)
    return bare / len(words) >= PUNCTUATION_WORD_SHARE


def dehyphenate(text: str) -> str:
    """Join words a PDF line break split with a hyphen (pure). 2e.

    `max- imizing` becomes `maximizing`, `func-\\ntion` becomes `function`.
    Stays out of real compounds with three signals, none of them a dictionary:
    the continuation is a function word (`fixed-\\nand random-effects`), the left
    fragment is a compound prefix (`well-\\nknown`), or the document spells the
    hyphenated form intact elsewhere. A soft hyphen is deleted outright.
    """
    text = text.replace("­", "")
    words = set(WORD_RE.findall(text))
    compounds = {
        (m.group("left"), m.group("right"))
        for m in re.finditer(r"(?P<left>[^\W\d_]+)-(?P<right>[^\W\d_]+)", text)
    }

    def join(match: re.Match[str]) -> str:
        left, right = match.group("left"), match.group("right")
        if not (left[-1].islower() and right[0].islower()) or len(left) < 2:
            return match.group(0)
        if right in FUNCTION_WORDS or (left, right) in compounds:
            return match.group(0)
        if left in COMPOUND_PREFIXES and left not in JOINING_PREFIXES and left + right not in words:
            return match.group(0)
        return left + right

    return HYPHEN_BREAK_RE.sub(join, text)


def structure_chunks(text: str, size: int, overlap: int) -> list[str]:
    """Chunks that start at headings and page breaks, windowed past `size` (pure). 2c.

    Each stretch between two boundaries is a chunk; one longer than `size`
    tokens falls back to `chunk_text`'s sliding window, which is also all that
    happens to text with no boundary at all.
    """
    cuts = sorted(
        {0, len(text)}
        | {m.start() for m in HEADING_RE.finditer(text)}
        | {m.end() for m in PAGE_BREAK_RE.finditer(text)}
    )
    encoding = tokenizer()
    chunks: list[str] = []
    for start, end in zip(cuts, cuts[1:], strict=False):
        segment = text[start:end]
        if not segment.strip():
            continue
        if len(encoding.encode(segment, disallowed_special=())) <= size:
            chunks.append(segment)
        else:
            chunks.extend(c.content for c in chunk_text(segment, size, overlap))
    return chunks


def rider_chunks(span: str, size: int, overlap: int) -> list[str]:
    """The span as the riders would chunk it: 2e, then 2c, then 2b (pure)."""
    chunks = structure_chunks(dehyphenate(span), size, overlap)
    return [c for c in chunks if not is_punctuation_dominated(c)]


def reconstruct_span(long_chunks: list[str]) -> str:
    """The span text back from its overlapping 1,300-token chunks (pure).

    `compare_sample` stores the span only as those chunks. Each next chunk
    starts inside the previous one, so the join is where its opening lands.
    """
    span = long_chunks[0]
    for chunk in long_chunks[1:]:
        at = span.rfind(chunk[:200])
        span = span[:at] + chunk if at >= 0 else span + chunk
    return span


#: Fuzzy-matching candidates (PLAN.md "Fuzzy matching"). pg_trgm's defaults for
#: `similarity_threshold` and `word_similarity_threshold`, and the share of a
#: query's terms the relaxed full-text match must hit.
TRIGRAM_SIMILARITY = 0.3
TRIGRAM_WORD_SIMILARITY = 0.6
RELAXED_TERM_SHARE = 0.6

#: Sets the fuzzy measurement reads by default: the exact phrases, their
#: perturbations, and the typo'd set the fallbacks exist for.
FUZZY_KINDS = ("phrase", "phrase-reordered", "phrase-dropped", "phrase-typo")

#: Candidate lexical arms, in report order. Each is a rule over a record's
#: stored arms, so adding one is a table row.
FUZZY_VARIANTS = ("shipped", "trigram similarity", "trigram word_similarity", "relaxed OR")


@dataclass(frozen=True)
class FuzzyRecord:
    """One query's baseline arms and each candidate lexical arm (document keys)."""

    query: str
    gold: str
    kind: str
    vector: list[str]
    lexical: list[str]
    shipped_lead: str
    similarity: list[str]
    word_similarity: list[str]
    relaxed: list[str]


def quote_lexeme(lexeme: str) -> str:
    """A lexeme as a quoted tsquery literal (pure)."""
    return "'" + lexeme.replace("'", "''") + "'"


def relaxed_tsquery(lexemes: list[str], share: float) -> str | None:
    """A tsquery matching any ``ceil(share * n)`` of the n lexemes (pure).

    An OR of every conjunction of that size, so the GIN index serves it and a
    row must still hold most of the terms: a bare OR of all terms matches every
    chunk containing one common word. Lexemes are quoted for a ``::tsquery``
    cast, which takes them as already stemmed.
    """
    terms = list(dict.fromkeys(lexemes))
    if not terms:
        return None
    need = max(1, math.ceil(share * len(terms) - 1e-9))
    return " | ".join(
        "(" + " & ".join(quote_lexeme(t) for t in combo) + ")"
        for combo in combinations(terms, need)
    )


def fuzzy_lexical(record: FuzzyRecord, variant: str) -> list[str]:
    """The lexical arm a variant would hand to fusion for this query (pure).

    The trigram variants are fallbacks and only replace an empty arm; the
    relaxed match is the arm itself, for every query.
    """
    if variant == "shipped":
        return record.lexical
    if variant == "relaxed OR":
        return record.relaxed
    fallback = record.similarity if variant == "trigram similarity" else record.word_similarity
    return record.lexical or fallback


def fuzzy_report(
    records: list[FuzzyRecord], depth: int
) -> dict[str, dict[str, tuple[float, float, float, int, int, int]]]:
    """Per kind, per variant: recall@1, recall@10, MRR, better, worse, changed (pure).

    ``better``/``worse`` pair each query's gold rank against the shipped arm's;
    ``changed`` counts queries whose ranking differs at all, so a candidate that
    never fires is visible as zero rather than as an equal score.
    """
    by_kind: dict[str, list[FuzzyRecord]] = {}
    for record in records:
        by_kind.setdefault(record.kind, []).append(record)
    out: dict[str, dict[str, tuple[float, float, float, int, int, int]]] = {}
    for kind, group in by_kind.items():
        base = [
            fused_ranking(r.vector, r.lexical, r.shipped_lead, depth) for r in group
        ]
        rows = {}
        for variant in FUZZY_VARIANTS:
            totals = [0.0, 0.0, 0.0]
            better = worse = changed = 0
            for record, shipped in zip(group, base, strict=True):
                ranking = fused_ranking(
                    record.vector, fuzzy_lexical(record, variant), record.shipped_lead, depth
                )
                r1, r10, rr = metrics(ranking, record.gold)
                totals[0] += r1
                totals[1] += r10
                totals[2] += rr
                a, b = gold_rank(ranking, record.gold), gold_rank(shipped, record.gold)
                better += a < b
                worse += a > b
                changed += ranking != shipped
            n = len(group)
            rows[variant] = (totals[0] / n, totals[1] / n, totals[2] / n, better, worse, changed)
        out[kind] = rows
    return out


def format_fuzzy(report: dict, counts: dict[str, int], depth: int) -> str:
    """Render `fuzzy_report` (pure)."""
    lines = []
    for kind, rows in report.items():
        lines.append(f"\n{kind}  (n={counts[kind]}, depth={depth})")
        lines.append(
            f"  {'lexical arm':<24} {'recall@1':>9} {'recall@10':>10} {'MRR':>7}"
            f" {'better':>7} {'worse':>6} {'changed':>8}"
        )
        for variant, (r1, r10, mrr, better, worse, changed) in rows.items():
            lines.append(
                f"  {variant:<24} {r1:>9.3f} {r10:>10.3f} {mrr:>7.3f}"
                f" {better:>7} {worse:>6} {changed:>8}"
            )
    return "\n".join(lines)


def rider_verdict(title: tuple[int, int, float], phrase: tuple[int, int, float]) -> bool:
    """The pre-registered rider rule over two recall@1 results (pure).

    Each result is `paired_at_1`'s (better, worse, p) against the control.
    Pass when one kind gains (more recall@1 wins than losses, sign test
    p < 0.01) and the other does not lose (no more recall@1 losses than wins).
    """
    def gains(result: tuple[int, int, float]) -> bool:
        return result[0] > result[1] and result[2] < RIDER_P

    def loses(result: tuple[int, int, float]) -> bool:
        return result[1] > result[0]

    return (gains(title) and not loses(phrase)) or (gains(phrase) and not loses(title))


def rider_census(spans: list[str]) -> dict[str, int]:
    """How much each rider has to act on across the spans (pure)."""
    census = {
        "spans": len(spans),
        "markdown heading lines": sum(len(HEADING_RE.findall(s)) for s in spans),
        "spans with a heading": sum(bool(HEADING_RE.search(s)) for s in spans),
        "page-break lines": sum(len(PAGE_BREAK_RE.findall(s)) for s in spans),
        "form feeds anywhere": sum(s.count("\f") for s in spans),
        "line-break hyphens": sum(len(HYPHEN_BREAK_RE.findall(s)) for s in spans),
        "line-break hyphens joined": sum(
            len(HYPHEN_BREAK_RE.findall(s)) - len(HYPHEN_BREAK_RE.findall(dehyphenate(s)))
            for s in spans
        ),
        "soft hyphens": sum(s.count("­") for s in spans),
    }
    return census


# --------------------------------------------------------------------------
# Corpus probes (shell)
# --------------------------------------------------------------------------


def matching_document_ids(conn, query: str, limit: int) -> list[int]:
    """Document ids of up to `limit` chunks matching `query`, GIN-driven.

    Deliberately *not* ``select distinct document_id ... limit n``. Measured
    2026-09-06 (notes/probe_hybrid_retrieval.py): that form makes the planner
    walk the document index in document order and apply the text match as a
    filter, never touching the GIN index -- 8 s+ for a token that matches
    nothing. Letting the text predicate drive an inner LIMIT gives a bitmap
    index scan instead.
    """
    rows = conn.execute(
        text(
            "select document_id from ("
            "  select document_id from chunks_v2 "
            f"  where to_tsvector('{LEXICAL_TEXT_CONFIG}', content) "
            f"        @@ plainto_tsquery('{LEXICAL_TEXT_CONFIG}', :q) "
            "  limit :lim"
            ") c"
        ),
        {"q": query, "lim": limit},
    ).fetchall()
    return [row[0] for row in rows]


def is_known_item(conn, query: str, gold_id: int) -> bool:
    """Whether `query` identifies `gold_id` narrowly enough to be scorable.

    Two conditions, both necessary. The match set must be small, or the query
    has many right answers. And the gold document must be in it, or the query
    is unanswerable and would score every system zero -- which looks like a
    finding and is an artefact of the query set.
    """
    distinct = set(matching_document_ids(conn, query, MATCH_PROBE_ROWS))
    return 0 < len(distinct) <= MAX_MATCHING_DOCUMENTS and gold_id in distinct


def sample_documents(conn, collection: str, rng: random.Random) -> list[tuple[int, str]]:
    """Every live document in the collection, shuffled."""
    docs = conn.execute(
        text(
            "select id, source_path from source_documents "
            "where collection = :col and status <> 'deleted'"
        ),
        {"col": collection},
    ).fetchall()
    docs = [(row[0], row[1]) for row in docs]
    rng.shuffle(docs)
    return docs


def build_rare_token_set(conn, docs, n: int) -> list[Query]:
    """Set A: a rare capitalised token, answered by the document it came from.

    Candidates are sorted longest-first. Measured 2026-09-06: "Hochreiter"
    appears in 1,697 documents and "Kalman" in 1,051, so ordinary capitalised
    words are nowhere near rare enough and testing them in random order wastes
    the budget. Long tokens are where names, identifiers and equation labels
    are.
    """
    out: list[Query] = []
    tried = 0
    for doc_id, source_path in docs:
        if len(out) >= n:
            break
        chunks = conn.execute(
            text(
                "select content from chunks_v2 where document_id = :d "
                "order by chunk_index limit 8"
            ),
            {"d": doc_id},
        ).fetchall()
        candidates: set[str] = set()
        for (chunk,) in chunks:
            candidates.update(t for t in TOKEN_RE.findall(chunk) if t not in STOPWORDS)
        for token in sorted(candidates, key=len, reverse=True)[:10]:
            tried += 1
            if is_known_item(conn, token, doc_id):
                out.append(Query(query=token, gold=document_key(source_path), kind="rare-token"))
                break
    print(f"  rare-token: tried {tried} candidates -> accepted {len(out)}", flush=True)
    return out


def build_title_set(conn, docs, n: int) -> list[Query]:
    """Set B: the document's own title, answered by that document.

    No corpus probe: a title is a known-item query by construction. The
    four-word floor drops filenames that are bare identifiers.
    """
    out: list[Query] = []
    for _doc_id, source_path in docs:
        if len(out) >= n:
            break
        title = title_from_path(source_path)
        if len(title.split()) >= 4:
            out.append(Query(query=title, gold=document_key(source_path), kind="title"))
    print(f"  title: accepted {len(out)}", flush=True)
    return out


def build_phrase_set(conn, docs, n: int) -> list[Query]:
    """Set E: a multi-word phrase from the body, answered by its own document.

    The bucket nothing has measured. Set A asks for a token a lexical index
    always wins; set B asks for a title a vector index always wins. A phrase
    sits between them, which is where `ts_rank`'s ranking quality -- as opposed
    to its recall -- should first become visible.
    """
    out: list[Query] = []
    tried = 0
    for doc_id, source_path in docs:
        if len(out) >= n:
            break
        chunks = conn.execute(
            text(
                "select content from chunks_v2 where document_id = :d "
                "order by chunk_index offset 2 limit 6"
            ),
            {"d": doc_id},
        ).fetchall()
        candidates: list[str] = []
        for (chunk,) in chunks:
            candidates.extend(phrase_candidates(chunk))
        for phrase in candidates[:12]:
            tried += 1
            if is_known_item(conn, phrase, doc_id):
                out.append(Query(query=phrase, gold=document_key(source_path), kind="phrase"))
                break
    print(f"  phrase: tried {tried} candidates -> accepted {len(out)}", flush=True)
    return out


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def command_build(args: argparse.Namespace) -> int:
    config = get_config()
    engine = get_engine(config.database.url)
    rng = random.Random(SEED)
    t0 = time.perf_counter()

    with engine.connect() as conn:
        docs = sample_documents(conn, args.collection, rng)
        if not docs:
            print(f"no live documents in collection {args.collection!r}", file=sys.stderr)
            return 1
        print(f"corpus: {len(docs)} live documents in {args.collection!r}", flush=True)
        # The gold label is a filename, so two documents sharing one makes a
        # gold that matches both and silently inflates every score. Refuse
        # rather than write a set nobody can trust.
        keys = [document_key(path) for _id, path in docs]
        if len(set(keys)) != len(keys):
            dupes = sorted({k for k in keys if keys.count(k) > 1})
            print(
                f"refusing to build: {len(dupes)} filename(s) are not unique in "
                f"{args.collection!r}, e.g. {dupes[:3]}",
                file=sys.stderr,
            )
            return 1
        queries = (
            build_rare_token_set(conn, docs, args.n)
            + build_title_set(conn, docs, args.n)
            + build_phrase_set(conn, docs, args.n)
        )

    payload = {
        "collection": args.collection,
        "seed": SEED,
        # Recorded for provenance only. `score --depth` is what decides the
        # depth a run measures at, because one query set is scored at several.
        "depth": DEFAULT_DEPTH,
        "queries": [asdict(q) for q in queries],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"\nwrote {len(queries)} queries to {out} in {time.perf_counter() - t0:.0f}s",
        flush=True,
    )
    return 0


def command_perturb(args: argparse.Namespace) -> int:
    """Add the perturbed sets to an existing query file, in place.

    No corpus access: this rewrites queries that are already checked in, which
    is what keeps the verbatim sets byte-identical across the edit. It is
    idempotent -- the derived kinds are dropped and regenerated, so running it
    twice does not perturb the perturbations.
    """
    path = Path(args.queries)
    payload = json.loads(path.read_text())
    queries = [Query(**q) for q in payload["queries"]]

    original = [q for q in queries if q.kind not in PERTURBED_KINDS]
    derived = perturbed_queries(original)
    sources = sum(1 for q in original if q.kind == "phrase")
    if not sources:
        print(f"no 'phrase' queries in {path}; nothing to derive from", file=sys.stderr)
        return 1

    for kind in PERTURBED_KINDS:
        got = sum(1 for q in derived if q.kind == kind)
        print(f"  {kind}: {got}/{sources} derived", flush=True)

    payload["perturb_seed"] = PERTURB_SEED
    payload["queries"] = [asdict(q) for q in original + derived]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {len(payload['queries'])} queries to {path}", flush=True)
    return 0


def command_score(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.queries).read_text())
    queries = [Query(**q) for q in payload["queries"]]
    collection = payload["collection"]

    config = get_config()
    searcher = Searcher(config)
    t0 = time.perf_counter()

    by_kind: dict[str, list[Query]] = {}
    for query in queries:
        by_kind.setdefault(query.kind, []).append(query)

    overall: dict[str, tuple[float, float, float]] = {}
    for kind, group in by_kind.items():
        totals = [0, 0, 0.0]
        for i, query in enumerate(group, start=1):
            ranked = deduplicate(
                [
                    document_key(r["source_path"])
                    for r in searcher.search(
                        query.query, top_k=args.depth, collections=[collection]
                    )
                ]
            )
            r1, r10, mrr = metrics(ranked, query.gold)
            totals[0] += r1
            totals[1] += r10
            totals[2] += mrr
            if i % 25 == 0:
                print(f"  {kind}: {i}/{len(group)} ({time.perf_counter() - t0:.0f}s)", flush=True)
        n = len(group)
        overall[kind] = (totals[0] / n, totals[1] / n, totals[2] / n)
        print(format_table(kind, {"search": overall[kind]}, n, args.depth), flush=True)

    print(f"\ntotal {time.perf_counter() - t0:.0f}s at depth {args.depth}", flush=True)
    if args.json:
        print(json.dumps({"depth": args.depth,
                          "sets": {k: {"recall@1": v[0], "recall@10": v[1], "mrr": v[2]}
                                   for k, v in overall.items()}}, indent=2, sort_keys=True))
    return 0


def command_arms(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.queries).read_text())
    queries = [Query(**q) for q in payload["queries"]]
    collection = payload["collection"]

    config = get_config()
    if not config.search.hybrid:
        print("search.hybrid is off: there is no lexical arm to ablate", file=sys.stderr)
        return 1
    searcher = Searcher(config)
    arm_limit = arm_fetch_limit(PRODUCTION_DEPTH, hybrid=True)
    t0 = time.perf_counter()

    records: list[dict] = []
    for i, query in enumerate(queries, start=1):
        candidates = searcher.candidates(
            query.query, top_k=PRODUCTION_DEPTH, collections=[collection]
        )
        vector, lexical = split_arms(candidates)
        records.append(
            asdict(
                ArmRecord(
                    query=query.query,
                    gold=query.gold,
                    kind=query.kind,
                    vector=[document_key(r["source_path"]) for r in vector],
                    lexical=[document_key(r["source_path"]) for r in lexical],
                    shipped_lead=(
                        "lexical" if searcher.lexical_should_lead(query.query) else "vector"
                    ),
                )
            )
        )
        if i % 30 == 0:
            print(f"  {i}/{len(queries)} ({time.perf_counter() - t0:.0f}s)", flush=True)

    out = Path(args.out).expanduser()
    out.write_text(
        json.dumps(
            {"queries": args.queries, "arm_limit": arm_limit, "records": records}, indent=1
        )
    )
    print(f"wrote {len(records)} records to {out} in {time.perf_counter() - t0:.0f}s")
    return 0


def command_ablate(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.arms).expanduser().read_text())
    # The cached lists are exactly what `search` fuses only at a depth whose
    # per-arm fetch limit is the one they were fetched at.
    if arm_fetch_limit(args.depth, hybrid=True) != payload["arm_limit"]:
        print(
            f"depth {args.depth} fetches {arm_fetch_limit(args.depth, hybrid=True)} per arm; "
            f"the cache holds {payload['arm_limit']}. Re-run `arms` for this depth.",
            file=sys.stderr,
        )
        return 1
    records = [ArmRecord(**r) for r in payload["records"]]
    for kind, rows in ablation(records, args.depth).items():
        n = sum(1 for r in records if r.kind == kind)
        print(format_ablation(kind, rows, n, args.depth))
    leads = sum(1 for r in records if r.shipped_lead == "lexical")
    print(f"\nshipped routing gave rank 1 to the lexical arm for {leads}/{len(records)} queries")
    return 0


def compare_sample(conn, collection: str, n: int, rng: random.Random) -> dict:
    """Draw papers, their stored span chunks and vectors, and the queries.

    Returns plain JSON-able data so the sample is written once and every model,
    including one run days later, embeds exactly the same text.
    """
    revision = active_revision(conn, collection)
    table = vector_table_name(int(revision.embedding_profile_id))
    candidates = conn.execute(
        text(
            "select sd.id, sd.source_path, ed.artifact_path from source_documents sd "
            "join extracted_documents ed on ed.document_id = sd.id "
            "and ed.extractor_profile_id = :ex and ed.status = 'done' "
            "where sd.collection = :c and sd.status <> 'deleted' "
            "and ed.artifact_path is not null"
        ),
        {"c": collection, "ex": revision.extractor_profile_id},
    ).fetchall()
    candidates = sorted(tuple(row) for row in candidates)
    rng.shuffle(candidates)

    first, last = SPAN_SKIP_CHUNKS, SPAN_SKIP_CHUNKS + SPAN_CHUNKS - 1
    start, end = span_token_range(
        STORED_CHUNK_SIZE, STORED_CHUNK_OVERLAP, SPAN_SKIP_CHUNKS, SPAN_CHUNKS
    )
    encoding = tokenizer()
    papers: list[dict] = []
    misaligned = 0
    for document_id, source_path, artifact in candidates:
        if len(papers) == n:
            break
        rows = conn.execute(
            text(
                "select c.chunk_index, c.content, ev.embedding::text as vector "
                f"from {table} ev join chunks_v2 c on c.id = ev.chunk_id "
                "where c.document_id = :d and ev.collection = :c "
                "and ev.chunk_profile_id = :cp and ev.extractor_profile_id = :ex "
                "and c.chunk_index between :first and :last order by c.chunk_index"
            ),
            {
                "d": document_id, "c": collection, "cp": revision.chunk_profile_id,
                "ex": revision.extractor_profile_id, "first": first, "last": last,
            },
        ).fetchall()
        if [r.chunk_index for r in rows] != list(range(first, last + 1)):
            continue
        try:
            body = read_extracted_text(Path(artifact))
        except OSError:
            continue
        # Encoded exactly as `chunk_text` encodes, special-token spellings
        # included as plain text, so the span's token offsets are its offsets.
        tokens = encoding.encode(body, disallowed_special=())
        if len(tokens) < end:
            continue
        # The stored chunks must be what chunking this artifact produces now,
        # or the control arm scores vectors of different text.
        rechunked = chunk_text(body, STORED_CHUNK_SIZE, STORED_CHUNK_OVERLAP)
        if rechunked[first].content != rows[0].content:
            misaligned += 1
            continue
        phrases = phrase_candidates(rows[SPAN_CHUNKS // 2].content)
        # Byte-level decode; a character cut at either edge is dropped rather
        # than rendered as U+FFFD.
        span = encoding.decode_bytes(tokens[start:end]).decode("utf-8", "ignore")
        papers.append(
            {
                "id": document_id,
                "title": title_from_path(source_path),
                "phrase": rng.choice(phrases) if phrases else None,
                "short": [r.content for r in rows],
                "stored": [[float(v) for v in r.vector.strip("[]").split(",")] for r in rows],
                "long": [c.content for c in chunk_text(span, LONG_CHUNK_SIZE, LONG_CHUNK_OVERLAP)],
            }
        )
    if misaligned > max(len(papers) // 20, 5):
        raise SystemExit(
            f"{misaligned} papers' stored chunks differ from a re-chunk of their artifact; "
            "the chunk profile no longer matches STORED_CHUNK_SIZE/OVERLAP"
        )
    return {"collection": collection, "papers": papers, "misaligned": misaligned}


def server_argv(config, spec: ModelSpec, port: int, models_dir: Path) -> list[str]:
    """The live daemon's binary and environment, serving `spec` on `port`.

    Taken from the configured `daemon_command` up to the executable, so every
    model runs on the build the corpus was indexed with.
    """
    prefix: list[str] = []
    for part in config.llama_cpp.daemon_command:
        prefix.append(part)
        if part.endswith("llama-server"):
            break
    else:
        prefix = ["llama-server"]
    return prefix + [
        "--host", "127.0.0.1", "--port", str(port),
        "--model", str(models_dir / spec.gguf), "--embeddings",
        "--pooling", spec.pooling,
        "--ctx-size", str(spec.ctx), "--batch-size", str(spec.ctx),
        "--ubatch-size", str(spec.ctx), "--n-gpu-layers", "99",
    ]


def _post_embeddings(url: str, texts: list[str]) -> tuple[list[list[float]], int]:
    request = urllib.request.Request(
        f"{url}/v1/embeddings",
        data=json.dumps({"input": texts}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        payload = json.load(response)
    rows = sorted(payload["data"], key=lambda r: r.get("index", 0))
    used = int(payload.get("usage", {}).get("prompt_tokens", 0))
    return [row["embedding"] for row in rows], used


def embed_all(
    url: str, texts: list[str], label: str
) -> tuple[list[list[float] | None], int, float]:
    """Vectors (None where refused), prompt tokens, and seconds spent."""
    out: list[list[float] | None] = []
    tokens, began = 0, time.perf_counter()
    for i in range(0, len(texts), 16):
        batch = texts[i : i + 16]
        try:
            vectors, used = _post_embeddings(url, batch)
            out.extend(vectors)
            tokens += used
        except Exception:
            for one in batch:
                try:
                    vectors, used = _post_embeddings(url, [one])
                    out.extend(vectors)
                    tokens += used
                except Exception:
                    out.append(None)
        if (i // 16) % 25 == 24:
            elapsed = time.perf_counter() - began
            print(
                f"    {label}: {len(out)}/{len(texts)} ({elapsed:.0f}s, "
                f"{tokens / elapsed:.0f} tok/s)",
                flush=True,
            )
    return out, tokens, time.perf_counter() - began


@contextlib.contextmanager
def serving(config, spec: ModelSpec, out_dir: Path, port: int) -> Iterator[str]:
    """Run `spec`'s llama-server for the block and yield its base URL."""
    models_dir = Path(config.llama_cpp.model_path).expanduser().parent
    log = out_dir / f"server-{spec.name}.log"
    with log.open("w") as handle:
        server = subprocess.Popen(
            server_argv(config, spec, port, models_dir), stdout=handle, stderr=subprocess.STDOUT
        )
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(180):
            try:
                with urllib.request.urlopen(f"{url}/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except Exception:
                pass
            if server.poll() is not None:
                raise SystemExit(f"{spec.name}: server exited; see {log}")
            time.sleep(1)
        else:
            raise SystemExit(f"{spec.name}: server not ready in 180s; see {log}")
        yield url
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()


def run_model(config, spec: ModelSpec, sample: dict, out_dir: Path, port: int) -> None:
    """Embed every text this model needs, resuming from whatever is cached.

    Each finished step is written to a partial file at once, so a run killed
    mid-model -- one was, by memory pressure from other work on the machine --
    loses at most the step in flight. The final file appears only when every
    step is done, and it is what scoring reads.
    """
    cache = out_dir / f"{spec.name}.json"
    if cache.exists():
        print(f"  {spec.name}: cached", flush=True)
        return
    partial = out_dir / f"{spec.name}.partial.json"
    result: dict = (
        json.loads(partial.read_text()) if partial.exists()
        else {"model": spec.name, "gguf": spec.gguf}
    )
    with serving(config, spec, out_dir, port) as url:
        papers = sample["papers"]
        for kind in ("title", "phrase"):
            if f"q_{kind}" in result:
                continue
            texts = [format_for(spec, p[kind], query=True) for p in papers if p[kind]]
            result[f"q_{kind}"], _, _ = embed_all(url, texts, f"{spec.name} {kind} queries")
            partial.write_text(json.dumps(result))
        # A 512-token window cannot hold the long chunks; the stored model only
        # ever runs at the stored granularity.
        granularities = ("short",) if spec.ctx < 1024 else ("short", "long")
        for granularity in granularities:
            if granularity in result:
                print(f"  {spec.name} {granularity}: cached", flush=True)
                continue
            texts = [format_for(spec, c, query=False) for p in papers for c in p[granularity]]
            vectors, tokens, seconds = embed_all(url, texts, f"{spec.name} {granularity}")
            result[granularity] = vectors
            result[f"{granularity}_tokens"] = tokens
            result[f"{granularity}_seconds"] = seconds
            refused = sum(v is None for v in vectors)
            print(
                f"  {spec.name} {granularity}: {len(texts)} chunks, {tokens} tokens in "
                f"{seconds:.0f}s ({tokens / seconds:.0f} tok/s), {refused} refused",
                flush=True,
            )
            partial.write_text(json.dumps(result))
        partial.rename(cache)



def score_arm(
    np, queries, query_docs, pool, pool_docs, keep: list[int] | None = None
) -> tuple[list[int], list[float]]:
    """Gold rank and margin per query for one (queries, pool) pair.

    `keep` is the rows to score. Arms that are compared with each other must be
    given the same rows (`shared_pool`): the default, each arm's own non-refused
    rows, is right only for an arm compared with nothing of the same chunking.
    """
    if keep is None:
        keep = [i for i, v in enumerate(pool) if v is not None]
    matrix = np.array([pool[i] for i in keep], dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    documents = [pool_docs[i] for i in keep]
    ranks, margins = [], []
    for vector, gold in zip(queries, query_docs, strict=True):
        if vector is None:
            ranks.append(len(set(documents)) + 1)
            margins.append(-math.inf)
            continue
        q = np.asarray(vector, dtype=np.float32)
        similarities = (matrix @ (q / np.linalg.norm(q))).tolist()
        rank, margin = gold_rank_and_margin(similarities, documents, gold)
        ranks.append(rank)
        margins.append(margin)
    return ranks, margins


def command_compare_models(args: argparse.Namespace) -> int:
    try:
        import numpy as np
    except ImportError:
        print("compare-models needs numpy in the environment running it", file=sys.stderr)
        return 1
    names = args.models.split(",")
    if BASELINE_MODEL not in names:
        print(f"--models must include {BASELINE_MODEL}, the control's query side", file=sys.stderr)
        return 1
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    config = get_config()
    sample_path = out_dir / "sample.json"
    if not sample_path.exists():
        with get_engine(config.database.url).connect() as conn:
            sample = compare_sample(conn, args.collection, args.documents, random.Random(SEED))
        sample_path.write_text(json.dumps(sample))
    sample = json.loads(sample_path.read_text())
    papers = sample["papers"]
    digest = hashlib.sha256(sample_path.read_bytes()).hexdigest()[:12]
    print(
        f"sample {digest}: {len(papers)} papers, {sample['misaligned']} misaligned skipped",
        flush=True,
    )

    for name in names:
        run_model(config, MODELS[name], sample, out_dir, args.port)

    short_docs = [p["id"] for p in papers for _ in p["short"]]
    long_docs = [p["id"] for p in papers for _ in p["long"]]
    stored = [v for p in papers for v in p["stored"]]
    loaded = {name: json.loads((out_dir / f"{name}.json").read_text()) for name in names}
    # Every arm of a granularity is scored on the rows all of them embedded. A
    # refusal used to delete a chunk from the fresh arm's pool only, and the
    # chunks the server refuses are dot-leader tables of contents that sit in
    # the stored index as hubs outranking the gold document: 21 refusals made
    # fresh v2-moe "beat" stored on 99 phrase queries it merely never saw them on.
    short_keep = shared_pool([stored] + [r["short"] for r in loaded.values()])
    long_keep = shared_pool([r["long"] for r in loaded.values() if "long" in r])
    lines = [
        f"\nscored on shared pools: {len(short_keep)}/{len(short_docs)} chunks at 320, "
        f"{len(long_keep)}/{len(long_docs)} at 1300"
    ]
    for kind in ("title", "phrase"):
        query_docs = [p["id"] for p in papers if p[kind]]
        # The control pairs fresh v2-moe queries with the stored index vectors,
        # which is what production search does.
        arms = {
            "v2-moe stored@320": score_arm(
                np, loaded[BASELINE_MODEL][f"q_{kind}"], query_docs, stored, short_docs, short_keep
            )
        }
        for name, result in loaded.items():
            arms[f"{name}@320"] = score_arm(
                np, result[f"q_{kind}"], query_docs, result["short"], short_docs, short_keep
            )
            if "long" in result:
                arms[f"{name}@1300"] = score_arm(
                    np, result[f"q_{kind}"], query_docs, result["long"], long_docs, long_keep
                )
        baseline = arms["v2-moe stored@320"][0]
        lines.append(f"\n{kind} -> body  (n={len(query_docs)}, pool {len(papers)} papers)")
        lines.append(
            f"  {'arm':<22} {'recall@1':>9} {'recall@10':>10} {'MRR':>7} {'margin':>8}"
            f" {'better':>7} {'worse':>6} {'sign p':>9}"
        )
        for arm, (ranks, margins) in arms.items():
            m = summarise(ranks, margins)
            better, worse, p = paired(ranks, baseline)
            lines.append(
                f"  {arm:<22} {m['recall@1']:>9.3f} {m['recall@10']:>10.3f} "
                f"{m['MRR']:>7.3f} {m['margin']:>8.4f} {better:>7} {worse:>6} {p:>9.2g}"
            )
    lines.append("\nthroughput through llama-server (same machine, sequential runs)")
    for name, result in loaded.items():
        for granularity in ("short", "long"):
            if f"{granularity}_tokens" in result:
                tokens = result[f"{granularity}_tokens"]
                seconds = result[f"{granularity}_seconds"]
                refused = sum(v is None for v in result[granularity])
                lines.append(
                    f"  {name:<14} {granularity:<6} {tokens:>9} tokens {seconds:>7.0f}s "
                    f"{tokens / seconds:>6.0f} tok/s  {refused} refused"
                )
    print("\n".join(lines))
    return 0


def active_revision(conn, collection: str):
    """The collection's active pipeline revision row."""
    return conn.execute(
        text(
            "select extractor_profile_id, chunk_profile_id, embedding_profile_id "
            "from pipeline_revisions where collection = :c and status = 'active'"
        ),
        {"c": collection},
    ).one()


#: Rows the trigram probe scans to extrapolate a full-corpus scan from.
TRIGRAM_PROBE_ROWS = 20_000


def fuzzy_scope(table: str) -> str:
    """The join and filters that keep a lexical candidate inside the searched rows.

    The same scope as `lexical_sql`: the per-profile vector table's own columns.
    """
    return (
        f"JOIN {table} ev ON ev.chunk_id = c.id "
        "JOIN source_documents sd ON sd.id = c.document_id "
        "WHERE ev.collection = :collection "
        "AND ev.chunk_profile_id = :chunk_profile_id "
        "AND ev.extractor_profile_id = :extractor_profile_id "
    )


def trigram_arms(conn, table: str, scope: dict, query: str) -> tuple[list[str], list[str]]:
    """(similarity arm, word_similarity arm) over every chunk, as document keys.

    A sequential scan: no index backs these functions here. `word_similarity`
    is the one that fits a short query against a long chunk; plain `similarity`
    divides by the chunk's whole trigram set and so cannot reach its threshold
    for a 25-character phrase against a 1,500-character chunk. Both are here
    because the plan named `similarity`.
    """
    rows = conn.execute(
        text(
            "SELECT sd.source_path AS source_path, c.s AS s, c.w AS w FROM ("
            # OFFSET 0 keeps the planner from inlining the two scores into the
            # filter, which would compute each twice per row of a 2.4M-row scan.
            "  SELECT id, document_id, similarity(content, :q) AS s, "
            "         word_similarity(:q, content) AS w FROM chunks_v2 OFFSET 0"
            ") c " + fuzzy_scope(table) +
            "AND (c.s >= :ts OR c.w >= :tw) ORDER BY c.w DESC LIMIT 300"
        ),
        {"q": query, "ts": TRIGRAM_SIMILARITY, "tw": TRIGRAM_WORD_SIMILARITY, **scope},
    ).fetchall()
    by_similarity = sorted((r for r in rows if r.s >= TRIGRAM_SIMILARITY), key=lambda r: -r.s)
    by_word = [r for r in rows if r.w >= TRIGRAM_WORD_SIMILARITY]
    return (
        deduplicate([document_key(r.source_path) for r in by_similarity]),
        deduplicate([document_key(r.source_path) for r in by_word]),
    )


def relaxed_arm(conn, table: str, scope: dict, query: str, limit: int) -> list[str]:
    """Document keys of chunks holding >= RELAXED_TERM_SHARE of the query's terms."""
    config = LEXICAL_TEXT_CONFIG
    lexemes = [
        r[0]
        for r in conn.execute(
            text(f"SELECT lexeme FROM unnest(to_tsvector('{config}', :q))"), {"q": query}
        )
    ]
    tsquery = relaxed_tsquery(lexemes, RELAXED_TERM_SHARE)
    if tsquery is None:
        return []
    rows = conn.execute(
        text(
            "SELECT sd.source_path AS source_path FROM ("
            "  SELECT id, document_id, content FROM chunks_v2 "
            f"  WHERE to_tsvector('{config}', content) @@ CAST(:tsq AS tsquery) LIMIT :pool"
            ") c " + fuzzy_scope(table) +
            f"ORDER BY ts_rank(to_tsvector('{config}', c.content), "
            "CAST(:orq AS tsquery)) DESC LIMIT :k"
        ),
        {
            "tsq": tsquery,
            "orq": " | ".join(quote_lexeme(t) for t in lexemes),
            "pool": LEXICAL_CANDIDATE_POOL,
            "k": limit,
            **scope,
        },
    ).fetchall()
    return deduplicate([document_key(r.source_path) for r in rows])


def command_fuzzy(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.queries).read_text())
    kinds = args.kinds.split(",")
    queries = [Query(**q) for q in payload["queries"] if q["kind"] in kinds]
    collection = payload["collection"]
    config = get_config()
    engine = get_engine(config.database.url)

    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
    with engine.connect() as conn:
        revision = active_revision(conn, collection)
        table = vector_table_name(int(revision.embedding_profile_id))
        scope = {
            "collection": collection,
            "chunk_profile_id": revision.chunk_profile_id,
            "extractor_profile_id": revision.extractor_profile_id,
        }
        chunks = conn.execute(
            text("SELECT reltuples::bigint FROM pg_class WHERE relname = 'chunks_v2'")
        ).scalar_one()
        typo = next((q.query for q in queries if q.kind == "phrase-typo"), queries[0].query)
        began = time.perf_counter()
        conn.execute(
            text(
                "SELECT max(word_similarity(:q, content)), max(similarity(content, :q)) "
                "FROM (SELECT content FROM chunks_v2 LIMIT :n) t"
            ),
            {"q": typo, "n": TRIGRAM_PROBE_ROWS},
        )
        probe = time.perf_counter() - began
        per_query = probe / TRIGRAM_PROBE_ROWS * chunks
        print(
            f"trigram scan: {probe:.2f}s for {TRIGRAM_PROBE_ROWS} of ~{chunks:,} chunks; "
            f"about {per_query:.0f}s per query on one core (parallel workers divide it), "
            f"~{per_query * sum(q.kind == 'phrase-typo' for q in queries) / 60:.0f} min "
            "for the typo'd set",
            flush=True,
        )
        if args.estimate_only:
            return 0

        searcher = Searcher(config)
        arm_limit = arm_fetch_limit(PRODUCTION_DEPTH, hybrid=True)
        records: list[FuzzyRecord] = []
        began = time.perf_counter()
        for i, query in enumerate(queries, start=1):
            vector, lexical = split_arms(
                searcher.candidates(query.query, top_k=PRODUCTION_DEPTH, collections=[collection])
            )
            lexical_docs = [document_key(r["source_path"]) for r in lexical]
            similarity: list[str] = []
            word: list[str] = []
            if not lexical_docs:
                similarity, word = trigram_arms(conn, table, scope, query.query)
            records.append(
                FuzzyRecord(
                    query=query.query,
                    gold=query.gold,
                    kind=query.kind,
                    vector=[document_key(r["source_path"]) for r in vector],
                    lexical=lexical_docs,
                    shipped_lead=(
                        "lexical" if searcher.lexical_should_lead(query.query) else "vector"
                    ),
                    similarity=similarity,
                    word_similarity=word,
                    relaxed=relaxed_arm(conn, table, scope, query.query, arm_limit),
                )
            )
            if i % 20 == 0:
                print(f"  {i}/{len(queries)} ({time.perf_counter() - began:.0f}s)", flush=True)

    out = Path(args.out).expanduser()
    out.write_text(json.dumps([asdict(r) for r in records]))
    counts = {k: sum(r.kind == k for r in records) for k in kinds}
    fired = sum(not r.lexical for r in records)
    print(f"{fired}/{len(records)} queries had an empty lexical arm; records in {out}")
    print(format_fuzzy(fuzzy_report(records, PRODUCTION_DEPTH), counts, PRODUCTION_DEPTH))
    return 0


def command_riders(args: argparse.Namespace) -> int:
    """Score the three chunking riders against fresh v2-moe, per document."""
    try:
        import numpy as np
    except ImportError:
        print("riders needs numpy in the environment running it", file=sys.stderr)
        return 1
    out_dir = Path(args.out).expanduser()
    control_path = out_dir / f"{BASELINE_MODEL}.json"
    if not (out_dir / "sample.json").exists() or not control_path.exists():
        print(f"riders needs sample.json and {control_path.name} (compare-models)", file=sys.stderr)
        return 1
    papers = json.loads((out_dir / "sample.json").read_text())["papers"]
    control = json.loads(control_path.read_text())

    spans = [reconstruct_span(p["long"]) for p in papers]
    texts = [rider_chunks(span, STORED_CHUNK_SIZE, STORED_CHUNK_OVERLAP) for span in spans]
    counts = [len(t) for t in texts]
    flat = [c for t in texts for c in t]
    short = [c for p in papers for c in p["short"]]
    encoding = tokenizer()

    def tiktoken_of(chunks: list[str]) -> int:
        return sum(len(encoding.encode(c, disallowed_special=())) for c in chunks)

    print(f"{len(papers)} papers, rider census over their spans:")
    for name, value in rider_census(spans).items():
        print(f"  {name:<28} {value:>7}")
    dropped = sum(is_punctuation_dominated(c) for c in short)
    print(
        f"  control chunks {len(short)}, riders chunks {len(flat)}, "
        f"punctuation-dominated control chunks {dropped}"
    )
    # Model tokens per tiktoken token, from the control run, projects the rider
    # arm's cost at the control's throughput.
    ratio = control["short_tokens"] / tiktoken_of(short)
    projected = tiktoken_of(flat) * ratio
    rate = control["short_tokens"] / control["short_seconds"]
    print(
        f"  embedding {projected:,.0f} model tokens at the control's {rate:.0f} tok/s "
        f"= about {projected / rate / 60:.0f} min"
    )
    if args.dry_run:
        return 0

    cache = out_dir / f"{RIDERS_ARM}.json"
    if cache.exists():
        print(f"  {RIDERS_ARM}: cached", flush=True)
    else:
        spec = MODELS[BASELINE_MODEL]
        with serving(get_config(), spec, out_dir, args.port) as url:
            vectors, tokens, seconds = embed_all(
                url, [format_for(spec, c, query=False) for c in flat], RIDERS_ARM
            )
        refused = sum(v is None for v in vectors)
        print(
            f"  {RIDERS_ARM}: {len(flat)} chunks, {tokens} tokens in {seconds:.0f}s "
            f"({tokens / seconds:.0f} tok/s), {refused} refused",
            flush=True,
        )
        cache.write_text(
            json.dumps(
                {"model": RIDERS_ARM, "counts": counts, "chunks": vectors,
                 "tokens": tokens, "seconds": seconds}
            )
        )
    riders = json.loads(cache.read_text())
    if riders["counts"] != counts:
        print(f"{cache} was built by different riders; delete it", file=sys.stderr)
        return 1

    docs = [p["id"] for p in papers for _ in p["short"]]
    rider_docs = [p["id"] for p, n in zip(papers, counts, strict=True) for _ in range(n)]
    stored = [v for p in papers for v in p["stored"]]
    keep = shared_pool([stored, control["short"]])
    # The control without the chunks 2b would drop, so 2c and 2e are visible
    # apart from the hubs the server's refusals already removed from it.
    keep_clean = [i for i in keep if not is_punctuation_dominated(short[i])]
    results: dict[str, dict[str, tuple[list[int], list[float]]]] = {}
    for kind in ("title", "phrase"):
        query_docs = [p["id"] for p in papers if p[kind]]
        queries = control[f"q_{kind}"]
        results[kind] = {
            "stored@320": score_arm(np, queries, query_docs, stored, docs, keep),
            "control: fresh v2-moe@320": score_arm(
                np, queries, query_docs, control["short"], docs, keep
            ),
            "control without 2b chunks": score_arm(
                np, queries, query_docs, control["short"], docs, keep_clean
            ),
            "riders (2b+2c+2e)": score_arm(np, queries, query_docs, riders["chunks"], rider_docs),
        }
    outcome = {}
    for kind, arms in results.items():
        baseline = arms["control: fresh v2-moe@320"][0]
        print(f"\n{kind} -> body  (n={len(arms['stored@320'][0])}, pool {len(papers)} papers)")
        print(
            f"  {'arm':<28} {'recall@1':>9} {'recall@10':>10} {'MRR':>7}"
            f" {'better':>7} {'worse':>6} {'sign p':>9}"
        )
        for arm, (ranks, margins) in arms.items():
            m = summarise(ranks, margins)
            better, worse, p = paired(ranks, baseline)
            print(
                f"  {arm:<28} {m['recall@1']:>9.3f} {m['recall@10']:>10.3f} {m['MRR']:>7.3f}"
                f" {better:>7} {worse:>6} {p:>9.2g}"
            )
            if arm.startswith("riders"):
                outcome[kind] = paired_at_1(ranks, baseline)
    passed = rider_verdict(outcome["title"], outcome["phrase"])
    for kind, (better, worse, p) in outcome.items():
        print(f"riders vs control, {kind} recall@1: better {better}, worse {worse}, p {p:.2g}")
    print(f"\nrider rule (recall@1 vs the control row; p < {RIDER_P} on one kind, "
          f"no loss on the other): {'PASS' if passed else 'FAIL'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="draw query sets from the corpus")
    build.add_argument("-c", "--collection", default="papers")
    build.add_argument("-n", type=int, default=60, help="queries per set")
    build.add_argument("-o", "--out", default="eval/queries.json")
    build.set_defaults(func=command_build)

    perturb = sub.add_parser("perturb", help="add perturbed phrase sets to a query file")
    perturb.add_argument("-q", "--queries", default="eval/queries.json")
    perturb.set_defaults(func=command_perturb)

    score = sub.add_parser("score", help="score the active revision against a query file")
    score.add_argument("-q", "--queries", default="eval/queries.json")
    score.add_argument(
        "--depth",
        type=int,
        default=DEFAULT_DEPTH,
        help=f"per-arm fetch limit and returned list length (default {DEFAULT_DEPTH}; "
        f"`cementic search` ships {PRODUCTION_DEPTH})",
    )
    score.add_argument("--json", action="store_true", help="also emit machine-readable totals")
    score.set_defaults(func=command_score)

    arms = sub.add_parser("arms", help="cache both arms' pre-fusion candidates per query")
    arms.add_argument("-q", "--queries", default="eval/queries.json")
    arms.add_argument("-o", "--out", default=DEFAULT_ARMS_CACHE)
    arms.set_defaults(func=command_arms)

    ablate = sub.add_parser("ablate", help="score rank-1 policies against the arms cache")
    ablate.add_argument("-a", "--arms", default=DEFAULT_ARMS_CACHE)
    ablate.add_argument(
        "--depth",
        type=int,
        default=PRODUCTION_DEPTH,
        help=f"returned list length (default {PRODUCTION_DEPTH}, what `cementic search` ships)",
    )
    ablate.set_defaults(func=command_ablate)

    compare = sub.add_parser("compare-models", help="embedding models head to head")
    compare.add_argument("-c", "--collection", default="papers")
    compare.add_argument("-n", "--documents", type=int, default=COMPARE_DOCUMENTS)
    compare.add_argument("-o", "--out", default=COMPARE_DIR)
    compare.add_argument(
        "--models",
        default=",".join(MODELS),
        help=f"comma-separated, from {', '.join(MODELS)}; must include {BASELINE_MODEL}",
    )
    compare.add_argument("--port", type=int, default=11557)
    compare.set_defaults(func=command_compare_models)

    riders = sub.add_parser("riders", help="score chunking riders 2b/2c/2e against fresh v2-moe")
    riders.add_argument("-o", "--out", default=COMPARE_DIR)
    riders.add_argument("--port", type=int, default=11557)
    riders.add_argument(
        "--dry-run", action="store_true", help="print the census and cost estimate; embed nothing"
    )
    riders.set_defaults(func=command_riders)

    fuzzy = sub.add_parser("fuzzy", help="score fuzzy lexical fallbacks against the live database")
    fuzzy.add_argument("-q", "--queries", default="eval/queries.json")
    fuzzy.add_argument("-o", "--out", default="~/.cache/cementic-waves/fuzzy-records.json")
    fuzzy.add_argument(
        "--kinds", default=",".join(FUZZY_KINDS), help="comma-separated query kinds to score"
    )
    fuzzy.add_argument(
        "--estimate-only", action="store_true", help="time a trigram probe scan and stop"
    )
    fuzzy.set_defaults(func=command_fuzzy)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
