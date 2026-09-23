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

The database is not reachable from an agent sandbox; run these unsandboxed.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from sqlalchemy import text

from cementic.config import get_config
from cementic.db import LEXICAL_TEXT_CONFIG, get_engine
from cementic.hybrid import combine, reciprocal_rank_fusion
from cementic.search import Searcher, arm_fetch_limit, split_arms

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

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
