#!/usr/bin/env python
"""Retrieval quality of the active revision, scored on synthetic known-item queries.

The harness PLAN.md's "Execution order — retrieval quality and the rebuild it
rides" calls Phase 0. It exists so that a change to the ranker, the chunker or
the embedding model can be scored rather than argued about, and so the score
survives the rebuild that produced it.

Two commands, because they cost different things and only one of them is
allowed to move between runs:

    build   Draw query sets from the corpus and write them to a JSON file.
            Thousands of document-frequency queries; minutes. Run once.
    score   Read that file, run each query through the real search path, and
            print recall@1 / recall@10 / MRR per set. Needs the embedding
            server; seconds per query.

Splitting them is the point. A judgment re-derived on every run is not a
judgment, it is a coin flip with a seed -- so `build`'s output is checked into
git and `score` never writes to it. Comparing two systems means running `score`
twice against the *same* file, including across a re-index.

Judgments are mechanical. Nobody hand-labels anything: each query is drawn from
a document that is by construction the answer, which is what makes the gold
label free and the query set re-buildable on a different corpus.

Usage:
    python scripts/measure_retrieval_quality.py build -o eval/queries.json
    python scripts/measure_retrieval_quality.py score -q eval/queries.json

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
from cementic.search import Searcher

#: Ranking depth requested from search. Matches `search.MAX_SEARCH_RESULTS`, so
#: recall@10 and MRR are measured over the deepest list the CLI can return.
DEPTH = 50

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


@dataclass(frozen=True)
class Query:
    """One scored query and the filename of the document that answers it."""

    query: str
    gold: str
    kind: str


# --------------------------------------------------------------------------
# Pure functions
# --------------------------------------------------------------------------


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


def format_table(name: str, rows: dict[str, tuple[float, float, float]], n: int) -> str:
    """Render one set's scores (pure)."""
    lines = [f"\n{name}  (n={n})", f"  {'arm':<10} {'recall@1':>9} {'recall@10':>10} {'MRR':>7}"]
    for arm, (r1, r10, mrr) in rows.items():
        lines.append(f"  {arm:<10} {r1:>9.3f} {r10:>10.3f} {mrr:>7.3f}")
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
        "depth": DEPTH,
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
                    for r in searcher.search(query.query, top_k=DEPTH, collections=[collection])
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
        print(format_table(kind, {"search": overall[kind]}, n), flush=True)

    print(f"\ntotal {time.perf_counter() - t0:.0f}s", flush=True)
    if args.json:
        print(json.dumps({k: {"recall@1": v[0], "recall@10": v[1], "mrr": v[2]}
                          for k, v in overall.items()}, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="draw query sets from the corpus")
    build.add_argument("-c", "--collection", default="papers")
    build.add_argument("-n", type=int, default=60, help="queries per set")
    build.add_argument("-o", "--out", default="eval/queries.json")
    build.set_defaults(func=command_build)

    score = sub.add_parser("score", help="score the active revision against a query file")
    score.add_argument("-q", "--queries", default="eval/queries.json")
    score.add_argument("--json", action="store_true", help="also emit machine-readable totals")
    score.set_defaults(func=command_score)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
