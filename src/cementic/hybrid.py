"""Combining a lexical ranking with a vector ranking (pure).

The functional core of hybrid retrieval: ranking lists in, one ranking list out,
no database and no embedding server. The shell in `search.py` owns fetching the
two rankings and deciding which arm leads; everything here is a total function
of its arguments.

Shape measured on the `papers` corpus, 2026-09-07 (PLAN.md, execution order
"hybrid retrieval"):

- Fusion is unambiguously right for the result *list*. recall@10 improves for
  both query kinds -- rare exact tokens 0.047 -> 1.000, semantic queries
  0.787 -> 0.840. There is no trade-off to manage.
- Fusion cannot win the *top slot*. Reciprocal rank fusion is symmetric in its
  inputs, so a document ranked first by one arm and absent from the other ties
  exactly with the other arm's first document -- both score 1/(k+1). Whichever
  way that tie is broken is a bet on an arm, and the bet that wins rare-token
  queries (0.047 -> 0.713) is the one that loses semantic queries
  (0.713 -> 0.567).

Hence the split this module implements: **the leading arm owns rank 1, and
reciprocal rank fusion owns everything below it.** Both measured optima are
reached at once, because the two metrics are decided in different places.
"""

from __future__ import annotations

from collections import defaultdict

#: Reciprocal-rank-fusion smoothing constant. 60 is the value from the original
#: RRF paper and the de-facto default; it is deliberately not tuned here, since
#: k trades rank-1 sharpness against robustness and the rank-1 question is
#: settled by `lead` instead.
RRF_K = 60


def deduplicate(paths: list[str]) -> list[str]:
    """Drop repeats, keeping first position (pure).

    Both arms return one row per *chunk*, and a document usually owns many
    chunks, so its best-ranked chunk is what should represent it.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered


def reciprocal_rank_fusion(
    *rankings: list[str],
    k: int = RRF_K,
) -> list[str]:
    """Merge ranked lists by summed reciprocal rank (pure).

    Each list contributes ``1/(k + rank)`` per document. Chosen over blending
    the arms' own scores because cosine distance and a text-rank score are on
    incomparable scales: blending them requires inventing a normalisation and a
    weight, and neither has a defensible value.

    Ties are left in the caller's hands rather than resolved here. A symmetric
    tie is real information -- it says the two arms disagree and neither ranking
    dominates -- and burying it behind an arbitrary sort order is what made the
    first measurement of this look like a fusion failure.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, path in enumerate(ranking, start=1):
            scores[path] += 1.0 / (k + rank)
    return [path for path, _ in sorted(scores.items(), key=lambda item: -item[1])]


def looks_like_identifier(query: str) -> bool:
    """Whether a query is a single bare word (pure).

    The syntactic half of the routing decision; the other half is how rare the
    word is, which only the index can answer. Split because this half is a total
    function of the string and should not need a database to test.

    A quoted or multi-word query is never treated as an identifier: "Kalman
    filter" is a topic, and the vector arm is better at topics.
    """
    stripped = query.strip()
    return bool(stripped) and len(stripped.split()) == 1


def combine(
    vector_ranking: list[str],
    lexical_ranking: list[str],
    *,
    lead: str = "vector",
    k: int = RRF_K,
) -> list[str]:
    """One ranking from two, with ``lead`` owning the first position (pure).

    ``lead`` names the arm trusted for the top slot: ``"lexical"`` for a query
    that is a single rare token, ``"vector"`` for everything else. Only the
    first position is routed. Taking more of the leading arm's ordering would
    give back the recall@10 that fusion earns -- for semantic queries the vector
    arm alone reaches 0.787 where the fused list reaches 0.840, and those extra
    documents are precisely the ones only the lexical arm found.

    Degenerate cases fall out without special-casing: if the leading arm
    returned nothing, the fused list is returned unchanged.
    """
    if lead not in ("vector", "lexical"):
        raise ValueError(f"lead must be 'vector' or 'lexical', got {lead!r}")
    leading = vector_ranking if lead == "vector" else lexical_ranking
    fused = reciprocal_rank_fusion(vector_ranking, lexical_ranking, k=k)
    return deduplicate(leading[:1] + fused)
