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

#: Reciprocal-rank-fusion smoothing constant, and the reason it is not tuned.
#:
#: 60 comes from the paper that introduced RRF -- Cormack, Clarke and Büttcher,
#: "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning
#: Methods" (2009) -- where it is not a derived constant but a pilot-tuned one:
#: "k = 60 was fixed during a pilot investigation and not altered during
#: subsequent validation", and their sweep "indicated that k = 60 was
#: near-optimal, but that the choice was not critical". It is also the
#: documented default in Elasticsearch and OpenSearch.
#:
#: So sweeping k here would be searching a flat region the authors already
#: reported as flat. The question k might otherwise have been asked to settle --
#: which arm wins rank 1 -- is settled by `lead` instead, and no value of k can
#: settle it: the tie it would need to break is exact and symmetric.
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


def scores_explain_order(scores: list[float]) -> bool:
    """Whether the scores are consistent with the order they are shown in (pure).

    The test for whether a score column is worth printing. After fusion the list
    is ordered by summed reciprocal rank while each result still carries its own
    arm's score, and those two orders differ -- which is how a terminal came to
    print `0.270, 0.089, 0.267` and look broken. A cosine similarity and a
    `ts_rank` are not on one scale and no amount of formatting makes them so.

    Checked rather than inferred from a flag: whether the numbers explain the
    order is a property of the numbers, and a flag saying "this was fused" would
    still be wrong for a fused list whose arms happened not to interleave.
    """
    return all(earlier >= later for earlier, later in zip(scores, scores[1:]))


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

    **Each ranking is collapsed to one entry per document before fusing, and
    that is load-bearing (2026-09-19).** Callers pass chunk-level rankings, so a
    document repeats once per matching chunk, and `reciprocal_rank_fusion` adds
    a term per occurrence. Fusing the raw lists therefore ranks a document by
    how many chunks it owns rather than by how well its best chunk matches: a
    one-chunk document cannot exceed ``1/(k+1)`` however good it is, while a
    two-chunk document at ranks 40 and 41 scores ``1/100 + 1/101`` and passes
    it. On the live corpus that cost rare-token recall@10 0.950 -> 0.817 when
    the per-arm fetch depth rose 10 -> 50, because depth manufactures
    multi-chunk documents (PLAN.md, Phase 1 step 1).

    `deduplicate` keeps first position, so each document enters fusion at its
    best-ranked chunk -- which is what its own docstring already said the
    representative should be.
    """
    if lead not in ("vector", "lexical"):
        raise ValueError(f"lead must be 'vector' or 'lexical', got {lead!r}")
    vector_documents = deduplicate(vector_ranking)
    lexical_documents = deduplicate(lexical_ranking)
    leading = vector_documents if lead == "vector" else lexical_documents
    fused = reciprocal_rank_fusion(vector_documents, lexical_documents, k=k)
    return deduplicate(leading[:1] + fused)
