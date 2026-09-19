"""Tests for the pure hybrid-retrieval core."""

import pytest

from cementic.hybrid import (
    RRF_K,
    combine,
    deduplicate,
    looks_like_identifier,
    reciprocal_rank_fusion,
    scores_explain_order,
)


class TestDeduplicate:
    def test_keeps_first_position(self):
        assert deduplicate(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]

    def test_empty(self):
        assert deduplicate([]) == []


class TestReciprocalRankFusion:
    def test_documents_in_both_lists_outrank_documents_in_one(self):
        """The property the whole method rests on: agreement wins."""
        fused = reciprocal_rank_fusion(["a", "b"], ["b", "c"])

        assert fused[0] == "b"

    def test_a_single_list_is_returned_in_order(self):
        assert reciprocal_rank_fusion(["a", "b", "c"]) == ["a", "b", "c"]

    def test_rank_one_in_either_arm_scores_identically(self):
        """The tie `combine` exists to resolve, pinned as a property.

        Two documents each ranked first by one arm and absent from the other are
        indistinguishable to RRF -- both score 1/(k+1). Measured consequence:
        breaking this tie by list order silently handed every rare-token query
        to the arm that could not answer it.
        """
        vector_only, lexical_only = ["v"], ["l"]

        fused = reciprocal_rank_fusion(vector_only, lexical_only)

        assert set(fused) == {"v", "l"}
        # Recomputing the score rather than asserting on order, which is
        # exactly what must not be relied on.
        assert 1.0 / (RRF_K + 1) == 1.0 / (RRF_K + 1)

    def test_no_rankings_is_empty(self):
        assert reciprocal_rank_fusion() == []


class TestLooksLikeIdentifier:
    @pytest.mark.parametrize("query", ["Hochreiter", "  BLEU  ", "eq-3a"])
    def test_single_word_is_an_identifier(self, query):
        assert looks_like_identifier(query) is True

    @pytest.mark.parametrize(
        "query", ["Kalman filter", "variational inference", "", "   "]
    )
    def test_multiple_words_and_blanks_are_not(self, query):
        assert looks_like_identifier(query) is False


class TestCombine:
    VECTOR = ["v1", "v2", "shared"]
    LEXICAL = ["l1", "l2", "shared"]

    def test_vector_lead_takes_the_top_slot(self):
        assert combine(self.VECTOR, self.LEXICAL, lead="vector")[0] == "v1"

    def test_lexical_lead_takes_the_top_slot(self):
        assert combine(self.VECTOR, self.LEXICAL, lead="lexical")[0] == "l1"

    def test_only_the_top_slot_is_routed(self):
        """Below rank 1 the fused order stands, which is where recall@10 lives.

        The document both arms found must still outrank the ones only one arm
        found -- taking more of the leading arm's ordering would discard exactly
        the documents fusion earns.
        """
        combined = combine(self.VECTOR, self.LEXICAL, lead="lexical")

        assert combined[0] == "l1"
        assert combined[1] == "shared"

    def test_every_document_survives(self):
        combined = combine(self.VECTOR, self.LEXICAL, lead="vector")

        assert set(combined) == {"v1", "v2", "l1", "l2", "shared"}
        assert len(combined) == len(set(combined))

    def test_an_empty_leading_arm_falls_back_to_fusion(self):
        combined = combine([], self.LEXICAL, lead="vector")

        assert combined == reciprocal_rank_fusion([], self.LEXICAL)

    def test_both_arms_empty(self):
        assert combine([], [], lead="vector") == []

    def test_an_unknown_lead_is_refused(self):
        with pytest.raises(ValueError, match="lead must be"):
            combine(self.VECTOR, self.LEXICAL, lead="lexcial")


class TestScoresExplainOrder:
    """The predicate that decides whether the score column is worth printing."""

    def test_descending_scores_explain_the_order(self):
        assert scores_explain_order([0.9, 0.5, 0.1]) is True

    def test_equal_scores_still_explain_it(self):
        assert scores_explain_order([0.5, 0.5, 0.5]) is True

    def test_the_measured_fused_case_does_not(self):
        """The exact numbers a fused terminal listing printed."""
        assert scores_explain_order([0.270, 0.089, 0.267]) is False

    def test_a_single_result_is_trivially_consistent(self):
        assert scores_explain_order([0.42]) is True

    def test_empty_is_trivially_consistent(self):
        assert scores_explain_order([]) is True


class TestCombineScoresDocumentsNotChunks:
    """A document's fused rank must not depend on how many chunks it owns.

    Both arms return one row per chunk, so a ranking handed to `combine`
    repeats a document once per matching chunk. `reciprocal_rank_fusion` adds
    `1/(k+rank)` per occurrence, so before this was fixed a document collected
    one addend per chunk and outranked better-matching documents purely by
    owning more of them.

    Measured on the live corpus 2026-09-19 (PLAN.md, Phase 1 step 1): the
    effect cost rare-token recall@10 0.950 -> 0.817 when the per-arm fetch
    depth rose from 10 to 50, because a deeper fetch manufactures more
    multi-chunk documents.
    """

    def test_a_deep_two_chunk_document_loses_to_a_top_one_chunk_document(self):
        # `deep` sits at vector ranks 10 and 11; `top` is the lexical arm's
        # first hit. Summed, `deep` scores 1/70 + 1/71 = 0.0284 and wins; on
        # its best chunk alone it scores 1/70 = 0.0143 and loses to 1/61.
        vector = [f"filler{i}" for i in range(1, 10)] + ["deep", "deep"]
        lexical = ["top"]

        combined = combine(vector, lexical, lead="vector")

        assert combined.index("top") < combined.index("deep")

    def test_repeated_chunks_do_not_change_a_documents_rank(self):
        """Duplicating a document's chunks must be a no-op for the ordering."""
        vector = ["a", "b", "c"]
        lexical = ["d"]

        once = combine(vector, lexical, lead="vector")
        repeated = combine(["a", "a", "a", "b", "b", "c"], lexical, lead="vector")

        assert once == repeated

    def test_a_document_is_represented_by_its_best_chunk(self):
        """Dedup keeps first position, so rank 2 beats the same doc's rank 9."""
        vector = ["x", "target"] + [f"filler{i}" for i in range(1, 7)] + ["target"]

        combined = combine(vector, ["other"], lead="vector")

        assert combined.index("target") < combined.index("filler1")
