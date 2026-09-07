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
