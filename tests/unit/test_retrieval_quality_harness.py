"""Pure functions of scripts/measure_retrieval_quality.py.

The harness is not importable as a package -- it is a script -- so it is loaded
by path. Only the pure half is tested: everything below `Corpus probes` needs a
database and belongs to the integration suite that does not exist for scripts.

These tests exist because the perturbations are the entire epistemic content of
the perturbed sets. A `drop_interior_word` that silently produced fragments
would still score, still print a table, and the table would be wrong in exactly
the direction the experiment is looking for -- which is how the first phrase
set shipped with a 3-character floor that deleted "to" out of the middle of its
queries.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "measure_retrieval_quality.py"
_spec = importlib.util.spec_from_file_location("measure_retrieval_quality", _PATH)
assert _spec is not None and _spec.loader is not None
harness = importlib.util.module_from_spec(_spec)
# Registered before execution because `@dataclass` resolves the defining
# module out of `sys.modules`, and raises AttributeError on None when the
# module is loaded by path without being registered first.
sys.modules[_spec.name] = harness
_spec.loader.exec_module(harness)


def rng() -> random.Random:
    return random.Random(0)


class TestSwapAdjacentWords:
    def test_preserves_the_multiset_of_words(self) -> None:
        phrase = "rule-based approaches work well for regular"
        swapped = harness.swap_adjacent_words(phrase, rng())
        assert swapped is not None
        assert sorted(swapped.split()) == sorted(phrase.split())

    def test_changes_the_order(self) -> None:
        phrase = "key features of cities and regions"
        assert harness.swap_adjacent_words(phrase, rng()) != phrase

    def test_moves_exactly_one_adjacent_pair(self) -> None:
        words = "alpha beta gamma delta".split()
        swapped = harness.swap_adjacent_words(" ".join(words), rng())
        assert swapped is not None
        differing = [i for i, (a, b) in enumerate(zip(words, swapped.split())) if a != b]
        assert len(differing) == 2
        assert differing[1] == differing[0] + 1

    def test_refuses_a_single_word(self) -> None:
        assert harness.swap_adjacent_words("alpha", rng()) is None

    def test_does_not_leave_a_function_word_at_an_edge(self) -> None:
        # Swapping either pair strands "the" or "of" on an edge.
        assert harness.swap_adjacent_words("the alpha", rng()) is None


class TestDropInteriorWord:
    def test_removes_exactly_one_word(self) -> None:
        phrase = "prior densities of the SSM depend"
        dropped = harness.drop_interior_word(phrase, rng())
        assert dropped is not None
        assert len(dropped.split()) == len(phrase.split()) - 1

    def test_keeps_both_edges(self) -> None:
        phrase = "prior densities of the SSM depend"
        dropped = harness.drop_interior_word(phrase, rng())
        assert dropped is not None
        assert dropped.split()[0] == "prior"
        assert dropped.split()[-1] == "depend"

    def test_keeps_the_remaining_words_in_order(self) -> None:
        words = "alpha beta gamma delta".split()
        dropped = harness.drop_interior_word(" ".join(words), rng())
        assert dropped is not None
        assert [w for w in words if w in dropped.split()] == dropped.split()

    def test_drops_short_function_words_rather_than_only_long_ones(self) -> None:
        # The defect this guards: a length floor that deletes "to" and leaves a
        # string nobody would type. Over every seed, "to" must be reachable.
        phrase = "enable planning to emerge while training"
        seen = {harness.drop_interior_word(phrase, random.Random(s)) for s in range(50)}
        assert "enable planning emerge while training" in seen

    def test_refuses_a_two_word_phrase(self) -> None:
        assert harness.drop_interior_word("alpha beta", rng()) is None


class TestTransposeCharacters:
    def test_preserves_word_count(self) -> None:
        phrase = "apparently contradictory phenomenon has become increasingly"
        typo = harness.transpose_characters(phrase, rng())
        assert typo is not None
        assert len(typo.split()) == len(phrase.split())

    def test_damages_the_longest_word(self) -> None:
        phrase = "quick extraordinarily red foxes"
        typo = harness.transpose_characters(phrase, rng())
        assert typo is not None
        changed = [(a, b) for a, b in zip(phrase.split(), typo.split()) if a != b]
        assert len(changed) == 1
        assert changed[0][0] == "extraordinarily"

    def test_refuses_a_phrase_whose_edge_is_already_a_function_word(self) -> None:
        # Not reachable from a real query set -- `phrase_candidates` bars such
        # edges when drawing -- but the perturbations must not be the thing
        # that introduces one.
        assert harness.transpose_characters("the extraordinarily fox", rng()) is None

    def test_the_damaged_word_is_an_anagram_of_the_original(self) -> None:
        phrase = "alpha extraordinarily beta"
        typo = harness.transpose_characters(phrase, rng())
        assert typo is not None
        changed = [(a, b) for a, b in zip(phrase.split(), typo.split()) if a != b]
        assert sorted(changed[0][0]) == sorted(changed[0][1])

    def test_actually_changes_the_word(self) -> None:
        phrase = "alpha extraordinarily beta"
        assert harness.transpose_characters(phrase, rng()) != phrase

    def test_refuses_when_every_word_is_too_short(self) -> None:
        assert harness.transpose_characters("a bc de", rng()) is None

    def test_skips_a_word_whose_characters_all_repeat(self) -> None:
        # "aaaa" has no transposition that changes it; the longer real word
        # must be damaged instead of returning the phrase unchanged.
        typo = harness.transpose_characters("aaaa gamma", rng())
        assert typo is not None
        assert typo.split()[0] == "aaaa"
        assert typo.split()[1] != "gamma"


class TestHasFunctionWordEdge:
    @pytest.mark.parametrize("phrase", ["the alpha beta", "alpha beta of", "The alpha beta"])
    def test_rejects_an_edge_function_word(self, phrase: str) -> None:
        assert harness.has_function_word_edge(phrase.split())

    def test_allows_one_in_the_middle(self) -> None:
        assert not harness.has_function_word_edge("alpha of beta".split())

    def test_an_empty_list_has_no_edge(self) -> None:
        assert not harness.has_function_word_edge([])


class TestPerturbedQueries:
    def source(self) -> list[object]:
        return [
            harness.Query(query="prior densities of the SSM depend", gold="a.pdf", kind="phrase"),
            harness.Query(query="key features of cities and regions", gold="b.pdf", kind="phrase"),
            harness.Query(query="Hochreiter", gold="c.pdf", kind="rare-token"),
        ]

    def test_derives_only_from_the_phrase_set(self) -> None:
        derived = harness.perturbed_queries(self.source())
        assert {q.kind for q in derived} == set(harness.PERTURBED_KINDS)
        assert all(q.gold in {"a.pdf", "b.pdf"} for q in derived)

    def test_carries_the_gold_label_of_its_source(self) -> None:
        # The pairing that makes the comparison a paired one. If a perturbed
        # query kept the wrong gold, every score would be zero and it would
        # read as a finding.
        derived = harness.perturbed_queries(self.source())
        by_gold = {q.gold for q in derived if "SSM" in q.query or "ssm" in q.query.lower()}
        assert by_gold == {"a.pdf"}

    def test_is_deterministic(self) -> None:
        first = harness.perturbed_queries(self.source())
        second = harness.perturbed_queries(self.source())
        assert [(q.query, q.gold, q.kind) for q in first] == [
            (q.query, q.gold, q.kind) for q in second
        ]

    def test_does_not_depend_on_source_order(self) -> None:
        forward = harness.perturbed_queries(self.source())
        backward = harness.perturbed_queries(list(reversed(self.source())))
        assert {(q.query, q.kind) for q in forward} == {(q.query, q.kind) for q in backward}

    def test_never_reproduces_its_source_verbatim(self) -> None:
        originals = {q.query for q in self.source()}
        assert not originals & {q.query for q in harness.perturbed_queries(self.source())}

    def test_is_idempotent_over_its_own_output(self) -> None:
        # What `perturb` relies on to be rerunnable: derived kinds are not
        # themselves a source, so a second pass adds nothing.
        derived = harness.perturbed_queries(self.source())
        assert harness.perturbed_queries(derived) == []


def record(vector, lexical, *, gold="g", shipped_lead="vector", kind="phrase"):
    return harness.ArmRecord(
        query="q", gold=gold, kind=kind, vector=vector, lexical=lexical,
        shipped_lead=shipped_lead,
    )


class TestFusedRanking:
    def test_the_lead_owns_rank_one(self) -> None:
        vector, lexical = ["v", "x"], ["l", "x"]
        assert harness.fused_ranking(vector, lexical, "vector", 10)[0] == "v"
        assert harness.fused_ranking(vector, lexical, "lexical", 10)[0] == "l"

    def test_no_lead_puts_the_document_both_arms_found_first(self) -> None:
        # Plain RRF: "x" is second in both arms and outscores each arm's top.
        assert harness.fused_ranking(["v", "x"], ["l", "x"], None, 10)[0] == "x"

    def test_is_cut_to_depth(self) -> None:
        vector = [f"d{i}" for i in range(30)]
        assert len(harness.fused_ranking(vector, ["d0"], "vector", 10)) == 10

    def test_one_empty_arm_cuts_chunks_before_collapsing_documents(self) -> None:
        # `_merge_arms` returns chunk rows [:top_k] when an arm is empty, so a
        # document repeated across those rows leaves fewer than top_k documents.
        vector = ["a", "a", "b", "c"]
        assert harness.fused_ranking(vector, [], "lexical", 2) == ["a"]
        assert harness.fused_ranking([], vector, "vector", 3) == ["a", "b"]


class TestPolicyRanking:
    def test_shipped_follows_the_recorded_lead(self) -> None:
        rec = record(["v"], ["l"], shipped_lead="lexical")
        assert harness.policy_ranking(rec, "shipped", 10)[0] == "l"

    def test_oracle_takes_the_lead_that_ranks_gold_higher(self) -> None:
        rec = record(["v", "x"], ["g", "x"], gold="g")
        assert harness.policy_ranking(rec, "oracle", 10)[0] == "g"

    def test_oracle_keeps_vector_lead_on_a_tie(self) -> None:
        rec = record(["v", "x"], ["l", "x"], gold="absent")
        assert harness.policy_ranking(rec, "oracle", 10)[0] == "v"

    def test_an_unknown_policy_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown policy"):
            harness.policy_ranking(record(["v"], ["l"]), "coin", 10)


class TestAblation:
    def test_shipped_is_neither_better_nor_worse_than_itself(self) -> None:
        rows = harness.ablation([record(["v", "g"], ["g", "v"])], 10)
        assert rows["phrase"]["shipped"][3:] == (0, 0)

    def test_counts_paired_wins_and_losses_against_shipped(self) -> None:
        records = [
            record(["v", "x"], ["g", "x"], gold="g"),  # lexical lead wins this one
            record(["g", "x"], ["l", "x"], gold="g"),  # and loses this one
        ]
        rows = harness.ablation(records, 10)["phrase"]
        assert rows["lexical"][3:] == (1, 1)
        assert rows["oracle"][3:] == (1, 0)

    def test_the_oracle_is_never_worse_than_shipped(self) -> None:
        records = [
            record(["v", "g"], ["l", "g"], shipped_lead="lexical"),
            record(["g"], ["l"], shipped_lead="lexical"),
            record(["v"], ["g"], shipped_lead="vector"),
        ]
        assert harness.ablation(records, 10)["phrase"]["oracle"][4] == 0

    def test_recall_at_one_is_the_share_with_gold_first(self) -> None:
        records = [record(["g"], ["l"]), record(["v"], ["g"])]
        assert harness.ablation(records, 10)["phrase"]["vector"][0] == 0.5
