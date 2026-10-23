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


class TestSpanTokenRange:
    def test_matches_the_windows_chunk_text_produces(self) -> None:
        import tiktoken

        from cementic.chunk import TOKENIZER, chunk_text

        encoding = tiktoken.get_encoding(TOKENIZER)
        body = " ".join(f"word{i}" for i in range(3000))
        tokens = encoding.encode(body)
        start, end = harness.span_token_range(320, 80, 2, 8)
        chunks = chunk_text(body, 320, 80)
        assert encoding.encode(chunks[2].content) == tokens[start : start + 320]
        assert encoding.encode(chunks[9].content) == tokens[end - 320 : end]

    def test_the_stored_span_is_2000_tokens(self) -> None:
        assert harness.span_token_range(320, 80, 2, 8) == (480, 2480)


class TestFormatFor:
    def test_nomic_uses_cementics_own_prefixes(self) -> None:
        spec = harness.MODELS["v2-moe"]
        assert harness.format_for(spec, "x", query=True) == "search_query: x"
        assert harness.format_for(spec, "x", query=False) == "search_document: x"

    def test_instruction_models_prefix_queries_only(self) -> None:
        spec = harness.MODELS["qwen3-0.6b"]
        assert harness.format_for(spec, "x", query=True).endswith("\nQuery: x")
        assert harness.format_for(spec, "x", query=False) == "x"

    def test_gemma_prefixes_both_sides(self) -> None:
        spec = harness.MODELS["gemma-300m"]
        assert harness.format_for(spec, "x", query=True) == "task: search result | query: x"
        assert harness.format_for(spec, "x", query=False) == "title: none | text: x"


class TestGoldRankAndMargin:
    def test_a_document_is_ranked_by_its_best_chunk(self) -> None:
        # Doc 2 owns the single best chunk; doc 1 has more good chunks.
        rank, margin = harness.gold_rank_and_margin([0.8, 0.7, 0.9], [1, 1, 2], gold=2)
        assert rank == 1
        assert margin == pytest.approx(0.1)

    def test_rank_counts_documents_not_chunks(self) -> None:
        rank, margin = harness.gold_rank_and_margin([0.9, 0.95, 0.5], [1, 1, 2], gold=2)
        assert rank == 2
        assert margin == pytest.approx(-0.45)

    def test_a_gold_document_absent_from_the_pool_ranks_last(self) -> None:
        rank, margin = harness.gold_rank_and_margin([0.9, 0.5], [1, 2], gold=3)
        assert rank == 3
        assert margin == float("-inf")


class TestSignTest:
    def test_reproduces_the_v1_5_runs_recorded_p_value(self) -> None:
        # PLAN.md: 133 better, 245 worse, "exact sign test p = 8.9e-09".
        assert harness.sign_test(133, 245) == pytest.approx(8.9e-09, rel=0.05)

    def test_an_even_split_is_not_significant(self) -> None:
        assert harness.sign_test(50, 50) == pytest.approx(1.0)

    def test_no_decided_pairs_is_uninformative(self) -> None:
        assert harness.sign_test(0, 0) == 1.0

    def test_is_symmetric(self) -> None:
        assert harness.sign_test(10, 30) == harness.sign_test(30, 10)


class TestSummariseAndPaired:
    def test_summarise(self) -> None:
        m = harness.summarise([1, 2, 20], [0.1, -0.1, float("-inf")])
        assert m["recall@1"] == pytest.approx(1 / 3)
        assert m["recall@10"] == pytest.approx(2 / 3)
        assert m["MRR"] == pytest.approx((1 + 0.5 + 0.05) / 3)
        assert m["margin"] == pytest.approx(0.0)

    def test_paired_counts_lower_rank_as_better(self) -> None:
        better, worse, _ = harness.paired([1, 3, 2], [2, 3, 1])
        assert (better, worse) == (1, 1)


class TestSharedPool:
    def test_keeps_rows_every_pool_has(self) -> None:
        assert harness.shared_pool([[[1], None, [1], [1]], [[1], [1], None, [1]]]) == [0, 3]

    def test_score_arm_scores_the_rows_it_is_given(self) -> None:
        np = pytest.importorskip("numpy")
        pool = [[1.0, 0.0], [0.0, 1.0], None]
        docs = [1, 2, 3]
        own = harness.score_arm(np, [[0.0, 1.0]], [2], pool, docs)
        forced = harness.score_arm(np, [[0.0, 1.0]], [2], pool, docs, keep=[0])
        assert own[0] == [1]
        assert forced[0] == [2]  # gold has no row in the pool: ranks after the others


class TestPunctuationDominated:
    def test_dot_leader_contents_line(self) -> None:
        assert harness.is_punctuation_dominated(". . . . . . . 117 C.5.4 Proof . . . . . .")

    def test_prose_and_equations_are_kept(self) -> None:
        assert not harness.is_punctuation_dominated("We show that x = y + 1 holds for all n.")

    def test_empty_chunk_is_dropped(self) -> None:
        assert harness.is_punctuation_dominated("  \n ")


class TestDehyphenate:
    @pytest.mark.parametrize(
        ("raw", "clean"),
        [
            ("by max- imizing the agreement", "by maximizing the agreement"),
            ("a func-\ntion of time", "a function of time"),
            ("a func- \n  tion of time", "a function of time"),
            ("a multi-\nvariate model", "a multivariate model"),
            ("well­-conducted", "well-conducted"),
        ],
    )
    def test_joins_a_split_word(self, raw: str, clean: str) -> None:
        assert harness.dehyphenate(raw) == clean

    @pytest.mark.parametrize(
        "kept",
        [
            "fixed-\nand random-effects",  # continuation is a function word
            "pre- and post-match",
            "a well-\nknown result",  # compound prefix
            "state-of-the-art",  # no break at all
            "x-\n3 and y",  # digit continuation
            "low-\nDimensional",  # capitalised continuation
        ],
    )
    def test_leaves_real_compounds(self, kept: str) -> None:
        assert harness.dehyphenate(kept) == kept

    def test_keeps_a_compound_the_document_spells_intact(self) -> None:
        text = "an over-\nparameterized net and an over-parameterized one"
        assert harness.dehyphenate(text) == text

    def test_joins_when_the_document_spells_the_word_whole(self) -> None:
        assert harness.dehyphenate("a well-\nknown fact; wellknown too").startswith("a wellknown")


class TestStructureChunks:
    def test_no_boundary_is_the_plain_window(self) -> None:
        text = "word " * 400
        plain = [c.content for c in harness.chunk_text(text, 100, 20)]
        assert harness.structure_chunks(text, 100, 20) == plain

    def test_cuts_at_headings_and_page_breaks(self) -> None:
        text = "intro text\n# Methods\nbody one\n\f\nbody two\n## Results\nbody three\n"
        assert harness.structure_chunks(text, 100, 20) == [
            "intro text\n",
            "# Methods\nbody one\n\f\n",
            "body two\n",
            "## Results\nbody three\n",
        ]

    def test_form_feed_inside_a_line_is_not_a_boundary(self) -> None:
        text = "a = \f\f det(x)\n b = c\n"
        assert harness.structure_chunks(text, 100, 20) == [text]

    def test_a_long_section_falls_back_to_the_window(self) -> None:
        text = "# Head\n" + "word " * 400
        chunks = harness.structure_chunks(text, 100, 20)
        assert len(chunks) > 3
        assert max(len(harness.tokenizer().encode(c)) for c in chunks) <= 100

    def test_a_hash_that_is_not_a_heading_is_ignored(self) -> None:
        assert harness.structure_chunks("x #notheading\n#also not", 100, 20) == [
            "x #notheading\n#also not"
        ]


class TestRiderChunks:
    def test_applies_all_three(self) -> None:
        span = "# Intro\nby max- imizing the agree-\nment\n# TOC\n" + ". " * 60 + "\n"
        assert harness.rider_chunks(span, 100, 20) == [
            "# Intro\nby maximizing the agreement\n"
        ]


class TestReconstructSpan:
    def test_rebuilds_the_span_from_overlapping_chunks(self) -> None:
        words = [f"w{i}" for i in range(2000)]
        text = " ".join(words)
        chunks = [c.content for c in harness.chunk_text(text, 1300, 325)]
        assert len(chunks) > 1
        assert harness.reconstruct_span(chunks) == text


class TestRiderVerdict:
    def test_passes_on_one_gain_and_no_loss(self) -> None:
        assert harness.rider_verdict((60, 20, 1e-5), (30, 28, 0.9))

    def test_fails_when_the_other_kind_loses(self) -> None:
        assert not harness.rider_verdict((60, 20, 1e-5), (10, 28, 0.9))

    def test_fails_without_significance(self) -> None:
        assert not harness.rider_verdict((30, 20, 0.2), (30, 28, 0.9))


class TestRiderCensus:
    def test_counts_what_each_rider_acts_on(self) -> None:
        census = harness.rider_census(["# A\nfunc-\ntion\n\f\nx­", "plain"])
        assert census["markdown heading lines"] == 1
        assert census["spans with a heading"] == 1
        assert census["page-break lines"] == 1
        assert census["line-break hyphens"] == 1
        assert census["line-break hyphens joined"] == 1
        assert census["soft hyphens"] == 1


class TestRelaxedTsquery:
    def test_ors_every_conjunction_of_the_required_size(self) -> None:
        # ceil(0.6 * 4) = 3 of 4 terms
        query = harness.relaxed_tsquery(["a", "b", "c", "d"], 0.6)
        assert query is not None
        assert query.count("|") == 3  # C(4, 3) = 4 conjunctions
        assert "('a' & 'b' & 'c')" in query
        assert "('b' & 'c' & 'd')" in query

    def test_a_single_term_is_itself(self) -> None:
        assert harness.relaxed_tsquery(["a"], 0.6) == "('a')"

    def test_repeated_terms_count_once(self) -> None:
        assert harness.relaxed_tsquery(["a", "a", "b"], 1.0) == "('a' & 'b')"

    def test_no_terms_is_no_query(self) -> None:
        assert harness.relaxed_tsquery([], 0.6) is None

    def test_quotes_are_escaped(self) -> None:
        assert harness.relaxed_tsquery(["o'neil"], 0.6) == "('o''neil')"


def fuzzy_record(**overrides) -> "harness.FuzzyRecord":
    fields = {
        "query": "q", "gold": "g.pdf", "kind": "phrase-typo",
        "vector": ["x.pdf", "y.pdf"], "lexical": [], "shipped_lead": "vector",
        "similarity": [], "word_similarity": [], "relaxed": [],
    }
    return harness.FuzzyRecord(**{**fields, **overrides})


class TestFuzzyReport:
    def test_a_trigram_fallback_only_fills_an_empty_arm(self) -> None:
        empty = fuzzy_record(word_similarity=["g.pdf"])
        full = fuzzy_record(lexical=["x.pdf"], word_similarity=["g.pdf"])
        assert harness.fuzzy_lexical(empty, "trigram word_similarity") == ["g.pdf"]
        assert harness.fuzzy_lexical(full, "trigram word_similarity") == ["x.pdf"]

    def test_the_relaxed_arm_replaces_the_lexical_arm(self) -> None:
        record = fuzzy_record(lexical=["x.pdf"], relaxed=["g.pdf"])
        assert harness.fuzzy_lexical(record, "relaxed OR") == ["g.pdf"]

    def test_counts_wins_against_the_shipped_arm(self) -> None:
        record = fuzzy_record(word_similarity=["g.pdf"])
        rows = harness.fuzzy_report([record], depth=10)["phrase-typo"]
        assert rows["shipped"][:3] == (0.0, 0.0, 0.0)
        assert rows["trigram word_similarity"][1] == 1.0  # gold now in the top 10
        assert rows["trigram word_similarity"][3:] == (1, 0, 1)
        assert rows["trigram similarity"][5] == 0  # never fired: ranking unchanged

    def test_formats_every_variant(self) -> None:
        report = harness.fuzzy_report([fuzzy_record()], depth=10)
        text = harness.format_fuzzy(report, {"phrase-typo": 1}, 10)
        for variant in harness.FUZZY_VARIANTS:
            assert variant in text
