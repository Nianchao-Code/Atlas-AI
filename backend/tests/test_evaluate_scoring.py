"""The scoring functions every published quality number comes out of.

The pipeline gets measured constantly and the ruler almost never does. These
pin what `_hit` counts as a retrieval hit, what the keyless correctness
fallback rewards, and one asymmetry in the abstention metric that is easy to
read the wrong way.
"""

from __future__ import annotations

import pytest

from app.evaluate import _hit, _keyword_correctness


def test_a_matching_filename_is_a_hit():
    assert _hit(["02-leave.md"], ["02-leave.md", "07-vendors.md"])


def test_case_does_not_matter():
    assert _hit(["02-Leave.MD"], ["02-leave.md"])


def test_no_expected_documents_is_never_a_hit():
    # A question with nothing to retrieve cannot be scored on retrieval, and
    # returning True would inflate recall with cases that measure nothing.
    assert not _hit([], ["02-leave.md"])
    assert not _hit([], [])


def test_retrieving_nothing_is_not_a_hit():
    assert not _hit(["02-leave.md"], [])


def test_one_of_several_expected_documents_is_enough():
    # This is the property that made recall saturate at eight documents: a
    # multi-hop question needing two sources scores a hit on either one.
    # docs/retrieval-ablation.md says so where the number is published.
    assert _hit(["02-leave.md", "06-seattle.md"], ["06-seattle.md"])


def test_a_substring_of_a_filename_counts():
    # Deliberate, so an expected value can name a document without its
    # extension -- and worth pinning, because it is also how a careless
    # expected value could match a distractor it was never meant to.
    assert _hit(["02-leave"], ["02-leave.md"])
    assert _hit(["leave"], ["02-leave.md"])


def test_an_unrelated_document_is_not_a_hit():
    assert not _hit(["02-leave.md"], ["d4145.md", "07-vendors.md"])


@pytest.mark.parametrize(
    ("answer", "points", "expected"),
    [
        ("first year is 15 working days", ["15"], 1.0),
        ("first year is 15 working days", ["15", "20"], 0.5),
        ("no idea", ["15", "20"], 0.0),
        ("FIFTEEN and 15", ["15"], 1.0),
    ],
)
def test_keyword_correctness_is_the_share_of_key_points_present(answer, points, expected):
    assert _keyword_correctness(answer, points) == expected


def test_keyword_correctness_with_no_key_points_is_zero_not_one():
    # An empty rubric scoring 1.0 would make every unscoreable case look
    # perfect, which is the direction that flatters.
    assert _keyword_correctness("anything at all", []) == 0.0


def test_keyword_correctness_rewards_substrings_not_meaning():
    # The keyless fallback is lexical. It scores a refusal that happens to
    # quote the figure, and this is why the LLM judge exists.
    assert _keyword_correctness("I cannot say whether it is 15 days", ["15"]) == 1.0


class _FakePipeline:
    """Answers or abstains according to a per-question script."""

    def __init__(self, abstain_on: set[str]) -> None:
        self.abstain_on = abstain_on

    async def ainvoke(self, question: str, use_cache: bool = False) -> dict:
        abstained = question in self.abstain_on
        return {
            "answer": "" if abstained else "the answer is 15",
            "citations": [] if abstained else [{"filename": "02-leave.md", "doc_id": "d1"}],
            "faithfulness": 0.0 if abstained else 1.0,
            "abstained": abstained,
            "prompt_tokens": 100,
            "tokens_saved_vs_naive": 100,
            "retrieval_ms": 10.0,
        }


def _cases() -> list[dict]:
    answerable = [
        {
            "id": f"a{i}",
            "question": f"answerable {i}",
            "expected_docs": ["02-leave.md"],
            "key_points": ["15"],
        }
        for i in range(4)
    ]
    refusable = [
        {
            "id": f"r{i}",
            "question": f"unanswerable {i}",
            "expected_docs": [],
            "key_points": [],
            "expect_abstain": True,
        }
        for i in range(2)
    ]
    return answerable + refusable


async def _run(monkeypatch, abstain_on: set[str]):
    from app import evaluate

    monkeypatch.setattr(evaluate, "load_golden", _cases)
    return await evaluate.run_eval(_FakePipeline(abstain_on))


async def test_a_perfect_run_scores_both_directions(monkeypatch):
    report = await _run(monkeypatch, {"unanswerable 0", "unanswerable 1"})
    assert report.abstention_accuracy == 1.0
    assert report.over_abstention_rate == 0.0


async def test_refusing_a_question_the_corpus_answers_is_counted(monkeypatch):
    # The failure the old metric could not see. `abstention_accuracy` stays
    # perfect here -- every case that should abstain did -- and the only thing
    # that moves is the new rate.
    report = await _run(
        monkeypatch, {"unanswerable 0", "unanswerable 1", "answerable 0", "answerable 1"}
    )
    assert report.abstention_accuracy == 1.0
    assert report.over_abstention_rate == 0.5


async def test_the_two_rates_are_independent(monkeypatch):
    # Answers everything: fails every refusal, over-abstains on nothing.
    report = await _run(monkeypatch, set())
    assert report.abstention_accuracy == 0.0
    assert report.over_abstention_rate == 0.0


async def test_refusing_everything_is_not_a_perfect_score(monkeypatch):
    # The degenerate pipeline the one-directional metric rewarded: refuse every
    # question and score 1.000 on abstention. It now also scores 1.000 on the
    # rate that says it is useless.
    report = await _run(monkeypatch, {c["question"] for c in _cases()})
    assert report.abstention_accuracy == 1.0
    assert report.over_abstention_rate == 1.0


async def test_every_case_is_scored_in_both_directions(monkeypatch):
    # `abstention_correct` used to be None for anything not expecting an
    # abstention, which is what made the metric one-sided.
    report = await _run(monkeypatch, {"unanswerable 0"})
    assert all(c.abstention_correct is not None for c in report.cases)
    wrong = [c.id for c in report.cases if not c.abstention_correct]
    assert wrong == ["r1"]
