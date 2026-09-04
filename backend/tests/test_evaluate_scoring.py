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
