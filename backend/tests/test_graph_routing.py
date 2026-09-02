from app.evaluate import _hit
from app.graph import Pipeline, _keyword_query


def test_keyword_query_strips_hyde():
    assert (
        _keyword_query("contract KV-2025-441\nHypothetical answer here.", "original")
        == "contract KV-2025-441"
    )
    assert _keyword_query("", "original") == "original"
    assert _keyword_query(None, "original") == "original"


def test_after_faith_regenerates_once_then_abstains():
    pipeline = Pipeline.__new__(Pipeline)
    assert pipeline._after_faith({"faithfulness": 0.85, "gen_retries": 0}) == "end"
    assert pipeline._after_faith({"faithfulness": 0.4, "gen_retries": 0}) == "generate"
    assert pipeline._after_faith({"faithfulness": 0.4, "gen_retries": 1}) == "abstain"


def test_retrieval_hit_requires_expected_docs():
    assert _hit([], ["01-company.md"]) is False
    assert _hit(["02-leave.md"], ["02-leave.md"]) is True
    assert _hit(["02-leave.md"], ["01-company.md"]) is False
