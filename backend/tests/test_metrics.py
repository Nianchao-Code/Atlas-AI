"""Prometheus instrumentation.

The interesting assertions are not "a counter went up". They are that a cache
hit does not land in the retrieval histogram, that /metrics is reachable
without a key while every /api/v1 route is not, and that no series carries a
question or an answer.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import metrics as m
from app.config import settings

KEYS = "alice:secret-alice"


def _value(counter, **labels) -> float:
    return counter.labels(**labels)._value.get() if labels else counter._value.get()


def _hist_count(hist) -> float:
    return hist._sum.get(), sum(b.get() for b in hist._buckets)


# ------------------------------------------------------------- outcomes ---

def test_answered_query_records_latency_and_tokens():
    before_sum, before_count = _hist_count(m.RETRIEVAL_SECONDS)
    before = _value(m.QUERIES, outcome="answered")

    m.observe_query(retrieval_ms=250.0, prompt_tokens=400, cache_hit=False)

    assert _value(m.QUERIES, outcome="answered") == before + 1
    after_sum, after_count = _hist_count(m.RETRIEVAL_SECONDS)
    assert after_count == before_count + 1
    assert after_sum == pytest.approx(before_sum + 0.25)


def test_cache_hit_is_its_own_outcome_and_skips_the_latency_histogram():
    before_cached = _value(m.QUERIES, outcome="cached")
    before_answered = _value(m.QUERIES, outcome="answered")
    _, before_count = _hist_count(m.RETRIEVAL_SECONDS)
    before_hits = _value(m.CACHE_HITS, kind="semantic")

    m.observe_query(retrieval_ms=0.4, prompt_tokens=0, cache_hit=True)

    assert _value(m.QUERIES, outcome="cached") == before_cached + 1
    # Folding cache hits into "answered" would hide the hit rate in the query
    # rate, which is the number worth watching when the cache changes.
    assert _value(m.QUERIES, outcome="answered") == before_answered
    assert _value(m.CACHE_HITS, kind="semantic") == before_hits + 1
    # A near-zero sample would drag p95 toward zero and hide real retrieval.
    _, after_count = _hist_count(m.RETRIEVAL_SECONDS)
    assert after_count == before_count


def test_abstained_and_blocked_are_distinguishable():
    before_abstain = _value(m.QUERIES, outcome="abstained")
    before_blocked = _value(m.QUERIES, outcome="blocked")

    m.observe_query(retrieval_ms=100.0, prompt_tokens=200, cache_hit=False, abstained=True)
    m.observe_query(retrieval_ms=1.0, prompt_tokens=0, cache_hit=False, blocked=True)

    assert _value(m.QUERIES, outcome="abstained") == before_abstain + 1
    assert _value(m.QUERIES, outcome="blocked") == before_blocked + 1


def test_faithfulness_is_recorded_only_when_scored():
    _, before = _hist_count(m.FAITHFULNESS)
    m.observe_query(retrieval_ms=10.0, prompt_tokens=10, cache_hit=False, faithfulness=None)
    _, mid = _hist_count(m.FAITHFULNESS)
    assert mid == before

    m.observe_query(retrieval_ms=10.0, prompt_tokens=10, cache_hit=False, faithfulness=0.95)
    _, after = _hist_count(m.FAITHFULNESS)
    assert after == before + 1


# --------------------------------------------------------------- exposition ---

def test_render_is_prometheus_text_format():
    m.set_corpus_size(8, 27)
    payload, content_type = m.render()
    body = payload.decode()

    assert "text/plain" in content_type
    assert "# TYPE atlas_queries_total counter" in body
    assert "atlas_corpus_documents 8.0" in body
    assert "atlas_corpus_chunks 27.0" in body


# --------------------------------------------------------------- the route ---

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "api_keys", KEYS)
    monkeypatch.setattr(settings, "rate_limit_per_minute", 0)
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_metrics_endpoint_needs_no_key(client):
    """It is not proxied by nginx, so it is in-cluster only; requiring a key
    would mean distributing one to the scraper for no gain."""
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "atlas_queries_total" in res.text


def test_json_sli_endpoint_still_requires_one(client):
    # /api/v1/metrics drives the UI and stays behind auth; the two endpoints
    # are easy to confuse, so pin the difference.
    assert client.get("/api/v1/metrics").status_code == 401
    assert client.get("/api/v1/metrics", headers={"X-API-Key": "secret-alice"}).status_code == 200


def test_no_series_carries_question_or_answer_text(client):
    m.observe_query(retrieval_ms=120.0, prompt_tokens=300, cache_hit=False, faithfulness=1.0)
    body = client.get("/metrics").text

    # Only our own series; prometheus_client also ships process and GC
    # collectors whose labels are not ours to police.
    label_names = {"outcome", "kind", "principal", "le"}
    checked = 0
    for line in body.splitlines():
        if not line.startswith("atlas_") or "{" not in line:
            continue
        labels = line[line.index("{") + 1 : line.rindex("}")]
        for pair in labels.split(","):
            if pair:
                checked += 1
                assert pair.split("=")[0] in label_names, line
    assert checked, "no labelled atlas_ series found; the check proved nothing"
