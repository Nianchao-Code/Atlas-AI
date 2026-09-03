"""Token accounting.

This exists because an estimate was wrong by roughly half and the account ran
dry mid-measurement. What these pin is the difference between a figure that can
be checked and one that has to be believed: tokens are counted, dollars are
arithmetic over a rate table, and a model with no rate reports null rather than
zero.
"""

from __future__ import annotations

import json

import pytest

from app.config import settings
from app.usage import ModelUsage, UsageLedger, cost_usd, rates, since


@pytest.fixture
def led():
    return UsageLedger()


def test_calls_accumulate_per_model(led):
    led.record("gpt-4o", 100, 20)
    led.record("gpt-4o", 50, 10)
    led.record("gpt-4o-mini", 300, 5)

    snap = led.snapshot()
    assert snap["gpt-4o"] == ModelUsage(calls=2, prompt_tokens=150, completion_tokens=30)
    assert snap["gpt-4o-mini"] == ModelUsage(calls=1, prompt_tokens=300, completion_tokens=5)


def test_a_snapshot_is_not_a_live_view(led):
    led.record("gpt-4o", 100, 20)
    snap = led.snapshot()
    led.record("gpt-4o", 999, 999)
    # Otherwise a baseline taken before a run would silently include the run.
    assert snap["gpt-4o"].prompt_tokens == 100


def test_since_reports_only_what_happened_after_the_baseline(led):
    led.record("gpt-4o", 100, 20)
    baseline = led.snapshot()
    led.record("gpt-4o", 40, 5)
    led.record("gpt-4o-mini", 10, 1)

    delta = since(baseline, led.snapshot())
    assert delta["gpt-4o"] == ModelUsage(calls=1, prompt_tokens=40, completion_tokens=5)
    assert delta["gpt-4o-mini"].calls == 1


def test_a_model_untouched_since_the_baseline_is_omitted(led):
    led.record("gpt-4o", 100, 20)
    baseline = led.snapshot()
    led.record("gpt-4o-mini", 10, 1)

    delta = since(baseline, led.snapshot())
    assert "gpt-4o" not in delta


def test_cost_is_arithmetic_over_the_configured_rates(monkeypatch):
    monkeypatch.setattr(settings, "model_prices", json.dumps({"m": [2.0, 10.0]}))
    usage = {"m": ModelUsage(calls=1, prompt_tokens=1_000_000, completion_tokens=100_000)}
    per_model, total = cost_usd(usage)
    assert per_model["m"] == pytest.approx(2.0 + 1.0)
    assert total == pytest.approx(3.0)


def test_an_unpriced_model_reports_null_rather_than_free(monkeypatch):
    # Zero would read as "this cost nothing", which is exactly the wrong thing
    # to believe about a model nobody has entered a price for.
    monkeypatch.setattr(settings, "model_prices", json.dumps({"priced": [1.0, 1.0]}))
    usage = {
        "priced": ModelUsage(calls=1, prompt_tokens=1_000_000, completion_tokens=0),
        "mystery": ModelUsage(calls=1, prompt_tokens=1_000_000, completion_tokens=0),
    }
    per_model, total = cost_usd(usage)
    assert per_model == {"priced": 1.0}
    assert total is None


def test_a_broken_price_table_degrades_to_tokens_only(monkeypatch):
    monkeypatch.setattr(settings, "model_prices", "{not json")
    assert rates() == {}
    _per_model, total = cost_usd({"m": ModelUsage(calls=1, prompt_tokens=10, completion_tokens=1)})
    assert total is None


def test_one_bad_entry_does_not_discard_the_rest(monkeypatch):
    monkeypatch.setattr(
        settings, "model_prices", json.dumps({"good": [1.0, 2.0], "bad": "nonsense"})
    )
    table = rates()
    assert table == {"good": (1.0, 2.0)}


def test_negative_token_counts_cannot_reduce_the_total(led):
    # A provider returning something odd should not make a run look cheaper
    # than a run that happened.
    led.record("gpt-4o", -5, -5)
    assert led.snapshot()["gpt-4o"] == ModelUsage(calls=1, prompt_tokens=0, completion_tokens=0)
