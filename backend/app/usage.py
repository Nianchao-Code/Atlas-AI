"""Count what the model calls actually cost, instead of estimating afterwards.

Every number in this repo is measured except one: what a run costs. That gap
produced a wrong answer today. Asked how much a full ablation would be, the
estimate used 232 prompt tokens per question -- the figure the *full* pipeline
reports after LLM grading trims the context. Most ablation rows run with
grading off and pack ~1000 tokens instead, on the more expensive model. The
estimate was low by roughly half, and the account ran dry mid-run.

So: every call records its own usage, keyed by model, and a run reports the
tokens it actually spent. Tokens are the measurement. Dollars are arithmetic on
top of a rate table that is configuration, not fact -- prices change, and a
price compiled into source is wrong the moment it is committed. The report
echoes the rates it used so the arithmetic can be checked rather than believed.

The ledger is per-process, like the SLI counters and for the same reason. A
harness that runs the pipeline in its own process sees its own spend, which is
exactly what it wants; a serving replica sees only its own share.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

import structlog

from app.config import settings

log = structlog.get_logger()


@dataclass(frozen=True)
class ModelUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __add__(self, other: ModelUsage) -> ModelUsage:
        return ModelUsage(
            calls=self.calls + other.calls,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )

    def __sub__(self, other: ModelUsage) -> ModelUsage:
        return ModelUsage(
            calls=self.calls - other.calls,
            prompt_tokens=self.prompt_tokens - other.prompt_tokens,
            completion_tokens=self.completion_tokens - other.completion_tokens,
        )


Snapshot = dict[str, ModelUsage]


class UsageLedger:
    def __init__(self) -> None:
        self._by_model: dict[str, ModelUsage] = {}

    def record(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        current = self._by_model.get(model, ModelUsage())
        self._by_model[model] = replace(
            current,
            calls=current.calls + 1,
            prompt_tokens=current.prompt_tokens + max(0, prompt_tokens),
            completion_tokens=current.completion_tokens + max(0, completion_tokens),
        )

    def snapshot(self) -> Snapshot:
        return dict(self._by_model)

    def reset(self) -> None:
        self._by_model.clear()


ledger = UsageLedger()


def since(before: Snapshot, after: Snapshot | None = None) -> Snapshot:
    """What was spent between two snapshots.

    Taking a baseline rather than resetting means a harness can measure one
    phase without erasing what another phase already counted.
    """
    after = ledger.snapshot() if after is None else after
    out: Snapshot = {}
    for model, used in after.items():
        delta = used - before.get(model, ModelUsage())
        if delta.calls > 0:
            out[model] = delta
    return out


def rates() -> dict[str, tuple[float, float]]:
    """USD per million tokens, (input, output), from MODEL_PRICES.

    Configuration on purpose. The defaults below were taken from the published
    price list and are not authoritative -- check them, and if they are wrong
    the fix is one setting rather than a hunt through the code.
    """
    try:
        raw = json.loads(settings.model_prices or "{}")
    except json.JSONDecodeError:
        log.warning("model_prices.unparseable", value=settings.model_prices[:80])
        return {}
    out: dict[str, tuple[float, float]] = {}
    for model, pair in raw.items():
        try:
            inp, outp = pair
            out[model] = (float(inp), float(outp))
        except (TypeError, ValueError):
            log.warning("model_prices.bad_entry", model=model)
    return out


def cost_usd(usage: Snapshot) -> tuple[dict[str, float], float | None]:
    """Per-model and total dollars, or None where no rate is configured.

    Returning None rather than zero for an unpriced model matters: zero reads
    as "this was free", which is the wrong thing to believe about a model
    nobody has entered a price for.
    """
    table = rates()
    per_model: dict[str, float] = {}
    priced_all = bool(usage)
    for model, used in usage.items():
        rate = table.get(model)
        if rate is None:
            priced_all = False
            continue
        inp, outp = rate
        per_model[model] = (used.prompt_tokens * inp + used.completion_tokens * outp) / 1_000_000
    total = round(sum(per_model.values()), 4) if priced_all else None
    return {m: round(v, 4) for m, v in per_model.items()}, total
