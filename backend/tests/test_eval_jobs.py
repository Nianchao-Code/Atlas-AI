"""The eval job runner.

What matters here is what the old shape got wrong: two clicks must not become
two runs against a paid API, a run must survive the client that started it, and
a process that dies mid-run must not leave a record claiming to be running
forever.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from app.eval_jobs import ACTIVE_KEY, STALE_SECONDS, EvalJob, EvalRunner
from app.models import EvalReport


class _FakeRedis:
    """Enough Redis for the runner: string get/set with NX, expire, delete."""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.expiry: dict[str, float] = {}

    async def set(self, key, value, nx: bool = False, ex: int | None = None):
        if nx and key in self.data:
            return None
        self.data[key] = value
        if ex:
            self.expiry[key] = ex
        return True

    async def get(self, key):
        return self.data.get(key)

    async def delete(self, key):
        self.data.pop(key, None)
        self.expiry.pop(key, None)

    async def expire(self, key, seconds):
        if key in self.data:
            self.expiry[key] = seconds


def _report(n: int = 2) -> EvalReport:
    return EvalReport(
        n=n,
        retrieval_recall=1.0,
        mean_context_precision=0.5,
        mean_faithfulness=1.0,
        mean_correctness=1.0,
        hallucination_rate=0.0,
        abstention_accuracy=1.0,
        p95_retrieval_ms=10.0,
        mean_prompt_tokens=100.0,
        naive_prompt_tokens=200.0,
        token_reduction_pct=50.0,
        cases=[],
    )


@pytest.fixture
def runner(monkeypatch):
    return EvalRunner(_FakeRedis(), pipeline=object())


async def _settle():
    """Let the background task run to completion."""
    for _ in range(50):
        await asyncio.sleep(0)


async def test_a_run_finishes_after_start_returns(runner, monkeypatch):
    started = asyncio.Event()

    async def fake_eval(pipeline, limit=None, on_case=None):
        started.set()
        if on_case:
            await on_case(1, 2)
            await on_case(2, 2)
        return _report()

    monkeypatch.setattr("app.eval_jobs.run_eval", fake_eval)

    job = await runner.start()
    # The point of the change: start does not wait for the work.
    assert job.status == "running"
    assert job.report is None

    await asyncio.wait_for(started.wait(), timeout=2)
    await _settle()

    finished = await runner.get(job.id)
    assert finished is not None
    assert finished.status == "done"
    assert finished.report["n"] == 2
    assert (finished.done, finished.total) == (2, 2)


async def test_a_second_start_joins_the_run_in_progress(runner, monkeypatch):
    release = asyncio.Event()

    async def slow_eval(pipeline, limit=None, on_case=None):
        await release.wait()
        return _report()

    monkeypatch.setattr("app.eval_jobs.run_eval", slow_eval)

    first = await runner.start()
    second = await runner.start()

    # One run, not two. The old endpoint billed for both.
    assert second.id == first.id
    assert second.joined is True

    release.set()
    await _settle()


async def test_the_claim_is_released_when_the_run_finishes(runner, monkeypatch):
    async def fake_eval(pipeline, limit=None, on_case=None):
        return _report()

    monkeypatch.setattr("app.eval_jobs.run_eval", fake_eval)

    first = await runner.start()
    await _settle()
    assert await runner.r.get(ACTIVE_KEY) is None

    second = await runner.start()
    assert second.id != first.id
    assert second.joined is False
    await _settle()


async def test_a_failed_run_records_why(runner, monkeypatch):
    async def boom(pipeline, limit=None, on_case=None):
        raise RuntimeError("no credits remaining")

    monkeypatch.setattr("app.eval_jobs.run_eval", boom)

    job = await runner.start()
    await _settle()

    failed = await runner.get(job.id)
    assert failed.status == "failed"
    assert "no credits remaining" in failed.error
    # And the claim is gone, so the next run is not blocked by the failure.
    assert await runner.r.get(ACTIVE_KEY) is None


async def test_a_job_that_stopped_beating_is_reported_abandoned(runner):
    # The one failure Redis cannot cover: the process holding the task died, so
    # nobody is left who can honestly mark the record failed.
    stale = EvalJob(
        id="dead",
        status="running",
        started_at=time.time() - 3600,
        updated_at=time.time() - STALE_SECONDS - 10,
        done=7,
        total=53,
    )
    await runner.r.set(
        "eval:job:dead", json.dumps({k: v for k, v in stale.__dict__.items() if k != "joined"})
    )

    job = await runner.get("dead")
    assert job.status == "failed"
    assert "abandoned" in job.error
    # Progress is preserved rather than zeroed: it says how far it got.
    assert (job.done, job.total) == (7, 53)


async def test_a_stale_claim_does_not_block_new_runs(runner, monkeypatch):
    async def fake_eval(pipeline, limit=None, on_case=None):
        return _report()

    monkeypatch.setattr("app.eval_jobs.run_eval", fake_eval)

    # A pointer left behind by a process that died, with no readable record.
    await runner.r.set(ACTIVE_KEY, "ghost")

    job = await runner.start()
    assert job.joined is False
    assert job.status == "running"
    await _settle()
    assert (await runner.get(job.id)).status == "done"


async def test_latest_points_at_the_most_recent_run(runner, monkeypatch):
    async def fake_eval(pipeline, limit=None, on_case=None):
        return _report()

    monkeypatch.setattr("app.eval_jobs.run_eval", fake_eval)

    first = await runner.start()
    await _settle()
    second = await runner.start()
    await _settle()

    latest = await runner.latest()
    assert latest.id == second.id != first.id


async def test_the_case_count_is_known_before_the_first_case_finishes(runner, monkeypatch):
    # Otherwise the UI shows "0 of ?" for however long one model call takes,
    # which is the same non-answer the old spinner gave.
    release = asyncio.Event()

    async def slow_eval(pipeline, limit=None, on_case=None):
        await release.wait()
        return _report()

    monkeypatch.setattr("app.eval_jobs.run_eval", slow_eval)

    job = await runner.start(total=53)
    assert (job.done, job.total) == (0, 53)
    assert (await runner.get(job.id)).total == 53

    release.set()
    await _settle()
