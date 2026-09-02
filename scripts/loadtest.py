#!/usr/bin/env python3
"""Measure what one replica can actually serve.

End-to-end throughput with a model in the path measures the model provider, not
this service, so the phases here are chosen to isolate what the app itself
contributes.

  warm  Cache-hit path, no model call. A concurrency ramp against one replica,
        which is the ceiling the app can offer before the model is involved.

  hol   Head-of-line blocking. Cache hits are timed while one cold request runs
        alongside them. If any synchronous work is left on the event loop, the
        fast requests inherit the slow one's latency -- this is the property
        moving synchronous work off the event loop exists to protect, and
        it is otherwise invisible.

  cold  A small ramp on unique questions. Expected to be bounded by the model,
        and included so the comparison with warm is on the page rather than
        assumed.

Run it from inside the cluster; kubectl port-forward serialises connections and
would measure itself:

    kubectl cp scripts/loadtest.py atlas/<worker-pod>:/tmp/loadtest.py
    kubectl exec -n atlas deploy/worker -- \
      sh -c 'ATLAS_KEY=... python /tmp/loadtest.py --phase warm'
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
import uuid

import httpx

BASE = os.environ.get("ATLAS_BASE", "http://api:8000")
KEY = os.environ.get("ATLAS_KEY", "")
HEADERS = {"X-API-Key": KEY, "Content-Type": "application/json"}

WARM_QUESTION = "What is the weekly on-call stipend?"


class Sample:
    __slots__ = ("ms", "status", "cache")

    def __init__(self, ms: float, status: int, cache: str) -> None:
        self.ms = ms
        self.status = status
        self.cache = cache


async def ask(client: httpx.AsyncClient, question: str, use_cache: bool = True) -> Sample:
    t0 = time.perf_counter()
    try:
        r = await client.post(
            "/api/v1/query",
            json={"question": question, "use_cache": use_cache},
            headers=HEADERS,
            timeout=120,
        )
    except Exception:
        return Sample((time.perf_counter() - t0) * 1000, 0, "error")
    ms = (time.perf_counter() - t0) * 1000
    detail = "?"
    if r.status_code == 200:
        trace = r.json().get("trace") or []
        detail = next((n.get("detail") for n in trace if n.get("node") == "cache"), "?")
    return Sample(ms, r.status_code, detail)


def report(name: str, samples: list[Sample], wall: float) -> dict:
    ok = [s for s in samples if s.status == 200]
    lat = sorted(s.ms for s in ok)
    rate_limited = sum(1 for s in samples if s.status == 429)
    errors = sum(1 for s in samples if s.status not in (200, 429))

    def pct(p: float) -> float:
        if not lat:
            return 0.0
        return lat[min(len(lat) - 1, int(round(p / 100 * (len(lat) - 1))))]

    row = {
        "name": name,
        "n": len(samples),
        "ok": len(ok),
        "rps": len(ok) / wall if wall else 0.0,
        "p50": pct(50),
        "p95": pct(95),
        "p99": pct(99),
        "max": lat[-1] if lat else 0.0,
        # One stalled request is what head-of-line blocking looks like, and a
        # p95 over thousands of samples will not show it.
        "over_100ms": sum(1 for v in lat if v > 100),
        "429": rate_limited,
        "err": errors,
    }
    print(
        f"  {name:<22} n={row['n']:<5} {row['rps']:6.1f} rps   "
        f"p50={row['p50']:6.1f}  p95={row['p95']:6.1f}  p99={row['p99']:7.1f}  "
        f"max={row['max']:7.1f}ms   >100ms={row['over_100ms']:<4} 429={rate_limited} err={errors}"
    )
    return row


async def phase_warm(client: httpx.AsyncClient, levels: list[int], per_level: int) -> None:
    print("=== warm: cache-hit path, no model call ===")
    seed = await ask(client, WARM_QUESTION)
    print(f"  seeded ({seed.ms:.0f}ms, cache={seed.cache!r})")
    confirm = await ask(client, WARM_QUESTION)
    if confirm.cache not in ("exact hit", "near-dup hit"):
        print(
            f"  WARNING: not serving from cache (cache={confirm.cache!r}); "
            f"this phase would measure the model instead"
        )

    async def bounded(sem: asyncio.Semaphore) -> Sample:
        # Taken as an argument rather than closed over: `sem` is rebound on each
        # iteration below, and a closure over it is a bug waiting for someone
        # to move the await outside the loop.
        async with sem:
            return await ask(client, WARM_QUESTION)

    for c in levels:
        sem = asyncio.Semaphore(c)
        t0 = time.perf_counter()
        samples = await asyncio.gather(*(bounded(sem) for _ in range(per_level)))
        report(f"concurrency {c}", list(samples), time.perf_counter() - t0)


async def phase_hol(client: httpx.AsyncClient, concurrency: int, rounds: int) -> None:
    print("=== hol: do cache hits inherit a slow request's latency? ===")
    await ask(client, WARM_QUESTION)

    async def burst(n: int) -> list[Sample]:
        return list(await asyncio.gather(*(ask(client, WARM_QUESTION) for _ in range(n))))

    async def burst_series(n_bursts: int, size: int) -> list[Sample]:
        out: list[Sample] = []
        for _ in range(n_bursts):
            out.extend(await burst(size))
        return out

    # An earlier version fired concurrency*rounds at once here and bursts of
    # `concurrency` below, so the "inflation" it reported was the gap between
    # two concurrency levels.
    t0 = time.perf_counter()
    baseline = await burst_series(rounds, concurrency)
    base_row = report("baseline (quiet)", baseline, time.perf_counter() - t0)

    # A cold question runs the whole graph, including a model call, while the
    # cache hits are timed alongside it.
    cold_q = f"What is the vendor defect rate threshold? ({uuid.uuid4().hex[:6]})"
    t0 = time.perf_counter()
    cold_task = asyncio.create_task(ask(client, cold_q, use_cache=False))
    during: list[Sample] = []
    while not cold_task.done():
        during.extend(await burst(concurrency))
    cold = await cold_task
    during_row = report("during a cold query", during, time.perf_counter() - t0)
    print(f"  the cold query itself took {cold.ms:.0f}ms (status {cold.status})")

    if base_row["p95"] and during_row["p95"]:
        print(
            f"\n  p95  quiet {base_row['p95']:.1f}ms -> busy {during_row['p95']:.1f}ms "
            f"({during_row['p95'] / base_row['p95']:.2f}x)"
        )
        print(
            f"  p99  quiet {base_row['p99']:.1f}ms -> busy {during_row['p99']:.1f}ms "
            f"({during_row['p99'] / base_row['p99']:.2f}x)"
            if base_row["p99"]
            else ""
        )
        print(
            f"  requests over 100ms: quiet {base_row['over_100ms']}/{base_row['n']}, "
            f"busy {during_row['over_100ms']}/{during_row['n']}"
        )
        print(
            "  Blocking work on the event loop shows up as a multiple here, "
            "and as a cluster of slow requests rather than one outlier."
        )


async def phase_cold(client: httpx.AsyncClient, concurrency: int, total: int) -> None:
    print("=== cold: unique questions, model in the path ===")
    sem = asyncio.Semaphore(concurrency)

    async def one(i: int):
        async with sem:
            return await ask(
                client, f"What is the on-call rotation length? (probe {i})", use_cache=False
            )

    t0 = time.perf_counter()
    samples = await asyncio.gather(*(one(i) for i in range(total)))
    report(f"concurrency {concurrency}", list(samples), time.perf_counter() - t0)


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["warm", "hol", "cold", "all"], default="all")
    parser.add_argument("--levels", default="1,2,4,8,16,32")
    parser.add_argument("--per-level", type=int, default=60)
    parser.add_argument("--cold-concurrency", type=int, default=4)
    parser.add_argument("--cold-total", type=int, default=8)
    args = parser.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    limits = httpx.Limits(max_connections=200, max_keepalive_connections=200)
    async with httpx.AsyncClient(base_url=BASE, limits=limits) as client:
        health = await client.get("/health", timeout=30)
        print(f"target {BASE} -> {health.json()}\n")

        if args.phase in ("warm", "all"):
            await phase_warm(client, levels, args.per_level)
            print()
        if args.phase in ("hol", "all"):
            await phase_hol(client, concurrency=8, rounds=40)
            print()
        if args.phase in ("cold", "all"):
            await phase_cold(client, args.cold_concurrency, args.cold_total)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
