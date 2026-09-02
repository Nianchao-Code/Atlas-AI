#!/usr/bin/env python3
"""Eval regression gate for CI and local releases.

Smoke mode (default in CI): no API key required, cross-encoder disabled.
Full mode: requires OPENAI_API_KEY and runs LLM-as-judge metrics.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


from app.config import settings
from app.evaluate import run_eval
from app.graph import Pipeline
from app.hybrid import BM25Index
from app.indexer import Indexer
from app.llm import llm_configured
from app.obs import Cache
from app.redis_client import create_redis
from app.store_docs import Catalog, IndexQueue
from app.vectors import VectorStore


def load_thresholds(mode: str) -> dict:
    path = ROOT / "samples" / "eval" / "thresholds.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data[mode]


async def seed_corpus(catalog: Catalog, queue: IndexQueue, indexer: Indexer) -> None:
    corpus = ROOT / "samples" / "corpus"
    for path in sorted(corpus.glob("*")):
        if path.suffix.lower() not in {".md", ".txt", ".pdf"}:
            continue
        doc_id = f"seed-{path.stem}"[:24]
        from app.models import DocumentRecord

        rec = DocumentRecord(
            id=doc_id, filename=path.name, bytes=path.stat().st_size, status="queued"
        )
        await catalog.upsert(rec)
        await indexer.index_job({"doc_id": doc_id, "filename": path.name, "path": str(path)})


async def run_gate(mode: str) -> int:
    if mode == "full" and not llm_configured():
        print("FULL mode requires OPENAI_API_KEY", file=sys.stderr)
        return 2

    thresholds = load_thresholds(mode)
    if mode == "smoke":
        settings.enable_cross_encoder = False

    r = create_redis()
    cache = Cache(r)
    vectors = VectorStore()
    vectors.ensure()
    catalog = Catalog(r)
    bm25 = BM25Index()
    queue = IndexQueue(r)
    await queue.start()
    indexer = Indexer(cache, vectors, catalog)
    pipeline = Pipeline(cache, vectors, bm25)

    await seed_corpus(catalog, queue, indexer)
    # Indexing only bumps bm25:rev; building the sparse snapshot from it is the
    # reader's job, here as in the API.
    await pipeline.warm()
    report = await run_eval(pipeline, limit=None)
    await queue.close()
    await r.aclose()

    checks = [
        (
            "retrieval_recall",
            report.retrieval_recall,
            thresholds.get("min_retrieval_recall", 0),
            "min",
        ),
        (
            "abstention_accuracy",
            report.abstention_accuracy,
            thresholds.get("min_abstention_accuracy", 0),
            "min",
        ),
        (
            "hallucination_rate",
            report.hallucination_rate,
            thresholds.get("max_hallucination_rate", 1),
            "max",
        ),
    ]
    if mode == "full":
        checks.extend(
            [
                (
                    "mean_faithfulness",
                    report.mean_faithfulness,
                    thresholds["min_mean_faithfulness"],
                    "min",
                ),
                (
                    "mean_correctness",
                    report.mean_correctness,
                    thresholds["min_mean_correctness"],
                    "min",
                ),
                (
                    "token_reduction_pct",
                    report.token_reduction_pct,
                    thresholds["min_token_reduction_pct"],
                    "min",
                ),
            ]
        )

    failed = []
    for name, value, bound, kind in checks:
        if bound is None:
            continue
        ok = value >= bound if kind == "min" else value <= bound
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {name}={value:.3f} ({kind} {bound})")
        if not ok:
            failed.append(name)

    print(json.dumps(report.model_dump(), indent=2)[:2000])
    if failed:
        print(f"Eval gate failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"Eval gate passed ({mode})")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Smoke thresholds, no LLM required")
    parser.add_argument(
        "--full", action="store_true", help="Full thresholds, requires OPENAI_API_KEY"
    )
    args = parser.parse_args()
    mode = "full" if args.full else "smoke"
    raise SystemExit(asyncio.run(run_gate(mode)))


if __name__ == "__main__":
    main()
