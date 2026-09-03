#!/usr/bin/env python3
"""Do the added documents actually compete, or are they scenery?

Scaling the corpus is only worth anything if the new documents are hard. Ten
thousand documents that no query ever retrieves leave recall exactly as
saturated as eight did, and the ablation would spend an hour confirming it.

This asks the cheap version of that question first: for every golden question,
retrieve and count how much of the top-k comes from the distractors rather than
the eight real handbook documents. No generation and no judge -- one embedding
per question, so it costs a fraction of a cent and answers whether the
expensive measurement is worth running.

Three numbers matter:

  intrusion   share of retrieved chunks that are distractors. Zero means the
              corpus got bigger and no harder.
  displaced   questions where a distractor outranks every real document. These
              are the ones recall can finally discriminate.
  survived    questions where a real document is still rank 1.

    kubectl exec -n atlas deploy/worker -- python /tmp/distractor_probe.py
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (ROOT / "backend", Path("/app")):
    if (_candidate / "app" / "__init__.py").exists():
        sys.path.insert(0, str(_candidate))
        break

from app.config import settings  # noqa: E402
from app.evaluate import load_golden  # noqa: E402
from app.llm import embed_texts  # noqa: E402
from app.usage import cost_usd, ledger, since  # noqa: E402
from app.vectors import VectorStore  # noqa: E402


def is_real(doc_id: str) -> bool:
    """The eight handbook documents are the ones a golden answer can come from."""
    return not doc_id.startswith("bulk-")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=settings.retrieve_k)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cases = load_golden()[: args.limit]
    vectors = VectorStore()
    total = vectors.count()
    spent = ledger.snapshot()

    rows = []
    for case in cases:
        q = case["question"]
        vec = (await embed_texts([q]))[0]
        for mode, hits in (
            ("dense", await asyncio.to_thread(vectors.search, vec, args.k)),
            ("sparse", await asyncio.to_thread(vectors.search_sparse, q, args.k)),
            ("hybrid", await asyncio.to_thread(vectors.search_hybrid, vec, q, args.k)),
        ):
            ids = [h.doc_id for h in hits]
            rows.append(
                {
                    "id": case["id"],
                    "mode": mode,
                    "n": len(ids),
                    "distractors": sum(1 for d in ids if not is_real(d)),
                    "top1_real": bool(ids) and is_real(ids[0]),
                    "any_real": any(is_real(d) for d in ids),
                }
            )

    print(f"corpus: {total:,} chunks, {len(cases)} questions, top-{args.k}\n")
    print(f"{'retriever':10} {'intrusion':>10} {'top-1 real':>11} {'any real in k':>14}")
    print("-" * 48)
    for mode in ("dense", "sparse", "hybrid"):
        sub = [r for r in rows if r["mode"] == mode]
        retrieved = sum(r["n"] for r in sub)
        intr = sum(r["distractors"] for r in sub)
        top1 = sum(1 for r in sub if r["top1_real"])
        anyr = sum(1 for r in sub if r["any_real"])
        print(
            f"{mode:10} {intr / max(retrieved, 1):>9.1%} {top1}/{len(sub):>9} {anyr}/{len(sub):>12}"
        )

    worst = sorted((r for r in rows if r["mode"] == "hybrid"), key=lambda r: -r["distractors"])[:8]
    print("\nmost contested questions (hybrid):")
    for r in worst:
        print(f"  {r['id']:32} {r['distractors']:>2}/{r['n']} distractors in top-k")

    _pm, usd = cost_usd(since(spent))
    if usd is not None:
        print(f"\ncost ${usd:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
