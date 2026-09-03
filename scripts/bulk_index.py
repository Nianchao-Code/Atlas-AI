#!/usr/bin/env python3
"""Index a large corpus directly, bypassing the upload API and the job queue.

Ten thousand documents through POST /api/v1/documents would be ten thousand
HTTP requests, ten thousand Redis Stream messages and ten thousand single-file
embedding calls. That path exists so a person can add a document; it is the
wrong shape for a corpus load, and pretending otherwise would measure the queue
rather than retrieval.

So this does what a bulk loader does: chunk everything, embed in large batches,
and write Qdrant and the catalogue in bulk. It uses the same `chunk_document`
and the same `VectorStore.upsert` as the worker, so the vectors are identical
to what the normal path would have produced -- only the delivery differs.

Resumable, because a rate limit or a dropped connection halfway through 50k
chunks should not mean paying to embed the first half again: documents already
present in Qdrant are skipped.

    python scripts/bulk_index.py --source samples/distractors --dry-run
    python scripts/bulk_index.py --source samples/distractors
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (ROOT / "backend", Path("/app")):
    if (_candidate / "app" / "__init__.py").exists():
        sys.path.insert(0, str(_candidate))
        break

from app.chunking import chunk_document  # noqa: E402
from app.llm import embed_texts  # noqa: E402
from app.models import DocumentRecord  # noqa: E402
from app.redis_client import create_redis  # noqa: E402
from app.usage import cost_usd, ledger, since  # noqa: E402
from app.vectors import VectorStore  # noqa: E402


def doc_id_for(path: Path) -> str:
    """Stable and derivable, so a rerun skips rather than duplicates."""
    return f"bulk-{path.stem}"[:48]


async def with_retry(label: str, fn, attempts: int = 6):
    """Back off and retry rather than losing a batch.

    Both sides of a bulk load fail transiently: the embedding API rate-limits,
    and Qdrant times out. An earlier version wrapped only the embedding call,
    so a write failure discarded vectors that had already been paid for.
    """
    delay = 2.0
    for attempt in range(1, attempts + 1):
        try:
            return await fn()
        except Exception as exc:
            if attempt == attempts:
                raise
            print(
                f"    {label} retry {attempt}/{attempts - 1} after {type(exc).__name__}, "
                f"sleeping {delay:.0f}s",
                file=sys.stderr,
                flush=True,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)
    raise RuntimeError("unreachable")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path("samples/distractors"))
    ap.add_argument("--batch", type=int, default=256, help="chunks per embedding call")
    ap.add_argument("--docs-per-flush", type=int, default=500)
    ap.add_argument("--concurrency", type=int, default=6, help="parallel embedding calls")
    ap.add_argument(
        "--upsert-batch",
        type=int,
        default=256,
        help="points per Qdrant write; 2,587 was 17.5MB and timed out",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="chunk and count, spend nothing")
    args = ap.parse_args()

    paths = sorted(args.source.glob("*.md"))[: args.limit]
    if not paths:
        print(f"no .md files under {args.source}", file=sys.stderr)
        return 2

    vectors = VectorStore()
    vectors.ensure()
    existing = await asyncio.to_thread(vectors.doc_ids)
    todo = [p for p in paths if doc_id_for(p) not in existing]
    print(
        f"{len(paths):,} documents, {len(paths) - len(todo):,} already indexed, {len(todo):,} to do"
    )

    if args.dry_run:
        sample = todo[:200]
        n_chunks = sum(
            len(
                chunk_document(
                    doc_id=doc_id_for(p), filename=p.name, text=p.read_text(encoding="utf-8")
                )
            )
            for p in sample
        )
        per_doc = n_chunks / len(sample)
        total = per_doc * len(todo)
        print(f"  {per_doc:.1f} chunks/document -> {total:,.0f} chunks")
        print(f"  ~{total * 60 / 1e6:.1f}M tokens, ~${total * 60 / 1e6 * 0.02:.2f} to embed")
        print("  dry run, nothing written")
        return 0

    r = create_redis()
    spent_before = ledger.snapshot()
    started = time.time()
    done_docs = done_chunks = 0
    pending_chunks: list = []
    pending_records: list[DocumentRecord] = []

    async def write_records(status: str) -> None:
        pipe = r.pipeline()
        for rec in pending_records:
            rec.status = status  # type: ignore[assignment]
            pipe.hset("docs", rec.id, rec.model_dump_json())
        await pipe.execute()

    async def flush() -> None:
        nonlocal pending_chunks, pending_records, done_chunks
        if not pending_chunks:
            return
        # Claim the documents as `indexing` BEFORE writing their vectors.
        #
        # The first version wrote vectors first and the catalogue second, and
        # the reconciler -- which deletes vectors whose document the catalogue
        # does not list -- ran inside that window and removed 107 documents.
        # It was not wrong; the window was. Writing the record first makes the
        # transient state "listed but not yet retrievable", which reconcile
        # skips while it says `indexing` and requeues if the loader dies. The
        # recoverable failure instead of the destructive one.
        await write_records("indexing")
        # Embedding batches are independent, and running them one at a time
        # measured 3.1 doc/s -- 54 minutes for this corpus and nine hours for a
        # ten-times-larger one. Bounded concurrency, because the ceiling here is
        # the provider's rate limit rather than anything local.
        batches = [
            pending_chunks[i : i + args.batch] for i in range(0, len(pending_chunks), args.batch)
        ]
        gate = asyncio.Semaphore(args.concurrency)

        async def embed_batch(b):
            async with gate:
                return await with_retry("embed", lambda: embed_texts([c.text for c in b]))

        # gather preserves order, which is what keeps vectors aligned to chunks.
        vecs: list[list[float]] = []
        for got in await asyncio.gather(*(embed_batch(b) for b in batches)):
            vecs.extend(got)
        # Batched for the same reason as the embeddings, which an earlier
        # version missed: flushing 500 documents meant 2,587 points and a
        # 17.5MB request body against a 30s client timeout, and it timed out
        # every time. The flush size was tuned for the embedding API and never
        # checked against the write side.
        for i in range(0, len(pending_chunks), args.upsert_batch):
            cs = pending_chunks[i : i + args.upsert_batch]
            vs = vecs[i : i + args.upsert_batch]
            await with_retry("upsert", lambda c=cs, v=vs: asyncio.to_thread(vectors.upsert, c, v))
        await write_records("ready")
        done_chunks += len(pending_chunks)
        pending_chunks, pending_records = [], []

    try:
        for n, path in enumerate(todo, 1):
            doc_id = doc_id_for(path)
            text = path.read_text(encoding="utf-8")
            chunks = chunk_document(doc_id=doc_id, filename=path.name, text=text)
            pending_chunks.extend(chunks)
            pending_records.append(
                DocumentRecord(
                    id=doc_id,
                    filename=path.name,
                    bytes=len(text.encode("utf-8")),
                    status="indexing",
                    chunks=len(chunks),
                )
            )
            done_docs = n
            if len(pending_records) >= args.docs_per_flush:
                await flush()
                rate = done_docs / max(1e-9, time.time() - started)
                eta = (len(todo) - done_docs) / max(rate, 1e-9)
                _pm, usd = cost_usd(since(spent_before))
                money = f"  ${usd:.3f}" if usd is not None else ""
                print(
                    f"  {done_docs:>7,}/{len(todo):,} docs  {done_chunks:>8,} chunks  "
                    f"{rate:>5.1f} doc/s  eta {eta / 60:>4.1f}m{money}",
                    flush=True,
                )
        await flush()
    finally:
        await r.aclose()

    used = since(spent_before)
    per_model, usd = cost_usd(used)
    elapsed = time.time() - started
    print(f"\nindexed {done_docs:,} documents / {done_chunks:,} chunks in {elapsed / 60:.1f} min")
    for model, u in sorted(used.items()):
        print(
            f"  {model:26} {u.calls:>6,} calls  {u.prompt_tokens:>10,} tokens  "
            f"${per_model.get(model, 0):.4f}"
        )
    if usd is not None:
        print(f"  total ${usd:.4f}")
    print(f"  collection now holds {vectors.count():,} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
