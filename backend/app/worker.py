from __future__ import annotations

import asyncio
import logging

import redis.asyncio as redis
import structlog

from app.config import settings
from app.hybrid import BM25Index
from app.indexer import Indexer
from app.obs import Cache
from app.redis_client import create_redis
from app.store_docs import Catalog, IndexQueue
from app.vectors import VectorStore

log = structlog.get_logger()


async def run_worker() -> None:
    logging.basicConfig(level=settings.log_level)
    r = create_redis()
    cache = Cache(r)
    vectors = VectorStore()
    vectors.ensure()
    catalog = Catalog(r)
    bm25 = BM25Index()
    bm25.rebuild(vectors.scroll_all())
    indexer = Indexer(cache, vectors, catalog, bm25)
    queue = IndexQueue(r)
    await queue.start()
    log.info("worker.started", kafka=bool(settings.kafka_brokers))
    try:
        async for envelope in queue.jobs():
            job = envelope.job
            try:
                n = await indexer.index_job(job)
                await queue.ack(envelope)
                log.info("worker.indexed", doc_id=job.get("doc_id"), chunks=n)
            except Exception:
                log.exception("worker.failed", doc_id=job.get("doc_id"))
                rec = await catalog.get(job["doc_id"])
                if rec:
                    rec.status = "failed"
                    rec.error = "index_failed"
                    await catalog.upsert(rec)
    finally:
        await queue.close()
        await r.aclose()


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
