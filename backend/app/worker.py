from __future__ import annotations

import asyncio
import logging

import structlog

from app.config import settings
from app.indexer import Indexer
from app.metrics import INDEX_JOBS, set_corpus_size
from app.obs import Cache
from app.redis_client import create_redis
from app.startup import await_dependency
from app.store_docs import Catalog, IndexQueue
from app.vectors import VectorStore

log = structlog.get_logger()


async def run_worker() -> None:
    logging.basicConfig(level=settings.log_level)
    # The worker serves no HTTP, so without its own listener its counters are
    # unscrapeable and index throughput is invisible. Every process needs its
    # own target; Prometheus does the summing.
    if settings.metrics_port:
        from prometheus_client import start_http_server

        start_http_server(settings.metrics_port)
        log.info("metrics.listening", port=settings.metrics_port)
    r = create_redis()
    await await_dependency("redis", r.ping)
    cache = Cache(r)
    vectors = VectorStore()
    await await_dependency("qdrant", lambda: asyncio.to_thread(vectors.client.get_collections))
    vectors.ensure()
    catalog = Catalog(r)
    indexer = Indexer(cache, vectors, catalog)
    queue = IndexQueue(r)
    await queue.start()
    log.info("worker.started", kafka=bool(settings.kafka_brokers), queue="poll-v3")
    try:
        while True:
            try:
                async for envelope in queue.jobs():
                    job = envelope.job
                    try:
                        n = await indexer.index_job(job)
                        await queue.ack(envelope)
                        INDEX_JOBS.labels(outcome="indexed").inc()
                        set_corpus_size(*await catalog.counts())
                        log.info("worker.indexed", doc_id=job.get("doc_id"), chunks=n)
                    except Exception:
                        INDEX_JOBS.labels(outcome="failed").inc()
                        log.exception("worker.failed", doc_id=job.get("doc_id"))
                        rec = await catalog.get(job.get("doc_id", ""))
                        if rec:
                            rec.status = "failed"
                            rec.error = "index_failed"
                            await catalog.upsert(rec)
                        await queue.ack(envelope)
            except Exception:
                log.exception("worker.loop.retry")
                await asyncio.sleep(1)
    finally:
        await queue.close()
        await r.aclose()


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
