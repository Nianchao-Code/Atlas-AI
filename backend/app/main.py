from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import redis.asyncio as redis
import structlog
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from app.auth import (
    auth_enabled,
    cors_origins,
    enforce_rate_limit,
    key_from_headers,
    log_auth_mode,
    resolve_principal,
)
from app.chunking import parse_file
from app.config import settings
from app.evaluate import run_eval
from app.graph import Pipeline
from app.hybrid import BM25Index
from app.indexer import Indexer
from app.llm import llm_configured
from app.metrics import (
    AUTH_REJECTIONS,
    INDEX_JOBS,
    UPLOAD_REJECTIONS,
    observe_query,
    set_corpus_size,
)
from app.metrics import (
    render as render_metrics,
)
from app.models import DocumentRecord, MetricsSnapshot, QueryRequest, QueryResponse
from app.obs import Cache, obs
from app.qa_cache import QACache
from app.redis_client import create_redis
from app.startup import await_dependency
from app.store_docs import Catalog, IndexQueue
from app.uploads import UploadRejected, destination, save
from app.vectors import VectorStore

log = structlog.get_logger()


class AppState:
    redis: redis.Redis
    cache: Cache
    vectors: VectorStore
    catalog: Catalog
    bm25: BM25Index
    qa_cache: QACache | None
    queue: IndexQueue
    pipeline: Pipeline
    indexer: Indexer
    worker_task: asyncio.Task | None = None
    refresh_task: asyncio.Task | None = None


state = AppState()


def get_state() -> AppState:
    return state


async def _background_refresh() -> None:
    """Keep the BM25 snapshot and the corpus gauges fresh, off the request path.

    The worker reindexes in its own process and bumps bm25:rev; polling it here
    means no query ever pays for the rebuild. The cost is bounded staleness on
    the sparse side, which the interval names. The corpus gauges ride along
    rather than being computed during a scrape, so scraping stays O(1).
    """
    ticks = 0
    sweep_every = max(1, int(60 / max(settings.bm25_refresh_seconds, 0.1)))
    while True:
        await asyncio.sleep(settings.bm25_refresh_seconds)
        ticks += 1
        try:
            if await state.pipeline.warm():
                log.info("bm25.refreshed")
            documents, chunks = await state.catalog.counts()
            set_corpus_size(documents, chunks)
            if state.qa_cache is not None and ticks % sweep_every == 0:
                await asyncio.to_thread(state.qa_cache.purge_expired)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("background_refresh_failed")


async def _consume_index_jobs() -> None:
    """Same process as the API so the in-memory BM25 index stays coherent.

    Docker compose runs a dedicated worker instead (EMBEDDED_WORKER=false)
    and the API refreshes BM25 from Qdrant when the revision key changes.
    """
    async for envelope in state.queue.jobs(consumer="api-embedded"):
        job = envelope.job
        try:
            n = await state.indexer.index_job(job)
            await state.queue.ack(envelope)
            INDEX_JOBS.labels(outcome="indexed").inc()
            log.info("index.ok", doc_id=job.get("doc_id"), chunks=n)
        except Exception:
            INDEX_JOBS.labels(outcome="failed").inc()
            log.exception("index.failed", doc_id=job.get("doc_id"))
            rec = await state.catalog.get(job.get("doc_id", ""))
            if rec:
                rec.status = "failed"
                rec.error = "index_failed"
                await state.catalog.upsert(rec)


StateDep = Annotated[AppState, Depends(get_state)]


async def require_principal(
    s: StateDep,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> str:
    """Authenticate the caller and charge the request against its budget.

    Rate limiting lives here rather than in middleware so it runs after the key
    is known: an unauthenticated flood should be rejected as 401 without
    consuming anyone's quota.
    """
    principal = resolve_principal(key_from_headers(x_api_key, authorization))
    if principal is None:
        AUTH_REJECTIONS.inc()
        raise HTTPException(
            status_code=401,
            detail="missing or invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    await enforce_rate_limit(s.redis, principal)
    return principal


PrincipalDep = Annotated[str, Depends(require_principal)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    log_auth_mode()
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    state.redis = create_redis()
    await await_dependency("redis", state.redis.ping)
    state.cache = Cache(state.redis)
    state.vectors = VectorStore()
    await await_dependency(
        "qdrant", lambda: asyncio.to_thread(state.vectors.client.get_collections)
    )
    state.vectors.ensure()
    state.catalog = Catalog(state.redis)
    state.qa_cache = QACache()
    try:
        state.qa_cache.ensure()
    except Exception:
        # A missing paraphrase cache degrades to exact-match only; it must not
        # stop the API from serving.
        log.warning("qa_cache.unavailable")
        state.qa_cache = None
    state.bm25 = BM25Index()
    state.queue = IndexQueue(state.redis)
    await state.queue.start()
    state.indexer = Indexer(state.cache, state.vectors, state.catalog)
    state.pipeline = Pipeline(state.cache, state.vectors, state.bm25, qa_cache=state.qa_cache)
    try:
        # Build the BM25 snapshot now so the first query does not pay for it.
        await state.pipeline.warm()
    except Exception:
        log.warning("bm25.rebuild_skipped")
    state.refresh_task = asyncio.create_task(_background_refresh())
    if settings.embedded_worker:
        state.worker_task = asyncio.create_task(_consume_index_jobs())
    yield
    if state.refresh_task:
        state.refresh_task.cancel()
    if state.worker_task:
        state.worker_task.cancel()
    await state.queue.close()
    await state.redis.aclose()


app = FastAPI(title="Atlas AI", lifespan=lifespan)
# allow_origins=["*"] with credentials is the combination that turns a
# browser into a confused deputy. In the image the frontend is same-origin
# behind nginx, so this list only needs the Vite dev server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
)


@app.get("/health")
async def health():
    return {
        "ok": True,
        "llm": llm_configured(),
        "kafka": bool(settings.kafka_brokers),
        "auth": auth_enabled(),
    }


@app.get("/api/v1/documents")
async def list_docs(s: StateDep, principal: PrincipalDep):
    return await s.catalog.list()


@app.post("/api/v1/documents")
async def upload_doc(s: StateDep, principal: PrincipalDep, file: UploadFile = File(...)):
    doc_id = uuid.uuid4().hex[:12]
    try:
        dest = destination(settings.upload_dir, doc_id, file.filename or "")
        size = await save(file, dest, settings.max_upload_bytes)
    except UploadRejected as exc:
        UPLOAD_REJECTIONS.labels(reason=str(exc.status)).inc()
        raise HTTPException(status_code=exc.status, detail=exc.detail) from exc
    # Stored under the sanitised name, but reported and cited under the one the
    # user recognises. The two are only ever equal by luck.
    filename = file.filename or dest.name
    rec = DocumentRecord(id=doc_id, filename=filename, bytes=size, status="queued")
    await s.catalog.upsert(rec)
    await s.queue.publish(
        {
            "doc_id": doc_id,
            "filename": filename,
            "path": str(dest),
            "text": await asyncio.to_thread(parse_file, dest),
        }
    )
    return rec


@app.post("/api/v1/documents/seed")
async def seed(s: StateDep, principal: PrincipalDep):
    root = Path(settings.samples_dir) / "corpus"
    if not root.exists():
        root = Path(__file__).resolve().parents[2] / "samples" / "corpus"
    if not root.exists():
        raise HTTPException(404, "samples/corpus not found")
    created: list[DocumentRecord] = []
    for path in sorted(root.glob("*")):
        if path.suffix.lower() not in {".md", ".txt", ".pdf"}:
            continue
        doc_id = f"seed-{path.stem}"[:24]
        existing = await s.catalog.get(doc_id)
        if existing and existing.status == "ready":
            created.append(existing)
            continue
        rec = DocumentRecord(
            id=doc_id, filename=path.name, bytes=path.stat().st_size, status="queued"
        )
        await s.catalog.upsert(rec)
        await s.queue.publish(
            {
                "doc_id": doc_id,
                "filename": path.name,
                "path": str(path),
                "text": await asyncio.to_thread(parse_file, path),
            }
        )
        created.append(rec)
    return created


@app.delete("/api/v1/documents/{doc_id}")
async def delete_doc(doc_id: str, s: StateDep, principal: PrincipalDep):
    await asyncio.to_thread(s.vectors.delete_doc, doc_id)
    await s.catalog.delete(doc_id)
    # Bump the revision and let the background refresher rebuild in a thread.
    # Rebuilding here instead put an O(corpus) scroll and index build directly
    # on the event loop -- exactly the stall the refresher was written to end,
    # left behind on the one path that was never load-tested.
    await s.cache.r.incr("bm25:rev")
    return {"ok": True}


@app.post("/api/v1/query", response_model=QueryResponse)
async def query(req: QueryRequest, s: StateDep, principal: PrincipalDep):
    payload = await s.pipeline.ainvoke(req.question, use_cache=req.use_cache, principal=principal)
    obs.record_query(
        retrieval_ms=payload["retrieval_ms"],
        prompt_tokens=payload["prompt_tokens"],
        cache_hit=payload["cache_hit"],
    )
    observe_query(
        retrieval_ms=payload["retrieval_ms"],
        prompt_tokens=payload["prompt_tokens"],
        cache_hit=payload["cache_hit"],
        abstained=bool(payload.get("abstained")),
        blocked=bool(payload.get("blocked")),
        faithfulness=payload.get("faithfulness"),
    )
    return payload


@app.post("/api/v1/query/stream")
async def query_stream(req: QueryRequest, s: StateDep, principal: PrincipalDep):
    async def events():
        async for evt in s.pipeline.astream(
            req.question, use_cache=req.use_cache, principal=principal
        ):
            if evt["type"] == "done":
                data = evt["data"]
                obs.record_query(
                    retrieval_ms=float(data.get("retrieval_ms") or 0),
                    prompt_tokens=int(data.get("prompt_tokens") or 0),
                    cache_hit=bool(data.get("cache_hit")),
                )
                observe_query(
                    retrieval_ms=float(data.get("retrieval_ms") or 0),
                    prompt_tokens=int(data.get("prompt_tokens") or 0),
                    cache_hit=bool(data.get("cache_hit")),
                    abstained=bool(data.get("abstained")),
                    blocked=bool(data.get("blocked")),
                    faithfulness=data.get("faithfulness"),
                )
            yield f"event: {evt['type']}\ndata: {json.dumps(evt['data'], ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/api/v1/eval")
async def eval_run(s: StateDep, principal: PrincipalDep, limit: int | None = None):
    try:
        return await run_eval(s.pipeline, limit=limit)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/metrics")
async def prometheus_metrics():
    """Scrape endpoint, deliberately unauthenticated.

    It is not proxied by the frontend nginx, so it is reachable only from
    inside the cluster, where the scraper lives and where an API key would be
    one more secret to distribute for no gain. It carries no question text and
    no answers -- see app/metrics.py for what is and is not labelled.
    """
    payload, content_type = render_metrics()
    return Response(content=payload, media_type=content_type)


@app.get("/api/v1/metrics", response_model=MetricsSnapshot)
async def metrics(s: StateDep, principal: PrincipalDep):
    docs, chunks = await s.catalog.counts()
    return obs.snapshot(docs, chunks)
