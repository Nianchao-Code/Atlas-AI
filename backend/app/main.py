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
from app.eval_jobs import EvalJob, EvalRunner
from app.evaluate import load_golden
from app.graph import Pipeline
from app.indexer import Indexer, consume
from app.llm import llm_configured
from app.metrics import (
    AUTH_REJECTIONS,
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
from app.reconcile import reconcile
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
    qa_cache: QACache | None
    queue: IndexQueue
    pipeline: Pipeline
    evals: EvalRunner
    indexer: Indexer
    worker_task: asyncio.Task | None = None
    refresh_task: asyncio.Task | None = None


state = AppState()


def get_state() -> AppState:
    return state


async def _background_refresh() -> None:
    """Housekeeping that should never happen during a request.

    The corpus gauges are computed here rather than during a scrape, so
    scraping stays O(1); the expired-entry sweep of the paraphrase cache rides
    along on a slower tick. This used to also rebuild an in-process sparse
    index, which is what the interval was originally sized for -- sparse
    retrieval now lives in Qdrant and is current the moment a write lands.
    """
    ticks = 0
    tick = max(settings.refresh_seconds, 0.1)
    sweep_every = max(1, int(60 / tick))
    reconcile_every = int(settings.reconcile_seconds / tick) if settings.reconcile_seconds else 0
    while True:
        await asyncio.sleep(settings.refresh_seconds)
        ticks += 1
        try:
            documents, chunks = await state.catalog.counts()
            set_corpus_size(documents, chunks)
            if state.qa_cache is not None and ticks % sweep_every == 0:
                await asyncio.to_thread(state.qa_cache.purge_expired)
            if reconcile_every and ticks % reconcile_every == 0:
                await reconcile(state.catalog, state.vectors, state.queue)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("background_refresh_failed")


async def _consume_index_jobs() -> None:
    """Index in-process, for the one-command demo.

    Docker compose and Kubernetes both run a dedicated worker instead
    (EMBEDDED_WORKER=false). Either way the API sees the result immediately:
    indexing writes to Qdrant and both retrievers read from there.

    The loop itself lives in app.indexer and is shared with the standalone
    worker, because two copies of it diverged once already.
    """
    await consume(state.indexer, state.queue, state.catalog, consumer="api-embedded")


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
    state.queue = IndexQueue(state.redis)
    await state.queue.start()
    state.indexer = Indexer(state.cache, state.vectors, state.catalog)
    state.pipeline = Pipeline(state.cache, state.vectors, qa_cache=state.qa_cache)
    state.evals = EvalRunner(state.redis, state.pipeline)
    try:
        # Before serving, not after: startup is when the two stores are most
        # likely to disagree, because that is when a collection gets rebuilt.
        await reconcile(state.catalog, state.vectors, state.queue)
    except Exception:
        # A reconciler that cannot run is not a reason to refuse to serve.
        log.exception("reconcile.failed_at_startup")
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
async def list_docs(
    s: StateDep,
    principal: PrincipalDep,
    limit: int = 200,
    offset: int = 0,
):
    """A page of the catalogue, plus the true total.

    It used to return everything. At eight documents that was a nicety; at ten
    thousand it is a 10MB response the browser renders as an unusable list, and
    the count the UI actually wants is one number.
    """
    documents, chunks = await s.catalog.counts()
    limit = max(1, min(limit, 1000))
    return {
        "documents": await s.catalog.list(limit=limit, offset=max(0, offset)),
        "total": documents,
        "chunks": chunks,
        "limit": limit,
        "offset": max(0, offset),
    }


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
    # One delete, both retrievers: the sparse vectors are these same points.
    await asyncio.to_thread(s.vectors.delete_doc, doc_id)
    await s.catalog.delete(doc_id)
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


def _job_response(job: EvalJob) -> dict:
    return {
        "job_id": job.id,
        "status": job.status,
        "done": job.done,
        "total": job.total,
        "elapsed_s": round(job.updated_at - job.started_at, 1),
        "joined": job.joined,
        "error": job.error,
        "report": job.report,
    }


@app.post("/api/v1/eval", status_code=202)
async def eval_start(s: StateDep, principal: PrincipalDep, limit: int | None = None):
    """Start a run and return immediately.

    The golden set takes minutes; holding the request open for it made a closed
    tab throw away the model calls it had already paid for, and made a second
    click bill for a second run. Poll GET /api/v1/eval/{job_id}.
    """
    try:
        # Fail here rather than inside the background task, where a missing
        # golden set would surface as a job that failed for no visible reason.
        cases = load_golden()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    total = len(cases[:limit] if limit else cases)
    return _job_response(await s.evals.start(limit=limit, total=total))


@app.get("/api/v1/eval")
async def eval_latest(s: StateDep, principal: PrincipalDep):
    job = await s.evals.latest()
    if job is None:
        raise HTTPException(status_code=404, detail="no eval has been run yet")
    return _job_response(job)


@app.get("/api/v1/eval/{job_id}")
async def eval_status(job_id: str, s: StateDep, principal: PrincipalDep):
    job = await s.evals.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown or expired job")
    return _job_response(job)


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


@app.post("/api/v1/documents/reconcile")
async def reconcile_now(s: StateDep, principal: PrincipalDep, dry_run: bool = False):
    """Repair disagreement between the catalogue and the vector store.

    Also runs at startup and on a slow timer; this exists so the state can be
    inspected and fixed without waiting for either, which is what the demo
    needs after a collection rebuild.
    """
    report = await reconcile(s.catalog, s.vectors, s.queue, dry_run=dry_run)
    return {
        "dry_run": dry_run,
        "checked": report.checked,
        "skipped_in_flight": report.skipped_in_flight,
        "requeued": report.requeued,
        "marked_failed": report.marked_failed,
        "orphans_deleted": report.orphans_deleted,
        "clean": report.clean,
    }


@app.get("/api/v1/metrics", response_model=MetricsSnapshot)
async def metrics(s: StateDep, principal: PrincipalDep):
    docs, chunks = await s.catalog.counts()
    return obs.snapshot(docs, chunks)
