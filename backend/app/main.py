from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import structlog
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, StreamingResponse

from app.chunking import parse_file
from app.config import settings
from app.evaluate import run_eval
from app.graph import Pipeline
from app.hybrid import BM25Index
from app.indexer import Indexer
from app.llm import llm_configured
from app.models import DocumentRecord, MetricsSnapshot, QueryRequest, QueryResponse
from app.obs import Cache, obs
from app.redis_client import create_redis
from app.store_docs import Catalog, IndexQueue
from app.vectors import VectorStore

log = structlog.get_logger()


class AppState:
    redis: redis.Redis
    cache: Cache
    vectors: VectorStore
    catalog: Catalog
    bm25: BM25Index
    queue: IndexQueue
    pipeline: Pipeline
    indexer: Indexer
    worker_task: asyncio.Task | None = None


state = AppState()


def get_state() -> AppState:
    return state


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
            log.info("index.ok", doc_id=job.get("doc_id"), chunks=n)
        except Exception:
            log.exception("index.failed", doc_id=job.get("doc_id"))
            rec = await state.catalog.get(job.get("doc_id", ""))
            if rec:
                rec.status = "failed"
                rec.error = "index_failed"
                await state.catalog.upsert(rec)


StateDep = Annotated[AppState, Depends(get_state)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    state.redis = create_redis()
    state.cache = Cache(state.redis)
    state.vectors = VectorStore()
    state.vectors.ensure()
    state.catalog = Catalog(state.redis)
    state.bm25 = BM25Index()
    try:
        state.bm25.rebuild(state.vectors.scroll_all())
    except Exception:
        log.warning("bm25.rebuild_skipped")
    state.queue = IndexQueue(state.redis)
    await state.queue.start()
    state.indexer = Indexer(state.cache, state.vectors, state.catalog, state.bm25)
    state.pipeline = Pipeline(state.cache, state.vectors, state.bm25)
    if settings.embedded_worker:
        state.worker_task = asyncio.create_task(_consume_index_jobs())
    yield
    if state.worker_task:
        state.worker_task.cancel()
    await state.queue.close()
    await state.redis.aclose()


app = FastAPI(title="Atlas AI", default_response_class=ORJSONResponse, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "ok": True,
        "llm": llm_configured(),
        "kafka": bool(settings.kafka_brokers),
    }


@app.get("/api/v1/documents")
async def list_docs(s: StateDep):
    return await s.catalog.list()


@app.post("/api/v1/documents")
async def upload_doc(s: StateDep, file: UploadFile = File(...)):
    raw = await file.read()
    doc_id = uuid.uuid4().hex[:12]
    filename = file.filename or "upload.bin"
    dest = Path(settings.upload_dir) / f"{doc_id}_{filename}"
    dest.write_bytes(raw)
    rec = DocumentRecord(id=doc_id, filename=filename, bytes=len(raw), status="queued")
    await s.catalog.upsert(rec)
    await s.queue.publish(
        {
            "doc_id": doc_id,
            "filename": filename,
            "path": str(dest),
            "text": parse_file(dest),
        }
    )
    return rec


@app.post("/api/v1/documents/seed")
async def seed(s: StateDep):
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
        rec = DocumentRecord(id=doc_id, filename=path.name, bytes=path.stat().st_size, status="queued")
        await s.catalog.upsert(rec)
        await s.queue.publish(
            {
                "doc_id": doc_id,
                "filename": path.name,
                "path": str(path),
                "text": parse_file(path),
            }
        )
        created.append(rec)
    return created


@app.delete("/api/v1/documents/{doc_id}")
async def delete_doc(doc_id: str, s: StateDep):
    s.vectors.delete_doc(doc_id)
    await s.catalog.delete(doc_id)
    s.bm25.rebuild(s.vectors.scroll_all())
    await s.cache.r.incr("bm25:rev")
    return {"ok": True}


@app.post("/api/v1/query", response_model=QueryResponse)
async def query(req: QueryRequest, s: StateDep):
    payload = await s.pipeline.ainvoke(req.question, use_cache=req.use_cache)
    obs.record_query(
        retrieval_ms=payload["retrieval_ms"],
        prompt_tokens=payload["prompt_tokens"],
        cache_hit=payload["cache_hit"],
    )
    return payload


@app.post("/api/v1/query/stream")
async def query_stream(req: QueryRequest, s: StateDep):
    async def events():
        async for evt in s.pipeline.astream(req.question, use_cache=req.use_cache):
            if evt["type"] == "done":
                data = evt["data"]
                obs.record_query(
                    retrieval_ms=float(data.get("retrieval_ms") or 0),
                    prompt_tokens=int(data.get("prompt_tokens") or 0),
                    cache_hit=bool(data.get("cache_hit")),
                )
            yield f"event: {evt['type']}\ndata: {json.dumps(evt['data'], ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/api/v1/eval")
async def eval_run(s: StateDep, limit: int | None = None):
    return await run_eval(s.pipeline, limit=limit)


@app.get("/api/v1/metrics", response_model=MetricsSnapshot)
async def metrics(s: StateDep):
    docs, chunks = await s.catalog.counts()
    return obs.snapshot(docs, chunks)
