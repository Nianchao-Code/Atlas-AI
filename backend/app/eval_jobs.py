"""Run the golden set as a job instead of holding an HTTP request open.

Evaluating 53 questions takes minutes, and it used to happen inside the request
that asked for it. That shape had four problems and only one of them was
visible: the nginx in front of the API needed `proxy_read_timeout 600s` to stop
cutting the connection, which is a symptom being configured around rather than
a cause being fixed.

  work is lost on disconnect   A closed browser tab threw away four minutes of
                               model calls that had already been paid for.
  concurrent runs duplicate    Two clicks started two full runs against the
                               same corpus and billed for both.
  no progress                  A spinner for four minutes is indistinguishable
                               from a hang, which is how the timeout got found.
  a proxy decides the limit    Any hop with an idle timeout shorter than the
                               run silently truncates it.

So: start returns immediately with a job id, the run continues in the
background, and progress and the report are read back separately. State lives
in Redis rather than in the process, for the same reason the sparse index does
-- a poll can land on a different replica than the one doing the work.

The one thing Redis cannot cover is the process disappearing mid-run. The job
heartbeats as it goes; a record that stops beating is reported as abandoned
rather than left claiming to be running forever.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import redis.asyncio as redis
import structlog

from app.evaluate import run_eval
from app.graph import Pipeline

log = structlog.get_logger()

ACTIVE_KEY = "eval:active"
JOB_PREFIX = "eval:job:"
LATEST_KEY = "eval:latest"

# How long a job may go without a heartbeat before it is presumed dead. Well
# above the slowest single case, which is a model call plus a judge call.
STALE_SECONDS = 120
# How long a finished report stays readable. Long enough to reload the tab.
RETAIN_SECONDS = 3600

Status = Literal["running", "done", "failed"]


@dataclass
class EvalJob:
    id: str
    status: Status
    started_at: float
    updated_at: float
    done: int = 0
    total: int = 0
    limit: int | None = None
    report: dict[str, Any] | None = None
    error: str | None = None
    # True when start() joined a run that was already going rather than
    # beginning one. The caller usually wants to say so.
    joined: bool = field(default=False, compare=False)

    @property
    def stale(self) -> bool:
        return self.status == "running" and (time.time() - self.updated_at) > STALE_SECONDS


class EvalRunner:
    def __init__(self, client: redis.Redis, pipeline: Pipeline) -> None:
        self.r = client
        self.pipeline = pipeline
        # asyncio only holds a weak reference to a bare task; without this the
        # garbage collector can cancel the run mid-flight.
        self._tasks: set[asyncio.Task] = set()

    async def start(self, limit: int | None = None, total: int = 0) -> EvalJob:
        """Begin a run, or return the one already going.

        Single-flight through SET NX: two clicks are one run. The key carries a
        TTL so a crashed process cannot block every future run.

        `total` is known before the first case finishes, and passing it in means
        the UI can say "0 of 53" rather than "0 of ?" for the length of the
        first question.
        """
        job_id = uuid.uuid4().hex[:12]
        claimed = await self.r.set(ACTIVE_KEY, job_id, nx=True, ex=STALE_SECONDS)
        if not claimed:
            existing = await self._active_job()
            if existing is not None:
                existing.joined = True
                return existing
            # The pointer was there but the record was not, or it had gone
            # stale. Take it over rather than refusing to run.
            await self.r.set(ACTIVE_KEY, job_id, ex=STALE_SECONDS)

        now = time.time()
        job = EvalJob(
            id=job_id,
            status="running",
            started_at=now,
            updated_at=now,
            limit=limit,
            total=total,
        )
        await self._save(job)
        task = asyncio.create_task(self._run(job))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        log.info("eval.started", job_id=job_id, limit=limit)
        return job

    async def get(self, job_id: str) -> EvalJob | None:
        raw = await self.r.get(JOB_PREFIX + job_id)
        if raw is None:
            return None
        return self._decode(raw)

    async def latest(self) -> EvalJob | None:
        job_id = await self.r.get(LATEST_KEY)
        if not job_id:
            return None
        return await self.get(job_id if isinstance(job_id, str) else job_id.decode())

    async def _active_job(self) -> EvalJob | None:
        job_id = await self.r.get(ACTIVE_KEY)
        if not job_id:
            return None
        job = await self.get(job_id if isinstance(job_id, str) else job_id.decode())
        if job is None or job.stale:
            return None
        return job

    async def _run(self, job: EvalJob) -> None:
        async def progress(done: int, total: int) -> None:
            job.done, job.total, job.updated_at = done, total, time.time()
            await self._save(job)
            # Refresh the single-flight claim alongside the heartbeat, so the
            # two cannot disagree about whether this run is still alive.
            await self.r.expire(ACTIVE_KEY, STALE_SECONDS)

        try:
            report = await run_eval(self.pipeline, limit=job.limit, on_case=progress)
            job.status = "done"
            job.report = report.model_dump()
            log.info("eval.finished", job_id=job.id, n=report.n)
        except asyncio.CancelledError:
            job.status = "failed"
            job.error = "cancelled"
            raise
        except Exception as exc:
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"[:300]
            log.exception("eval.failed", job_id=job.id)
        finally:
            job.updated_at = time.time()
            await self._save(job)
            await self.r.delete(ACTIVE_KEY)

    async def _save(self, job: EvalJob) -> None:
        payload = {k: v for k, v in asdict(job).items() if k != "joined"}
        await self.r.set(
            JOB_PREFIX + job.id, json.dumps(payload, ensure_ascii=False), ex=RETAIN_SECONDS
        )
        await self.r.set(LATEST_KEY, job.id, ex=RETAIN_SECONDS)

    @staticmethod
    def _decode(raw: str | bytes) -> EvalJob:
        data = json.loads(raw if isinstance(raw, str) else raw.decode())
        job = EvalJob(**data)
        if job.stale:
            # Reported, not rewritten: the process that owned it is gone, so
            # nobody is left who can honestly update the record.
            job.status = "failed"
            job.error = job.error or "abandoned: no heartbeat"
        return job
