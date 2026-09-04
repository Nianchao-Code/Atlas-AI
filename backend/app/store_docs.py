from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import redis.asyncio as redis

from app.config import settings
from app.models import DocumentRecord


def worker_consumer_id(default: str = "atlas-worker") -> str:
    return os.environ.get("WORKER_ID") or default


@dataclass
class IndexJobEnvelope:
    job: dict[str, Any]
    msg_id: str | None = None  # Redis stream id; None for Kafka


SEP = chr(31)  # unit separator: no filename this service accepts can hold it
DOCS_KEY = "docs"
CHUNKS_KEY = "docs:chunks"
ORDER_KEY = "docs:by_name"

# Three keys kept in step by Lua, so a writer cannot leave two of them
# disagreeing. HSET followed by INCRBY from the client is two round trips with
# a window between them, and that window is where a counter goes wrong.
#
# The separator is interpolated from SEP rather than written twice. Two places
# that must agree, with nothing forcing them to, is the shape of most of the
# bugs this project has found.
_UPSERT = f"""
local old = redis.call('HGET', KEYS[1], ARGV[1])
local delta = tonumber(ARGV[3])
if old then
  local prev = cjson.decode(old)
  delta = delta - tonumber(prev['chunks'])
  redis.call('ZREM', KEYS[3], prev['filename'] .. string.char({ord(SEP)}) .. ARGV[1])
end
redis.call('HSET', KEYS[1], ARGV[1], ARGV[2])
redis.call('INCRBY', KEYS[2], delta)
redis.call('ZADD', KEYS[3], 0, ARGV[4] .. string.char({ord(SEP)}) .. ARGV[1])
return delta
"""

_DELETE = f"""
local old = redis.call('HGET', KEYS[1], ARGV[1])
if not old then return 0 end
local prev = cjson.decode(old)
redis.call('INCRBY', KEYS[2], -tonumber(prev['chunks']))
redis.call('ZREM', KEYS[3], prev['filename'] .. string.char({ord(SEP)}) .. ARGV[1])
redis.call('HDEL', KEYS[1], ARGV[1])
return 1
"""


class Catalog:
    """Document records, plus the two indexes that keep reads O(1).

    `counts()` used to load every record and sum it, and the background
    refresher calls it every two seconds on every replica. At eight documents
    that was 0.5ms. At ten thousand it measured 56ms -- 2.8% of a core, per
    replica, forever, to recompute a number that only changes on a write.

    So the chunk total is maintained on write, and ordering lives in a sorted
    set so a page costs O(log n + page) instead of loading the corpus. Both are
    derived data and both can drift; `rebuild_indexes()` recomputes them from
    the records, which are the only thing here that is authoritative.
    """

    def __init__(self, client: redis.Redis) -> None:
        self.r = client
        self._upsert = client.register_script(_UPSERT)
        self._delete = client.register_script(_DELETE)

    async def upsert(self, rec: DocumentRecord) -> None:
        await self._upsert(
            keys=[DOCS_KEY, CHUNKS_KEY, ORDER_KEY],
            args=[rec.id, rec.model_dump_json(), rec.chunks, rec.filename],
        )

    async def get(self, doc_id: str) -> DocumentRecord | None:
        raw = await self.r.hget(DOCS_KEY, doc_id)
        return DocumentRecord.model_validate_json(raw) if raw else None

    async def list(self, limit: int | None = None, offset: int = 0) -> list[DocumentRecord]:
        """A page of records, filename-ordered.

        Unbounded by default because reconciliation and the eval harness both
        want the whole catalogue; the HTTP endpoint passes a limit, because
        returning ten thousand records to a browser is a 10MB response nobody
        reads.
        """
        if limit is None and offset == 0:
            raw = await self.r.hgetall(DOCS_KEY)
            docs = [DocumentRecord.model_validate_json(v) for v in raw.values()]
            return sorted(docs, key=lambda d: d.filename)
        stop = offset + limit - 1 if limit is not None else -1
        members = await self.r.zrange(ORDER_KEY, offset, stop)
        ids = [_text(m).rsplit(SEP, 1)[-1] for m in members]
        if not ids:
            return []
        rows = await self.r.hmget(DOCS_KEY, ids)
        return [DocumentRecord.model_validate_json(v) for v in rows if v]

    async def delete(self, doc_id: str) -> None:
        await self._delete(keys=[DOCS_KEY, CHUNKS_KEY, ORDER_KEY], args=[doc_id])

    async def counts(self) -> tuple[int, int]:
        documents = int(await self.r.hlen(DOCS_KEY))
        chunks = await self.r.get(CHUNKS_KEY)
        if chunks is None:
            # A corpus written before the counter existed, or one whose counter
            # was lost. Pay the scan once rather than every two seconds.
            return documents, await self.rebuild_indexes()
        return documents, int(chunks)

    async def rebuild_indexes(self) -> int:
        """Recompute the derived keys from the records. Returns the chunk total."""
        raw = await self.r.hgetall(DOCS_KEY)
        docs = [DocumentRecord.model_validate_json(v) for v in raw.values()]
        total = sum(d.chunks for d in docs)
        pipe = self.r.pipeline()
        pipe.set(CHUNKS_KEY, total)
        pipe.delete(ORDER_KEY)
        if docs:
            pipe.zadd(ORDER_KEY, {f"{d.filename}{SEP}{d.id}": 0 for d in docs})
        await pipe.execute()
        return total


def _text(value) -> str:
    return value if isinstance(value, str) else value.decode("utf-8")


class IndexQueue:
    """One job schema, two transports.

    Local / compose default: Redis Streams. Production: Kafka (or Redpanda).
    The worker does not care which bus delivered the bytes.
    """

    def __init__(self, client: redis.Redis) -> None:
        self.r = client
        # AIOKafkaProducer when a broker is configured; the import is optional.
        self._kafka: Any | None = None

    async def start(self) -> None:
        if settings.kafka_brokers:
            from aiokafka import AIOKafkaProducer

            self._kafka = AIOKafkaProducer(bootstrap_servers=settings.kafka_brokers)
            await self._kafka.start()
        try:
            await self.r.xgroup_create(settings.index_stream, "indexers", id="0", mkstream=True)
        except Exception as exc:  # redis.ResponseError BUSYGROUP
            if "BUSYGROUP" not in str(exc):
                raise

    async def close(self) -> None:
        if self._kafka is not None:
            await self._kafka.stop()

    async def publish(self, job: dict[str, Any]) -> None:
        payload = json.dumps(job, ensure_ascii=False)
        if self._kafka is not None:
            await self._kafka.send_and_wait(settings.kafka_topic, payload.encode("utf-8"))
            return
        await self.r.xadd(settings.index_stream, {"job": payload})

    async def jobs(self, consumer: str | None = None) -> AsyncIterator[IndexJobEnvelope]:
        consumer = consumer or worker_consumer_id()
        if settings.kafka_brokers:
            async for job in self._kafka_jobs():
                yield IndexJobEnvelope(job=job)
            return
        while True:
            try:
                envelope = await self._next_job(consumer)
            except (redis.TimeoutError, TimeoutError, redis.RedisError):
                await asyncio.sleep(1)
                continue
            if envelope is None:
                await asyncio.sleep(1)
                continue
            yield envelope

    async def _next_job(self, consumer: str) -> IndexJobEnvelope | None:
        """Claim stuck PEL entries first, then read never-delivered jobs.

        A crashed worker leaves messages in the consumer group's pending list.
        XREADGROUP with '>' cannot see those; XAUTOCLAIM steals them.
        """
        try:
            claimed = await self.r.xautoclaim(
                settings.index_stream,
                "indexers",
                consumer,
                min_idle_time=0,
                start_id="0-0",
                count=1,
            )
            messages = _autoclaim_messages(claimed)
            if messages:
                return _envelope(messages[0])
        except redis.ResponseError:
            pass

        # Annotated loosely on purpose: the declared return type does not
        # describe the [[stream, [(id, fields)]]] shape redis actually sends.
        resp: Any = await self.r.xreadgroup(
            "indexers",
            consumer,
            {settings.index_stream: ">"},
            count=1,
        )
        if not resp:
            return None
        for _stream, messages in resp:
            if messages:
                return _envelope(messages[0])
        return None

    async def ack(self, envelope: IndexJobEnvelope) -> None:
        if envelope.msg_id is None:
            return
        await self.r.xack(settings.index_stream, "indexers", envelope.msg_id)

    async def _kafka_jobs(self) -> AsyncIterator[dict[str, Any]]:
        from aiokafka import AIOKafkaConsumer

        consumer = AIOKafkaConsumer(
            settings.kafka_topic,
            bootstrap_servers=settings.kafka_brokers,
            group_id="atlas-indexers",
            auto_offset_reset="earliest",
        )
        await consumer.start()
        try:
            async for msg in consumer:
                yield json.loads(msg.value.decode("utf-8"))
        finally:
            await consumer.stop()


def _autoclaim_messages(claimed: Any) -> list:
    if not claimed:
        return []
    # redis-py: (next_id, messages) or (next_id, messages, deleted)
    if isinstance(claimed, (list, tuple)) and len(claimed) >= 2:
        messages = claimed[1]
        return list(messages or [])
    return []


def _envelope(item: Any) -> IndexJobEnvelope:
    msg_id, fields = item
    raw = fields.get("job") or fields.get(b"job")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return IndexJobEnvelope(job=json.loads(raw), msg_id=msg_id)
