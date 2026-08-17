from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from typing import Any, AsyncIterator

import redis.asyncio as redis

from app.config import settings
from app.models import DocumentRecord


def worker_consumer_id(default: str = "worker-1") -> str:
    return os.environ.get("WORKER_ID") or socket.gethostname() or default


@dataclass
class IndexJobEnvelope:
    job: dict[str, Any]
    msg_id: str | None = None  # Redis stream id; None for Kafka


class Catalog:
    def __init__(self, client: redis.Redis) -> None:
        self.r = client

    def _key(self, doc_id: str) -> str:
        return f"doc:{doc_id}"

    async def upsert(self, rec: DocumentRecord) -> None:
        await self.r.hset("docs", rec.id, rec.model_dump_json())

    async def get(self, doc_id: str) -> DocumentRecord | None:
        raw = await self.r.hget("docs", doc_id)
        return DocumentRecord.model_validate_json(raw) if raw else None

    async def list(self) -> list[DocumentRecord]:
        raw = await self.r.hgetall("docs")
        docs = [DocumentRecord.model_validate_json(v) for v in raw.values()]
        return sorted(docs, key=lambda d: d.filename)

    async def delete(self, doc_id: str) -> None:
        await self.r.hdel("docs", doc_id)

    async def counts(self) -> tuple[int, int]:
        docs = await self.list()
        return len(docs), sum(d.chunks for d in docs)


class IndexQueue:
    """One job schema, two transports.

    Local / compose default: Redis Streams. Production: Kafka (or Redpanda).
    The worker does not care which bus delivered the bytes.
    """

    def __init__(self, client: redis.Redis) -> None:
        self.r = client
        self._kafka = None

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
            resp = await self.r.xreadgroup(
                "indexers",
                consumer,
                {settings.index_stream: ">"},
                count=1,
                block=5000,
            )
            if not resp:
                continue
            for _stream, messages in resp:
                for msg_id, fields in messages:
                    raw = fields.get("job") or fields.get(b"job")
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8")
                    yield IndexJobEnvelope(job=json.loads(raw), msg_id=msg_id)

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
