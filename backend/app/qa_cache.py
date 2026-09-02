"""Paraphrase cache for answered questions, in its own Qdrant collection.

The previous implementation kept a 200-entry Redis list per principal and, on
every miss, fetched each entry with two round trips and scored it in Python:
up to 400 round trips and 200 cosine computations over 1536 dimensions, on the
query path. This is one ANN query, and it drops the window -- the index decides
what is near, not a recency list.

It also fixes what made the feature dead rather than merely slow. The stored
vector used to be the embedding of the rewritten query plus its HyDE
paragraph, while lookups embed the raw question. Those are different texts:
the same question asked twice scored 0.817 against its own cache entry, under
a 0.92 threshold, so the paraphrase path had never served a hit. Both sides now
embed the raw question.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any

import structlog
from qdrant_client import QdrantClient, models

from app.config import settings

log = structlog.get_logger()


def _point_id(principal: str, question: str) -> str:
    """Deterministic, so asking the same question again overwrites its entry
    instead of accumulating near-identical points."""
    digest = hashlib.md5(f"{principal}\x00{question.strip().lower()}".encode()).hexdigest()
    return str(uuid.UUID(digest))


class QACache:
    def __init__(self, client: QdrantClient | None = None, collection: str | None = None) -> None:
        self.client = client or QdrantClient(url=settings.qdrant_url, timeout=30)
        self.collection = collection or settings.qa_cache_collection

    def ensure(self) -> None:
        names = {c.name for c in self.client.get_collections().collections}
        if self.collection in names:
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=settings.embedding_dim,
                distance=models.Distance.COSINE,
            ),
        )
        # Both fields are filtered on every lookup, so both are indexed.
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="principal",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="expires_at",
            field_schema=models.PayloadSchemaType.INTEGER,
        )

    def nearest(self, principal: str, vector: list[float], threshold: float) -> dict[str, Any] | None:
        """Closest cached answer for this principal above `threshold`.

        The principal filter is a `must`, not a re-ranking step: an answer is
        built from whatever documents its asker could retrieve, so serving it
        to anyone else would undo the access control the API key established.
        """
        res = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=1,
            score_threshold=threshold,
            with_payload=True,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="principal", match=models.MatchValue(value=principal)
                    ),
                    models.FieldCondition(
                        key="expires_at", range=models.Range(gt=int(time.time()))
                    ),
                ]
            ),
        )
        if not res.points:
            return None
        raw = (res.points[0].payload or {}).get("answer_payload")
        return json.loads(raw) if raw else None

    def remember(
        self,
        principal: str,
        question: str,
        vector: list[float],
        payload: dict[str, Any],
        ttl_seconds: int = 60 * 60 * 12,
    ) -> None:
        self.client.upsert(
            collection_name=self.collection,
            points=[
                models.PointStruct(
                    id=_point_id(principal, question),
                    vector=vector,
                    payload={
                        "principal": principal,
                        "expires_at": int(time.time()) + ttl_seconds,
                        "answer_payload": json.dumps(payload, ensure_ascii=False),
                    },
                )
            ],
        )

    def purge_expired(self) -> None:
        """Qdrant has no TTL, so expiry is a payload field plus this sweep.

        Lookups already filter on it, so a missed sweep costs disk rather than
        correctness.
        """
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="expires_at", range=models.Range(lt=int(time.time()))
                        )
                    ]
                )
            ),
        )
