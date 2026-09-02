from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog
from qdrant_client import QdrantClient, models

from app.chunking import Chunk
from app.config import settings
from app.sparse import SPARSE_VECTOR_NAME, sparse_vector

log = structlog.get_logger()


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    filename: str
    text: str
    parent_text: str
    section: str
    score: float
    source: str  # dense | sparse | rrf


class VectorStore:
    """Dense and sparse vectors in one collection, fused by Qdrant.

    Both retrievers read the same points, so a document is visible to both the
    moment it is written and gone from both the moment it is deleted. The
    previous split -- dense in Qdrant, sparse in the memory of every process
    that served a query -- is what made a revision counter, a poller and an
    atomically swapped snapshot necessary.
    """

    def __init__(self, url: str | None = None) -> None:
        self.client = QdrantClient(url=url or settings.qdrant_url, timeout=30)
        self.collection = settings.qdrant_collection

    def ensure(self) -> None:
        names = {c.name for c in self.client.get_collections().collections}
        if self.collection in names:
            if self._has_sparse():
                return
            # Qdrant cannot add a sparse vector to a live collection -- verified
            # against 1.12.5, where update_collection answers "Not existing
            # vector name" -- so a collection written before this change has to
            # be rebuilt. Loud, because it drops vectors while the Redis
            # catalogue still lists the documents: they need reindexing.
            log.warning(
                "collection.rebuilt_for_sparse",
                collection=self.collection,
                detail="existing vectors dropped; documents must be reindexed",
            )
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=settings.embedding_dim,
                distance=models.Distance.COSINE,
            ),
            # IDF is computed by Qdrant across the collection. That is the
            # corpus statistic which would otherwise have to live in a process
            # and be kept in step with every other process.
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="doc_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    def _has_sparse(self) -> bool:
        params = self.client.get_collection(self.collection).config.params
        return SPARSE_VECTOR_NAME in (params.sparse_vectors or {})

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        points = [
            models.PointStruct(
                id=_point_id(c.chunk_id),
                # The dense vector stays unnamed, so every dense query written
                # before sparse existed still means exactly what it did.
                vector={"": v, SPARSE_VECTOR_NAME: sparse_vector(c.text)},
                payload={
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "filename": c.filename,
                    "text": c.text,
                    "parent_text": c.parent_text,
                    "section": c.section,
                },
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, vector: list[float], k: int) -> list[Hit]:
        res = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=k,
            with_payload=True,
        )
        return [_to_hit(p, "dense") for p in res.points]

    def search_sparse(self, query: str, k: int) -> list[Hit]:
        vec = sparse_vector(query)
        if not vec.indices:
            return []
        res = self.client.query_points(
            collection_name=self.collection,
            query=vec,
            using=SPARSE_VECTOR_NAME,
            limit=k,
            with_payload=True,
        )
        return [_to_hit(p, "sparse") for p in res.points]

    def search_hybrid(self, vector: list[float], query: str, k: int) -> list[Hit]:
        """Dense and sparse retrieved and fused in one round trip.

        RRF is the same fusion the client used to do. Running it where the
        candidates already live means one network call instead of two, and no
        second ranking to keep consistent with the first.
        """
        prefetch = [models.Prefetch(query=vector, limit=k)]
        vec = sparse_vector(query)
        if vec.indices:
            prefetch.append(models.Prefetch(query=vec, using=SPARSE_VECTOR_NAME, limit=k))
        res = self.client.query_points(
            collection_name=self.collection,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k,
            with_payload=True,
        )
        return [_to_hit(p, "rrf") for p in res.points]

    def delete_doc(self, doc_id: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))
                    ]
                )
            ),
        )

    def doc_ids(self) -> set[str]:
        """Every doc_id holding at least one vector, for reconciling with Redis.

        Scrolls only the doc_id field: the payloads carry whole passages, and
        this runs over the entire collection.
        """
        seen: set[str] = set()
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection,
                limit=1024,
                offset=offset,
                with_payload=["doc_id"],
                with_vectors=False,
            )
            for p in records:
                doc_id = (p.payload or {}).get("doc_id")
                if doc_id:
                    seen.add(str(doc_id))
            if offset is None:
                return seen

    def count(self) -> int:
        return int(self.client.count(self.collection, exact=True).count)


def _to_hit(point: Any, source: str) -> Hit:
    payload = point.payload or {}
    return Hit(
        chunk_id=payload.get("chunk_id", str(point.id)),
        doc_id=payload.get("doc_id", ""),
        filename=payload.get("filename", ""),
        text=payload.get("text", ""),
        # A payload that stores an explicit null returns None from .get with a
        # default, and None reaches a field declared str.
        parent_text=payload.get("parent_text") or payload.get("text") or "",
        section=payload.get("section", ""),
        score=float(getattr(point, "score", 0.0) or 0.0),
        source=source,
    )


def _point_id(chunk_id: str) -> str:
    import hashlib
    import uuid

    return str(uuid.UUID(hashlib.md5(chunk_id.encode()).hexdigest()))
