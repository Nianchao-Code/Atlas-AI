from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from app.chunking import Chunk
from app.config import settings


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    filename: str
    text: str
    parent_text: str
    section: str
    score: float
    source: str  # dense | bm25 | rrf


class VectorStore:
    def __init__(self, url: str | None = None) -> None:
        self.client = QdrantClient(url=url or settings.qdrant_url, timeout=30)
        self.collection = settings.qdrant_collection

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
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="doc_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        points = [
            models.PointStruct(
                id=_point_id(c.chunk_id),
                vector=v,
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
        hits: list[Hit] = []
        for p in res.points:
            payload = p.payload or {}
            hits.append(
                Hit(
                    chunk_id=payload.get("chunk_id", str(p.id)),
                    doc_id=payload.get("doc_id", ""),
                    filename=payload.get("filename", ""),
                    text=payload.get("text", ""),
                    parent_text=payload.get("parent_text", payload.get("text", "")),
                    section=payload.get("section", ""),
                    score=float(p.score or 0.0),
                    source="dense",
                )
            )
        return hits

    def scroll_all(self) -> list[Hit]:
        hits: list[Hit] = []
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in records:
                payload = p.payload or {}
                hits.append(
                    Hit(
                        chunk_id=payload.get("chunk_id", str(p.id)),
                        doc_id=payload.get("doc_id", ""),
                        filename=payload.get("filename", ""),
                        text=payload.get("text", ""),
                        parent_text=payload.get("parent_text", payload.get("text", "")),
                        section=payload.get("section", ""),
                        score=0.0,
                        source="store",
                    )
                )
            if offset is None:
                break
        return hits

    def delete_doc(self, doc_id: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
                )
            ),
        )

    def count(self) -> int:
        return int(self.client.count(self.collection, exact=True).count)


def _point_id(chunk_id: str) -> str:
    import hashlib
    import uuid

    return str(uuid.UUID(hashlib.md5(chunk_id.encode()).hexdigest()))
