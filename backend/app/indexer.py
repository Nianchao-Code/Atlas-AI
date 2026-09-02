from __future__ import annotations

import asyncio
from typing import Any

from app.chunking import chunk_document, parse_file
from app.llm import embed_texts
from app.obs import Cache
from app.store_docs import Catalog
from app.vectors import VectorStore


class Indexer:
    def __init__(self, cache: Cache, vectors: VectorStore, catalog: Catalog) -> None:
        self.cache = cache
        self.vectors = vectors
        self.catalog = catalog

    async def index_job(self, job: dict[str, Any]) -> int:
        from pathlib import Path

        doc_id = job["doc_id"]
        rec = await self.catalog.get(doc_id)
        if rec:
            rec.status = "indexing"
            await self.catalog.upsert(rec)

        path = Path(job.get("path") or "")
        text = str(job.get("text") or "")
        if not text:
            if not path.exists():
                raise FileNotFoundError(f"index job missing text and path does not exist: {path}")
            # PDF extraction is CPU-bound and unbounded by document size, so it
            # does not belong on the loop the embedded worker shares with the API.
            text = await asyncio.to_thread(parse_file, path)
        chunks = chunk_document(doc_id=doc_id, filename=job["filename"], text=text)
        vectors = await self._embed_cached([c.text for c in chunks])
        # Writes dense and sparse together, so both retrievers see the
        # document at the same moment and neither needs to be told about it.
        await asyncio.to_thread(self.vectors.upsert, chunks, vectors)

        if rec:
            rec.status = "ready"
            rec.chunks = len(chunks)
            await self.catalog.upsert(rec)
        return len(chunks)

    async def _embed_cached(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float] | None] = [None] * len(texts)
        missing: list[int] = []
        for i, t in enumerate(texts):
            cached = await self.cache.get_embedding(t)
            if cached is None:
                missing.append(i)
            else:
                out[i] = cached
        if missing:
            fresh = await embed_texts([texts[i] for i in missing])
            for i, vec in zip(missing, fresh, strict=True):
                out[i] = vec
                await self.cache.set_embedding(texts[i], vec)
        return [v for v in out if v is not None]
