from __future__ import annotations

import re
from collections import defaultdict

from rank_bm25 import BM25Okapi

from app.vectors import Hit

_TOKEN = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]


class BM25Index:
    def __init__(self) -> None:
        self._ids: list[str] = []
        self._hits: dict[str, Hit] = {}
        self._corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def rebuild(self, hits: list[Hit]) -> None:
        self._hits = {h.chunk_id: h for h in hits}
        self._ids = [h.chunk_id for h in hits]
        self._corpus = [tokenize(h.text) for h in hits]
        self._bm25 = BM25Okapi(self._corpus) if self._corpus else None

    def search(self, query: str, k: int) -> list[Hit]:
        if not self._bm25 or not self._ids:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        out: list[Hit] = []
        for i, score in ranked:
            if score <= 0:
                continue
            base = self._hits[self._ids[i]]
            out.append(
                Hit(
                    chunk_id=base.chunk_id,
                    doc_id=base.doc_id,
                    filename=base.filename,
                    text=base.text,
                    parent_text=base.parent_text,
                    section=base.section,
                    score=float(score),
                    source="bm25",
                )
            )
        return out


def reciprocal_rank_fusion(lists: list[list[Hit]], k: int = 60, limit: int = 24) -> list[Hit]:
    """RRF is the boring fusion that actually shows up in production RAG.

    We do not train a learning-to-rank model on 10k docs. We fuse dense
    and BM25 ranks, then cut to `limit` for the reranker.
    """
    scores: dict[str, float] = defaultdict(float)
    keep: dict[str, Hit] = {}
    for ranked in lists:
        for rank, hit in enumerate(ranked, start=1):
            scores[hit.chunk_id] += 1.0 / (k + rank)
            prev = keep.get(hit.chunk_id)
            if prev is None or hit.score > prev.score:
                keep[hit.chunk_id] = hit
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
    fused: list[Hit] = []
    for chunk_id, score in ordered:
        h = keep[chunk_id]
        fused.append(
            Hit(
                chunk_id=h.chunk_id,
                doc_id=h.doc_id,
                filename=h.filename,
                text=h.text,
                parent_text=h.parent_text,
                section=h.section,
                score=score,
                source="rrf",
            )
        )
    return fused
