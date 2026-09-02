from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.vectors import Hit

_TOKEN = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]


@dataclass(frozen=True)
class _Snapshot:
    """One coherent view of the corpus: ids, payloads, and the scorer built
    from exactly those ids, in that order."""

    ids: list[str]
    hits: dict[str, Hit]
    bm25: BM25Okapi | None


class BM25Index:
    """Read-mostly index published by a single assignment.

    rebuild() now runs in a worker thread while queries are being served, so
    the new state is assembled off to the side and swapped in one rebinding.
    Assigning ids, payloads and scorer one field at a time let a concurrent
    search pair fresh ids with a stale scorer: a KeyError if it was lucky,
    silently mismatched hits if it was not.
    """

    def __init__(self) -> None:
        self._snap = _Snapshot(ids=[], hits={}, bm25=None)

    def rebuild(self, hits: list[Hit]) -> None:
        corpus = [tokenize(h.text) for h in hits]
        self._snap = _Snapshot(
            ids=[h.chunk_id for h in hits],
            hits={h.chunk_id: h for h in hits},
            bm25=BM25Okapi(corpus) if corpus else None,
        )

    def search(self, query: str, k: int) -> list[Hit]:
        snap = self._snap  # one read; a swap mid-search cannot split this view
        if not snap.bm25 or not snap.ids:
            return []
        scores = snap.bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        out: list[Hit] = []
        for i, score in ranked:
            if score <= 0:
                continue
            base = snap.hits[snap.ids[i]]
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
