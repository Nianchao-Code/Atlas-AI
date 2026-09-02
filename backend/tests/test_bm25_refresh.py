"""The BM25 snapshot is now rebuilt off the request path, in a thread.

That combination is only safe if two things hold, so both are pinned here: a
query reads one coherent snapshot even when a refresh lands mid-search, and
warm() rebuilds once per revision no matter how many callers race it.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.graph import Pipeline
from app.hybrid import BM25Index, _Snapshot
from app.vectors import Hit


def _hit(chunk_id: str) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        doc_id="d",
        filename=f"{chunk_id}.md",
        text=f"leave policy {chunk_id}",
        parent_text="parent",
        section="s",
        score=0.0,
        source="store",
    )


# ------------------------------------------------------------- atomicity ---

def test_rebuild_publishes_a_coherent_snapshot():
    idx = BM25Index()
    idx.rebuild([_hit("a"), _hit("b")])
    snap = idx._snap
    assert snap.ids == ["a", "b"]
    assert set(snap.hits) == {"a", "b"}
    assert snap.bm25 is not None


def test_search_is_immune_to_a_refresh_landing_mid_query():
    idx = BM25Index()
    idx.rebuild([_hit("old-a"), _hit("old-b")])
    original = idx._snap

    class _ScorerThatTriggersARefresh:
        def get_scores(self, _tokens):
            # A background refresh completes while this query is scoring.
            idx.rebuild([_hit("new-x")])
            return [2.0, 1.0]

    idx._snap = _Snapshot(
        ids=original.ids, hits=original.hits, bm25=_ScorerThatTriggersARefresh()
    )

    out = idx.search("leave policy", k=2)

    # Reading fields off self one at a time would pair these two scores with
    # the single id of the new snapshot.
    assert [h.chunk_id for h in out] == ["old-a", "old-b"]
    assert idx._snap.ids == ["new-x"]


def test_empty_corpus_searches_to_nothing():
    idx = BM25Index()
    assert idx.search("anything", k=5) == []
    idx.rebuild([])
    assert idx.search("anything", k=5) == []


# ------------------------------------------------------------------ warm ---

class _Redis:
    def __init__(self, rev: str | None) -> None:
        self.rev = rev

    async def get(self, _key: str) -> str | None:
        return self.rev


class _CountingVectors:
    def __init__(self) -> None:
        self.scrolls = 0

    def scroll_all(self):
        self.scrolls += 1
        return [_hit("a")]


class _CountingIndex:
    def __init__(self) -> None:
        self.rebuilds = 0

    def rebuild(self, _hits) -> None:
        self.rebuilds += 1

    def search(self, _q, _k):
        return []


def _pipeline(rev: str | None = "1"):
    p = Pipeline.__new__(Pipeline)
    p.cache = SimpleNamespace(r=_Redis(rev))
    p.vectors = _CountingVectors()
    p.bm25 = _CountingIndex()
    p._bm25_rev = None
    p._warm_lock = asyncio.Lock()
    return p


async def test_warm_rebuilds_once_then_reports_no_work():
    p = _pipeline("1")
    assert await p.warm() is True
    assert p.bm25.rebuilds == 1
    # Same revision: nothing to do, and no corpus scroll to pay for.
    assert await p.warm() is False
    assert p.bm25.rebuilds == 1
    assert p.vectors.scrolls == 1


async def test_warm_rebuilds_again_when_the_revision_moves():
    p = _pipeline("1")
    await p.warm()
    p.cache.r.rev = "2"
    assert await p.warm() is True
    assert p.bm25.rebuilds == 2


async def test_concurrent_warms_rebuild_once():
    p = _pipeline("1")
    results = await asyncio.gather(*(p.warm() for _ in range(5)))
    assert sum(results) == 1
    assert p.bm25.rebuilds == 1


async def test_retrieve_does_not_rebuild(monkeypatch):
    """The whole point: a query must never trigger a corpus scroll."""
    from app.graph import PipelineConfig
    from app.obs import Tracer

    async def fake_embed(texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr("app.graph.embed_texts", fake_embed)

    p = _pipeline("999")
    p.config = PipelineConfig()
    p.vectors.search = lambda _v, _k: []

    await p.retrieve({"question": "how many leave days", "tracer": Tracer()})

    assert p.vectors.scrolls == 0
    assert p.bm25.rebuilds == 0
