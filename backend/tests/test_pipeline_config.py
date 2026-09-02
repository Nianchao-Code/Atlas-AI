"""The retrieve node is what the ablation harness varies, so pin its wiring.

These cover the branches the ablation depends on: that switching a retriever
off really removes it, that a lone retriever keeps its own ranking instead of
being rewritten by RRF, and that rerank/grade honour the per-pipeline config
rather than only the global setting.
"""
from __future__ import annotations

import pytest

from app.config import settings
from app.graph import Pipeline, PipelineConfig
from app.obs import Tracer
from app.rerank import cross_encoder_rerank
from app.vectors import Hit


def _hit(chunk_id: str, score: float, source: str) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        doc_id="d",
        filename=f"{chunk_id}.md",
        text=chunk_id,
        parent_text=chunk_id,
        section="s",
        score=score,
        source=source,
    )


class _FakeRedis:
    async def get(self, _key: str) -> str:
        return "1"


class _FakeCache:
    def __init__(self) -> None:
        self.r = _FakeRedis()


class _FakeVectors:
    def __init__(self) -> None:
        self.calls = 0

    def search(self, _vec, _k):
        self.calls += 1
        return [_hit("dense-a", 0.9, "dense"), _hit("dense-b", 0.8, "dense")]

    def scroll_all(self):
        return []


class _FakeBM25:
    def __init__(self) -> None:
        self.calls = 0

    def search(self, _q, _k):
        self.calls += 1
        return [_hit("bm25-a", 12.0, "bm25"), _hit("dense-b", 3.0, "bm25")]

    def rebuild(self, _hits) -> None:
        pass


def _pipeline(config: PipelineConfig) -> tuple[Pipeline, _FakeVectors, _FakeBM25]:
    p = Pipeline.__new__(Pipeline)
    vectors, bm25 = _FakeVectors(), _FakeBM25()
    p.cache = _FakeCache()
    p.vectors = vectors
    p.bm25 = bm25
    p.config = config
    p._bm25_rev = "1"  # already warm; keeps the rebuild out of the assertions
    return p, vectors, bm25


@pytest.fixture(autouse=True)
def _stub_embeddings(monkeypatch):
    async def fake_embed(texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr("app.graph.embed_texts", fake_embed)


def _state() -> dict:
    return {"question": "how many leave days", "tracer": Tracer()}


async def test_dense_only_skips_bm25():
    p, vectors, bm25 = _pipeline(PipelineConfig(sparse=False))
    out = await p.retrieve(_state())
    assert vectors.calls == 1
    assert bm25.calls == 0
    assert [h.chunk_id for h in out["hits"]] == ["dense-a", "dense-b"]
    # A single retriever keeps its own ranking and its own source label.
    assert {h.source for h in out["hits"]} == {"dense"}


async def test_bm25_only_skips_dense():
    p, vectors, bm25 = _pipeline(PipelineConfig(dense=False))
    out = await p.retrieve(_state())
    assert vectors.calls == 0
    assert bm25.calls == 1
    assert {h.source for h in out["hits"]} == {"bm25"}


async def test_hybrid_fuses_both_and_relabels():
    p, vectors, bm25 = _pipeline(PipelineConfig())
    out = await p.retrieve(_state())
    assert vectors.calls == 1 and bm25.calls == 1
    ids = [h.chunk_id for h in out["hits"]]
    assert set(ids) == {"dense-a", "dense-b", "bm25-a"}
    assert {h.source for h in out["hits"]} == {"rrf"}
    # dense-b is the only chunk both retrievers return, so RRF ranks it first.
    assert ids[0] == "dense-b"


async def test_grade_disabled_short_circuits_without_llm():
    p, _v, _b = _pipeline(PipelineConfig(grade=False))
    hits = [_hit(f"c{i}", 1.0, "rrf") for i in range(10)]
    out = await p.grade({**_state(), "hits": hits})
    assert out["grade"] == "sufficient"
    assert len(out["hits"]) == settings.rerank_k


async def test_rewrite_disabled_passes_question_through():
    p, _v, _b = _pipeline(PipelineConfig(rewrite=False))
    out = await p.rewrite(_state())
    assert out["rewritten"] == "how many leave days"


def test_rerank_enabled_argument_overrides_global(monkeypatch):
    monkeypatch.setattr(settings, "enable_cross_encoder", True)
    hits = [_hit("a", 1.0, "rrf"), _hit("b", 0.5, "rrf"), _hit("c", 0.1, "rrf")]
    out = cross_encoder_rerank("q", hits, top_k=2, enabled=False)
    assert [h.chunk_id for h in out] == ["a", "b"]
