"""The retrieve node is what the ablation harness varies, so pin its wiring.

These cover the branches the ablation depends on: that switching a retriever
off really removes it, that a lone retriever is queried on its own rather than
sent through a fusion with nothing to fuse against, and that rerank/grade
honour the per-pipeline config rather than only the global setting.
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


class _FakeVectors:
    """Records which of the three retrieval calls the node actually made."""

    def __init__(self) -> None:
        self.dense = 0
        self.sparse = 0
        self.hybrid = 0

    def search(self, _vec, _k):
        self.dense += 1
        return [_hit("dense-a", 0.9, "dense"), _hit("dense-b", 0.8, "dense")]

    def search_sparse(self, _q, _k):
        self.sparse += 1
        return [_hit("sparse-a", 12.0, "sparse")]

    def search_hybrid(self, _vec, _q, _k):
        self.hybrid += 1
        return [_hit("dense-b", 0.9, "rrf"), _hit("sparse-a", 0.5, "rrf")]


def _pipeline(config: PipelineConfig) -> tuple[Pipeline, _FakeVectors]:
    p = Pipeline.__new__(Pipeline)
    vectors = _FakeVectors()
    p.vectors = vectors
    p.config = config
    return p, vectors


@pytest.fixture(autouse=True)
def _stub_embeddings(monkeypatch):
    async def fake_embed(texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr("app.graph.embed_texts", fake_embed)


def _state() -> dict:
    return {"question": "how many leave days", "tracer": Tracer()}


async def test_dense_only_queries_dense_alone():
    p, vectors = _pipeline(PipelineConfig(sparse=False))
    out = await p.retrieve(_state())
    assert (vectors.dense, vectors.sparse, vectors.hybrid) == (1, 0, 0)
    assert [h.chunk_id for h in out["hits"]] == ["dense-a", "dense-b"]
    # A single retriever keeps its own ranking and its own source label.
    assert {h.source for h in out["hits"]} == {"dense"}


async def test_sparse_only_queries_sparse_alone():
    p, vectors = _pipeline(PipelineConfig(dense=False))
    out = await p.retrieve(_state())
    assert (vectors.dense, vectors.sparse, vectors.hybrid) == (0, 1, 0)
    assert {h.source for h in out["hits"]} == {"sparse"}


async def test_hybrid_makes_one_fused_call():
    p, vectors = _pipeline(PipelineConfig())
    out = await p.retrieve(_state())
    # The point of moving fusion into Qdrant: one round trip, not two plus a
    # client-side merge.
    assert (vectors.dense, vectors.sparse, vectors.hybrid) == (0, 0, 1)
    assert {h.source for h in out["hits"]} == {"rrf"}


async def test_no_retriever_returns_nothing_instead_of_raising():
    p, vectors = _pipeline(PipelineConfig(dense=False, sparse=False))
    out = await p.retrieve(_state())
    assert out["hits"] == []
    assert (vectors.dense, vectors.sparse, vectors.hybrid) == (0, 0, 0)


async def test_grade_disabled_short_circuits_without_llm():
    p, _v = _pipeline(PipelineConfig(grade=False))
    hits = [_hit(f"c{i}", 1.0, "rrf") for i in range(10)]
    out = await p.grade({**_state(), "hits": hits})
    assert out["grade"] == "sufficient"
    assert len(out["hits"]) == settings.rerank_k


async def test_rewrite_disabled_passes_question_through():
    p, _v = _pipeline(PipelineConfig(rewrite=False))
    out = await p.rewrite(_state())
    assert out["rewritten"] == "how many leave days"


def test_rerank_enabled_argument_overrides_global(monkeypatch):
    monkeypatch.setattr(settings, "enable_cross_encoder", True)
    hits = [_hit("a", 1.0, "rrf"), _hit("b", 0.5, "rrf"), _hit("c", 0.1, "rrf")]
    out = cross_encoder_rerank("q", hits, top_k=2, enabled=False)
    assert [h.chunk_id for h in out] == ["a", "b"]
