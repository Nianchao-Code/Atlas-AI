"""What happens when the cross-encoder is asked for and is not installed.

The base image ships without `sentence-transformers`, so this path runs in
production whenever ENABLE_CROSS_ENCODER is turned on there. It has to
degrade to a pass-through rather than fail a query -- and, less obviously, the
pass-through has to be indistinguishable from having the stage switched off,
because the ablation harness compares those two configurations and would
otherwise attribute the difference to the reranker.
"""

from __future__ import annotations

import app.rerank as rerank_module
from app.rerank import cross_encoder_rerank, reranker_available
from app.vectors import Hit


def _hits(n: int) -> list[Hit]:
    return [
        Hit(
            chunk_id=f"c{i}",
            doc_id="d",
            filename="f.md",
            text=f"chunk {i}",
            parent_text=f"parent {i}",
            section="s",
            score=1.0 - i / 100,
            source="rrf",
        )
        for i in range(n)
    ]


def _without_the_package(monkeypatch):
    def missing():
        raise RuntimeError("sentence-transformers is not installed")

    monkeypatch.setattr(rerank_module, "_load_cross_encoder", missing)


def test_missing_model_passes_candidates_through(monkeypatch, caplog):
    _without_the_package(monkeypatch)
    out = cross_encoder_rerank("q", _hits(20), top_k=12, enabled=True)
    assert [h.chunk_id for h in out] == [f"c{i}" for i in range(12)]


def test_missing_model_is_indistinguishable_from_disabled(monkeypatch):
    # The ablation ran "Hybrid + RRF" and "+ cross-encoder" as separate rows
    # and they are the same pipeline whenever the model is absent. Pinning that
    # is what makes the difference between those rows readable as noise rather
    # than as an effect of reranking.
    _without_the_package(monkeypatch)
    hits = _hits(20)
    degraded = cross_encoder_rerank("q", hits, top_k=12, enabled=True)
    disabled = cross_encoder_rerank("q", hits, top_k=12, enabled=False)
    assert degraded == disabled


def test_availability_reports_false_without_the_package(monkeypatch):
    _without_the_package(monkeypatch)
    assert reranker_available() is False


def test_no_candidates_is_not_an_error(monkeypatch):
    _without_the_package(monkeypatch)
    assert cross_encoder_rerank("q", [], top_k=12, enabled=True) == []
