from app.config import settings
from app.rerank import cross_encoder_rerank
from app.vectors import Hit


def _hit(n: str) -> Hit:
    return Hit(
        chunk_id=n,
        doc_id="d",
        filename="f.md",
        text=f"text about {n}",
        parent_text=f"parent {n}",
        section="s",
        score=0.5,
        source="dense",
    )


def test_rerank_disabled_passthrough(monkeypatch):
    monkeypatch.setattr(settings, "enable_cross_encoder", False)
    hits = [_hit("a"), _hit("b"), _hit("c")]
    out = cross_encoder_rerank("question", hits, top_k=2)
    assert len(out) == 2
    assert out[0].chunk_id == "a"
