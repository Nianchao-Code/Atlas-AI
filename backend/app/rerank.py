from __future__ import annotations

import structlog

from app.config import settings
from app.vectors import Hit

log = structlog.get_logger()
_cross_encoder = None


def _load_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder(settings.cross_encoder_model)
    return _cross_encoder


def cross_encoder_rerank(question: str, hits: list[Hit], top_k: int | None = None) -> list[Hit]:
    """RRF candidates → cross-encoder scores. Skipped when disabled or empty."""
    if not hits or not settings.enable_cross_encoder:
        return hits[: top_k or settings.rerank_k]
    try:
        model = _load_cross_encoder()
    except Exception:
        log.warning("rerank.model_unavailable")
        return hits[: top_k or settings.rerank_k]

    limit = top_k or settings.rerank_k
    pairs = [[question, h.text[:512]] for h in hits]
    scores = model.predict(pairs)
    ranked = sorted(zip(hits, scores, strict=True), key=lambda x: float(x[1]), reverse=True)
    out: list[Hit] = []
    for hit, score in ranked[:limit]:
        out.append(
            Hit(
                chunk_id=hit.chunk_id,
                doc_id=hit.doc_id,
                filename=hit.filename,
                text=hit.text,
                parent_text=hit.parent_text,
                section=hit.section,
                score=float(score),
                source="cross-encoder",
            )
        )
    return out
