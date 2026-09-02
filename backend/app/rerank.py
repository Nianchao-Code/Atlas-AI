from __future__ import annotations

import structlog

from app.config import settings
from app.vectors import Hit

log = structlog.get_logger()
_cross_encoder = None


def _load_cross_encoder():
    """Import lazily: sentence-transformers is an optional extra, and the base
    image does not carry it."""
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # not installed rather than not downloadable
            raise RuntimeError(
                "ENABLE_CROSS_ENCODER is on but sentence-transformers is not "
                'installed. Install the extra: pip install -e ".[rerank]"'
            ) from exc
        _cross_encoder = CrossEncoder(settings.cross_encoder_model)
    return _cross_encoder


def cross_encoder_rerank(
    question: str,
    hits: list[Hit],
    top_k: int | None = None,
    enabled: bool | None = None,
) -> list[Hit]:
    """RRF candidates → cross-encoder scores. Skipped when disabled or empty.

    `enabled` lets a caller override the global switch; the ablation harness
    uses it to compare pipelines within one process.
    """
    if enabled is None:
        enabled = settings.enable_cross_encoder
    if not hits or not enabled:
        return hits[: top_k or settings.rerank_k]
    try:
        model = _load_cross_encoder()
    except Exception as exc:
        # Passing the candidates through unranked is the right degradation:
        # the ablation says this stage changes nothing on this corpus anyway.
        log.warning("rerank.model_unavailable", error=str(exc)[:160])
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
