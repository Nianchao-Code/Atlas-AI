"""Prometheus instrumentation.

`/api/v1/metrics` reports this replica's own counters and drives the SLI tab in
the UI. It cannot answer "what is p95 across the fleet", because the numbers
live in process memory and every replica has its own. This module is the
answer to that: each process exposes a scrape endpoint, and aggregation is
Prometheus's job rather than the app's.

Cardinality is deliberate. Nothing here is labelled by principal except the
rate-limit counter, where knowing who is being throttled is the entire point
and the label space is bounded by the configured keys. Everywhere else a
principal label would put an identity into series that outlive it, for a
breakdown nobody reads.
"""
from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

QUERIES = Counter(
    "atlas_queries_total",
    "Queries served, by how they ended",
    ["outcome"],  # answered | abstained | cached | blocked
)

RETRIEVAL_SECONDS = Histogram(
    "atlas_retrieval_seconds",
    "Time in the retrieve node: embed, vector search, BM25, fusion",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

PROMPT_TOKENS = Histogram(
    "atlas_prompt_tokens",
    "Prompt tokens sent to the generation model after packing",
    buckets=(100, 250, 500, 1000, 2000, 4000, 8000),
)

FAITHFULNESS = Histogram(
    "atlas_faithfulness_score",
    "Faithfulness score assigned to served answers",
    # 0.7 is the serving gate, so it gets its own edge: the bucket below it is
    # the regeneration-or-abstain rate.
    buckets=(0.5, 0.7, 0.8, 0.9, 0.95, 1.0),
)

CACHE_HITS = Counter(
    "atlas_cache_hits_total",
    "Cache hits, by which cache answered",
    ["kind"],  # semantic | embedding
)

AUTH_REJECTIONS = Counter(
    "atlas_auth_rejections_total",
    "Requests refused for a missing or invalid API key",
)

RATE_LIMITED = Counter(
    "atlas_rate_limited_total",
    "Requests refused because the principal exhausted its window",
    ["principal"],
)

INDEX_JOBS = Counter(
    "atlas_index_jobs_total",
    "Index jobs consumed from the queue",
    ["outcome"],  # indexed | failed
)

CORPUS_DOCUMENTS = Gauge("atlas_corpus_documents", "Documents in the catalogue")
CORPUS_CHUNKS = Gauge("atlas_corpus_chunks", "Chunks across all documents")


def observe_query(
    *,
    retrieval_ms: float,
    prompt_tokens: int,
    cache_hit: bool,
    abstained: bool = False,
    blocked: bool = False,
    faithfulness: float | None = None,
) -> None:
    """Record one served query.

    A cached answer is counted as `cached` rather than `answered`: rolling the
    two together makes the cache hit rate invisible in the query rate, which is
    the number most worth watching when the cache changes.
    """
    if blocked:
        outcome = "blocked"
    elif cache_hit:
        outcome = "cached"
    elif abstained:
        outcome = "abstained"
    else:
        outcome = "answered"
    QUERIES.labels(outcome=outcome).inc()

    if cache_hit:
        CACHE_HITS.labels(kind="semantic").inc()
    else:
        # A cache hit skips retrieval entirely; recording a near-zero sample
        # would drag the latency histogram toward zero and hide real work.
        RETRIEVAL_SECONDS.observe(retrieval_ms / 1000.0)
        if prompt_tokens:
            PROMPT_TOKENS.observe(prompt_tokens)

    if faithfulness is not None:
        FAITHFULNESS.observe(float(faithfulness))


def set_corpus_size(documents: int, chunks: int) -> None:
    CORPUS_DOCUMENTS.set(documents)
    CORPUS_CHUNKS.set(chunks)


def render() -> tuple[bytes, str]:
    """Scrape payload. Single process per pod, so the default registry is
    correct; running uvicorn with --workers would need multiprocess mode."""
    return generate_latest(), CONTENT_TYPE_LATEST
