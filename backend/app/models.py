from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DocumentRecord(BaseModel):
    id: str
    filename: str
    bytes: int
    status: Literal["queued", "indexing", "ready", "failed"]
    chunks: int = 0
    error: str | None = None


class Citation(BaseModel):
    n: int
    doc_id: str
    filename: str
    chunk_id: str
    score: float
    text: str


class TraceNode(BaseModel):
    node: str
    ms: float
    detail: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    question: str
    use_cache: bool = True


class QueryResponse(BaseModel):
    answer: str
    abstained: bool
    citations: list[Citation]
    trace: list[TraceNode]
    retrieval_ms: float
    total_ms: float
    prompt_tokens: int
    completion_tokens: int
    tokens_saved_vs_naive: int
    cache_hit: bool
    rewritten_query: str | None = None
    faithfulness: float | None = None


class EvalCaseResult(BaseModel):
    id: str
    question: str
    retrieval_hit: bool
    context_precision: float
    faithfulness: float
    answer_correctness: float
    hallucinated: bool
    abstained: bool
    abstention_correct: bool | None = None
    retrieval_ms: float
    prompt_tokens: int
    answer: str


class ModelSpend(BaseModel):
    calls: int
    prompt_tokens: int
    completion_tokens: int
    usd: float | None = None


class RunCost(BaseModel):
    """What a run actually spent.

    Tokens are measured. `usd` is arithmetic over a configurable rate table,
    and `rates` echoes the table used so the figure can be checked instead of
    believed. A model with no configured rate leaves `usd` as null rather than
    zero -- zero reads as free.
    """

    by_model: dict[str, ModelSpend] = Field(default_factory=dict)
    total_usd: float | None = None
    per_case_usd: float | None = None
    rates: dict[str, list[float]] = Field(default_factory=dict)


class EvalReport(BaseModel):
    n: int
    retrieval_recall: float
    mean_context_precision: float
    mean_faithfulness: float
    mean_correctness: float
    hallucination_rate: float
    abstention_accuracy: float
    p95_retrieval_ms: float
    mean_prompt_tokens: float
    naive_prompt_tokens: float
    token_reduction_pct: float
    cost: RunCost = Field(default_factory=RunCost)
    cases: list[EvalCaseResult]


class MetricsSnapshot(BaseModel):
    documents: int
    chunks: int
    queries: int
    cache_hits: int
    cache_hit_rate: float
    p50_retrieval_ms: float
    p95_retrieval_ms: float
    mean_prompt_tokens: float
    embedding_cache_hits: int
