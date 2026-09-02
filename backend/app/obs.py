from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import redis.asyncio as redis

from app.metrics import CACHE_HITS


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round((p / 100) * (len(s) - 1)))))
    return s[idx]


class Observability:
    """In-process SLIs plus a Redis-backed trace ring.

    Traces are served by the app itself rather than shipped to LangSmith or
    Phoenix: one less external dependency to run, and the trace is visible to
    anyone who can already reach the UI. The cost is that these counters live
    in the process, so they are per-replica -- see the scaling note in README.
    """

    def __init__(self) -> None:
        self._retrieval_ms: list[float] = []
        self._prompt_tokens: list[int] = []
        self.queries = 0
        self.cache_hits = 0
        self.embedding_cache_hits = 0

    def record_query(
        self,
        *,
        retrieval_ms: float,
        prompt_tokens: int,
        cache_hit: bool,
    ) -> None:
        self.queries += 1
        self._retrieval_ms.append(retrieval_ms)
        self._prompt_tokens.append(prompt_tokens)
        if cache_hit:
            self.cache_hits += 1
        if len(self._retrieval_ms) > 2000:
            self._retrieval_ms = self._retrieval_ms[-1000:]
            self._prompt_tokens = self._prompt_tokens[-1000:]

    def snapshot(self, documents: int, chunks: int) -> dict[str, Any]:
        return {
            "documents": documents,
            "chunks": chunks,
            "queries": self.queries,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": (self.cache_hits / self.queries) if self.queries else 0.0,
            "p50_retrieval_ms": _percentile(self._retrieval_ms, 50),
            "p95_retrieval_ms": _percentile(self._retrieval_ms, 95),
            "mean_prompt_tokens": (
                sum(self._prompt_tokens) / len(self._prompt_tokens) if self._prompt_tokens else 0.0
            ),
            "embedding_cache_hits": self.embedding_cache_hits,
        }


obs = Observability()


class Tracer:
    def __init__(self) -> None:
        self.nodes: list[dict[str, Any]] = []
        self._t0 = time.perf_counter()
        self._mark = self._t0

    def span(self, node: str, detail: str = "", **data: Any) -> dict[str, Any]:
        now = time.perf_counter()
        item = {
            "node": node,
            "ms": round((now - self._mark) * 1000, 2),
            "detail": detail,
            "data": data,
        }
        self._mark = now
        self.nodes.append(item)
        return item

    @property
    def total_ms(self) -> float:
        return round((time.perf_counter() - self._t0) * 1000, 2)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Cache:
    def __init__(self, client: redis.Redis) -> None:
        self.r = client

    async def get_embedding(self, text: str) -> list[float] | None:
        raw = await self.r.get(f"emb:{content_hash(text)}")
        if not raw:
            return None
        obs.embedding_cache_hits += 1
        CACHE_HITS.labels(kind="embedding").inc()
        return json.loads(raw)

    async def set_embedding(self, text: str, vector: list[float]) -> None:
        await self.r.set(f"emb:{content_hash(text)}", json.dumps(vector), ex=60 * 60 * 24 * 14)

    @staticmethod
    def _qa_key(principal: str, question: str) -> str:
        # The principal is part of the key, not a filter applied after the
        # read. Answers are built from whatever the caller may retrieve, so a
        # shared key would serve one caller's answer to another.
        return f"qa:{principal}:{content_hash(question.strip().lower())}"

    async def get_semantic(self, principal: str, question: str) -> dict[str, Any] | None:
        raw = await self.r.get(self._qa_key(principal, question))
        return json.loads(raw) if raw else None

    async def set_semantic(self, principal: str, question: str, payload: dict[str, Any]) -> None:
        await self.r.set(
            self._qa_key(principal, question),
            json.dumps(payload, ensure_ascii=False),
            ex=60 * 60 * 12,
        )


class TokenCounter:
    def __init__(self) -> None:
        self._enc = None

    def _get(self):
        if self._enc is None:
            import tiktoken

            try:
                self._enc = tiktoken.encoding_for_model("gpt-4o")
            except Exception:
                self._enc = tiktoken.get_encoding("o200k_base")
        return self._enc

    def count(self, text: str) -> int:
        return len(self._get().encode(text or ""))

    def pack(self, chunks: list[str], budget: int) -> tuple[list[str], int, int]:
        """Greedy pack until budget. Returns kept, used tokens, dropped tokens."""
        kept: list[str] = []
        used = 0
        dropped = 0
        for chunk in chunks:
            n = self.count(chunk)
            if used + n <= budget:
                kept.append(chunk)
                used += n
            else:
                dropped += n
        return kept, used, dropped


tokens = TokenCounter()
