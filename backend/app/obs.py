from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import redis.asyncio as redis


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round((p / 100) * (len(s) - 1)))))
    return s[idx]


class Observability:
    """In-process SLIs plus a Redis-backed trace ring.

    LangSmith/Phoenix are fine in a company that already pays for them.
    A portfolio demo should not require a SaaS login to show a trace —
    the product itself is the observability surface.
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
        return json.loads(raw)

    async def set_embedding(self, text: str, vector: list[float]) -> None:
        await self.r.set(f"emb:{content_hash(text)}", json.dumps(vector), ex=60 * 60 * 24 * 14)

    async def get_semantic(self, question: str) -> dict[str, Any] | None:
        raw = await self.r.get(f"qa:{content_hash(question.strip().lower())}")
        return json.loads(raw) if raw else None

    async def set_semantic(self, question: str, payload: dict[str, Any]) -> None:
        await self.r.set(
            f"qa:{content_hash(question.strip().lower())}",
            json.dumps(payload, ensure_ascii=False),
            ex=60 * 60 * 12,
        )

    async def nearest_semantic(
        self,
        query_vec: list[float],
        threshold: float,
    ) -> dict[str, Any] | None:
        """Exact-question cache is cheap; this catches paraphrases.

        We keep a small Redis ZSET of recent query embeddings rather than
        standing up a second vector index. At this QPS it is enough, and
        the interview story is 'cache the question, not just the chunk'.
        """
        keys = await self.r.lrange("qa:recent", 0, 199)
        best: tuple[float, dict[str, Any]] | None = None
        for key in keys:
            blob = await self.r.get(f"qa:vec:{key}")
            cached = await self.r.get(f"qa:{key}")
            if not blob or not cached:
                continue
            vec = json.loads(blob)
            sim = _cosine(query_vec, vec)
            if sim >= threshold and (best is None or sim > best[0]):
                best = (sim, json.loads(cached))
        return None if best is None else best[1]

    async def remember_query_vec(self, question: str, vec: list[float], payload: dict[str, Any]) -> None:
        key = content_hash(question.strip().lower())
        await self.set_semantic(question, payload)
        await self.r.set(f"qa:vec:{key}", json.dumps(vec), ex=60 * 60 * 12)
        await self.r.lpush("qa:recent", key)
        await self.r.ltrim("qa:recent", 0, 199)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


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
