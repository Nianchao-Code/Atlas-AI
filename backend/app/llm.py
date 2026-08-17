from __future__ import annotations

import json
from typing import Any

from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from app.config import settings

_client: AsyncOpenAI | None = None


def llm() -> AsyncOpenAI:
    global _client
    if _client is None:
        kwargs: dict[str, Any] = {"api_key": settings.openai_api_key or "sk-missing"}
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        _client = AsyncOpenAI(**kwargs)
    return _client


def llm_configured() -> bool:
    return bool(settings.openai_api_key)


async def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    if not llm_configured():
        return [_hashed_vector(t) for t in texts]
    resp = await llm().embeddings.create(model=settings.embedding_model, input=texts)
    ordered = sorted(resp.data, key=lambda d: d.index)
    return [d.embedding for d in ordered]


async def chat_json(
    *,
    system: str,
    user: str,
    model: str | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    resp = await llm().chat.completions.create(
        model=model or settings.cheap_model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = resp.choices[0].message.content or "{}"
    usage = resp.usage
    data = json.loads(content)
    data["_usage"] = {
        "prompt": usage.prompt_tokens if usage else 0,
        "completion": usage.completion_tokens if usage else 0,
    }
    return data


async def chat_text_stream(
    *,
    system: str,
    user: str,
    model: str | None = None,
    temperature: float = 0.1,
) -> AsyncIterator[str]:
    stream = await llm().chat.completions.create(
        model=model or settings.chat_model,
        temperature=temperature,
        stream=True,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            yield delta


async def chat_text(
    *,
    system: str,
    user: str,
    model: str | None = None,
    temperature: float = 0.1,
) -> tuple[str, int, int]:
    resp = await llm().chat.completions.create(
        model=model or settings.chat_model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    usage = resp.usage
    return (
        resp.choices[0].message.content or "",
        usage.prompt_tokens if usage else 0,
        usage.completion_tokens if usage else 0,
    )


def _hashed_vector(text: str) -> list[float]:
    """Deterministic fallback so tests and dry-runs work without an API key.

    Not a substitute for a real embedding model — retrieval quality in
    keyless mode is lexical-only (BM25 still works).
    """
    import hashlib
    import struct

    dim = settings.embedding_dim
    seed = hashlib.sha256(text.encode("utf-8")).digest()
    out: list[float] = []
    block = seed
    while len(out) < dim:
        block = hashlib.sha256(block).digest()
        for i in range(0, len(block), 4):
            if len(out) >= dim:
                break
            val = struct.unpack(">i", block[i : i + 4])[0] / 2**31
            out.append(val)
    n = sum(x * x for x in out) ** 0.5 or 1.0
    return [x / n for x in out]
