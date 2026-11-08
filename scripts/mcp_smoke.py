#!/usr/bin/env python3
"""Drive the MCP server against a live Atlas, the way a client would.

The unit tests mock the HTTP layer and the end-to-end test stubs the backend,
so neither of them has ever seen a real retrieval. This launches the actual
`atlas-mcp` process over stdio -- the same transport Claude Desktop uses --
points it at a running API, and calls the tools.

It also times the two question-answering tools against each other, because
the argument for `search_documents` existing is that a caller which is itself
a model should not pay for Atlas's generation. That argument deserves a
number rather than a paragraph.

    docker compose up -d
    python scripts/mcp_smoke.py --question "How many annual leave days in the first year?"
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

from mcp import Client, StdioServerParameters

QUESTION = "How many annual leave days in the first year?"


def _params(base_url: str, api_key: str) -> StdioServerParameters:
    env = {"ATLAS_BASE_URL": base_url, "ATLAS_API_KEY": api_key}
    # The child needs enough of the parent environment to find its own
    # interpreter and packages; it does not need the rest.
    for name in ("PATH", "SYSTEMROOT", "VIRTUAL_ENV", "PYTHONPATH"):
        if name in os.environ:
            env[name] = os.environ[name]
    return StdioServerParameters(command=sys.executable, args=["-m", "atlas_mcp.server"], env=env)


def _text(result) -> str:
    return "\n".join(b.text for b in result.content if getattr(b, "type", None) == "text")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("ATLAS_BASE_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--api-key", default=os.environ.get("ATLAS_API_KEY", ""))
    ap.add_argument("--question", default=QUESTION)
    ap.add_argument("--chars", type=int, default=700, help="how much of each answer to print")
    args = ap.parse_args()

    # Read the backend's mode before timing anything. Without a model key the
    # generation step returns a retrieved extract instead of calling a model,
    # and a latency comparison printed without saying so would be reporting a
    # model call that never happened.
    import httpx

    async with httpx.AsyncClient(timeout=10) as probe:
        health = (await probe.get(f"{args.base_url}/health")).json()
    live_llm = bool(health.get("llm"))
    print(f"backend {args.base_url} -> {health}")
    if not live_llm:
        print(
            "  NOTE: no model key configured. Dense vectors are the hashed "
            "fallback (retrieval is lexical-only) and ask_handbook returns an "
            "extract rather than a generated answer. Quality below is not "
            "representative; the plumbing is.\n"
        )

    async with Client(_params(args.base_url, args.api_key)) as client:
        tools = sorted(t.name for t in (await client.list_tools()).tools)
        print(f"tools: {', '.join(tools)}\n")

        t0 = time.perf_counter()
        search = await client.call_tool("search_documents", {"question": args.question, "top_k": 3})
        search_ms = (time.perf_counter() - t0) * 1000

        # use_cache=False deliberately. With it on, a second run of this
        # script times the paraphrase cache rather than the graph -- the
        # first run here reported 1703ms and the second 5ms for the same
        # call, which is a cache hit wearing a benchmark's clothes.
        t0 = time.perf_counter()
        ask = await client.call_tool(
            "ask_handbook", {"question": args.question, "use_cache": False}
        )
        ask_ms = (time.perf_counter() - t0) * 1000

        print("=" * 72)
        print(f"search_documents  {search_ms:7.0f}ms   error={search.is_error}")
        print("=" * 72)
        print(_text(search)[: args.chars])

        print()
        print("=" * 72)
        print(f"ask_handbook      {ask_ms:7.0f}ms   error={ask.is_error}")
        print("=" * 72)
        print(_text(ask)[: args.chars])

        print()
        if not search.is_error and not ask.is_error and search_ms > 0:
            ratio = f"{ask_ms / search_ms:.1f}x"
            if live_llm:
                print(
                    f"ask_handbook took {ratio} the wall clock of search_documents, "
                    f"and one chat completion that search_documents did not spend."
                )
            else:
                print(
                    f"ask_handbook took {ratio} the wall clock of search_documents "
                    f"WITHOUT making a model call -- that gap is the rest of the "
                    f"graph (guard, cache, rewrite, grade, compress, faithfulness). "
                    f"With a key the gap is larger; this run cannot say by how much."
                )

        # The injection bait is in the sample corpus on purpose. Pull it and
        # show what the envelope does to planted instructions, because this
        # is the path Atlas's measured injection result does not cover.
        bait = await client.call_tool(
            "search_documents", {"question": "payroll password stored in the handbook", "top_k": 2}
        )
        print("\n" + "=" * 72)
        print("search_documents against the planted-injection document")
        print("=" * 72)
        print(_text(bait)[: args.chars])

        # The gate, exercised rather than asserted: this run did not set
        # ATLAS_MCP_ALLOW_WRITES, so the spending tool must refuse.
        gated = await client.call_tool("run_eval", {"limit": 1})
        print(f"\nrun_eval without writes enabled -> error={gated.is_error}")
        print("  " + _text(gated)[:200])

        return 0 if not (search.is_error or ask.is_error) and gated.is_error else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
