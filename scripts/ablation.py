#!/usr/bin/env python3
"""Retrieval ablation: measure what each stage of the stack actually buys.

A single set of eval numbers proves nothing about design choices. This runs
the golden set through progressively richer pipelines and prints the deltas,
so claims like "hybrid beats dense" or "the cross-encoder is worth it" are
answered with measurements from this corpus instead of folklore.

Runs anywhere the backend package imports and Redis/Qdrant are reachable:

    python scripts/ablation.py --repeats 3 --out docs/ablation.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (ROOT / "backend", Path("/app")):
    if (_candidate / "app" / "__init__.py").exists():
        sys.path.insert(0, str(_candidate))
        break

from app.config import settings  # noqa: E402
from app.evaluate import load_golden, run_eval  # noqa: E402
from app.graph import Pipeline, PipelineConfig  # noqa: E402
from app.hybrid import BM25Index  # noqa: E402
from app.llm import embed_texts, llm_configured  # noqa: E402
from app.obs import Cache  # noqa: E402
from app.redis_client import create_redis  # noqa: E402
from app.vectors import VectorStore  # noqa: E402


@dataclass(frozen=True)
class Variant:
    label: str
    note: str
    config: PipelineConfig


# Cumulative build-up, then one knock-out row. Each step adds exactly one
# stage so a delta can be attributed to that stage and nothing else.
VARIANTS = [
    Variant(
        "Dense only",
        "vector search, no fusion",
        PipelineConfig(sparse=False, rerank=False, grade=False),
    ),
    Variant(
        "BM25 only",
        "lexical search, no fusion",
        PipelineConfig(dense=False, rerank=False, grade=False),
    ),
    Variant("Hybrid + RRF", "dense + BM25 fused", PipelineConfig(rerank=False, grade=False)),
    Variant("+ cross-encoder", "reranked top candidates", PipelineConfig(grade=False)),
    Variant("+ LLM grading (full)", "corrective loop, abstains", PipelineConfig()),
    Variant("Full - query rewrite", "isolates HyDE/rewrite", PipelineConfig(rewrite=False)),
]

METRICS = [
    ("retrieval_recall", "Recall", "{:.3f}"),
    ("mean_context_precision", "Ctx precision", "{:.3f}"),
    ("mean_faithfulness", "Faithful", "{:.3f}"),
    ("mean_correctness", "Correct", "{:.3f}"),
    ("hallucination_rate", "Halluc.", "{:.3f}"),
    ("p95_retrieval_ms", "p95 ms", "{:.0f}"),
    ("mean_prompt_tokens", "Tokens", "{:.0f}"),
]


def _agg(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def _cell(mean: float, sd: float, fmt: str, repeats: int) -> str:
    if repeats < 2:
        return fmt.format(mean)
    return f"{fmt.format(mean)} ±{fmt.format(sd)}"


def render_markdown(rows: list[dict], repeats: int, n_cases: int) -> str:
    header = ["Configuration"] + [label for _, label, _ in METRICS]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for row in rows:
        cells = [f"**{row['label']}**"]
        for key, _label, fmt in METRICS:
            mean, sd = row["metrics"][key]
            cells.append(_cell(mean, sd, fmt, repeats))
        lines.append("| " + " | ".join(cells) + " |")
    footer = (
        f"\n{n_cases} golden questions, {repeats} run(s) per configuration"
        f"{', mean ±sd' if repeats > 1 else ''}. "
        f"Judge: `{settings.cheap_model}`, generation: `{settings.chat_model}`. "
        "Query embeddings are content-hash cached in Redis, so p95 covers vector "
        "search + BM25 + rerank, not embedding API time."
    )
    return "\n".join(lines) + "\n" + footer + "\n"


def checkpoint(rows, raw, repeats, n_cases, out: Path | None, js: Path | None) -> None:
    """Flush after every variant.

    An earlier run died 13 configurations in and lost everything, because
    results were only written at the end. Partial output beats no output.
    """
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_markdown(rows, repeats, n_cases), encoding="utf-8")
    if js:
        js.parent.mkdir(parents=True, exist_ok=True)
        js.write_text(json.dumps(raw, indent=2), encoding="utf-8")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3, help="runs per configuration")
    parser.add_argument("--limit", type=int, default=None, help="cap golden cases (smoke runs)")
    parser.add_argument("--out", type=Path, default=None, help="write the markdown table here")
    parser.add_argument("--json", type=Path, default=None, help="write raw per-run reports here")
    args = parser.parse_args()

    if not llm_configured():
        print("WARNING: no API key, judge degrades to keyword overlap", file=sys.stderr)

    # The cross-encoder rows need the model reachable; PipelineConfig decides
    # per-variant whether it actually runs.
    settings.enable_cross_encoder = True

    r = create_redis()
    cache = Cache(r)
    vectors = VectorStore()
    vectors.ensure()
    bm25 = BM25Index()
    bm25.rebuild(vectors.scroll_all())

    cases = load_golden()[: args.limit or None]
    if not cases:
        print("golden set is empty", file=sys.stderr)
        return 2

    # Prime the embedding cache on the raw questions so the first variant is
    # not charged for cold embeddings the later ones get for free.
    await embed_texts([c["question"] for c in cases])

    rows: list[dict] = []
    raw: dict[str, list[dict]] = {}
    started = time.time()

    for variant in VARIANTS:
        pipeline = Pipeline(cache, vectors, bm25, config=variant.config)
        # Sync the BM25 snapshot before timing: a cold rebuild inside the first
        # measured query added ~19s to p95 and told us nothing about retrieval.
        await pipeline.warm()
        runs: list[dict] = []
        for i in range(args.repeats):
            t0 = time.time()
            report = await run_eval(pipeline, limit=args.limit)
            runs.append(report.model_dump())
            print(
                f"  {variant.label} run {i + 1}/{args.repeats} "
                f"recall={report.retrieval_recall:.3f} "
                f"correct={report.mean_correctness:.3f} "
                f"({time.time() - t0:.0f}s)",
                file=sys.stderr,
                flush=True,
            )
        raw[variant.label] = runs
        rows.append(
            {
                "label": variant.label,
                "note": variant.note,
                "metrics": {key: _agg([run[key] for run in runs]) for key, _l, _f in METRICS},
            }
        )
        checkpoint(rows, raw, args.repeats, len(cases), args.out, args.json)

    await r.aclose()

    table = render_markdown(rows, args.repeats, len(cases))
    print(f"\nTotal wall clock: {time.time() - started:.0f}s\n", file=sys.stderr)
    print(table)

    checkpoint(rows, raw, args.repeats, len(cases), args.out, args.json)
    for path in (args.out, args.json):
        if path:
            print(f"wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
