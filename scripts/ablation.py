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
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (ROOT / "backend", Path("/app")):
    if (_candidate / "app" / "__init__.py").exists():
        sys.path.insert(0, str(_candidate))
        break

from app.config import settings  # noqa: E402
from app.evaluate import load_golden, run_eval  # noqa: E402
from app.graph import Pipeline, PipelineConfig  # noqa: E402
from app.llm import embed_texts, llm_configured  # noqa: E402
from app.obs import Cache  # noqa: E402
from app.redis_client import create_redis  # noqa: E402
from app.rerank import reranker_available  # noqa: E402
from app.vectors import VectorStore  # noqa: E402


@dataclass(frozen=True)
class Variant:
    label: str
    note: str
    config: PipelineConfig
    # True only for the row whose entire point is the cross-encoder. Without
    # the optional model the stage degrades to a pass-through, so that row has
    # to be skipped; the rows after it still mean something with reranking off,
    # which is also how the service is deployed.
    measures_rerank: bool = False


# Cumulative build-up, then one knock-out row. Each step adds exactly one
# stage so a delta can be attributed to that stage and nothing else.
#
# The control row is the same configuration as the first, run again under a
# different name. It exists because a table of small deltas is unreadable
# without knowing how large a delta of zero looks: generation is sampled, so
# two identical pipelines can disagree, and the size of that disagreement is
# the smallest difference anything else in the table is allowed to mean. This
# was not a hypothetical -- an earlier run reported "Hybrid + RRF" at 0.925 and
# "+ cross-encoder" at 0.906 while the cross-encoder was not installed, which
# made them the same pipeline.
VARIANTS = [
    Variant(
        "Dense only",
        "vector search, no fusion",
        PipelineConfig(sparse=False, rerank=False, grade=False),
    ),
    Variant(
        "Control: dense only again",
        "identical to the row above",
        PipelineConfig(sparse=False, rerank=False, grade=False),
    ),
    Variant(
        "Sparse only",
        "Qdrant sparse vectors, IDF-weighted",
        PipelineConfig(dense=False, rerank=False, grade=False),
    ),
    Variant("Hybrid + RRF", "fused inside Qdrant", PipelineConfig(rerank=False, grade=False)),
    Variant(
        "+ cross-encoder",
        "reranked top candidates",
        PipelineConfig(grade=False),
        measures_rerank=True,
    ),
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
        "search and fusion, not embedding API time."
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
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="LABEL",
        help="run only variants whose label contains one of these (case-insensitive)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep configurations already present in --json and run the rest",
    )
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

    cases = load_golden()[: args.limit or None]
    if not cases:
        print("golden set is empty", file=sys.stderr)
        return 2

    # Prime the embedding cache on the raw questions so the first variant is
    # not charged for cold embeddings the later ones get for free.
    await embed_texts([c["question"] for c in cases])

    # A 50-minute measurement should not be all-or-nothing. --resume reloads
    # what a previous run finished; the loop below skips those labels.
    raw: dict[str, list[dict]] = {}
    if args.resume and args.json and args.json.exists():
        raw = json.loads(args.json.read_text(encoding="utf-8"))
        print(f"  resuming, {len(raw)} configuration(s) already measured", file=sys.stderr)
    rows: list[dict] = [
        {
            "label": v.label,
            "note": v.note,
            "metrics": {k: _agg([run[k] for run in raw[v.label]]) for k, _l, _f in METRICS},
        }
        for v in VARIANTS
        if v.label in raw
    ]
    started = time.time()

    # A row that says "+ cross-encoder" while the model is absent measures a
    # pass-through, which is worse than no row: it reads as evidence.
    no_reranker = not reranker_available()
    if no_reranker:
        print(
            "  note: sentence-transformers is absent. The cross-encoder row is skipped\n"
            "        and the rows after it run with reranking off, as deployed."
        )

    for variant in VARIANTS:
        if variant.label in raw:
            continue
        if args.only and not any(t.lower() in variant.label.lower() for t in args.only):
            continue
        if no_reranker and variant.measures_rerank:
            continue
        if no_reranker and variant.config.rerank:
            variant = replace(variant, config=replace(variant.config, rerank=False))
        # Nothing to warm: both retrievers read Qdrant, so there is no
        # process-local snapshot for a first measured query to pay for.
        pipeline = Pipeline(cache, vectors, config=variant.config)
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
