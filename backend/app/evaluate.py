from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from statistics import quantiles

from app.config import settings
from app.graph import Pipeline
from app.llm import chat_json, llm_configured
from app.models import EvalCaseResult, EvalReport, ModelSpend, RunCost
from app.usage import cost_usd, ledger, rates, since


def load_golden() -> list[dict]:
    # Walk every ancestor rather than indexing a fixed depth. samples/ sits three
    # levels above this file in the repo but only two inside the image, where
    # parents[3] does not exist -- and building the candidate list eagerly meant
    # that IndexError fired before any path was even tried.
    candidates = [
        Path(settings.samples_dir) / "eval" / "golden.json",
        Path("/app/samples/eval/golden.json"),
        *(p / "samples" / "eval" / "golden.json" for p in Path(__file__).resolve().parents),
    ]
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    searched = ", ".join(str(p) for p in seen)
    raise FileNotFoundError(f"golden.json not found. Looked in: {searched}")


def _cost_of(baseline: dict, n_cases: int) -> RunCost:
    used = since(baseline)
    per_model, total = cost_usd(used)
    return RunCost(
        by_model={
            model: ModelSpend(
                calls=u.calls,
                prompt_tokens=u.prompt_tokens,
                completion_tokens=u.completion_tokens,
                usd=per_model.get(model),
            )
            for model, u in sorted(used.items())
        },
        total_usd=total,
        per_case_usd=round(total / n_cases, 5) if total is not None and n_cases else None,
        rates={m: [i, o] for m, (i, o) in sorted(rates().items())},
    )


def _hit(expected_docs: list[str], filenames: list[str]) -> bool:
    if not expected_docs:
        return False
    expected = {e.lower() for e in expected_docs}
    got = {f.lower() for f in filenames}
    return bool(expected & got) or any(e in f.lower() for e in expected for f in got)


def _keyword_correctness(answer: str, points: list[str]) -> float:
    if not points:
        return 0.0
    a = answer.lower()
    return sum(1 for p in points if p.lower() in a) / len(points)


async def judge_correctness(question: str, answer: str, points: list[str]) -> float:
    if not llm_configured():
        return _keyword_correctness(answer, points)
    data = await chat_json(
        system=(
            "Grade answer correctness against key points. "
            "Return JSON {score: number 0-1, reason: string}."
        ),
        user=f"Question: {question}\nKey points: {points}\nAnswer: {answer}",
    )
    return float(data.get("score") or 0)


async def run_eval(
    pipeline: Pipeline,
    limit: int | None = None,
    on_case: Callable[[int, int], Awaitable[None]] | None = None,
) -> EvalReport:
    """Run the golden set.

    `on_case` is called after each question with (done, total). It exists so a
    caller can report progress; a four-minute run with no output is
    indistinguishable from a hang.
    """
    cases = load_golden()[: limit or None]
    results: list[EvalCaseResult] = []
    naive_tokens: list[int] = []
    # A baseline rather than a reset: another harness phase may already be
    # counting, and erasing its total to measure this one is not a trade worth
    # making.
    spent_before = ledger.snapshot()

    for case in cases:
        q = case["question"]
        payload = await pipeline.ainvoke(q, use_cache=False)
        filenames = [c["filename"] for c in payload["citations"]]
        doc_ids = [c["doc_id"] for c in payload["citations"]]
        expected = case.get("expected_docs") or []
        expect_abstain = bool(case.get("expect_abstain"))
        retrieval_hit = _hit(expected, filenames + doc_ids) if expected else False
        relevant = sum(1 for f in filenames if _hit(expected, [f])) if expected else 0
        precision = (relevant / len(filenames)) if filenames and expected else 0.0
        faith = float(payload.get("faithfulness") or 0)
        if expect_abstain:
            # Asking the judge to grade a refusal against key points it was
            # never meant to contain scored correct abstentions as failures,
            # which dragged mean_correctness down every time the pipeline did
            # the right thing. For these cases correctness is exactly whether
            # the pipeline held the line.
            correctness = 1.0 if payload["abstained"] else 0.0
        else:
            correctness = await judge_correctness(
                q, payload["answer"], case.get("key_points") or []
            )
        hallucinated = (not payload["abstained"]) and faith < 0.5
        abstention_correct = payload["abstained"] == expect_abstain if expect_abstain else None
        naive = payload["prompt_tokens"] + payload["tokens_saved_vs_naive"]
        naive_tokens.append(naive or 8 * 240)
        results.append(
            EvalCaseResult(
                id=case["id"],
                question=q,
                retrieval_hit=retrieval_hit,
                context_precision=round(precision, 3),
                faithfulness=round(faith, 3),
                answer_correctness=round(correctness, 3),
                hallucinated=hallucinated,
                abstained=payload["abstained"],
                abstention_correct=abstention_correct,
                retrieval_ms=payload["retrieval_ms"],
                prompt_tokens=payload["prompt_tokens"],
                answer=payload["answer"][:500],
            )
        )
        if on_case is not None:
            await on_case(len(results), len(cases))

    cost = _cost_of(spent_before, len(results))

    n = len(results) or 1
    recall_cases = [r for r, c in zip(results, cases, strict=True) if c.get("expected_docs")]
    abstain_cases = [(r, c) for r, c in zip(results, cases, strict=True) if c.get("expect_abstain")]
    retrievals = [r.retrieval_ms for r in results]
    p95 = retrievals[0] if len(retrievals) < 2 else quantiles(retrievals, n=20)[18]
    mean_prompt = sum(r.prompt_tokens for r in results) / n
    mean_naive = sum(naive_tokens) / n if naive_tokens else mean_prompt
    reduction = 0.0 if mean_naive == 0 else max(0.0, (mean_naive - mean_prompt) / mean_naive * 100)

    return EvalReport(
        n=len(results),
        retrieval_recall=(
            sum(r.retrieval_hit for r in recall_cases) / len(recall_cases) if recall_cases else 0.0
        ),
        mean_context_precision=sum(r.context_precision for r in results) / n,
        mean_faithfulness=sum(r.faithfulness for r in results) / n,
        mean_correctness=sum(r.answer_correctness for r in results) / n,
        hallucination_rate=sum(r.hallucinated for r in results) / n,
        abstention_accuracy=(
            sum(1 for r, _ in abstain_cases if r.abstention_correct) / len(abstain_cases)
            if abstain_cases
            else 0.0
        ),
        p95_retrieval_ms=round(p95, 2),
        mean_prompt_tokens=round(mean_prompt, 1),
        naive_prompt_tokens=round(mean_naive, 1),
        token_reduction_pct=round(reduction, 1),
        cost=cost,
        cases=results,
    )
