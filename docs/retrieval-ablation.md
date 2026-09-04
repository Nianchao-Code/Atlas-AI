# What each stage actually buys

`scripts/ablation.py` runs the golden set through progressively richer
pipelines. Each row adds exactly one stage, so a delta belongs to that stage
and nothing else — and one row adds nothing at all, on purpose.

The same six configurations were measured twice: once on the eight-document
handbook, and once with [10,000 distractors](scaling-the-corpus.md) around it.
Reading them side by side is the point of this page, because **three of the
conclusions the eight-document run supported are false at ten thousand.**

## Ten thousand documents

| Configuration | Recall | Ctx precision | Faithful | Correct | Halluc. | Tokens |
|---|---|---|---|---|---|---|
| **Dense only** | 0.618 ±0.014 | 0.165 ±0.002 | 0.980 | **0.580** ±0.007 | 0.000 | 1093 |
| **Control: dense only again** | 0.618 ±0.014 | 0.165 ±0.007 | 0.979 | 0.585 ±0.013 | 0.000 | 1111 |
| **Sparse only** | 0.922 ±0.000 | 0.285 ±0.002 | 0.977 | 0.849 ±0.000 | 0.000 | 954 |
| **Hybrid + RRF** | **0.961** ±0.000 | 0.267 ±0.000 | 0.977 | 0.877 ±0.013 | 0.000 | 1041 |
| **+ LLM grading (full)** | 0.912 ±0.014 | **0.775** ±0.013 | 0.975 | 0.873 ±0.020 | 0.000 | **266** |
| **Full − query rewrite** | 0.922 ±0.000 | 0.796 ±0.000 | 0.977 | 0.882 ±0.007 | 0.000 | 270 |

53 questions, 2 runs each, mean ±sd. Judge `gpt-4o-mini`, generation `gpt-4o`.
Whole run **$1.14**, 45 minutes, `gpt-4o` 88% of the spend.

## Eight documents

| Configuration | Recall | Ctx precision | Correct | Tokens |
|---|---|---|---|---|
| **Dense only** | 1.000 | 0.505 | 0.925 | 1010 |
| **Control: dense only again** | 1.000 | 0.505 | 0.925 | 1019 |
| **Sparse only** | 1.000 | 0.412 | 0.896 | 1047 |
| **Hybrid + RRF** | 1.000 | 0.458 | 0.925 | 1067 |
| **+ LLM grading (full)** | 0.961 | 0.850 | 0.934 | 288 |
| **Full − query rewrite** | 0.961 | 0.850 | 0.925 | 277 |

Every retriever scored recall 1.000, which is the whole problem: with eight
documents to choose from, the metric had nothing to separate.

## What changed, and what it costs to have believed the old numbers

**Dense retrieval does not survive scale.** Correctness 0.925 → **0.580**,
recall 1.000 → **0.618**. Not a degradation at the margins: on 22 of 53
questions it retrieves no relevant document at all, and generation does not
rescue it — a pipeline given the wrong passages answers wrongly. The
eight-document page said dense "earns its place"; at ten thousand it is the
worst retriever measured.

**Fusing sparse into dense is what holds the system together.** Hybrid reaches
recall **0.961** — 34pp above dense, 4pp above sparse alone — and correctness
0.877 against dense's 0.580. The old page called this stage one that "buys
nothing and costs precision". It bought nothing *at a corpus size where
everything worked*.

**Sparse is the stable one.** Its two runs agreed on all 53 answers and all 53
retrievals, while dense differed on two answers between repeats. Dense
retrieval is sensitive to query phrasing, and query rewrite is an LLM call that
phrases the query differently each time. At eight documents that variance was
invisible because every rewrite retrieved everything.

**Grading still pays, and now it costs recall.** Context precision 0.267 →
0.775 and tokens 1041 → 266 (−74%), which is the same trade it made before. But
recall drops 0.961 → 0.912: at this corpus size the grader discards passages
that were worth keeping. That trade did not exist at eight documents, where
nothing needed discarding.

**Query rewrite still buys nothing, and now has a cost.** Full and full−rewrite
differ by 1.0pp recall and 0.9pp correctness, inside the noise. What is new is
that it is the identified source of dense retrieval's instability — no
measurable benefit, and a measurable variance.

## The noise floor, and why the control row is not decoration

Two identical configurations, `Dense only` and the control:

```
run 1 vs run 1: 4 of 53 answers differ
repeats within a configuration: dense 2, hybrid 1, full 2, sparse 0
```

So **anything under about 4 questions — 7.5pp — is not a finding at this corpus
size.** At eight documents the same control pair agreed on all 53 and the floor
was one question. The floor grew with the corpus, and a reader working from the
old page's ±0.000 columns would have read a 5pp gap as real.

Every difference called out above clears it by a wide margin: dense against
hybrid is 30pp of correctness, dense against sparse 27pp. The ones that do not
clear it — rewrite, and grading's effect on correctness — are reported as not
clearing it.

## What is still true from the eight-document run

The cross-encoder row is absent from both tables because the model is not in
the slim image and the harness [skips it rather than reporting a pass-through
as a measurement](sparse-in-qdrant.md). Its measured contribution of zero
stands as an eight-document result and has not been re-tested at scale.

Hallucination is 0.000 in every configuration at both corpus sizes, including
the ones whose retrieval collapsed. The faithfulness gate holds: given bad
passages the pipeline answers wrongly from them or abstains, rather than
inventing. That is the one claim scaling did not dent.

---

Back to the [README](../README.md).
