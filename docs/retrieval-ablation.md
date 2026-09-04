# What each stage actually buys

`scripts/ablation.py` runs the golden set through progressively richer
pipelines. Each row adds exactly one stage, so a delta belongs to that stage
and nothing else — and one row adds nothing at all.

> **Every number on this page was measured against eight documents.** The
> corpus is now [10,008](scaling-the-corpus.md), and retrieval measured there
> contradicts the conclusion below about dense retrieval: at eight documents
> dense earns its place and fusing sparse into it buys nothing, while at ten
> thousand dense finds no relevant document at all for 22 of 53 questions and
> sparse finds one for every question. The ablation has not been re-run at the
> larger corpus — it needs generation rather than retrieval alone — so this page
> stands as the eight-document result and should be read as one.



| Configuration | Recall | Ctx precision | Faithful | Correct | Halluc. | p95 ms | Tokens | Cost |
|---|---|---|---|---|---|---|---|---|
| **Dense only** | 1.000 ±0.000 | **0.505** ±0.002 | 0.981 ±0.000 | 0.925 ±0.000 | 0.000 | 462 ±135 | 1010 ±2 | $0.220 |
| **Control: dense only again** | 1.000 ±0.000 | 0.505 ±0.002 | 0.981 ±0.000 | 0.925 ±0.000 | 0.000 | 1094 ±1060 | 1019 ±6 | $0.223 |
| **Sparse only** | 1.000 ±0.000 | 0.412 ±0.002 | 0.981 ±0.000 | 0.896 ±0.013 | 0.000 | 805 ±614 | 1047 ±18 | $0.224 |
| **Hybrid + RRF** | 1.000 ±0.000 | 0.458 ±0.002 | 0.980 ±0.001 | 0.925 ±0.000 | 0.000 | 426 ±62 | 1067 ±1 | $0.228 |
| **+ LLM grading (full)** | 0.961 ±0.000 | **0.850** ±0.000 | 0.975 ±0.003 | **0.934** ±0.013 | 0.000 | 377 ±29 | **288** ±0 | **$0.115** |
| **Full − query rewrite** | 0.961 ±0.000 | 0.850 ±0.000 | 0.976 ±0.001 | 0.925 ±0.000 | 0.000 | 1189 ±1191 | 277 ±0 | $0.107 |

53 questions, 2 runs per configuration, mean ±sd. Judge `gpt-4o-mini`,
generation `gpt-4o`. Whole run: **$1.12**, 45 minutes, `gpt-4o` 88% of the
spend. Reproduce with `python scripts/ablation.py --repeats 2`. The
cross-encoder row is missing because the model is not installed in the slim
image and the harness [skips it rather than reporting a pass-through as a
measurement](sparse-in-qdrant.md).

**Read the control row first.** It is `Dense only` run a second time under a
different name — the same configuration, so every difference between those two
rows is noise and nothing else. Case by case, they agreed on **53 of 53
questions**. Their correctness, precision, faithfulness and cost are identical
to three decimals.

**Their p95 latency differs by 2.4x** — 462ms against 1094ms, ±1060.

That is the control row paying for itself. The p95 column cannot distinguish
configurations at this sample size: it is dominated by variance in the model
provider's response time, not by retrieval. `Full − query rewrite` at 1189ms
against the full pipeline's 377ms is the same artifact, not a finding. Retrieval
latency is measured properly under [Throughput](operations.md#throughput), against a warm
cache and with the model out of the path. **Nothing in this column should be
read as a difference between rows.**

Aggregates hide where the differences live, so the same runs by question type
(answer correctness, both runs averaged):

| Category | n | Dense | Sparse | Full | What it stresses |
|---|---|---|---|---|---|
| `lookup` | 10 | **1.000** | 0.950 | 1.000 | single stated figure |
| `policy` | 9 | 1.000 | 1.000 | **0.889** | the answer is a rule, not a figure |
| `distractor` | 8 | 1.000 | 1.000 | **0.875** | the same figure appears in another document |
| `abstain` | 6 | 0.333 | 0.333 | **0.750** | document retrievable, fact absent from it |
| `multi-hop` | 5 | 1.000 | 1.000 | 1.000 | answer spans two documents |
| `lexical` | 5 | 1.000 | 1.000 | 1.000 | exact identifiers (`KV-2025-441`, `IR-4`) |
| `semantic` | 4 | **1.000** | **0.750** | 1.000 | paraphrased away from corpus wording |
| `injection` | 3 | 1.000 | 1.000 | 1.000 | user and corpus prompt injection |
| `deferral` | 2 | 1.000 | 1.000 | 1.000 | corpus says where the answer lives, not what it is |
| `withheld` | 1 | 1.000 | 1.000 | 1.000 | corpus names a figure as unpublished |

**What the measurements support:**

- **Grading is the stage that pays, and it pays twice.** Context precision
  0.458 → 0.850, prompt tokens 1067 → 288 (−73%), abstention correctness more
  than doubled (0.333 → 0.750). It also **halves the bill**: $0.228 → $0.115
  per two runs, because a smaller context is a smaller prompt on the expensive
  model. The token-reduction claim shows up on the invoice, not just in a
  counter.
- **Dense retrieval earns its place, on the strength of one category.** It ties
  sparse everywhere except `semantic` — questions deliberately worded away from
  the corpus vocabulary, where sparse scores 0.750 against dense 1.000 — and
  `lookup`, 0.950 against 1.000.
- **Fusing sparse into dense buys nothing and costs precision.** Hybrid matches
  dense on correctness to three decimals (0.925 both) and gives up context
  precision to do it (0.505 → 0.458): the lexical hits displace passages the
  dense ranker had already ordered correctly.
- **Grading no longer costs correctness.** It used to look like a trade — the
  previous implementation put the full pipeline at 0.915 against dense-only
  0.925. It is now 0.934 against 0.925, and that 0.9pp is well inside the noise
  floor, so the honest reading is that the two are level. What survives is the
  *shape* of the trade below.
- **Grading still over-abstains.** It is the only reason `policy` falls to
  0.889 and `distractor` to 0.875: on `pii-to-llm` and `personal-claude-account`
  the corpus states the rule plainly, dense-only answers correctly, and the full
  pipeline refuses. That is a real regression and it is why the corrective loop
  is a trade rather than a free win.
- **Query rewrite has no measurable effect.** Full and full−rewrite differ by
  0.9pp, which is half of one question. An earlier 14-question run credited
  rewrite with +3.5pp; that was noise, and it is the clearest argument in this
  repo for sizing an eval set before trusting it.

**The noise floor, measured three ways.** Two identical configurations agreed
on all 53 questions. But `Sparse only` disagreed with *itself* across its two
repeats — `leave-approval-four-days`, 1.00 → 0.00 — and so did the full
pipeline, `karm-payload`, 0.00 → 1.00. One question is **1.9pp**, and that is
the smallest difference anything in this table is allowed to mean. Every
disagreement observed, in this run and the previous one, has been exactly one
question. Two rows that differ by less than that are the same row.

**What the measurements do not support:** the `lexical` category was built to
show lexical search winning on exact identifiers and it did not — dense
retrieves `KV-2025-441` and `IR-4` perfectly well at this corpus size. Recall is
also saturated at 1.000 for every retriever, because `_hit` only requires one of
the expected documents and there are only eight to choose from. Recall cannot
discriminate anything here regardless of how the questions are written; that is
a property of the metric and the corpus size, not of the question set.

Six of 53 cases fail under the full pipeline. They are kept deliberately: four
are the over-abstention and cross-document-conflation bugs named above, and the
set exists to keep catching them.


---

Back to the [README](../README.md).
