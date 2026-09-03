# What a run costs

Every number in this repo is measured except one: what it costs to produce
them. That gap produced a wrong answer. Asked to estimate a full ablation, the
arithmetic used **232 prompt tokens per question** — the figure the *full*
pipeline reports after LLM grading trims the context — when most ablation rows
run with grading off and pack ~1000 instead, on the more expensive model. The
estimate was low by about half, and the account ran dry mid-run.

The correction was low too, in the other direction. Both were guesses. So every
model call now records its own usage, and a run reports what it actually spent:

```
gpt-4o                       3 calls     738 in     77 out  $0.0026
gpt-4o-mini                 12 calls    3058 in    535 out  $0.0008
text-embedding-3-small       3 calls     132 in      0 out  $0.0
total $0.0034   per case $0.00113
```

Three full-pipeline questions. **$0.0011 per case**, so a 53-question run is
about **$0.06** — an order of magnitude below the estimate that replaced the
first estimate.

The full ablation then billed **$1.12** for twelve 53-question runs, and the
per-configuration figures say something the token counter alone did not:

| | Tokens/question | Cost per 2 runs |
| --- | --- | --- |
| Retrieval only (no grading) | ~1050 | ~$0.22 |
| Full pipeline (grading on) | 288 | **$0.11** |

**LLM grading halves the bill.** It is the same finding as "−73% prompt tokens",
arriving through the invoice instead of a counter — and the two do not match
exactly, because grading spends its own `gpt-4o-mini` calls to save `gpt-4o`
ones. `gpt-4o` was 88% of the run's spend on 24% of its calls.

**Tokens are the measurement; dollars are arithmetic.** The rate table is
configuration (`MODEL_PRICES`), not fact, because a price compiled into source
is wrong the moment it is committed. The report echoes the rates it used so the
figure can be checked rather than believed, and a model with **no** configured
rate reports `null` rather than `0` — zero reads as "this was free", which is
the wrong thing to believe about a model nobody has priced.

One hole was worth closing deliberately: streamed answers report no usage
unless `stream_options={"include_usage": True}` is set, and streaming is the
path real users take. An accounting system blind to its most-used path is worse
than none, because the total still looks authoritative.


---

Back to the [README](../README.md).
