# The sparse index moved into Qdrant

[Limits](limits.md) used to open with "runs as a single replica," and one
subsystem was the reason. Sparse retrieval was an in-process `BM25Okapi`,
rebuilt from the whole corpus by every process that answered queries. Keeping
those copies in step took a `bm25:rev` counter in Redis, a background poller, a
worker thread so the rebuild stayed off the event loop, and a frozen snapshot
published in a single assignment so a concurrent search could not read a
half-swapped index. Every one of those pieces existed to make one piece of
mutable state agree across processes.

Qdrant stores sparse vectors beside the dense ones, computes IDF across the
collection itself, and fuses two rankings with RRF server-side. So the state
could be deleted rather than coordinated.

| | Before | After |
| --- | --- | --- |
| Sparse index | `BM25Okapi`, one per process | vectors on the same Qdrant points |
| IDF | recomputed per process on every rebuild | Qdrant, across the collection |
| Fusion | client-side RRF over two result lists | `Prefetch` + `FusionQuery(RRF)`, one round trip |
| Keeping replicas in step | `bm25:rev`, a 2s poller, a rebuild thread, a snapshot swap | nothing |
| A write becomes visible | dense at once, sparse up to one tick later | both at once |
| Deleting a document | delete vectors, bump a counter, rebuild | delete the points |

`hybrid.py`, `Pipeline.warm()`, `test_bm25_refresh.py` and the `rank-bm25`
dependency all went with it.

**It is TF-IDF, not BM25, and that was a choice.** Qdrant applies IDF; the
values this code sends are the term-frequency half. BM25 adds saturation (`k1`)
and length normalisation (`b`), and `b` needs a corpus average — a statistic
that would have to be maintained somewhere and would restate every stored
vector each time it moved, which is the coordination this change exists to
delete. Two measurements say what that gives up here: **83.7% of term
occurrences inside a chunk are singletons**, so `k1` has almost nothing to act
on, and chunk length varies with a **coefficient of variation of 0.333**, so
`b` is the real loss. Whether that costs anything is a retrieval question, and
the ablation answers it.

> The comparison below was measured on eight documents. At
> [ten thousand](retrieval-ablation.md) the same configurations separate far
> more sharply — sparse 0.849 against dense 0.580 — so read this table as
> evidence that the Qdrant implementation is at least as good as the library it
> replaced, not as a measurement of what sparse retrieval is worth.

**The replacement retrieves better than the library it replaced.** Same corpus,
same golden set, same generation and judge models:

| Retriever | Ctx precision | Correctness |
| --- | --- | --- |
| Dense only | 0.508 → 0.505 | 0.925 → 0.925 |
| Sparse only | 0.403 → **0.412** | 0.863 → **0.896** |
| Hybrid + RRF | 0.434 → **0.458** | 0.906 → **0.925** |

Dense is unchanged, which is the control: nothing about the dense path moved.
Sparse-only correctness gains **3.3pp** — under two questions of 53, so only
just above [what this harness can resolve](retrieval-ablation.md) —
and context precision gains for both sparse and hybrid, where the run-to-run sd
of ±0.002 makes the difference solid. One likely reason for the correctness
gain, not measured: the old implementation dropped hits scoring zero, so it
often returned fewer than `retrieve_k` candidates, while the Qdrant query
returns a full ranking.

**The hybrid result also retracts an earlier claim.** This README used to say
hybrid retrieval was worse than dense — "fusing BM25 into dense *lowers*
correctness (0.925 → 0.906)". That gap is 1.9pp, which is one question of 53,
and the same runs showed 1.9pp is the smallest difference the harness can
resolve. So the old claim was over-read. What the measurements support now is
narrower and duller: **hybrid and dense are equal on correctness, and hybrid has
lower context precision.** Dense alone is still what the deployment would choose
on the evidence; hybrid no longer costs anything to keep.

**How the resolution floor got measured — by accident.** A run reported
`Hybrid + RRF` at 0.925 and `+ cross-encoder` at 0.906. Those are the same
pipeline: the slim image has no `sentence-transformers`, so the rerank stage
degrades to a pass-through that is byte-identical to having it switched off
(`test_rerank_degradation.py` pins exactly that). Two identical configurations
disagreed on one question, `seattle-sick-day-count`, and agreed on the other
52.

Three things changed because of it:

- `scripts/ablation.py` now checks `reranker_available()` and **skips** the
  cross-encoder row rather than printing a pass-through as if it were a
  measurement, running the rows after it with reranking off, as deployed.
- Every table now carries a **control row** — the first configuration run again
  under a second name — so a reader can see how large a difference of zero is
  before reading anything into a small one.
- `--resume` and `--only`, because a 50-minute measurement should not be
  all-or-nothing. It died twice and lost everything the first time.

**Verified across two replicas.** `scripts/replica_check.py` addresses each API
pod directly, which is the one thing a Service exists to prevent, and compares
what each retrieves rather than what each answers — retrieval is what moved,
and generation would only add a model's sampling to the comparison:

```
replicas: 2
  count   agree   27
  sparse  agree   7 chunks, top=['seed-02-leave:1', 'seed-06-seattle:1']
  dense   agree   8 chunks, top=['seed-02-leave:1', 'seed-02-leave:3']
  hybrid  agree   8 chunks, top=['seed-02-leave:1', 'seed-06-seattle:1']
PASS
```

Byte-identical rankings from both pods for all three retrievers.


---

Back to the [README](../README.md).
