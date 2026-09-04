# What ten thousand documents changed

The corpus was eight documents for most of this project's life, and the
[retrieval ablation](retrieval-ablation.md) said so in its own results: recall
scored 1.000 for every retriever, because `_hit` needs one of the expected
documents and there were eight to choose from. A metric that cannot separate
anything is not measuring; it is decorating.

`scripts/generate_corpus.py` builds ten thousand distractors around the real
handbook and `scripts/bulk_index.py` loads them. What follows is what that
exposed — in the corpus, in the code, and in the method.

## The generator took three tries, and the first two produced findings

**v1 was unusable and looked fine.** It shaped every distractor after the real
documents on the theory that distractors should be hard, which produced 1,649
leave policies — 16.5% of the corpus — all opening `Annual leave - First year
of employment`, identical to the real one. Dense retrieval then returned 24 of
24 distractors for the leave questions, with top scores spanning 0.7942 to
0.7903. A 0.3% spread across documents that are the same sentence.

That reads as "dense retrieval collapses at scale". It means "there was nothing
to discriminate".

**The statistics check passed it.** Heaps β 0.588 and hapax 42.3%, both inside
the natural band, because vocabulary statistics measure words and say nothing
about how many documents are about the same thing in the same words. Necessary
and insufficient — so `corpus_stats.py` gained two checks that would have
caught it: the largest share any single opening line holds, and Jaccard overlap
against the real corpus.

**v2 fixed the topic mix and still flattered sparse retrieval to 53/53.** Every
distractor named a different fictional company, so 84 of the 102 terms the real
handbook uses twice or more — `kepler`, `seattle`, `mercer`, `stipend` —
appeared in no distractor at all. IDF made every golden question separable by
its proper nouns alone.

**v3 lets the distractors borrow the handbook's vocabulary**, the way a
company's own documents do: 41% name Kepler, 52% of its signal terms appear.
Vocabulary only, never phrasing, which is exactly what v1 got wrong.

| | v1 | v2 | v3 |
| --- | --- | --- | --- |
| Largest single topic | 16.5% | 6.1% | 6.0% |
| Documents naming the real company | 0% | 0% | **41%** |
| Real corpus signal terms present | — | 18% | **52%** |
| Max Jaccard overlap vs real | — | 0.105 | **0.127** |
| Dense: real document at rank 1 | 15/53 | 30/53 | 28/53 |
| Sparse: real document at rank 1 | 48/53 | **53/53** | 51/53 |

The sparse column is the tell. A retriever that scores perfectly is usually
being flattered, and it was: v3 removes the flattery and it drops to 51/53.
Dense stays degraded across all three (15/30/28), which is what makes that one
a property of retrieval rather than of a generator.

**One number is still out of band, and it stays.** Zipf slope −1.72 against
−0.86 measured on real prose with the same estimator. It was attributed rather
than guessed: splitting the documents into their structured policy skeleton and
their prose measured −1.82 and −1.33. The skeleton is the cause, and the
skeleton is the point — it is what makes a distractor collide with a handbook.
The practical effect is that template boilerplate is discounted harder than it
would be in real text; distinctive terms and identifiers are unaffected, which
is the half that matters here.

## Two bugs the load found, both mine

**A 17.5MB write against a 30-second timeout.** The loader flushed 500
documents at a time, which is a sensible batch for the embedding API and 2,587
points in a single Qdrant upsert. Measured: 15.9MB of vectors plus 1.6MB of
payloads in one request body. It timed out every time. The flush size had been
chosen for one side of the work and never checked against the other; upserts
are now batched at 256 points, 1.7MB per request. Not by raising the timeout —
a 30s timeout is reasonable and a 17.5MB request is not.

The retry wrapper had the same shape of blind spot: it wrapped `embed_texts`
and nothing else, so a transient *write* failure would have discarded vectors
already paid for. It wraps both sides now.

**The reconciler deleted 107 documents, correctly.** The loader wrote vectors
first and the catalogue second. The background reconciler ran inside that
window, saw points whose `doc_id` the catalogue did not list, classified them
as orphans — which is exactly what it is for — and removed them:

```
22:42:30 reconcile.repaired checked=8008
  orphans_deleted=['bulk-d8000', 'bulk-d8001', ... 107 documents]
```

Two features that are individually right, fighting. The fix is an ordering
decision rather than a lock: the loader now claims each record as `indexing`
*before* writing its vectors, so the transient state is "listed but not yet
retrievable" — which reconcile skips while it says `indexing`, and requeues if
the loader dies. **Order the writes so the window between them is the
recoverable state.**

It was caught only because the loader's own count (40,052 chunks) disagreed
with the collection's (39,635). Neither the loader nor the reconciler reported
anything wrong.

## What broke in the service

**`counts()` went from 0.5ms to 56ms**, and the background refresher calls it
every two seconds on every replica — 2.8% of a core, forever, recomputing a
number that only changes on a write. It loaded every record and summed them.

The chunk total is now maintained on write and the document count comes from
`HLEN`, both O(1). Ordering moved into a sorted set so a page of the catalogue
costs O(log n + page). Both are derived data that can drift, so the reconciler
— which already walks every record — re-derives them on each pass, which bounds
drift to one sweep.

The writes go through Lua, because `HSET` followed by `INCRBY` from the client
is two round trips with a window between them, and that window is where a
counter goes wrong.

**`/api/v1/documents` returned every record.** Fine at eight, a 10MB response
at ten thousand that a browser renders as an unusable list. It pages now, and
returns the true total separately, because the count the UI wants is one
number and not the length of an array.

## What is measured, and what is not

Measured on the loaded corpus, one embedding per question and no generation:

| | dense | sparse | hybrid |
| --- | --- | --- | --- |
| Real document at rank 1 | 28/53 | **51/53** | 34/53 |
| Real document anywhere in top-24 | 31/53 | **53/53** | **53/53** |
| Distractor share of retrieved chunks | 94.8% | 85.1% | 85.7% |

**Recall is a metric again.** It scored 1.000 for every retriever on eight
documents. Dense now finds no real document at all for 22 of 53 questions,
which is a number that can move when something changes.

**Dense degrades and sparse holds.** That reverses what the eight-document
ablation concluded — that dense earned its place and fusing sparse into it
bought nothing. The reversal survived all three generators (dense at 15, 30 and
28 of 53), which is what separates it from the artefacts the first two
produced.

**What has not been re-measured**: the full ablation, the eval gate, and
throughput, all of which need generation rather than retrieval alone. Until
those run, [the ablation's numbers](retrieval-ablation.md) describe an
eight-document corpus and say so. The retrieval numbers above are not a
substitute for them.

## Reproducing it

```bash
python scripts/generate_corpus.py --docs 10000 --out samples/distractors
python scripts/corpus_stats.py samples/distractors --against samples/corpus
python scripts/bulk_index.py --source samples/distractors --dry-run
python scripts/bulk_index.py --source samples/distractors
python scripts/distractor_probe.py
```

The corpus is deterministic from `--seed` and is not committed: 42MB of derived
files belongs in a generator, not in git. Loading it costs about $0.06 in
embeddings, reported by the loader itself rather than estimated.

---

Back to the [README](../README.md).
