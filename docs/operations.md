# Operations

Auth, metrics, caching and measured throughput. Split out of the
[README](../README.md), which was 39 minutes of reading.

## Auth

Every `/api/v1` route requires a key; `/health` stays open so probes work.

```bash
curl -H "X-API-Key: $ATLAS_API_KEY" localhost:8000/api/v1/metrics
curl -H "Authorization: Bearer $ATLAS_API_KEY" localhost:8000/api/v1/metrics
```

Keys are configured as `ATLAS_API_KEYS="principal:secret,principal:secret"`.
The principal is the unit of isolation, not just a label:

- **The semantic cache is keyed by it.** `qa:{principal}:{hash}`, and the
  near-duplicate list is per principal too. Without that, a cached answer built
  from one caller's documents would be served to the next caller asking the
  same question — the cache would quietly undo the access control.
- **Rate limits are charged to it.** A fixed window of
  `RATE_LIMIT_PER_MINUTE` per principal per minute, one Redis `INCR` per
  request. A fixed window permits a 2x burst across a boundary; a sliding
  window costs a sorted set and a read-modify-write per call.
- **Key comparison walks every candidate** with `secrets.compare_digest`
  rather than a dict lookup, which would short-circuit and leak key length and
  prefix through timing.

**Leaving `ATLAS_API_KEYS` empty disables auth** and makes every caller the
`dev` principal. Quick start and CI run that way on purpose, and `/health`
reports `"auth": false` so it is never a silent default.

**The browser never holds the key.** nginx injects it server-side when
proxying `/api/`, so the SPA calls a same-origin path with no credential in
its bundle. `scripts/k8s-deploy.ps1` generates the key on first deploy and
leaves it alone afterwards — rotating on every deploy would invalidate it for
no reason. Pass `-RotateApiKey` to replace it:

```powershell
kubectl get secret atlas-auth -n atlas -o jsonpath='{.data.ATLAS_FRONTEND_KEY}' | base64 -d
```

## Observability

Two endpoints, easy to confuse:

| Path | Auth | Purpose |
| --- | --- | --- |
| `/metrics` | none | Prometheus scrape, in-cluster only |
| `/api/v1/metrics` | API key | JSON snapshot for this replica, drives the SLI tab |

`/metrics` is unauthenticated on purpose: nginx does not proxy it, so it is
reachable only from inside the cluster, where the scraper lives and where a key
would be one more secret to distribute for nothing.

**Both processes are scrape targets.** The worker serves no HTTP of its own, so
it runs a listener on `:9100` purely to be scraped — without it, index
throughput and failures are invisible. Discovery is by pod annotation rather
than a `ServiceMonitor`, since that needs the Prometheus Operator. Aggregation
is Prometheus's job; each process only reports itself. An ingest bumps
`atlas_index_jobs_total` on the worker and leaves the API's copy at zero, which
is exactly the split that made a single in-process counter the wrong shape.

What is exported, and the reasoning behind the shape:

- **`atlas_queries_total{outcome}`** — `answered`, `abstained`, `cached`,
  `blocked`. A cache hit is its own outcome; folding it into `answered` hides
  the hit rate inside the query rate, which is the number worth watching when
  the cache changes.
- **`atlas_retrieval_seconds`** — a cache hit records nothing here. It never
  ran retrieval, and a near-zero sample would drag p95 down and hide real work.
- **`atlas_faithfulness_score`** — bucketed at 0.7, because that is the serving
  gate. The bucket below it is the regenerate-or-abstain rate.
- **`atlas_reconcile_actions_total{action}`** — `requeued`, `marked_failed`,
  `orphans_deleted`. A steady zero is the expected reading; anything else means
  the two stores drifted and says which way.
- **`atlas_upload_rejections_total{reason}`** — the HTTP status, not the
  filename: `400` is a traversal attempt, `413` an oversized body, `415` a type
  the parser cannot read. Labelling the filename would put attacker-controlled
  text into the metric namespace.
- **`atlas_rate_limited_total{principal}`** — the only series carrying an
  identity, because knowing who is being throttled is the entire point and the
  label space is bounded by the configured keys. Nothing else is labelled by
  principal: an identity in a label outlives the request in dashboards and long
  term storage, for a breakdown nobody reads.

No series carries question or answer text, and a test enforces it.

## Caching

Two layers, and the second one had to be measured before it was worth keeping.

**Exact match** is a Redis hash lookup on the normalised question: 4ms against
6.7s for the cold path.

**Paraphrase match** is an ANN query against a second Qdrant collection,
filtered to the asking principal. A reworded question returns in 247ms instead
of 6.7s.

It was rewritten for two reasons, and only one of them was speed.

**It had never served a hit.** The stored vector was the embedding of the
rewritten query plus its HyDE paragraph, while lookups embed the raw question.
Those are different texts: the *same question asked twice* scored 0.817 against
its own cache entry, under a 0.92 threshold. Every hit the system had ever
recorded came from the exact-match path. The scan being slow was the less
interesting half — it was scanning for something it could not match.

**The threshold was guessed.** `scripts/calibrate_cache.py` measures two sets
of pairs: the same question reworded, and questions one decisive word apart.

| Threshold | Paraphrases caught | Wrong answers served |
| --- | --- | --- |
| 0.80 | 6/9 | 0 |
| **0.82** | **6/9** | **0** |
| 0.92 (previous) | 2/9 | 0 |

The worst false pair is *"Is Seattle sick leave also capped at 10 days?"*
against *"How many paid sick days does a Seattle employee receive?"* at 0.768 —
nearly identical wording, opposite answers. 0.82 clears it by five points.
Verified end to end: those two still do not fuse.

The old implementation also walked up to 200 Redis entries with two round trips
each and scored them in Python on every miss. The ANN query replaces that and
drops the 200-entry window, so what counts as near is decided by the index
rather than by a recency list.

## Throughput

`scripts/loadtest.py` drives one replica from inside the cluster. It runs
there because `kubectl port-forward` serialises connections and would measure
itself.

End-to-end throughput with a model in the path measures the model provider, so
the phases separate what this service contributes from what it waits on.

Everything here was measured at 27 chunks and again at
[40,079 chunks](scaling-the-corpus.md), and then — because the second run
appeared to show a 10% drop — five more times, to find out what a 10% number
is worth here. It is worth nothing. **The run-to-run spread on an unchanged
system is 9.7%.**

**The noise floor.** Five consecutive runs of the same phase against the same
corpus on the same replica, nothing changed between them
(`python scripts/loadtest.py --phase noise --repeats 5`, run in the cluster —
the rate limit has to be lifted first or the benchmark measures the throttle):

| Concurrency | min | max | mean | spread | as % of mean |
| --- | --- | --- | --- | --- | --- |
| 1 | 339.9 | 370.9 | 349.8 | 31.0 | 8.9% |
| 4 | 425.2 | 470.7 | 449.1 | 45.5 | 10.1% |
| 8 | 487.3 | 535.8 | 511.1 | 48.5 | 9.5% |
| 16 | 485.5 | 535.3 | 512.3 | 49.8 | 9.7% |
| 32 | 348.4 | 498.4 | 457.2 | 150.0 | **32.8%** |

Every level below saturation moves by about a tenth of itself between
identical runs, and concurrency 32 moves by a third — one of the five collapsed
to 348 rps with thirty requests over 100ms while its neighbours held near 490.
The concurrency that peaks is not stable either: it landed on 16 three times
and on 8 twice.

**Cache-hit path**, no model call, one replica at 2 CPU. p50 and p95 are from
one 40,079-chunk run; the rps column is that same single run, which is exactly
the kind of number the table above says to distrust:

| Concurrency | rps @ 27 (n=1) | rps @ 40,079 (n=1) | p50 | p95 |
| --- | --- | --- | --- | --- |
| 1 | 418 | 374 | 2.6ms | 2.9ms |
| 2 | — | 416 | 4.8ms | 5.3ms |
| 4 | 531 | 473 | 8.3ms | 10.2ms |
| 8 | 562 | 496 | 15.7ms | 19.3ms |
| **16** | **592** | **534** | 29.2ms | 36.6ms |
| 32 | 524 | 500 | 55.9ms | 95.2ms |
| 64 | 528 | 399 | 124.7ms | 139.5ms |

**The 10% drop this page used to report was the noise floor.** 534 sits inside
the 485–535 band five repeats produced on the current corpus; it is a
high-ish draw, not a lower ceiling. The honest reading of the second column is
"the current corpus sustains roughly 510 rps at its peak, ±10%".

What survives, and what does not:

- **Survives:** the shape. Throughput rises to a saturation region around
  concurrency 8–16 and then flattens while latency grows linearly, with no
  errors at any level in any run. That held in all seven runs.
- **Does not survive:** any claim built on comparing two single runs. The old
  592 does sit above the best of the five current draws, by 10.6% over the
  maximum — so a real difference is not excluded. But 592 was itself n=1 and
  its own noise floor was never measured, so nothing here establishes one.

The prediction this page used to make — that the cache-hit ceiling would not
move, because a cache hit is answered from Redis without touching Qdrant — is
therefore neither confirmed nor refuted. It is untested, and the instrument
was too coarse to test it. Settling it needs repeats on both corpora, not a
better explanation.

**Cold path**, model in the loop: 1.0 rps at concurrency 4, p50 3.5s, p95 4.6s.
At 27 chunks it was 1.1 rps at the same concurrency with p50 3.4s — unchanged
within noise, which is what a model-bound path should do.

**Retrieval latency did move, and this is where it shows.** The eval gate
measures p95 retrieval at **600.71ms** across its 53 questions at 40,079
chunks. That is the HNSW search and the sparse index growing with the corpus,
and it is separate from the cache-hit path above.

**Head-of-line blocking**, which is the number moving synchronous work off the
event loop exists to protect. Cache hits at concurrency 8, measured quiet, then
measured again while one cold model request is in flight:

| | p50 | p95 | p99 | over 100ms |
| --- | --- | --- | --- | --- |
| quiet | 18.2ms | 21.1ms | 21.9ms | 0 / 320 |
| during a 5.6s request | 16.1ms | 21.1ms | 73.8ms | 15 / 1560 |

**This one compares quiet against busy inside a single run**, which is what
makes it worth more than the table above it: both halves saw the same host, the
same minute and the same replica, so the ~10% run-to-run drift cancels instead
of accumulating. p95 is identical under load — 1.00x — so the property holds
at the percentile describing the common case, and p99 rising 21.9ms to 73.8ms
is the cost, paid by a handful of requests rather than by all of them.
Synchronous work on the event loop would show up here as every request slowing
together, not fifteen of 1,560.

The cross-run half of this — 1.00x here against 1.36x at 27 chunks, fifteen
over 100ms against none of 2,360 — is the comparison the noise floor says not
to lean on. Both sides are n=1. Reported because it was measured, not because
it shows anything.

**The rate limit was the binding constraint, and it was a guess.** At 60/min
the first run rate-limited 22 of 40 requests at concurrency 4, and all 40 at
concurrency 16 — a service capable of 592 rps was capped at 1. The limit exists
to bound spend rather than to protect the app, so it is now 300/min: far above
any interactive session, far below what a runaway client could burn.

300/min then ate the re-run. The first attempt at 40,079 chunks returned 429
for every request from concurrency 32 onward, and for all 328 requests of the
head-of-line and cold phases behind it — a load test measuring the throttle
instead of the service. The numbers above come from a second run with
`RATE_LIMIT_PER_MINUTE` temporarily raised, restored to 300 afterwards. Worth
writing down twice: a spend guard sized for humans is not sized for the
benchmark, and a benchmark that silently measures the guard reports whatever
the guard does.

