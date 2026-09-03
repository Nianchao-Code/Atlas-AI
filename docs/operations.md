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

**Cache-hit path**, no model call, one replica at 2 CPU:

| Concurrency | rps | p50 | p95 |
| --- | --- | --- | --- |
| 1 | 418 | 2.3ms | 2.7ms |
| 4 | 531 | 7.5ms | 9.0ms |
| 8 | 562 | 14.0ms | 16.6ms |
| **16** | **592** | 26.4ms | 31.7ms |
| 32 | 524 | 51.7ms | 96.6ms |
| 64 | 528 | 97.2ms | 237.3ms |

Throughput peaks near concurrency 16 and then flattens while latency grows
linearly — the saturation point, not a cliff. No errors at any level.

**Cold path**, model in the loop: 0.3 rps at concurrency 1 (p50 3.8s), 1.1 rps
at concurrency 4 (p50 3.4s). Latency per request is flat as concurrency rises,
so the service is holding requests rather than adding to them.

**Head-of-line blocking**, which is the number moving synchronous work off the
event loop exists to protect. Cache hits at concurrency 8, measured quiet, then measured again while
one 5.2s model request is in flight:

| | p50 | p95 | over 100ms |
| --- | --- | --- | --- |
| quiet | 14.2ms | 15.8ms | 0 / 320 |
| during a 5.2s request | 14.6ms | 21.4ms | 0 / 2360 |

Not one of 2,360 concurrent requests crossed 100ms while a five-second request
was running. Synchronous work on the event loop would show up here as a cluster
of slow requests; a 1.36x p95 is ordinary contention.

**The rate limit was the binding constraint, and it was a guess.** At 60/min
the first run rate-limited 22 of 40 requests at concurrency 4, and all 40 at
concurrency 16 — a service capable of 592 rps was capped at 1. The limit exists
to bound spend rather than to protect the app, so it is now 300/min: far above
any interactive session, far below what a runaway client could burn.

