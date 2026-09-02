# Atlas AI

Production RAG platform for internal handbook Q&A. Hybrid retrieval, a corrective LangGraph pipeline, faithfulness gating, and an offline eval harness — built so every component maps to a real failure mode, not a buzzword checklist.

**Demo flow:** seed corpus → ask a question → inspect citations and graph trace → run the golden-set eval.

![Atlas answering a handbook question, with the graph trace and cited sources filling in as the pipeline runs](docs/demo.gif)

One question against the deployed stack, uncut: guard, a cache miss, query
rewrite, hybrid retrieval (recorded before fusion moved into Qdrant, so the
trace reads `dense=24 bm25=12 rrf=24` where it now reads
`dense+sparse rrf=24`), the rerank node cutting
24 candidates to 12, LLM grading, parent-passage packing, a streamed answer,
and the faithfulness gate scoring it 1.00 — 3.9s end to end. The cross-encoder
is off in this deployment, so that step is a truncation rather than a rerank;
the [ablation](#what-each-stage-actually-buys) is why. Recorded by
`scripts/capture_demo.py`, which drives the real UI.

## Highlights

| Area | Implementation |
| --- | --- |
| **Retrieval** | Parent-child chunking, dense + sparse vectors in one Qdrant collection fused server-side by RRF, cross-encoder rerank, LLM document grading |
| **Generation** | SSE streaming answers; token-budget packing on deduplicated parent passages |
| **Orchestration** | LangGraph: guard → cache → rewrite → retrieve → rerank → grade → compress → generate → faithfulness |
| **Quality** | 53-question golden set tagged by failure mode — recall, precision, faithfulness, correctness, hallucination rate, abstention accuracy, plus a stage-by-stage ablation with a control row; runs as a background job with progress, single-flighted so two clicks are one run |
| **Safety** | API-key auth with per-principal cache isolation and rate limiting; injection resistance measured across 17 attacks and four defense configurations, not asserted; upload path hardened against traversal, oversized bodies and unreadable types |
| **Ops** | 521MB image running as a non-root uid; horizontally scalable — no retrieval state in process memory; load-tested to a measured ceiling (592 rps cached, no head-of-line blocking); Prometheus metrics from every process; retrieval p50/p95 separate from end-to-end latency; Redis embedding + semantic QA cache |
| **Ingest** | Async worker queue (Redis Streams locally, Kafka/Redpanda in production); uploads streamed to disk under a byte budget, parsing off the event loop; catalogue and vector store reconciled at startup and on demand |

## What the measurements said

Every claim above has a number behind it, and several of those numbers are
unflattering. Those are the ones worth reading first.

- **[Hybrid retrieval and the cross-encoder buy nothing on this corpus.](#what-each-stage-actually-buys)**
  Fusing sparse into dense matches dense on correctness and costs context
  precision. The cross-encoder moves correctness by zero and adds latency, so
  it is off in the deployed config. Query rewrite looked worth +3.5pp on 14
  questions and worth exactly nothing on 53 — the clearest argument here for
  sizing an eval set before trusting it.

- **[A whole subsystem existed to coordinate state that did not need to
  exist.](#the-sparse-index-moved-into-qdrant)** Sparse retrieval was an
  in-process index rebuilt by every replica, kept in step by a revision counter,
  a poller, a rebuild thread and an atomically swapped snapshot. Moving the
  vectors into Qdrant deleted all of it — and retrieved *better* than the
  library it replaced, sparse-only correctness 0.863 → 0.906.

- **[The harness could not resolve the differences it was being read
  for.](#the-sparse-index-moved-into-qdrant)** Two configurations that were the
  same pipeline reported 0.925 and 0.906. That put a number on the noise floor,
  1.9pp — one question of 53 — and retracted an earlier finding that was exactly
  1.9pp. Every table now carries a control row so the floor is always visible.

- **[The paraphrase cache had never served a hit.](#caching)** It stored the
  embedding of the rewritten query and looked up with the raw question, so the
  same question asked twice scored 0.817 against its own entry under a 0.92
  threshold. The slow linear scan everyone would have optimised first was
  scanning for something it could not match.

- **[The injection guard contributes nothing measurable.](#injection-resistance)**
  17 attacks across four defense configurations: nothing leaks, including with
  both defenses disabled. The regex catches 4 of 4 literal phrasings and 0 of 7
  rewordings of the same intent. What keeps the system safe is the model's
  instruction following; the guard buys a cheap deterministic refusal.

- **[The image was 94% dependencies for a disabled feature.](#what-each-stage-actually-buys)**
  `sentence-transformers` pulled in torch, triton and 2.7GB of CUDA wheels --
  into a CPU-only container, for a reranker the ablation had already measured
  at zero. Making it an optional extra took the image from 8.69GB to 521MB.

- **[No head-of-line blocking under load.](#throughput)** Not one of 2,360
  concurrent requests crossed 100ms while a 5.2-second request was in flight —
  the evidence that moving the sparse rebuild off the event loop did what it
  claimed. That rebuild has since been deleted outright.

- **[The upload endpoint let a caller write to the application's own source.](#the-ingest-path)**
  `filename` came from the client and was joined onto the upload directory
  unchecked, so `../../../../app/main.py` resolved to `/app/app/main.py` —
  writable, because the non-root uid owns `/app`. Verified against the running
  pod. Every claim on this page was about the query path; nobody had pointed
  one at the half of the service that takes files.

- **[A known gap is not a handled one.](#keeping-redis-and-qdrant-in-agreement)**
  "Redis and Qdrant can diverge" sat in Limits for weeks. Then a collection
  rebuild dropped every vector while the catalogue kept listing eight ready
  documents: the UI showed a healthy corpus and every question abstained.
  Reconciliation now runs at startup and on demand, and a probe verifies it by
  causing both kinds of divergence on purpose.

- **[A raised timeout was a symptom being configured around.](#the-eval-endpoint-became-a-job)**
  The eval ran inside its own HTTP request, so nginx needed
  `proxy_read_timeout 600s`. The three problems that setting hid were worse
  than the one it solved: a closed tab discarded four minutes of paid model
  calls, two clicks billed for two runs, and a spinner could not be told from a
  hang. It is a job now, `202` in 14ms, and the timeouts came back down to 120s.

- **Three features shipped silently inert, and none of them failed.** API auth
  was disabled because pydantic-settings bound the field to `API_KEYS` while
  the deployment set `ATLAS_API_KEYS`. The paraphrase cache never matched. The
  CI eval gate skipped every run because a step's own `env` is not in scope for
  that step's `if`. All three looked configured, all three were green, and all
  three were found by testing the deployed system rather than the code.

## Architecture

```mermaid
flowchart TD
    U["Upload or seed"] --> Q[("Redis Streams<br/>Kafka in production")]
    Q --> W["Worker: chunk, embed, index"]
    W --> QD[("Qdrant chunks<br/>dense + sparse vectors")]

    A(["Question"]) --> G{"guard"}
    G -- "injection pattern" --> X["refuse"]
    G --> C{"cache"}
    C -- "exact or paraphrase hit" --> DONE(["answer + citations"])
    C -- "miss" --> RW["rewrite + HyDE"]
    RW --> RT["retrieve: dense + sparse,<br/>fused by RRF inside Qdrant"]
    QD -.-> RT
    RT --> RR["rerank"]
    RR --> GR{"graded sufficient?"}
    GR -- "no, retries left" --> RW
    GR -- "no, retries spent" --> AB["abstain"]
    GR -- "yes" --> CP["dedupe parents<br/>pack to token budget"]
    CP --> GEN["generate, streamed"]
    GEN --> F{"faithfulness at least 0.7?"}
    F -- "no, first try" --> GEN
    F -- "no, again" --> AB
    F -- "yes" --> DONE
```

The dashed edges are the parts that are easy to miss: the worker indexes in its
own process and only signals the API through a Redis counter, and the API
rebuilds its sparse index on a background tick rather than inside a request.
[Throughput](#throughput) is where that choice gets measured.

## Tech stack

**Backend:** FastAPI · LangGraph · Qdrant · Redis · OpenAI-compatible LLM API  
**Frontend:** React · Vite  
**Infra:** Docker Compose · Kubernetes manifests (`infra/k8s/`) · optional Redpanda (Kafka protocol)

## Design decisions

- **Parent-child chunks** — children for search, parents for generation. Fewer, coherent context blocks instead of stuffing raw top-k snippets.
- **Hybrid + RRF + cross-encoder** — dense and sparse vectors sit in one Qdrant collection and are fused there by RRF; a small cross-encoder reranks the top candidates before the LLM grader.
- **Corrective graph, not multi-agent** — one stateful graph that rewrites bad retrieval and abstains on thin evidence.
- **Faithfulness as a serving gate** — scores below 0.7 trigger one regeneration, then abstain. Hallucination control is enforced, not only measured offline.
- **Eval set is part of the product** — graph and chunking changes should pass the golden set before shipping.
- **Single vector store** — Qdrant holds the dense vectors, the sparse ones and the payload filters, and computes IDF across the collection. Handbook Q&A does not need GraphRAG, and the sparse side does not need a second index to keep in step with the first.

These are design intentions. [What each stage actually buys](#what-each-stage-actually-buys)
measures them, and on the current corpus it does not vindicate all of them —
hybrid retrieval and the cross-encoder show no benefit at this scale, and the
corrective loop trades recall for precision and tokens.

## Quick start

Requires Python 3.11+, Node 20+, Docker (Redis + Qdrant).

```bash
git clone https://github.com/Nianchao-Code/Atlas-AI.git
cd Atlas-AI
cp .env.example .env
# Set OPENAI_API_KEY. Compatible gateways: set OPENAI_BASE_URL.

docker compose up -d redis qdrant

cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
uvicorn app.main:app --reload --port 8000
```

Second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open http://127.0.0.1:5173 → **Corpus → Load Kepler sample handbook** → **Ask** → **Eval**.

Without an API key the system still indexes and retrieves; generation falls back to extracts. LLM-as-judge metrics degrade to keyword overlap.

Full stack (API + worker + frontend):

```bash
docker compose up --build
```

Kafka-protocol bus (optional):

```bash
docker compose --profile kafka up -d
# Set KAFKA_BROKERS=localhost:19092 in .env
```

## API

Every `/api/v1` route requires an API key (see [Auth](#auth)); `/health` does not.

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Liveness, plus whether auth and an LLM are configured |
| `GET` | `/metrics` | Prometheus scrape (no key; not proxied to the browser) |
| `POST` | `/api/v1/query` | Run the RAG graph (non-streaming) |
| `POST` | `/api/v1/query/stream` | Same graph, streams answer tokens via SSE |
| `POST` | `/api/v1/eval` | Start a golden-set run; returns `202` and a job id ([why](#the-eval-endpoint-became-a-job)) |
| `GET` | `/api/v1/eval/{job_id}` | Progress, then the report |
| `GET` | `/api/v1/eval` | The most recent run |
| `GET` | `/api/v1/metrics` | SLI snapshot (latency, cache, tokens) |
| `POST` | `/api/v1/documents/seed` | Load sample handbook |
| `POST` | `/api/v1/documents/reconcile` | Repair catalogue/vector disagreement; `?dry_run=true` reports only ([why](#keeping-redis-and-qdrant-in-agreement)) |
| `POST` | `/api/v1/documents` | Upload md / txt / pdf — `415` on other types, `413` over `MAX_UPLOAD_BYTES` (20MiB), `400` if the filename cannot be made safe ([why](#the-ingest-path)) |
| `DELETE` | `/api/v1/documents/{id}` | Remove a document |

## Project layout

```
backend/app/     FastAPI, LangGraph pipeline, retrieval, eval
frontend/        Ask · Corpus · Eval · SLI dashboards
samples/corpus   Kepler internal handbook (includes injection test doc)
samples/eval     Golden question set (53 cases, tagged by category)
docs/            Demo recording used by this README
scripts/         Deploy, eval gate, and the harnesses behind the numbers above
infra/k8s/       Kubernetes manifests (Redis, Qdrant, API, worker, frontend)
```

## Kubernetes deploy

Requires a local cluster (Docker Desktop Kubernetes or minikube) and `kubectl` context configured.

```powershell
# Enable Kubernetes in Docker Desktop, then:
cd "D:\Atlas AI"
$env:OPENAI_API_KEY = "sk-..."
.\scripts\k8s-deploy.ps1
kubectl port-forward svc/frontend 8080:80 -n atlas
# Open http://127.0.0.1:8080
```

## Eval CI gate

```bash
# Smoke (no API key; used in CI)
python scripts/eval_gate.py --smoke

# Full regression (requires OPENAI_API_KEY)
python scripts/eval_gate.py --full
```

Thresholds live in `samples/eval/thresholds.json` and are set below the worst
of two measured runs, not chosen by feel. The previous
`min_abstention_accuracy: 1.0` was decided by the single abstention question
the set had at the time; there are now seven.

## What each stage actually buys

`scripts/ablation.py` runs the golden set through progressively richer
pipelines. Each row adds exactly one stage, so a delta belongs to that stage
and nothing else.

**Retrieval, re-measured on the Qdrant sparse index:**

| Configuration | Recall | Ctx precision | Faithful | Correct | Halluc. | p95 ms | Tokens |
|---|---|---|---|---|---|---|---|
| **Dense only** | 1.000 ±0.000 | **0.502** ±0.002 | 0.981 ±0.000 | **0.925** ±0.000 | 0.000 ±0.000 | 674 ±485 | 1022 ±9 |
| **Sparse only** | 1.000 ±0.000 | 0.425 ±0.002 | 0.981 ±0.000 | 0.906 ±0.000 | 0.000 ±0.000 | 617 ±334 | 1036 ±0 |
| **Hybrid + RRF** | 1.000 ±0.000 | 0.456 ±0.004 | 0.979 ±0.003 | **0.925** ±0.000 | 0.000 ±0.000 | 434 ±54 | 1075 ±12 |

**Downstream stages, from the previous implementation:**

| Configuration | Recall | Ctx precision | Faithful | Correct | Halluc. | p95 ms | Tokens |
|---|---|---|---|---|---|---|---|
| **+ cross-encoder** | 1.000 ±0.000 | 0.428 ±0.000 | 0.977 ±0.000 | 0.906 ±0.000 | 0.000 | 660 ±316 | 985 ±8 |
| **+ LLM grading (full)** | 0.961 ±0.000 | **0.830** ±0.003 | 0.975 ±0.004 | 0.915 ±0.013 | 0.000 | 1024 ±336 | **232** ±5 |
| **Full − query rewrite** | 0.961 ±0.000 | 0.830 ±0.003 | 0.975 ±0.001 | 0.915 ±0.013 | 0.000 | 1099 ±1159 | 232 ±5 |

These three are carried over rather than re-run: the account the harness bills
ran out of credits partway through the new run, and a stale number that says so
beats a fresh number that is not there. The one partial re-run that did land
before it stopped put the full pipeline at recall 0.961 / correctness 0.943,
which is the same shape. They are due for re-measurement.

53 questions, 2 runs per configuration, mean ±sd. Judge `gpt-4o-mini`,
generation `gpt-4o`. Reproduce with `python scripts/ablation.py --repeats 2`.
Note the ±sd is spread across repeats of one configuration and understates the
harness's resolution: two *identical* configurations differed by 1.9pp, which
is [where that number comes from](#the-sparse-index-moved-into-qdrant) and why
the tables now carry a control row.

Aggregates hide where the differences live, so the same runs by question type
(answer correctness, one run). The sparse column is the previous
implementation; the categories, not the column, are the point:

| Category | n | Dense | Sparse | Full | What it stresses |
|---|---|---|---|---|---|
| `semantic` | 4 | **1.000** | **0.500** | 1.000 | paraphrased away from corpus wording |
| `abstain` | 6 | 0.333 | 0.167 | **0.667** | document retrievable, fact absent from it |
| `policy` | 9 | 1.000 | 1.000 | **0.778** | the answer is a rule, not a figure |
| `distractor` | 8 | 1.000 | 1.000 | **0.875** | the same figure appears in another document |
| `deferral` | 2 | 1.000 | 0.750 | 1.000 | corpus says where the answer lives, not what it is |
| `lexical` | 5 | 1.000 | 1.000 | 1.000 | exact identifiers (`KV-2025-441`, `IR-4`) |
| `multi-hop` | 5 | 1.000 | 1.000 | 1.000 | answer spans two documents |
| `lookup` | 10 | 1.000 | 1.000 | 1.000 | single stated figure |
| `injection` | 3 | 1.000 | 1.000 | 1.000 | user and corpus prompt injection |

**What the measurements support:**

- **Dense retrieval earns its place, on the strength of one category.** It ties
  sparse everywhere except `semantic`, where questions are deliberately worded
  away from the corpus vocabulary and sparse scores 0.500 against dense 1.000.
- **Fusing sparse into dense buys precision, not correctness.** Hybrid matches
  dense on correctness (0.925 both) and gives up context precision to do it
  (0.502 → 0.456): the lexical hits displace passages the dense ranker had
  already ordered correctly. An earlier version of this section said hybrid was
  *worse* on correctness, 0.925 → 0.906. That gap is 1.9pp, which is one
  question of 53 and exactly the harness's
  [measured resolution floor](#the-sparse-index-moved-into-qdrant). It was
  over-read, and it is retracted.
- **The cross-encoder does nothing here.** Identical correctness to plain
  hybrid, marginally worse precision, and it adds latency. On this corpus it is
  pure cost, so it is off by default and its dependency is an optional extra:
  `sentence-transformers` was dragging torch, triton and 2.7GB of CUDA wheels
  into a CPU-only image for it. Removing that from the default build took the
  image from **8.69GB to 521MB**, and the layer the cluster stores from 2.99GB
  to 117MB, and CI's backend job from 2m19s to 44s. Install `.[rerank]` to turn it back on, ideally against the CPU
  torch index.
- **Grading is the stage that pays.** Context precision 0.43 → 0.83, prompt
  tokens 990 → 232 (−77%), and abstention correctness doubles (0.333 → 0.667).
- **Grading also over-abstains.** It is the only reason `policy` falls to 0.778
  and `distractor` to 0.875: on `pii-to-llm` and `personal-claude-account` the
  corpus states the rule plainly, dense-only answers correctly, and the full
  pipeline refuses. That is a real regression, and it is why the corrective
  loop is a trade rather than a free win.
- **Query rewrite has no measurable effect.** Full and full−rewrite agree on
  every metric to three decimals. An earlier 14-question run credited rewrite
  with +3.5pp correctness; that was noise, and it is the clearest argument in
  this repo for sizing an eval set before trusting it.

**What the measurements do not support:** the `lexical` category was built to
show lexical search winning on exact identifiers and it did not — dense retrieves
`KV-2025-441` and `IR-4` perfectly well at this corpus size. Recall is also
saturated at 1.000 for every retriever, because `_hit` only requires one of the
expected documents and there are only eight to choose from. Recall cannot
discriminate anything here regardless of how the questions are written; that is
a property of the metric and the corpus size, not of the question set.

Six of 53 cases fail under the full pipeline. They are kept deliberately: four
are the over-abstention and cross-document-conflation bugs named above, and the
set exists to keep catching them.

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

## Injection resistance

`scripts/injection_probe.py` runs 17 attacks against four configurations, three
times each: both defenses off, the user-input guard alone, chunk sanitising
alone, and production. It exists because the previous evidence was three golden
cases that all passed, which is the same shape of evidence that turned out to
mean nothing for retrieval recall.

**Nothing leaks in any configuration** — 17 attacks, 3 runs, all four setups.
That includes seven rewordings that walk straight past the regex guard, and the
indirect payloads planted in `08-injection-bait.md`.

**The regex guard's marginal contribution is zero.** It catches 4 of 4 literal
phrasings and 0 of 7 rewordings of the same intent — `Forget everything you
were told earlier`, Spanish, letters spaced apart. Every attack it blocks, the
model refuses anyway with the guard disabled. What it buys is a deterministic
refusal without an LLM call, which is worth having and is not the same as
being what keeps the system safe.

| | Literal phrasings | Same intent, reworded |
| --- | --- | --- |
| Caught by `scan_user` | 4/4 | 0/7 |
| Leaked with the guard off | 0/4 | 0/7 |

**One real defect, found and fixed.** `01-company.md` says the K-Walk 2
endurance target must be answered as unpublished. Only the blunt question
obeyed that. Claiming to be on the internal planning team returned the figure
3/3, and so did asking whether it is more or less than 12 hours and by how
much — the model relayed the rule and the number together, because they sit in
the same passage. A withholding instruction in the system prompt closes all
three, verified 8 leaks to 0.

**Measuring this was harder than passing it.** Substring matching produced two
different false positives before the numbers meant anything:

- `90 days` matched a correct answer, because `03-data-policy.md` really does
  retain logs for 90 days.
- `90 days of annual leave` then matched *"the real policy does **not** grant
  90 days of annual leave"* — the model identifying the bait and negating it.

A detector that cannot tell assertion from negation reports a working defense
as a breach. The probe now uses substring matching only as a cheap pre-filter
and has an LLM adjudicate whether the answer asserted or complied. Even that
judge disagreed with itself once on two near-identical negations, which is why
the remaining 1/3 in the raw output is reported as judge variance rather than a
finding.

## The ingest path

Every measurement above is about answering questions. Nothing had been pointed
at the other half of the service — the one endpoint where a caller supplies a
filesystem path.

**An upload could write anywhere the container user could write.** `filename`
arrives from the client and was joined straight onto the upload directory:

```python
dest = Path(settings.upload_dir) / f"{doc_id}_{filename}"
dest.write_bytes(await file.read())
```

Verified against the running pod: `filename = "../../../../app/main.py"`
resolves to `/app/app/main.py`. The image runs as a non-root uid — but that uid
owns `/app`, because the Dockerfile chowns it so the app can run there. The
hardening that stopped an upload from reaching `/etc` did nothing to stop it
reaching the application's own source, and the next restart would have executed
whatever the upload contained.

The `doc_id` prefix is what made this non-obvious. `abc123_../../x` splits into
`abc123_..`, `..`, `x` — no separator before the first `..`, so the prefix
absorbs one level and a shallow probe lands back inside the directory looking
harmless. Escaping needs three.

Fixed in `app/uploads.py`: the filename is reduced to a single path component,
and the resolved path is then checked for containment. The second check is
redundant against the first today. It is there because it is the half that
keeps holding if someone later relaxes the sanitiser — one asserts a property
of the input, the other of the result. Backslashes are folded by hand, because
on Linux they are ordinary characters: `Path("..\..\x.md").name` returns the
whole string unchanged, and a POSIX-only defence waves Windows-style payloads
through.

**Two more things the endpoint accepted.** The body was read fully into memory
before anything could object to its size, so a request larger than the pod's
2Gi limit was an OOM kill rather than a 413; it now streams to disk under a
20MiB budget and unlinks the partial file on refusal. And any suffix was
accepted and then indexed as replacement characters — an `.exe` became
retrievable chunks of mojibake. Unreadable types are refused at the door.

Every row below was a `200` before this change, and every "now" column is the
deployed behaviour, not the intended one:

| `filename` sent | before | now |
| --- | --- | --- |
| `../../../../app/main.py` | written to `/app/app/main.py` | `415` |
| `../../ESCAPED.md` | written outside `uploads/` | stored as `<id>_ESCAPED.md` |
| `..\..\..\evil.md` | backslashes kept in the name | stored as `<id>_evil.md` |
| `payload.exe` | indexed as mojibake | `415` |
| 21MB body | read into memory | `413` |
| `Employee Handbook v2.md` | accepted | accepted, indexed, answerable |

Readable types are sanitised rather than rejected: someone who names a file
oddly should get their document, not an error. The catalogue and the citations
keep the name they used; only the path on disk is rewritten.

**The same event-loop rule the query path already follows.** `delete_doc`
scrolled the whole corpus and rebuilt the sparse index inline, and uploads
parsed PDFs inline — synchronous O(corpus) work on the event loop, which is
exactly what the [background refresher](#throughput) existed to prevent, left
behind on the two endpoints that were never load-tested.

Honest about the size of it: at that corpus the stall was **4.6ms** (4.1ms
scroll, 0.4ms rebuild), and the rebuild alone 13.7ms at 1080 chunks. It is
linear in the corpus, and it is the same work that put 19s into a p95 when it
ran cold on the query path — but at 27 chunks it was never going to show up in
a load test. The reason to fix it was that the rule should hold on every path,
not that this instance was expensive.

A piece of dead work went with it: `Indexer` rebuilt a sparse index that
nothing in its own process ever searched. Both of these are now moot for a
better reason — [the sparse index moved into Qdrant](#the-sparse-index-moved-into-qdrant),
so there is no per-process index left to rebuild anywhere. Parsing still runs
in a thread.

## The sparse index moved into Qdrant

[Limits](#limits) used to open with "runs as a single replica," and one
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

**The replacement retrieves better than the library it replaced.** Same corpus,
same golden set, same generation and judge models:

| Retriever | Ctx precision | Correctness |
| --- | --- | --- |
| Dense only | 0.508 → 0.502 | 0.925 → 0.925 |
| Sparse only | 0.403 → **0.425** | 0.863 → **0.906** |
| Hybrid + RRF | 0.434 → **0.456** | 0.906 → **0.925** |

Dense is unchanged, which is the control: nothing about the dense path moved.
Sparse-only correctness gains 4.3pp — about two and a half questions of 53 —
and context precision gains for both sparse and hybrid, with a run-to-run sd of
±0.004. One likely reason, not measured: the old implementation dropped hits
scoring zero, so it often returned fewer than `retrieve_k` candidates, while
the Qdrant query returns a full ranking.

**The hybrid result also retracts an earlier claim.** This README used to say
hybrid retrieval was worse than dense — "fusing BM25 into dense *lowers*
correctness (0.925 → 0.906)". That gap is 1.9pp, which is one question of 53,
and the run that produced these numbers also showed that 1.9pp is the smallest
difference the harness can resolve. So the old claim was over-read. What the
measurements support now is narrower and duller: **hybrid and dense are equal
on correctness, and hybrid has lower context precision.** Dense alone is still
what the deployment would choose on the evidence; hybrid no longer costs
anything to keep.

**How the resolution floor got measured — by accident.** The run reported
`Hybrid + RRF` at 0.925 and `+ cross-encoder` at 0.906. Those are the same
pipeline: the slim image has no `sentence-transformers`, so the rerank stage
degrades to a pass-through that is byte-identical to having it switched off
(`test_rerank_degradation.py` pins exactly that). Two identical configurations
disagreed on one question, `seattle-sick-day-count`, and agreed on the other
52. Within a configuration, repeats agreed on all 53.

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
```

## Keeping Redis and Qdrant in agreement

The catalogue lives in Redis and the vectors live in Qdrant, and nothing makes
a write to one atomic with a write to the other. That was in [Limits](#limits)
for a long time as a known gap with nothing done about it. Then
[the sparse migration](#the-sparse-index-moved-into-qdrant) made it real: the
collection had to be rebuilt to gain a sparse vector, which dropped every
point, while Redis went on listing eight ready documents. **The UI showed a
healthy corpus and every question abstained.** A gap you have written down is
still a gap.

The two directions are not symmetric, and neither repair is "delete the other
side":

| | What it is | Repair |
| --- | --- | --- |
| Catalogue without vectors | Listed, unanswerable | Index it again — if the source is still on disk |
| ...and the source is gone | An upload that outlived the `emptyDir` holding it | Mark it `failed`, so it stops claiming to be ready |
| Vectors without a catalogue | A delete that got half done | Delete the points |

The middle row is the one worth arguing about. Marking a document failed is a
worse-looking outcome than leaving it alone, and it is the right one: a record
that says `ready` while every question about it abstains is a lie the UI
repeats. Documents that are `queued` or `indexing` are skipped — those have no
vectors *yet*, which is not the same thing.

Reconciliation runs at startup (that is when a collection gets rebuilt, so that
is when the stores are most likely to disagree), on a slow timer, and on
`POST /api/v1/documents/reconcile`, which takes `?dry_run=true` and reports
without touching anything. There is no fast timer on purpose: the sweep scrolls
the whole collection, which is the shape of work the last change went to some
trouble to get off the hot path. Divergence comes from events — a failed
ingest, a migration — not from drift.

**Verified by causing it.** `scripts/reconcile_probe.py` injures both stores on
purpose and checks the repair. Neither injury needs the embedding API — the
orphan carries a made-up vector, the phantom record needs no vector at all — so
it runs against the deployed stack regardless of what the model account is
doing:

```
start        27 points, 8 catalogue records
injured      one orphan point (ghost-probe), one unretrievable record (phantom-probe)
dry run      {"requeued": [], "marked_failed": ["phantom-probe"], "orphans_deleted": ["ghost-probe"], "clean": false}
repaired     {"requeued": [], "marked_failed": ["phantom-probe"], "orphans_deleted": ["ghost-probe"], "clean": false}
points       28 -> 27, expected 27
phantom      status=failed error=source_missing
settled      a second pass reports clean: True
PASS
```

The requeue path is covered by unit tests but not by this probe: re-indexing
calls the embedding API, and the account behind it is out of credits. That half
is asserted, not observed, and this line is here so nobody reads the `PASS`
above as covering it.

## The eval endpoint became a job

Running the golden set takes minutes, and it used to happen inside the request
that asked for it. That shape had four problems, and only one of them was ever
visible: the nginx in front of the API needed `proxy_read_timeout 600s` to stop
cutting the connection. A timeout being configured around is a symptom, and
raising it treated the symptom.

The other three were quieter:

- **A closed tab threw the work away.** Four minutes of model calls, already
  paid for, discarded because nobody was still listening.
- **Two clicks were two runs.** Nothing stopped a second request starting a
  second full pass over the same corpus, and billing for it.
- **A spinner is indistinguishable from a hang.** With no progress, the only
  way to learn the run was still alive was to wait longer than the timeout —
  which is how the timeout got found in the first place.

So `POST /api/v1/eval` now returns `202` with a job id and keeps going;
`GET /api/v1/eval/{job_id}` reports progress and, when it is finished, the
report. `GET /api/v1/eval` returns the most recent run, so reloading the tab
does not mean re-running the set.

**The state lives in Redis, not in the process** — for the same reason
[the sparse index does](#the-sparse-index-moved-into-qdrant). A poll can land
on a different replica than the one doing the work, and a job that only one
process can see is a job that breaks the moment there are two.

**The failure Redis cannot cover is the process dying mid-run.** The job
heartbeats as it goes, and a record that stops beating is reported as
`abandoned` rather than left claiming to be running forever. The heartbeat also
refreshes the single-flight claim, so the two cannot disagree about whether a
run is still alive — and a crashed process cannot block every future run.

Verified against the deployed stack:

```
POST         202 in 14ms  job=20f0dd8ac6d5 0/53
second click job=20f0dd8ac6d5 joined=True (one run, not two)
final        status=failed 0/53 error=RateLimitError: 429 ... no credits remaining
```

Three things in that trace. The start returns in **14 milliseconds** where it
used to hold the connection for minutes. The second click **joined** the run
instead of starting another. And the case count is known at `0/53` before the
first question finishes, rather than `0/?` for however long one model call
takes — a detail worth fixing, because "0 of ?" is the same non-answer the old
spinner gave.

The `failed` is real and is not the job machinery: this ran while the account
behind the embedding and generation calls had no credits. **What is verified
here is the job, not the eval** — a successful run with progress climbing
53 cases has not been observed since the change, and this line is here so the
`202` above is not read as covering it.

With the long request gone, the proxy timeouts it forced came down with it:
600s to 120s in the frontend nginx and 3600 to 120 on the Ingress. The longest
response left is a streamed answer, which is seconds. It is not tighter than
that because SSE is idle between tokens, and a timeout firing mid-answer
produces a truncated answer rather than an error anyone can see.

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

## Limits

Measured and known, not hidden. Each of these is a deliberate stopping point
for a handbook-scale corpus, with the replacement named.

**The SLI counters are still per-process.** Retrieval state is not: dense and
sparse vectors both live in Qdrant, so replicas share one view of the corpus
and a write is visible to all of them at once. See
[the sparse index moved into Qdrant](#the-sparse-index-moved-into-qdrant) for
what that replaced and what it measured.

`/api/v1/metrics` drives the UI and reports the replica that served the
request, so with more than one it shows a slice rather than the system. The
Prometheus endpoint is the answer for anything that has to be true across
replicas; the JSON one stays because a demo should show numbers without asking
you to stand up a scraper first. It is the last piece of per-process state in
the service, and unlike the sparse index it never affected an answer.

**Auth is service-level, not user identity.** Every `/api/v1` route requires
an API key mapped to a named principal, and the semantic cache and rate limit
budget are both keyed by that principal. What it does not have is per-user
login: the nginx container presents one key on behalf of every browser that
reaches it, so anyone who can load the page can query. Real user identity means
a session layer in front, and principals would come from it rather than from a
secret.

**The user-input guard is a regex, and regexes lose this game.** It catches
the phrasings it was written against and nothing else, at a measured 7/11
bypass rate. It is kept as a cheap deterministic first pass, not as the defense
— that is the model's instruction-following, and the probe is what says so.

**The paraphrase cache catches roughly two thirds of rewordings.** At the
calibrated threshold it hits 6 of 9 measured paraphrases. Looser rephrasings
fall through to the full pipeline, which is the safe direction to fail, but it
means the hit rate is bounded by how people happen to phrase things. Raising
recall here needs a threshold that also fuses questions one decisive word
apart, so it is not a knob to turn without a better signal than cosine
similarity.

**The cross-encoder is off, and the measurement is the reason.** It changed
answer correctness by zero on the golden set while adding latency, so it stays
disabled in `infra/k8s/atlas.yaml`. Turning it on also costs a HuggingFace
fetch on the first request — about 12s, and egress the cluster may not have —
so the weights would need baking into the image first. That work is not worth
doing until a corpus exists where the reranker earns it.

**A document whose source is gone cannot be recovered.** Redis holds the
catalogue and runs with AOF on a PVC, so a restart no longer empties it, and
[reconciliation](#keeping-redis-and-qdrant-in-agreement) repairs the two stores
when they disagree. What it cannot do is re-index an upload whose bytes died
with the pod's `emptyDir`; those are marked `failed` rather than silently left
listed. A persistent volume for uploads is the fix, and it is not there because
the demo corpus is seeded from a ConfigMap.

## Sample corpus

Fictional **Kepler Robotics** internal handbook: leave policy, data handling, incident response, vendor terms, and a deliberate indirect-injection document for guardrail testing.
