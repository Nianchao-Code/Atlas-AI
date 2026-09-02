# Atlas AI

Production RAG platform for internal handbook Q&A. Hybrid retrieval, a corrective LangGraph pipeline, faithfulness gating, and an offline eval harness — built so every component maps to a real failure mode, not a buzzword checklist.

**Demo flow:** seed corpus → ask a question → inspect citations and graph trace → run the golden-set eval.

![Atlas answering a handbook question, with the graph trace and cited sources filling in as the pipeline runs](docs/demo.gif)

One question against the deployed stack, uncut: guard, a cache miss, query
rewrite, hybrid retrieval (`dense=24 bm25=12 rrf=24`), the rerank node cutting
24 candidates to 12, LLM grading, parent-passage packing, a streamed answer,
and the faithfulness gate scoring it 1.00 — 3.9s end to end. The cross-encoder
is off in this deployment, so that step is a truncation rather than a rerank;
the [ablation](#what-each-stage-actually-buys) is why. Recorded by
`scripts/capture_demo.py`, which drives the real UI.

## Highlights

| Area | Implementation |
| --- | --- |
| **Retrieval** | Parent-child chunking, BM25 + dense RRF, cross-encoder rerank, LLM document grading |
| **Generation** | SSE streaming answers; token-budget packing on deduplicated parent passages |
| **Orchestration** | LangGraph: guard → cache → rewrite → retrieve → rerank → grade → compress → generate → faithfulness |
| **Quality** | 53-question golden set tagged by failure mode — recall, precision, faithfulness, correctness, hallucination rate, abstention accuracy, plus a stage-by-stage ablation |
| **Safety** | API-key auth with per-principal cache isolation and rate limiting; user prompt-injection blocking + indirect corpus injection handling (`08-injection-bait.md`) |
| **Ops** | Retrieval p50/p95 separate from end-to-end latency; Redis embedding + semantic QA cache |
| **Ingest** | Async worker queue (Redis Streams locally, Kafka/Redpanda in production) |

## Architecture

```
Upload / seed documents
    → Redis Streams (local) or Kafka/Redpanda (production)
    → Worker: chunk → embed (Redis content-hash cache) → Qdrant

Query
    → Semantic cache hit? return
    → LangGraph
         guard → rewrite → hybrid retrieve → cross-encoder rerank → grade
              not enough? rewrite and search again (max 2) or abstain
              enough? parent dedupe + token-budget pack → generate (SSE) → faithfulness
                   score < 0.7 → regenerate once, still weak → abstain
    → Citations + graph trace in the UI

Offline eval
    → recall / context precision / faithfulness / correctness / hallucination / tokens vs naive
```

## Tech stack

**Backend:** FastAPI · LangGraph · Qdrant · Redis · OpenAI-compatible LLM API  
**Frontend:** React · Vite  
**Infra:** Docker Compose · Kubernetes manifests (`infra/k8s/`) · optional Redpanda (Kafka protocol)

## Design decisions

- **Parent-child chunks** — children for search, parents for generation. Fewer, coherent context blocks instead of stuffing raw top-k snippets.
- **Hybrid + RRF + cross-encoder** — RRF fuses dense and BM25; a small cross-encoder reranks the top candidates before the LLM grader.
- **Corrective graph, not multi-agent** — one stateful graph that rewrites bad retrieval and abstains on thin evidence.
- **Faithfulness as a serving gate** — scores below 0.7 trigger one regeneration, then abstain. Hallucination control is enforced, not only measured offline.
- **Eval set is part of the product** — graph and chunking changes should pass the golden set before shipping.
- **Single vector store** — Qdrant for dense search and payload filters. Handbook Q&A does not need GraphRAG or a second index.

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

Without an API key the system still indexes and BM25-retrieves; generation falls back to extracts. LLM-as-judge metrics degrade to keyword overlap.

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
| `POST` | `/api/v1/query` | Run the RAG graph (non-streaming) |
| `POST` | `/api/v1/query/stream` | Same graph, streams answer tokens via SSE |
| `POST` | `/api/v1/eval` | Run offline golden-set eval |
| `GET` | `/api/v1/metrics` | SLI snapshot (latency, cache, tokens) |
| `POST` | `/api/v1/documents/seed` | Load sample handbook |
| `POST` | `/api/v1/documents` | Upload md / txt / pdf |
| `DELETE` | `/api/v1/documents/{id}` | Remove a document |

## Project layout

```
backend/app/     FastAPI, LangGraph pipeline, retrieval, eval
frontend/        Ask · Corpus · Eval · SLI dashboards
samples/corpus   Kepler internal handbook (includes injection test doc)
samples/eval     Golden question set (53 cases, tagged by category)
docs/            Demo recording used by this README
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

| Configuration | Recall | Ctx precision | Faithful | Correct | Halluc. | p95 ms | Tokens |
|---|---|---|---|---|---|---|---|
| **Dense only** | 1.000 ±0.000 | 0.508 ±0.002 | 0.981 ±0.000 | **0.925** ±0.000 | 0.000 | 1392 ±1469 | 961 |
| **BM25 only** | 1.000 ±0.000 | 0.403 ±0.002 | 0.977 ±0.000 | 0.863 ±0.007 | 0.000 | 304 ±8 | 942 ±7 |
| **Hybrid + RRF** | 1.000 ±0.000 | 0.434 ±0.000 | 0.980 ±0.001 | 0.906 ±0.000 | 0.000 | 485 ±93 | 990 |
| **+ cross-encoder** | 1.000 ±0.000 | 0.428 ±0.000 | 0.977 ±0.000 | 0.906 ±0.000 | 0.000 | 660 ±316 | 985 ±8 |
| **+ LLM grading (full)** | 0.961 ±0.000 | **0.830** ±0.003 | 0.975 ±0.004 | 0.915 ±0.013 | 0.000 | 1024 ±336 | **232** ±5 |
| **Full − query rewrite** | 0.961 ±0.000 | 0.830 ±0.003 | 0.975 ±0.001 | 0.915 ±0.013 | 0.000 | 1099 ±1159 | 232 ±5 |

53 questions, 2 runs per configuration, mean ±sd. Judge `gpt-4o-mini`,
generation `gpt-4o`. Reproduce with `python scripts/ablation.py --repeats 2`.

Aggregates hide where the differences live, so the same runs by question type
(answer correctness, one run):

| Category | n | Dense | BM25 | Full | What it stresses |
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
  BM25 everywhere except `semantic`, where questions are deliberately worded
  away from the corpus vocabulary and BM25 scores 0.500 against dense 1.000.
- **BM25 and RRF do not.** Fusing BM25 into dense *lowers* overall correctness
  (0.925 → 0.906) and context precision (0.508 → 0.434): the lexical hits
  displace passages the dense ranker had already ordered correctly.
- **The cross-encoder does nothing here.** Identical correctness to plain
  hybrid, marginally worse precision, and it adds latency. On this corpus it is
  pure cost, and `ENABLE_CROSS_ENCODER` stays off in the deployed config.
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
show BM25 winning on exact identifiers and it did not — dense retrieves
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

## Limits

Measured and known, not hidden. Each of these is a deliberate stopping point
for a handbook-scale corpus, with the replacement named.

**Runs as a single replica.** The BM25 index and the SLI counters both live in
process memory. A `bm25:rev` counter in Redis invalidates the snapshot when
another process reindexes, so correctness holds across the API and the worker,
but every process still keeps its own copy.

No request pays to rebuild it. A background task polls the revision every
`BM25_REFRESH_SECONDS` and rebuilds in a worker thread, because both the Qdrant
scroll and the index build are synchronous and O(corpus) -- on the event loop
they stall every other in-flight request, which is how a cold rebuild once put
19s into a p95. Measured across an ingest, retrieval stays at 232ms against a
130-324ms steady-state baseline, and the worker's reindex shows up in the API
roughly two seconds later.

The remaining cost is memory and duplicated work: every replica rebuilds the
whole corpus for itself. Past a few tens of thousands of chunks the sparse
index belongs in Qdrant alongside the dense one, which removes the per-process
copy entirely.

**`/api/v1/metrics` is per-process.** With more than one replica each pod
reports only its own traffic. Horizontal scaling needs the counters exported to
Prometheus rather than summarised in the app.

**Auth is service-level, not user identity.** Every `/api/v1` route requires
an API key mapped to a named principal, and the semantic cache and rate limit
budget are both keyed by that principal. What it does not have is per-user
login: the nginx container presents one key on behalf of every browser that
reaches it, so anyone who can load the page can query. Real user identity means
a session layer in front, and principals would come from it rather than from a
secret.

**The semantic cache scans linearly.** A miss walks up to 200 recent entries
with two Redis round trips each before falling through to the graph. A second
Qdrant collection is the obvious replacement once that cost matters.

**The cross-encoder downloads at first use.** `ENABLE_CROSS_ENCODER` is off in
`infra/k8s/atlas.yaml` because the model is fetched from HuggingFace on the
first request, which needs egress and adds ~12s to that request. Bake the
weights into the image before turning it on.

**Redis and Qdrant can still diverge.** Redis holds the document catalogue and
now runs with AOF on a PVC, so a restart no longer empties it. There is still
no reconciliation job: if the two stores disagree, uploaded documents can leave
orphaned vectors that nothing points at.

## Sample corpus

Fictional **Kepler Robotics** internal handbook: leave policy, data handling, incident response, vendor terms, and a deliberate indirect-injection document for guardrail testing.
