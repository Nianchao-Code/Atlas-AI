# Atlas AI

Production RAG platform for internal handbook Q&A. Hybrid retrieval, a corrective LangGraph pipeline, faithfulness gating, and an offline eval harness — built so every component maps to a real failure mode, not a buzzword checklist.

**Demo flow:** seed corpus → ask a question → inspect citations and graph trace → run the golden-set eval.

## Highlights

| Area | Implementation |
| --- | --- |
| **Retrieval** | Parent-child chunking, BM25 + dense RRF, cross-encoder rerank, LLM document grading |
| **Generation** | SSE streaming answers; token-budget packing on deduplicated parent passages |
| **Orchestration** | LangGraph: guard → cache → rewrite → retrieve → rerank → grade → compress → generate → faithfulness |
| **Quality** | Offline eval on `samples/eval/golden.json` — recall, precision, faithfulness, correctness, hallucination rate, abstention accuracy |
| **Safety** | User prompt-injection blocking + indirect corpus injection handling (`08-injection-bait.md`) |
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

| Method | Path | Description |
| --- | --- | --- |
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
samples/eval     Golden question set
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

Thresholds live in `samples/eval/thresholds.json`.

## What each stage actually buys

`scripts/ablation.py` runs the golden set through progressively richer
pipelines. Each row adds exactly one stage, so a delta belongs to that stage
and nothing else.

| Configuration | Recall | Ctx precision | Faithful | Correct | Halluc. | p95 ms | Tokens |
|---|---|---|---|---|---|---|---|
| **Dense only** | 1.000 ±0.000 | 0.536 ±0.017 | 1.000 ±0.000 | 1.000 ±0.000 | 0.000 | 1469 ±1579 | 932 ±2 |
| **BM25 only** | 1.000 ±0.000 | 0.405 ±0.000 | 1.000 ±0.000 | 1.000 ±0.000 | 0.000 | 3460 ±3630 | 971 ±0 |
| **Hybrid + RRF** | 1.000 ±0.000 | 0.452 ±0.000 | 1.000 ±0.000 | 0.929 ±0.000 | 0.000 | 1767 ±1944 | 989 ±0 |
| **+ cross-encoder** | 1.000 ±0.000 | 0.476 ±0.000 | 1.000 ±0.000 | 0.964 ±0.051 | 0.000 | 628 ±10 | 968 ±0 |
| **+ LLM grading (full)** | 0.923 ±0.000 | 0.786 ±0.000 | 0.993 ±0.010 | 0.839 ±0.025 | 0.000 | 283 ±77 | 239 ±0 |
| **Full − query rewrite** | 0.923 ±0.000 | 0.779 ±0.010 | 1.000 ±0.000 | 0.804 ±0.025 | 0.000 | 1648 ±1850 | 239 ±0 |

14 questions, 2 runs per configuration, mean ±sd. Judge `gpt-4o-mini`,
generation `gpt-4o`. Reproduce with `python scripts/ablation.py --repeats 2`.

**Read it honestly:**

- **The retrieval stack is not earning its keep on this corpus.** Dense, BM25,
  hybrid and hybrid+rerank all reach recall 1.000 — 8 documents and 27 chunks
  are not enough to separate them. Hybrid actually *lowers* context precision
  against dense alone (0.536 → 0.452), because fusing in BM25 hits pulls in
  passages the dense ranker had already pushed down.
- **Grading is the one stage that moves the numbers, and it is a trade, not a
  win.** Context precision 0.45 → 0.79 and prompt tokens 989 → 239 (−76%), paid
  for with recall 1.000 → 0.923 and correctness 0.964 → 0.839. On a corpus this
  small the trade is bad; the token saving is what would justify it at scale.
- **Query rewrite/HyDE is worth roughly +3.5pp correctness** (0.804 → 0.839) at
  identical recall — the smallest defensible component in the stack.
- **The latency column is noise at n=2.** Several standard deviations exceed
  their means. Treat only `+ cross-encoder` (628 ±10) and `full` (283 ±77) as
  measured; the rest needs more runs on an idle machine.

**What the ablation exposed about the eval harness itself**, which matters more
than any row above:

- **Recall saturates at 1.000 for every retriever**, so the golden set cannot
  currently justify any retrieval design decision. It needs harder questions —
  multi-hop, near-duplicate distractors, questions whose answer sits in the
  tail of a long document.
- **`abstention_accuracy` is computed over exactly one case**, and
  `thresholds.json` gates CI at `1.0`. A single question decides whether the
  build is red.
- **`mean_correctness` punishes correct abstentions.** The `unknown` case is
  abstained on correctly (`abstention_accuracy` 1.0) and still scored 0.50 by
  the correctness judge, because the judge grades the refusal text against key
  points it was never supposed to contain.
- **Grading over-abstains.** On `pii-to-llm` the dense-only pipeline retrieves
  the right passage and answers correctly; the full pipeline abstains. That is
  a real regression the aggregate numbers hide.

## Limits

Measured and known, not hidden. Each of these is a deliberate stopping point
for a handbook-scale corpus, with the replacement named.

**Runs as a single replica.** The BM25 index and the SLI counters both live in
process memory. A `bm25:rev` counter in Redis invalidates the snapshot when
another process reindexes, so correctness holds across the API and the worker
-- but every process still keeps its own copy and rebuilds it with a full
Qdrant scroll, synchronously, inside the request that noticed the change.
`Pipeline.warm()` moves that cost to startup; it does not remove it. Past a few
tens of thousands of chunks the rebuild belongs in a background task, or the
sparse index belongs in Qdrant alongside the dense one.

**`/api/v1/metrics` is per-process.** With more than one replica each pod
reports only its own traffic. Horizontal scaling needs the counters exported to
Prometheus rather than summarised in the app.

**No authentication.** Query, upload, and delete are all open, and CORS is
`*`. This is a demo deployment, not a hardened one. Adding auth also means
keying the semantic cache per tenant: today `qa:` keys hash the question text
alone, so two users asking the same thing would share an answer.

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
