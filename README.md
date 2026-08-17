# Atlas AI

A production RAG system built for AI-engineer interviews: every component exists because a real failure mode exists, not because the 2026 skill list needed another logo.

Kepler Robotics internal handbook → hybrid retrieval → LangGraph corrective graph → abstain instead of inventing → offline eval dashboard.

## What 2026 AI roles actually screen for

Job posts for AI Engineer / LLM Engineer / RAG Engineer rarely reward “I can prompt GPT.” They reward systems you can defend:

| What the JD asks | What this repo has | How to talk about it |
| --- | --- | --- |
| RAG pipeline design (chunking, hybrid search, rerank) | Parent-child chunks + BM25/dense RRF + document grading | Short spans retrieve well; parent passages generate well. RRF does not need a learned ranker on 10k docs. |
| Evals (shows up in roughly half of postings) | `samples/eval/golden.json` + faithfulness / correctness / hallucination rate | The eval set is part of the product. Change the graph only after this table still holds. |
| Agents / stateful orchestration | **One** LangGraph: rewrite → retrieve → grade → compress → generate → faithfulness gate | Not a multi-agent play. Bad retrieval must retry; thin evidence must refuse. |
| Production SLIs: latency, cost, cache | Retrieval p50/p95 reported separately from generation; Redis caches embeddings and near-duplicate questions | “Sub-second” on a resume may only include retrieval. Counting GPT generation will get you caught. |
| Guardrails | User injection + indirect corpus injection (`08-injection-bait.md`) | RAG’s distinctive hole is instructions that arrive *through retrieval*, not the user’s chat. |

The original stack (FastAPI / React / LangChain / Kafka / Redis / K8s / GCP) is still represented. The three additions were forced by failure modes, not by keyword stuffing.

## Architecture (problem → component)

```
Upload / seed documents
    → Redis Streams (local) or Kafka/Redpanda (production)
    → Worker: chunk → embed (Redis content-hash cache) → Qdrant
Query
    → Semantic cache hit? return
    → LangGraph
         guard → rewrite → hybrid retrieve → grade
              not enough? rewrite and search again (max 2) or abstain
              enough? parent dedupe + token-budget pack → generate → faithfulness
                   < 0.7 regenerate once, still weak → abstain
    → Citations + graph trace in the UI
Offline eval
    → recall / context precision / faithfulness / correctness / hallucination / tokens vs naive
```

### Why these, and not the rest

**Keep**

- **FastAPI + React**: interviewers need to see citations and the graph trace. Streamlit reads as a tutorial.
- **Qdrant**: one store for dense search and payload filters. A 10k-doc corpus does not need three vector databases.
- **Redis**: skip re-embedding identical chunks; skip the full graph on the same / near-duplicate question. That is how “3× indexing throughput / 40% fewer tokens” can be a real mechanism.
- **Kafka protocol**: ingest must be decoupled from the API or upload latency includes embedding. Local default is Redis Streams; `docker compose --profile kafka` swaps in Redpanda. **Same IndexJob schema.** If asked why Kafka at 10k docs: it is not the document count, it is two different SLOs (sync embed vs API latency).
- **K8s manifests**: a deploy path, not the product. On GCP use Memorystore plus a Qdrant StatefulSet. No extra Terraform just for the resume.

**Add (once each)**

- **LangGraph**: a linear chain cannot say “retrieval was bad, rewrite.” 2026 JDs ask for agents; they mean a **stateful, branching graph**, not CrewAI stacked on AutoGen.
- **Hybrid retrieval + RRF**: keywords (contract KV-2025-441, 1.2%) drop on dense-only search. The original project claimed hybrid; this repo makes it measurable.
- **LLM-as-judge on the serving path**: the original used a judge for offline scores. Here faithfulness below threshold **abstains**. Hallucination control is a gate, not another chart.
- **Parent-child + token budget**: the concrete 40% token story — retrieve children, send unique parents, greedy-pack to a budget.

**Deliberately skip**

| Not included | Why |
| --- | --- |
| GraphRAG / Neo4j | The corpus is a policy handbook, not a multi-hop entity graph. You could not defend it. |
| LlamaIndex + LangChain together | One orchestration layer. LangGraph here; retrieval is custom. |
| CrewAI / AutoGen | One corrective graph is enough. Multi-agent needs multiple roles; handbook Q&A does not. |
| LoRA / fine-tuning | Policies change. Prefer RAG where overnight updates matter. |
| vLLM / self-hosted inference | No GPU-cost story. Adding it looks like stacking. |
| LangSmith account | Traces render in the product. Wire an exporter if the company already pays for LangSmith. |
| Weaviate + Pinecone + Qdrant | Qdrant only. |

## Run locally

Needs Python 3.11+, Node 20+, Docker (Redis + Qdrant).

```powershell
cd "D:\Atlas AI"
copy .env.example .env
# Set OPENAI_API_KEY. Compatible gateways: change OPENAI_BASE_URL.

docker compose up -d redis qdrant

cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
uvicorn app.main:app --reload --port 8000
```

Second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open http://127.0.0.1:5173 → **Corpus → Load Kepler sample handbook** → Ask. Then run **Eval**.

Without an API key the system still chunks and BM25-retrieves; generation falls back to extracts. LLM-as-judge in eval degrades to keyword overlap.

Full compose (worker + frontend image):

```powershell
docker compose up --build
```

Kafka-protocol bus:

```powershell
docker compose --profile kafka up -d
# Set KAFKA_BROKERS=localhost:19092 in .env and pip install -e ".[kafka]"
```

## Resume bullets (replace with numbers from your eval page)

Do not copy the original 3× / 40% / sub-second claims unless the eval page actually shows them.

> Built Atlas, a corrective RAG system (FastAPI, LangGraph, Qdrant, Redis) for an internal handbook corpus. Hybrid BM25+dense retrieval with parent-child packing cut prompt tokens versus naive top-8 stuffing; a faithfulness gate abstains instead of answering ungrounded questions. Offline eval tracks retrieval recall, faithfulness, correctness, and hallucination rate on a golden set that includes indirect prompt injection.

## Questions interviewers will ask (run the product first)

1. **Does sub-second include generation?** No. Use retrieval p95 on the SLI page versus end-to-end ms on the answer.
2. **Why not GraphRAG?** Handbook QA is clause lookup, not “the CEO of the parent of the vendor of A.”
3. **Is Kafka overkill?** The queue abstraction is Redis Streams; Kafka is the production bus. 10k docs can skip Kafka. Embedding inside the API thread cannot.
4. **How do you cut hallucination?** Not a longer prompt. Grade + faith < 0.7 → abstain, plus injection cases in the golden set.
5. **How do you regress a model swap?** `POST /api/v1/eval`. That is why evals beat prompt-tinkering in 2026 JDs.

## Layout

```
backend/app/     FastAPI, graph, retrieval, eval
frontend/        Ask / Corpus / Eval / SLI
samples/corpus   Kepler internal handbook (includes injection bait)
samples/eval     golden set
infra/k8s        deploy sketch
```
