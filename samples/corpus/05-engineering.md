# Engineering onboarding · knowledge base and RAG rules

New hires treat internal Q&A like ChatGPT. Kepler’s Atlas knowledge base has hard rules.

## Chunking

- Retrieval unit (child) targets ~180 words with 40-word overlap
- The model receives the **parent passage** (~900 words), not the retrieved fragment
- Filename and section title are prefixed on each chunk for hybrid retrieval

## Retrieval

1. Dense vectors (Qdrant)
2. BM25
3. Reciprocal Rank Fusion
4. Are the documents enough? If not, rewrite the query and search again, at most twice
5. Faithfulness below 0.7 means abstain — not “it feels right”

## Latency wording

Externally, “sub-second” covers **retrieval only** (embed + fusion), not generation. Generation is gated by the model gateway and reported as its own p95. Mixing the two on a resume will get called out.
