export type Citation = {
  n: number;
  doc_id: string;
  filename: string;
  chunk_id: string;
  score: number;
  text: string;
};

export type TraceNode = {
  node: string;
  ms: number;
  detail: string;
  data: Record<string, unknown>;
};

export type QueryResponse = {
  answer: string;
  abstained: boolean;
  citations: Citation[];
  trace: TraceNode[];
  retrieval_ms: number;
  total_ms: number;
  prompt_tokens: number;
  completion_tokens: number;
  tokens_saved_vs_naive: number;
  cache_hit: boolean;
  rewritten_query?: string | null;
  faithfulness?: number | null;
};

export type DocumentRecord = {
  id: string;
  filename: string;
  bytes: number;
  status: string;
  chunks: number;
  error?: string | null;
};

export type MetricsSnapshot = {
  documents: number;
  chunks: number;
  queries: number;
  cache_hits: number;
  cache_hit_rate: number;
  p50_retrieval_ms: number;
  p95_retrieval_ms: number;
  mean_prompt_tokens: number;
  embedding_cache_hits: number;
};

export type EvalReport = {
  n: number;
  retrieval_recall: number;
  mean_context_precision: number;
  mean_faithfulness: number;
  mean_correctness: number;
  hallucination_rate: number;
  abstention_accuracy: number;
  p95_retrieval_ms: number;
  mean_prompt_tokens: number;
  naive_prompt_tokens: number;
  token_reduction_pct: number;
  cases: Array<{
    id: string;
    question: string;
    retrieval_hit: boolean;
    context_precision: number;
    faithfulness: number;
    answer_correctness: number;
    hallucinated: boolean;
    abstained: boolean;
    abstention_correct?: boolean | null;
    retrieval_ms: number;
    prompt_tokens: number;
    answer: string;
  }>;
};

async function ensureOk(res: Response) {
  if (!res.ok) {
    const t = await res.text();
    throw new Error(t || res.statusText);
  }
}

export async function askStream(
  question: string,
  handlers: {
    onMeta?: (data: { retrieval_ms: number; trace: TraceNode[]; citations: Citation[] }) => void;
    onToken?: (text: string) => void;
  },
): Promise<QueryResponse> {
  const res = await fetch("/api/v1/query/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, use_cache: true }),
  });
  await ensureOk(res);
  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";
  let finalPayload: QueryResponse | null = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";
    for (const part of parts) {
      const lines = part.split("\n");
      const eventLine = lines.find((l) => l.startsWith("event:"));
      const dataLine = lines.find((l) => l.startsWith("data:"));
      if (!eventLine || !dataLine) continue;
      const event = eventLine.slice(6).trim();
      const data = JSON.parse(dataLine.slice(5).trim());
      if (event === "meta") handlers.onMeta?.(data);
      if (event === "token") handlers.onToken?.(data.text as string);
      if (event === "done") finalPayload = data as QueryResponse;
    }
  }
  if (!finalPayload) throw new Error("Stream ended without final payload");
  return finalPayload;
}

export async function ask(question: string): Promise<QueryResponse> {
  const res = await fetch("/api/v1/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, use_cache: true }),
  });
  await ensureOk(res);
  return res.json();
}

export async function listDocs(): Promise<DocumentRecord[]> {
  const res = await fetch("/api/v1/documents");
  await ensureOk(res);
  return res.json();
}

export async function seedCorpus(): Promise<DocumentRecord[]> {
  const res = await fetch("/api/v1/documents/seed", { method: "POST" });
  await ensureOk(res);
  return res.json();
}

export async function uploadDoc(file: File): Promise<DocumentRecord> {
  const body = new FormData();
  body.append("file", file);
  const res = await fetch("/api/v1/documents", { method: "POST", body });
  await ensureOk(res);
  return res.json();
}

export async function deleteDoc(docId: string): Promise<void> {
  const res = await fetch(`/api/v1/documents/${docId}`, { method: "DELETE" });
  await ensureOk(res);
}

export async function runEval(): Promise<EvalReport> {
  const res = await fetch("/api/v1/eval", { method: "POST" });
  await ensureOk(res);
  return res.json();
}

export async function metrics(): Promise<MetricsSnapshot> {
  const res = await fetch("/api/v1/metrics");
  await ensureOk(res);
  return res.json();
}

export async function health(): Promise<{ ok: boolean; llm: boolean; kafka: boolean }> {
  const res = await fetch("/health");
  await ensureOk(res);
  return res.json();
}
