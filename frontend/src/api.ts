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

export type DocumentPage = {
  documents: DocumentRecord[];
  total: number;
  chunks: number;
  limit: number;
  offset: number;
};

/**
 * A page of the catalogue, plus the true total.
 *
 * The endpoint used to return every record. That was fine at eight documents
 * and a 10MB response at ten thousand, so the count the header shows now comes
 * from `total` rather than from the length of the list.
 */
export async function listDocs(limit = 200, offset = 0): Promise<DocumentPage> {
  const res = await fetch(`/api/v1/documents?limit=${limit}&offset=${offset}`);
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

export type EvalJob = {
  job_id: string;
  status: "running" | "done" | "failed";
  done: number;
  total: number;
  elapsed_s: number;
  joined: boolean;
  error: string | null;
  report: EvalReport | null;
};

export async function startEval(): Promise<EvalJob> {
  const res = await fetch("/api/v1/eval", { method: "POST" });
  await ensureOk(res);
  return res.json();
}

export async function evalStatus(jobId: string): Promise<EvalJob> {
  const res = await fetch(`/api/v1/eval/${jobId}`);
  await ensureOk(res);
  return res.json();
}

/**
 * Start a run and follow it to the end.
 *
 * The run outlives this request: the server returns a job id straight away and
 * keeps going whether or not anyone is still polling. Closing the tab mid-run
 * no longer throws the work away, and a second click joins the run in progress
 * rather than starting a second one.
 */
export async function runEval(
  onProgress?: (job: EvalJob) => void,
  poll = (ms: number) => new Promise((r) => setTimeout(r, ms)),
): Promise<EvalReport> {
  let job = await startEval();
  onProgress?.(job);
  while (job.status === "running") {
    await poll(2000);
    job = await evalStatus(job.job_id);
    onProgress?.(job);
  }
  if (job.status !== "done" || !job.report) {
    throw new Error(job.error || "eval failed");
  }
  return job.report;
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
