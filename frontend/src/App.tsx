import { useEffect, useMemo, useState } from "react";
import {
  askStream,
  deleteDoc,
  health,
  listDocs,
  metrics,
  runEval,
  seedCorpus,
  uploadDoc,
  type DocumentRecord,
  type EvalReport,
  type MetricsSnapshot,
  type QueryResponse,
} from "./api";

type Tab = "ask" | "corpus" | "eval" | "sli";

const STARTERS = [
  "How many annual leave days in the first year?",
  "Can I drop customer warehouse camera frames into ChatGPT?",
  "What is the latest a formal SEV-1 RCA may go out?",
  "What is the K-Walk 2 endurance target in hours?",
];

export default function App() {
  const [tab, setTab] = useState<Tab>("ask");
  const [question, setQuestion] = useState(STARTERS[0]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [docs, setDocs] = useState<DocumentRecord[]>([]);
  const [sli, setSli] = useState<MetricsSnapshot | null>(null);
  const [report, setReport] = useState<EvalReport | null>(null);
  const [progress, setProgress] = useState<string | null>(null);
  const [llm, setLlm] = useState<boolean | null>(null);

  async function refresh() {
    try {
      const [d, m, h] = await Promise.all([listDocs(), metrics(), health()]);
      setDocs(d);
      setSli(m);
      setLlm(h.llm);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  useEffect(() => {
    void refresh();
  }, []);

  async function onAsk() {
    setBusy(true);
    setError(null);
    setResult({
      answer: "",
      abstained: false,
      citations: [],
      trace: [],
      retrieval_ms: 0,
      total_ms: 0,
      prompt_tokens: 0,
      completion_tokens: 0,
      tokens_saved_vs_naive: 0,
      cache_hit: false,
    });
    let streamed = "";
    try {
      const r = await askStream(question, {
        onMeta: (meta) => {
          setResult((prev) =>
            prev
              ? {
                  ...prev,
                  retrieval_ms: meta.retrieval_ms,
                  trace: meta.trace,
                  citations: meta.citations,
                }
              : prev,
          );
        },
        onToken: (text) => {
          streamed += text;
          setResult((prev) => (prev ? { ...prev, answer: streamed } : prev));
        },
      });
      setResult(r);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function onSeed() {
    setBusy(true);
    setError(null);
    try {
      await seedCorpus();
      for (let i = 0; i < 8; i++) {
        await new Promise((r) => setTimeout(r, 700));
        await refresh();
        const latest = await listDocs();
        if (latest.length && latest.every((d) => d.status === "ready" || d.status === "failed")) break;
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function onEval() {
    setBusy(true);
    setError(null);
    setProgress(null);
    try {
      setReport(
        await runEval((job) =>
          setProgress(job.status === "running" ? `${job.done}/${job.total || "?"}` : null),
        ),
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
      setProgress(null);
    }
  }

  const ready = docs.filter((d) => d.status === "ready").length;

  return (
    <div className="shell">
      <header className="top">
        <div>
          <p className="kicker">Kepler Robotics · internal</p>
          <h1>Atlas</h1>
        </div>
        <div className="status">
          <span className={llm ? "pill ok" : "pill"}>{llm ? "Model connected" : "No API key · retrieval only"}</span>
          <span className="pill">
            {ready}/{docs.length || 0} docs indexed
          </span>
        </div>
      </header>

      <nav className="tabs">
        {(
          [
            ["ask", "Ask"],
            ["corpus", "Corpus"],
            ["eval", "Eval"],
            ["sli", "SLI"],
          ] as const
        ).map(([id, label]) => (
          <button key={id} className={tab === id ? "tab on" : "tab"} onClick={() => setTab(id)}>
            {label}
          </button>
        ))}
      </nav>

      {error && <p className="err">{error}</p>}

      {tab === "ask" && (
        <section className="grid two">
          <div className="panel">
            <label>Question</label>
            <textarea value={question} onChange={(e) => setQuestion(e.target.value)} rows={4} />
            <div className="starters">
              {STARTERS.map((s) => (
                <button key={s} className="ghost" onClick={() => setQuestion(s)}>
                  {s}
                </button>
              ))}
            </div>
            <button className="primary" disabled={busy} onClick={() => void onAsk()}>
              {busy ? "Streaming answer…" : "Retrieve and answer"}
            </button>
            {result && (
              <div className={`answer ${result.abstained ? "abstain" : ""}`}>
                <p>{result.answer}</p>
                <p className="meta">
                  retrieval {result.retrieval_ms.toFixed(0)}ms · e2e {result.total_ms.toFixed(0)}ms ·
                  prompt {result.prompt_tokens} tok · saved vs naive {result.tokens_saved_vs_naive} ·
                  faithfulness {result.faithfulness?.toFixed(2) ?? "—"}
                  {result.cache_hit ? " · cache" : ""}
                </p>
              </div>
            )}
          </div>
          <div className="panel">
            <h2>Graph trace</h2>
            <ol className="trace">
              {(result?.trace ?? []).map((n, i) => (
                <li key={`${n.node}-${i}`}>
                  <strong>{n.node}</strong>
                  <span>{n.ms.toFixed(0)}ms</span>
                  <em>{n.detail}</em>
                </li>
              ))}
            </ol>
            <h2>Citations</h2>
            <ul className="cites">
              {(result?.citations ?? []).map((c) => (
                <li key={c.chunk_id}>
                  <span>[{c.n}] {c.filename}</span>
                  <p>{c.text}</p>
                </li>
              ))}
            </ul>
          </div>
        </section>
      )}

      {tab === "corpus" && (
        <section className="panel">
          <div className="row">
            <button className="primary" disabled={busy} onClick={() => void onSeed()}>
              Load Kepler sample handbook
            </button>
            <label className="upload">
              Upload md / txt / pdf
              <input
                type="file"
                accept=".md,.txt,.pdf"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (!f) return;
                  void uploadDoc(f).then(() => refresh());
                }}
              />
            </label>
          </div>
          <table>
            <thead>
              <tr>
                <th>File</th>
                <th>Status</th>
                <th>Chunks</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {docs.map((d) => (
                <tr key={d.id}>
                  <td>{d.filename}</td>
                  <td>{d.status}</td>
                  <td>{d.chunks}</td>
                  <td>
                    <button
                      className="ghost"
                      disabled={busy}
                      onClick={() => void deleteDoc(d.id).then(() => refresh())}
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {tab === "eval" && (
        <section className="panel">
          <p className="lead">
            The eval set is part of the product, not a script you run later. Change chunking, the graph, or the model only after this table still holds.
          </p>
          <button className="primary" disabled={busy} onClick={() => void onEval()}>
            {busy ? `Running golden set… ${progress ?? ""}`.trim() : "Run offline eval"}
          </button>
          {report && <EvalView report={report} />}
        </section>
      )}

      {tab === "sli" && sli && (
        <section className="stats">
          <Stat k="Documents" v={String(sli.documents)} />
          <Stat k="Chunks" v={String(sli.chunks)} />
          <Stat k="Retrieval p50" v={`${sli.p50_retrieval_ms.toFixed(0)}ms`} />
          <Stat k="Retrieval p95" v={`${sli.p95_retrieval_ms.toFixed(0)}ms`} />
          <Stat k="Cache hit" v={`${(sli.cache_hit_rate * 100).toFixed(0)}%`} />
          <Stat k="Embedding cache hits" v={String(sli.embedding_cache_hits)} />
          <Stat k="Mean prompt tokens" v={sli.mean_prompt_tokens.toFixed(0)} />
        </section>
      )}
    </div>
  );
}

function Stat({ k, v }: { k: string; v: string }) {
  return (
    <div className="stat">
      <span>{k}</span>
      <strong>{v}</strong>
    </div>
  );
}

function EvalView({ report }: { report: EvalReport }) {
  const cards = useMemo(
    () => [
      ["Retrieval recall", pct(report.retrieval_recall)],
      ["Context precision", pct(report.mean_context_precision)],
      ["Faithfulness", pct(report.mean_faithfulness)],
      ["Correctness", pct(report.mean_correctness)],
      ["Hallucination rate", pct(report.hallucination_rate)],
      ["Abstention accuracy", pct(report.abstention_accuracy)],
      ["Tokens vs naive", `-${report.token_reduction_pct.toFixed(0)}%`],
    ],
    [report]
  );
  return (
    <div>
      <div className="stats">
        {cards.map(([k, v]) => (
          <Stat key={k} k={k} v={v} />
        ))}
      </div>
      <table>
        <thead>
          <tr>
            <th>id</th>
            <th>Hit</th>
            <th>Faith</th>
            <th>Correct</th>
            <th>Abstain</th>
            <th>Halluc.</th>
          </tr>
        </thead>
        <tbody>
          {report.cases.map((c) => (
            <tr key={c.id}>
              <td>{c.id}</td>
              <td>{c.retrieval_hit ? "Y" : "N"}</td>
              <td>{c.faithfulness.toFixed(2)}</td>
              <td>{c.answer_correctness.toFixed(2)}</td>
              <td>
                {c.abstained ? "Y" : "N"}
                {c.abstention_correct != null ? (c.abstention_correct ? " ✓" : " ✗") : ""}
              </td>
              <td>{c.hallucinated ? "Y" : ""}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function pct(n: number) {
  return `${(n * 100).toFixed(0)}%`;
}
