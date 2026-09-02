import { describe, expect, it, vi, afterEach } from "vitest";
import { runEval } from "./api";

/**
 * The eval run outlives the request that starts it, so the client's job is to
 * start it once and then follow it. What these pin is the part that is easy to
 * get wrong and quiet when it is: polling until the job actually finishes
 * rather than reading the first response as the answer, and turning a failed
 * job into a thrown error instead of a report-shaped null.
 */

const REPORT = {
  n: 2,
  retrieval_recall: 1,
  mean_context_precision: 0.5,
  mean_faithfulness: 1,
  mean_correctness: 1,
  hallucination_rate: 0,
  abstention_accuracy: 1,
  p95_retrieval_ms: 10,
  mean_prompt_tokens: 100,
  naive_prompt_tokens: 200,
  token_reduction_pct: 50,
  cases: [],
};

function job(over: Record<string, unknown> = {}) {
  return {
    job_id: "abc123",
    status: "running",
    done: 0,
    total: 2,
    elapsed_s: 0,
    joined: false,
    error: null,
    report: null,
    ...over,
  };
}

function mockSequence(bodies: unknown[]) {
  const fn = vi.fn();
  for (const b of bodies) {
    fn.mockResolvedValueOnce(new Response(JSON.stringify(b), { status: 200 }));
  }
  globalThis.fetch = fn as unknown as typeof fetch;
  return fn;
}

const noWait = () => Promise.resolve();

afterEach(() => vi.restoreAllMocks());

describe("runEval", () => {
  it("polls until the job is done and returns the report", async () => {
    const fetchMock = mockSequence([
      job(),
      job({ done: 1 }),
      job({ status: "done", done: 2, report: REPORT }),
    ]);

    const report = await runEval(undefined, noWait);

    expect(report).toEqual(REPORT);
    // One POST to start, then GETs. The start must not be repeated: a second
    // POST would be a second run against a paid API.
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(fetchMock.mock.calls[0][1]).toMatchObject({ method: "POST" });
    expect(fetchMock.mock.calls[1][0]).toBe("/api/v1/eval/abc123");
    expect(fetchMock.mock.calls[2][1]).toBeUndefined();
  });

  it("reports progress for every poll, including the first", async () => {
    mockSequence([job(), job({ done: 1 }), job({ status: "done", done: 2, report: REPORT })]);
    const seen: string[] = [];

    await runEval((j) => seen.push(`${j.status}:${j.done}/${j.total}`), noWait);

    expect(seen).toEqual(["running:0/2", "running:1/2", "done:2/2"]);
  });

  it("throws the job's own error when the run fails", async () => {
    mockSequence([job(), job({ status: "failed", error: "RateLimitError: no credits" })]);

    await expect(runEval(undefined, noWait)).rejects.toThrow("RateLimitError: no credits");
  });

  it("does not mistake a finished job with no report for success", async () => {
    // A record that says done but carries nothing is a server bug; returning
    // null here would surface as an empty table rather than an error.
    mockSequence([job({ status: "done", report: null })]);

    await expect(runEval(undefined, noWait)).rejects.toThrow();
  });

  it("returns straight away when the first response is already finished", async () => {
    // Joining a run that finished between the click and the request.
    const fetchMock = mockSequence([job({ status: "done", done: 2, report: REPORT, joined: true })]);

    expect(await runEval(undefined, noWait)).toEqual(REPORT);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
