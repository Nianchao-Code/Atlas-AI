import { describe, expect, it, vi, afterEach } from "vitest";
import { askStream } from "./api";

/**
 * The SSE reader is the one piece of real parsing in the frontend, and its
 * failure mode is quiet: a token split across two network reads is dropped or
 * mangled, and the answer just looks slightly wrong. These drive it with
 * chunk boundaries chosen to land in the worst places.
 */

function streamOf(chunks: string[]): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const c of chunks) controller.enqueue(encoder.encode(c));
      controller.close();
    },
  });
  return new Response(body, { status: 200 });
}

function mockFetch(res: Response) {
  const fn = vi.fn().mockResolvedValue(res);
  globalThis.fetch = fn as unknown as typeof fetch;
  return fn;
}

const DONE = `event: done\ndata: ${JSON.stringify({
  answer: "15 days",
  citations: [],
  trace: [],
  abstained: false,
})}\n\n`;

afterEach(() => vi.restoreAllMocks());

describe("askStream", () => {
  it("routes meta, token and done to their handlers", async () => {
    mockFetch(
      streamOf([
        `event: meta\ndata: ${JSON.stringify({ retrieval_ms: 12, trace: [], citations: [] })}\n\n`,
        `event: token\ndata: ${JSON.stringify({ text: "15 " })}\n\n`,
        `event: token\ndata: ${JSON.stringify({ text: "days" })}\n\n`,
        DONE,
      ]),
    );

    const tokens: string[] = [];
    let metaSeen = 0;
    const payload = await askStream("how many days?", {
      onMeta: () => (metaSeen += 1),
      onToken: (t) => tokens.push(t),
    });

    expect(metaSeen).toBe(1);
    expect(tokens.join("")).toBe("15 days");
    expect(payload.answer).toBe("15 days");
  });

  it("reassembles an event split across reads", async () => {
    // The blank-line delimiter itself lands on a chunk boundary, which is the
    // case a naive per-chunk parser drops.
    const event = `event: token\ndata: ${JSON.stringify({ text: "hello" })}\n\n`;
    const cut = event.length - 1;
    mockFetch(streamOf([event.slice(0, cut), event.slice(cut), DONE]));

    const tokens: string[] = [];
    await askStream("q", { onToken: (t) => tokens.push(t) });
    expect(tokens).toEqual(["hello"]);
  });

  it("survives a chunk boundary inside the JSON payload", async () => {
    const event = `event: token\ndata: ${JSON.stringify({ text: "a token with spaces" })}\n\n`;
    const mid = Math.floor(event.length / 2);
    mockFetch(streamOf([event.slice(0, mid), event.slice(mid), DONE]));

    const tokens: string[] = [];
    await askStream("q", { onToken: (t) => tokens.push(t) });
    expect(tokens).toEqual(["a token with spaces"]);
  });

  it("handles several events arriving in one read", async () => {
    const two =
      `event: token\ndata: ${JSON.stringify({ text: "one " })}\n\n` +
      `event: token\ndata: ${JSON.stringify({ text: "two" })}\n\n`;
    mockFetch(streamOf([two + DONE]));

    const tokens: string[] = [];
    await askStream("q", { onToken: (t) => tokens.push(t) });
    expect(tokens).toEqual(["one ", "two"]);
  });

  it("keeps newlines inside a token intact", async () => {
    // JSON escapes them, so they must not be mistaken for event delimiters.
    const text = "line one\nline two";
    mockFetch(
      streamOf([`event: token\ndata: ${JSON.stringify({ text })}\n\n`, DONE]),
    );

    const tokens: string[] = [];
    await askStream("q", { onToken: (t) => tokens.push(t) });
    expect(tokens).toEqual([text]);
  });

  it("throws when the stream ends without a final payload", async () => {
    mockFetch(streamOf([`event: token\ndata: ${JSON.stringify({ text: "partial" })}\n\n`]));
    await expect(askStream("q", {})).rejects.toThrow(/without final payload/i);
  });

  it("surfaces the server's error body rather than a bare status", async () => {
    mockFetch(new Response("rate limit exceeded: 300 requests/minute", { status: 429 }));
    await expect(askStream("q", {})).rejects.toThrow(/rate limit exceeded/);
  });
});
