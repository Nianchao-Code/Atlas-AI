import { describe, expect, it, vi, afterEach } from "vitest";
import { listDocs } from "./api";

/**
 * The catalogue endpoint used to return every record, and the header counted
 * documents by taking the length of that array. At ten thousand documents that
 * is a 10MB response and a wrong-looking count if anything is ever truncated,
 * so the endpoint pages and reports the total separately.
 *
 * What these pin is the part that would fail quietly: asking for a page at all,
 * and keeping `total` distinct from the number of rows returned.
 */

function mockJson(body: unknown, status = 200) {
  const fn = vi.fn().mockResolvedValue(new Response(JSON.stringify(body), { status }));
  globalThis.fetch = fn as unknown as typeof fetch;
  return fn;
}

const PAGE = {
  documents: [
    { id: "a", filename: "a.md", bytes: 10, status: "ready", chunks: 3, error: null },
    { id: "b", filename: "b.md", bytes: 10, status: "ready", chunks: 4, error: null },
  ],
  total: 10008,
  chunks: 40079,
  limit: 200,
  offset: 0,
};

afterEach(() => vi.restoreAllMocks());

describe("listDocs", () => {
  it("asks for a bounded page rather than the whole catalogue", async () => {
    const fetchMock = mockJson(PAGE);

    await listDocs();

    const url = fetchMock.mock.calls[0][0] as string;
    expect(url).toContain("limit=");
    expect(url).toContain("offset=");
  });

  it("passes an explicit page through", async () => {
    const fetchMock = mockJson({ ...PAGE, limit: 50, offset: 100 });

    await listDocs(50, 100);

    expect(fetchMock.mock.calls[0][0]).toBe("/api/v1/documents?limit=50&offset=100");
  });

  it("keeps the total distinct from the number of rows returned", async () => {
    mockJson(PAGE);

    const page = await listDocs();

    // The header shows `total`. Counting page.documents would say 2 of 10,008
    // documents are indexed, which is the bug this shape exists to prevent.
    expect(page.documents).toHaveLength(2);
    expect(page.total).toBe(10008);
    expect(page.chunks).toBe(40079);
  });

  it("surfaces an error status instead of returning an empty page", async () => {
    // An empty page and a failed request look identical to the caller unless
    // the failure throws, and "0 documents" is a believable lie.
    mockJson({ detail: "unauthorised" }, 401);

    await expect(listDocs()).rejects.toThrow();
  });
});
