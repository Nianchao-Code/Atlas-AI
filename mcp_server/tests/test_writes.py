"""The gated tools, and the gate itself.

`upload_document` changes the corpus and `run_eval` spends money -- about
$0.055 a call, measured over three runs rather than estimated. Both are off
unless the operator turns them on, because a model picks tools from a
sentence and "check that the system is working" is a plausible sentence.

There is no delete tool to test. That is the point of it.
"""

from __future__ import annotations

import httpx
import pytest
from mcp import Client

from tests.conftest import json_route, make_client, text_of


async def test_upload_is_refused_by_default(atlas):
    client = make_client(json_route({"/api/v1/documents": (200, {})}), atlas)
    async with client:
        result = await client.call_tool("upload_document", {"filename": "x.md", "content": "hello"})

    assert result.is_error
    assert "disabled" in text_of(result)
    assert "ATLAS_MCP_ALLOW_WRITES" in text_of(result)
    # Refused here, not at the API: no request was made at all.
    assert atlas.requests == []


async def test_run_eval_is_refused_by_default(atlas):
    client = make_client(json_route({"/api/v1/eval": (202, {})}), atlas)
    async with client:
        result = await client.call_tool("run_eval", {})

    assert result.is_error
    assert "disabled" in text_of(result)
    assert atlas.requests == []


async def test_the_refusal_tells_the_model_not_to_route_around_it(atlas):
    client = make_client(json_route({}), atlas)
    async with client:
        out = text_of(await client.call_tool("run_eval", {}))
    assert "rather than trying another tool" in out


async def test_upload_sends_multipart_when_writes_are_enabled(atlas):
    queued = {"id": "ab12", "filename": "x.md", "bytes": 5, "status": "queued", "chunks": 0}
    client = make_client(json_route({"/api/v1/documents": (200, queued)}), atlas, allow_writes=True)
    async with client:
        result = await client.call_tool("upload_document", {"filename": "x.md", "content": "hello"})

    assert not result.is_error
    assert result.structured_content["status"] == "queued"

    sent = atlas.requests[0]
    assert sent.headers["content-type"].startswith("multipart/form-data")
    assert b'filename="x.md"' in sent.content and b"hello" in sent.content


@pytest.mark.parametrize(
    ("status", "detail", "expected"),
    [
        (415, "unsupported file type; expected one of ['.md', '.pdf', '.txt']", ".md, .txt"),
        (413, "file exceeds the 20971520 byte limit", "too large"),
        (400, "invalid filename", "rejected the request (400)"),
    ],
)
async def test_the_upload_rejections_are_explained(atlas, status, detail, expected):
    client = make_client(
        json_route({"/api/v1/documents": (status, {"detail": detail})}), atlas, allow_writes=True
    )
    async with client:
        result = await client.call_tool("upload_document", {"filename": "x.exe", "content": "x"})

    assert result.is_error
    out = text_of(result)
    assert expected in out
    # The server's own detail is carried through, not replaced by a guess.
    assert detail[:20] in out


async def test_run_eval_returns_a_job_rather_than_a_report(atlas):
    job = {"job_id": "job-1", "status": "running", "done": 0, "total": 53, "report": None}
    client = make_client(json_route({"/api/v1/eval": (202, job)}), atlas, allow_writes=True)
    async with client:
        result = await client.call_tool("run_eval", {"limit": 5})

    assert result.structured_content["job_id"] == "job-1"
    assert atlas.requests[0].url.params["limit"] == "5"


async def test_get_eval_is_readable_without_enabling_writes(atlas):
    # Polling costs nothing and starting does, so only starting is gated.
    done = {"job_id": "job-1", "status": "done", "done": 53, "total": 53, "report": {"n": 53}}
    client = make_client(json_route({"/api/v1/eval/job-1": (200, done)}), atlas)
    async with client:
        result = await client.call_tool("get_eval", {"job_id": "job-1"})

    assert not result.is_error
    assert result.structured_content["report"]["n"] == 53


async def test_an_expired_job_id_says_the_run_is_gone(atlas):
    client = make_client(
        json_route({"/api/v1/eval/old": (404, {"detail": "unknown or expired job"})}), atlas
    )
    async with client:
        result = await client.call_tool("get_eval", {"job_id": "old"})

    assert result.is_error
    assert "expired" in text_of(result)


async def test_there_is_no_delete_tool():
    from atlas_mcp.config import Settings
    from atlas_mcp.server import build_server

    server = build_server(Settings(base_url="http://x", api_key="", timeout_s=1, allow_writes=True))
    async with Client(server) as client:
        names = {t.name for t in (await client.list_tools()).tools}

    # Even with writes enabled. The capability is absent, not guarded.
    assert not any("delete" in n or "remove" in n for n in names)


async def test_list_documents_clamps_an_oversized_limit(atlas):
    page = {"documents": [], "total": 10008, "chunks": 40079, "limit": 200, "offset": 0}
    client = make_client(json_route({"/api/v1/documents": (200, page)}), atlas)
    async with client:
        result = await client.call_tool("list_documents", {"limit": 99999})

    # A model asking for everything gets a page, and the true total alongside
    # it so it knows there is more.
    assert atlas.requests[0].url.params["limit"] == "200"
    assert result.structured_content["total_documents"] == 10008


async def test_an_unreachable_backend_names_the_url_and_the_fix(atlas):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = make_client(refuse, atlas)
    async with client:
        result = await client.call_tool("search_documents", {"question": "q"})

    out = text_of(result)
    assert result.is_error
    assert "http://atlas.test" in out
    assert "docker compose up" in out


async def test_a_timeout_is_not_reported_as_a_dead_backend(atlas):
    def stall(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    client = make_client(stall, atlas)
    async with client:
        out = text_of(await client.call_tool("ask_handbook", {"question": "q"}))

    assert "did not respond within 5s" in out
    assert "Do not retry immediately" in out


async def test_a_bad_key_says_it_is_configuration_not_bad_luck(atlas):
    client = make_client(
        json_route({"/api/v1/retrieve": (401, {"detail": "missing or invalid API key"})}), atlas
    )
    async with client:
        out = text_of(await client.call_tool("search_documents", {"question": "q"}))

    assert "401" in out
    assert "retrying will not help" in out


async def test_rate_limiting_explains_the_shared_budget(atlas):
    def limited(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429, json={"detail": "rate limit exceeded"}, headers={"Retry-After": "60"}
        )

    client = make_client(limited, atlas)
    async with client:
        out = text_of(await client.call_tool("search_documents", {"question": "q"}))

    assert "Wait 60s" in out
    # The thing a caller cannot otherwise know: one key, one budget, shared.
    assert "shared" in out


async def test_a_backend_crash_is_marked_as_not_worth_retrying(atlas):
    client = make_client(json_route({"/api/v1/retrieve": (500, {"detail": "boom"})}), atlas)
    async with client:
        out = text_of(await client.call_tool("search_documents", {"question": "q"}))

    assert "backend bug, not a bad request" in out


async def test_the_api_key_never_appears_in_an_error(atlas):
    def leaky(request: httpx.Request) -> httpx.Response:
        # A library error that happens to contain the key, which is the case
        # redaction exists for.
        raise httpx.ConnectError(
            f"failed connecting with header test-secret-key to {request.url}", request=request
        )

    client = make_client(leaky, atlas)
    async with client:
        out = text_of(await client.call_tool("search_documents", {"question": "q"}))

    assert "test-secret-key" not in out
    assert "***" in out
