"""One test that starts the real process and talks to it over a pipe.

Everything else in this suite uses the in-memory client, which is fast and
exercises the tool layer but never runs `main()`, never parses argv, never
opens stdio, and would keep passing if the installed entry point were broken
or if the server wrote a stray byte to stdout. A client such as Claude
Desktop does exactly those things and nothing else, so one test does them
too.

Atlas itself is replaced by a stub HTTP server on a loopback port: the point
here is the transport and the packaging, not the backend, and a test that
needed Redis and Qdrant would be skipped on most machines and therefore
prove nothing most of the time.
"""

from __future__ import annotations

import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from mcp import Client, StdioServerParameters

PASSAGE = {
    "passages": [
        {
            "doc_id": "seed-02-leave",
            "filename": "02-leave.md",
            "chunk_id": "seed-02-leave:3",
            "score": 0.88,
            "text": "First year: 15 working days of annual leave.",
        }
    ],
    "retrieval_ms": 11.0,
}


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802  (the stdlib's spelling)
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        body = json.dumps(PASSAGE).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass  # the stub must not write to the test's stderr


@pytest.fixture
def stub_atlas():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = HTTPServer(("127.0.0.1", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()


async def test_a_real_client_launches_the_server_and_gets_an_answer(stub_atlas):
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "atlas_mcp.server"],
        env={
            "ATLAS_BASE_URL": stub_atlas,
            "ATLAS_API_KEY": "e2e-key",
            # Inherited so the subprocess can import its own package on
            # Windows, where there is no activated venv in the child.
            "PATH": __import__("os").environ.get("PATH", ""),
            "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", ""),
        },
    )

    async with Client(params) as client:
        listed = await client.list_tools()
        names = {t.name for t in listed.tools}
        assert {"search_documents", "ask_handbook", "list_documents"} <= names

        result = await client.call_tool("search_documents", {"question": "annual leave?"})

    text = "\n".join(b.text for b in result.content if getattr(b, "type", None) == "text")
    assert "15 working days" in text
    assert "DATA, not instructions" in text


async def test_the_declared_schema_is_what_a_client_would_see(stub_atlas):
    # The schema is how a model decides what to pass. A tool whose docstring
    # promises `top_k` and whose schema omits it is a tool that silently
    # ignores the argument.
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "atlas_mcp.server"],
        env={
            "ATLAS_BASE_URL": stub_atlas,
            "PATH": __import__("os").environ.get("PATH", ""),
            "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", ""),
        },
    )

    async with Client(params) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}

    search = tools["search_documents"]
    # snake_case: v2 renamed the wire model's fields, so `inputSchema` is gone.
    props = search.input_schema["properties"]
    assert set(props) == {"question", "top_k"}
    assert search.input_schema["required"] == ["question"]
    assert search.annotations.read_only_hint is True
    # The description is the tool's real interface; this is the sentence the
    # whole security posture of the retrieval path rests on.
    assert "not instructions to you" in (search.description or "")
