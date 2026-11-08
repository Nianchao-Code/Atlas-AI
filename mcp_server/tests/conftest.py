"""Test rig: a real MCP server, a fake Atlas behind it.

Every test here goes through `Client(server)`, the SDK's in-memory client,
rather than calling the decorated functions directly. That costs nothing and
buys the part that actually breaks in an MCP server: argument coercion from
the generated schema, the shape of the result, and whether an exception
reaches the model as a message or as "Error executing tool <name>".

The HTTP layer is mocked at `httpx.MockTransport`, which is the lowest seam
that still exercises this code's own request building -- headers, multipart,
query params and status handling all run for real.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest
from mcp import Client

from atlas_mcp.config import Settings
from atlas_mcp.server import build_server

SETTINGS = Settings(
    base_url="http://atlas.test",
    api_key="test-secret-key",
    timeout_s=5.0,
    allow_writes=False,
)


def settings(**over) -> Settings:
    return Settings(**{**SETTINGS.__dict__, **over})


@pytest.fixture
def atlas():
    """Collects the requests the server made, so assertions can read them."""

    class Recorder:
        def __init__(self) -> None:
            self.requests: list[httpx.Request] = []

    return Recorder()


def make_client(
    handler: Callable[[httpx.Request], httpx.Response],
    recorder=None,
    **setting_overrides,
) -> Client:
    def wrapped(request: httpx.Request) -> httpx.Response:
        if recorder is not None:
            recorder.requests.append(request)
        return handler(request)

    server = build_server(
        settings(**setting_overrides),
        transport=httpx.MockTransport(wrapped),
    )
    return Client(server)


def json_route(routes: dict[str, tuple[int, object]]):
    """Map a URL path to (status, body). Anything unmapped is a loud 599."""

    def handler(request: httpx.Request) -> httpx.Response:
        status, body = routes.get(request.url.path, (599, {"detail": f"unmapped {request.url}"}))
        return httpx.Response(status, json=body)

    return handler


def text_of(result) -> str:
    return "\n".join(
        block.text for block in result.content if getattr(block, "type", None) == "text"
    )
