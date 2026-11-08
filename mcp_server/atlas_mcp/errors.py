"""Turn transport and HTTP failures into something a model can act on.

`ToolError` is the only exception class whose message reaches the calling
model; anything else arrives as "Error executing tool <name>", which tells it
nothing and invites a retry of the identical call. So every failure here is
translated, and the translation says what to do next rather than restating
the status code.

Nothing in these messages contains the API key. `redact` is applied to text
that came from a library rather than from this module, because a URL in an
httpx error is assembled somewhere that made no such promise.
"""

from __future__ import annotations

import httpx
from mcp.server.mcpserver.exceptions import ToolError

from atlas_mcp.config import Settings, redact


def _detail(resp: httpx.Response) -> str:
    try:
        body = resp.json()
    except ValueError:
        return resp.text[:200]
    if isinstance(body, dict):
        return str(body.get("detail") or body)[:300]
    return str(body)[:300]


def translate(failure: httpx.Response | httpx.RequestError, settings: Settings) -> ToolError:
    base = settings.base_url

    if isinstance(failure, httpx.TimeoutException):
        return ToolError(
            f"Atlas did not respond within {settings.timeout_s:g}s. A cold query takes about "
            "3.5s normally; this is longer than that, so the backend is likely overloaded or "
            "stuck. Do not retry immediately."
        )
    if isinstance(failure, httpx.RequestError):
        return ToolError(
            f"Cannot reach Atlas at {base}. Check that the backend is running "
            f"(`docker compose up -d`) and that ATLAS_BASE_URL points at it. "
            f"Underlying error: {redact(str(failure), settings.api_key)[:200]}"
        )

    status, detail = failure.status_code, _detail(failure)

    if status == 401:
        return ToolError(
            "Atlas rejected the API key (401). ATLAS_API_KEY is missing or wrong. "
            "This is a configuration problem the user must fix; retrying will not help."
        )
    if status == 429:
        retry = failure.headers.get("Retry-After", "60")
        return ToolError(
            f"Rate limited by Atlas (429). The limit is per principal and this server "
            f"presents one key for all of its callers, so the budget is shared. "
            f"Wait {retry}s before trying again."
        )
    if status == 413:
        return ToolError(f"The file is too large for Atlas to accept (413). {detail}")
    if status == 415:
        return ToolError(
            f"Atlas does not index this file type (415). It accepts .md, .txt and .pdf; "
            f"anything else would be stored as replacement characters and pollute "
            f"retrieval. {detail}"
        )
    if status == 400:
        return ToolError(f"Atlas rejected the request (400). {detail}")
    if status == 404:
        return ToolError(f"Not found (404). {detail}")
    if status == 422:
        return ToolError(f"The arguments did not match what Atlas expects (422). {detail}")
    if status >= 500:
        return ToolError(
            f"Atlas failed internally ({status}). This is a backend bug, not a bad request; "
            f"the same call will fail the same way. {detail}"
        )
    return ToolError(f"Atlas returned {status}. {detail}")


def writes_disabled(tool: str) -> ToolError:
    return ToolError(
        f"`{tool}` is disabled. This server starts read-only: tools that change the corpus "
        f"or spend money are off unless the operator sets ATLAS_MCP_ALLOW_WRITES=true. "
        f"Tell the user that, rather than trying another tool to achieve the same thing."
    )
