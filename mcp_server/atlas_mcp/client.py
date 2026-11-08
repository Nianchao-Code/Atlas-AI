"""A thin HTTP client over the Atlas API.

Thin on purpose. Every tool in this server could have imported
`app.graph.Pipeline` and called it directly, which would have been one process
instead of two -- and would have bypassed the three things the API puts in
front of that pipeline: key authentication, the per-principal rate limit, and
the per-principal cache namespace. `Pipeline.ainvoke` takes a `principal`
argument precisely so those stay true, and an in-process caller would have had
to invent one.

The cost of the extra hop is a local round trip against a measured p50 of
3.5s on the cold path. The cost of skipping it is re-implementing the parts of
the service that decide who is allowed to ask.
"""

from __future__ import annotations

from typing import Any

import httpx

from atlas_mcp.config import Settings
from atlas_mcp.errors import translate


class AtlasClient:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings
        # No default Content-Type. httpx sets `application/json` for `json=`
        # and the multipart type -- with its generated boundary -- for
        # `files=`, and a fixed default overrides the second one into a body
        # the server cannot parse.
        headers = {}
        if settings.api_key:
            headers["X-API-Key"] = settings.api_key
        self._client = httpx.AsyncClient(
            base_url=settings.base_url,
            headers=headers,
            timeout=settings.timeout_s,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, **kw: Any) -> Any:
        try:
            resp = await self._client.request(method, path, **kw)
        except httpx.RequestError as exc:
            raise translate(exc, self.settings) from exc
        if resp.status_code >= 400:
            raise translate(resp, self.settings)
        if not resp.content:
            return None
        return resp.json()

    async def health(self) -> dict:
        return await self._request("GET", "/health")

    async def query(self, question: str, use_cache: bool = True) -> dict:
        return await self._request(
            "POST", "/api/v1/query", json={"question": question, "use_cache": use_cache}
        )

    async def retrieve(self, question: str, top_k: int) -> dict:
        return await self._request(
            "POST", "/api/v1/retrieve", json={"question": question, "top_k": top_k}
        )

    async def list_documents(self, limit: int, offset: int) -> dict:
        return await self._request(
            "GET", "/api/v1/documents", params={"limit": limit, "offset": offset}
        )

    async def upload_document(self, filename: str, content: bytes) -> dict:
        # multipart, matching the API's UploadFile parameter. Content type is
        # left to the server: it decides what it will accept from the
        # extension, and a client-asserted type would only be a second
        # opinion it is right to ignore.
        return await self._request("POST", "/api/v1/documents", files={"file": (filename, content)})

    async def start_eval(self, limit: int | None) -> dict:
        params = {"limit": limit} if limit else None
        return await self._request("POST", "/api/v1/eval", params=params)

    async def get_eval(self, job_id: str) -> dict:
        return await self._request("GET", f"/api/v1/eval/{job_id}")
