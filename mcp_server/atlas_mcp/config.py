"""Settings, read once from the environment.

Kept apart from the backend's `app.config` on purpose: this process runs on a
laptop next to Claude Desktop, not in the cluster, and the two have almost no
configuration in common. The one thing they share -- the API key -- travels in
the opposite direction, so it is read here and never logged, never echoed in
an error, and never returned by a tool.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    base_url: str
    api_key: str
    timeout_s: float
    allow_writes: bool

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            base_url=os.environ.get("ATLAS_BASE_URL", "http://127.0.0.1:8000").rstrip("/"),
            api_key=os.environ.get("ATLAS_API_KEY", ""),
            # Longer than the backend's own p50 cold path (3.5s measured, p95
            # 4.6s) by enough margin that a slow model call is not reported to
            # the caller as a dead backend, and short enough that a hung
            # request does not sit in an MCP client forever.
            timeout_s=float(os.environ.get("ATLAS_TIMEOUT_S", "60")),
            # Writes and spend are off unless asked for. See docs/mcp.md: the
            # model picks tools from their descriptions, and `run_eval` costs
            # $0.055 a call at the measured rate.
            allow_writes=_flag("ATLAS_MCP_ALLOW_WRITES"),
        )


def redact(text: str, secret: str) -> str:
    """Remove a secret from text that is about to be shown to someone.

    Belt and braces. Nothing here deliberately formats the key into a message,
    but error strings are assembled from URLs and headers by libraries that
    make no such promise, and a leaked key in an MCP client's transcript is a
    leaked key.
    """
    if not secret:
        return text
    return text.replace(secret, "***")
