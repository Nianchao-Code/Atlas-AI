from __future__ import annotations

import secrets
import time
from typing import Any

import structlog
from fastapi import HTTPException

from app.config import settings
from app.metrics import RATE_LIMITED

log = structlog.get_logger()

# Principal used when no keys are configured. Quick start and CI run this way
# on purpose: the demo should come up with one command. /health reports which
# mode is live so it is never a silent default.
DEV_PRINCIPAL = "dev"


def parse_api_keys(raw: str) -> dict[str, str]:
    """Parse "principal:secret,principal:secret" into {secret: principal}.

    Malformed entries are dropped rather than raising: a typo in one key should
    not stop the service from starting with the others.
    """
    keys: dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        principal, secret = entry.split(":", 1)
        principal, secret = principal.strip(), secret.strip()
        if principal and secret:
            keys[secret] = principal
    return keys


def auth_enabled() -> bool:
    return bool(parse_api_keys(settings.api_keys))


def key_from_headers(x_api_key: str | None, authorization: str | None) -> str | None:
    if x_api_key and x_api_key.strip():
        return x_api_key.strip()
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[len("bearer ") :].strip() or None
    return None


def resolve_principal(api_key: str | None) -> str | None:
    """Map a presented key to its principal, or None to reject."""
    keys = parse_api_keys(settings.api_keys)
    if not keys:
        return DEV_PRINCIPAL
    if not api_key:
        return None
    # Walk every candidate with compare_digest instead of a dict lookup: a
    # hash lookup short-circuits, which leaks key length and prefix in timing.
    match: str | None = None
    for secret, principal in keys.items():
        if secrets.compare_digest(secret, api_key):
            match = principal
    return match


async def enforce_rate_limit(redis_client: Any, principal: str) -> None:
    """Fixed-window counter, one Redis key per principal per minute.

    A fixed window lets a caller send up to 2x the limit across a window
    boundary. That is the accepted cost of one INCR per request; a sliding
    window needs a sorted set and a read-modify-write per call.
    """
    limit = settings.rate_limit_per_minute
    if limit <= 0:
        return
    window = int(time.time() // 60)
    key = f"rl:{principal}:{window}"
    used = await redis_client.incr(key)
    if used == 1:
        # Two windows of slack so the key outlives its own window under clock skew.
        await redis_client.expire(key, 120)
    if used > limit:
        RATE_LIMITED.labels(principal=principal).inc()
        raise HTTPException(
            status_code=429,
            detail=f"rate limit exceeded: {limit} requests/minute",
            headers={"Retry-After": "60"},
        )


def cors_origins() -> list[str]:
    return [o.strip() for o in settings.cors_origins.split(",") if o.strip()]


def log_auth_mode() -> None:
    if auth_enabled():
        principals = sorted(set(parse_api_keys(settings.api_keys).values()))
        log.info("auth.enabled", principals=principals)
    else:
        log.warning(
            "auth.disabled",
            detail="ATLAS_API_KEYS is unset; every caller is treated as the dev principal",
        )
