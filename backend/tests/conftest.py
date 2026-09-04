"""Skip the service-backed tests when the services are not there, and say so.

Most of this suite runs against fakes and needs nothing. Two files go through
FastAPI itself, which starts the application's lifespan, which waits on Redis
and Qdrant: `await_dependency` retries thirty times at one second apiece,
deliberately, so a pod that starts before its dependencies does not crashloop.

Run without those services and the same patience becomes a minute of silence
per file with no explanation. A developer sees a hung test run, kills it, and
learns not to run the suite locally -- which costs far more than the two tests
are worth. CI has the service containers and runs them for real; here they are
skipped with the reason attached.
"""

from __future__ import annotations

import socket
from urllib.parse import urlparse

import pytest

from app.config import settings

NEEDS_SERVICES = {"test_auth.py", "test_metrics.py"}


def _reachable(url: str, default_port: int) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or default_port
    try:
        with socket.create_connection((host, port), timeout=0.4):
            return True
    except OSError:
        return False


def pytest_collection_modifyitems(config, items):
    missing = []
    if not _reachable(settings.redis_url, 6379):
        missing.append(f"redis at {settings.redis_url}")
    if not _reachable(settings.qdrant_url, 6333):
        missing.append(f"qdrant at {settings.qdrant_url}")
    if not missing:
        return

    skip = pytest.mark.skip(
        reason=(
            "needs a live backend: "
            + ", ".join(missing)
            + ". Start them with `docker compose up -d redis qdrant`, or run the "
            "rest of the suite -- everything else uses fakes."
        )
    )
    for item in items:
        if item.path.name in NEEDS_SERVICES:
            item.add_marker(skip)
