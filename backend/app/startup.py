"""Wait for dependencies instead of exiting on the first refused connection.

Kubernetes restarts a pod that exits, so this is not about availability. It is
about not turning a few seconds of dependency startup into a crash loop: adding
a PVC to Redis made it slower to become ready, and the API and worker both
crash-looped twice on the next deploy purely because they raced it. A restart
loop looks identical to a real fault in `kubectl get pods`, which is what makes
it expensive.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import structlog

log = structlog.get_logger()


async def await_dependency(
    name: str,
    probe: Callable[[], Awaitable[object]],
    attempts: int = 30,
    delay: float = 1.0,
) -> None:
    """Poll `probe` until it succeeds, then return.

    Raises after `attempts`, because a dependency that is still refusing
    connections half a minute in is a real failure and should be visible as
    one rather than retried forever behind a healthy-looking pod.
    """
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            await probe()
            if attempt > 1:
                log.info("dependency.ready", dependency=name, attempts=attempt)
            return
        except Exception as exc:  # noqa: BLE001 - any failure is a retry
            last = exc
            if attempt == 1:
                log.info("dependency.waiting", dependency=name, error=str(exc)[:120])
            await asyncio.sleep(delay)
    raise RuntimeError(
        f"{name} was unreachable after {attempts} attempts ({attempts * delay:.0f}s)"
    ) from last
