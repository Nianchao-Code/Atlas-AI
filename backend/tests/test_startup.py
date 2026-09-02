"""Dependency waiting.

The API and worker both exited on the first refused connection. Adding a PVC to
Redis made it slower to become ready, and the next deploy crash-looped both of
them twice for no reason other than the race. A restart loop is indistinguishable
from a real fault at a glance, which is what made it worth fixing.
"""

from __future__ import annotations

import pytest

from app.startup import await_dependency


async def test_returns_immediately_when_the_probe_succeeds():
    calls = 0

    async def probe():
        nonlocal calls
        calls += 1

    await await_dependency("thing", probe, attempts=5, delay=0)
    assert calls == 1


async def test_retries_until_the_dependency_comes_up():
    calls = 0

    async def probe():
        nonlocal calls
        calls += 1
        if calls < 4:
            raise ConnectionError("connection refused")

    await await_dependency("redis", probe, attempts=10, delay=0)
    assert calls == 4


async def test_gives_up_rather_than_retrying_forever():
    """A dependency still refusing after the budget is a real failure, and
    should look like one instead of hiding behind a pod that never gets ready.
    """
    calls = 0

    async def probe():
        nonlocal calls
        calls += 1
        raise ConnectionError("connection refused")

    with pytest.raises(RuntimeError, match="unreachable after 3 attempts"):
        await await_dependency("qdrant", probe, attempts=3, delay=0)
    assert calls == 3


async def test_original_error_is_kept_as_the_cause():
    async def probe():
        raise ConnectionError("connection refused")

    with pytest.raises(RuntimeError) as exc:
        await await_dependency("redis", probe, attempts=2, delay=0)
    assert isinstance(exc.value.__cause__, ConnectionError)
