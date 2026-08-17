from __future__ import annotations

import redis.asyncio as redis

from app.config import settings


def create_redis() -> redis.Redis:
    """Redis client tuned for blocking stream reads (XREADGROUP BLOCK).

    redis-py's default socket timeout is shorter than our 5s block window,
    which kills the worker while idle-waiting for index jobs.
    """
    return redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_timeout=30,
        socket_connect_timeout=10,
        health_check_interval=0,
    )
