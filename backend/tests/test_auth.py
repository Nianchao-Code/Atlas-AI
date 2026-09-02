"""Auth, per-principal cache isolation, and rate limiting.

The API-level cases here are the first tests in the repo that go through
FastAPI itself rather than calling a function directly, so they also cover the
wiring: a route that forgets the dependency shows up as a 200 where a 401 is
expected.
"""
from __future__ import annotations

import json

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.auth import (
    DEV_PRINCIPAL,
    enforce_rate_limit,
    key_from_headers,
    parse_api_keys,
    resolve_principal,
)
from app.config import settings
from app.obs import Cache

KEYS = "alice:secret-alice,bob:secret-bob"


# --------------------------------------------------------------- parsing ---

def test_parse_api_keys_maps_secret_to_principal():
    assert parse_api_keys(KEYS) == {"secret-alice": "alice", "secret-bob": "bob"}


def test_parse_api_keys_drops_malformed_without_losing_the_rest():
    # A typo in one entry must not take the whole service down.
    assert parse_api_keys("alice:a, ,nocolon,  bob:b ,:noprincipal,trailing:") == {
        "a": "alice",
        "b": "bob",
    }


def test_parse_api_keys_secret_may_contain_colons():
    assert parse_api_keys("svc:aa:bb:cc") == {"aa:bb:cc": "svc"}


# ------------------------------------------------------------- resolution ---

def test_no_keys_configured_means_dev_principal(monkeypatch):
    monkeypatch.setattr(settings, "api_keys", "")
    assert resolve_principal(None) == DEV_PRINCIPAL
    assert resolve_principal("anything") == DEV_PRINCIPAL


def test_configured_keys_reject_missing_and_wrong(monkeypatch):
    monkeypatch.setattr(settings, "api_keys", KEYS)
    assert resolve_principal("secret-alice") == "alice"
    assert resolve_principal("secret-bob") == "bob"
    assert resolve_principal(None) is None
    assert resolve_principal("") is None
    assert resolve_principal("secret-alic") is None
    assert resolve_principal("secret-alice ") is None


def test_key_from_headers_accepts_both_schemes():
    assert key_from_headers("k1", None) == "k1"
    assert key_from_headers(None, "Bearer k2") == "k2"
    assert key_from_headers(None, "bearer k3") == "k3"
    assert key_from_headers(None, "Basic k4") is None
    assert key_from_headers(None, "Bearer   ") is None
    assert key_from_headers("  ", "Bearer k5") == "k5"


# ------------------------------------------------------ cache isolation ---

class _FakeRedis:
    def __init__(self) -> None:
        self.kv: dict[str, str] = {}
        self.lists: dict[str, list[str]] = {}
        self.counters: dict[str, int] = {}

    async def get(self, key):
        return self.kv.get(key)

    async def set(self, key, value, ex=None):
        self.kv[key] = value

    async def lpush(self, key, value):
        self.lists.setdefault(key, []).insert(0, value)

    async def ltrim(self, key, start, end):
        self.lists[key] = self.lists.get(key, [])[start : end + 1]

    async def lrange(self, key, start, end):
        return self.lists.get(key, [])[start : end + 1]

    async def incr(self, key):
        self.counters[key] = self.counters.get(key, 0) + 1
        return self.counters[key]

    async def expire(self, key, seconds):
        return True


async def test_semantic_cache_is_scoped_per_principal():
    cache = Cache(_FakeRedis())
    await cache.set_semantic("alice", "what is the leave policy", {"answer": "15 days"})

    assert (await cache.get_semantic("alice", "what is the leave policy")) == {"answer": "15 days"}
    # bob must not be served an answer built from alice's retrievable documents
    assert (await cache.get_semantic("bob", "what is the leave policy")) is None


async def test_near_duplicate_cache_is_scoped_per_principal():
    cache = Cache(_FakeRedis())
    vec = [1.0, 0.0, 0.0]
    await cache.remember_query_vec("alice", "leave policy", vec, {"answer": "15 days"})

    assert (await cache.nearest_semantic("alice", vec, threshold=0.9)) == {"answer": "15 days"}
    assert (await cache.nearest_semantic("bob", vec, threshold=0.9)) is None


# ---------------------------------------------------------- rate limiting ---

async def test_rate_limit_trips_after_the_configured_count(monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_per_minute", 3)
    r = _FakeRedis()
    for _ in range(3):
        await enforce_rate_limit(r, "alice")
    with pytest.raises(HTTPException) as exc:
        await enforce_rate_limit(r, "alice")
    assert exc.value.status_code == 429


async def test_rate_limit_budgets_are_per_principal(monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_per_minute", 2)
    r = _FakeRedis()
    await enforce_rate_limit(r, "alice")
    await enforce_rate_limit(r, "alice")
    # alice is spent; bob still has a full budget
    await enforce_rate_limit(r, "bob")
    await enforce_rate_limit(r, "bob")
    with pytest.raises(HTTPException):
        await enforce_rate_limit(r, "bob")


async def test_rate_limit_disabled_at_zero(monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_per_minute", 0)
    r = _FakeRedis()
    for _ in range(50):
        await enforce_rate_limit(r, "alice")


# -------------------------------------------------------------- the API ---

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "api_keys", KEYS)
    monkeypatch.setattr(settings, "rate_limit_per_minute", 0)
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_health_needs_no_key_and_reports_the_mode(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["auth"] is True


@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/api/v1/documents"),
        ("get", "/api/v1/metrics"),
        ("post", "/api/v1/documents/seed"),
        ("post", "/api/v1/eval"),
        ("delete", "/api/v1/documents/does-not-matter"),
    ],
)
def test_api_routes_reject_anonymous_callers(client, method, path):
    assert getattr(client, method)(path).status_code == 401


def test_wrong_key_is_rejected(client):
    res = client.get("/api/v1/metrics", headers={"X-API-Key": "not-a-key"})
    assert res.status_code == 401


def test_valid_key_is_accepted_in_either_header(client):
    assert client.get("/api/v1/metrics", headers={"X-API-Key": "secret-alice"}).status_code == 200
    assert client.get(
        "/api/v1/metrics", headers={"Authorization": "Bearer secret-bob"}
    ).status_code == 200


# --------------------------------------------------- settings binding ---
# Every test above monkeypatches settings.api_keys directly, so none of them
# exercise the environment -> settings step. A deployment once shipped with
# ATLAS_API_KEYS set, auth silently off, and the whole suite green, because
# pydantic-settings was binding the field to API_KEYS instead.

def test_atlas_api_keys_env_var_binds_to_settings(monkeypatch):
    from app.config import Settings

    monkeypatch.setenv("ATLAS_API_KEYS", "svc:s3cret")
    assert Settings(_env_file=None).api_keys == "svc:s3cret"


def test_unprefixed_api_keys_env_var_is_not_read(monkeypatch):
    from app.config import Settings

    monkeypatch.delenv("ATLAS_API_KEYS", raising=False)
    monkeypatch.setenv("API_KEYS", "svc:s3cret")
    assert Settings(_env_file=None).api_keys == ""


def test_rate_limit_and_cors_bind_from_env(monkeypatch):
    from app.config import Settings

    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "7")
    monkeypatch.setenv("CORS_ORIGINS", "https://example.test")
    s = Settings(_env_file=None)
    assert s.rate_limit_per_minute == 7
    assert s.cors_origins == "https://example.test"
