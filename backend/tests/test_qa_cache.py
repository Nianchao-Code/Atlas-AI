"""The paraphrase cache, which was dead code before it was slow code.

Two things are worth pinning. The principal must be a `must` filter rather
than anything softer, because an answer is built from the documents its asker
could reach. And the entry must be keyed and scored on the raw question: the
previous version stored the embedding of the rewritten query plus its HyDE
paragraph, so the same question asked twice scored 0.817 against its own entry
and never hit.
"""
from __future__ import annotations

import json
import time
from typing import Any

from app.qa_cache import QACache, _point_id


class _Point:
    def __init__(self, payload: dict[str, Any], score: float = 0.99) -> None:
        self.payload = payload
        self.score = score
        self.id = "p"


class _Result:
    def __init__(self, points):
        self.points = points


class _FakeQdrant:
    """Records what it was asked, and returns what the test stages."""

    def __init__(self, points=None) -> None:
        self.points = points or []
        self.queries: list[dict] = []
        self.upserts: list[dict] = []
        self.deletes: list[Any] = []
        self.created: list[str] = []
        self.indexes: list[str] = []

    def get_collections(self):
        class _C:
            collections = []

        return _C()

    def create_collection(self, collection_name, **_kw):
        self.created.append(collection_name)

    def create_payload_index(self, collection_name, field_name, **_kw):
        self.indexes.append(field_name)

    def query_points(self, **kwargs):
        self.queries.append(kwargs)
        return _Result(self.points)

    def upsert(self, collection_name, points):
        self.upserts.append({"collection": collection_name, "points": points})

    def delete(self, collection_name, points_selector):
        self.deletes.append(points_selector)


def _cache(points=None) -> tuple[QACache, _FakeQdrant]:
    fake = _FakeQdrant(points)
    return QACache(client=fake, collection="test_qa"), fake


# ------------------------------------------------------------- point ids ---

def test_point_id_is_deterministic_and_principal_scoped():
    a = _point_id("alice", "How many leave days?")
    assert a == _point_id("alice", "  how many leave days?  ")  # normalised
    assert a != _point_id("bob", "How many leave days?")


def test_same_question_overwrites_rather_than_accumulating():
    cache, fake = _cache()
    cache.remember("alice", "How many leave days?", [0.1] * 3, {"answer": "15"})
    cache.remember("alice", "how many leave days?", [0.2] * 3, {"answer": "15"})
    ids = [p.id for u in fake.upserts for p in u["points"]]
    assert ids[0] == ids[1]


# --------------------------------------------------------------- lookups ---

def test_lookup_filters_by_principal_and_expiry():
    cache, fake = _cache()
    cache.nearest("alice", [0.1] * 3, threshold=0.82)

    q = fake.queries[0]
    assert q["limit"] == 1
    assert q["score_threshold"] == 0.82

    must = q["query_filter"].must
    keys = {c.key for c in must}
    assert keys == {"principal", "expires_at"}
    principal_cond = next(c for c in must if c.key == "principal")
    assert principal_cond.match.value == "alice"
    # Scoping has to be a filter, not a post-hoc check on the result.
    assert len(must) == 2


def test_lookup_returns_the_stored_payload():
    payload = {"answer": "15 working days", "citations": [{"filename": "02-leave.md"}]}
    cache, _ = _cache([_Point({"answer_payload": json.dumps(payload)})])
    assert cache.nearest("alice", [0.1] * 3, threshold=0.82) == payload


def test_no_match_returns_none():
    cache, _ = _cache([])
    assert cache.nearest("alice", [0.1] * 3, threshold=0.82) is None


def test_malformed_entry_is_treated_as_a_miss():
    cache, _ = _cache([_Point({})])
    assert cache.nearest("alice", [0.1] * 3, threshold=0.82) is None


# ---------------------------------------------------------------- writes ---

def test_remember_stores_principal_and_an_expiry():
    cache, fake = _cache()
    before = int(time.time())
    cache.remember("alice", "q", [0.1] * 3, {"answer": "a"}, ttl_seconds=100)

    point = fake.upserts[0]["points"][0]
    assert point.payload["principal"] == "alice"
    assert before + 100 <= point.payload["expires_at"] <= before + 101
    assert json.loads(point.payload["answer_payload"]) == {"answer": "a"}


def test_ensure_indexes_both_filtered_fields():
    cache, fake = _cache()
    cache.ensure()
    assert fake.created == ["test_qa"]
    # An unindexed filter field turns every lookup into a scan.
    assert set(fake.indexes) == {"principal", "expires_at"}


def test_purge_deletes_by_expiry_filter():
    cache, fake = _cache()
    cache.purge_expired()
    cond = fake.deletes[0].filter.must[0]
    assert cond.key == "expires_at"
    assert cond.range.lt <= int(time.time())
