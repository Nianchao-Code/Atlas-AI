"""Collection setup and hit construction.

`ensure()` is the code that decides whether to drop a collection. It ran once
for real -- Qdrant cannot add a sparse vector to a live collection, verified
against 1.12.5, so gaining sparse retrieval meant rebuilding and every vector
went with it. That is the correct behaviour and it is also destructive, which
is a combination worth pinning: the mistake would be rebuilding a collection
that did not need it.

`_to_hit` is pinned because it reads attacker-influenced payloads and its
failure mode is a citation pointing at the wrong text.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.sparse import SPARSE_VECTOR_NAME
from app.vectors import Hit, VectorStore, _to_hit


class _Collection:
    def __init__(self, name: str) -> None:
        self.name = name


class _Collections:
    def __init__(self, names: list[str]) -> None:
        self.collections = [_Collection(n) for n in names]


class _Params:
    def __init__(self, sparse: dict | None) -> None:
        self.sparse_vectors = sparse


class _Config:
    def __init__(self, sparse: dict | None) -> None:
        self.params = _Params(sparse)


class _Info:
    def __init__(self, sparse: dict | None) -> None:
        self.config = _Config(sparse)


class _FakeQdrant:
    def __init__(self, existing: list[str], sparse: dict | None = None) -> None:
        self.existing = list(existing)
        self.sparse = sparse
        self.deleted: list[str] = []
        self.created: list[str] = []
        self.indexed: list[str] = []

    def get_collections(self):
        return _Collections(self.existing)

    def get_collection(self, name):
        return _Info(self.sparse)

    def delete_collection(self, name):
        self.deleted.append(name)
        self.existing = [n for n in self.existing if n != name]

    def create_collection(self, collection_name, **kwargs):
        self.created.append(collection_name)
        self.existing.append(collection_name)
        assert SPARSE_VECTOR_NAME in kwargs["sparse_vectors_config"], (
            "a collection created without the sparse vector would need rebuilding again"
        )

    def create_payload_index(self, collection_name, field_name, **kwargs):
        self.indexed.append(field_name)


def _store(client: _FakeQdrant) -> VectorStore:
    store = VectorStore.__new__(VectorStore)
    store.client = client  # type: ignore[assignment]
    store.collection = "atlas_chunks"
    return store


def test_a_collection_that_already_has_sparse_is_left_alone():
    client = _FakeQdrant(["atlas_chunks"], sparse={SPARSE_VECTOR_NAME: object()})
    _store(client).ensure()

    # The important half: no delete. Rebuilding here would drop a live corpus.
    assert client.deleted == []
    assert client.created == []


def test_a_collection_without_sparse_is_rebuilt():
    # Qdrant cannot add a sparse vector to a live collection, so this is the
    # only way forward -- and it drops every point, which is why it logs.
    client = _FakeQdrant(["atlas_chunks"], sparse=None)
    _store(client).ensure()

    assert client.deleted == ["atlas_chunks"]
    assert client.created == ["atlas_chunks"]


def test_an_empty_sparse_config_counts_as_missing():
    client = _FakeQdrant(["atlas_chunks"], sparse={})
    _store(client).ensure()
    assert client.deleted == ["atlas_chunks"]


def test_a_collection_under_another_name_does_not_count():
    client = _FakeQdrant(["something_else"], sparse=None)
    _store(client).ensure()

    assert client.deleted == []
    assert client.created == ["atlas_chunks"]


def test_a_fresh_collection_gets_the_doc_id_index():
    # Deletion filters on doc_id, and an unindexed keyword filter is a scan.
    client = _FakeQdrant([])
    _store(client).ensure()
    assert client.indexed == ["doc_id"]


class _Point:
    def __init__(self, payload: dict[str, Any] | None, score: float | None = 0.5) -> None:
        self.payload = payload
        self.score = score
        self.id = "point-1"


def test_a_hit_carries_the_payload_it_was_given():
    hit = _to_hit(
        _Point(
            {
                "chunk_id": "c1",
                "doc_id": "d1",
                "filename": "f.md",
                "text": "child",
                "parent_text": "parent",
                "section": "S",
            }
        ),
        "dense",
    )
    assert hit == Hit("c1", "d1", "f.md", "child", "parent", "S", 0.5, "dense")


def test_a_null_parent_falls_back_to_the_child_text():
    # A payload storing an explicit null returns None from .get with a default,
    # and None reaching a field declared str is how this broke before.
    hit = _to_hit(_Point({"chunk_id": "c", "text": "child", "parent_text": None}), "sparse")
    assert hit.parent_text == "child"


def test_a_payload_missing_everything_still_yields_a_usable_hit():
    hit = _to_hit(_Point(None), "rrf")
    assert hit.chunk_id == "point-1"  # falls back to the point id
    assert hit.parent_text == ""
    assert hit.source == "rrf"


def test_a_null_score_is_zero_rather_than_a_crash():
    assert _to_hit(_Point({"chunk_id": "c"}, score=None), "dense").score == 0.0


@pytest.mark.parametrize("source", ["dense", "sparse", "rrf"])
def test_the_source_label_is_whatever_the_caller_says(source):
    # The ablation reads this to tell which retriever produced a hit.
    assert _to_hit(_Point({"chunk_id": "c"}), source).source == source


class _FlakyClient:
    """Fails the first N calls with a given exception, then succeeds."""

    def __init__(self, failures: int, exc: Exception) -> None:
        self.remaining = failures
        self.exc = exc
        self.calls = 0

    def query_points(self, **kwargs):
        self.calls += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise self.exc

        class _Res:
            points = [_Point({"chunk_id": "c1", "text": "t"})]

        return _Res()


def _searchable(client) -> VectorStore:
    store = VectorStore.__new__(VectorStore)
    store.client = client  # type: ignore[assignment]
    store.collection = "atlas_chunks"
    return store


def test_a_stale_connection_is_retried_once():
    # This ended a 45-minute measurement mid-run. On the serving path it is a
    # failed query for a user who did nothing wrong.
    client = _FlakyClient(1, Exception("Server disconnected without sending a response."))
    hits = _searchable(client).search([0.1], k=4)

    assert client.calls == 2
    assert [h.chunk_id for h in hits] == ["c1"]


def test_a_query_the_server_rejected_is_not_retried():
    # Retrying something the server evaluated and refused turns a bug into
    # latency, and it will fail the second time for the same reason.
    client = _FlakyClient(5, ValueError("Wrong input: Not existing vector name"))
    with pytest.raises(ValueError):
        _searchable(client).search([0.1], k=4)

    assert client.calls == 1


def test_a_connection_that_stays_broken_still_fails():
    client = _FlakyClient(5, Exception("Connection reset by peer"))
    with pytest.raises(Exception, match="Connection reset"):
        _searchable(client).search([0.1], k=4)

    # One retry, not a loop: a dead backend should surface, not be waited on.
    assert client.calls == 2


def test_the_retry_covers_sparse_and_hybrid_too():
    for call, args in (
        ("search_sparse", ("annual leave", 4)),
        ("search_hybrid", ([0.1], "annual leave", 4)),
    ):
        client = _FlakyClient(1, Exception("Server disconnected without sending a response."))
        getattr(_searchable(client), call)(*args)
        assert client.calls == 2, f"{call} did not retry"
