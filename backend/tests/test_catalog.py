"""The catalogue's derived keys.

`counts()` is called by the background refresher every two seconds on every
replica. Summing every record to answer it measured 0.5ms at eight documents
and 56ms at ten thousand -- 2.8% of a core, forever, recomputing a number that
only changes on a write.

So the chunk total and the name ordering are maintained on write. Both are
derived, both can drift, and drift in a counter is the kind of bug that is
invisible until someone compares it against the thing it summarises. These pin
the cases where drift would happen.
"""

from __future__ import annotations

import json

import pytest

from app.models import DocumentRecord
from app.store_docs import CHUNKS_KEY, DOCS_KEY, ORDER_KEY, SEP, Catalog


class _FakeRedis:
    """Enough Redis to run the catalogue, including its Lua scripts.

    The scripts are re-implemented here rather than executed: the point of the
    test is the bookkeeping they perform, and a fake that silently did nothing
    would pass every assertion below.
    """

    def __init__(self) -> None:
        self.h: dict[str, dict[str, str]] = {}
        self.s: dict[str, str] = {}
        self.z: dict[str, dict[str, float]] = {}

    def register_script(self, body: str):
        upsert = "ZADD" in body and "HSET" in body

        async def run(keys, args):
            docs, chunks, order = keys
            if upsert:
                doc_id, payload, new_chunks, filename = args
                delta = int(new_chunks)
                old = self.h.get(docs, {}).get(doc_id)
                if old:
                    prev = json.loads(old)
                    delta -= int(prev["chunks"])
                    self.z.get(order, {}).pop(f"{prev['filename']}{SEP}{doc_id}", None)
                self.h.setdefault(docs, {})[doc_id] = payload
                self.s[chunks] = str(int(self.s.get(chunks, "0")) + delta)
                self.z.setdefault(order, {})[f"{filename}{SEP}{doc_id}"] = 0.0
                return delta
            (doc_id,) = args
            old = self.h.get(docs, {}).pop(doc_id, None)
            if not old:
                return 0
            prev = json.loads(old)
            self.s[chunks] = str(int(self.s.get(chunks, "0")) - int(prev["chunks"]))
            self.z.get(order, {}).pop(f"{prev['filename']}{SEP}{doc_id}", None)
            return 1

        return run

    async def hget(self, key, field):
        return self.h.get(key, {}).get(field)

    async def hgetall(self, key):
        return dict(self.h.get(key, {}))

    async def hmget(self, key, fields):
        return [self.h.get(key, {}).get(f) for f in fields]

    async def hlen(self, key):
        return len(self.h.get(key, {}))

    async def get(self, key):
        return self.s.get(key)

    async def zrange(self, key, start, stop):
        members = sorted(self.z.get(key, {}))
        return members[start:] if stop == -1 else members[start : stop + 1]

    def pipeline(self):
        return _Pipe(self)


class _Pipe:
    def __init__(self, r: _FakeRedis) -> None:
        self.r = r
        self.ops: list = []

    def set(self, k, v):
        self.ops.append(("set", k, v))

    def delete(self, k):
        self.ops.append(("delete", k, None))

    def zadd(self, k, mapping):
        self.ops.append(("zadd", k, mapping))

    async def execute(self):
        for op, k, v in self.ops:
            if op == "set":
                self.r.s[k] = str(v)
            elif op == "delete":
                self.r.z.pop(k, None)
            elif op == "zadd":
                self.r.z.setdefault(k, {}).update(v)
        self.ops = []


def _rec(doc_id: str, chunks: int, filename: str | None = None) -> DocumentRecord:
    return DocumentRecord(
        id=doc_id,
        filename=filename or f"{doc_id}.md",
        bytes=100,
        status="ready",
        chunks=chunks,
    )


@pytest.fixture
def cat():
    return Catalog(_FakeRedis())


async def test_counts_does_not_read_every_record(cat):
    for i in range(5):
        await cat.upsert(_rec(f"d{i}", chunks=3))
    assert await cat.counts() == (5, 15)
    # The whole point: answered from HLEN and a counter, not from the records.
    assert cat.r.s[CHUNKS_KEY] == "15"


async def test_reindexing_a_document_adjusts_rather_than_adds(cat):
    await cat.upsert(_rec("a", chunks=10))
    await cat.upsert(_rec("a", chunks=4))
    # A naive INCRBY on every write would report 14 for one document.
    assert await cat.counts() == (1, 4)


async def test_delete_subtracts_what_the_document_held(cat):
    await cat.upsert(_rec("a", chunks=7))
    await cat.upsert(_rec("b", chunks=5))
    await cat.delete("a")
    assert await cat.counts() == (1, 5)


async def test_deleting_something_absent_changes_nothing(cat):
    await cat.upsert(_rec("a", chunks=7))
    await cat.delete("nope")
    assert await cat.counts() == (1, 7)


async def test_a_missing_counter_is_rebuilt_rather_than_reported_as_zero(cat):
    for i in range(3):
        await cat.upsert(_rec(f"d{i}", chunks=2))
    # A corpus written before the counter existed. Reporting 0 chunks would be
    # a confident wrong answer; the scan is paid once instead.
    del cat.r.s[CHUNKS_KEY]
    assert await cat.counts() == (3, 6)
    assert cat.r.s[CHUNKS_KEY] == "6"


async def test_rebuild_corrects_a_drifted_counter(cat):
    for i in range(4):
        await cat.upsert(_rec(f"d{i}", chunks=5))
    cat.r.s[CHUNKS_KEY] = "999"  # a writer died between the record and the counter
    assert await cat.rebuild_indexes() == 20
    assert await cat.counts() == (4, 20)


async def test_pages_are_filename_ordered_and_bounded(cat):
    for name in ("delta", "alpha", "charlie", "bravo"):
        await cat.upsert(_rec(name, chunks=1, filename=f"{name}.md"))

    first = await cat.list(limit=2, offset=0)
    second = await cat.list(limit=2, offset=2)
    assert [d.filename for d in first] == ["alpha.md", "bravo.md"]
    assert [d.filename for d in second] == ["charlie.md", "delta.md"]


async def test_an_offset_past_the_end_is_empty_not_an_error(cat):
    await cat.upsert(_rec("a", chunks=1))
    assert await cat.list(limit=10, offset=99) == []


async def test_renaming_a_document_leaves_one_entry_in_the_order(cat):
    # The order index keys on filename, so an update that changes it has to
    # remove the old member or the document appears twice.
    await cat.upsert(_rec("a", chunks=1, filename="old.md"))
    await cat.upsert(_rec("a", chunks=1, filename="new.md"))
    page = await cat.list(limit=10, offset=0)
    assert [d.filename for d in page] == ["new.md"]
    assert len(cat.r.z[ORDER_KEY]) == 1


async def test_unpaged_list_still_returns_everything(cat):
    for i in range(3):
        await cat.upsert(_rec(f"d{i}", chunks=1))
    # Reconciliation and the eval harness both want the whole catalogue.
    assert len(await cat.list()) == 3
    assert DOCS_KEY in cat.r.h
