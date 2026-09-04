"""The ingest path, which had no tests at all.

Coverage said 0% for this module while it was the thing every document goes
through. What is pinned here is mostly about alignment and about not writing
vectors nobody can cite -- the two ways this can be wrong quietly. A chunk
paired with another chunk's vector produces answers that cite the wrong
document, and a document that reappears after deletion looks like the delete
failed.
"""

from __future__ import annotations

import pytest

from app.indexer import Indexer
from app.models import DocumentRecord


class _FakeCache:
    def __init__(self, prefilled: dict[str, list[float]] | None = None) -> None:
        self.store = dict(prefilled or {})
        self.reads: list[str] = []
        self.writes: list[str] = []

    async def get_embedding(self, text: str):
        self.reads.append(text)
        return self.store.get(text)

    async def set_embedding(self, text: str, vec: list[float]) -> None:
        self.writes.append(text)
        self.store[text] = vec


class _FakeVectors:
    def __init__(self) -> None:
        self.upserts: list[tuple[list, list]] = []

    def upsert(self, chunks, vectors) -> None:
        # The real one zips with strict=True; mirroring that here is what makes
        # a misalignment fail in the test rather than only in production.
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks against {len(vectors)} vectors")
        self.upserts.append((chunks, vectors))


class _FakeCatalog:
    def __init__(self, records: list[DocumentRecord] | None = None) -> None:
        self.records = {r.id: r for r in (records or [])}
        self.history: list[tuple[str, str]] = []

    async def get(self, doc_id: str):
        return self.records.get(doc_id)

    async def upsert(self, rec: DocumentRecord) -> None:
        self.records[rec.id] = rec
        self.history.append((rec.id, rec.status))


def _rec(doc_id: str = "d1") -> DocumentRecord:
    return DocumentRecord(id=doc_id, filename="d1.md", bytes=10, status="queued")


def _job(text: str) -> dict:
    return {"doc_id": "d1", "filename": "d1.md", "path": "", "text": text}


@pytest.fixture(autouse=True)
def _stub_embeddings(monkeypatch):
    calls: list[list[str]] = []

    async def fake_embed(texts):
        calls.append(list(texts))
        return [[float(len(t)), 0.0, 1.0] for t in texts]

    monkeypatch.setattr("app.indexer.embed_texts", fake_embed)
    return calls


DOC = "# Leave\n\n## Annual\n\nFirst year: 15 days.\n\n## Sick\n\nTen days per year.\n"


async def test_indexing_pairs_every_chunk_with_its_own_vector():
    cache, vectors, catalog = _FakeCache(), _FakeVectors(), _FakeCatalog([_rec()])
    n = await Indexer(cache, vectors, catalog).index_job(_job(DOC))

    chunks, vecs = vectors.upserts[0]
    assert n == len(chunks) == len(vecs)
    # The vector encodes its own text's length, so a shuffle is detectable.
    assert [v[0] for v in vecs] == [float(len(c.text)) for c in chunks]


async def test_status_goes_queued_to_indexing_to_ready():
    catalog = _FakeCatalog([_rec()])
    await Indexer(_FakeCache(), _FakeVectors(), catalog).index_job(_job(DOC))

    assert [s for _id, s in catalog.history] == ["indexing", "ready"]
    assert catalog.records["d1"].chunks > 0


async def test_a_document_deleted_while_queued_is_not_indexed():
    # The catalogue no longer lists it, so writing vectors would make a deleted
    # document reappear in search results until reconciliation notices.
    vectors, catalog = _FakeVectors(), _FakeCatalog([])
    n = await Indexer(_FakeCache(), vectors, catalog).index_job(_job(DOC))

    assert n == 0
    assert vectors.upserts == []
    assert catalog.history == []


async def test_cached_embeddings_are_not_paid_for_twice(_stub_embeddings):
    cache, catalog = _FakeCache(), _FakeCatalog([_rec()])
    await Indexer(cache, _FakeVectors(), catalog).index_job(_job(DOC))
    first_call_count = len(_stub_embeddings)

    # Same text again: every chunk is already in the cache.
    catalog2 = _FakeCatalog([_rec()])
    await Indexer(cache, _FakeVectors(), catalog2).index_job(_job(DOC))

    assert len(_stub_embeddings) == first_call_count, "re-embedded what the cache held"


async def test_a_partial_cache_hit_keeps_the_order(_stub_embeddings):
    # The dangerous case: some chunks cached, some not. The fresh vectors come
    # back in the order they were requested, not in the order of the chunks,
    # and putting them back wrongly pairs text with someone else's embedding.
    cache, vectors, catalog = _FakeCache(), _FakeVectors(), _FakeCatalog([_rec()])
    await Indexer(cache, vectors, catalog).index_job(_job(DOC))
    chunks, _ = vectors.upserts[0]

    warm = _FakeCache({chunks[1].text: [999.0, 0.0, 1.0]})
    vectors2, catalog2 = _FakeVectors(), _FakeCatalog([_rec()])
    await Indexer(warm, vectors2, catalog2).index_job(_job(DOC))

    chunks2, vecs2 = vectors2.upserts[0]
    assert vecs2[1] == [999.0, 0.0, 1.0]
    for c, v in zip(chunks2, vecs2, strict=True):
        expected = 999.0 if v[0] == 999.0 else float(len(c.text))
        assert v[0] == expected


async def test_missing_text_and_missing_file_is_an_error_not_an_empty_document():
    # Indexing nothing would mark the record ready with zero chunks, which reads
    # as a successful ingest of an empty file.
    catalog = _FakeCatalog([_rec()])
    job = {"doc_id": "d1", "filename": "d1.md", "path": "/nope/missing.md", "text": ""}

    with pytest.raises(FileNotFoundError):
        await Indexer(_FakeCache(), _FakeVectors(), catalog).index_job(job)


async def test_text_in_the_job_is_used_without_touching_the_path():
    # The worker may not share a filesystem with whoever published the job, so
    # the text travels with it.
    catalog = _FakeCatalog([_rec()])
    job = {"doc_id": "d1", "filename": "d1.md", "path": "/nope/missing.md", "text": DOC}

    assert await Indexer(_FakeCache(), _FakeVectors(), catalog).index_job(job) > 0


class _Envelope:
    def __init__(self, job: dict) -> None:
        self.job = job
        self.msg_id = job["doc_id"]


class _FakeQueue:
    """Yields a fixed set of jobs once, and records what was acked."""

    def __init__(self, jobs: list[dict]) -> None:
        self.pending = [_Envelope(j) for j in jobs]
        self.acked: list[str] = []

    async def jobs(self, consumer: str | None = None):
        for env in list(self.pending):
            yield env

    async def ack(self, envelope) -> None:
        self.acked.append(envelope.msg_id)


class _ExplodingIndexer:
    def __init__(self, fail_on: set[str]) -> None:
        self.fail_on = fail_on
        self.attempts: list[str] = []

    async def index_job(self, job: dict) -> int:
        doc_id = job["doc_id"]
        self.attempts.append(doc_id)
        if doc_id in self.fail_on:
            raise RuntimeError("cannot index this one")
        return 3


async def test_a_job_that_cannot_be_indexed_is_still_acked():
    # An unacked message stays in the consumer group's pending list and
    # XAUTOCLAIM redelivers it, so one bad document loops forever. The
    # standalone worker acked on failure and the embedded copy did not; this is
    # the behaviour that divergence cost.
    from app.indexer import consume

    queue = _FakeQueue([{"doc_id": "bad", "filename": "bad.md", "text": "x"}])
    catalog = _FakeCatalog([_rec("bad")])

    await consume(_ExplodingIndexer({"bad"}), queue, catalog, consumer="test")

    assert queue.acked == ["bad"]
    assert catalog.records["bad"].status == "failed"
    assert catalog.records["bad"].error == "index_failed"


async def test_a_successful_job_is_acked_once():
    from app.indexer import consume

    queue = _FakeQueue([{"doc_id": "ok", "filename": "ok.md", "text": "x"}])
    await consume(_ExplodingIndexer(set()), queue, _FakeCatalog([_rec("ok")]), consumer="test")

    assert queue.acked == ["ok"]


async def test_one_poisonous_job_does_not_stop_the_ones_behind_it():
    from app.indexer import consume

    jobs = [{"doc_id": d, "filename": f"{d}.md", "text": "x"} for d in ("a", "bad", "b")]
    queue = _FakeQueue(jobs)
    indexer = _ExplodingIndexer({"bad"})

    await consume(indexer, queue, _FakeCatalog([_rec("a"), _rec("bad"), _rec("b")]), consumer="t")

    assert indexer.attempts == ["a", "bad", "b"]
    assert queue.acked == ["a", "bad", "b"]


async def test_a_failure_for_an_unlisted_document_is_acked_too():
    # No catalogue record to mark failed, and nothing to retry against either.
    from app.indexer import consume

    queue = _FakeQueue([{"doc_id": "ghost", "filename": "g.md", "text": "x"}])
    await consume(_ExplodingIndexer({"ghost"}), queue, _FakeCatalog([]), consumer="test")

    assert queue.acked == ["ghost"]
