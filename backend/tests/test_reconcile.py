"""Reconciling the catalogue against the vector store.

The scenario these are written from is the one that actually happened: the
Qdrant collection was rebuilt to gain a sparse vector, every point went with
it, and Redis carried on listing eight ready documents that no query could
retrieve.
"""

from __future__ import annotations

import pytest

from app.models import DocumentRecord
from app.reconcile import reconcile, source_path


class _FakeCatalog:
    def __init__(self, records: list[DocumentRecord]) -> None:
        self.records = {r.id: r for r in records}

    async def list(self) -> list[DocumentRecord]:
        return list(self.records.values())

    async def upsert(self, rec: DocumentRecord) -> None:
        self.records[rec.id] = rec


class _FakeVectors:
    def __init__(self, doc_ids: set[str]) -> None:
        self._doc_ids = set(doc_ids)
        self.deleted: list[str] = []

    def doc_ids(self) -> set[str]:
        return set(self._doc_ids)

    def delete_doc(self, doc_id: str) -> None:
        self.deleted.append(doc_id)
        self._doc_ids.discard(doc_id)


class _FakeQueue:
    def __init__(self) -> None:
        self.published: list[dict] = []

    async def publish(self, job: dict) -> None:
        self.published.append(job)


def _rec(doc_id: str, status: str = "ready", filename: str = "x.md") -> DocumentRecord:
    return DocumentRecord(id=doc_id, filename=filename, bytes=10, status=status, chunks=3)


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """A samples dir the reconciler can find a seed document in."""
    from app.config import settings

    root = tmp_path / "samples" / "corpus"
    root.mkdir(parents=True)
    (root / "02-leave.md").write_text("# Leave\n\nFifteen days.\n", encoding="utf-8")
    monkeypatch.setattr(settings, "samples_dir", str(tmp_path / "samples"))
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))
    (tmp_path / "uploads").mkdir()
    return tmp_path


async def test_agreement_changes_nothing(corpus):
    catalog = _FakeCatalog([_rec("a"), _rec("b")])
    vectors = _FakeVectors({"a", "b"})
    queue = _FakeQueue()
    report = await reconcile(catalog, vectors, queue)
    assert report.clean
    assert report.checked == 2
    assert queue.published == [] and vectors.deleted == []


async def test_a_seed_document_with_no_vectors_is_reindexed(corpus):
    # The migration case: listed as ready, retrievable by nothing, source still
    # mounted.
    rec = _rec("seed-02-leave", filename="02-leave.md")
    catalog = _FakeCatalog([rec])
    queue = _FakeQueue()
    report = await reconcile(catalog, _FakeVectors(set()), queue)

    assert report.requeued == ["seed-02-leave"]
    assert catalog.records["seed-02-leave"].status == "queued"
    assert len(queue.published) == 1
    job = queue.published[0]
    assert job["doc_id"] == "seed-02-leave"
    # The text is read here rather than left for the worker, because the worker
    # may not have the same filesystem.
    assert "Fifteen days" in job["text"]


async def test_an_unrecoverable_document_is_marked_failed(corpus):
    # An upload whose bytes died with the pod's emptyDir. Nothing can bring it
    # back, so it must stop claiming to be ready.
    rec = _rec("abc123", filename="handbook.md")
    catalog = _FakeCatalog([rec])
    queue = _FakeQueue()
    report = await reconcile(catalog, _FakeVectors(set()), queue)

    assert report.marked_failed == ["abc123"]
    assert report.requeued == []
    stored = catalog.records["abc123"]
    assert stored.status == "failed"
    assert stored.error == "source_missing"
    assert stored.chunks == 0
    assert queue.published == []


async def test_orphan_vectors_are_deleted(corpus):
    # A half-finished delete. These still match queries and displace real
    # passages, so leaving them is worse than leaving nothing.
    catalog = _FakeCatalog([_rec("a")])
    vectors = _FakeVectors({"a", "ghost"})
    report = await reconcile(catalog, vectors, _FakeQueue())

    assert report.orphans_deleted == ["ghost"]
    assert vectors.deleted == ["ghost"]


async def test_in_flight_documents_are_left_alone(corpus):
    # queued and indexing have no vectors yet by definition; repairing them
    # would fight the worker that is mid-job.
    catalog = _FakeCatalog([_rec("q", status="queued"), _rec("i", status="indexing")])
    queue = _FakeQueue()
    report = await reconcile(catalog, _FakeVectors(set()), queue)

    assert report.clean
    assert report.skipped_in_flight == 2
    assert queue.published == []


async def test_already_failed_documents_are_not_retried_forever(corpus):
    catalog = _FakeCatalog([_rec("dead", status="failed")])
    queue = _FakeQueue()
    report = await reconcile(catalog, _FakeVectors(set()), queue)
    assert report.clean
    assert queue.published == []


async def test_dry_run_reports_without_changing_anything(corpus):
    rec = _rec("seed-02-leave", filename="02-leave.md")
    catalog = _FakeCatalog([rec])
    vectors = _FakeVectors({"ghost"})
    queue = _FakeQueue()
    report = await reconcile(catalog, vectors, queue, dry_run=True)

    assert report.requeued == ["seed-02-leave"]
    assert report.orphans_deleted == ["ghost"]
    assert queue.published == []
    assert vectors.deleted == []
    assert catalog.records["seed-02-leave"].status == "ready"


def test_upload_source_is_found_under_its_sanitised_name(corpus):
    from app.config import settings
    from app.uploads import destination

    rec = _rec("abc123", filename="Employee Handbook v2.md")
    dest = destination(settings.upload_dir, rec.id, rec.filename)
    dest.write_bytes(b"# Handbook\n")
    # The record stores the name the user typed; the file is on disk under the
    # sanitised one. Reconciliation has to bridge that.
    assert source_path(rec) == dest
