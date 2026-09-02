"""Make the Redis catalogue and the Qdrant collection agree, or say why they cannot.

The two stores can disagree, and the README has admitted so for a while without
doing anything about it. Then the sparse migration made it happen for real: the
collection had to be rebuilt to gain a sparse vector, which dropped every
point, while the catalogue went on listing eight ready documents that nothing
could retrieve. The UI showed a healthy corpus and every question abstained.

Two directions of divergence, and they are not symmetric:

  catalogue without vectors   The document is listed and unanswerable. The fix
                              is to index it again, which is possible only if
                              the source is still on disk -- uploads live in an
                              emptyDir that does not survive the pod. When it
                              is gone the record is marked failed, because a
                              document that cannot be recovered should say so
                              rather than sit at "ready" and abstain.

  vectors without a catalogue Points nobody can cite, from a delete that got
                              half done. They still match queries, so they are
                              worse than useless: they push real passages out
                              of the top k. Deleted.

Documents that are queued or indexing are skipped: those are in flight, not
divergent.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from app.chunking import parse_file
from app.config import settings
from app.metrics import RECONCILE_ACTIONS
from app.models import DocumentRecord
from app.store_docs import Catalog, IndexQueue
from app.uploads import UploadRejected, destination
from app.vectors import VectorStore

log = structlog.get_logger()

# Set by /api/v1/documents/seed, which names records after the sample file.
SEED_PREFIX = "seed-"


@dataclass
class ReconcileReport:
    checked: int = 0
    requeued: list[str] = field(default_factory=list)
    marked_failed: list[str] = field(default_factory=list)
    orphans_deleted: list[str] = field(default_factory=list)
    skipped_in_flight: int = 0

    @property
    def clean(self) -> bool:
        return not (self.requeued or self.marked_failed or self.orphans_deleted)


def source_path(rec: DocumentRecord) -> Path | None:
    """Where this document's bytes should still be, if anywhere.

    Neither location is stored on the record, and both are derivable: seeds
    come from the mounted corpus, uploads from the sanitised destination the
    upload handler chose. Deriving beats storing a path that can go stale.
    """
    if rec.id.startswith(SEED_PREFIX):
        candidates = [
            Path(settings.samples_dir) / "corpus" / rec.filename,
            Path(__file__).resolve().parents[2] / "samples" / "corpus" / rec.filename,
        ]
        return next((p for p in candidates if p.exists()), None)
    try:
        dest = destination(settings.upload_dir, rec.id, rec.filename)
    except UploadRejected:
        return None
    return dest if dest.exists() else None


async def reconcile(
    catalog: Catalog,
    vectors: VectorStore,
    queue: IndexQueue,
    *,
    dry_run: bool = False,
) -> ReconcileReport:
    report = ReconcileReport()
    records = await catalog.list()
    indexed = await asyncio.to_thread(vectors.doc_ids)
    report.checked = len(records)

    known = {rec.id for rec in records}
    for rec in records:
        if rec.status in ("queued", "indexing"):
            report.skipped_in_flight += 1
            continue
        if rec.id in indexed:
            continue
        if rec.status == "failed":
            # Already reported as broken; nothing to reconcile.
            continue

        path = source_path(rec)
        if path is None:
            report.marked_failed.append(rec.id)
            if not dry_run:
                rec.status = "failed"
                rec.chunks = 0
                rec.error = "source_missing"
                await catalog.upsert(rec)
            continue

        report.requeued.append(rec.id)
        if dry_run:
            continue
        rec.status = "queued"
        rec.chunks = 0
        rec.error = None
        await catalog.upsert(rec)
        await queue.publish(
            {
                "doc_id": rec.id,
                "filename": rec.filename,
                "path": str(path),
                "text": await asyncio.to_thread(parse_file, path),
            }
        )

    for doc_id in sorted(indexed - known):
        report.orphans_deleted.append(doc_id)
        if not dry_run:
            await asyncio.to_thread(vectors.delete_doc, doc_id)

    if not dry_run:
        for action, ids in (
            ("requeued", report.requeued),
            ("marked_failed", report.marked_failed),
            ("orphans_deleted", report.orphans_deleted),
        ):
            if ids:
                RECONCILE_ACTIONS.labels(action=action).inc(len(ids))

    if report.clean:
        log.info("reconcile.clean", checked=report.checked)
    else:
        log.warning(
            "reconcile.repaired",
            checked=report.checked,
            requeued=report.requeued,
            marked_failed=report.marked_failed,
            orphans_deleted=report.orphans_deleted,
        )
    return report
