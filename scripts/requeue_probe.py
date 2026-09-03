#!/usr/bin/env python3
"""Delete a document's vectors and check that reconciliation puts them back.

`scripts/reconcile_probe.py` covers the two repairs that need no model calls:
deleting orphan vectors, and marking an unrecoverable record failed. It cannot
cover the third, because re-indexing calls the embedding API -- so that path
was asserted by unit tests and nothing else for as long as the account behind
those calls had no credits.

This is that third path, end to end and against the deployed stack. It removes
one seed document's vectors while leaving its catalogue entry claiming `ready`,
which is exactly the state a rebuilt collection leaves behind, and then checks
that the document comes back and is retrievable again.

Destructive by design and self-healing by the thing under test: if reconcile
does its job the corpus ends where it started.

    kubectl exec -n atlas deploy/worker -- python /tmp/requeue_probe.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (ROOT / "backend", Path("/app")):
    if (_candidate / "app" / "__init__.py").exists():
        sys.path.insert(0, str(_candidate))
        break

from app.vectors import VectorStore  # noqa: E402

BASE = os.environ.get("ATLAS_BASE", "http://api:8000")
_raw = os.environ.get("ATLAS_API_KEYS", "")
HEADERS = {
    "X-API-Key": _raw.split(",")[0].split(":", 1)[1].strip() if ":" in _raw else "",
    "Content-Type": "application/json",
}
TARGET = os.environ.get("REQUEUE_TARGET", "seed-02-leave")
QUESTION = "How many days of annual leave after three years?"


def call(path: str, body: dict | None = None, method: str = "GET"):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers=HEADERS, method=method)
    return json.loads(urllib.request.urlopen(req, timeout=300).read())


def status_of(doc_id: str) -> tuple[str, int]:
    rec = next((d for d in call("/api/v1/documents") if d["id"] == doc_id), None)
    return (rec["status"], rec["chunks"]) if rec else ("missing", 0)


def main() -> int:
    vectors = VectorStore()
    before_total = vectors.count()
    before_status, before_chunks = status_of(TARGET)
    print(f"start        {before_total} points, {TARGET} is {before_status}/{before_chunks} chunks")
    if before_status != "ready":
        print(f"FAIL: {TARGET} is not ready to begin with")
        return 2

    # The injury: vectors gone, catalogue still says ready. What a collection
    # rebuild leaves behind.
    vectors.delete_doc(TARGET)
    time.sleep(1)
    after_delete = vectors.count()
    injured_status, _ = status_of(TARGET)
    print(f"injured      {after_delete} points, catalogue still says {injured_status}")

    dry = call("/api/v1/documents/reconcile?dry_run=true", {}, "POST")
    print(f"dry run      requeued={dry['requeued']} marked_failed={dry['marked_failed']}")
    detected = TARGET in dry["requeued"]

    real = call("/api/v1/documents/reconcile", {}, "POST")
    print(f"reconciled   requeued={real['requeued']}")

    deadline = time.time() + 180
    status, chunks = status_of(TARGET)
    while time.time() < deadline and status not in ("ready", "failed"):
        time.sleep(3)
        status, chunks = status_of(TARGET)
    restored = vectors.count()
    print(f"reindexed    {restored} points, {TARGET} is {status}/{chunks} chunks")

    answer = call("/api/v1/query", {"question": QUESTION, "use_cache": False}, "POST")
    cited = [c["doc_id"] for c in answer.get("citations", [])]
    retrievable = TARGET in cited
    print(f"retrievable  cited={sorted(set(cited))}")
    print(f"             answer: {answer.get('answer', '')[:90]}")

    settled = call("/api/v1/documents/reconcile", {}, "POST")["clean"]
    print(f"settled      a second pass reports clean: {settled}")

    ok = (
        detected
        and TARGET in real["requeued"]
        and status == "ready"
        and restored == before_total
        and chunks == before_chunks
        and retrievable
        and settled
    )
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
