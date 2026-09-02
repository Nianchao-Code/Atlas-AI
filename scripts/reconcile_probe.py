#!/usr/bin/env python3
"""Break both stores on purpose, then check the reconciler puts them back.

The catalogue lives in Redis and the vectors live in Qdrant, and nothing makes
a write to one atomic with a write to the other. The README admitted they could
disagree long before anything did something about it; then the sparse migration
made it happen for real, and the UI showed eight healthy documents while every
question abstained.

Two injuries, one per direction:

  orphan    A point whose document the catalogue no longer lists. It still
            matches queries, so it displaces real passages.
  phantom   A record listed ready whose bytes are gone -- an upload that
            outlived the emptyDir holding it.

Neither needs the embedding API: the orphan carries a vector this script makes
up, and the phantom needs no vector at all. So this runs even when the account
that pays for embeddings cannot.

    kubectl exec -n atlas deploy/worker -- python /tmp/reconcile_probe.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (ROOT / "backend", Path("/app")):
    if (_candidate / "app" / "__init__.py").exists():
        sys.path.insert(0, str(_candidate))
        break

import redis  # noqa: E402
from qdrant_client import QdrantClient, models  # noqa: E402

from app.config import settings  # noqa: E402
from app.sparse import SPARSE_VECTOR_NAME, sparse_vector  # noqa: E402

BASE = os.environ.get("ATLAS_BASE", "http://api:8000")
_raw = os.environ.get("ATLAS_API_KEYS", "")
HEADERS = {
    "X-API-Key": _raw.split(",")[0].split(":", 1)[1].strip() if ":" in _raw else "",
    "Content-Type": "application/json",
}

GHOST = "ghost-probe"
PHANTOM = "phantom-probe"
GHOST_POINT = "00000000-0000-0000-0000-0000000000ff"


def call(path: str, body: dict | None = None, method: str = "GET"):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers=HEADERS, method=method)
    return json.loads(urllib.request.urlopen(req, timeout=300).read())


def brief(report: dict) -> str:
    return json.dumps({k: v for k, v in report.items() if k not in ("checked", "dry_run")})


def main() -> int:
    client = QdrantClient(url=settings.qdrant_url, timeout=30)
    r = redis.Redis.from_url(settings.redis_url, decode_responses=True)

    before = client.count(settings.qdrant_collection, exact=True).count
    print(f"start        {before} points, {len(call('/api/v1/documents'))} catalogue records")

    text = "[source=ghost.md section=Ghost] orphaned passage nobody can cite"
    client.upsert(
        collection_name=settings.qdrant_collection,
        points=[
            models.PointStruct(
                id=GHOST_POINT,
                vector={
                    "": [0.01] * settings.embedding_dim,
                    SPARSE_VECTOR_NAME: sparse_vector(text),
                },
                payload={
                    "chunk_id": f"{GHOST}:0",
                    "doc_id": GHOST,
                    "filename": "ghost.md",
                    "text": text,
                    "parent_text": text,
                    "section": "Ghost",
                },
            )
        ],
    )
    r.hset(
        "docs",
        PHANTOM,
        json.dumps(
            {
                "id": PHANTOM,
                "filename": "vanished.md",
                "bytes": 100,
                "status": "ready",
                "chunks": 4,
                "error": None,
            }
        ),
    )
    print(f"injured      one orphan point ({GHOST}), one unretrievable record ({PHANTOM})")

    dry = call("/api/v1/documents/reconcile?dry_run=true", {}, "POST")
    print(f"dry run      {brief(dry)}")
    wrote = client.count(settings.qdrant_collection, exact=True).count != before + 1
    dry_ok = GHOST in dry["orphans_deleted"] and PHANTOM in dry["marked_failed"] and not wrote

    real = call("/api/v1/documents/reconcile", {}, "POST")
    print(f"repaired     {brief(real)}")

    after = client.count(settings.qdrant_collection, exact=True).count
    phantom = next(d for d in call("/api/v1/documents") if d["id"] == PHANTOM)
    print(f"points       {before + 1} -> {after}, expected {before}")
    print(f"phantom      status={phantom['status']} error={phantom['error']}")

    settled = call("/api/v1/documents/reconcile", {}, "POST")["clean"]
    print(f"settled      a second pass reports clean: {settled}")

    r.hdel("docs", PHANTOM)

    ok = (
        dry_ok
        and after == before
        and phantom["status"] == "failed"
        and phantom["error"] == "source_missing"
        and settled
    )
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
