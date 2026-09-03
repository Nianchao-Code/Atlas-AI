# Keeping Redis and Qdrant in agreement

The catalogue lives in Redis and the vectors live in Qdrant, and nothing makes
a write to one atomic with a write to the other. That was in [Limits](limits.md)
for a long time as a known gap with nothing done about it. Then
[the sparse migration](sparse-in-qdrant.md) made it real: the
collection had to be rebuilt to gain a sparse vector, which dropped every
point, while Redis went on listing eight ready documents. **The UI showed a
healthy corpus and every question abstained.** A gap you have written down is
still a gap.

The two directions are not symmetric, and neither repair is "delete the other
side":

| | What it is | Repair |
| --- | --- | --- |
| Catalogue without vectors | Listed, unanswerable | Index it again — if the source is still on disk |
| ...and the source is gone | An upload that outlived the `emptyDir` holding it | Mark it `failed`, so it stops claiming to be ready |
| Vectors without a catalogue | A delete that got half done | Delete the points |

The middle row is the one worth arguing about. Marking a document failed is a
worse-looking outcome than leaving it alone, and it is the right one: a record
that says `ready` while every question about it abstains is a lie the UI
repeats. Documents that are `queued` or `indexing` are skipped — those have no
vectors *yet*, which is not the same thing.

Reconciliation runs at startup (that is when a collection gets rebuilt, so that
is when the stores are most likely to disagree), on a slow timer, and on
`POST /api/v1/documents/reconcile`, which takes `?dry_run=true` and reports
without touching anything. There is no fast timer on purpose: the sweep scrolls
the whole collection, which is the shape of work the last change went to some
trouble to get off the hot path. Divergence comes from events — a failed
ingest, a migration — not from drift.

**Verified by causing it.** `scripts/reconcile_probe.py` injures both stores on
purpose and checks the repair. Neither injury needs the embedding API — the
orphan carries a made-up vector, the phantom record needs no vector at all — so
it runs against the deployed stack regardless of what the model account is
doing:

```
start        27 points, 8 catalogue records
injured      one orphan point (ghost-probe), one unretrievable record (phantom-probe)
dry run      {"requeued": [], "marked_failed": ["phantom-probe"], "orphans_deleted": ["ghost-probe"], "clean": false}
repaired     {"requeued": [], "marked_failed": ["phantom-probe"], "orphans_deleted": ["ghost-probe"], "clean": false}
points       28 -> 27, expected 27
phantom      status=failed error=source_missing
settled      a second pass reports clean: True
PASS
```

The third repair — re-index a document whose vectors are gone — needs the
embedding API, so it has its own probe. `scripts/requeue_probe.py` deletes one
seed document's vectors while leaving its record claiming `ready`, which is
exactly the state a rebuilt collection leaves behind:

```
start        27 points, seed-02-leave is ready/5 chunks
injured      22 points, catalogue still says ready
dry run      requeued=['seed-02-leave'] marked_failed=[]
reconciled   requeued=['seed-02-leave']
reindexed    27 points, seed-02-leave is ready/5 chunks
retrievable  cited=['seed-02-leave']
settled      a second pass reports clean: True
PASS
```

Destructive by design and self-healing by the thing under test: the corpus ends
where it started, and the repaired document is cited in an answer rather than
merely counted.


---

Back to the [README](../README.md).
