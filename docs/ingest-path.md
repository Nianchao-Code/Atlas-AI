# The ingest path

Every measurement above is about answering questions. Nothing had been pointed
at the other half of the service — the one endpoint where a caller supplies a
filesystem path.

**An upload could write anywhere the container user could write.** `filename`
arrives from the client and was joined straight onto the upload directory:

```python
dest = Path(settings.upload_dir) / f"{doc_id}_{filename}"
dest.write_bytes(await file.read())
```

Verified against the running pod: `filename = "../../../../app/main.py"`
resolves to `/app/app/main.py`. The image runs as a non-root uid — but that uid
owns `/app`, because the Dockerfile chowns it so the app can run there. The
hardening that stopped an upload from reaching `/etc` did nothing to stop it
reaching the application's own source, and the next restart would have executed
whatever the upload contained.

The `doc_id` prefix is what made this non-obvious. `abc123_../../x` splits into
`abc123_..`, `..`, `x` — no separator before the first `..`, so the prefix
absorbs one level and a shallow probe lands back inside the directory looking
harmless. Escaping needs three.

Fixed in `app/uploads.py`: the filename is reduced to a single path component,
and the resolved path is then checked for containment. The second check is
redundant against the first today. It is there because it is the half that
keeps holding if someone later relaxes the sanitiser — one asserts a property
of the input, the other of the result. Backslashes are folded by hand, because
on Linux they are ordinary characters: `Path("..\..\x.md").name` returns the
whole string unchanged, and a POSIX-only defence waves Windows-style payloads
through.

**Two more things the endpoint accepted.** The body was read fully into memory
before anything could object to its size, so a request larger than the pod's
2Gi limit was an OOM kill rather than a 413; it now streams to disk under a
20MiB budget and unlinks the partial file on refusal. And any suffix was
accepted and then indexed as replacement characters — an `.exe` became
retrievable chunks of mojibake. Unreadable types are refused at the door.

Every row below was a `200` before this change, and every "now" column is the
deployed behaviour, not the intended one:

| `filename` sent | before | now |
| --- | --- | --- |
| `../../../../app/main.py` | written to `/app/app/main.py` | `415` |
| `../../ESCAPED.md` | written outside `uploads/` | stored as `<id>_ESCAPED.md` |
| `..\..\..\evil.md` | backslashes kept in the name | stored as `<id>_evil.md` |
| `payload.exe` | indexed as mojibake | `415` |
| 21MB body | read into memory | `413` |
| `Employee Handbook v2.md` | accepted | accepted, indexed, answerable |

Readable types are sanitised rather than rejected: someone who names a file
oddly should get their document, not an error. The catalogue and the citations
keep the name they used; only the path on disk is rewritten.

**The same event-loop rule the query path already follows.** `delete_doc`
scrolled the whole corpus and rebuilt the sparse index inline, and uploads
parsed PDFs inline — synchronous O(corpus) work on the event loop, which is
exactly what the [background refresher](operations.md#throughput) existed to prevent, left
behind on the two endpoints that were never load-tested.

Honest about the size of it: at that corpus the stall was **4.6ms** (4.1ms
scroll, 0.4ms rebuild), and the rebuild alone 13.7ms at 1080 chunks. It is
linear in the corpus, and it is the same work that put 19s into a p95 when it
ran cold on the query path — but at 27 chunks it was never going to show up in
a load test. The reason to fix it was that the rule should hold on every path,
not that this instance was expensive.

A piece of dead work went with it: `Indexer` rebuilt a sparse index that
nothing in its own process ever searched. Both of these are now moot for a
better reason — [the sparse index moved into Qdrant](sparse-in-qdrant.md),
so there is no per-process index left to rebuild anywhere. Parsing still runs
in a thread.


---

Back to the [README](../README.md).
