#!/usr/bin/env python3
"""Check that every API replica retrieves the same thing.

Sparse retrieval used to live in each process's memory, rebuilt from the whole
corpus and kept in step through a revision counter in Redis. Every replica held
its own copy and could be a refresh tick behind a write. Sparse vectors now sit
in Qdrant beside the dense ones, so there is no process-local retrieval state
left to diverge -- which is a claim about replicas, and therefore needs more
than one replica to test.

It probes retrieval rather than answers, deliberately. Retrieval is what moved;
generation would only add a model's sampling to the comparison and a bill to
the run.

    kubectl scale deploy/api -n atlas --replicas=2
    python scripts/replica_check.py
    kubectl scale deploy/api -n atlas --replicas=1

Runs from the operator's machine: it needs to address each pod individually,
which is the one thing a Service is designed to stop you doing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

PROBE = r"""
import asyncio, json, sys

from app.vectors import VectorStore

question = sys.argv[1] if len(sys.argv) > 1 else "annual leave carryover"
v = VectorStore()
out = {"count": v.count(), "sparse": [h.chunk_id for h in v.search_sparse(question, 8)]}

# Dense needs the embedding API. Sparse is the half that moved, so the check
# still means something without it -- but say so rather than quietly compare
# two fields instead of four.
try:
    from app.llm import embed_texts

    vec = asyncio.run(embed_texts([question]))[0]
    out["dense"] = [h.chunk_id for h in v.search(vec, 8)]
    out["hybrid"] = [h.chunk_id for h in v.search_hybrid(vec, question, 8)]
except Exception as exc:
    out["embedding_error"] = type(exc).__name__

print("RESULT " + json.dumps(out))
"""


def run(args: list[str]) -> str:
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed:\n{proc.stderr[:400]}")
    return proc.stdout


def pods(namespace: str) -> list[str]:
    out = run(
        [
            "kubectl",
            "get",
            "pod",
            "-n",
            namespace,
            "-l",
            "app=api",
            "-o",
            "jsonpath={range .items[*]}{.metadata.name}{'\\n'}{end}",
        ]
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


def probe(namespace: str, pod: str, question: str) -> dict:
    out = run(["kubectl", "exec", "-n", namespace, pod, "--", "python", "-c", PROBE, question])
    for line in out.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :])
    raise RuntimeError(f"no result from {pod}:\n{out[-400:]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", default="atlas")
    parser.add_argument(
        "--question",
        default="annual leave carryover",
        help="anything the corpus can answer; only the ranking is compared",
    )
    args = parser.parse_args()

    names = pods(args.namespace)
    print(f"replicas: {len(names)}")
    for n in names:
        print(f"  {n}")
    if len(names) < 2:
        print("\nNeed at least two. kubectl scale deploy/api -n atlas --replicas=2")
        return 2

    results = {n: probe(args.namespace, n, args.question) for n in names}

    skipped = {n: r["embedding_error"] for n, r in results.items() if "embedding_error" in r}
    if skipped:
        print(f"  note: dense and hybrid skipped, embedding call failed ({set(skipped.values())})")

    ok = True
    fields = [
        f for f in ("count", "sparse", "dense", "hybrid") if all(f in r for r in results.values())
    ]
    for field in fields:
        values = {n: json.dumps(r[field]) for n, r in results.items()}
        agree = len(set(values.values())) == 1
        ok = ok and agree
        first = next(iter(results.values()))[field]
        detail = first if field == "count" else f"{len(first)} chunks, top={first[:2]}"
        print(f"  {field:7} {'agree' if agree else 'DISAGREE'}   {detail}")
        if not agree:
            for n, v in values.items():
                print(f"            {n}: {v}")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
