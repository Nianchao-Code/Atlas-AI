"""Sparse term vectors, built here and scored inside Qdrant.

Sparse retrieval used to be an in-process `BM25Okapi` that every query-serving
process rebuilt from the whole corpus, kept coherent through a `bm25:rev`
counter in Redis, a background poller and an atomically swapped snapshot. All
of that machinery existed to make one piece of mutable state agree across
processes. Storing the sparse vectors next to the dense ones removes the state
instead of coordinating it.

Qdrant applies IDF from the collection's own statistics (`Modifier.IDF`), so
what this module produces is the term-frequency half of the score.
"""

from __future__ import annotations

import re
import zlib
from collections import Counter

from qdrant_client import models

_TOKEN = re.compile(r"[A-Za-z0-9_]+")

# One name, shared by the collection config and every query that touches it.
SPARSE_VECTOR_NAME = "text"


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]


def term_index(token: str) -> int:
    """Map a token to a stable sparse index.

    crc32 rather than `hash()`: Python randomises string hashing per process,
    so `hash()` would give the worker and the API different indices for the
    same word, and nothing one of them indexed would be findable by the other.
    A silent, restart-dependent retrieval failure is a bad thing to build on a
    detail of the interpreter's startup.
    """
    return zlib.crc32(token.encode("utf-8")) & 0x7FFFFFFF


def sparse_vector(text: str) -> models.SparseVector:
    """Term frequencies for `text`, keyed by hashed token.

    Values are raw counts, which makes this TF-IDF rather than BM25: BM25's
    length normalisation needs a corpus average that would have to be
    maintained somewhere and would restate every stored vector each time it
    moved -- the coordination this change exists to delete. Two measurements
    say what that costs on this corpus: 83.7% of term occurrences inside a
    chunk are singletons, so the k1 saturation term has almost nothing to act
    on, and chunk length varies with a coefficient of 0.333, so the b term is
    the one being given up. Whether that matters is a retrieval question, and
    the ablation answers it rather than this comment.
    """
    counts = Counter(term_index(t) for t in tokenize(text))
    return models.SparseVector(
        indices=list(counts.keys()),
        values=[float(v) for v in counts.values()],
    )
