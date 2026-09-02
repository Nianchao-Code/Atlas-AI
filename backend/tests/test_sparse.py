"""Sparse vectors are written by one process and queried by another.

Everything here is about that split: the index a term maps to has to be the
same in the worker that wrote it and the API that looks it up, forever, or
retrieval fails silently rather than loudly.
"""

from __future__ import annotations

import os
import subprocess
import sys

from app.sparse import sparse_vector, term_index, tokenize


def test_term_index_is_stable_across_processes():
    # hash() is randomised per interpreter unless PYTHONHASHSEED is pinned, so
    # a hash()-based index would give the worker and the API different vectors
    # for the same word and neither would ever know. Two fresh interpreters
    # with different seeds have to agree.
    code = "from app.sparse import term_index; print(term_index('annual'))"
    runs = {
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout.strip()
        for seed in ("0", "12345")
    }
    assert len(runs) == 1
    assert runs == {str(term_index("annual"))}


def test_term_index_fits_the_sparse_index_space():
    for token in ("annual", "leave", "kv-2025-441", "200ms", ""):
        assert 0 <= term_index(token) < 2**31


def test_values_are_term_frequencies():
    vec = sparse_vector("leave leave leave policy")
    by_index = dict(zip(vec.indices, vec.values, strict=True))
    assert by_index[term_index("leave")] == 3.0
    assert by_index[term_index("policy")] == 1.0


def test_repeated_terms_collapse_to_one_index():
    vec = sparse_vector("leave leave leave")
    assert len(vec.indices) == len(set(vec.indices)) == 1


def test_case_is_folded_so_a_query_matches_a_heading():
    assert sparse_vector("Annual Leave").indices == sparse_vector("annual leave").indices


def test_text_with_no_terms_yields_an_empty_vector():
    # search_sparse checks for this and skips the query: Qdrant rejects a
    # sparse query with no indices, and a punctuation-only rewrite is not a
    # reason to fail a request.
    for text in ("", "   ", "!!! ... ???"):
        vec = sparse_vector(text)
        assert vec.indices == []
        assert vec.values == []


def test_tokenize_mixed_language():
    toks = tokenize("K-Fleet latency 200ms")
    assert "fleet" in toks or "k" in toks
    assert "200" in toks or "200ms" in toks
    assert "latency" in toks
