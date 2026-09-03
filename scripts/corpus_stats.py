#!/usr/bin/env python3
"""Measure whether a corpus behaves like natural language before trusting it.

A synthetic corpus can manufacture retrieval findings. Sparse retrieval scores
on IDF, which Qdrant computes across the collection, so a generator whose whole
vocabulary fits on one screen produces a term-frequency distribution unlike any
real document set -- and then "sparse retrieval wins on identifiers" would be a
fact about the generator, not about retrieval.

Three checks, and the third exists because the first two passed a corpus that
was unusable. The first generator scored Heaps beta 0.588 and hapax 42.3%, both
squarely natural, while 16.5% of its documents were leave policies copying the
real one's phrasing verbatim. Vocabulary statistics measure words; they say
nothing about how many documents are about the same thing in the same words.

Two properties of natural text are cheap to check and hard to fake by accident:

  Heaps' law   Vocabulary grows as V = K * N^beta with beta roughly 0.4-0.6 for
               natural language. A generator that recycles a small word list
               saturates: beta collapses toward 0.
  Zipf's law   Rank-frequency on a log-log plot is close to a straight line
               with slope near -1. A flat slope means no common function words;
               a steep one means a handful of tokens dominate everything.

Neither is a pass/fail oracle -- they are the reference values published for
natural language, and the point is to print the corpus's numbers next to a real
one rather than to assert the corpus is fine.

  Concentration  The largest share any single opening line holds, and the
                 closest lexical overlap between a generated document and the
                 real corpus. Both catch a generator that has quietly turned
                 into a copier.

    python scripts/corpus_stats.py samples/corpus samples/distractors
    python scripts/corpus_stats.py samples/distractors --against samples/corpus
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
from collections import Counter
from pathlib import Path

TOKEN = re.compile(r"[A-Za-z0-9_]+")


def tokens_of(paths: list[Path]) -> list[str]:
    out: list[str] = []
    for p in paths:
        out.extend(t.lower() for t in TOKEN.findall(p.read_text(encoding="utf-8")))
    return out


def heaps_beta(tokens: list[str]) -> tuple[float, int]:
    """Fit log V = log K + beta * log N over growing prefixes of the stream."""
    points: list[tuple[float, float]] = []
    seen: set[str] = set()
    step = max(1, len(tokens) // 40)
    for i, tok in enumerate(tokens, 1):
        seen.add(tok)
        if i % step == 0 and i > 100:
            points.append((math.log(i), math.log(len(seen))))
    if len(points) < 3:
        return float("nan"), len(seen)
    n = len(points)
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    num = sum((x - mx) * (y - my) for x, y in points)
    den = sum((x - mx) ** 2 for x, _ in points)
    return (num / den if den else float("nan")), len(seen)


def zipf_slope(tokens: list[str]) -> float:
    freqs = sorted(Counter(tokens).values(), reverse=True)
    pts = [(math.log(r), math.log(f)) for r, f in enumerate(freqs[:2000], 1) if f > 1]
    if len(pts) < 10:
        return float("nan")
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    num = sum((x - mx) * (y - my) for x, y in pts)
    den = sum((x - mx) ** 2 for x, _ in pts)
    return num / den if den else float("nan")


def concentration(paths: list[Path]) -> tuple[str, float]:
    """The most repeated opening line, and what share of documents carry it.

    A corpus where one clause opens a sixth of the documents is a corpus of
    copies, whatever its vocabulary statistics say.
    """
    firsts = Counter()
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                firsts[line[:60]] += 1
                break
    if not firsts:
        return "", 0.0
    line, n = firsts.most_common(1)[0]
    return line, n / len(paths)


def closest_overlap(paths: list[Path], against: list[Path]) -> tuple[float, float]:
    """Max and mean Jaccard overlap against the reference corpus.

    Verbatim reuse of the target's phrasing shows up here and nowhere else.
    """
    ref = [set(TOKEN.findall(p.read_text(encoding="utf-8").lower())) for p in against]
    best: list[float] = []
    for p in paths:
        toks = set(TOKEN.findall(p.read_text(encoding="utf-8").lower()))
        best.append(max((len(toks & r) / len(toks | r)) for r in ref) if ref and toks else 0.0)
    return max(best), sum(best) / len(best)


def report(name: str, root: Path, against: list[Path] | None = None) -> None:
    paths = sorted(root.glob("*.md"))
    if not paths:
        print(f"{name}: no .md files under {root}")
        return
    toks = tokens_of(paths)
    beta, vocab = heaps_beta(toks)
    slope = zipf_slope(toks)
    digests = Counter(
        hashlib.md5(p.read_text(encoding="utf-8").encode()).hexdigest() for p in paths
    )
    exact_dupes = sum(c - 1 for c in digests.values() if c > 1)
    counts = Counter(toks)
    hapax = sum(1 for c in counts.values() if c == 1)

    print(f"\n{name}  ({len(paths):,} documents)")
    print(f"  tokens            {len(toks):>12,}")
    print(f"  vocabulary        {vocab:>12,}")
    print(f"  type/token ratio  {vocab / len(toks):>12.4f}")
    print(f"  hapax legomena    {hapax / vocab:>11.1%}  (once-only terms; real prose ~40-60%)")
    print(f"  Heaps beta        {beta:>12.3f}  (natural language 0.4-0.6)")
    print(f"  Zipf slope        {slope:>12.3f}  (natural language near -1.0)")
    print(f"  exact duplicates  {exact_dupes:>12,}")
    line, share = concentration(paths)
    flag = "  <-- a sixth of the corpus opened alike in v1" if share > 0.10 else ""
    print(f"  top opening line  {share:>11.1%}{flag}")
    if line:
        print(f"                    {line!r}")
    if against:
        hi, avg = closest_overlap(paths, against)
        print(f"  overlap vs real   max {hi:.3f}, mean {avg:.3f}  (Jaccard; 1.0 is a copy)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument(
        "--against",
        type=Path,
        default=None,
        help="reference corpus to measure lexical overlap against",
    )
    args = ap.parse_args()
    ref = sorted(args.against.glob("*.md")) if args.against else None
    for root in args.roots:
        report(root.as_posix(), root, ref)
    print("\nThese are reference values, not a pass mark. Read them before")
    print("reading any retrieval result measured on this corpus.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
