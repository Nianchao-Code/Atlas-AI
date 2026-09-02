#!/usr/bin/env python3
"""Read an ablation's raw output and ask how much of a gap is real.

The ±sd the table prints is spread across repeats of one configuration. That
is not the same as the noise floor: two configurations can be behaviourally
identical and still disagree, because generation and the judge are both
sampled. When they do, the size of that disagreement is the smallest gap the
harness can distinguish, and any smaller gap in the table means nothing.

    python scripts/ablation_variance.py docs/ablation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def per_case(run: dict) -> dict[str, float]:
    return {c["id"]: float(c["answer_correctness"]) for c in run["cases"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument(
        "--pair",
        nargs=2,
        metavar=("A", "B"),
        help="two configuration labels to diff case by case",
    )
    args = parser.parse_args()

    raw = json.loads(args.path.read_text(encoding="utf-8"))
    labels = list(raw)
    print(f"{len(labels)} configurations: {labels}\n")

    print("within a configuration, repeat to repeat:")
    for label, runs in raw.items():
        if len(runs) < 2:
            print(f"  {label:24} only one run")
            continue
        first, *rest = [per_case(r) for r in runs]
        flipped = {cid for other in rest for cid, v in other.items() if first.get(cid) != v}
        mean = [r["mean_correctness"] for r in runs]
        print(
            f"  {label:24} correctness {min(mean):.3f}-{max(mean):.3f}"
            f"  cases that flipped: {len(flipped)}"
        )

    if args.pair:
        a, b = args.pair
        if a not in raw or b not in raw:
            print(f"\nno such labels; have {labels}")
            return 2
        ca, cb = per_case(raw[a][0]), per_case(raw[b][0])
        diff = {cid: (ca[cid], cb[cid]) for cid in ca if cid in cb and ca[cid] != cb[cid]}
        print(f"\n{a}  vs  {b}: {len(diff)} of {len(ca)} cases differ")
        for cid, (x, y) in sorted(diff.items()):
            print(f"  {cid:28} {x:.2f} -> {y:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
