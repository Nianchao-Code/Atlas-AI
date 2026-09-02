"""Calibrate against what a paraphrase cache is actually for.

The first pass used golden questions that share an answer but not an intent
("How many leave days in year one?" vs "Somebody just joined, how much time
off?"). Those are different questions that happen to land on the same fact, and
a cache is not obliged to fuse them. What it must catch is the same question
asked in different words, and what it must never fuse is two questions that
differ in one decisive word.
"""

import asyncio
import sys

sys.path.insert(0, "/app")

from app.llm import embed_texts
from app.obs import _cosine

# Same question, reworded. A hit here is the whole point of the feature.
POSITIVE = [
    (
        "How many annual leave days in the first year?",
        "How many days of annual leave do I get in my first year?",
    ),
    (
        "How many annual leave days in the first year?",
        "What is the first-year annual leave entitlement?",
    ),
    ("What is the weekly on-call stipend?", "How much is the on-call stipend per week?"),
    (
        "About how many full-time employees does Kepler have?",
        "Roughly how many full-time staff does Kepler employ?",
    ),
    (
        "What is the latest a formal SEV-1 RCA may go out?",
        "What is the deadline for publishing the formal SEV-1 RCA?",
    ),
    (
        "How long are retrieval and answer logs kept for hallucination audits?",
        "What is the retention period for retrieval and answer logs?",
    ),
    (
        "May L3 data be sent to an external LLM?",
        "Is it allowed to send L3 data to an external LLM?",
    ),
    (
        "What is the daily cross-timezone on-call handoff window?",
        "When is the daily on-call handoff between timezones?",
    ),
    (
        "By how much may the backup reducer supplier unit price rise?",
        "What is the maximum price increase allowed for the backup supplier?",
    ),
]

# One decisive word apart. A hit here serves a confidently wrong answer.
NEGATIVE = [
    (
        "How many annual leave days in the first year?",
        "How many annual leave days after three years of tenure?",
    ),
    (
        "Is Seattle sick leave also capped at 10 days per year?",
        "Exactly how many paid sick days per year does a Seattle employee receive?",
    ),
    (
        "How long are retrieval and answer logs kept for hallucination audits?",
        "How long does the production Q&A model gateway retain its logs?",
    ),
    (
        "How many minutes into a SEV-1 must the first written status go out?",
        "What is the latest a formal SEV-1 RCA may go out?",
    ),
    (
        "What is the K-Arm end-effector repeatability?",
        "What is the K-Arm rated payload in kilograms?",
    ),
    (
        "How far in advance must leave attached to a public holiday be submitted?",
        "When does sick leave require a hospital note?",
    ),
    (
        "What are Kitagawa Precision payment terms, and at what defect rate can a lot be rejected?",
        "By how much may the backup reducer supplier unit price rise?",
    ),
]


async def main():
    texts = sorted({t for pair in POSITIVE + NEGATIVE for t in pair})
    vecs = dict(zip(texts, await embed_texts(texts), strict=True))

    print("SAME question, reworded -- a hit is correct:")
    pos = []
    for a, b in POSITIVE:
        s = _cosine(vecs[a], vecs[b])
        pos.append(s)
        print(f"  {s:.4f}  {b[:62]}")

    print()
    print("ONE decisive word apart -- a hit is a wrong answer:")
    neg = []
    for a, b in NEGATIVE:
        s = _cosine(vecs[a], vecs[b])
        neg.append(s)
        print(f"  {s:.4f}  {a[:40]!r} vs {b[:40]!r}")

    print()
    print(f"positives: min={min(pos):.4f}  max={max(pos):.4f}")
    print(f"negatives: min={min(neg):.4f}  max={max(neg):.4f}")
    print()
    if max(neg) < min(pos):
        lo, hi = max(neg), min(pos)
        print(f"SEPARABLE. Safe threshold band: ({lo:.4f}, {hi:.4f})")
        print(f"  midpoint {(lo + hi) / 2:.4f}")
    else:
        print(f"OVERLAP: worst wrong pair {max(neg):.4f} >= best-case cutoff {min(pos):.4f}")
        for t in (0.80, 0.85, 0.88, 0.90, 0.92, 0.95):
            tp = sum(1 for s in pos if s >= t)
            fp = sum(1 for s in neg if s >= t)
            print(f"  threshold {t:.2f}: catches {tp}/{len(pos)} paraphrases, {fp} wrong hit(s)")


asyncio.run(main())
