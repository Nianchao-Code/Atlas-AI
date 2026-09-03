#!/usr/bin/env python3
"""Generate a realistic distractor corpus around the Kepler handbook.

Why generate rather than download. The golden set asks 53 questions about eight
real handbook documents, and with eight documents to choose from every retriever
scores recall 1.000. Recall cannot discriminate anything at that corpus size,
which makes it a formality rather than a metric. Scaling fixes that.

This is the second generator, and the first one's failure is worth recording
because neither check in place caught it.

It shaped every distractor after the real documents -- same section headings,
same opening clauses -- on the theory that distractors should be hard. The
result was 1,649 leave policies, 16.5% of the corpus, all beginning "Annual
leave - First year of employment", identical to the real one. Dense retrieval
then returned 24 of 24 distractors for the leave questions, top scores spanning
0.7942 to 0.7903: a 0.3% spread across documents that are the same sentence.
That reads like "dense retrieval collapses at scale" and actually says "there
was nothing to discriminate". A fact about the generator, not about retrieval.

The vocabulary statistics were healthy throughout -- Heaps beta 0.588, hapax
42.3%, both inside the natural band -- because they measure words, not semantic
concentration. Necessary and insufficient.

So this one is built to be realistic rather than maximally hard:

  topic mix    Twenty-two document kinds spread evenly, so no single topic is
               more than a few percent. A real company's ten thousand documents
               are mostly not leave policies.
  collisions   A small deliberate slice (--collide-rate, 1% by default) shares
               the handbook's subject matter: other organisations' leave
               entitlements, on-call stipends, retention windows, IR-style
               severities, KV-style ticket ids. Few and pointed.
  phrasing     Several openings and section shapes per kind, so two documents
               on one topic do not begin with the same clause.
  near-dups    Two percent are one figure apart from a neighbour, which is the
               case the golden set's `distractor` questions are about.
  shared words The second generator got the topic mix right and still flattered
               sparse retrieval to a perfect 53/53. It gave every distractor a
               different fictional company, so 84 of the 102 terms occurring
               twice or more in the real handbook -- kepler, seattle, mercer,
               stipend -- appeared in no distractor at all. IDF then made every
               golden question trivially separable by its proper nouns. A real
               company's corpus mentions the company in most of its documents,
               so --share-rate of these documents borrow the handbook's
               vocabulary: its name, its sites, its products, its ticket
               prefixes. Vocabulary only. Copying its phrasing is what the
               first generator did.

Vocabulary comes from one lexicon sampled at 1/rank. Tiering it -- boilerplate
everywhere, topic words in every sixth document, names once -- creates frequency
plateaus, and fitting a line through a staircase gave a Zipf slope near -1.9
against -0.86 measured on real prose with the same estimator. Six variants of
the tiered model were measured and none moved it; a single pool measures -0.99.

    python scripts/generate_corpus.py --docs 10000 --out samples/distractors
    python scripts/corpus_stats.py samples/corpus samples/distractors
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

ONSET = [
    "b",
    "d",
    "f",
    "g",
    "h",
    "k",
    "l",
    "m",
    "n",
    "p",
    "r",
    "s",
    "t",
    "v",
    "w",
    "z",
    "br",
    "cl",
    "dr",
    "fl",
    "gr",
    "kr",
    "pl",
    "st",
    "th",
    "tr",
    "vr",
    "sk",
    "sp",
    "sw",
    "ch",
    "sh",
    "ph",
    "qu",
]
NUCLEUS = ["a", "e", "i", "o", "u", "ai", "ea", "ie", "oa", "ou", "au", "ei", "ee", "oo"]
CODA = [
    "",
    "n",
    "r",
    "l",
    "m",
    "s",
    "k",
    "t",
    "d",
    "ng",
    "rk",
    "st",
    "lm",
    "nd",
    "ll",
    "ss",
    "rn",
    "ft",
]

# Real words at the frequent head so the prose reads like company documents;
# synthesised names form the tail so identifiers behave like identifiers
# under IDF.
PROSE = [
    "review",
    "approve",
    "escalate",
    "document",
    "record",
    "submit",
    "verify",
    "audit",
    "confirm",
    "schedule",
    "assign",
    "transfer",
    "reconcile",
    "publish",
    "archive",
    "retain",
    "disclose",
    "notify",
    "authorise",
    "delegate",
    "inspect",
    "calibrate",
    "replace",
    "decommission",
    "migrate",
    "provision",
    "deprecate",
    "monitor",
    "threshold",
    "baseline",
    "exception",
    "waiver",
    "quorum",
    "custodian",
    "steward",
    "attestation",
    "remediation",
    "mitigation",
    "dependency",
    "prerequisite",
    "obligation",
    "liability",
    "indemnity",
    "warranty",
    "tolerance",
    "variance",
    "deviation",
    "nonconformance",
    "corrective",
    "preventive",
    "systemic",
    "recurring",
    "intermittent",
    "transient",
    "degraded",
    "nominal",
    "elevated",
    "critical",
    "routine",
    "quarterly",
    "annual",
    "monthly",
    "interim",
    "provisional",
    "binding",
    "advisory",
    "mandatory",
    "discretionary",
    "confidential",
    "restricted",
    "anonymised",
    "aggregated",
    "derived",
    "upstream",
    "downstream",
    "inbound",
    "outbound",
    "residual",
    "cumulative",
    "marginal",
    "retrospective",
    "prospective",
    "contractual",
    "regulatory",
    "statutory",
    "procedural",
    "operational",
    "logistical",
    "territorial",
    "jurisdictional",
    "reciprocal",
    "unilateral",
    "conditional",
    "supplementary",
    "ancillary",
    "preliminary",
    "definitive",
    "outstanding",
    "overdue",
    "reconciled",
    "deferred",
    "amortised",
    "accrued",
    "escalated",
    "withdrawn",
    "superseded",
    "rescinded",
    "ratified",
    "endorsed",
    "contested",
    "arbitrated",
    "backlog",
    "capacity",
    "throughput",
    "latency",
    "coverage",
    "rollout",
    "rollback",
    "staging",
    "canary",
    "quota",
    "budget",
    "forecast",
    "headcount",
    "attrition",
    "onboarding",
    "offboarding",
    "procurement",
    "inventory",
    "shipment",
    "calibration",
    "firmware",
    "schema",
    "partition",
    "replica",
    "checkpoint",
    "snapshot",
    "migration",
    "index",
    "cursor",
    "batch",
    "queue",
    "consumer",
    "producer",
]

UNIT = [
    "Platform",
    "Reliability",
    "Fulfilment",
    "Procurement",
    "Actuarial",
    "Field-Service",
    "Instrumentation",
    "Metrology",
    "Compliance",
    "Payroll",
    "Facilities",
    "Chemistry",
    "Firmware",
    "Packaging",
    "Validation",
    "Warehouse",
    "Calibration",
    "Tooling",
    "Radiology",
    "Underwriting",
    "Localisation",
    "Accessibility",
    "Sustainability",
    "Partnerships",
]
GIVEN = [
    "Amara",
    "Bao",
    "Cleo",
    "Devi",
    "Elias",
    "Farid",
    "Greta",
    "Hana",
    "Ines",
    "Jonas",
    "Kaito",
    "Lucia",
    "Mateo",
    "Nadia",
    "Omar",
    "Priya",
    "Quinn",
    "Rafael",
    "Sena",
    "Tobias",
    "Uma",
    "Viktor",
    "Wren",
    "Xiulan",
    "Yusuf",
    "Zofia",
    "Anders",
    "Bianca",
    "Cormac",
    "Delphine",
]
ORG_TAIL = [
    "Robotics",
    "Dynamics",
    "Systems",
    "Logistics",
    "Analytics",
    "Instruments",
    "Networks",
    "Materials",
    "Automation",
    "Avionics",
    "Biotech",
    "Ceramics",
    "Diagnostics",
    "Energy",
    "Geospatial",
    "Hydraulics",
    "Imaging",
    "Kinetics",
    "Laboratories",
    "Optics",
]


def shared_terms(root: Path) -> tuple[list[str], list[str]]:
    """Proper nouns and domain words the real corpus actually uses.

    Vocabulary, never phrasing. The point is that a distractor can say "Kepler"
    and "Seattle" the way a real internal document would, without reproducing
    the sentence those words appear in.
    """
    if not root.exists():
        return [], []
    text = " ".join(p.read_text(encoding="utf-8") for p in sorted(root.glob("*.md")))
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text)
    counts = Counter(w.lower() for w in words)
    # A capitalised word is not a proper noun just because it starts a heading.
    # Counting capitalisation alone ranked Leave, Customer, Are, For and
    # Internal above Kepler, so the company name reached only 11% of documents.
    # A real proper noun is almost never written lowercase; an ordinary word
    # that happens to open a line is written both ways.
    upper = Counter(w for w in words if w[0].isupper())
    lower = Counter(w for w in words if not w[0].isupper())
    seen_proper = {
        w
        for w, n in upper.items()
        if n >= 2 and lower.get(w.lower(), 0) <= n * 0.25 and w.lower() not in _STOPISH
    }
    proper = sorted(seen_proper, key=lambda w: (-upper[w], w))
    domain = sorted(
        {w.lower() for w in words if not w[0].isupper() and counts[w.lower()] >= 3 and len(w) > 4}
    )
    return proper, domain


_STOPISH = {"the", "this", "that", "these", "those", "there", "when", "where", "after", "with"}
SHARED_PROPER: list[str] = []


class Lexicon:
    """One pool sampled at 1/rank: common words recur, the tail stays rare."""

    def __init__(self, rng: random.Random, size: int, head: list[str]) -> None:
        self.words = list(head)
        seen = set(self.words)
        tail: list[str] = []
        while len(seen) < size:
            w = self._coin(rng)
            if w not in seen:
                seen.add(w)
                tail.append(w)
        rng.shuffle(tail)
        self.words += tail
        total = 0.0
        self.cum: list[float] = []
        for i in range(len(self.words)):
            total += 1.0 / (i + 1)
            self.cum.append(total)
        self.total = total

    @staticmethod
    def _coin(rng: random.Random) -> str:
        return "".join(
            rng.choice(ONSET) + rng.choice(NUCLEUS) + rng.choice(CODA)
            for _ in range(rng.choice((2, 2, 3)))
        ).capitalize()

    def pick(self, rng: random.Random) -> str:
        r = rng.random() * self.total
        lo, hi = 0, len(self.cum) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self.cum[mid] < r:
                lo = mid + 1
            else:
                hi = mid
        return self.words[lo]


LEX: Lexicon


def org(rng: random.Random, share_rate: float = 0.0) -> str:
    """Most documents in a company's corpus name that company.

    Giving every distractor a fictional org is what let sparse retrieval score
    a perfect 53/53: the golden questions' proper nouns existed nowhere else.
    """
    if SHARED_PROPER and rng.random() < share_rate:
        # 1/rank over the frequency-ordered list: the top name carries most
        # documents, exactly as a company's own name does in its own corpus.
        weights = [1.0 / (i + 1) for i in range(len(SHARED_PROPER))]
        head = rng.choices(SHARED_PROPER, weights=weights, k=1)[0]
        tail = rng.choices(SHARED_PROPER, weights=weights, k=1)[0]
        return head if head == tail else f"{head} {tail}"
    return f"{LEX.pick(rng)} {rng.choice(ORG_TAIL)}"


def person(rng: random.Random) -> str:
    return f"{rng.choice(GIVEN)} {LEX.pick(rng)}"


def ref(rng: random.Random) -> str:
    letters = rng.choice("BCDFGHJLMNPRSTVWXZ") + rng.choice("AEIOU")
    return f"{letters}-{rng.randint(2019, 2026)}-{rng.randint(100, 999)}"


# The first five can collide with the handbook's subject matter and are rationed
# by --collide-rate. The rest are ordinary company documents about something
# else, which is what a real corpus mostly is.
COLLIDING = ["leave", "oncall", "retention", "incident", "vendor"]
GENERAL = [
    "meeting",
    "status",
    "spec",
    "adr",
    "release",
    "testplan",
    "postmortem",
    "budget",
    "training",
    "role",
    "brief",
    "contract",
    "research",
    "schema",
    "evaluation",
    "facilities",
    "audit",
]

OPENINGS = [
    "Prepared by {who} for the {unit} group at {org}.",
    "{org} internal. Owner: {unit}. Contact {who}.",
    "This note supersedes the revision previously held by {unit} at {org}.",
    "Circulated to {unit} leads across {org}; comments to {who}.",
    "Filed under {ref} by {who}, {unit}, {org}.",
    "Working document. {unit} at {org} maintains it and {who} signs it off.",
    "Distribution limited to {unit}. Raise questions against {ref}.",
]

SECTIONS = {
    "meeting": ["Attendees", "Decisions", "Actions", "Parked", "Next steps"],
    "status": ["Progress", "Risks", "Blockers", "Forecast", "Dependencies"],
    "spec": ["Scope", "Interfaces", "Constraints", "Non-goals", "Open questions"],
    "adr": ["Context", "Decision", "Consequences", "Alternatives", "Status"],
    "release": ["Included", "Known issues", "Rollout", "Rollback", "Verification"],
    "testplan": ["Coverage", "Environments", "Entry criteria", "Exit criteria", "Fixtures"],
    "postmortem": ["What happened", "Contributing factors", "What went well", "Follow-ups"],
    "budget": ["Allocation", "Variance", "Commitments", "Forecast", "Approvals"],
    "training": ["Objectives", "Modules", "Assessment", "Prerequisites", "Refresher"],
    "role": ["Responsibilities", "Requirements", "Team", "Progression", "Location"],
    "brief": ["Audience", "Message", "Channels", "Measurement", "Timeline"],
    "contract": ["Parties", "Deliverables", "Milestones", "Change control", "Exit"],
    "research": ["Question", "Method", "Findings", "Threats to validity", "Next"],
    "schema": ["Entities", "Keys", "Indexes", "Retention", "Migration"],
    "evaluation": ["Candidates", "Criteria", "Scoring", "Recommendation", "Caveats"],
    "facilities": ["Access", "Hours", "Works", "Safety", "Contacts"],
    "audit": ["Scope", "Sampling", "Observations", "Severity", "Response"],
    "leave": ["Entitlement", "Accrual", "Approval", "Carryover", "Absence"],
    "oncall": ["Rotation", "Compensation", "Response", "Escalation", "Handover"],
    "retention": ["Windows", "Classification", "Transfers", "Disposal", "Audit"],
    "incident": ["Summary", "Classification", "Timeline", "Actions", "Review"],
    "vendor": ["Payment", "Service levels", "Security", "Termination", "Contacts"],
}

LINES = [
    "- {w} {w2}: **{n}** {unit_word} per {period}",
    "- The {unit} owner will {w} the {w2} within {n} {unit_word}",
    "- {w} sits {w2} above {n}%, reviewed each {period}",
    "- Threshold {n} {unit_word}; beyond that escalate to {who}",
    "- {w2} records are kept {n} {unit_word} under {ref}",
    "- Sign-off from {unit} is required when {w} exceeds {n}",
    "{who} confirmed the {w} {w2} on behalf of {unit}.",
    "The {w} remains {w2} until {unit} publishes a revision.",
    "Where {w} conflicts with {w2}, the {unit} position takes precedence.",
    "No {w2} is implied by this {w}; {ref} carries the binding text.",
    "{unit} reviews the {w} every {period} and records the outcome in {ref}.",
]
UNIT_WORDS = ["days", "weeks", "months", "hours", "minutes", "units", "percent", "items", "records"]
PERIODS = ["year", "quarter", "month", "cycle", "rotation", "release"]


def render(rng: random.Random, kind: str, share_rate: float = 0.0) -> str:
    heads = rng.sample(SECTIONS[kind], rng.randint(2, 4))

    def word() -> str:
        # A borrowed noun sits in the same slots an invented one would, so the
        # handbook's vocabulary appears without its sentences.
        if SHARED_PROPER and rng.random() < share_rate * 0.25:
            return rng.choice(SHARED_PROPER)
        return LEX.pick(rng)

    out = [f"# {kind.capitalize()} - {word()} {word()}", ""]
    out.append(
        rng.choice(OPENINGS).format(
            who=person(rng), unit=rng.choice(UNIT), org=org(rng, share_rate), ref=ref(rng)
        )
    )
    for h in heads:
        out += ["", f"## {h}", ""]
        for _ in range(rng.randint(3, 6)):
            out.append(
                rng.choice(LINES).format(
                    w=word(),
                    w2=word(),
                    n=rng.randint(1, 400),
                    unit_word=rng.choice(UNIT_WORDS),
                    period=rng.choice(PERIODS),
                    unit=rng.choice(UNIT),
                    who=person(rng),
                    ref=ref(rng),
                )
            )
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=10_000)
    ap.add_argument("--out", type=Path, default=Path("samples/distractors"))
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--name-pool", type=int, default=0, help="0 scales with --docs")
    ap.add_argument(
        "--collide-rate",
        type=float,
        default=0.01,
        help="share sharing the handbook's subject matter; the first generator "
        "reached 0.165 by accident and made dense retrieval meaningless",
    )
    ap.add_argument("--near-dup-rate", type=float, default=0.02)
    ap.add_argument(
        "--shared-from",
        type=Path,
        default=Path("samples/corpus"),
        help="real corpus whose vocabulary the distractors borrow (not its phrasing)",
    )
    ap.add_argument(
        "--share-rate",
        type=float,
        default=0.7,
        help="share of documents naming the real organisation, as a company's own corpus does",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    global LEX
    # Four lexicon words per document, swept rather than picked: x10 leaves the
    # tail too fat (beta 0.676), x2 exhausts it (hapax 30.9%).
    pool_size = args.name_pool or max(20_000, args.docs * 4)
    proper, domain = shared_terms(args.shared_from)
    global SHARED_PROPER
    SHARED_PROPER = proper
    # Domain words join the frequent head, which is where they sit in a real
    # corpus: common, low-IDF, and useless for telling documents apart.
    LEX = Lexicon(random.Random(args.seed ^ 0x5EED), pool_size, list(PROSE) + domain)

    args.out.mkdir(parents=True, exist_ok=True)
    for old in args.out.glob("*.md"):
        old.unlink()

    width = len(str(args.docs - 1))
    kinds: dict[str, int] = {}
    previous: str | None = None

    for i in range(args.docs):
        if previous and rng.random() < args.near_dup_rate:
            body = previous
            digits = [w for w in body.split() if w.strip("*-.,%:").isdigit()]
            if digits:
                d = rng.choice(digits).strip("*-.,%:")
                body = body.replace(d, str(int(d) + rng.choice([-2, -1, 1, 2, 5])), 1)
            kind = "near-dup"
        else:
            kind = (
                rng.choice(COLLIDING) if rng.random() < args.collide_rate else rng.choice(GENERAL)
            )
            body = render(rng, kind, args.share_rate)
            previous = body
        kinds[kind] = kinds.get(kind, 0) + 1
        (args.out / f"d{i:0{width}d}.md").write_text(body, encoding="utf-8")

    manifest = args.out.parent / "distractors_manifest.json"
    manifest.write_text(
        json.dumps({"docs": args.docs, "seed": args.seed, "kinds": kinds}, indent=2),
        encoding="utf-8",
    )
    words = sum(len(p.read_text(encoding="utf-8").split()) for p in args.out.glob("*.md"))
    collide = sum(v for k, v in kinds.items() if k in COLLIDING)
    biggest = max(kinds.items(), key=lambda kv: kv[1])
    print(f"wrote {args.docs:,} documents to {args.out} (lexicon {pool_size:,})")
    print(f"  {words:,} words, {words / args.docs:.0f} per document")
    print(f"  colliding with handbook topics: {collide:,} ({collide / args.docs:.1%})")
    print(f"  largest single kind: {biggest[0]} {biggest[1]:,} ({biggest[1] / args.docs:.1%})")
    print(f"  borrowed vocabulary: {len(proper)} proper nouns, {len(domain)} domain words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
