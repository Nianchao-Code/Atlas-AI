#!/usr/bin/env python3
"""Check that every link between the markdown files actually resolves.

The README was one 1019-line page until the deep dives moved into docs/. That
split turned a dozen same-page anchors into cross-file links, and a cross-file
link is the kind of thing that looks right in a diff and 404s for the reader.

Three failures, all silent in a browser until someone clicks:

  dead file      docs/foo.md does not exist
  dead anchor    the file exists, the heading does not
  dead local     a link to a source file that was moved or renamed

Run it anywhere:

    python scripts/check_links.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.M)


def slug(heading: str) -> str:
    """GitHub's heading-to-anchor rule, near enough for these documents."""
    text = re.sub(r"`([^`]*)`", r"\1", heading)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[*_]", "", text).lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"\s+", "-", text.strip())


def anchors_of(path: Path) -> set[str]:
    return {slug(h) for h in HEADING.findall(path.read_text(encoding="utf-8"))}


def main() -> int:
    files = sorted(
        [ROOT / "README.md", *(ROOT / "docs").glob("*.md"), *(ROOT / "docs/data").glob("*.md")]
    )
    anchor_cache: dict[Path, set[str]] = {}
    problems: list[str] = []
    checked = 0

    for md in files:
        if not md.exists():
            continue
        text = md.read_text(encoding="utf-8")
        here = md.relative_to(ROOT).as_posix()
        for _label, target in LINK.findall(text):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            checked += 1
            path_part, _, anchor = target.partition("#")
            anchor = unquote(anchor)

            if path_part:
                dest = (md.parent / unquote(path_part)).resolve()
                if not dest.exists():
                    problems.append(f"{here}: dead file -> {target}")
                    continue
            else:
                dest = md

            if anchor and dest.suffix == ".md":
                if dest not in anchor_cache:
                    anchor_cache[dest] = anchors_of(dest)
                if anchor not in anchor_cache[dest]:
                    problems.append(f"{here}: dead anchor -> {target}")

    print(f"checked {checked} links across {len(files)} files")
    for p in problems:
        print(f"  {p}")
    if problems:
        print(f"\n{len(problems)} broken")
        return 1
    print("all resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
