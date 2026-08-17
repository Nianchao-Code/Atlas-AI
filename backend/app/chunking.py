from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

_HEADING = re.compile(r"^(#{1,3})\s+(.+)$", re.M)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    filename: str
    text: str
    parent_text: str
    section: str
    index: int
    metadata: dict = field(default_factory=dict)


def parse_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            pages.append(f"## Page {i}\n{text}")
        return "\n\n".join(pages)
    return path.read_text(encoding="utf-8", errors="replace")


def _split_windows(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    out: list[str] = []
    i = 0
    step = max(1, size - overlap)
    while i < len(words):
        window = words[i : i + size]
        out.append(" ".join(window))
        if i + size >= len(words):
            break
        i += step
    return out


def _sections(text: str) -> list[tuple[str, str]]:
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [("Document", text.strip())]
    sections: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        lead = text[: matches[0].start()].strip()
        if lead:
            sections.append(("Preamble", lead))
    for i, m in enumerate(matches):
        title = m.group(2).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end() : end].strip()
        sections.append((title, f"{title}\n{body}"))
    return sections


def chunk_document(
    *,
    doc_id: str,
    filename: str,
    text: str,
    child_words: int = 180,
    child_overlap: int = 40,
    parent_words: int = 900,
) -> list[Chunk]:
    """Parent-child chunking with a cheap contextual prefix.

    Children are what we embed and search. Parents are what we send to
    the model. That is the main token-reduction lever: dense fragments
    retrieve well, but answering from a paragraph instead of a sentence
    means fewer, more coherent context blocks.

    The prefix (`source=... section=...`) is a poor-man's version of
    Anthropic contextual retrieval — no extra LLM call per chunk.
    """
    chunks: list[Chunk] = []
    n = 0
    for section, body in _sections(text):
        parents = _split_windows(body, parent_words, 80) or [body]
        for parent in parents:
            children = _split_windows(parent, child_words, child_overlap) or [parent]
            for child in children:
                prefix = f"[source={filename} section={section}] "
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}:{n}",
                        doc_id=doc_id,
                        filename=filename,
                        text=prefix + child,
                        parent_text=prefix + parent,
                        section=section,
                        index=n,
                    )
                )
                n += 1
    return chunks
