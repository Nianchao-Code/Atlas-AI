"""Turn a client-supplied upload into a file this service is willing to keep.

Everything a caller sends is attacker-controlled, and `filename` is the part
that is easiest to forget about because it arrives looking like data. It is
not: it was being joined straight onto the upload directory, and

    filename = "../../../../app/main.py"

resolved to /app/app/main.py -- the running application's own source, inside a
container whose user owns /app and can write it. The next restart would have
executed whatever the upload contained. See tests/test_uploads.py, which pins
that exact payload.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Protocol

# What app.chunking can actually read. Anything else is indexed as replacement
# characters and pollutes retrieval, so it is rejected at the door rather than
# discovered as a bad answer later.
SUPPORTED_SUFFIXES = frozenset({".md", ".txt", ".pdf"})

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_CHUNK = 1 << 20


class UploadRejected(Exception):
    """A rejection the caller can act on: carries the status it maps to."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


class _Readable(Protocol):
    async def read(self, size: int = -1) -> bytes: ...


def safe_name(filename: str | None) -> str:
    r"""Reduce a client filename to a single, inert path component.

    Backslashes are split by hand: on Linux they are ordinary characters, so
    `Path("..\\..\\x.md").name` returns the whole string unchanged, and a
    payload would survive a POSIX-only defence.
    """
    raw = (filename or "").replace("\\", "/")
    base = Path(raw).name
    base = _UNSAFE.sub("_", base).strip("._")
    return base[:120] or "upload.bin"


def destination(upload_dir: str, doc_id: str, filename: str) -> Path:
    """Where an upload may be written, or raise.

    The containment check is redundant against `safe_name` today. It is here
    because it is the half that keeps holding if someone later relaxes the
    sanitiser: one asserts a property of the result, the other only of the
    input.
    """
    root = Path(upload_dir).resolve()
    dest = (root / f"{doc_id}_{safe_name(filename)}").resolve()
    if not dest.is_relative_to(root):
        raise UploadRejected(400, "invalid filename")
    if dest.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise UploadRejected(
            415, f"unsupported file type; expected one of {sorted(SUPPORTED_SUFFIXES)}"
        )
    return dest


async def save(file: _Readable, dest: Path, max_bytes: int) -> int:
    """Stream an upload to disk under a byte budget, off the event loop.

    The previous `raw = await file.read()` bought the whole body into memory
    before anything could object to its size, so a single request larger than
    the container's memory limit was an OOM kill rather than a 413.
    """
    total = 0
    try:
        with dest.open("wb") as fh:
            while chunk := await file.read(_CHUNK):
                total += len(chunk)
                if total > max_bytes:
                    raise UploadRejected(413, f"file exceeds the {max_bytes} byte limit")
                await asyncio.to_thread(fh.write, chunk)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    return total
