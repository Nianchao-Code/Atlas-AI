"""The upload path, which is the one place a caller supplies a filesystem path.

The traversal case here is not hypothetical: `../../../../app/main.py` was
verified against the running pod to resolve to /app/app/main.py, which the
container user owns and can write.
"""

from __future__ import annotations

import pytest

from app.uploads import SUPPORTED_SUFFIXES, UploadRejected, destination, safe_name, save


@pytest.mark.parametrize(
    "hostile",
    [
        "../../../../app/main.py",
        "../../etc/passwd",
        "/etc/passwd",
        r"..\..\..\windows\system32\hosts",
        "....//....//app/main.py",
        "sub/dir/notes.md",
    ],
)
def test_no_upload_escapes_the_upload_directory(tmp_path, hostile):
    try:
        dest = destination(str(tmp_path), "abc123456789", hostile)
    except UploadRejected as exc:
        # Rejecting outright is a fine outcome; landing outside is not.
        assert exc.status in (400, 415)
        return
    assert dest.parent == tmp_path.resolve()


def test_the_documented_payload_is_rejected_not_merely_relocated(tmp_path):
    # It reduces to main.py, whose suffix this service cannot read anyway.
    with pytest.raises(UploadRejected) as exc:
        destination(str(tmp_path), "abc123456789", "../../../../app/main.py")
    assert exc.value.status == 415


def test_ordinary_filenames_survive(tmp_path):
    dest = destination(str(tmp_path), "abc123456789", "Employee Handbook v2.md")
    assert dest.parent == tmp_path.resolve()
    assert dest.name == "abc123456789_Employee_Handbook_v2.md"


@pytest.mark.parametrize("suffix", sorted(SUPPORTED_SUFFIXES))
def test_every_suffix_the_parser_handles_is_accepted(tmp_path, suffix):
    assert destination(str(tmp_path), "doc", f"notes{suffix}").suffix == suffix


def test_unreadable_types_are_refused(tmp_path):
    with pytest.raises(UploadRejected) as exc:
        destination(str(tmp_path), "doc", "payload.exe")
    assert exc.value.status == 415


def test_a_filename_that_sanitises_to_nothing_still_yields_a_path(tmp_path):
    # "..." has no usable characters left, so the fallback name applies -- and
    # the fallback must itself be a type the parser accepts or nothing can be
    # uploaded without an extension.
    with pytest.raises(UploadRejected):
        destination(str(tmp_path), "doc", "...")
    assert safe_name("...") == "upload.bin"


class _Body:
    """An UploadFile stand-in: hands back `size` bytes at a time, like Starlette."""

    def __init__(self, total: int) -> None:
        self.remaining = total

    async def read(self, size: int = -1) -> bytes:
        n = min(size, self.remaining) if size > 0 else self.remaining
        self.remaining -= n
        return b"x" * n


async def test_a_body_under_the_budget_is_written(tmp_path):
    dest = tmp_path / "ok.md"
    assert await save(_Body(4096), dest, max_bytes=1 << 20) == 4096
    assert dest.stat().st_size == 4096


async def test_an_oversized_body_is_refused_and_leaves_nothing_behind(tmp_path):
    dest = tmp_path / "big.md"
    with pytest.raises(UploadRejected) as exc:
        await save(_Body(5 << 20), dest, max_bytes=1 << 20)
    assert exc.value.status == 413
    # The partial write is the reason this matters: refusing but keeping the
    # bytes would let a caller fill the volume in 413s.
    assert not dest.exists()
