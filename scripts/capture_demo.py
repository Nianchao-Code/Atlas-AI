"""Record the README demo GIF against a running Atlas deployment.

Drives the real UI; nothing here is mocked.

    kubectl port-forward svc/frontend 8080:80 -n atlas

    # Clear the semantic cache first, or the run is an exact cache hit and the
    # graph trace shows guard -> cache and stops -- which is the opposite of
    # what the GIF is meant to show.
    kubectl exec -n atlas deploy/redis -- \
      sh -c 'redis-cli --scan --pattern "qa:*" | xargs -r redis-cli DEL'

    python capture_demo.py
"""
from __future__ import annotations

import io
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8080"
QUESTION = "How many annual leave days in the first year?"
VIEWPORT = {"width": 1280, "height": 880}
GIF_WIDTH = 900
FRAME_MS = 150
OUT = Path("demo.gif")

frames: list[Image.Image] = []


def grab(page) -> None:
    frames.append(Image.open(io.BytesIO(page.screenshot())).convert("RGB"))


def hold(page, ms: int) -> None:
    deadline = time.time() + ms / 1000
    while time.time() < deadline:
        grab(page)
        page.wait_for_timeout(FRAME_MS)


with sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_page(viewport=VIEWPORT, device_scale_factor=1)
    page.goto(URL, wait_until="networkidle")
    page.wait_for_timeout(1200)

    hold(page, 800)

    box = page.locator("textarea")
    box.click()
    box.fill("")
    for i in range(0, len(QUESTION), 3):
        box.fill(QUESTION[: i + 3])
        grab(page)
    hold(page, 400)

    page.get_by_role("button", name="Retrieve and answer").click()

    # Done when the button label reverts and the metrics line has a real
    # faithfulness score. Two earlier attempts got this wrong: waiting on the
    # page being unchanged fires during the pre-answer pause while the panel is
    # still empty, and substring-matching "faith" always passes because
    # "faithfulness" contains it.
    deadline = time.time() + 60
    while time.time() < deadline:
        grab(page)
        page.wait_for_timeout(FRAME_MS)
        labels = [
            b.inner_text().strip()
            for b in page.get_by_role("button").all()
            if "Retrieve and answer" in b.inner_text() or "Streaming" in b.inner_text()
        ]
        metrics = next(
            (l for l in page.inner_text("body").splitlines()
             if "retrieval" in l and "faithfulness" in l),
            "",
        )
        if labels and "Streaming" not in labels[0] and metrics and "faithfulness —" not in metrics:
            break
    else:
        raise SystemExit("answer never completed; is the API reachable and the cache cleared?")

    # End on the finished state: answer, full trace, cited sources. An earlier
    # cut closed on the SLI tab, whose panel is a third of this height and left
    # the last two seconds looking like a rendering bug.
    hold(page, 3000)

    browser.close()

# Collapse consecutive identical frames but keep their time, so the pauses
# survive. Dropping them outright turned a 9s walkthrough into a 2.7s flicker.
timed: list[tuple[Image.Image, int]] = []
for f in frames:
    if timed and f.tobytes() == timed[-1][0].tobytes():
        timed[-1] = (timed[-1][0], timed[-1][1] + FRAME_MS)
    else:
        timed.append((f, FRAME_MS))

scale = GIF_WIDTH / VIEWPORT["width"]
height = int(VIEWPORT["height"] * scale)
# Near-monochrome UI, so a small adaptive palette is visually lossless.
images = [
    f.resize((GIF_WIDTH, height), Image.LANCZOS).quantize(
        colors=96, method=Image.MEDIANCUT, dither=Image.NONE
    )
    for f, _ in timed
]
durations = [d for _, d in timed]

images[0].save(
    OUT,
    save_all=True,
    append_images=images[1:],
    duration=durations,
    loop=0,
    optimize=True,
    disposal=2,
)

print(f"captured {len(frames)} frames -> {len(images)} after collapsing duplicates")
print(f"runtime {sum(durations) / 1000:.1f}s, {GIF_WIDTH}x{height}")
print(f"wrote {OUT} ({OUT.stat().st_size / 1_000_000:.2f} MB)")
