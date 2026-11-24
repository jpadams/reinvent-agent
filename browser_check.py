# /// script
# requires-python = ">=3.10"
# dependencies = ["playwright==1.57.0"]
# ///
"""Drive the booth demo in a headless browser and save screenshots, to check the page by eye.

  uv run planner.py                           # in one terminal
  uv run browser_check.py [output-directory]  # in another (default: ./screenshots)

It uses your installed Google Chrome (headless, with a throwaway profile), so there's no browser
download. It clicks New visitor, runs the demo button, clicks a session, runs the curveball, and
shows the Strip map and the Venetian inside view. It prints any JavaScript errors it saw.
"""

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(sys.argv[1] if sys.argv[1:] else "screenshots")
OUT.mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    page = browser.new_page(viewport={"width": 1600, "height": 950})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: m.type == "error" and errors.append(m.text))
    page.goto("http://localhost:8026/")
    page.click("#reset")
    time.sleep(1)
    page.screenshot(path=OUT / "0-empty.png")

    page.click("#chips .chip >> nth=0")  # Demo: plan my Tuesday afternoon
    page.wait_for_function("document.querySelectorAll('.card').length >= 2 && !document.querySelector('.thinking')",
                           timeout=120000)
    time.sleep(1.5)
    page.screenshot(path=OUT / "1-plan.png")

    page.click(".card >> nth=1")  # The "why this session" graph and the venue's rooms.
    time.sleep(1.5)
    page.screenshot(path=OUT / "2-why.png")

    page.click("#chips .chip >> nth=1")  # Curveball: a meeting on the hands-on session.
    time.sleep(1)
    page.wait_for_function("document.querySelector('.block') && !document.querySelector('.thinking')", timeout=120000)
    time.sleep(1.5)
    page.screenshot(path=OUT / "3-replan.png")

    page.click("#mapTabs button >> nth=0")
    time.sleep(1)
    page.screenshot(path=OUT / "4-strip.png")
    page.click("#mapTabs button:has-text('Venetian')")
    time.sleep(1)
    page.screenshot(path=OUT / "5-venetian.png")

    agenda = page.evaluate("fetch('/api/state').then(r => r.json()).then(s => s.agenda.items.map(i => i.code))")
    print("final agenda:", agenda)
    print("errors:", errors or "none")
    print(f"screenshots in {OUT.resolve()}")
    browser.close()
