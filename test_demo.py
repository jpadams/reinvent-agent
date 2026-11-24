# /// script
# requires-python = ">=3.10"
# dependencies = ["strands-agents[openai]==1.64.0", "openai==3.49.0", "relate-db==1.2.0"]
# ///
"""Drive the booth demo's two turns through the planner agent and check the agenda it saves.

  uv run test_demo.py

Turn 1 builds a Tuesday afternoon with one hands-on session and one talk. Turn 2 blocks time for a
customer meeting; the agenda must still check out, keep a hands-on session, and keep what still fits.
"""

import time

import planner

GOAL = ("I'm building a production GenAI agent. I know RAG, but I want to learn tool permissions and "
        "evaluation. I only have Tuesday afternoon. Give me one hands-on session and one architecture talk.")
# The page's curveball button does the same: a meeting that lands on the hands-on session, so the replan
# has something to fix whichever sessions turn 1 picked.
def curveball(agenda):
    start = next(s["start"] for s in agenda["items"] if s["kind"] == "hands-on")
    h, m = map(int, start.split(":"))
    return f"I just got a customer meeting at {h % 12 or 12}:{m:02d} {'pm' if h >= 12 else 'am'} for an hour. Keep a hands-on session if you can.", start


def show(agenda):
    for s in agenda["items"]:
        print(f"   {s['start']}–{s['end']}  {s['code']:<10} {s['kind']:<8} {s['venue']:<15} {s['title'][:60]}")
        print(f"      why: {s['why']}")
    for leg in agenda["legs"]:
        print(f"   travel {leg['fromVenue']} → {leg['toVenue']}: {leg['minutes']} min needed, {leg['gap']} min free")
    for p in agenda["passedOver"]:
        print(f"   passed over {p['code']} {p['start']} {p['venue']} ({p['slot']}): {p['reason']}")
    for b in agenda["blocked"]:
        print(f"   blocked {b['start']}–{b['end']} {b.get('label', '')}")
    print(f"   changes: {agenda['changes']}")


def verify(agenda, meeting=None):
    items = agenda["items"]
    assert agenda["date"] == "2026-12-01", agenda["date"]
    problems = planner.check(agenda["date"], tuple(agenda["window"]), items, agenda["blocked"])
    assert not problems, problems
    assert all(planner.minutes(s["start"]) >= 12 * 60 for s in items), "outside the afternoon"
    assert any(s["kind"] == "hands-on" for s in items), "no hands-on session"
    assert any(s["kind"] == "talk" for s in items), "no talk"
    if meeting:
        assert any(b["start"] == meeting for b in agenda["blocked"]), agenda["blocked"]


meeting = None
for turn in (1, 2):
    text = GOAL if turn == 1 else curveball(planner.state["agenda"])[0]
    if turn == 2:
        meeting = curveball(planner.state["agenda"])[1]
    started = time.time()
    reply = planner.chat(text)
    print(f"\n== Turn {turn} ({time.time() - started:.1f}s) ==\n{reply}\n")
    tools = [e["name"] for e in planner.state["events"] if e["kind"] == "tool"]
    rejected = sum(e["kind"] == "rejected" for e in planner.state["events"])
    print(f"   tools so far: {tools}  (rejected agendas: {rejected})")
    agenda = planner.state["agenda"]
    assert agenda, f"no agenda saved; events: {[(e['kind'], e.get('name') or e.get('problems')) for e in planner.state['events'] if e['kind'] != 'cypher']}"
    show(agenda)
    verify(agenda, meeting)
    if turn == 2:
        before = {s["code"] for s in planner.state["previous"]["items"]}
        after = {s["code"] for s in agenda["items"]}
        still_fit = [s for s in planner.state["previous"]["items"]
                     if not planner.check(agenda["date"], tuple(agenda["window"]), [s], agenda["blocked"])]
        kept = {s["code"] for s in still_fit} & after
        assert kept == {s["code"] for s in still_fit}, f"dropped a session that still fit: {before - after}"
print("\nOK: both turns produced valid agendas.")
