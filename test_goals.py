# /// script
# requires-python = ">=3.10"
# dependencies = ["strands-agents[openai]==1.64.0", "openai==3.49.0", "relate-db==1.2.0"]
# ///
"""Run visitor-style requests through the planner agent and check what it saves.

  uv run test_goals.py            # every conversation, one at a time (a few minutes)
  uv run test_goals.py 2          # just conversation 2

test_demo.py checks the scripted booth demo. These are the unscripted requests that broke earlier
versions: vague slot labels, "far from" a venue, an evening event, and plain questions. Each
conversation runs in a fresh planner, so they don't share an agenda. The agent's picks vary from run
to run, so the checks are the things that must always hold, plus the expected sessions as hints.
"""

import subprocess
import sys
import time

CONVERSATIONS = [
    # (turns, checks on the final agenda)
    (["I'm a data engineer getting into vector search. Plan my Wednesday morning."],
     {"date": "2026-12-02"}),
    (["Give me three sessions on serverless and event-driven architecture Thursday afternoon, at least one hands-on."],
     {"date": "2026-12-03", "kinds": ["hands-on"]}),
    (["Plan Tuesday afternoon: one talk on Kubernetes cost optimization and one hands-on EKS session.",
      "I have lunch with a friend 12:00 to 1:00."],
     {"date": "2026-12-01", "blocked": "12:00"}),
    # Regression: a vague slot label once let the scheduler's own off-topic finds (AIM218) in.
    (["I want talks about Strands Agents", "Give me a tuesday schedule"],
     {"date": "2026-12-01", "title_has": "strands"}),
    # Regression: "far from" once landed across the street at Caesars Palace. Expect MGM Grand.
    (["Plan Wednesday afternoon with two AI agent talks as far away from the Venetian as possible"],
     {"date": "2026-12-02", "venues_not": ["Venetian", "Wynn/Encore", "Caesars Forum"]}),
    # Expect OPN406-R1 at MGM Grand, then NEOAD at The Cosmopolitan.
    (["Plan my Tuesday from 2pm: one Strands Agents talk, then the Neo4j After Dark reception in the evening"],
     {"date": "2026-12-01", "codes": ["NEOAD"]}),
    # A question, not a plan: no agenda is required. Expect AIM265-S only.
    (["Which MCP talks are at the Venetian on Wednesday?"], {}),
    # Regression: naming a session that overlaps one on the agenda used to change nothing, because keeping
    # the agenda outranked it and a dropped session came back from the saved slots. Expect AIM278-S in.
    (["Is Jeremy Adams speaking on Tuesday? I'd like to see him and some talks on running agents in production.",
      "perfect, let's see Jeremy and as many agent talks as fit on Tues",
      "Let's opt for AIM278-S",
      "drop IND302-R"],
     {"date": "2026-12-01", "codes": ["AIM278-S", "NEO04"], "codes_not": ["IND302-R"]}),
]


def run_one(index: int) -> bool:
    import planner  # Imported here so each conversation gets a fresh process and planner.

    turns, want = CONVERSATIONS[index]
    for text in turns:
        started = time.time()
        reply = planner.chat(text)
        print(f"\n>>> {text}  ({time.time() - started:.1f}s)\n{reply[:600]}")
    tools = [e["name"] for e in planner.state["events"] if e["kind"] == "tool"]
    print(f"tools: {tools}")
    agenda = planner.state["agenda"]
    problems = []
    if want and not agenda:
        problems.append("no agenda saved")
    if agenda:
        items = agenda["items"]
        print("agenda:", [(s["code"], s["start"], s["end"], s["venue"], s["kind"]) for s in items])
        problems += planner.check(agenda["date"], tuple(agenda["window"]), items, agenda["blocked"])
        if want.get("date") and agenda["date"] != want["date"]:
            problems.append(f"date {agenda['date']}, expected {want['date']}")
        for k in want.get("kinds", []):
            if not any(s["kind"] == k for s in items):
                problems.append(f"no {k} session")
        if want.get("blocked") and not any(b["start"] == want["blocked"] for b in agenda["blocked"]):
            problems.append(f"blocked time at {want['blocked']} missing")
        if want.get("title_has") and not all(want["title_has"] in s["title"].lower() for s in items):
            problems.append(f"a session without {want['title_has']!r} in its title")
        for v in want.get("venues_not", []):
            if any(s["venue"] == v for s in items):
                problems.append(f"a session at {v}")
        for c in want.get("codes", []):
            if c not in [s["code"] for s in items]:
                problems.append(f"{c} not on the agenda")
        for c in want.get("codes_not", []):
            if c in [s["code"] for s in items]:
                problems.append(f"{c} is on the agenda")
    print("RESULT:", "OK" if not problems else f"PROBLEMS {problems}")
    return not problems


if __name__ == "__main__":
    if sys.argv[1:] and sys.argv[1] != "--all":
        sys.exit(0 if run_one(int(sys.argv[1])) else 1)
    failed = [i for i in range(len(CONVERSATIONS))
              if subprocess.run([sys.executable, __file__, str(i)]).returncode != 0]
    print(f"\n{len(CONVERSATIONS) - len(failed)}/{len(CONVERSATIONS)} conversations OK", f"(failed: {failed})" if failed else "")
    sys.exit(1 if failed else 0)
