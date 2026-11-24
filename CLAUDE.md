# Notes for picking up this project

A booth demo and 20-minute talk for AWS re:Invent 2026 (Las Vegas, 30 Nov to 4 Dec): **Build an Expert Agent with AWS Strands Agents and Neo4j Relate**. Jeremy gives the talk at the Neo4j booth theater (booth 732, Venetian Expo Hall B), Tue and Wed at 17:30.
- **The planner** (`planner.py` with `web/index.html`) is the visual booth demo.
- **The command-line guide** (`reinvent-relate-agent.py`) answers questions.
- **Both** share `reinvent_graph.py`.

Read `README.md` first for what the demo does, how to run it, and the graph model. This file covers what the README doesn't: why things are the way they are, how to check changes, and what's left.

## Where things stand

- **Branch:** the work is on the `planner-demo` branch, several commits ahead of `main`. It hasn't been merged or pushed.
- **Commits:** sign them off with `git commit -s`, and commit only when Jeremy asks.
- **Running:** the planner runs locally (`uv run planner.py`, http://localhost:8026). After a change to `planner.py` or `reinvent_graph.py`, restart it; the page itself (`web/index.html`) is re-read on every request.
- **The graph file:** `reinvent-2026.rl` is committed and must be committed with any change to the build, the schema, `booth-theater.csv`, or `neo4j-events.json`. Rebuild with `uv run planner.py build` (needs `OPENAI_API_KEY`; 10–30 s).
- **Versions:** the scripts pin `relate-db==1.2.0`. The installed `relate` CLI is 1.0.1. It clones and queries 1.2.0 stores fine, including full-text `SEARCH`. `relate upgrade` updates it, but don't run that without asking.

## Design decisions and why

- **The schedule is checked by plain Python, not the LLM** (`check`, `solve`, and `plan_agenda` in `planner.py`). Early versions let the model do the timing. It invented conflicts ("too much travel" between sessions in the same venue), gave up on replans, and stopped calling tools after a refusal. Now `plan_agenda` always saves the best valid agenda, and the page renders only that saved state. Passed-over reasons come from the checker.
- **Candidates are ranked by a hybrid search for the slot's label, with the agent's order as a tiebreak.** That makes the same request plan the same day. The relevance search uses the label alone: adding the whole goal let its other subject ("evaluation") pull in the wrong sessions for the "permissions" slot.
- **The scheduler adds its own finds only on a replan.** They become substitutes that must fit around the sessions being kept. On first plans they used to outrank the agent's picks. For example, a vague slot label ("Talk 1") let AIM218 (Nova Sonic) in for "talks about Strands Agents". The agent can reject those picks with `reject_scheduler_picks`. That list never applies to its own candidates or to the current agenda: the agent once put the whole agenda in it.
- **On a replan, sessions that still fit are kept first**, then moved to another run of the same session (`REPEAT_OF`, same-day repeats), and only then substituted. Slots and candidates persist across turns in `state["agenda"]["slots"]`.
- **The attendee's explicit asks beat keeping: `include` and `drop` on `plan_agenda`.** Before these, "let's opt for AIM278-S" and "drop IND302-R" both failed. Keeping outranked the new pick, the old slot candidates merged the dropped session back in, and `reject_scheduler_picks` ignores anything on the agenda. Now:
  - **`include`:** ranks the named sessions first in `solve`, by series. Sessions it can't place come back as `notPlaced`, with the checker's reason.
  - **`drop`:** removes a session's whole series from keep, candidates, repeats, and fallback finds. It persists in `state["agenda"]["dropped"]` until the session is included again.
- **Small guards for real failures:**
  - A slot labelled "talk" or "hands-on" gets that kind when the agent omits it.
  - Blocked time equal to a single-candidate slot's own time is ignored. The agent would "block" the reception it wanted, but a lunch block that shares a talk's time is kept.
  - Keyword phrases match when all their words appear, so "Strands Agents" matches "agents with Strands".
- **Hybrid search is one Cypher statement** (`HYBRID_SEARCH` in `reinvent_graph.py`). A vector `SEARCH` pre-filters inside the index `WITH [c.date, c.venue, c.start, c.kind]`. A full-text `SEARCH` filters afterwards, because full-text can't filter inside. Reciprocal rank fusion combines them. `find_sessions` splits results into `containKeywords` and `similarOnly`, so the model doesn't claim a merely similar session is "about MCP".
- **Travel times are `TRAVEL` edges built from `TRAVEL_MINUTES`.** They're estimates and are labelled as such on the page. `venues_by_distance` walks them for "far from" and "near" questions.
- **The map is a stylized inline SVG**, stretched east to west so the pins don't crowd. It uses no map tiles, so it works on booth Wi-Fi. Venue inside views are schematics built from catalog room strings, deliberately not traced from official floor plans.
- **Neo4j content:**
  - **Booth theater talks** come from `booth-theater.csv` (codes NEO01–NEO06, `-R` for repeats).
  - **Other events** come from `neo4j-events.json` (NEOAD, Neo4j After Dark, at The Cosmopolitan, which was added as a venue).
  - **The agents know about both:** both agents' prompts describe them, and the page badges them NEO4J BOOTH or NEO4J EVENT.

## How to check a change

1. `uv run test_demo.py` runs the scripted booth demo. The expected result is COM317-R plus SEC407-R, then COM317-R plus OPN309 after the curveball. Run it 3 or more times: the agent varies, and single runs have hidden regressions.
2. `uv run test_goals.py` runs 7 unscripted conversations, including past regressions; `uv run test_goals.py N` runs one. The last full run passed 6/7. The failing one (lunch block, #2) was then fixed and passed 2/2, but the full suite hasn't been rerun since.
3. `uv run browser_check.py` checks the page. Run it with the planner running, then look at the screenshots in `./screenshots`.
4. `uv run reinvent-relate-agent.py "Which MCP talks are at the Venetian on Wednesday?"` should return AIM265-S only.
5. Run `uvx ruff check --select F,E9 *.py` before committing.

When a test fails intermittently, log the tool arguments; `ToolCall` events carry them, and `planner.state["events"]` holds them. Then replay that exact `plan_agenda` call against `options_for` and `solve` directly. That's how the `exclude` misuse was found.

## Open items and ideas

- **Take-away for visitors:** a QR code with the agenda. Phones can't reach localhost, so either encode the agenda as text in the QR, or host it.
- **Booth schedule:** it's a draft. Update `booth-theater.csv` and rebuild when it changes. The Neo4j After Dark details came from a PDF of the event page, because events.neo4j.com blocks automated fetches.
- **Agent variance worth watching:**
  - "I want talks about X on Tuesday" sometimes gets a plain list instead of a planned agenda.
  - The agent occasionally nominates a loosely related session (IND315-R for Strands talks).
  - The curveball swap is OPN309 most of the time. Recent runs: 6/6, then 5/5, then 7/8, with IND421-R or SEC352-R otherwise.
- **The "why this session" graph is sparse for Neo4j sessions**, because they have no topics or services. Tagging them, say with Topic "Artificial Intelligence" and a Neo4j or Graph topic, would make the graph path richer.
- **Duplicated prompt text:** both agents' prompts repeat the booth and After Dark notes. Consider keeping that text once in `reinvent_graph.py`.
- **The browser tab title** is still "re:Invent Planner". The header was retitled "re:Invent 2026 day planner".
- **`.reinvent-2026/`** (85 MB: the original script's catalog and embedding cache) is still tracked in git, but nothing uses it now. Remove it from git if Jeremy agrees.
- **Talk outline idea** (20 min):
  1. Relate basics and the in-browser WASM playground at relatedb.ai, about 5–8 min. See relatedb.ai/docs/concepts/where-relate-fits/ for what it's for and not for.
  2. This planner live, about 5 min.
  3. The "under the hood" panel: the Cypher, hybrid `SEARCH`, `TRAVEL` walks, and the deterministic scheduler.
- **Possible extra:** "the same graph in the browser" with `@relate/relate-wasm`. It can't open `.rl` files, so it would need a Cypher-script export.
