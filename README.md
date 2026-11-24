# re:Invent 2026 planner: Strands Agents + Neo4j Relate

A demo for **Build an Expert Agent with AWS Strands Agents and Neo4j Relate**. A [Strands](https://strandsagents.com/) agent plans an attendee's re:Invent day from a [Relate](https://relatedb.ai/) graph of the catalog. Relate is an embedded graph database, so the whole graph, with its vector and full-text indexes, is one file, `reinvent-2026.rl`, opened inside the app.

| File | What it is |
|---|---|
| `planner.py` | The booth demo: planner agent, schedule checker, and web page at http://localhost:8026 |
| `reinvent-relate-agent.py` | The command-line guide: ask questions about sessions, speakers, and schedule |
| `reinvent_graph.py` | The shared graph: the one-time build, hybrid search, and the read-only graph |
| `web/index.html` | The page: chat, tool and Cypher log, agenda timeline, Strip map, venue schematics, "why this session" graph |
| `test_demo.py` | Runs the two booth turns through the agent and checks the agenda it saves |

## Run

You need [uv](https://docs.astral.sh/uv/), an OpenAI API key, and (optionally) the [`relate` CLI](https://relatedb.ai/), which the build uses to compact the file.

```sh
export OPENAI_API_KEY=sk-...
uv run planner.py build      # one time, about 10 s: fetch the catalog, embed 2,655 sessions, write reinvent-2026.rl
uv run planner.py            # open http://localhost:8026
uv run reinvent-relate-agent.py "Which MCP talks are at the Venetian on Wednesday?"
uv run test_demo.py          # the two booth turns, checked
```

Set `OPENAI_MODEL` to use a different chat model (default `gpt-5.4-mini`). Click **New visitor** to start over between visitors.

## The booth demo (90 seconds)

1. Click **Demo: plan my Tuesday afternoon**. The request is *"I'm building a production GenAI agent. I know RAG, but I want to learn tool permissions and evaluation. I only have Tuesday afternoon. Give me one hands-on session and one architecture talk."* It typically plans:
   - **COM317-R** *Ship Reliable AI Agents with Strands Evaluation and AgentCore Evaluations* (13:00, Wynn)
   - **SEC407-R** *Constrain AI agents with permissions boundaries and Cedar in AgentCore* (15:30, Wynn)

   "Considered, not chosen" shows the sessions it passed over, with the scheduler's real conflict for each (overlap, or not enough travel time).
2. Click a session. The map switches to that venue's rooms, and **Why this session** draws its graph path: its topics and services, the agenda session they connect it to, and its other runs.
3. Click **Curveball: meeting at …**. The button aims the meeting at the hands-on session, so the replan always has something to fix. Typically:
   - COM317-R is **KEPT**.
   - SEC407-R is **removed**.
   - **OPN309** *Defense in depth for Strands AI agents with Cedar, AgentCore, and Rex* (14:30, Caesars Forum) comes in as **NEW**, with ~30 min to travel from Wynn and 30 min free.

The agent's choices can vary from run to run. `test_demo.py` checks what must hold every time: no overlaps, enough travel time, the meeting respected, a hands-on session and a talk, and every session that still fits kept.

## How it works

- **Strands** turns the goal into subjects, a day, a window, and slots ("hands-on: tool permissions", "talk: evaluation"). It searches for candidates for each slot and writes a one-line why for each.
- **Relate** answers every question from the graph:
  - **Hybrid search** is one Cypher statement that fuses two ranked lists by reciprocal rank. The vector `SEARCH` filters by day, venue, start time, and kind inside the index. The full-text `SEARCH` catches exact words like "MCP" or "Cedar".
  - **Graph walks:**
    - `REPEAT_OF` finds other runs of the same session.
    - Shared `HAS_TOPIC` and `MENTIONS_SERVICE` neighbors pair sessions.
    - `TRAVEL` edges between venues give the travel minutes.
- **The scheduler** (`plan_agenda`) is plain Python, not the LLM.
  - **Candidates:** for each slot, it takes the agent's candidates and their other runs that day. It ranks them by a hybrid search for the slot's subject, so the same request plans the same day.
  - **Choosing:** it tries every combination, rejects overlaps, blocked time, and gaps too short to travel, and always saves the best valid agenda. If none of the agent's picks fit a slot, it leaves the slot empty; it doesn't fill it with something off-subject.
  - **Replanning:** it keeps sessions that still fit, then moves a session to another run of it, and only then substitutes. It can substitute from its own search when none of the agent's candidates fit around what it keeps. Those cards are badged SCHEDULER PICK, and the agent can reject them (`reject_scheduler_picks`).
  - **Explaining:** the agent explains the result, and can rewrite the why for a scheduler pick (`explain_pick`). The page shows only what the scheduler saved, so passed-over reasons are real conflicts, not the model's guesses.

### Graph

```
(:Speaker)-[:PRESENTS]->(:Session {code, title, abstract, format, kind, level, url})
(:Session)-[:REPEAT_OF]->(:Series)
(:Session)-[:HAS_SLOT]->(:Slot {date, start, end, room})-[:AT_VENUE]->(:Venue {name, lat, lon})
(:Session)-[:HAS_TOPIC]->(:Topic)   (:Session)-[:MENTIONS_SERVICE]->(:Service)
(:Venue)-[:TRAVEL {minutes}]->(:Venue)
(:SessionChunk {text, embedding, date, venue, start, kind})-[:FROM_DOCUMENT]->(:Session)
```

### Caveats

- The catalog comes from a public community mirror (https://ri26.awsclassroom.kr/). The official AWS catalog is the authority.
- Travel minutes are planning estimates in `reinvent_graph.py` (`TRAVEL_MINUTES`), not AWS guidance. Edit them and rebuild.
- The Strip map is stylized, and each venue's inside view is a schematic built from the catalog's room names, not a floor plan.
