# /// script
# requires-python = ">=3.10"
# dependencies = ["strands-agents[openai]==1.64.0", "openai==3.49.0", "relate-db==1.2.0"]
# ///
"""An AWS re:Invent 2026 guide: a Strands agent with tools that query a Relate graph.

Needs OPENAI_API_KEY.
Usage:
  uv run reinvent-relate-agent.py build         # one time: fetch the catalog, embed it, write reinvent-2026.rl
  uv run reinvent-relate-agent.py ["question"]  # answer one question and exit, or chat until Ctrl-C or Ctrl-D

The build reads a public community mirror of the catalog (https://ri26.awsclassroom.kr/).
The official AWS catalog is the authority, so check times there before you go.
"""

import difflib
import json
import os
import sys
from datetime import date
from pathlib import Path

if not os.environ.get("OPENAI_API_KEY"):
    sys.exit("OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY=sk-...")

import relate
from openai import OpenAI

import reinvent_graph as rg  # noqa: E402  The shared graph build and hybrid search, also used by planner.py.

DATABASE = rg.DATABASE
EMBED_MODEL = rg.EMBED_MODEL
openai = OpenAI()
fold = rg.fold
build = lambda: rg.build(openai)  # noqa: E731


if sys.argv[1:] == ["build"]:
    build()
    sys.exit()

if not DATABASE.exists():
    sys.exit(f"{DATABASE.name} not found. Run once: uv run {Path(__file__).name} build")

# --- The graph (opened read-only; see reinvent_graph.Graph) ------------------------------------

from strands import Agent, tool  # noqa: E402
from strands.models.openai import OpenAIModel  # noqa: E402

graph = rg.Graph(on_query=(lambda c, p, n, t: print("   [cypher]", {k: v for k, v in p.items() if k != "embedding"}, n, "rows")) if os.environ.get("DEBUG_CYPHER") else None)
run = graph.run


def to_json(value) -> str:
    """Query results as JSON for the model, with nodes and relationships as their properties."""
    return json.dumps(value, default=lambda v: getattr(v, "properties", str(v)), ensure_ascii=False)


NAME_WORDS = sorted({w for r in run("MATCH (p:Speaker) RETURN p.nameFolded AS n") for w in r["n"].split()})
DAYS = run("MATCH (sl:Slot) RETURN DISTINCT sl.date AS date, sl.day AS day ORDER BY date")
VENUES = [r["v"] for r in run("MATCH (v:Venue) RETURN v.name AS v ORDER BY v")]
FORMATS = [r["f"] for r in run("MATCH (s:Session) RETURN DISTINCT s.format AS f ORDER BY f")]

# Filters shared by the listing tools. Empty strings mean "any".
FILTERS = """
MATCH (s:Session)-[:HAS_SLOT]->(sl:Slot)-[:AT_VENUE]->(v:Venue)
WHERE ($date = '' OR sl.date = $date)
  AND ($venue = '' OR toLower(v.name) CONTAINS toLower($venue))
  AND ($format = '' OR toLower(s.format) CONTAINS toLower($format))
  AND ($after = '' OR sl.start >= $after)
  AND ($before = '' OR sl.start < $before)
"""
# What each tool returns per session occurrence.
DETAILS = """
OPTIONAL MATCH (s)<-[:PRESENTS]-(sp:Speaker)
WITH s, sl, v, extra, collect(sp.name + CASE sp.company WHEN '' THEN '' ELSE ' (' + sp.company + ')' END) AS speakers
RETURN s.code AS code, s.title AS title, s.format AS format, s.level AS level, sl.displayTime AS time,
       v.name AS venue, sl.room AS room, speakers, s.url AS link, extra
"""
def filters(date="", venue="", format="", after="", before="") -> dict:
    return {"date": date, "venue": venue, "format": format, "after": after, "before": before}


# --- The tools -------------------------------------------------------------------------------
# Strands turns each function into a tool: its name, its docstring as the description, and
# its arguments (with the descriptions under Args) as the input schema.


@tool
def find_sessions(query: str, keywords: list[str] | None = None, date: str = "", venue: str = "", format: str = "", after: str = "", before: str = "", limit: int = 15) -> str:
    """Semantic search: the sessions whose title, description, topics, services, and speakers best
    match a subject, optionally only on a day, at a venue, of a format, or in a time window. Put only the
    subject in query, spelled out (e.g. "Model Context Protocol (MCP) servers and agent tools"), any literal
    terms in keywords (e.g. ["MCP", "Model Context Protocol"]), and the day, venue, and format in their
    arguments. Sessions containing a keyword rank first: embeddings alone match acronyms poorly.

    Args:
        query: The subject to search for, in natural language.
        keywords: Literal terms or acronyms (case-insensitive) that mark a session as on-subject, or [].
        date: Day as YYYY-MM-DD (see the day table in your instructions), or "" for any day.
        venue: Part of a venue name, e.g. "Venetian", "Wynn", "MGM", "Caesars Forum", several separated
            by commas, or "" for any.
        format: Part of a session format, e.g. "Breakout", "Chalk talk", "Workshop", "Builders", or "" for any.
        after: Earliest start time, 24-hour "HH:MM" Pacific, or "" for any.
        before: Start before this time, 24-hour "HH:MM" Pacific, or "" for any.
        limit: How many sessions to return, at most 40.
    """
    embed = lambda text: openai.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding  # noqa: E731
    rows = rg.find_sessions(graph, embed, query, keywords, date, venue, "", after, before, limit)
    if format:
        rows = [r for r in rows if format.lower() in r["format"].lower()]
    if not keywords:
        return to_json(rows)
    return to_json({"containKeywords": [r for r in rows if r["matchedKeywords"]],
                    "similarOnly (don't say these are about the keywords)": [r for r in rows if not r["matchedKeywords"]]})


@tool
def list_sessions(keyword: str = "", topic: str = "", service: str = "", date: str = "", venue: str = "", format: str = "",
                  after: str = "", before: str = "") -> str:
    """Every session matching exact filters, in time order: a literal keyword in the code, title, or
    description, a catalog topic (e.g. "Artificial Intelligence", "Serverless"), or an AWS service
    (e.g. "Amazon Bedrock", "Amazon S3"), plus day, venue, format, and time window. Use it for
    schedules and complete lists; use find_sessions for subjects that need meaning, not words.

    Args:
        keyword: Literal text in the session code, title, or description (case-insensitive), or "".
        topic: Part of a catalog topic name, or "".
        service: Part of an AWS service name, or "".
        date: Day as YYYY-MM-DD (see the day table in your instructions), or "" for any day.
        venue: Part of a venue name, e.g. "Venetian", "Wynn", "MGM", "Caesars Forum", or "" for any.
        format: Part of a session format, e.g. "Breakout", "Chalk talk", "Workshop", "Builders", or "" for any.
        after: Earliest start time, 24-hour "HH:MM" Pacific, or "" for any.
        before: Start before this time, 24-hour "HH:MM" Pacific, or "" for any.
    """
    rows = run(FILTERS + """
        AND ($keyword = '' OR toLower(s.code + ' ' + s.title + ' ' + s.abstract) CONTAINS toLower($keyword))
        AND ($topic = '' OR exists { (s)-[:HAS_TOPIC]->(t) WHERE toLower(t.name) CONTAINS toLower($topic) })
        AND ($service = '' OR exists { (s)-[:MENTIONS_SERVICE]->(x) WHERE toLower(x.name) CONTAINS toLower($service) })
        WITH s, sl, v, null AS extra
        """ + DETAILS + " ORDER BY sl.date, sl.start, code",
        keyword=keyword, topic=topic, service=service, **filters(date, venue, format, after, before))
    return to_json({"count": len(rows), "sessions": rows[:150], "truncated": len(rows) > 150})


SPEAKER_SESSIONS = """
MATCH (sp:Speaker) WHERE all(w IN split($name, ' ') WHERE sp.nameFolded CONTAINS w)
MATCH (sp)-[:PRESENTS]->(s:Session)-[:HAS_SLOT]->(sl:Slot)-[:AT_VENUE]->(v:Venue)
WITH s, sl, v, sp.name + CASE sp.company WHEN '' THEN '' ELSE ' (' + sp.company + ')' END AS extra
""" + DETAILS + " ORDER BY extra, sl.date, sl.start LIMIT 40"


@tool
def venues_by_distance(from_venue: str, date: str = "") -> str:
    """Every other venue, farthest first, with its estimated travel minutes from a venue and how many sessions
    it has. Use it for "near X", "far from X", or "away from X", then pass the venues you pick to find_sessions
    (several at once as "MGM Grand, Allegiant Stadium").

    Args:
        from_venue: Part of a venue name, e.g. "Venetian".
        date: Day as YYYY-MM-DD to count that day's sessions, or "".
    """
    return to_json(rg.venues_by_travel(graph, from_venue, date))


@tool
def sessions_by_speaker(name: str) -> str:
    """The sessions a speaker presents, by speaker name or part of it. Accents and small typos are OK.

    Args:
        name: Speaker name or part of it.
    """
    words = fold(name).split()
    if not words:
        return "[]"
    rows = run(SPEAKER_SESSIONS, name=" ".join(words))
    if not rows:  # Nobody matched: try again with each word swapped for the closest word in a speaker's name.
        close = [difflib.get_close_matches(w, NAME_WORDS, n=1, cutoff=0.8) or [w] for w in words]
        rows = run(SPEAKER_SESSIONS, name=" ".join(c[0] for c in close))
    return to_json(rows)


model = OpenAIModel(model_id=os.environ.get("OPENAI_MODEL", "gpt-5.4-mini"))

CYPHER_WRITER_PROMPT = """Write one read-only Cypher query that answers the question, for this graph:
(:Speaker {name, company})-[:PRESENTS]->(:Session {code, title, abstract, format, kind, level, url})
(:Session)-[:HAS_SLOT]->(:Slot {date, day, start, end, displayTime, room, place})-[:AT_VENUE]->(:Venue {name})
(:Session)-[:REPEAT_OF]->(:Series {code, title})   (other runs of a session share its Series)
(:Venue)-[:TRAVEL {minutes}]->(:Venue)   (estimated travel time between venues)
(:Session)-[:HAS_TOPIC]->(:Topic {name})
(:Session)-[:MENTIONS_SERVICE]->(:Service {name})
Slot.date is a 'YYYY-MM-DD' string, Slot.day is like 'Wed', and start and end are 'HH:MM' Pacific strings,
so compare them as strings. Return Slot.displayTime to show a time. Match names with toLower(...) CONTAINS.
Return every row the question asks for: add a LIMIT only when it asks for a few, like "the top 5".
Reply with the Cypher only, no explanation and no code fences."""


@tool
def ask_reinvent_graph(question: str) -> str:
    """Answer any other question about re:Invent sessions, speakers, companies, venues, topics, services,
    and time slots by writing a Cypher query. Good for counts, rankings, and comparisons.

    Args:
        question: The question to answer from the graph.
    """
    # Agents as tools: a specialist agent writes the Cypher. If the query fails, it sees the
    # error and gets one more try.
    cypher_writer = Agent(model=model, system_prompt=CYPHER_WRITER_PROMPT, callback_handler=None)
    prompt = question
    for _ in range(2):
        cypher = str(cypher_writer(prompt)).strip().removeprefix("```cypher").strip("`\n ")
        try:
            return to_json({"cypher": cypher, "rows": run(cypher)[:300]})
        except relate.Error as err:
            error = str(err)
            prompt = f"That failed: {error}. Fix the query."
    return to_json({"cypher": cypher, "error": error})


# --- The agent -------------------------------------------------------------------------------

day_table = "\n".join(f"  {date.fromisoformat(d['date']):%A %-d %B %Y} = {d['date']}" for d in DAYS)
travel_table = "\n".join(f"  {a} ↔ {b}: {m} min" for (a, b), m in sorted(rg.travel_minutes(graph).items()) if a < b)
agent = Agent(
    model=model,
    tools=[find_sessions, venues_by_distance, list_sessions, sessions_by_speaker, ask_reinvent_graph],
    system_prompt=f"""You are a helpful guide to AWS re:Invent 2026 in Las Vegas. Answer using only the tools.

Conference days (use these dates; don't work out weekdays yourself):
{day_table}
For "near", "far from", or "away from" a venue, call venues_by_distance and search the venues it ranks.
If the farthest (or nearest) venues have nothing on the subject, move to the next ones in its ranking.
Estimated travel between venues, walking or by shuttle (planning estimates, not AWS guidance):
{travel_table}
Venues: {", ".join(VENUES)}
Formats: {", ".join(FORMATS)}

All times are Las Vegas local time (Pacific). Write times as the tools return them. Pull the day,
venue, format, and time window out of the question and pass them as tool arguments, and keep only
the subject in a semantic query. Venue filters match part of a name, so "Venetian" also finds places
"located in The Venetian". find_sessions ranks by similarity, so drop results that aren't really
about the subject, and don't claim a semantic list is complete. Include each session's code, title,
time, venue and room, speakers, and catalog link. The data comes from a community mirror of the catalog,
so say the official catalog is the authority when times matter. Neo4j booth theater talks (format
"Booth theater", codes NEO01...) are 30-minute talks at booth 732 in the Venetian Expo (Hall B); they
aren't in the AWS catalog, have no catalog link, and their schedule is a draft. Neo4j After Dark (code NEOAD) is
Neo4j's evening reception with a live set by almost monday: Tuesday 1 Dec, 19:00-22:00, at The Barbershop
Cuts & Cocktails in The Cosmopolitan; guests request a pass on its event page, and a request isn't a pass. If the tools don't have
the answer, say so. Refer to speakers by name or as "they"; don't guess anyone's pronouns.""",
)

if __name__ == "__main__":
    try:
        if sys.argv[1:]:
            agent(" ".join(sys.argv[1:]))
        else:
            print("Ask about re:Invent 2026 sessions, speakers, and schedule. Press Ctrl-D to quit.")
            while True:
                if question := input("\nyou> ").strip():
                    agent(question)
                    print()
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        graph.close()
    print()
