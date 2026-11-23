# /// script
# requires-python = ">=3.10"
# dependencies = ["strands-agents[openai]==1.64.0", "openai==3.49.0", "relate-db==1.0.0"]
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
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode, urljoin
from urllib.request import urlopen

if not os.environ.get("OPENAI_API_KEY"):
    sys.exit("OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY=sk-...")

import relate
from openai import OpenAI

DATABASE = Path(__file__).with_name("reinvent-2026.rl")
MIRROR = "https://ri26.awsclassroom.kr/"
CATALOG = "https://registration.awsevents.com/flow/awsevents/reinvent2026/eventcatalog/page/eventcatalog"
EMBED_MODEL = "text-embedding-3-small"
openai = OpenAI()


def fold(text: str) -> str:
    """Lowercase text without accents: "Gołąb" becomes "golab"."""
    text = unicodedata.normalize("NFKD", text.lower().translate(str.maketrans({"ł": "l", "ø": "o", "đ": "d", "ß": "ss", "æ": "ae"})))
    return "".join(c for c in text if not unicodedata.combining(c))


# --- The build (one time) --------------------------------------------------------------------


class ModuleScripts(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sources = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "script" and attrs.get("type") == "module" and attrs.get("src"):
            self.sources.append(attrs["src"])


def download(url: str) -> str:
    with urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8")


def fetch_sessions() -> list[dict]:
    """The session array from the mirror's JS bundle, decoded without running any JS."""
    page = ModuleScripts()
    page.feed(download(MIRROR))
    if len(page.sources) != 1:
        raise ValueError("Expected one module bundle on the mirror page")
    bundle = download(urljoin(MIRROR, page.sources[0]))
    match = re.search(r'JSON\.parse\(`(\[\{"id":(?:\\.|[^`\\])*)`\)', bundle, re.S)
    if not match:
        raise ValueError("Could not find the session array in the mirror bundle")
    escapes = {"\\": "\\", "`": "`", "$": "$", "n": "\n", "r": "\r", "t": "\t"}
    sessions = json.loads(re.sub(r"\\(.)", lambda m: escapes[m[1]], match[1]))
    if len(sessions) < 1000 or len({s["id"] for s in sessions}) != len(sessions):
        raise ValueError("Unexpected catalog size or duplicate session IDs")
    return sessions


def session_text(s: dict) -> str:
    speakers = [f"{p['name']} ({p['company']})" if p.get("company") else p["name"] for p in s.get("speakers", []) if p.get("name")]
    return "\n".join([
        f"{s['code']}: {s['title'].strip()}", s.get("type") or "", s.get("abstract") or "",
        "Topics: " + ", ".join(s.get("topics", [])), "Services: " + ", ".join(s.get("services", [])),
        "Speakers: " + ", ".join(speakers),
    ])


def embed_all(texts: list[str]) -> list[list[float]]:
    """Embed every text in a few large requests, in parallel. About 2,700 sessions take seconds."""
    batches = [texts[i:i + 500] for i in range(0, len(texts), 500)]
    with ThreadPoolExecutor(max_workers=len(batches)) as pool:
        results = pool.map(lambda b: openai.embeddings.create(model=EMBED_MODEL, input=b).data, batches)
    return [d.embedding for batch in results for d in batch]


def load(path: Path, sessions: list[dict], embeddings: list[list[float]]) -> None:
    db = relate.connect(str(path))
    try:
        for label, key in [("Session", "id"), ("Venue", "name"), ("Topic", "name"), ("Service", "name"), ("Speaker", "key")]:
            db.execute(f"CREATE CONSTRAINT FOR (n:{label}) REQUIRE n.{key} IS UNIQUE")
        rows = [{
            "id": s["id"], "code": s["code"], "title": s["title"].strip(), "abstract": s.get("abstract") or "",
            "format": s.get("type") or "", "level": s.get("level") or "", "text": session_text(s), "embedding": e,
            "url": CATALOG + "?" + urlencode({"search": s["code"]}),
        } for s, e in zip(sessions, embeddings)]
        for i in range(0, len(rows), 500):  # Relate caps one commit at 64 MB, so store the embeddings in batches.
            db.execute("""
            UNWIND $rows AS row
            CREATE (s:Session {id: row.id, code: row.code, title: row.title, abstract: row.abstract,
                               format: row.format, level: row.level, url: row.url})
            CREATE (:SessionChunk {text: row.text, embedding: row.embedding})-[:FROM_DOCUMENT]->(s)
            """, {"rows": rows[i:i + 500]})
        slots = []
        for s in sessions:
            for i, sl in enumerate(s.get("slots", [])):
                day = date.fromisoformat(sl["date"])
                slots.append({**sl, "id": f"{s['id']}:{i}", "session": s["id"], "day": day.strftime("%a"),
                              "display": f"{day:%a} {day.day} {day:%b}, {sl['start']}–{sl['end']} PT"})
        db.execute("""
            UNWIND $rows AS row
            MATCH (s:Session {id: row.session})
            MERGE (v:Venue {name: row.venue})
            CREATE (s)-[:HAS_SLOT]->(:Slot {id: row.id, date: row.date, day: row.day, start: row.start, end: row.end,
                                            displayTime: row.display, room: row.room})-[:AT_VENUE]->(v)
            """, {"rows": slots})
        for label, rel, field in [("Topic", "HAS_TOPIC", "topics"), ("Service", "MENTIONS_SERVICE", "services")]:
            db.execute(f"""
                UNWIND $rows AS row
                MATCH (s:Session {{id: row.id}})
                MERGE (n:{label} {{name: row.name}})
                MERGE (s)-[:{rel}]->(n)
                """, {"rows": [{"id": s["id"], "name": v} for s in sessions for v in s.get(field, [])]})
        db.execute("""
            UNWIND $rows AS row
            MATCH (s:Session {id: row.id})
            MERGE (p:Speaker {key: row.key})
            SET p.name = row.name, p.company = row.company, p.nameFolded = row.folded
            MERGE (p)-[:PRESENTS]->(s)
            """, {"rows": [{"id": s["id"], "name": p["name"], "company": p.get("company") or "", "folded": fold(p["name"]),
                            "key": json.dumps([p["name"], p.get("company") or ""])}
                           for s in sessions for p in s.get("speakers", []) if p.get("name")]})
        db.execute("""
            CREATE VECTOR INDEX session_chunk_embeddings IF NOT EXISTS
            FOR (c:SessionChunk) ON c.embedding
            OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}
            """)
        counts = db.execute("""
            MATCH (s:Session) WITH count(s) AS sessions
            MATCH (sl:Slot) WITH sessions, count(sl) AS slots
            MATCH (c:SessionChunk) WHERE c.embedding IS NOT NULL
            RETURN sessions, slots, count(c) AS embedded
            """).fetchall()
        print(f"Loaded: {counts[0]}")
    finally:
        db.close()


def build() -> None:
    started = time.time()
    sessions = fetch_sessions()
    print(f"Fetched {len(sessions)} sessions from {MIRROR} ({time.time() - started:.1f}s)")
    embeddings = embed_all([session_text(s) for s in sessions])
    print(f"Embedded {len(embeddings)} sessions with {EMBED_MODEL} ({time.time() - started:.1f}s)")
    with tempfile.TemporaryDirectory() as tmp:
        store = Path(tmp) / "store"
        load(store, sessions, embeddings)
        # A .rl file keeps the space of files it no longer needs, like the load's write-ahead log,
        # so load into a directory store and `relate clone` it into a compact .rl when the CLI is here.
        if cli := shutil.which("relate"):
            bundle = Path(tmp) / DATABASE.name
            subprocess.run([cli, "clone", str(store), str(bundle)], check=True, stdout=subprocess.DEVNULL)
            os.replace(bundle, DATABASE)
        else:
            print("No relate CLI found, so writing the .rl directly (larger). See https://relatedb.ai/")
            DATABASE.unlink(missing_ok=True)
            load(DATABASE, sessions, embeddings)
    print(f"Wrote {DATABASE.name} ({DATABASE.stat().st_size / 1e6:.1f} MB) in {time.time() - started:.1f}s")


if sys.argv[1:] == ["build"]:
    build()
    sys.exit()

if not DATABASE.exists():
    sys.exit(f"{DATABASE.name} not found. Run once: uv run {Path(__file__).name} build")

# --- The graph -------------------------------------------------------------------------------
# A Relate connection only works on the thread that opened it, and Strands runs tools on
# worker threads, so one dedicated thread opens the database and runs every query.

from strands import Agent, tool  # noqa: E402
from strands.models.openai import OpenAIModel  # noqa: E402

db_thread = ThreadPoolExecutor(max_workers=1)
db = db_thread.submit(relate.connect, str(DATABASE), read_only=True).result()


def run(cypher: str, **params) -> list[dict]:
    return db_thread.submit(lambda: [row.data() for row in db.execute(cypher, params)]).result()


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
        venue: Part of a venue name, e.g. "Venetian", "Wynn", "MGM", "Caesars Forum", or "" for any.
        format: Part of a session format, e.g. "Breakout", "Chalk talk", "Workshop", "Builders", or "" for any.
        after: Earliest start time, 24-hour "HH:MM" Pacific, or "" for any.
        before: Start before this time, 24-hour "HH:MM" Pacific, or "" for any.
        limit: How many sessions to return, at most 40.
    """
    embedding = openai.embeddings.create(model=EMBED_MODEL, input=query).data[0].embedding
    # Exact cosine over the filtered sessions: a few thousand vectors take tens of milliseconds, and
    # unlike a top-k from the vector index, it can't drop matches that the filters would keep.
    return to_json(run(FILTERS + """
        MATCH (c:SessionChunk)-[:FROM_DOCUMENT]->(s)
        WITH s, sl, v, vector.similarity.cosine(c.embedding, $embedding) AS score,
             [k IN $keywords WHERE toLower(c.text) CONTAINS k] AS hits
        WITH s, sl, v, {score: round(score, 3), keywordHits: hits} AS extra, score + CASE WHEN size(hits) > 0 THEN 1 ELSE 0 END AS rank
        ORDER BY rank DESC LIMIT $limit
        """ + DETAILS + " ORDER BY extra.keywordHits = [] , extra.score DESC", embedding=embedding,
        keywords=[k.lower() for k in keywords or [] if k.strip()], limit=max(1, min(limit, 40)), **filters(date, venue, format, after, before)))


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
(:Speaker {name, company})-[:PRESENTS]->(:Session {code, title, abstract, format, level, url})
(:Session)-[:HAS_SLOT]->(:Slot {date, day, start, end, displayTime, room})-[:AT_VENUE]->(:Venue {name})
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
agent = Agent(
    model=model,
    tools=[find_sessions, list_sessions, sessions_by_speaker, ask_reinvent_graph],
    system_prompt=f"""You are a helpful guide to AWS re:Invent 2026 in Las Vegas. Answer using only the tools.

Conference days (use these dates; don't work out weekdays yourself):
{day_table}
Venues: {", ".join(VENUES)}
Formats: {", ".join(FORMATS)}

All times are Las Vegas local time (Pacific). Write times as the tools return them. Pull the day,
venue, format, and time window out of the question and pass them as tool arguments, and keep only
the subject in a semantic query. Venue filters match part of a name, so "Venetian" also finds places
"located in The Venetian". find_sessions ranks by similarity, so drop results that aren't really
about the subject, and don't claim a semantic list is complete. Include each session's code, title,
time, venue and room, speakers, and catalog link. The data comes from a community mirror of the catalog,
so say the official catalog is the authority when times matter. If the tools don't have the answer,
say so. Refer to speakers by name or as "they"; don't guess anyone's pronouns.""",
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
        db_thread.submit(db.close).result()  # Close the database on its own thread.
    print()
