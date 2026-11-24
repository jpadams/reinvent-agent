"""The re:Invent 2026 graph in Relate: the one-time build, and the queries the agents share.

  (:Speaker {name, company})-[:PRESENTS]->(:Session {code, title, abstract, format, kind, level, url})
  (:Session)-[:REPEAT_OF]->(:Series {code, title})            every time a session runs
  (:Session)-[:HAS_SLOT]->(:Slot {date, day, start, end, displayTime, room, place})-[:AT_VENUE]->(:Venue)
  (:Session)-[:HAS_TOPIC]->(:Topic)   (:Session)-[:MENTIONS_SERVICE]->(:Service)
  (:Venue {name, lat, lon})-[:TRAVEL {minutes}]->(:Venue)    estimated door-to-door minutes
  (:SessionChunk {text, embedding, date, venue, start, kind})-[:FROM_DOCUMENT]->(:Session)

SessionChunk has a vector index that pre-filters on date, venue, start, and kind, and a full-text
index on its text. find_sessions fuses the two by reciprocal rank, in one Cypher statement.
"""

import atexit
import csv
import json
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import unicodedata
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode, urljoin
from urllib.request import urlopen

import relate

DATABASE = Path(__file__).with_name("reinvent-2026.rl")
MIRROR = "https://ri26.awsclassroom.kr/"
CATALOG = "https://registration.awsevents.com/flow/awsevents/reinvent2026/eventcatalog/page/eventcatalog"
EMBED_MODEL = "text-embedding-3-small"

# Formats where attendees build, not watch.
HANDS_ON = {"Workshop", "Builders' session", "Lab", "Bootcamp", "Gamified learning"}
TALKS = {"Breakout session", "Chalk talk", "Code talk", "Lightning talk", "Keynote", "Mainstage", "Booth theater"}

# Talks at the Neo4j booth theater, which aren't in the AWS catalog. The schedule is a draft.
BOOTH_THEATER = Path(__file__).with_name("booth-theater.csv")
BOOTH_ROOM = "Venetian | Level 2 | Hall B | Expo | Booth 732"

# Campus locations (approximate) for the map. Places like "Flight Club (located in The Venetian)"
# belong to the campus they name.
VENUES = {
    "Wynn/Encore": (36.1266, -115.1657),
    "Venetian": (36.1215, -115.1690),
    "Caesars Forum": (36.1183, -115.1675),
    "Caesars Palace": (36.1162, -115.1745),
    "MGM Grand": (36.1025, -115.1698),
    "Allegiant Stadium": (36.0909, -115.1833),
}
# Estimated door-to-door minutes between campuses, walking or by conference shuttle, including
# getting out of one building and into the next. These are planning estimates, not AWS guidance:
# edit them here and rebuild.
SAME_VENUE_MINUTES = 10
TRAVEL_MINUTES = {
    ("Wynn/Encore", "Venetian"): 20,
    ("Venetian", "Caesars Forum"): 20,
    ("Caesars Forum", "Caesars Palace"): 15,
    ("Venetian", "Caesars Palace"): 25,
    ("Wynn/Encore", "Caesars Forum"): 30,
    ("Wynn/Encore", "Caesars Palace"): 35,
    ("MGM Grand", "Caesars Palace"): 30,
    ("MGM Grand", "Caesars Forum"): 35,
    ("MGM Grand", "Venetian"): 40,
    ("MGM Grand", "Wynn/Encore"): 45,
    ("Allegiant Stadium", "MGM Grand"): 30,
    ("Allegiant Stadium", "Caesars Palace"): 35,
    ("Allegiant Stadium", "Caesars Forum"): 40,
    ("Allegiant Stadium", "Venetian"): 45,
    ("Allegiant Stadium", "Wynn/Encore"): 50,
}


def campus(place: str) -> str:
    """The campus a catalog venue belongs to: "Canonita (located in The Venetian)" is the Venetian."""
    for name in VENUES:
        if name.lower() in place.lower() or (name == "Wynn/Encore" and "wynn" in place.lower()):
            return name
    return place


def fold(text: str) -> str:
    """Lowercase text without accents: "Gołąb" becomes "golab"."""
    text = unicodedata.normalize("NFKD", text.lower().translate(str.maketrans({"ł": "l", "ø": "o", "đ": "d", "ß": "ss", "æ": "ae"})))
    return "".join(c for c in text if not unicodedata.combining(c))


def booth_sessions() -> list[dict]:
    """The Neo4j booth theater talks, in the catalog's shape. Each title gets a code (NEO01...), and each
    repeat of it a -R, -R1... suffix, so repeats share a Series like the catalog's do."""
    sessions, codes, runs = [], {}, {}
    with open(BOOTH_THEATER, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            title = row["title"].strip()
            base = codes.setdefault(title, f"NEO{len(codes) + 1:02d}")
            n = runs[title] = runs.get(title, -1) + 1
            code = base if n == 0 else f"{base}-R" if n == 1 else f"{base}-R{n - 1}"
            sessions.append({
                "id": f"booth-{code}", "code": code, "title": title, "type": "Booth theater", "level": "",
                "abstract": "A 30-minute talk at the Neo4j booth theater, booth 732 in the Venetian Expo (Hall B). "
                            "The booth schedule is a draft.",
                "topics": [], "services": [], "url": "",
                "speakers": [{"name": row["speaker"].strip(), "company": "Neo4j"}],
                "slots": [{"date": row["date"], "start": row["start"], "end": row["end"], "venue": "Venetian",
                           "room": BOOTH_ROOM}],
            })
    return sessions


def series_code(code: str) -> str:
    """Repeats of a session share a code up to a -R, -R1, -R2... suffix."""
    return re.sub(r"-R\d*$", "", code)


def kind(fmt: str) -> str:
    return "hands-on" if fmt in HANDS_ON else "talk" if fmt in TALKS else "other"


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


def embed_all(client, texts: list[str]) -> list[list[float]]:
    """Embed every text in a few large requests, in parallel. About 2,700 sessions take seconds."""
    batches = [texts[i:i + 500] for i in range(0, len(texts), 500)]
    with ThreadPoolExecutor(max_workers=len(batches)) as pool:
        results = pool.map(lambda b: client.embeddings.create(model=EMBED_MODEL, input=b).data, batches)
    return [d.embedding for batch in results for d in batch]


def load(path: Path, sessions: list[dict], embeddings: list[list[float]]) -> dict:
    db = relate.connect(str(path))
    try:
        for label, key in [("Session", "id"), ("Series", "code"), ("Venue", "name"), ("Topic", "name"),
                           ("Service", "name"), ("Speaker", "key")]:
            db.execute(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{key} IS UNIQUE")
        db.execute("UNWIND $rows AS row CREATE (:Venue {name: row.name, lat: row.lat, lon: row.lon})",
                   {"rows": [{"name": n, "lat": lat, "lon": lon} for n, (lat, lon) in VENUES.items()]})
        travel = [{"a": a, "b": b, "m": m} for (a, b), m in TRAVEL_MINUTES.items()]
        travel += [{"a": b, "b": a, "m": m} for (a, b), m in TRAVEL_MINUTES.items()]
        travel += [{"a": n, "b": n, "m": SAME_VENUE_MINUTES} for n in VENUES]
        db.execute("""
            UNWIND $rows AS row MATCH (a:Venue {name: row.a}), (b:Venue {name: row.b})
            CREATE (a)-[:TRAVEL {minutes: row.m}]->(b)
            """, {"rows": travel})
        rows = []
        for s, e in zip(sessions, embeddings):
            slot = (s.get("slots") or [{}])[0]  # Every record in this catalog has exactly one slot.
            rows.append({
                "id": s["id"], "code": s["code"], "series": series_code(s["code"]), "title": s["title"].strip(),
                "abstract": s.get("abstract") or "", "format": s.get("type") or "", "kind": kind(s.get("type") or ""),
                "level": s.get("level") or "", "text": session_text(s), "embedding": e,
                "url": s["url"] if "url" in s else CATALOG + "?" + urlencode({"search": s["code"]}),
                "date": slot.get("date", ""), "start": slot.get("start", ""), "venue": campus(slot.get("venue", "")),
            })
        for i in range(0, len(rows), 500):  # Relate caps one commit at 64 MB, so store the embeddings in batches.
            db.execute("""
                UNWIND $rows AS row
                CREATE (s:Session {id: row.id, code: row.code, title: row.title, abstract: row.abstract,
                                   format: row.format, kind: row.kind, level: row.level, url: row.url})
                MERGE (r:Series {code: row.series}) ON CREATE SET r.title = row.title
                CREATE (s)-[:REPEAT_OF]->(r)
                CREATE (:SessionChunk {text: row.text, embedding: vector(row.embedding, 1536, FLOAT32),
                                       date: row.date, start: row.start, venue: row.venue, kind: row.kind})
                       -[:FROM_DOCUMENT]->(s)
                """, {"rows": rows[i:i + 500]})
        slots = []
        for s in sessions:
            for i, sl in enumerate(s.get("slots", [])):
                day = date.fromisoformat(sl["date"])
                slots.append({**sl, "id": f"{s['id']}:{i}", "session": s["id"], "day": day.strftime("%a"),
                              "campus": campus(sl["venue"]),
                              "display": f"{day:%a} {day.day} {day:%b}, {sl['start']}–{sl['end']} PT"})
        db.execute("""
            UNWIND $rows AS row
            MATCH (s:Session {id: row.session})
            MERGE (v:Venue {name: row.campus})
            CREATE (s)-[:HAS_SLOT]->(:Slot {id: row.id, date: row.date, day: row.day, start: row.start, end: row.end,
                                            displayTime: row.display, room: row.room, place: row.venue})-[:AT_VENUE]->(v)
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
            FOR (c:SessionChunk) ON (c.embedding) WITH [c.date, c.venue, c.start, c.kind]
            OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}
            """)
        db.execute("CREATE FULLTEXT INDEX session_chunk_text IF NOT EXISTS FOR (c:SessionChunk) ON EACH [c.text]")
        return db.execute("""
            MATCH (s:Session) WITH count(s) AS sessions
            MATCH (r:Series) WITH sessions, count(r) AS series
            MATCH (c:SessionChunk) WHERE c.embedding IS NOT NULL
            RETURN sessions, series, count(c) AS embedded
            """).fetchall()[0]
    finally:
        db.close()


def build(client) -> None:
    started = time.time()
    sessions = fetch_sessions()
    booth = booth_sessions()
    print(f"Fetched {len(sessions)} sessions from {MIRROR}, plus {len(booth)} Neo4j booth theater talks "
          f"({time.time() - started:.1f}s)")
    sessions += booth
    embeddings = embed_all(client, [session_text(s) for s in sessions])
    print(f"Embedded {len(embeddings)} sessions with {EMBED_MODEL} ({time.time() - started:.1f}s)")
    with tempfile.TemporaryDirectory() as tmp:
        store = Path(tmp) / "store"
        print(f"Loaded: {load(store, sessions, embeddings)}")
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


# --- The graph at run time -------------------------------------------------------------------


class Graph:
    """The graph, opened read-only. A Relate connection only works on the thread that opened it,
    and agent frameworks run tools on worker threads, so one dedicated thread runs every query.
    Pass on_query to see each statement and how long it took."""

    def __init__(self, path: Path = DATABASE, on_query=None):
        if not path.exists():
            raise SystemExit(f"{path.name} not found. Build it once: uv run planner.py build")
        self.jobs = queue.Queue()
        # A daemon thread is still running when atexit handlers run (a ThreadPoolExecutor's isn't),
        # so close() can close the connection on its own thread before the interpreter shuts down.
        threading.Thread(target=self._work, daemon=True, name="relate").start()
        self.db = self._call(relate.connect, str(path), read_only=True)
        self.on_query = on_query
        atexit.register(self.close)

    def _work(self):
        while True:
            fn, done = self.jobs.get()
            try:
                done.set_result(fn())
            except BaseException as err:
                done.set_exception(err)

    def _call(self, fn, *args, **kwargs):
        done = Future()
        self.jobs.put((lambda: fn(*args, **kwargs), done))
        return done.result()

    def run(self, cypher: str, **params) -> list[dict]:
        started = time.time()
        rows = self._call(lambda: [row.data() for row in self.db.execute(cypher, params)])
        if self.on_query:
            self.on_query(cypher, params, len(rows), time.time() - started)
        return rows

    def close(self):
        if self.db is not None:
            db, self.db = self.db, None
            self._call(db.close)


def to_json(value) -> str:
    """Query results as JSON for the model, with nodes and relationships as their properties."""
    return json.dumps(value, default=lambda v: getattr(v, "properties", str(v)), ensure_ascii=False)


# What every session-listing query returns. Expects s, sl, and v in scope.
SESSION_FIELDS = """s.code AS code, s.title AS title, s.format AS format, s.kind AS kind, s.level AS level,
       sl.date AS date, sl.start AS start, sl.end AS end, sl.displayTime AS time, v.name AS venue, sl.room AS room,
       [(s)<-[:PRESENTS]-(sp:Speaker) | sp.name + CASE sp.company WHEN '' THEN '' ELSE ' (' + sp.company + ')' END] AS speakers,
       s.url AS link"""

# Hybrid search: a vector search that pre-filters inside the index, and a full-text search, fused by
# reciprocal rank (each list gives a session 1 / (60 + rank)). The full-text search can't filter
# inside its index, so it asks for more and filters after.
HYBRID_SEARCH = """
CALL () {
  MATCH (c:SessionChunk)
  SEARCH c IN (VECTOR INDEX session_chunk_embeddings FOR $embedding
               WHERE c.date >= $dateFrom AND c.date <= $dateTo AND c.venue IN $venues
                 AND c.start >= $after AND c.start < $before AND c.kind IN $kinds
               LIMIT $k) SCORE AS score
  WITH c ORDER BY score DESC
  WITH collect(c) AS cs
  UNWIND range(0, size(cs) - 1) AS i
  RETURN cs[i] AS c, 'vector' AS list, i + 1 AS rank
  UNION ALL
  MATCH (c:SessionChunk)
  SEARCH c IN (FULLTEXT INDEX session_chunk_text FOR $words LIMIT 400) SCORE AS score
  WITH c, score WHERE c.date >= $dateFrom AND c.date <= $dateTo AND c.venue IN $venues
                  AND c.start >= $after AND c.start < $before AND c.kind IN $kinds
  WITH c ORDER BY score DESC LIMIT $k
  WITH collect(c) AS cs
  UNWIND range(0, size(cs) - 1) AS i
  RETURN cs[i] AS c, 'fulltext' AS list, i + 1 AS rank
}
WITH c, sum(1.0 / (60 + rank)) AS rrf,
     min(CASE list WHEN 'vector' THEN rank END) AS vectorRank,
     min(CASE list WHEN 'fulltext' THEN rank END) AS fulltextRank
ORDER BY rrf DESC LIMIT $limit
MATCH (c)-[:FROM_DOCUMENT]->(s:Session)-[:HAS_SLOT]->(sl:Slot)-[:AT_VENUE]->(v:Venue)
RETURN """ + SESSION_FIELDS + """, substring(s.abstract, 0, 240) AS summary,
       [w IN $phrases WHERE toLower(c.text) CONTAINS w] AS matchedKeywords, vectorRank, fulltextRank
ORDER BY rrf DESC
"""

ALL_VENUES = list(VENUES) + ["Various Locations"]


def search_words(query: str, keywords: list[str] | None) -> str:
    """A full-text query from keywords (or the query): plain words only, so typed ^ or ~ can't break it."""
    words = " ".join(keywords) if keywords else query
    return " ".join(re.findall(r"[\w+#.-]+", words)) or "session"


def find_sessions(graph: Graph, embed, query: str, keywords: list[str] | None = None, date: str = "",
                  venue: str = "", kind: str = "", after: str = "", before: str = "", limit: int = 15) -> list[dict]:
    venues = [v for v in ALL_VENUES if venue.lower() in v.lower()] if venue else ALL_VENUES
    kinds = [kind] if kind in ("hands-on", "talk", "other") else ["hands-on", "talk", "other"]
    phrases = [k.lower().strip() for k in keywords or [] if k.strip()]
    return graph.run(HYBRID_SEARCH, embedding=embed(query), words=search_words(query, keywords), phrases=phrases,
                     dateFrom=date or "0000", dateTo=date or "9999", venues=venues or ["(none)"],
                     after=after or "00:00", before=before or "99:99", kinds=kinds, k=60, limit=max(1, min(limit, 40)))


def conference_days(graph: Graph) -> list[dict]:
    return graph.run("MATCH (sl:Slot) RETURN DISTINCT sl.date AS date, sl.day AS day ORDER BY date")


def travel_minutes(graph: Graph) -> dict[tuple[str, str], int]:
    rows = graph.run("MATCH (a:Venue)-[t:TRAVEL]->(b:Venue) RETURN a.name AS a, b.name AS b, t.minutes AS m")
    return {(r["a"], r["b"]): r["m"] for r in rows}
