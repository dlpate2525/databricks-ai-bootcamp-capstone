# AI Movie Night Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Databricks App where a group rates movies and an AI agent recommends one everyone will enjoy, retrieving semantically over TMDB text and writing its decisions back to Lakebase.

**Architecture:** A Spark notebook harvests TMDB into bronze/silver Delta tables, composes one text document per movie, embeds it via the Foundation Model API, and loads movies plus `vector(1024)` embeddings into Lakebase Postgres. A Streamlit Databricks App runs a tool-calling loop against `databricks-llama-4-maverick` with three read tools and three write tools, all backed by hybrid pgvector + SQL retrieval.

**Tech Stack:** Python 3.12, PySpark + Delta (Unity Catalog), Lakebase Postgres 16 + pgvector, `psycopg` (v3) with a pooled connection, Databricks Foundation Model APIs (`databricks-gte-large-en`, `databricks-llama-4-maverick`), Streamlit, pytest.

## Global Constraints

- **No secrets in the repo.** TMDB credentials live in secret scope `movie_night`, keys `tmdb-api-key` and `tmdb-read-token`. Read via `WorkspaceClient().secrets.get_secret()` or an `app.yaml` `valueFrom`. Never hardcode, never log, never commit.
- **Use the TMDB v4 read token only** (`Authorization: Bearer`). The v3 api key returns 401 on this account.
- **Embedding batch size is 8, maximum.** Batches of 16 fail with `429 REQUEST_LIMIT_EXCEEDED` at any spacing. Measured, not assumed.
- **Embedding dimension is 1024** (`databricks-gte-large-en`). DDL uses `vector(1024)`.
- **Target corpus: 1,000 movies** = 50 `discover` pages at `vote_count.gte=200`. (~3,000 enrichment requests, 125 embedding requests, ~6 min of embedding.)
- **Lakebase instance is `bootcamp-support-db`** (existing, `AVAILABLE`, PG 16, host `ep-bitter-sea-d8tty11q.database.us-east-2.cloud.databricks.com`). Schema `movie_night`. Do not create a second instance.
- **Lakebase auth: OAuth token as the Postgres password, expiring after ~1 hour.** Connections must mint a fresh token per connection; pool `max_lifetime` under 45 minutes.
- **No raw `requests`/`urllib` calls inside agent tool functions or Streamlit callbacks.** HTTP lives in `tmdb_client.py` and `embeddings.py`.
- **All SQL is parameterised.** No f-string or `%`-formatted values into SQL, ever — the agent chooses these arguments.
- **Free Edition caps the workspace at 3 apps.** Two are in use (`support-desk`, `weather-mcp`); the planner takes the third.
- **Databricks CLI profile is `dbc-ff09ef2e-7294`.** Run CLI commands from a neutral directory; a `databricks.yml` in the working tree overrides host resolution.
- **PowerShell 5.1 has no `&&`.** Chain with `;`.

---

## File Structure

The repo root **is** the Databricks App source directory, so `app.yaml` and `requirements.txt` sit at the top level. Shared pure logic lives in `movienight/` so the notebook and the app import the same code rather than duplicating it.

```
databricks-ai-bootcamp-capstone/
├── app.yaml                     App entrypoint + env (Task 9)
├── requirements.txt             App runtime deps (Task 9)
├── app.py                       Streamlit UI only — no SQL, no HTTP (Task 9)
├── setup_secrets.py             DONE
├── movienight/
│   ├── __init__.py
│   ├── tmdb_client.py           All TMDB HTTP + normalisation (Task 1)
│   ├── documents.py             Pure: compose_document, doc_hash (Task 2)
│   ├── embeddings.py            FM API embedding client, backoff (Task 3)
│   ├── db.py                    Lakebase pool + all SQL statements (Task 4)
│   ├── retrieval.py             Hybrid search query builder (Task 5)
│   ├── tools.py                 Tool schemas + dispatch + validation (Task 6)
│   └── agent.py                 Tool-calling loop (Task 7)
├── sql/
│   ├── 00_schema.sql            Tables (Task 4)
│   ├── 01_indexes.sql           HNSW + supporting indexes (Task 4)
│   └── 02_seed_demo.sql         Demo users, group, ratings (Task 4)
├── scripts/
│   ├── bootstrap_db.py          Apply sql/*.sql idempotently (Task 4)
│   └── e2e_test.py              End-to-end against deployed app (Task 10)
├── notebooks/
│   └── ingest_movies.py         Spark pipeline (Task 8)
├── tests/
│   ├── fixtures/                Captured TMDB JSON (Task 1)
│   ├── test_tmdb_client.py      (Task 1)
│   ├── test_documents.py        (Task 2)
│   ├── test_embeddings.py       (Task 3)
│   ├── test_retrieval.py        (Task 5)
│   ├── test_tools.py            (Task 6)
│   ├── test_agent.py            (Task 7)
│   └── test_retrieval_live.py   Discrimination test, needs DB (Task 10)
└── README.md                    (Task 10)
```

**Responsibility boundaries.** `tmdb_client.py` is the only module that knows TMDB's JSON shape. `documents.py` is pure text with no I/O so it is fast to test and cheap to iterate. `db.py` owns every SQL string; `retrieval.py` builds the one query complex enough to deserve its own module. `tools.py` is the trust boundary — it validates everything the model produces. `agent.py` knows nothing about movies, only about the tool-calling protocol.

---

## Task 1: TMDB client

**Files:**
- Create: `movienight/__init__.py`, `movienight/tmdb_client.py`
- Create: `tests/test_tmdb_client.py`, `tests/fixtures/movie_550.json`, `tests/fixtures/keywords_550.json`, `tests/fixtures/credits_550.json`, `tests/fixtures/release_dates_550.json`
- Create: `scripts/capture_fixtures.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `TMDBError(Exception)`
  - `TMDBClient(token: str, timeout: int = 20)`
  - `TMDBClient.discover_page(page: int, min_votes: int = 200) -> list[dict]` — raw result dicts
  - `TMDBClient.movie_bundle(movie_id: int) -> dict` — one merged raw dict with keys `detail`, `keywords`, `credits`, `release_dates`
  - `normalize_movie(bundle: dict) -> dict` — pure; returns the typed record below
  - `us_certification(release_dates: dict) -> str | None` — pure

The normalized record (this exact shape is what Tasks 2, 4, and 8 consume):

```python
{
  "movie_id": int, "title": str, "release_year": int | None,
  "tagline": str, "overview": str, "runtime": int | None,
  "certification": str | None,          # "PG-13", or None when unknown
  "genres": list[str], "keywords": list[str],
  "cast": list[str],                    # top 6 billed
  "director": str | None,
  "poster_path": str | None,            # TMDB relative path, e.g. "/abc.jpg"
  "vote_average": float, "vote_count": int, "popularity": float,
}
```

- [ ] **Step 1: Write the failing test**

Create `tests/test_tmdb_client.py`:

```python
import json
import pathlib

import pytest

from movienight.tmdb_client import normalize_movie, us_certification

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def load_bundle(movie_id):
    return {
        "detail": json.loads((FIXTURES / f"movie_{movie_id}.json").read_text()),
        "keywords": json.loads((FIXTURES / f"keywords_{movie_id}.json").read_text()),
        "credits": json.loads((FIXTURES / f"credits_{movie_id}.json").read_text()),
        "release_dates": json.loads((FIXTURES / f"release_dates_{movie_id}.json").read_text()),
    }


def test_normalize_extracts_typed_fields():
    rec = normalize_movie(load_bundle(550))
    assert rec["movie_id"] == 550
    assert rec["title"] == "Fight Club"
    assert rec["release_year"] == 1999
    assert rec["runtime"] == 139
    assert rec["certification"] == "R"
    assert "Drama" in rec["genres"]
    assert "nihilism" in rec["keywords"]
    assert rec["director"] == "David Fincher"
    assert len(rec["cast"]) <= 6
    assert rec["poster_path"].startswith("/")


def test_us_certification_returns_none_when_absent():
    assert us_certification({"results": []}) is None


def test_us_certification_ignores_non_us_and_blank():
    payload = {"results": [
        {"iso_3166_1": "GB", "release_dates": [{"certification": "18"}]},
        {"iso_3166_1": "US", "release_dates": [{"certification": ""},
                                               {"certification": "PG-13"}]},
    ]}
    assert us_certification(payload) == "PG-13"


def test_normalize_tolerates_missing_optional_fields():
    bundle = {"detail": {"id": 1, "title": "Bare"}, "keywords": {},
              "credits": {}, "release_dates": {}}
    rec = normalize_movie(bundle)
    assert rec["movie_id"] == 1 and rec["title"] == "Bare"
    assert rec["runtime"] is None and rec["certification"] is None
    assert rec["genres"] == [] and rec["keywords"] == [] and rec["cast"] == []
    assert rec["release_year"] is None
```

- [ ] **Step 2: Capture the fixtures**

Create `scripts/capture_fixtures.py`:

```python
"""Capture real TMDB responses as test fixtures. Run once."""
import json
import pathlib

from databricks.sdk import WorkspaceClient
from movienight.tmdb_client import TMDBClient, read_token

OUT = pathlib.Path(__file__).parent.parent / "tests" / "fixtures"
OUT.mkdir(parents=True, exist_ok=True)

client = TMDBClient(read_token(WorkspaceClient()))
for movie_id in (550, 27205, 862):
    bundle = client.movie_bundle(movie_id)
    for part, payload in bundle.items():
        (OUT / f"{part.replace('detail', 'movie')}_{movie_id}.json").write_text(
            json.dumps(payload, indent=2)
        )
    print(f"captured {movie_id}")
```

Run: `python scripts/capture_fixtures.py`
Expected: three sets of four JSON files under `tests/fixtures/`.

- [ ] **Step 3: Run the test to verify it fails**

Run: `python -m pytest tests/test_tmdb_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'movienight.tmdb_client'`

- [ ] **Step 4: Write the implementation**

Create `movienight/__init__.py` (empty) and `movienight/tmdb_client.py`:

```python
"""All TMDB HTTP and JSON normalisation. Nothing else in the codebase
knows TMDB's response shape."""

import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://api.themoviedb.org/3"
SCOPE = "movie_night"
IMAGE_BASE = "https://image.tmdb.org/t/p/w342"


class TMDBError(Exception):
    """A TMDB call failed in a way the caller should surface, not retry."""


def read_token(workspace_client):
    """Fetch the v4 read token from the secret scope. Never log the result."""
    raw = workspace_client.secrets.get_secret(scope=SCOPE, key="tmdb-read-token").value
    return base64.b64decode(raw).decode().strip()


class TMDBClient:
    def __init__(self, token, timeout=20, max_retries=3):
        self._headers = {"Authorization": f"Bearer {token}",
                         "accept": "application/json"}
        self._timeout = timeout
        self._max_retries = max_retries

    def _get(self, path, **params):
        url = f"{BASE}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        last = None
        for attempt in range(self._max_retries):
            req = urllib.request.Request(url, headers=self._headers)
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as r:
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                if e.code == 429:                       # TMDB throttle
                    time.sleep(float(e.headers.get("Retry-After", 2)))
                    last = e
                    continue
                if 500 <= e.code < 600:
                    time.sleep(2 ** attempt)
                    last = e
                    continue
                raise TMDBError(f"TMDB {path} returned {e.code}") from e
            except Exception as e:                      # timeout, DNS, reset
                time.sleep(2 ** attempt)
                last = e
        raise TMDBError(f"TMDB {path} failed after {self._max_retries} attempts: {last}")

    def discover_page(self, page, min_votes=200):
        body = self._get("/discover/movie", sort_by="popularity.desc",
                         include_adult="false", page=page,
                         **{"vote_count.gte": min_votes})
        return body.get("results", [])

    def movie_bundle(self, movie_id):
        return {
            "detail": self._get(f"/movie/{movie_id}"),
            "keywords": self._get(f"/movie/{movie_id}/keywords"),
            "credits": self._get(f"/movie/{movie_id}/credits"),
            "release_dates": self._get(f"/movie/{movie_id}/release_dates"),
        }


def us_certification(release_dates):
    """First non-blank US certification, or None. TMDB often has blanks."""
    for entry in (release_dates or {}).get("results", []):
        if entry.get("iso_3166_1") != "US":
            continue
        for release in entry.get("release_dates", []):
            cert = (release.get("certification") or "").strip()
            if cert:
                return cert
    return None


def normalize_movie(bundle):
    """Bundle of raw TMDB payloads -> one typed record. Pure."""
    detail = bundle.get("detail") or {}
    credits = bundle.get("credits") or {}

    date = (detail.get("release_date") or "")[:4]
    director = next(
        (c["name"] for c in credits.get("crew", []) if c.get("job") == "Director"),
        None,
    )

    return {
        "movie_id": detail.get("id"),
        "title": detail.get("title") or "",
        "release_year": int(date) if date.isdigit() else None,
        "tagline": detail.get("tagline") or "",
        "overview": detail.get("overview") or "",
        "runtime": detail.get("runtime") or None,
        "certification": us_certification(bundle.get("release_dates")),
        "genres": [g["name"] for g in detail.get("genres", [])],
        "keywords": [k["name"] for k in
                     (bundle.get("keywords") or {}).get("keywords", [])],
        "cast": [c["name"] for c in credits.get("cast", [])[:6]],
        "director": director,
        "poster_path": detail.get("poster_path"),
        "vote_average": float(detail.get("vote_average") or 0.0),
        "vote_count": int(detail.get("vote_count") or 0),
        "popularity": float(detail.get("popularity") or 0.0),
    }
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_tmdb_client.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add movienight/ tests/ scripts/capture_fixtures.py; git commit -m "feat: TMDB client with normalisation and fixtures"
```

---

## Task 2: Document composition

**Files:**
- Create: `movienight/documents.py`
- Create: `tests/test_documents.py`

**Interfaces:**
- Consumes: the normalized record from Task 1.
- Produces:
  - `compose_document(record: dict, reviews: list[str] | None = None) -> str`
  - `doc_hash(document: str) -> str` — 64-char hex SHA-256

This is the context-engineering core. The spec (§4) sets the format; the tests below pin it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_documents.py`:

```python
from movienight.documents import compose_document, doc_hash

FULL = {
    "movie_id": 550, "title": "Fight Club", "release_year": 1999,
    "tagline": "Mischief. Mayhem. Soap.",
    "overview": "An insomniac office worker forms an underground fight club.",
    "runtime": 139, "certification": "R",
    "genres": ["Drama", "Thriller"],
    "keywords": ["nihilism", "rage and hate", "dual identity"],
    "cast": ["Edward Norton", "Brad Pitt"], "director": "David Fincher",
    "poster_path": "/x.jpg", "vote_average": 8.4, "vote_count": 2000,
    "popularity": 60.0,
}


def test_document_contains_every_signal_field():
    doc = compose_document(FULL)
    for expected in ["Fight Club", "1999", "Mischief. Mayhem. Soap.",
                     "Drama", "Thriller", "insomniac", "nihilism",
                     "rage and hate", "Edward Norton", "David Fincher",
                     "139", "R"]:
        assert expected in doc, f"{expected!r} missing from composed document"


def test_document_omits_empty_sections_without_stray_labels():
    sparse = {**FULL, "tagline": "", "keywords": [], "cast": [],
              "director": None, "certification": None, "runtime": None}
    doc = compose_document(sparse)
    assert "Themes:" not in doc
    assert "Starring" not in doc
    assert "Rated" not in doc
    assert "Runtime" not in doc
    assert "Fight Club" in doc          # the real content survives


def test_reviews_are_included_and_capped_at_three():
    doc = compose_document(FULL, reviews=["one", "two", "three", "four"])
    assert "one" in doc and "three" in doc
    assert "four" not in doc


def test_review_excerpts_are_truncated():
    doc = compose_document(FULL, reviews=["x" * 900])
    assert "x" * 400 in doc
    assert "x" * 700 not in doc


def test_hash_is_stable_and_sensitive():
    a = compose_document(FULL)
    assert doc_hash(a) == doc_hash(compose_document(FULL))
    assert len(doc_hash(a)) == 64
    changed = compose_document({**FULL, "overview": "Something else entirely."})
    assert doc_hash(a) != doc_hash(changed)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_documents.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'movienight.documents'`

- [ ] **Step 3: Write the implementation**

Create `movienight/documents.py`:

```python
"""Compose one embeddable text document per movie.

TMDB overviews average ~295 characters, which retrieves poorly on its own:
a query like "funny sci-fi that isn't too violent" finds nothing to match
against. Keywords carry most of the tone signal ("nihilism" vs "friendship"),
so they are always included when present.

Pure text in, pure text out. No I/O, so this is cheap to iterate on - and
because embeddings are gated on doc_hash, changing this format only re-embeds
the movies whose text actually changed.
"""

import hashlib

MAX_REVIEWS = 3
REVIEW_CHARS = 500


def compose_document(record, reviews=None):
    lines = []

    year = f" ({record['release_year']})" if record.get("release_year") else ""
    lines.append(f"{record['title']}{year}.")

    if record.get("tagline"):
        lines.append(record["tagline"])

    if record.get("genres"):
        lines.append("Genres: " + ", ".join(record["genres"]) + ".")

    if record.get("overview"):
        lines.append(record["overview"])

    if record.get("keywords"):
        lines.append("Themes: " + ", ".join(record["keywords"]) + ".")

    people = []
    if record.get("cast"):
        people.append("Starring " + ", ".join(record["cast"]))
    if record.get("director"):
        people.append(f"Directed by {record['director']}")
    if people:
        lines.append(". ".join(people) + ".")

    facts = []
    if record.get("runtime"):
        facts.append(f"Runtime {record['runtime']} minutes")
    if record.get("certification"):
        facts.append(f"Rated {record['certification']}")
    if facts:
        lines.append(". ".join(facts) + ".")

    for review in (reviews or [])[:MAX_REVIEWS]:
        text = (review or "").strip()
        if text:
            lines.append(text[:REVIEW_CHARS])

    return "\n".join(lines)


def doc_hash(document):
    """Stable content hash. Gates re-embedding, so it must depend only on text."""
    return hashlib.sha256(document.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_documents.py -v`
Expected: 5 passed

- [ ] **Step 5: Eyeball one real document**

Run:
```bash
python -c "import json,pathlib; from movienight.tmdb_client import normalize_movie; from movienight.documents import compose_document; f=pathlib.Path('tests/fixtures'); b={k:json.loads((f/(k.replace('detail','movie')+'_550.json')).read_text()) for k in ['detail','keywords','credits','release_dates']}; print(compose_document(normalize_movie(b)))"
```
Expected: a readable paragraph, roughly 400–700 characters, containing themes and cast. If it reads like noise, fix the format now — every embedding downstream depends on it.

- [ ] **Step 6: Commit**

```bash
git add movienight/documents.py tests/test_documents.py; git commit -m "feat: composed movie documents with content hashing"
```

---

## Task 3: Embedding client

**Files:**
- Create: `movienight/embeddings.py`
- Create: `tests/test_embeddings.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `EmbeddingError(Exception)`
  - `EmbeddingClient(host: str, auth_headers: dict, endpoint: str = "databricks-gte-large-en")`
  - `EmbeddingClient.embed(texts: list[str]) -> list[list[float]]` — handles batching and backoff internally; returns one 1024-float vector per input, in order
  - `EmbeddingClient.embed_one(text: str) -> list[float]`
  - `MAX_BATCH = 8`, `DIM = 1024`
  - `client_from_workspace(workspace_client) -> EmbeddingClient`

The measured limits (batch ≤ 8, `429` above it, ~2s spacing) are the whole reason this module exists.

- [ ] **Step 1: Write the failing test**

Create `tests/test_embeddings.py`:

```python
import pytest

from movienight.embeddings import DIM, MAX_BATCH, EmbeddingClient, EmbeddingError


class FakeTransport:
    """Records calls and replays scripted responses, so no network is needed."""

    def __init__(self, script=None):
        self.batches = []
        self.script = list(script or [])
        self.sleeps = []

    def __call__(self, texts):
        self.batches.append(list(texts))
        if self.script:
            outcome = self.script.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
        return [[0.1] * DIM for _ in texts]


def make(transport, script=None):
    client = EmbeddingClient.__new__(EmbeddingClient)
    client._post = transport
    client._sleep = transport.sleeps.append
    client._max_retries = 5
    return client


def test_splits_into_batches_of_at_most_eight():
    t = FakeTransport()
    client = make(t)
    vectors = client.embed([f"doc {i}" for i in range(20)])
    assert len(vectors) == 20
    assert all(len(v) == DIM for v in vectors)
    assert [len(b) for b in t.batches] == [8, 8, 4]
    assert all(len(b) <= MAX_BATCH for b in t.batches)


def test_preserves_input_order_across_batches():
    class Ordered(FakeTransport):
        def __call__(self, texts):
            self.batches.append(list(texts))
            return [[float(int(t.split()[-1]))] * DIM for t in texts]

    t = Ordered()
    vectors = make(t).embed([f"doc {i}" for i in range(11)])
    assert [v[0] for v in vectors] == [float(i) for i in range(11)]


def test_retries_on_rate_limit_then_succeeds():
    t = FakeTransport(script=[EmbeddingError("429"), EmbeddingError("429"), None])
    client = make(t)
    vectors = client.embed(["a", "b"])
    assert len(vectors) == 2
    assert len(t.batches) == 3           # two failures then success
    assert t.sleeps == sorted(t.sleeps)  # backoff is non-decreasing
    assert t.sleeps[-1] > t.sleeps[0]


def test_gives_up_after_max_retries():
    t = FakeTransport(script=[EmbeddingError("429")] * 9)
    with pytest.raises(EmbeddingError):
        make(t).embed(["a"])


def test_empty_input_makes_no_calls():
    t = FakeTransport()
    assert make(t).embed([]) == []
    assert t.batches == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_embeddings.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'movienight.embeddings'`

- [ ] **Step 3: Write the implementation**

Create `movienight/embeddings.py`:

```python
"""Embedding client for the Foundation Model API.

Measured against this Free Edition workspace on 2026-08-09:
  batch 4, 2s gap  -> 6/6 OK
  batch 8, 2s gap  -> 6/6 OK   (~2.9 docs/sec)
  batch 16, 3s gap -> 0/5, all 429 REQUEST_LIMIT_EXCEEDED
  batch 256        -> 400 BAD_REQUEST

So MAX_BATCH is 8, and the limit's exact window is unknown - which is why the
backoff is adaptive rather than a fixed sleep. It degrades instead of failing
if the budget behaves differently on another day.
"""

import json
import time
import urllib.error
import urllib.request

MAX_BATCH = 8
DIM = 1024
BASE_GAP = 2.0
DEFAULT_ENDPOINT = "databricks-gte-large-en"


class EmbeddingError(Exception):
    """Embedding failed after retries. Callers should stop, not silently skip."""


class EmbeddingClient:
    def __init__(self, host, auth_headers, endpoint=DEFAULT_ENDPOINT,
                 max_retries=6, timeout=120):
        self._url = f"{host.rstrip('/')}/serving-endpoints/{endpoint}/invocations"
        self._headers = {**auth_headers, "Content-Type": "application/json"}
        self._timeout = timeout
        self._max_retries = max_retries
        self._sleep = time.sleep

    def _post(self, texts):
        req = urllib.request.Request(
            self._url, data=json.dumps({"input": list(texts)}).encode(),
            headers=self._headers, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as r:
                body = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:200]
            raise EmbeddingError(f"HTTP {e.code}: {detail}") from e
        except Exception as e:
            raise EmbeddingError(f"{type(e).__name__}: {e}") from e

        rows = sorted(body["data"], key=lambda d: d.get("index", 0))
        return [row["embedding"] for row in rows]

    def _post_with_backoff(self, batch):
        delay = BASE_GAP
        last = None
        for attempt in range(self._max_retries):
            try:
                return self._post(batch)
            except EmbeddingError as e:
                last = e
                self._sleep(delay)
                delay = min(delay * 2, 60.0)
        raise EmbeddingError(f"gave up after {self._max_retries} attempts: {last}")

    def embed(self, texts):
        texts = list(texts)
        if not texts:
            return []
        out = []
        for start in range(0, len(texts), MAX_BATCH):
            batch = texts[start:start + MAX_BATCH]
            out.extend(self._post_with_backoff(batch))
            if start + MAX_BATCH < len(texts):
                self._sleep(BASE_GAP)     # stay under the budget on the happy path
        return out

    def embed_one(self, text):
        return self.embed([text])[0]


def client_from_workspace(workspace_client, endpoint=DEFAULT_ENDPOINT):
    return EmbeddingClient(
        host=workspace_client.config.host,
        auth_headers=workspace_client.config.authenticate(),
        endpoint=endpoint,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_embeddings.py -v`
Expected: 5 passed

- [ ] **Step 5: Verify against the real endpoint once**

Run:
```bash
python -c "from databricks.sdk import WorkspaceClient; from movienight.embeddings import client_from_workspace, DIM; c=client_from_workspace(WorkspaceClient()); v=c.embed(['a funny space comedy','a brutal war film']); print(len(v), len(v[0]), len(v[0])==DIM)"
```
Expected: `2 1024 True`

- [ ] **Step 6: Commit**

```bash
git add movienight/embeddings.py tests/test_embeddings.py; git commit -m "feat: embedding client with measured batch cap and adaptive backoff"
```

---

## Task 4: Lakebase schema and connection layer

**Files:**
- Create: `sql/00_schema.sql`, `sql/01_indexes.sql`, `sql/02_seed_demo.sql`
- Create: `movienight/db.py`
- Create: `scripts/bootstrap_db.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `get_pool() -> ConnectionPool` — module-level singleton, tokens minted per connection
  - `connection()` — context manager yielding a `psycopg` connection with `dict_row`
  - `SCHEMA = "movie_night"`

**Preflight:** confirm `pgvector` is installable on this instance before writing DDL that depends on it. The weather assignment used pgvector 0.8.0 on Lakebase, but verify rather than assume.

- [ ] **Step 1: Write the schema DDL**

Create `sql/00_schema.sql`:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS movie_night;
SET search_path TO movie_night, public;

CREATE TABLE IF NOT EXISTS users (
    user_id     bigserial PRIMARY KEY,
    email       text UNIQUE NOT NULL,
    display_name text NOT NULL,
    is_demo     boolean NOT NULL DEFAULT false,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS groups (
    group_id    bigserial PRIMARY KEY,
    name        text NOT NULL,
    created_by  bigint REFERENCES users(user_id),
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS group_members (
    group_id    bigint NOT NULL REFERENCES groups(group_id) ON DELETE CASCADE,
    user_id     bigint NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    joined_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (group_id, user_id)
);

CREATE TABLE IF NOT EXISTS movies (
    movie_id      bigint PRIMARY KEY,          -- TMDB id, natural key
    title         text NOT NULL,
    release_year  int,
    tagline       text NOT NULL DEFAULT '',
    overview      text NOT NULL DEFAULT '',
    runtime       int,
    certification text,                        -- NULL = unknown, not "safe"
    genres        text[] NOT NULL DEFAULT '{}',
    keywords      text[] NOT NULL DEFAULT '{}',
    cast_names    text[] NOT NULL DEFAULT '{}',
    director      text,
    poster_path   text,
    vote_average  real NOT NULL DEFAULT 0,
    vote_count    int  NOT NULL DEFAULT 0,
    popularity    real NOT NULL DEFAULT 0,
    updated_at    timestamptz NOT NULL DEFAULT now()
);

-- Separate table so re-embedding never rewrites movies, and so a movie can
-- exist before its vector does (the pipeline is resumable).
CREATE TABLE IF NOT EXISTS movie_embeddings (
    movie_id   bigint PRIMARY KEY REFERENCES movies(movie_id) ON DELETE CASCADE,
    document   text NOT NULL,
    doc_sha256 char(64) NOT NULL,     -- gates re-embedding on content change
    embedding  vector(1024) NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ratings (
    user_id    bigint NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    movie_id   bigint NOT NULL REFERENCES movies(movie_id) ON DELETE CASCADE,
    score      int NOT NULL CHECK (score BETWEEN 1 AND 10),
    rated_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, movie_id)   -- re-rating is an UPSERT, never a duplicate
);

CREATE TABLE IF NOT EXISTS watchlist_items (
    item_id    bigserial PRIMARY KEY,
    group_id   bigint NOT NULL REFERENCES groups(group_id) ON DELETE CASCADE,
    movie_id   bigint NOT NULL REFERENCES movies(movie_id) ON DELETE CASCADE,
    status     text NOT NULL DEFAULT 'queued'
               CHECK (status IN ('queued', 'watched', 'skipped')),
    reason     text,
    added_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (group_id, movie_id)       -- makes add_to_watchlist idempotent
);

-- Audit trail of what the agent did and why.
CREATE TABLE IF NOT EXISTS recommendations (
    rec_id            bigserial PRIMARY KEY,
    group_id          bigint NOT NULL REFERENCES groups(group_id) ON DELETE CASCADE,
    user_query        text NOT NULL,
    candidate_ids     bigint[] NOT NULL DEFAULT '{}',
    chosen_movie_id   bigint REFERENCES movies(movie_id),
    rationale         text,
    created_at        timestamptz NOT NULL DEFAULT now()
);
```

Create `sql/01_indexes.sql`:

```sql
SET search_path TO movie_night, public;

-- Approximate NN search. m/ef_construction are pgvector defaults; at 5k rows
-- the build is seconds.
CREATE INDEX IF NOT EXISTS movie_embeddings_hnsw
    ON movie_embeddings USING hnsw (embedding vector_cosine_ops);

-- Filters applied alongside the vector ordering (see retrieval.py).
CREATE INDEX IF NOT EXISTS movies_runtime_idx      ON movies (runtime);
CREATE INDEX IF NOT EXISTS movies_year_idx         ON movies (release_year);
CREATE INDEX IF NOT EXISTS movies_cert_idx         ON movies (certification);
CREATE INDEX IF NOT EXISTS movies_genres_gin       ON movies USING gin (genres);
CREATE INDEX IF NOT EXISTS ratings_movie_idx       ON ratings (movie_id);
CREATE INDEX IF NOT EXISTS watchlist_group_idx     ON watchlist_items (group_id);
```

Create `sql/02_seed_demo.sql` — a Free Edition workspace has one real account, so a
group needs demo members or "please everyone" is vacuous:

```sql
SET search_path TO movie_night, public;

INSERT INTO users (email, display_name, is_demo) VALUES
    ('ava@example.com',  'Ava',  true),
    ('ben@example.com',  'Ben',  true),
    ('cleo@example.com', 'Cleo', true)
ON CONFLICT (email) DO NOTHING;

INSERT INTO groups (name, created_by)
SELECT 'Friday Movie Night', (SELECT user_id FROM users WHERE email='ava@example.com')
WHERE NOT EXISTS (SELECT 1 FROM groups WHERE name = 'Friday Movie Night');

INSERT INTO group_members (group_id, user_id)
SELECT g.group_id, u.user_id
FROM groups g CROSS JOIN users u
WHERE g.name = 'Friday Movie Night' AND u.is_demo
ON CONFLICT DO NOTHING;
```

- [ ] **Step 2: Write the connection layer**

Create `movienight/db.py`:

```python
"""Lakebase connection management.

Lakebase authenticates with an OAuth token used as the Postgres password, and
that token expires after about an hour. A long-lived pool holding one password
therefore dies silently mid-session. The fix is a connection class that mints a
fresh token every time the pool opens a connection, plus a max_lifetime well
under the expiry so connections are recycled before their credential dies.
"""

import os
from contextlib import contextmanager

from databricks.sdk import WorkspaceClient
from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

SCHEMA = "movie_night"
INSTANCE = os.environ.get("LAKEBASE_INSTANCE", "bootcamp-support-db")
DATABASE = os.environ.get("LAKEBASE_DATABASE", "databricks_postgres")

_workspace = None
_pool = None


def _ws():
    global _workspace
    if _workspace is None:
        _workspace = WorkspaceClient()
    return _workspace


class _TokenConnection(Connection):
    """Mints a fresh OAuth token for each new physical connection."""

    @classmethod
    def connect(cls, conninfo="", **kwargs):
        w = _ws()
        instance = w.database.get_database_instance(name=INSTANCE)
        cred = w.database.generate_database_credential(
            request_id=os.urandom(8).hex(), instance_names=[INSTANCE]
        )
        kwargs.update(
            host=instance.read_write_dns,
            port=5432,
            dbname=DATABASE,
            user=w.current_user.me().user_name,
            password=cred.token,
            sslmode="require",
            options=f"-c search_path={SCHEMA},public",
        )
        return super().connect("", **kwargs)


def get_pool():
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo="",
            connection_class=_TokenConnection,
            kwargs={"row_factory": dict_row},
            min_size=1,
            max_size=4,
            max_lifetime=45 * 60,      # under the ~60 min token expiry
            open=True,
        )
    return _pool


@contextmanager
def connection():
    with get_pool().connection() as conn:
        yield conn
```

- [ ] **Step 3: Write the bootstrap script**

Create `scripts/bootstrap_db.py`:

```python
"""Apply sql/*.sql in order. Idempotent - safe to re-run."""
import pathlib
import sys

from movienight.db import connection

SQL_DIR = pathlib.Path(__file__).parent.parent / "sql"


def main():
    files = sorted(SQL_DIR.glob("*.sql"))
    if not files:
        print("no SQL files found")
        return 1
    with connection() as conn:
        for path in files:
            print(f"applying {path.name} ...", end=" ")
            conn.execute(path.read_text())
            conn.commit()
            print("ok")

        counts = {}
        for table in ["users", "groups", "group_members", "movies",
                      "movie_embeddings", "ratings", "watchlist_items",
                      "recommendations"]:
            row = conn.execute(f"SELECT count(*) AS n FROM {table}").fetchone()
            counts[table] = row["n"]
    print("\nrow counts:")
    for table, n in counts.items():
        print(f"  {table:20s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Verify pgvector is available, then bootstrap**

Run:
```bash
python -c "from movienight.db import connection; c=connection().__enter__(); print(c.execute(\"SELECT * FROM pg_available_extensions WHERE name='vector'\").fetchall())"
```
Expected: one row naming `vector` with a version. **If this returns empty, stop** — the design's retrieval depends on pgvector and an alternative must be chosen before continuing.

Then run: `python scripts/bootstrap_db.py`
Expected: each file reports `ok`; `users` = 3, `groups` = 1, `group_members` = 3, everything else 0.

- [ ] **Step 5: Verify idempotency**

Run: `python scripts/bootstrap_db.py`
Expected: identical output and identical row counts — no duplicate demo users, no errors.

- [ ] **Step 6: Commit**

```bash
git add sql/ movienight/db.py scripts/bootstrap_db.py; git commit -m "feat: Lakebase schema, pooled OAuth connections, idempotent bootstrap"
```

---

## Task 5: Hybrid retrieval

**Files:**
- Create: `movienight/retrieval.py`
- Create: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: `movienight.db.connection`, `movienight.embeddings`.
- Produces:
  - `SearchFilters` dataclass: `max_runtime_minutes: int | None`, `min_runtime_minutes: int | None`, `exclude_violent: bool`, `min_year: int | None`, `max_year: int | None`, `genres: list[str] | None`, `exclude_movie_ids: list[int]`
  - `build_search_sql(filters: SearchFilters, k: int) -> tuple[str, dict]` — pure; returns SQL and a params dict
  - `search(conn, query_vector: list[float], filters: SearchFilters, k: int = 8) -> list[dict]`
  - `VIOLENT_CERTS = ("R", "NC-17")`

The point of splitting `build_search_sql` out as a pure function is that the SQL shape becomes testable without a database.

- [ ] **Step 1: Write the failing test**

Create `tests/test_retrieval.py`:

```python
from movienight.retrieval import SearchFilters, build_search_sql


def test_no_filters_produces_only_vector_ordering():
    sql, params = build_search_sql(SearchFilters(), k=5)
    assert "ORDER BY" in sql and "<=>" in sql
    assert "runtime" not in sql
    assert "certification" not in sql
    assert params["k"] == 5


def test_runtime_filter_appears_and_is_parameterised():
    sql, params = build_search_sql(
        SearchFilters(max_runtime_minutes=120), k=5)
    assert "m.runtime <= %(max_runtime)s" in sql
    assert params["max_runtime"] == 120
    assert "120" not in sql          # value must not be interpolated


def test_exclude_violent_filters_certs_and_treats_null_as_unknown():
    sql, params = build_search_sql(SearchFilters(exclude_violent=True), k=5)
    assert "certification" in sql
    assert "IS NULL" in sql, "NULL certification must be excluded, not assumed safe"
    assert params["violent_certs"] == ("R", "NC-17")


def test_exclusions_are_passed_as_a_parameter():
    sql, params = build_search_sql(
        SearchFilters(exclude_movie_ids=[1, 2, 3]), k=5)
    assert "%(exclude_ids)s" in sql
    assert params["exclude_ids"] == [1, 2, 3]


def test_empty_exclusion_list_adds_no_clause():
    sql, _ = build_search_sql(SearchFilters(exclude_movie_ids=[]), k=5)
    assert "exclude_ids" not in sql


def test_genre_filter_uses_array_overlap():
    sql, params = build_search_sql(SearchFilters(genres=["Comedy"]), k=5)
    assert "m.genres && %(genres)s" in sql
    assert params["genres"] == ["Comedy"]


def test_overfetch_multiplier_applied_to_candidate_limit():
    sql, params = build_search_sql(SearchFilters(max_runtime_minutes=90), k=5)
    assert params["candidate_limit"] >= 50, "filters need over-fetch headroom"


def test_year_range_both_bounds():
    sql, params = build_search_sql(
        SearchFilters(min_year=1990, max_year=1999), k=5)
    assert params["min_year"] == 1990 and params["max_year"] == 1999
    assert "m.release_year >= %(min_year)s" in sql
    assert "m.release_year <= %(max_year)s" in sql
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_retrieval.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'movienight.retrieval'`

- [ ] **Step 3: Write the implementation**

Create `movienight/retrieval.py`:

```python
"""Hybrid retrieval: semantic ranking plus structured filters, one query.

"A funny sci-fi movie that isn't too violent and is under two hours" splits
into parts that need different mechanisms. "Funny sci-fi" is semantic. "Under
two hours" is runtime < 120 and no embedding will ever enforce it. Doing only
the vector half returns plausible-looking wrong answers.

HNSW is approximate and Postgres applies WHERE *after* the index returns
candidates, so a narrow filter can yield fewer than k rows even when matches
exist. Hence over-fetching a candidate pool and, if that still comes up short,
falling back to an exact scan - at 5k rows exact is still fast, so correctness
is never traded for speed.
"""

from dataclasses import dataclass, field

VIOLENT_CERTS = ("R", "NC-17")
OVERFETCH = 10
MIN_CANDIDATES = 50


@dataclass
class SearchFilters:
    max_runtime_minutes: int | None = None
    min_runtime_minutes: int | None = None
    exclude_violent: bool = False
    min_year: int | None = None
    max_year: int | None = None
    genres: list[str] | None = None
    exclude_movie_ids: list[int] = field(default_factory=list)


def build_search_sql(filters, k):
    """Pure. Returns (sql, params). Every value is a parameter, never inlined."""
    where = []
    params = {
        "k": k,
        "candidate_limit": max(k * OVERFETCH, MIN_CANDIDATES),
    }

    if filters.max_runtime_minutes is not None:
        where.append("m.runtime IS NOT NULL AND m.runtime <= %(max_runtime)s")
        params["max_runtime"] = filters.max_runtime_minutes

    if filters.min_runtime_minutes is not None:
        where.append("m.runtime IS NOT NULL AND m.runtime >= %(min_runtime)s")
        params["min_runtime"] = filters.min_runtime_minutes

    if filters.exclude_violent:
        # NULL certification means unknown. Treating unknown as safe would let
        # unrated films leak into a "not too violent" result, so exclude both.
        where.append(
            "m.certification IS NOT NULL "
            "AND m.certification <> ALL (%(violent_certs)s)"
        )
        params["violent_certs"] = VIOLENT_CERTS

    if filters.min_year is not None:
        where.append("m.release_year >= %(min_year)s")
        params["min_year"] = filters.min_year

    if filters.max_year is not None:
        where.append("m.release_year <= %(max_year)s")
        params["max_year"] = filters.max_year

    if filters.genres:
        where.append("m.genres && %(genres)s")
        params["genres"] = list(filters.genres)

    if filters.exclude_movie_ids:
        where.append("m.movie_id <> ALL (%(exclude_ids)s)")
        params["exclude_ids"] = list(filters.exclude_movie_ids)

    clause = ("WHERE " + " AND ".join(where)) if where else ""

    sql = f"""
        WITH candidates AS (
            SELECT e.movie_id, e.embedding <=> %(qvec)s::vector AS distance
            FROM movie_embeddings e
            ORDER BY e.embedding <=> %(qvec)s::vector
            LIMIT %(candidate_limit)s
        )
        SELECT m.movie_id, m.title, m.release_year, m.tagline, m.overview,
               m.runtime, m.certification, m.genres, m.keywords,
               m.cast_names, m.director, m.poster_path, m.vote_average,
               1 - c.distance AS similarity
        FROM candidates c
        JOIN movies m USING (movie_id)
        {clause}
        ORDER BY c.distance
        LIMIT %(k)s
    """
    return sql, params


def _exact_sql(filters, k):
    """Same filters, no candidate pre-limit. Used when HNSW under-delivers."""
    sql, params = build_search_sql(filters, k)
    sql = sql.replace("LIMIT %(candidate_limit)s", "")
    return sql, params


def search(conn, query_vector, filters=None, k=8):
    filters = filters or SearchFilters()
    sql, params = build_search_sql(filters, k)
    params["qvec"] = str(query_vector)
    rows = conn.execute(sql, params).fetchall()

    if len(rows) < k:
        # Approximate index plus post-filtering came up short; redo exactly.
        sql, params = _exact_sql(filters, k)
        params["qvec"] = str(query_vector)
        rows = conn.execute(sql, params).fetchall()

    return rows
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_retrieval.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add movienight/retrieval.py tests/test_retrieval.py; git commit -m "feat: hybrid pgvector + SQL retrieval with exact-scan fallback"
```

---

## Task 6: Agent tools

**Files:**
- Create: `movienight/tools.py`
- Create: `tests/test_tools.py`

**Interfaces:**
- Consumes: `movienight.retrieval`, `movienight.db`, `movienight.embeddings`.
- Produces:
  - `TOOL_SCHEMAS: list[dict]` — OpenAI-format function definitions
  - `ToolContext(conn, embedder, group_id: int, user_id: int)` — the trust anchor; `group_id`/`user_id` come from session state, never from the model
  - `dispatch(name: str, arguments: dict, ctx: ToolContext) -> dict` — returns `{"status": "ok", ...}` or `{"status": "error", "message": ...}`, never raises
  - `ToolError(Exception)` — internal only

This module is the trust boundary. Everything here assumes the arguments were chosen by a language model and may be wrong, malformed, or invented.

- [ ] **Step 1: Write the failing test**

Create `tests/test_tools.py`:

```python
import pytest

from movienight.tools import TOOL_SCHEMAS, ToolContext, dispatch


class FakeCursor:
    def __init__(self, rows): self._rows = rows
    def fetchall(self): return self._rows
    def fetchone(self): return self._rows[0] if self._rows else None


class FakeConn:
    """Records SQL and params so tests can assert parameterisation."""

    def __init__(self, responses=None):
        self.calls = []
        self.responses = responses or {}
        self.committed = False

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        for needle, rows in self.responses.items():
            if needle in sql:
                return FakeCursor(rows)
        return FakeCursor([])

    def commit(self): self.committed = True


class FakeEmbedder:
    def embed_one(self, text): return [0.0] * 1024


def ctx(conn):
    return ToolContext(conn=conn, embedder=FakeEmbedder(), group_id=7, user_id=42)


def test_every_schema_is_wellformed_and_dispatchable():
    names = set()
    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] and fn["description"]
        assert fn["parameters"]["type"] == "object"
        names.add(fn["name"])
    assert names == {"search_movies", "get_group_context", "compare_movies",
                     "add_to_watchlist", "record_rating", "save_recommendation"}


def test_unknown_tool_returns_error_not_exception():
    result = dispatch("drop_tables", {}, ctx(FakeConn()))
    assert result["status"] == "error"
    assert "unknown" in result["message"].lower()


def test_add_to_watchlist_rejects_unknown_movie_id():
    conn = FakeConn(responses={"FROM movies WHERE movie_id": []})
    result = dispatch("add_to_watchlist", {"movie_id": 999999}, ctx(conn))
    assert result["status"] == "error"
    assert "999999" in result["message"]
    assert not conn.committed, "must not write when the id does not exist"


def test_add_to_watchlist_uses_context_group_not_model_supplied_one():
    conn = FakeConn(responses={"FROM movies WHERE movie_id": [{"movie_id": 5,
                                                               "title": "X"}]})
    dispatch("add_to_watchlist", {"movie_id": 5, "group_id": 99999}, ctx(conn))
    insert = [c for c in conn.calls if "INSERT INTO watchlist_items" in c[0]]
    assert insert, "expected an insert"
    assert insert[0][1]["group_id"] == 7, "group_id must come from ToolContext"


def test_record_rating_clamps_score_out_of_range():
    conn = FakeConn(responses={"FROM movies WHERE movie_id": [{"movie_id": 5,
                                                               "title": "X"}]})
    dispatch("record_rating", {"movie_id": 5, "score": 47}, ctx(conn))
    insert = [c for c in conn.calls if "INSERT INTO ratings" in c[0]]
    assert insert[0][1]["score"] == 10
    dispatch("record_rating", {"movie_id": 5, "score": -3}, ctx(conn))
    insert = [c for c in conn.calls if "INSERT INTO ratings" in c[0]]
    assert insert[-1][1]["score"] == 1


def test_record_rating_rejects_non_numeric_score():
    conn = FakeConn(responses={"FROM movies WHERE movie_id": [{"movie_id": 5}]})
    result = dispatch("record_rating", {"movie_id": 5, "score": "great"}, ctx(conn))
    assert result["status"] == "error"


def test_search_movies_missing_query_is_an_error():
    result = dispatch("search_movies", {}, ctx(FakeConn()))
    assert result["status"] == "error"


def test_writes_never_interpolate_values_into_sql():
    conn = FakeConn(responses={"FROM movies WHERE movie_id": [{"movie_id": 5}]})
    dispatch("record_rating", {"movie_id": 5, "score": 8}, ctx(conn))
    for sql, params in conn.calls:
        assert "'" not in sql.replace("'{}'", ""), f"literal quote in SQL: {sql}"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_tools.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'movienight.tools'`

- [ ] **Step 3: Write the implementation**

Create `movienight/tools.py`:

```python
"""Agent tools - the trust boundary.

Every argument here was chosen by a language model, so it may be malformed,
out of range, or reference a movie that does not exist. Each tool therefore:
validates ids against the catalog before writing, uses only parameterised SQL,
is idempotent so a retry cannot double-write, clamps numeric ranges, and takes
group_id/user_id from ToolContext rather than from the model.

Tools return dicts and never raise; the agent needs a sentence it can relay.
"""

from dataclasses import dataclass

from .retrieval import SearchFilters, search

MAX_COMPARE = 4
MAX_K = 12


@dataclass
class ToolContext:
    conn: object
    embedder: object
    group_id: int
    user_id: int


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "search_movies",
        "description": (
            "Semantic search over the movie catalog with structured filters. "
            "Use for any request about what to watch. Movies already watched "
            "or disliked by the group are excluded automatically."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string",
                      "description": "Mood, theme, or plot in natural language"},
            "max_runtime_minutes": {"type": "integer"},
            "min_year": {"type": "integer"},
            "genres": {"type": "array", "items": {"type": "string"}},
            "exclude_violent": {"type": "boolean"},
            "k": {"type": "integer", "description": "How many results, max 12"},
        }, "required": ["query"]}}},

    {"type": "function", "function": {
        "name": "get_group_context",
        "description": (
            "Members of the current group, their ratings, and what they have "
            "already watched or disliked. Call this before recommending."),
        "parameters": {"type": "object", "properties": {}}}},

    {"type": "function", "function": {
        "name": "compare_movies",
        "description": "Side-by-side facts for 2-4 movies by id.",
        "parameters": {"type": "object", "properties": {
            "movie_ids": {"type": "array", "items": {"type": "integer"}},
        }, "required": ["movie_ids"]}}},

    {"type": "function", "function": {
        "name": "add_to_watchlist",
        "description": "Add a movie to the group's watchlist. Idempotent.",
        "parameters": {"type": "object", "properties": {
            "movie_id": {"type": "integer"},
            "reason": {"type": "string"},
        }, "required": ["movie_id"]}}},

    {"type": "function", "function": {
        "name": "record_rating",
        "description": "Record the signed-in user's 1-10 rating for a movie.",
        "parameters": {"type": "object", "properties": {
            "movie_id": {"type": "integer"},
            "score": {"type": "integer"},
        }, "required": ["movie_id", "score"]}}},

    {"type": "function", "function": {
        "name": "save_recommendation",
        "description": (
            "Record the final pick and the reasoning behind it, for the "
            "group's history."),
        "parameters": {"type": "object", "properties": {
            "user_query": {"type": "string"},
            "candidate_ids": {"type": "array", "items": {"type": "integer"}},
            "chosen_movie_id": {"type": "integer"},
            "rationale": {"type": "string"},
        }, "required": ["user_query", "chosen_movie_id", "rationale"]}}},
]


def _err(message):
    return {"status": "error", "message": message}


def _movie_exists(ctx, movie_id):
    row = ctx.conn.execute(
        "SELECT movie_id, title FROM movies WHERE movie_id = %(id)s",
        {"id": movie_id},
    ).fetchone()
    return row


def _as_int(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _excluded_ids(ctx):
    """Movies the group has watched, skipped, or any member rated <= 4."""
    rows = ctx.conn.execute(
        """
        SELECT movie_id FROM watchlist_items
        WHERE group_id = %(g)s AND status IN ('watched', 'skipped')
        UNION
        SELECT r.movie_id FROM ratings r
        JOIN group_members gm ON gm.user_id = r.user_id
        WHERE gm.group_id = %(g)s AND r.score <= 4
        """,
        {"g": ctx.group_id},
    ).fetchall()
    return [r["movie_id"] for r in rows]


def _tool_search_movies(args, ctx):
    query = (args.get("query") or "").strip()
    if not query:
        return _err("search_movies needs a non-empty 'query'.")

    k = _as_int(args.get("k")) or 8
    k = max(1, min(k, MAX_K))

    filters = SearchFilters(
        max_runtime_minutes=_as_int(args.get("max_runtime_minutes")),
        min_year=_as_int(args.get("min_year")),
        genres=args.get("genres") or None,
        exclude_violent=bool(args.get("exclude_violent")),
        exclude_movie_ids=_excluded_ids(ctx),
    )
    vector = ctx.embedder.embed_one(query)
    rows = search(ctx.conn, vector, filters, k=k)
    return {"status": "ok", "count": len(rows), "results": rows}


def _tool_get_group_context(args, ctx):
    members = ctx.conn.execute(
        """
        SELECT u.user_id, u.display_name, u.is_demo
        FROM group_members gm JOIN users u USING (user_id)
        WHERE gm.group_id = %(g)s ORDER BY u.display_name
        """, {"g": ctx.group_id}).fetchall()

    ratings = ctx.conn.execute(
        """
        SELECT u.display_name, m.movie_id, m.title, r.score
        FROM ratings r
        JOIN group_members gm ON gm.user_id = r.user_id AND gm.group_id = %(g)s
        JOIN users u ON u.user_id = r.user_id
        JOIN movies m ON m.movie_id = r.movie_id
        ORDER BY r.score DESC
        """, {"g": ctx.group_id}).fetchall()

    watchlist = ctx.conn.execute(
        """
        SELECT w.movie_id, m.title, w.status
        FROM watchlist_items w JOIN movies m USING (movie_id)
        WHERE w.group_id = %(g)s ORDER BY w.added_at DESC
        """, {"g": ctx.group_id}).fetchall()

    return {"status": "ok", "members": members, "ratings": ratings,
            "watchlist": watchlist}


def _tool_compare_movies(args, ctx):
    ids = [i for i in (_as_int(x) for x in args.get("movie_ids") or []) if i]
    if not 2 <= len(ids) <= MAX_COMPARE:
        return _err(f"compare_movies needs between 2 and {MAX_COMPARE} movie ids.")
    rows = ctx.conn.execute(
        """
        SELECT movie_id, title, release_year, runtime, certification,
               genres, keywords, director, vote_average, overview
        FROM movies WHERE movie_id = ANY(%(ids)s)
        """, {"ids": ids}).fetchall()
    found = {r["movie_id"] for r in rows}
    missing = [i for i in ids if i not in found]
    if missing:
        return _err(f"No movie in the catalog with id(s): {missing}.")
    return {"status": "ok", "movies": rows}


def _tool_add_to_watchlist(args, ctx):
    movie_id = _as_int(args.get("movie_id"))
    if movie_id is None:
        return _err("add_to_watchlist needs an integer 'movie_id'.")
    movie = _movie_exists(ctx, movie_id)
    if not movie:
        return _err(f"No movie in the catalog with id {movie_id}.")

    ctx.conn.execute(
        """
        INSERT INTO watchlist_items (group_id, movie_id, reason)
        VALUES (%(group_id)s, %(movie_id)s, %(reason)s)
        ON CONFLICT (group_id, movie_id)
        DO UPDATE SET reason = COALESCE(EXCLUDED.reason,
                                        watchlist_items.reason)
        """,
        {"group_id": ctx.group_id, "movie_id": movie_id,
         "reason": args.get("reason")},
    )
    ctx.conn.commit()
    return {"status": "ok", "action": "added_to_watchlist",
            "movie_id": movie_id, "title": movie.get("title")}


def _tool_record_rating(args, ctx):
    movie_id = _as_int(args.get("movie_id"))
    score = _as_int(args.get("score"))
    if movie_id is None:
        return _err("record_rating needs an integer 'movie_id'.")
    if score is None:
        return _err("record_rating needs an integer 'score' between 1 and 10.")
    if not _movie_exists(ctx, movie_id):
        return _err(f"No movie in the catalog with id {movie_id}.")

    score = max(1, min(score, 10))
    ctx.conn.execute(
        """
        INSERT INTO ratings (user_id, movie_id, score)
        VALUES (%(user_id)s, %(movie_id)s, %(score)s)
        ON CONFLICT (user_id, movie_id)
        DO UPDATE SET score = EXCLUDED.score, rated_at = now()
        """,
        {"user_id": ctx.user_id, "movie_id": movie_id, "score": score},
    )
    ctx.conn.commit()
    return {"status": "ok", "action": "recorded_rating",
            "movie_id": movie_id, "score": score}


def _tool_save_recommendation(args, ctx):
    chosen = _as_int(args.get("chosen_movie_id"))
    if chosen is None:
        return _err("save_recommendation needs an integer 'chosen_movie_id'.")
    if not _movie_exists(ctx, chosen):
        return _err(f"No movie in the catalog with id {chosen}.")

    candidates = [i for i in (_as_int(x) for x in
                              args.get("candidate_ids") or []) if i]
    ctx.conn.execute(
        """
        INSERT INTO recommendations
            (group_id, user_query, candidate_ids, chosen_movie_id, rationale)
        VALUES (%(g)s, %(q)s, %(c)s, %(chosen)s, %(r)s)
        """,
        {"g": ctx.group_id, "q": args.get("user_query") or "",
         "c": candidates, "chosen": chosen,
         "r": args.get("rationale") or ""},
    )
    ctx.conn.commit()
    return {"status": "ok", "action": "saved_recommendation",
            "chosen_movie_id": chosen}


_HANDLERS = {
    "search_movies": _tool_search_movies,
    "get_group_context": _tool_get_group_context,
    "compare_movies": _tool_compare_movies,
    "add_to_watchlist": _tool_add_to_watchlist,
    "record_rating": _tool_record_rating,
    "save_recommendation": _tool_save_recommendation,
}


def dispatch(name, arguments, ctx):
    handler = _HANDLERS.get(name)
    if handler is None:
        return _err(f"Unknown tool {name!r}.")
    try:
        return handler(arguments or {}, ctx)
    except Exception as exc:                      # never surface a traceback
        return _err(f"{name} failed: {type(exc).__name__}: {exc}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_tools.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add movienight/tools.py tests/test_tools.py; git commit -m "feat: agent tools with validation, idempotent writes, context-anchored ids"
```

---

## Task 7: Agent loop

**Files:**
- Create: `movienight/agent.py`
- Create: `tests/test_agent.py`

**Interfaces:**
- Consumes: `movienight.tools.TOOL_SCHEMAS`, `movienight.tools.dispatch`, `ToolContext`.
- Produces:
  - `SYSTEM_PROMPT: str`
  - `AgentStep` dataclass: `kind: str` (`"tool_call"` or `"answer"`), `name: str | None`, `arguments: dict | None`, `result: dict | None`, `content: str | None`
  - `run_agent(llm, ctx, user_message, history=None, max_iterations=6) -> list[AgentStep]`
  - `llm` is any callable `(messages, tools) -> dict` returning an OpenAI-shaped assistant message. This is what makes the loop testable offline.
  - `chat_llm_from_workspace(workspace_client, endpoint="databricks-llama-4-maverick") -> callable`

- [ ] **Step 1: Write the failing test**

Create `tests/test_agent.py`:

```python
from movienight.agent import SYSTEM_PROMPT, run_agent
from movienight.tools import ToolContext


class ScriptedLLM:
    """Replays canned assistant messages so the loop runs with no network."""

    def __init__(self, script):
        self.script = list(script)
        self.seen = []

    def __call__(self, messages, tools):
        self.seen.append(list(messages))
        return self.script.pop(0)


def tool_call(name, args, call_id="c1"):
    import json
    return {"role": "assistant", "tool_calls": [
        {"id": call_id, "type": "function",
         "function": {"name": name, "arguments": json.dumps(args)}}]}


def answer(text):
    return {"role": "assistant", "content": text}


class StubCtx(ToolContext):
    pass


def make_ctx(results):
    class Conn:
        def execute(self, *a, **k):
            class C:
                def fetchall(inner): return []
                def fetchone(inner): return None
            return C()
        def commit(self): pass
    ctx = ToolContext(conn=Conn(), embedder=None, group_id=1, user_id=1)
    ctx._results = results
    return ctx


def test_answer_without_tool_calls_returns_single_step(monkeypatch):
    steps = run_agent(ScriptedLLM([answer("Watch Toy Story.")]),
                      make_ctx({}), "what should we watch?")
    assert len(steps) == 1
    assert steps[0].kind == "answer"
    assert "Toy Story" in steps[0].content


def test_tool_call_is_executed_then_answer_returned(monkeypatch):
    import movienight.agent as agent_mod
    monkeypatch.setattr(agent_mod, "dispatch",
                        lambda n, a, c: {"status": "ok", "count": 1})
    llm = ScriptedLLM([tool_call("search_movies", {"query": "funny"}),
                       answer("Here you go.")])
    steps = run_agent(llm, make_ctx({}), "something funny")
    assert [s.kind for s in steps] == ["tool_call", "answer"]
    assert steps[0].name == "search_movies"
    assert steps[0].arguments == {"query": "funny"}
    assert steps[0].result["status"] == "ok"


def test_tool_results_are_fed_back_to_the_model(monkeypatch):
    import movienight.agent as agent_mod
    monkeypatch.setattr(agent_mod, "dispatch",
                        lambda n, a, c: {"status": "ok", "marker": "XYZZY"})
    llm = ScriptedLLM([tool_call("get_group_context", {}), answer("done")])
    run_agent(llm, make_ctx({}), "hi")
    second_call_messages = llm.seen[1]
    assert any(m.get("role") == "tool" and "XYZZY" in m.get("content", "")
               for m in second_call_messages)


def test_loop_stops_at_max_iterations(monkeypatch):
    import movienight.agent as agent_mod
    monkeypatch.setattr(agent_mod, "dispatch", lambda n, a, c: {"status": "ok"})
    llm = ScriptedLLM([tool_call("search_movies", {"query": "x"})] * 10)
    steps = run_agent(llm, make_ctx({}), "loop forever", max_iterations=3)
    assert sum(1 for s in steps if s.kind == "tool_call") == 3
    assert steps[-1].kind == "answer"
    assert "could not" in steps[-1].content.lower()


def test_malformed_tool_arguments_do_not_crash(monkeypatch):
    import movienight.agent as agent_mod
    monkeypatch.setattr(agent_mod, "dispatch", lambda n, a, c: {"status": "ok"})
    bad = {"role": "assistant", "tool_calls": [
        {"id": "c1", "type": "function",
         "function": {"name": "search_movies", "arguments": "{not json"}}]}
    steps = run_agent(ScriptedLLM([bad, answer("recovered")]),
                      make_ctx({}), "hi")
    assert steps[0].kind == "tool_call"
    assert steps[0].result["status"] == "error"
    assert steps[-1].content == "recovered"


def test_system_prompt_forbids_answering_without_tools():
    lowered = SYSTEM_PROMPT.lower()
    assert "tool" in lowered
    assert "never" in lowered or "do not" in lowered
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_agent.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'movienight.agent'`

- [ ] **Step 3: Write the implementation**

Create `movienight/agent.py`:

```python
"""Tool-calling loop.

Deliberately knows nothing about movies - only about the protocol. The llm
argument is any callable (messages, tools) -> assistant message, which is what
lets the whole loop be tested offline against a scripted model.

No Agent Bricks and no AI Gateway: registering an MCP service there requires a
per-user OAuth grant that a service call cannot supply. Calling the serving
endpoint directly has no such dependency.
"""

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from .tools import TOOL_SCHEMAS, dispatch

DEFAULT_ENDPOINT = "databricks-llama-4-maverick"
MAX_TOOL_RESULT_CHARS = 6000

SYSTEM_PROMPT = """\
You help a group of friends decide what to watch tonight.

Answer only from tool results. Never state a plot, runtime, rating, or
release year that did not come from a tool call, and never invent a movie id.

Order of work:
1. Call get_group_context first, so you know who is in the group, what they
   have rated, and what they have already seen.
2. Call search_movies with the mood or theme in `query`, and put hard
   constraints in the structured arguments - runtime limits in
   max_runtime_minutes, "nothing too violent" in exclude_violent, decades in
   min_year. Do not put those constraints only in the query text.
3. Recommend from what search_movies returned. Say why it fits the group,
   citing ratings or constraints - not vibes.

When the user agrees on a film, call add_to_watchlist, then
save_recommendation with the candidates you considered and your reasoning.
Tell the user plainly whenever you have written something.

If a tool returns status "error", relay the message in plain language and ask
the user to clarify. Do not retry with a made-up id and do not fall back on
general knowledge.

Keep answers to a few sentences unless asked for detail.
"""


@dataclass
class AgentStep:
    kind: str                  # "tool_call" | "answer"
    name: str | None = None
    arguments: dict | None = None
    result: dict | None = None
    content: str | None = None


def _truncate(payload):
    text = json.dumps(payload, default=str)
    if len(text) > MAX_TOOL_RESULT_CHARS:
        return text[:MAX_TOOL_RESULT_CHARS] + ' ...(truncated)"}'
    return text


def run_agent(llm, ctx, user_message, history=None, max_iterations=6):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": user_message})

    steps = []
    for _ in range(max_iterations):
        assistant = llm(messages, TOOL_SCHEMAS)
        tool_calls = assistant.get("tool_calls") or []

        if not tool_calls:
            content = assistant.get("content") or ""
            steps.append(AgentStep(kind="answer", content=content))
            return steps

        messages.append(assistant)

        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            raw = fn.get("arguments") or "{}"
            try:
                arguments = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                arguments = None

            if arguments is None:
                result = {"status": "error",
                          "message": f"Arguments for {name} were not valid JSON."}
            else:
                result = dispatch(name, arguments, ctx)

            steps.append(AgentStep(kind="tool_call", name=name,
                                   arguments=arguments or {}, result=result))
            messages.append({"role": "tool", "tool_call_id": call.get("id"),
                             "name": name, "content": _truncate(result)})

    steps.append(AgentStep(
        kind="answer",
        content=("I could not settle on an answer - I kept needing more "
                 "lookups. Try narrowing the request.")))
    return steps


def chat_llm_from_workspace(workspace_client, endpoint=DEFAULT_ENDPOINT,
                            timeout=120):
    host = workspace_client.config.host.rstrip("/")
    headers = {**workspace_client.config.authenticate(),
               "Content-Type": "application/json"}
    url = f"{host}/serving-endpoints/{endpoint}/invocations"

    def call(messages, tools):
        body = {"messages": messages, "tools": tools,
                "tool_choice": "auto", "max_tokens": 1024}
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            return {"role": "assistant",
                    "content": f"The model endpoint returned "
                               f"HTTP {e.code}. Try again in a moment."}
        return payload["choices"][0]["message"]

    return call
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_agent.py -v`
Expected: 6 passed

- [ ] **Step 5: Run the whole offline suite**

Run: `python -m pytest tests -v`
Expected: all tests pass, no network required.

- [ ] **Step 6: Commit**

```bash
git add movienight/agent.py tests/test_agent.py; git commit -m "feat: tool-calling agent loop with offline-testable LLM seam"
```

---

## Task 8: Spark ingestion pipeline

**Files:**
- Create: `notebooks/ingest_movies.py` (Databricks notebook source format)

**Interfaces:**
- Consumes: `movienight.tmdb_client`, `movienight.documents`, `movienight.embeddings`, `movienight.db`.
- Produces: Delta tables `workspace.default.movies_bronze`, `workspace.default.movies_silver`; rows in Lakebase `movie_night.movies` and `movie_night.movie_embeddings`.

**Widgets:** `pages` (default `50`), `min_votes` (default `200`), `reembed_all` (default `false`).

- [ ] **Step 1: Write the notebook**

Create `notebooks/ingest_movies.py`:

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # Movie ingestion pipeline
# MAGIC TMDB -> bronze Delta -> silver Delta -> embeddings -> Lakebase.
# MAGIC
# MAGIC Resumable by design: 250 discover pages plus 15,000 enrichment calls plus
# MAGIC ~29 minutes of embedding will eventually be interrupted. Bronze is the
# MAGIC checkpoint; the embedding stage only processes movies whose document hash
# MAGIC differs from what is already stored.

# COMMAND ----------
%pip install psycopg[binary] psycopg_pool
dbutils.library.restartPython()

# COMMAND ----------
import sys
REPO = "/Workspace/Users/dlpate2525@gmail.com/movie-night"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

dbutils.widgets.text("pages", "50")
dbutils.widgets.text("min_votes", "200")
dbutils.widgets.dropdown("reembed_all", "false", ["true", "false"])

PAGES = int(dbutils.widgets.get("pages"))
MIN_VOTES = int(dbutils.widgets.get("min_votes"))
REEMBED_ALL = dbutils.widgets.get("reembed_all") == "true"

CATALOG, SCHEMA = "workspace", "default"
BRONZE = f"{CATALOG}.{SCHEMA}.movies_bronze"
SILVER = f"{CATALOG}.{SCHEMA}.movies_silver"

# COMMAND ----------
# MAGIC %md ## Stage 1 - discover, into bronze
import json

from databricks.sdk import WorkspaceClient
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType, LongType

from movienight.tmdb_client import TMDBClient, normalize_movie, read_token

w = WorkspaceClient()
tmdb = TMDBClient(read_token(w))

spark.sql(f"CREATE TABLE IF NOT EXISTS {BRONZE} "
          "(movie_id BIGINT, payload STRING, fetched_at TIMESTAMP) USING DELTA")

already = set()
if spark.catalog.tableExists(BRONZE):
    already = {r.movie_id for r in spark.table(BRONZE).select("movie_id").collect()}
print(f"bronze already holds {len(already)} movies")

discovered = []
for page in range(1, PAGES + 1):
    for row in tmdb.discover_page(page, MIN_VOTES):
        discovered.append(row["id"])
    if page % 25 == 0:
        print(f"  discovered through page {page}: {len(discovered)} ids")

todo = [mid for mid in dict.fromkeys(discovered) if mid not in already]
print(f"{len(todo)} movies need enrichment")

# COMMAND ----------
# MAGIC %md ## Stage 2 - enrich, appending to bronze in chunks so a failure keeps progress
CHUNK = 200
schema = StructType([
    StructField("movie_id", LongType()),
    StructField("payload", StringType()),
])

for start in range(0, len(todo), CHUNK):
    batch = todo[start:start + CHUNK]
    rows = []
    for movie_id in batch:
        try:
            rows.append((movie_id, json.dumps(tmdb.movie_bundle(movie_id))))
        except Exception as exc:
            print(f"  skipping {movie_id}: {exc}")
    if rows:
        (spark.createDataFrame(rows, schema)
             .withColumn("fetched_at", F.current_timestamp())
             .write.mode("append").saveAsTable(BRONZE))
    print(f"  bronze +{len(rows)} ({start + len(batch)}/{len(todo)})")

# COMMAND ----------
# MAGIC %md ## Stage 3 - silver: typed columns and the composed document
from movienight.documents import compose_document, doc_hash

bronze = spark.table(BRONZE).dropDuplicates(["movie_id"])

def to_silver(iterator):
    import pandas as pd
    for pdf in iterator:
        records = []
        for _, row in pdf.iterrows():
            rec = normalize_movie(json.loads(row["payload"]))
            if not rec["movie_id"] or not rec["title"]:
                continue
            document = compose_document(rec)
            rec["document"] = document
            rec["doc_sha256"] = doc_hash(document)
            records.append(rec)
        yield pd.DataFrame(records)

silver_schema = ("movie_id long, title string, release_year int, tagline string, "
                 "overview string, runtime int, certification string, "
                 "genres array<string>, keywords array<string>, cast array<string>, "
                 "director string, poster_path string, vote_average double, "
                 "vote_count int, popularity double, document string, "
                 "doc_sha256 string")

silver = bronze.mapInPandas(to_silver, schema=silver_schema)
silver.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(SILVER)
print(f"silver rows: {spark.table(SILVER).count()}")
display(spark.table(SILVER).select("title", "runtime", "certification",
                                   "document").limit(5))

# COMMAND ----------
# MAGIC %md ## Stage 4 - upsert movies into Lakebase
from movienight.db import connection

rows = spark.table(SILVER).collect()
print(f"upserting {len(rows)} movies")

UPSERT = """
INSERT INTO movies (movie_id, title, release_year, tagline, overview, runtime,
                    certification, genres, keywords, cast_names, director,
                    poster_path, vote_average, vote_count, popularity, updated_at)
VALUES (%(movie_id)s, %(title)s, %(release_year)s, %(tagline)s, %(overview)s,
        %(runtime)s, %(certification)s, %(genres)s, %(keywords)s, %(cast)s,
        %(director)s, %(poster_path)s, %(vote_average)s, %(vote_count)s,
        %(popularity)s, now())
ON CONFLICT (movie_id) DO UPDATE SET
    title=EXCLUDED.title, release_year=EXCLUDED.release_year,
    tagline=EXCLUDED.tagline, overview=EXCLUDED.overview,
    runtime=EXCLUDED.runtime, certification=EXCLUDED.certification,
    genres=EXCLUDED.genres, keywords=EXCLUDED.keywords,
    cast_names=EXCLUDED.cast_names, director=EXCLUDED.director,
    poster_path=EXCLUDED.poster_path, vote_average=EXCLUDED.vote_average,
    vote_count=EXCLUDED.vote_count, popularity=EXCLUDED.popularity,
    updated_at=now()
"""

with connection() as conn:
    for i in range(0, len(rows), 500):
        conn.cursor().executemany(UPSERT, [r.asDict() for r in rows[i:i + 500]])
        conn.commit()
        print(f"  {min(i + 500, len(rows))}/{len(rows)}")

# COMMAND ----------
# MAGIC %md ## Stage 5 - embed only what changed
from movienight.embeddings import MAX_BATCH, client_from_workspace

embedder = client_from_workspace(w)

with connection() as conn:
    existing = {r["movie_id"]: r["doc_sha256"] for r in conn.execute(
        "SELECT movie_id, doc_sha256 FROM movie_embeddings").fetchall()}

pending = [r.asDict() for r in rows
           if REEMBED_ALL or existing.get(r["movie_id"]) != r["doc_sha256"]]
print(f"{len(pending)} of {len(rows)} need embedding "
      f"({len(rows) - len(pending)} unchanged)")

EMBED_UPSERT = """
INSERT INTO movie_embeddings (movie_id, document, doc_sha256, embedding, updated_at)
VALUES (%(movie_id)s, %(document)s, %(doc_sha256)s, %(embedding)s::vector, now())
ON CONFLICT (movie_id) DO UPDATE SET
    document=EXCLUDED.document, doc_sha256=EXCLUDED.doc_sha256,
    embedding=EXCLUDED.embedding, updated_at=now()
"""

with connection() as conn:
    for i in range(0, len(pending), MAX_BATCH):
        chunk = pending[i:i + MAX_BATCH]
        vectors = embedder.embed([c["document"] for c in chunk])
        conn.cursor().executemany(EMBED_UPSERT, [
            {"movie_id": c["movie_id"], "document": c["document"],
             "doc_sha256": c["doc_sha256"], "embedding": str(v)}
            for c, v in zip(chunk, vectors)])
        conn.commit()
        if i % (MAX_BATCH * 25) == 0:
            print(f"  embedded {min(i + MAX_BATCH, len(pending))}/{len(pending)}")

# COMMAND ----------
# MAGIC %md ## Stage 6 - verify
with connection() as conn:
    for label, sql in [
        ("movies", "SELECT count(*) AS n FROM movies"),
        ("embeddings", "SELECT count(*) AS n FROM movie_embeddings"),
        ("missing vectors",
         "SELECT count(*) AS n FROM movies m LEFT JOIN movie_embeddings e "
         "USING (movie_id) WHERE e.movie_id IS NULL"),
        ("with runtime", "SELECT count(*) AS n FROM movies WHERE runtime IS NOT NULL"),
        ("with certification",
         "SELECT count(*) AS n FROM movies WHERE certification IS NOT NULL"),
    ]:
        print(f"{label:20s} {conn.execute(sql).fetchone()['n']}")
```

- [ ] **Step 2: Sync the repo to the workspace**

Run from a neutral directory:
```bash
databricks sync "C:/Users/dlpat/OneDrive/Desktop/Projects/databricks-ai-bootcamp-capstone" /Workspace/Users/dlpate2525@gmail.com/movie-night --full -p dbc-ff09ef2e-7294
```
Expected: `Initial Sync Complete`. If it uploads thousands of files, `.gitignore` is not excluding `.venv` — stop and fix.

- [ ] **Step 3: Smoke-run with a tiny page count**

In the notebook UI set `pages = 2` and Run All.
Expected: ~40 movies through every stage; Stage 6 reports `movies` ≈ 40, `missing vectors` = 0.

- [ ] **Step 4: Confirm resumability**

Re-run with `pages = 2` unchanged.
Expected: Stage 2 enriches 0 movies ("bronze already holds..."), Stage 5 reports `0 of ~40 need embedding`. If it re-embeds everything, the hash gate is broken — fix before the full run.

- [ ] **Step 5: Full run**

Set `pages = 50` and Run All. Expect roughly 10–15 minutes, mostly Stage 2 (~3,000
TMDB calls) and Stage 5 (~6 minutes of embedding).
Expected: Stage 6 reports ~1,000 movies, ~1,000 embeddings, 0 missing vectors.

- [ ] **Step 6: Commit**

```bash
git add notebooks/ingest_movies.py; git commit -m "feat: resumable Spark ingestion pipeline with hash-gated embedding"
```

---

## Task 9: Streamlit app

**Files:**
- Create: `app.py`, `app.yaml`, `requirements.txt`

**Interfaces:**
- Consumes: everything in `movienight/`.
- Produces: the deployed app.

`app.py` contains **no SQL and no HTTP** — it calls `movienight` modules only.

- [ ] **Step 1: Write the app manifest and dependencies**

Create `requirements.txt`:

```
streamlit>=1.40
databricks-sdk>=0.57
psycopg[binary]>=3.2
psycopg_pool>=3.2
```

Create `app.yaml`:

```yaml
command: ["streamlit", "run", "app.py", "--server.port", "8000",
          "--server.address", "0.0.0.0"]
env:
  - name: LAKEBASE_INSTANCE
    value: bootcamp-support-db
  - name: LAKEBASE_DATABASE
    value: databricks_postgres
```

No TMDB secret is needed at runtime: the app never calls TMDB. Posters are loaded
from `image.tmdb.org` using the `poster_path` already stored by the pipeline.

- [ ] **Step 2: Write the app**

Create `app.py`:

```python
"""Movie Night Planner - Streamlit UI.

Contains no SQL and no HTTP. Everything goes through movienight/*, so the
logic stays testable outside Streamlit.

Tool calls are rendered inline as the agent makes them. That is deliberate:
the capstone asks to demonstrate an agent taking actions, and a visible
add_to_watchlist(...) -> ok is the evidence.
"""

import streamlit as st
from databricks.sdk import WorkspaceClient

from movienight.agent import chat_llm_from_workspace, run_agent
from movienight.db import connection
from movienight.embeddings import client_from_workspace
from movienight.tools import ToolContext

st.set_page_config(page_title="Movie Night Planner", page_icon="🎬",
                   layout="wide")
POSTER = "https://image.tmdb.org/t/p/w185"


@st.cache_resource
def workspace():
    return WorkspaceClient()


@st.cache_resource
def embedder():
    return client_from_workspace(workspace())


@st.cache_resource
def llm():
    return chat_llm_from_workspace(workspace())


def current_user(conn):
    """Identity comes from the Apps proxy header, never from user input."""
    email = (st.context.headers.get("X-Forwarded-Email")
             if hasattr(st, "context") else None) or "you@example.com"
    row = conn.execute(
        """
        INSERT INTO users (email, display_name, is_demo)
        VALUES (%(e)s, %(n)s, false)
        ON CONFLICT (email) DO UPDATE SET display_name = EXCLUDED.display_name
        RETURNING user_id, display_name
        """,
        {"e": email, "n": email.split("@")[0]},
    ).fetchone()
    conn.commit()
    return row


def ensure_membership(conn, group_id, user_id):
    conn.execute(
        "INSERT INTO group_members (group_id, user_id) VALUES (%(g)s, %(u)s) "
        "ON CONFLICT DO NOTHING", {"g": group_id, "u": user_id})
    conn.commit()


with connection() as conn:
    me = current_user(conn)
    groups = conn.execute(
        "SELECT group_id, name FROM groups ORDER BY group_id").fetchall()

    st.title("🎬 Movie Night Planner")

    with st.sidebar:
        st.caption(f"Signed in as **{me['display_name']}**")
        if not groups:
            st.warning("No groups yet. Run scripts/bootstrap_db.py.")
            st.stop()
        names = {g["name"]: g["group_id"] for g in groups}
        chosen = st.selectbox("Group", list(names))
        group_id = names[chosen]
        ensure_membership(conn, group_id, me["user_id"])

        members = conn.execute(
            "SELECT u.display_name, u.is_demo FROM group_members gm "
            "JOIN users u USING (user_id) WHERE gm.group_id = %(g)s "
            "ORDER BY u.display_name", {"g": group_id}).fetchall()
        st.write("**Members**")
        for m in members:
            st.write(("· " + m["display_name"]) +
                     (" _(demo)_" if m["is_demo"] else ""))

    ctx = ToolContext(conn=conn, embedder=embedder(),
                      group_id=group_id, user_id=me["user_id"])

    chat_tab, rate_tab, list_tab = st.tabs(
        ["Ask the agent", "Browse & rate", "Watchlist"])

    with chat_tab:
        if "history" not in st.session_state:
            st.session_state.history = []

        for entry in st.session_state.history:
            with st.chat_message(entry["role"]):
                st.markdown(entry["content"])

        prompt = st.chat_input(
            "e.g. a funny sci-fi movie that isn't too violent, under two hours")
        if prompt:
            st.session_state.history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    steps = run_agent(llm(), ctx, prompt)
                for step in steps:
                    if step.kind == "tool_call":
                        status = (step.result or {}).get("status", "?")
                        icon = "✅" if status == "ok" else "⚠️"
                        with st.expander(
                                f"{icon} `{step.name}` — {status}", expanded=False):
                            st.write("**Arguments**")
                            st.json(step.arguments)
                            st.write("**Result**")
                            st.json(step.result)
                    else:
                        st.markdown(step.content)
                        st.session_state.history.append(
                            {"role": "assistant", "content": step.content})

    with rate_tab:
        who = st.selectbox(
            "Rating as", members,
            format_func=lambda m: m["display_name"], key="rating_as")
        search = st.text_input("Find a movie by title")
        if search:
            found = conn.execute(
                "SELECT movie_id, title, release_year, poster_path, runtime, "
                "certification FROM movies WHERE title ILIKE %(q)s "
                "ORDER BY popularity DESC LIMIT 12",
                {"q": f"%{search}%"}).fetchall()
            for chunk_start in range(0, len(found), 4):
                for col, movie in zip(st.columns(4),
                                      found[chunk_start:chunk_start + 4]):
                    with col:
                        if movie["poster_path"]:
                            st.image(POSTER + movie["poster_path"])
                        st.caption(f"**{movie['title']}** ({movie['release_year']})")
                        score = st.slider("Score", 1, 10, 7,
                                          key=f"s{movie['movie_id']}")
                        if st.button("Save", key=f"b{movie['movie_id']}"):
                            uid = conn.execute(
                                "SELECT user_id FROM users WHERE display_name=%(n)s",
                                {"n": who["display_name"]}).fetchone()["user_id"]
                            conn.execute(
                                "INSERT INTO ratings (user_id, movie_id, score) "
                                "VALUES (%(u)s, %(m)s, %(s)s) "
                                "ON CONFLICT (user_id, movie_id) DO UPDATE SET "
                                "score = EXCLUDED.score, rated_at = now()",
                                {"u": uid, "m": movie["movie_id"], "s": score})
                            conn.commit()
                            st.success(f"Saved {score}/10")

    with list_tab:
        items = conn.execute(
            "SELECT w.movie_id, m.title, m.release_year, m.poster_path, "
            "w.status, w.reason FROM watchlist_items w "
            "JOIN movies m USING (movie_id) WHERE w.group_id = %(g)s "
            "ORDER BY w.added_at DESC", {"g": group_id}).fetchall()
        if not items:
            st.info("Nothing queued yet — ask the agent for a recommendation.")
        for item in items:
            cols = st.columns([1, 5])
            with cols[0]:
                if item["poster_path"]:
                    st.image(POSTER + item["poster_path"])
            with cols[1]:
                st.subheader(f"{item['title']} ({item['release_year']})")
                st.caption(f"Status: {item['status']}")
                if item["reason"]:
                    st.write(item["reason"])
```

- [ ] **Step 3: Run locally against Lakebase**

Run: `streamlit run app.py`
Expected: the app loads, the sidebar lists the demo group and members, and the
Browse tab finds movies by title. The agent tab will work only if the pipeline
has run.

- [ ] **Step 4: Deploy**

Run from a neutral directory:
```bash
databricks sync "C:/Users/dlpat/OneDrive/Desktop/Projects/databricks-ai-bootcamp-capstone" /Workspace/Users/dlpate2525@gmail.com/movie-night --full -p dbc-ff09ef2e-7294
```
Then:
```bash
databricks apps create movie-night -p dbc-ff09ef2e-7294
```
Then:
```bash
databricks apps deploy movie-night --source-code-path /Workspace/Users/dlpate2525@gmail.com/movie-night -p dbc-ff09ef2e-7294
```
Expected: `SUCCEEDED App started successfully`.

- [ ] **Step 5: Grant the app's service principal access to Lakebase**

The app runs as its own service principal, which needs Postgres privileges on
`movie_night` and the ability to mint database credentials. Run:
```bash
databricks apps get movie-night -p dbc-ff09ef2e-7294 -o json
```
Note the `service_principal_client_id`, then grant it in the Lakebase instance:
```sql
GRANT USAGE ON SCHEMA movie_night TO "<service-principal-id>";
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA movie_night TO "<service-principal-id>";
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA movie_night TO "<service-principal-id>";
```
Expected: opening the app URL shows data rather than a permissions error.

- [ ] **Step 6: Commit**

```bash
git add app.py app.yaml requirements.txt; git commit -m "feat: Streamlit app with inline agent tool-call display"
```

---

## Task 10: Retrieval quality proof, docs, and submission

**Files:**
- Create: `tests/test_retrieval_live.py`, `scripts/e2e_test.py`, `README.md`

**Interfaces:** none produced; this task proves and documents.

The discrimination test is the most important artifact here. It is the only thing
that proves the embeddings carry signal rather than merely existing.

- [ ] **Step 1: Write the discrimination test**

Create `tests/test_retrieval_live.py`:

```python
"""Retrieval quality against the populated database. Requires Lakebase.

Skipped automatically when the catalog is empty, so the offline suite stays green.
"""

import pytest
from databricks.sdk import WorkspaceClient

from movienight.db import connection
from movienight.embeddings import client_from_workspace
from movienight.retrieval import SearchFilters, search


@pytest.fixture(scope="module")
def db():
    with connection() as conn:
        n = conn.execute("SELECT count(*) AS n FROM movie_embeddings").fetchone()["n"]
        if n < 100:
            pytest.skip(f"only {n} embeddings; run the pipeline first")
        yield conn


@pytest.fixture(scope="module")
def embed():
    return client_from_workspace(WorkspaceClient())


def titles(rows):
    return [r["title"] for r in rows]


def test_family_query_outranks_dark_thriller(db, embed):
    rows = search(db, embed.embed_one(
        "a heartwarming animated movie about friendship for kids"),
        SearchFilters(), k=20)
    found = titles(rows)
    assert any("Toy Story" in t or "Finding" in t or "Up" in t for t in found), found
    assert "Fight Club" not in found


def test_runtime_filter_is_actually_enforced(db, embed):
    rows = search(db, embed.embed_one("epic adventure"),
                  SearchFilters(max_runtime_minutes=95), k=10)
    assert rows, "filter returned nothing - check the exact-scan fallback"
    assert all(r["runtime"] <= 95 for r in rows), [r["runtime"] for r in rows]


def test_exclude_violent_drops_r_rated_and_unrated(db, embed):
    rows = search(db, embed.embed_one("intense crime story"),
                  SearchFilters(exclude_violent=True), k=10)
    assert all(r["certification"] not in (None, "R", "NC-17") for r in rows)


def test_exclusions_are_honoured(db, embed):
    baseline = search(db, embed.embed_one("space adventure"),
                      SearchFilters(), k=5)
    drop = baseline[0]["movie_id"]
    rows = search(db, embed.embed_one("space adventure"),
                  SearchFilters(exclude_movie_ids=[drop]), k=5)
    assert drop not in [r["movie_id"] for r in rows]


def test_similarity_is_ordered_descending(db, embed):
    rows = search(db, embed.embed_one("romantic comedy in Paris"),
                  SearchFilters(), k=10)
    sims = [r["similarity"] for r in rows]
    assert sims == sorted(sims, reverse=True)
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/test_retrieval_live.py -v`
Expected: 5 passed. **If the family-vs-thriller test fails, the composed document
in Task 2 is not carrying enough signal** — revisit it before writing the README.

- [ ] **Step 3: Write the end-to-end agent script**

Create `scripts/e2e_test.py`:

```python
"""Drive the agent through three natural-language questions and print the
tool calls it made. Produces the transcript the submission needs."""

from databricks.sdk import WorkspaceClient

from movienight.agent import chat_llm_from_workspace, run_agent
from movienight.db import connection
from movienight.embeddings import client_from_workspace
from movienight.tools import ToolContext

QUESTIONS = [
    "We want something funny and light tonight, nothing over two hours.",
    "Compare your top two picks and tell us which suits the group better.",
    "Add that one to our watchlist and save why you chose it.",
]


def main():
    w = WorkspaceClient()
    llm = chat_llm_from_workspace(w)
    with connection() as conn:
        group = conn.execute("SELECT group_id FROM groups ORDER BY group_id "
                             "LIMIT 1").fetchone()
        user = conn.execute("SELECT user_id FROM users ORDER BY user_id "
                            "LIMIT 1").fetchone()
        ctx = ToolContext(conn=conn, embedder=client_from_workspace(w),
                          group_id=group["group_id"], user_id=user["user_id"])

        history = []
        for question in QUESTIONS:
            print("=" * 70)
            print(f"Q: {question}")
            print("=" * 70)
            for step in run_agent(llm, ctx, question, history=history):
                if step.kind == "tool_call":
                    status = (step.result or {}).get("status")
                    print(f"  -> {step.name}({step.arguments}) [{status}]")
                else:
                    print(f"\n  {step.content}\n")
                    history.append({"role": "user", "content": question})
                    history.append({"role": "assistant", "content": step.content})

        writes = conn.execute(
            "SELECT count(*) AS n FROM watchlist_items WHERE group_id = %(g)s",
            {"g": group["group_id"]}).fetchone()["n"]
        recs = conn.execute(
            "SELECT count(*) AS n FROM recommendations WHERE group_id = %(g)s",
            {"g": group["group_id"]}).fetchone()["n"]
        print(f"\nwatchlist rows: {writes}   recommendation rows: {recs}")
        print("PASS" if writes and recs else "FAIL - the agent never wrote anything")


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run it and capture the transcript**

Run: `python scripts/e2e_test.py > docs/agent-transcript.txt`
Expected: visible `search_movies`, `get_group_context`, `add_to_watchlist`, and
`save_recommendation` calls, and a final `PASS`. This transcript is submission
evidence for "an agent that does stuff."

- [ ] **Step 5: Write the README**

Create `README.md` covering: what the project is; an architecture diagram matching
the spec's §3; the TMDB endpoints used and why TMDB; the composed-document rationale
with the measured 295-char overview problem; the hybrid retrieval explanation with
the runtime clause as the motivating example; the six tools in a table marking reads
and writes; the write-safety rules; setup steps (`setup_secrets.py`, `bootstrap_db.py`,
sync, notebook, deploy); the app URL; the measured rate limits; and honest known
limitations. Map each of the five capstone requirements to where it is satisfied.

- [ ] **Step 6: Commit and push**

```bash
git add tests/test_retrieval_live.py scripts/e2e_test.py README.md docs/agent-transcript.txt; git commit -m "feat: retrieval discrimination tests, e2e agent transcript, README"; git push -u origin main
```

- [ ] **Step 7: Attempt the optional dashboard**

Run: `databricks apps create movie-night-dashboard -p dbc-ff09ef2e-7294`
If it fails with a quota error, the 3-app cap binds and the dashboard is out of
scope as agreed — record that in the README. If it succeeds, build it as a second
Streamlit app reading `recommendations` and `ratings`.

---

## Self-Review

**Spec coverage.** Every section of the design maps to a task: §1 requirements → Tasks 8/1/2/9/6-7; §2 constraints → Global Constraints; §3 architecture → file structure; §4 documents → Task 2; §5 hybrid retrieval → Task 5; §6 schema and identity → Tasks 4 and 9; §7 agent and write safety → Tasks 6 and 7; §8 frontend → Task 9; §9 scale and resumability → Task 8; §10 testing → Tasks 1-7 and 10; §11 risks → mitigations in Tasks 3, 5, and 8.

**Type consistency.** The normalized record from Task 1 is consumed unchanged by Tasks 2, 4, and 8. `cast` in the Python record maps to `cast_names` in SQL (`cast` is a reserved word) — Task 8's upsert handles this explicitly. `SearchFilters` field names in Task 5 match the argument names `tools.py` passes in Task 6. `AgentStep.kind` values (`"tool_call"`, `"answer"`) are the same strings `app.py` branches on in Task 9.

**Known gap.** Task 9 Step 5 grants Lakebase privileges to the app's service principal; the exact grantee syntax may need adjustment once the principal id is known, since Lakebase maps Databricks identities to Postgres roles. This is flagged rather than guessed.
