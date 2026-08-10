# AI Movie Night Planner — Databricks AI Capstone

A group of friends rate movies, describe what they're in the mood for, and an AI agent
recommends something everyone will enjoy — then writes its decision back to the
database.

Built on Databricks Free Edition: Lakebase (Postgres + pgvector), Foundation Model
APIs, and Databricks Apps.

## Capstone requirements

| Requirement | Where it lives |
| --- | --- |
| **Data pipeline in Spark** | `notebooks/ingest_movies.py` — TMDB → bronze Delta → silver Delta → embeddings → Lakebase. `scripts/load_movies.py` is a plain-Python loader used to populate the demo catalog quickly. |
| **Third-party API** | TMDB — `discover`, `movie/{id}`, `/keywords`, `/credits`, `/release_dates`. All HTTP is confined to `movienight/tmdb_client.py`. |
| **Unstructured data processing** | One composed text document per movie (`movienight/documents.py`), embedded to `vector(1024)` and searched semantically. |
| **Databricks App with a frontend** | `app.py` — Streamlit. Group picker, agent chat with visible tool calls, watchlist. |
| **An AI agent that does stuff** | `movienight/agent.py` + `movienight/tools.py` — 3 read tools and **3 write tools** against live Postgres. |

## Architecture

```
TMDB API
   |  (v4 bearer token, Databricks secret scope `movie_night`)
   v
Pipeline  --> movies + composed documents
   |
   +--> Foundation Model API (databricks-gte-large-en, 1024-dim)
   |
   v
Lakebase Postgres 16 + pgvector 0.8.0
   |   movies, movie_embeddings, users, groups, group_members,
   |   ratings, watchlist_items, recommendations
   v
Streamlit Databricks App
   +-- tool-calling loop --> databricks-llama-4-maverick
                                  |
                                  +--> 3 read tools, 3 write tools --> Lakebase
```

There is deliberately **no AI Gateway, no Unity Catalog MCP service, and no Agent
Bricks**. That path was tried on an earlier assignment and fails with
`Cannot update a connection from BEARER_TOKEN to OAUTH_DCR` — the platform wants a
per-user OAuth login grant that a service call cannot supply. Calling the serving
endpoint directly from the app has no such dependency.

## The interesting part: why retrieval is hybrid

The motivating query is *"a funny sci-fi movie that isn't too violent and is under two
hours."* It decomposes into clauses that need different mechanisms:

| Clause | Mechanism |
| --- | --- |
| "funny sci-fi" | semantic — genres and keywords in the embedded document |
| "isn't too violent" | semantic (tone keywords) **and** structured (`certification`) |
| "under two hours" | **purely structured** — `runtime <= 120` |

No embedding will ever enforce a runtime bound. Both halves run in one SQL statement:
pgvector `<=>` cosine ordering, with `WHERE` predicates on runtime, certification,
year and genre array overlap (`movienight/retrieval.py`).

Two consequences that are easy to get wrong and are handled explicitly:

- **HNSW is approximate and Postgres filters *after* the index returns candidates.**
  A narrow filter can return fewer than `k` rows even when matches exist. The search
  over-fetches a candidate pool and falls back to an exact scan when the filtered
  result is short.
- **`certification IS NULL` means unknown, not safe.** An unrated film must not leak
  into a "nothing too violent" result, so NULL is excluded rather than assumed benign.

### Why the embedded document is composed, not just the plot

Measured: TMDB overviews average **~295 characters**. Embedding an overview alone
retrieves badly — there is nothing for "not too violent" to match against. Each
movie's document concatenates title, tagline, genres, overview, keywords, top cast,
director, runtime and certification. Keywords carry most of the tone signal:
*Fight Club* → `nihilism`, `rage and hate`; *Toy Story* → `friendship`,
`anthropomorphism`.

## The agent's tools

| Tool | Kind | Notes |
| --- | --- | --- |
| `search_movies` | read | Hybrid search; group exclusions applied automatically |
| `get_group_context` | read | Members, ratings, already-watched, disliked |
| `compare_movies` | read | Side-by-side for 2–4 titles |
| `add_to_watchlist` | **write** | Idempotent on `(group_id, movie_id)` |
| `record_rating` | **write** | UPSERT; score clamped to 1–10 |
| `save_recommendation` | **write** | Audit row: query, candidates, pick, rationale |

### Write safety

Every argument reaching these tools was chosen by a language model, so the tool layer
is treated as a trust boundary:

- **Ids are validated against the catalog before any write** — including every id in
  `candidate_ids`, because `bigint[]` cannot be foreign-key constrained and a
  hallucinated id would otherwise persist silently into the audit trail.
- **`group_id` and `user_id` come from session state, never from the model.** If the
  model supplies a `group_id`, it is ignored.
- **All SQL is parameterised.** No value is ever interpolated into SQL text.
- **Writes are idempotent**, so a retry after a timeout cannot double-write.
- **Ranges are clamped, not trusted** (`score` → 1–10).
- **Tools return dicts and never raise**, so the agent gets a sentence it can relay.

## Setup

```bash
pip install -e .
pip install "psycopg[binary]" psycopg_pool databricks-sdk pytest
python setup_secrets.py                    # stores the TMDB token, getpass, never echoed
python scripts/bootstrap_db.py             # creates the schema, idempotent
python scripts/load_movies.py --pages 15   # populates the catalog
```

Then deploy:

```bash
databricks sync . /Workspace/Users/<you>/movie-night --full -p <profile>
databricks apps deploy movie-night --source-code-path /Workspace/Users/<you>/movie-night -p <profile>
```

## Tests

```bash
python -m pytest tests -q
```

52 tests, all offline — no database and no network. The LLM is injected as a callable,
so the agent loop is tested against a scripted model.

The tests worth knowing about, because they encode bugs that were actually caught:

- **Embedding order preservation.** Vectors must return in input order across batch
  boundaries; if they don't, every movie silently gets another movie's numbers. The
  test was verified by deleting the sort line and confirming it failed.
- **Model-supplied values never reach SQL text**, including a SQL-injection attempt
  that must survive as an inert parameter.
- **Rejected writes must not commit** — asserted via both the commit flag and the
  absence of the INSERT, because returning an error *and* writing anyway is the real
  failure mode.

## Measured constraints (Free Edition)

| Fact | Value |
| --- | --- |
| Embedding dimension | 1024 (`databricks-gte-large-en`) |
| Embedding batch limit | **8** — batches of 16 fail with `429` at any spacing |
| Sustained embedding rate | ~2.9 docs/sec at batch 8 with a 2s gap |
| Tool calling | Works on llama-4-maverick, llama-3.3-70b, gpt-oss-120b, qwen3 |
| App quota | 3 apps per workspace |
| TMDB corpus | 14,752 movies at `vote_count >= 200` |

## Known limitations

Stated honestly rather than omitted:

- **The agent can write and then under-report it.** If the tool-calling loop hits its
  6-iteration cap on a turn that already executed a write, the write is committed but
  the fallback message says it could not settle on an answer. Low probability, real.
- **Large tool results are truncated by slicing serialised JSON**, which can produce a
  structurally invalid fragment. Tool content is a plain string so nothing breaks at
  the protocol level, but a model could be tempted to complete the missing data.
- **Network errors other than HTTP status codes are unhandled** in the model client —
  `URLError` and timeouts propagate rather than degrading gracefully.
- **Parallel tool calls are implemented but untested.** The loop iterates all calls
  correctly by inspection; no test emits more than one call per turn.
- **The catalog is a popularity-sorted slice**, so it skews recent and mainstream. A
  request for an obscure or older title can legitimately find nothing, and the agent
  is instructed to say so rather than substitute a loosely similar film.
- **Keyword lists include generic TMDB tags** (`based on novel or book`,
  `duringcreditsstinger`) that dilute the distinctive ones. A stoplist would sharpen
  retrieval; not implemented.
- **The optional dashboard app was not built** — the workspace is at its 3-app cap.

## Repository layout

```
app.py                     Streamlit UI (no SQL, no HTTP)
movienight/
  tmdb_client.py           All TMDB HTTP + normalisation
  documents.py             Pure: compose_document, doc_hash
  embeddings.py            Foundation Model embedding client, adaptive backoff
  db.py                    Lakebase pool; fresh OAuth token per connection
  retrieval.py             Hybrid pgvector + SQL search
  tools.py                 The 6 tools — the trust boundary
  agent.py                 Tool-calling loop; knows protocol, not movies
sql/                       Schema, indexes, demo seed
scripts/                   bootstrap_db.py, load_movies.py, capture_fixtures.py
notebooks/ingest_movies.py Spark pipeline
tests/                     52 offline tests
docs/superpowers/          Design spec and implementation plan
```

## A note on Lakebase auth

Lakebase uses a short-lived OAuth token as the Postgres password, and it expires after
about an hour. A connection pool holding one password dies silently mid-session. The
pool therefore mints a **fresh token per physical connection** and sets `max_lifetime`
to 45 minutes so connections are recycled before their credential dies
(`movienight/db.py`).
