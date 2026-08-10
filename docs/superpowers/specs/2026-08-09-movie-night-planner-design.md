# AI Movie Night Planner — Design

**Date:** 2026-08-09
**Capstone option:** 1 — AI Movie Night Planner
**Workspace:** `dbc-ff09ef2e-7294` (Databricks Free Edition)

A group of people rate movies, describe what they're in the mood for, and an agent
recommends something everyone will enjoy — then writes the decision back to the
database.

## 1. Requirement mapping

The capstone has five hard requirements. Naming what satisfies each, so nothing is
assumed to be covered by something adjacent.

| Requirement | Satisfied by |
| --- | --- |
| A data pipeline in Spark | Notebook job: TMDB → bronze Delta → silver Delta → embeddings → Lakebase. Spark DataFrames and Delta tables in Unity Catalog, not a loop in pandas. |
| Third-party API | TMDB — `discover`, `movie/{id}`, `/keywords`, `/credits`, `/release_dates`, `/reviews`, `/watch/providers` |
| Unstructured data processing | A composed text document per movie (§4), embedded to `vector(1024)` and searched semantically |
| Databricks App with a frontend | One Streamlit app: group management, rating, agent chat, live watchlist |
| An agent that *does stuff* | Tool-calling loop with three read tools and three **write** tools (§7) |

## 2. Measured constraints

Everything below was measured against this workspace before the design was written,
not assumed. These numbers drive several decisions.

| Fact | Value | Consequence |
| --- | --- | --- |
| TMDB v4 read token | works | Use `Authorization: Bearer` header |
| TMDB v3 api key | **401** | Unused. Header auth also keeps the credential out of URLs |
| `discover` corpus at `vote_count>=200` | 14,752 movies / 738 pages | Far more than needed; page count is a parameter |
| Movie `overview` length | ~295 chars | **Too short to embed alone** — drives §4 |
| `/reviews` | 11 for a popular title, first 188 chars | Sparse. Included when present, never depended on |
| `/keywords` | 14–26 per movie, e.g. `nihilism`, `rage and hate`, `friendship` | Highest-signal field for tone queries |
| Embedding dim (`databricks-gte-large-en`) | **1024** | `vector(1024)` in DDL |
| Embedding batch ≤ 8, ~2s gap | 2.9 docs/sec sustained | 500 movies ≈ 3 min |
| Embedding batch ≥ 16 | `429 REQUEST_LIMIT_EXCEEDED` at any spacing | Hard cap: batch size 8 |
| Embedding batch 256 | `400 BAD_REQUEST` | True size ceiling is 128–256; irrelevant given the 429 |
| Tool calling on 4 chat endpoints | all return `tool_calls` in ~1s | Agent architecture is viable |

The rate limit is a Free Edition budget and its exact window was not pinned down —
batch 8 at a 2s gap sustained 6/6 twice, batch 16 failed 5/5. The pipeline therefore
uses **adaptive backoff rather than a fixed sleep**, so it degrades gracefully if the
budget behaves differently on a different day.

## 3. Architecture

```
TMDB API
   |  (Bearer, secret scope movie_night)
   v
Spark notebook -- bronze Delta (raw JSON) --> silver Delta (typed + composed doc)
   |                                              |
   |                                              v
   |                                    FM API databricks-gte-large-en
   |                                       batches of 8, backoff
   v                                              |
Lakebase Postgres 16 (pgvector) <-----------------+
   ^  movies, movie_embeddings, users, groups, group_members,
   |  ratings, watchlist_items, recommendations
   |
Streamlit Databricks App
   |-- UI: groups, rating, watchlist
   +-- agent loop --> FM API databricks-llama-4-maverick (tool calling)
                          |
                          +-- 3 read tools, 3 write tools --> Lakebase
```

Deliberately **no** AI Gateway, no Unity Catalog MCP service, and no Agent Bricks. That
path failed on Assignment 3 with `Cannot update a connection from BEARER_TOKEN to
OAUTH_DCR` — the platform wanted a per-user OAuth login grant that a service call can't
supply. Calling the serving endpoint directly from the app has no such dependency.

## 4. The composed document — context engineering

The naive version of this project embeds `overview` and calls it retrieval. Measured at
~295 characters, that fails the spec's own example query. One document per movie is
composed from:

```
{title} ({year}). {tagline}
Genres: {genre names}
{overview}
Themes: {keywords, comma-joined}
Starring {top 6 cast}. Directed by {director}.
Runtime {runtime} minutes. Rated {US certification}.
{up to 3 review excerpts, when present}
```

Keywords carry the most weight for tone. "Not too violent" is not present in any
overview, but `nihilism` / `rage and hate` / `fight` versus `friendship` /
`anthropomorphism` separates *Fight Club* from *Toy Story* cleanly.

Documents are ~400–600 characters, comfortably inside the embedding model's window, so
**no chunking is needed** — one vector per movie, unlike the sliding-window approach
that the weather assignment required.

## 5. Hybrid retrieval

The spec's example — *"a funny sci-fi movie that isn't too violent and is under two
hours"* — decomposes into parts that need different mechanisms:

| Clause | Mechanism |
| --- | --- |
| "funny sci-fi" | semantic — genre + keywords in the embedded document |
| "isn't too violent" | semantic (keywords) **and** structured (`certification NOT IN ('R','NC-17')`) |
| "under two hours" | **purely structured** — `runtime < 120` |

No embedding will ever enforce a runtime bound. Both halves run in one SQL statement:

```sql
SELECT m.*, 1 - (e.embedding <=> %s::vector) AS similarity
FROM movie_embeddings e JOIN movies m USING (movie_id)
WHERE (%(max_runtime)s IS NULL OR m.runtime <= %(max_runtime)s)
  AND (%(exclude_violent)s IS FALSE OR m.certification NOT IN ('R','NC-17'))
  AND (%(min_year)s IS NULL OR m.release_year >= %(min_year)s)
  AND m.movie_id <> ALL (%(exclude_ids)s)
ORDER BY e.embedding <=> %s::vector
LIMIT %(k)s;
```

`exclude_ids` is how "avoid movies already watched or disliked by group members" is
enforced — as a filter, not as a hope that the model remembers.

An HNSW index with `vector_cosine_ops` backs the ordering. At ~500 rows a sequential
scan would be fine; the index is there because the pipeline is designed to scale to the
full 14,752 without a schema change.

## 6. Lakebase schema

New schema `movie_night` on the **existing** `bootcamp-support-db` instance. A second
instance would be a second meter for no benefit.

| Table | Purpose | Notes |
| --- | --- | --- |
| `users` | person | see identity note below |
| `groups` | a movie-night group | |
| `group_members` | membership | PK `(group_id, user_id)` |
| `movies` | one row per TMDB title | `runtime`, `certification`, `release_year`, `genres text[]`, `keywords text[]`, `poster_path` |
| `movie_embeddings` | `vector(1024)` | HNSW `vector_cosine_ops`; separate table so re-embedding never rewrites `movies` |
| `ratings` | a user's score for a movie | PK `(user_id, movie_id)` — makes re-rating an UPSERT, not a duplicate |
| `watchlist_items` | group's queue | unique `(group_id, movie_id)`; `status` in `queued/watched/skipped` |
| `recommendations` | what the agent picked and why | audit trail of agent writes |

`recommendations` exists specifically so the agent's actions are inspectable after the
fact — it stores the query, the movie ids returned, the chosen pick, and the rationale.

### Identity

A Databricks App receives the signed-in user in the `X-Forwarded-Email` header. That
email is upserted into `users` on each request and is the *only* identity the app
trusts for writes attributed to "me".

A movie night needs several people, and a Free Edition workspace has one real account,
so groups are additionally populated with **seeded demo members** (`ava@example.com`,
`ben@example.com`, …) carrying pre-loaded ratings. This is what makes "recommend
something everyone will enjoy" a real constraint-satisfaction problem instead of a
single-user query.

The distinction is explicit in the schema: `users.is_demo boolean`. Demo members can be
rated on behalf of through the UI; the real user cannot be impersonated, and the agent
may never pass a `user_id` for the signed-in person other than the one resolved from
the header.

## 7. The agent

A tool-calling loop against `databricks-llama-4-maverick` over
`/serving-endpoints/{name}/invocations`. The loop: send messages + tool schemas →
if the response has `tool_calls`, execute them and append results → repeat until the
model answers in prose. Capped at 6 iterations so a confused model can't spin.

**Read tools**

| Tool | Behaviour |
| --- | --- |
| `search_movies(query, max_runtime_minutes, exclude_violent, min_year, genres, k)` | Hybrid query from §5, group exclusions applied automatically |
| `get_group_context(group_id)` | Members, their ratings, already-watched, disliked |
| `compare_movies(movie_ids)` | Side-by-side facts for 2–4 titles |

**Write tools**

| Tool | Behaviour |
| --- | --- |
| `add_to_watchlist(group_id, movie_id, reason)` | Idempotent on `(group_id, movie_id)` |
| `record_rating(user_id, movie_id, score)` | UPSERT; score clamped to 1–10 |
| `save_recommendation(group_id, movie_ids, chosen_movie_id, rationale)` | Audit row |

### Write safety

An LLM choosing arguments is untrusted input to SQL. Every write tool:

1. **Validates ids against the catalog** before writing — a hallucinated `movie_id`
   returns a clean error the model can react to, rather than a foreign-key stack trace
   or a silent orphan row.
2. **Uses parameterised SQL only.** No string interpolation anywhere in the tool layer.
3. **Is idempotent.** Re-running `add_to_watchlist` for the same pair is a no-op;
   `record_rating` updates. A retry after a timeout cannot double-write.
4. **Clamps ranges** rather than trusting the model (`score` to 1–10).
5. **Is scoped to the group in session state**, never to a `group_id` the model invents.

### System prompt guardrails

- Never state a plot, runtime, or rating that did not come from a tool result.
- Always call `get_group_context` before recommending, so exclusions are real.
- Cite *why* a pick fits — the numbers and constraints, not vibes.
- On a tool error, relay it and ask for clarification; do not retry with invented ids.
- Confirm in the reply whenever a write happened, naming what changed.

## 8. Frontend

One Streamlit app, four areas:

1. **Group** — create or pick a group, add members.
2. **Browse & rate** — poster grid from TMDB image URLs, 1–10 rating control.
3. **Ask the agent** — chat box; tool calls are rendered as they happen so the
   grader can see the agent working rather than being told it did.
4. **Watchlist** — updates live as the agent writes.

Showing tool calls inline is a deliberate choice: the capstone asks to *demonstrate*
the agent taking actions, and a visible `add_to_watchlist(...) → ok` is the evidence.

## 9. Scope and cost

Free Edition, educational use, so:

- **~500 movies**, not 14,752. Retrieval demonstrates identically; the page count is a
  notebook parameter, so scale is a config change, not a rewrite.
- **One app.** The optional dashboard from the spec is skipped.
- **Existing Lakebase instance**, new schema.
- **Job run on demand**, not scheduled — nothing bills while idle.
- Free Edition caps the workspace at 3 apps. `databricks-day-1-bootcamp-app` (the
  bootcamp's own reference app, distinct from the `support-desk` Assignment 1
  submission) was deleted to free the slot.

## 10. Testing

| Layer | How |
| --- | --- |
| Document composition | Pure function, unit tested on fixtures — no network |
| Hybrid query builder | Unit tested that filters appear/vanish correctly with `None` args |
| Write-tool validation | Unit tested: bad id, out-of-range score, duplicate insert |
| Retrieval quality | Discrimination test — "funny animated kids movie" must rank *Toy Story* above *Fight Club*; asserted, not eyeballed |
| Agent loop | Tested against a stub LLM returning canned `tool_calls`, so it runs offline |
| End-to-end | Script against the deployed app |

The retrieval discrimination test is the one that matters most: it is the only thing
that proves the embeddings carry signal rather than merely existing.

## 11. Risks

- **Rate limits are a Free Edition budget with an unknown window.** Mitigated by
  adaptive backoff and by embedding once into Lakebase, so the app never embeds at
  request time except for the query string itself (a single short call).
- **TMDB enrichment is 3 extra calls per movie.** At 500 movies that's ~1,500 requests;
  TMDB's limit is generous but the pipeline should checkpoint to bronze so a failure
  mid-run doesn't re-fetch everything.
- **Certification is US-only and often missing.** `exclude_violent` must treat NULL as
  unknown rather than safe, or unrated films leak into a "not too violent" result.
- **The model may not call a tool at all.** The system prompt forbids answering from
  memory, and the UI surfaces tool calls — if a recommendation appears with no tool
  call visible, that is itself the bug report.
