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
| Embedding batch ≤ 8, ~2s gap | 2.9 docs/sec sustained | 1,000 movies ≈ 6 min |
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

An HNSW index with `vector_cosine_ops` backs the ordering. At 1,000 rows a sequential
scan would also be fast; the index is there because it is the correct structure for the
job and because the page count is meant to be raised without a schema change.

One caveat that the index makes necessary: HNSW is an **approximate** search, and
Postgres applies the `WHERE` predicates *after* the index returns its candidate set. A
narrow filter (say, `runtime < 90` and G-rated) can therefore return fewer than `k`
rows, or none, even when matching movies exist. The search tool handles this by
over-fetching candidates (`ef_search` raised, `LIMIT k * 10` before filtering) and
falling back to an exact scan when the filtered result is short. At this corpus size an
exact scan is still fast, so correctness is never traded away for speed.

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

- **1,000 movies** (50 `discover` pages), out of the 14,752 available. Large enough
  that retrieval has real competition to rank against and the HNSW index does useful
  work, small enough that a full rebuild is ~10 minutes. The page count is a notebook
  parameter, so scaling up later is a config change rather than a rewrite.
- **Existing Lakebase instance**, new schema.
- **Job run on demand**, not scheduled — nothing bills while idle.
- Free Edition caps the workspace at 3 apps. `databricks-day-1-bootcamp-app` (the
  bootcamp's own reference app, distinct from the `support-desk` Assignment 1
  submission) was deleted to free the slot, leaving two in use.

### What 1,000 costs

| | |
| --- | --- |
| `discover` pages | 50 requests |
| Enrichment (`keywords`, `credits`, `release_dates`) | 3 × 1,000 = 3,000 requests |
| Embedding requests | 125 (batch of 8) |
| Embedding wall-clock | ~6 min at the measured 2.9 docs/sec |
| Vector storage | 1,000 × 1024 × 4B ≈ 4 MB |

Two consequences follow and are **requirements, not nice-to-haves**:

1. **The pipeline must be resumable.** A 3,000-request enrichment stage and 125
   embedding requests against a budget that was only ever measured over 6 consecutive
   calls will sometimes be interrupted. Bronze is written incrementally and the
   embedding stage processes only movies whose stored document hash differs, so a
   re-run continues instead of restarting.
2. **The document hash gates re-embedding.** `movie_embeddings` stores the SHA-256 of
   the composed document. Re-running the pipeline re-embeds only movies whose document
   actually changed, so iterating on the composition in §4 costs nothing for the
   movies it did not affect. This matters more than the raw scale suggests — §4 is the
   part most likely to need tuning after the first retrieval test.

### Optional dashboard app

The spec's dashboard is extra credit. With the planner deployed the workspace is at the
3-app cap, so it is **out of scope unless a slot is freed**. It will be attempted once
the planner is live, at which point the cap is a measured fact rather than an
assumption; if it blocks, the project ships without it as agreed.

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
- **TMDB enrichment is 3 extra calls per movie** — 3,000 requests at this scale. Still
  the longest-running stage and the most likely to be interrupted. Mitigated by the
  resumability requirement in §9; bronze is the checkpoint.
- **The embedding budget window is unknown**, and 125 requests is well past what was
  actually measured (6 consecutive). The stage may hit the limit partway through. This
  is survivable only because of hash-gated resumption — re-running picks up where it
  stopped. If the limit proves tighter than measured, the honest fallback is to lower
  the page count, not to pretend a partial corpus is complete.
- **A 1,000-movie catalog will miss things users ask for.** `discover` sorted by
  popularity skews recent and mainstream, so a request for an obscure or older title
  can legitimately find nothing. The agent must say so rather than substituting a
  loosely similar match — this is what the "relay the error, don't guess" guardrail in
  §7 is for.
- **Certification is US-only and often missing.** `exclude_violent` must treat NULL as
  unknown rather than safe, or unrated films leak into a "not too violent" result.
- **The model may not call a tool at all.** The system prompt forbids answering from
  memory, and the UI surfaces tool calls — if a recommendation appears with no tool
  call visible, that is itself the bug report.
