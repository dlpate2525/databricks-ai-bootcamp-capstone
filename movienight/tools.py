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


class ToolError(Exception):
    """Internal only - never surfaced past dispatch()."""


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
