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
        # unrated films leak into a "not too violent" result, so exclude both
        # explicitly: the IS NULL short-circuits the OR to TRUE, dropping the
        # row, rather than relying on NULL propagation through <> ALL.
        where.append(
            "NOT (m.certification IS NULL "
            "OR m.certification = ANY (%(violent_certs)s))"
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
        SELECT m.*, 1 - c.distance AS similarity
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
