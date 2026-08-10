import pytest

from movienight.retrieval import SearchFilters, build_search_sql, _exact_sql


def test_no_filters_produces_only_vector_ordering():
    sql, params = build_search_sql(SearchFilters(), k=5)
    assert "ORDER BY" in sql and "<=>" in sql
    assert "WHERE" not in sql, "no filters set means no filter clause at all"
    assert params["k"] == 5


def test_runtime_filter_appears_and_is_parameterised():
    sql, params = build_search_sql(
        SearchFilters(max_runtime_minutes=120), k=5)
    assert "m.runtime <= %(max_runtime)s" in sql
    assert params["max_runtime"] == 120
    assert "120" not in sql          # value must not be interpolated


def test_exclude_violent_filters_certs_and_treats_null_as_unknown():
    sql, params = build_search_sql(SearchFilters(exclude_violent=True), k=5)
    # certification is now always in the SELECT list, so "certification" in
    # sql alone would pass even if the filter clause vanished. Assert the
    # actual WHERE clause text instead.
    assert (
        "NOT (m.certification IS NULL "
        "OR m.certification = ANY (%(violent_certs)s))"
    ) in sql
    assert "IS NULL" in sql, "NULL certification must be excluded, not assumed safe"
    # psycopg3 needs a list (not a tuple) to bind as a Postgres array via
    # ANY/ALL. This is the literal expected value, not list(VIOLENT_CERTS) --
    # that would tautologically pass even if the constant's contents changed.
    assert params["violent_certs"] == ["R", "NC-17"]


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


def test_exact_sql_drops_the_candidate_prelimit():
    sql, _ = _exact_sql(SearchFilters(max_runtime_minutes=90), k=5)
    assert "candidate_limit" not in sql


def test_exact_sql_raises_loudly_if_the_replace_target_is_missing(monkeypatch):
    # Simulate the template drifting so the literal LIMIT string no longer
    # matches -- the removal must fail loudly, not silently no-op and leave
    # the candidate LIMIT in place.
    import movienight.retrieval as retrieval

    original_build_search_sql = retrieval.build_search_sql

    def drifted_build_search_sql(filters, k):
        sql, params = original_build_search_sql(filters, k)
        drifted = sql.replace(
            "LIMIT %(candidate_limit)s", "LIMIT %(candidate_cap)s")
        return drifted, params

    monkeypatch.setattr(retrieval, "build_search_sql", drifted_build_search_sql)
    with pytest.raises(RuntimeError):
        retrieval._exact_sql(SearchFilters(), k=5)
