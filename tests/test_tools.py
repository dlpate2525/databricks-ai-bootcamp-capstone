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


def test_model_supplied_values_reach_sql_only_as_parameters():
    """The model chooses these values, so none may be interpolated into SQL text."""
    conn = FakeConn(responses={"FROM movies WHERE movie_id": [{"movie_id": 5}]})
    dispatch("record_rating", {"movie_id": 5, "score": 8}, ctx(conn))
    insert = [c for c in conn.calls if "INSERT INTO ratings" in c[0]][0]
    sql, params = insert
    assert "5" not in sql and "8" not in sql, f"value interpolated into SQL: {sql}"
    assert params["movie_id"] == 5 and params["score"] == 8


def test_sql_injection_attempt_stays_inert_in_params():
    conn = FakeConn(responses={"FROM movies WHERE movie_id": [{"movie_id": 5}]})
    result = dispatch("add_to_watchlist",
                      {"movie_id": 5, "reason": "'; DROP TABLE movies; --"},
                      ctx(conn))
    assert result["status"] == "ok"
    insert = [c for c in conn.calls if "INSERT INTO watchlist_items" in c[0]][0]
    assert "DROP TABLE" not in insert[0]
    assert insert[1]["reason"] == "'; DROP TABLE movies; --"


def test_save_recommendation_candidate_ids_param_is_a_list_not_a_tuple():
    conn = FakeConn(responses={"FROM movies WHERE movie_id":
                                [{"movie_id": 1}, {"movie_id": 2}]})
    result = dispatch("save_recommendation",
                      {"user_query": "q", "candidate_ids": [1, 2],
                       "chosen_movie_id": 1, "rationale": "r"},
                      ctx(conn))
    assert result["status"] == "ok"
    insert = [c for c in conn.calls
              if "INSERT INTO recommendations" in c[0]][0]
    assert isinstance(insert[1]["c"], list), \
        f"candidate_ids param must be a list, got {type(insert[1]['c'])}"
    assert insert[1]["c"] == [1, 2]


def test_save_recommendation_rejects_unknown_chosen_movie_id():
    conn = FakeConn(responses={"FROM movies WHERE movie_id": []})
    result = dispatch("save_recommendation",
                      {"user_query": "q", "candidate_ids": [],
                       "chosen_movie_id": 999999, "rationale": "r"},
                      ctx(conn))
    assert result["status"] == "error"
    assert "999999" in result["message"]
    assert not conn.committed
    assert not [c for c in conn.calls if "INSERT INTO recommendations" in c[0]]


def test_save_recommendation_rejects_unknown_candidate_id():
    conn = FakeConn(responses={"FROM movies WHERE movie_id":
                                [{"movie_id": 5, "title": "X"}]})
    result = dispatch("save_recommendation",
                      {"user_query": "q", "candidate_ids": [5, 999999],
                       "chosen_movie_id": 5, "rationale": "r"},
                      ctx(conn))
    assert result["status"] == "error"
    assert "999999" in result["message"]
    assert not conn.committed
    assert not [c for c in conn.calls if "INSERT INTO recommendations" in c[0]]


def test_record_rating_rejects_infinite_score_cleanly():
    conn = FakeConn(responses={"FROM movies WHERE movie_id": [{"movie_id": 5}]})
    result = dispatch("record_rating", {"movie_id": 5, "score": float("inf")},
                      ctx(conn))
    assert result["status"] == "error"
    assert "OverflowError" not in result["message"]
