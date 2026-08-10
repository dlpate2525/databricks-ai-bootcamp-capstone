import json
import pathlib

from movienight.tmdb_client import TMDBClient, normalize_movie, us_certification

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
    assert len(rec["cast"]) == 6
    assert rec["cast"][0] == "Edward Norton"
    assert rec["poster_path"].startswith("/")
    assert set(rec) == {"movie_id", "title", "release_year", "tagline", "overview",
                         "runtime", "certification", "genres", "keywords", "cast",
                         "director", "poster_path", "vote_average", "vote_count",
                         "popularity"}


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


def test_discover_page_calls_expected_path_and_params(monkeypatch):
    captured = {}

    def fake_get(self, path, **params):
        captured["path"] = path
        captured["params"] = params
        return {"results": ["movie-a", "movie-b"]}

    monkeypatch.setattr(TMDBClient, "_get", fake_get)
    client = TMDBClient("fake-token")
    result = client.discover_page(page=3, min_votes=200)

    assert captured["path"] == "/discover/movie"
    assert captured["params"]["sort_by"] == "popularity.desc"
    assert captured["params"]["page"] == 3
    assert captured["params"]["vote_count.gte"] == 200
    assert result == ["movie-a", "movie-b"]
