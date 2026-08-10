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
        raise TMDBError(f"TMDB {path} failed after {self._max_retries} attempts: {last}") from last

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
