"""Populate the live Lakebase movie catalog end to end.

Fetches TMDB discover pages, enriches each movie (keywords/credits/release
dates), normalizes, composes a document, embeds it, and UPSERTs into
`movies` and `movie_embeddings`. Resumable: re-running skips movies whose
stored doc_sha256 already matches the freshly composed document.

    python scripts/load_movies.py --pages 15 --min-votes 200
"""

import argparse
import sys
import time

from databricks.sdk import WorkspaceClient

from movienight.tmdb_client import TMDBClient, normalize_movie, read_token
from movienight.documents import compose_document, doc_hash
from movienight.embeddings import client_from_workspace, MAX_BATCH
from movienight.db import connection

UPSERT_MOVIE = """
INSERT INTO movies (movie_id, title, release_year, tagline, overview, runtime,
                     certification, genres, keywords, cast_names, director,
                     poster_path, vote_average, vote_count, popularity, updated_at)
VALUES (%(movie_id)s, %(title)s, %(release_year)s, %(tagline)s, %(overview)s,
        %(runtime)s, %(certification)s, %(genres)s, %(keywords)s, %(cast_names)s,
        %(director)s, %(poster_path)s, %(vote_average)s, %(vote_count)s,
        %(popularity)s, now())
ON CONFLICT (movie_id) DO UPDATE SET
    title = EXCLUDED.title,
    release_year = EXCLUDED.release_year,
    tagline = EXCLUDED.tagline,
    overview = EXCLUDED.overview,
    runtime = EXCLUDED.runtime,
    certification = EXCLUDED.certification,
    genres = EXCLUDED.genres,
    keywords = EXCLUDED.keywords,
    cast_names = EXCLUDED.cast_names,
    director = EXCLUDED.director,
    poster_path = EXCLUDED.poster_path,
    vote_average = EXCLUDED.vote_average,
    vote_count = EXCLUDED.vote_count,
    popularity = EXCLUDED.popularity,
    updated_at = now()
"""

UPSERT_EMBEDDING = """
INSERT INTO movie_embeddings (movie_id, document, doc_sha256, embedding, updated_at)
VALUES (%(movie_id)s, %(document)s, %(doc_sha256)s, %(embedding)s::vector, now())
ON CONFLICT (movie_id) DO UPDATE SET
    document = EXCLUDED.document,
    doc_sha256 = EXCLUDED.doc_sha256,
    embedding = EXCLUDED.embedding,
    updated_at = now()
"""


def existing_hashes(conn, movie_ids):
    if not movie_ids:
        return {}
    rows = conn.execute(
        "SELECT movie_id, doc_sha256 FROM movie_embeddings WHERE movie_id = ANY(%s)",
        (list(movie_ids),),
    ).fetchall()
    return {r["movie_id"]: r["doc_sha256"] for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=15)
    ap.add_argument("--min-votes", type=int, default=200)
    args = ap.parse_args()

    start = time.time()
    ws = WorkspaceClient()
    token = read_token(ws)
    tmdb = TMDBClient(token)
    embedder = client_from_workspace(ws)

    # --- Fetch discover pages -------------------------------------------------
    movie_ids = []
    for page in range(1, args.pages + 1):
        try:
            results = tmdb.discover_page(page, min_votes=args.min_votes)
        except Exception as e:
            print(f"  page {page}: FAILED to fetch discover page: {e}")
            continue
        movie_ids.extend(r["id"] for r in results if r.get("id"))
    movie_ids = list(dict.fromkeys(movie_ids))  # dedupe, keep order
    print(f"discovered {len(movie_ids)} unique movie ids across {args.pages} pages")

    records = []
    failed = 0
    for i, mid in enumerate(movie_ids, start=1):
        try:
            bundle = tmdb.movie_bundle(mid)
            record = normalize_movie(bundle)
            if not record.get("movie_id"):
                raise ValueError("missing movie_id in normalized record")
            records.append(record)
        except Exception as e:
            failed += 1
            print(f"  movie {mid}: FAILED to fetch/normalize: {e}")
            continue
        if i % 50 == 0:
            print(f"  fetched {i}/{len(movie_ids)} movies ({failed} failed so far)")

    print(f"fetched {len(records)} movies ok, {failed} failed")

    # --- UPSERT movies, compose documents --------------------------------------
    docs = {}  # movie_id -> (document, hash)
    with connection() as conn:
        for i, record in enumerate(records, start=1):
            params = dict(record)
            params["genres"] = list(record.get("genres") or [])
            params["keywords"] = list(record.get("keywords") or [])
            params["cast_names"] = list(record.get("cast") or [])
            params.pop("cast", None)
            try:
                conn.execute(UPSERT_MOVIE, params)
            except Exception as e:
                print(f"  movie {record.get('movie_id')}: FAILED to upsert row: {e}")
                conn.rollback()
                continue

            doc = compose_document(record)
            docs[record["movie_id"]] = (doc, doc_hash(doc))

            if i % 50 == 0:
                conn.commit()
                print(f"  upserted {i}/{len(records)} movie rows")
        conn.commit()
        print(f"movies table: upserted {len(records)} rows")

        # --- Skip movies whose stored hash already matches -------------------
        existing = existing_hashes(conn, docs.keys())
        to_embed = [
            mid for mid, (doc, h) in docs.items()
            if existing.get(mid) != h
        ]
        print(f"{len(to_embed)}/{len(docs)} movies need (re-)embedding "
              f"({len(docs) - len(to_embed)} already up to date)")

        # --- Embed in batches of MAX_BATCH and upsert -------------------------
        embedded = 0
        for start_idx in range(0, len(to_embed), MAX_BATCH):
            batch_ids = to_embed[start_idx:start_idx + MAX_BATCH]
            texts = [docs[mid][0] for mid in batch_ids]
            try:
                vectors = embedder.embed(texts)
            except Exception as e:
                print(f"  embedding batch starting at {start_idx}: FAILED: {e}")
                continue

            for mid, vector in zip(batch_ids, vectors):
                doc, h = docs[mid]
                try:
                    conn.execute(UPSERT_EMBEDDING, {
                        "movie_id": mid,
                        "document": doc,
                        "doc_sha256": h,
                        "embedding": str(list(vector)),
                    })
                    embedded += 1
                except Exception as e:
                    print(f"  movie {mid}: FAILED to upsert embedding: {e}")
                    conn.rollback()
                    continue

            conn.commit()
            if embedded and embedded % 50 < MAX_BATCH:
                print(f"  embedded {embedded}/{len(to_embed)} movies")

        print(f"movie_embeddings table: embedded/upserted {embedded} rows")

        counts = {}
        for table in ["movies", "movie_embeddings"]:
            row = conn.execute(f"SELECT count(*) AS n FROM {table}").fetchone()
            counts[table] = row["n"]

    elapsed = time.time() - start
    print("\nFINAL ROW COUNTS:")
    for table, n in counts.items():
        print(f"  {table:20s} {n}")
    print(f"\nelapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
