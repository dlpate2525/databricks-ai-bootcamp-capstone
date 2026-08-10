# Databricks notebook source
# MAGIC %md
# MAGIC # Movie ingestion pipeline
# MAGIC TMDB -> bronze Delta -> silver Delta -> embeddings -> Lakebase.
# MAGIC
# MAGIC Resumable by design: 250 discover pages plus 15,000 enrichment calls plus
# MAGIC ~29 minutes of embedding will eventually be interrupted. Bronze is the
# MAGIC checkpoint; the embedding stage only processes movies whose document hash
# MAGIC differs from what is already stored.

# COMMAND ----------
%pip install psycopg[binary] psycopg_pool
dbutils.library.restartPython()

# COMMAND ----------
import sys
REPO = "/Workspace/Users/dlpate2525@gmail.com/movie-night"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

dbutils.widgets.text("pages", "50")
dbutils.widgets.text("min_votes", "200")
dbutils.widgets.dropdown("reembed_all", "false", ["true", "false"])

PAGES = int(dbutils.widgets.get("pages"))
MIN_VOTES = int(dbutils.widgets.get("min_votes"))
REEMBED_ALL = dbutils.widgets.get("reembed_all") == "true"

CATALOG, SCHEMA = "workspace", "default"
BRONZE = f"{CATALOG}.{SCHEMA}.movies_bronze"
SILVER = f"{CATALOG}.{SCHEMA}.movies_silver"

# COMMAND ----------
# MAGIC %md ## Stage 1 - discover, into bronze
import json

from databricks.sdk import WorkspaceClient
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType, LongType

from movienight.tmdb_client import TMDBClient, normalize_movie, read_token

w = WorkspaceClient()
tmdb = TMDBClient(read_token(w))

spark.sql(f"CREATE TABLE IF NOT EXISTS {BRONZE} "
          "(movie_id BIGINT, payload STRING, fetched_at TIMESTAMP) USING DELTA")

already = set()
if spark.catalog.tableExists(BRONZE):
    already = {r.movie_id for r in spark.table(BRONZE).select("movie_id").collect()}
print(f"bronze already holds {len(already)} movies")

discovered = []
for page in range(1, PAGES + 1):
    for row in tmdb.discover_page(page, MIN_VOTES):
        discovered.append(row["id"])
    if page % 25 == 0:
        print(f"  discovered through page {page}: {len(discovered)} ids")

todo = [mid for mid in dict.fromkeys(discovered) if mid not in already]
print(f"{len(todo)} movies need enrichment")

# COMMAND ----------
# MAGIC %md ## Stage 2 - enrich, appending to bronze in chunks so a failure keeps progress
CHUNK = 200
schema = StructType([
    StructField("movie_id", LongType()),
    StructField("payload", StringType()),
])

for start in range(0, len(todo), CHUNK):
    batch = todo[start:start + CHUNK]
    rows = []
    for movie_id in batch:
        try:
            rows.append((movie_id, json.dumps(tmdb.movie_bundle(movie_id))))
        except Exception as exc:
            print(f"  skipping {movie_id}: {exc}")
    if rows:
        (spark.createDataFrame(rows, schema)
             .withColumn("fetched_at", F.current_timestamp())
             .write.mode("append").saveAsTable(BRONZE))
    print(f"  bronze +{len(rows)} ({start + len(batch)}/{len(todo)})")

# COMMAND ----------
# MAGIC %md ## Stage 3 - silver: typed columns and the composed document
from movienight.documents import compose_document, doc_hash

bronze = spark.table(BRONZE).dropDuplicates(["movie_id"])

def to_silver(iterator):
    import pandas as pd
    for pdf in iterator:
        records = []
        for _, row in pdf.iterrows():
            rec = normalize_movie(json.loads(row["payload"]))
            if not rec["movie_id"] or not rec["title"]:
                continue
            document = compose_document(rec)
            rec["document"] = document
            rec["doc_sha256"] = doc_hash(document)
            records.append(rec)
        yield pd.DataFrame(records)

silver_schema = ("movie_id long, title string, release_year int, tagline string, "
                 "overview string, runtime int, certification string, "
                 "genres array<string>, keywords array<string>, cast array<string>, "
                 "director string, poster_path string, vote_average double, "
                 "vote_count int, popularity double, document string, "
                 "doc_sha256 string")

silver = bronze.mapInPandas(to_silver, schema=silver_schema)
silver.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(SILVER)
print(f"silver rows: {spark.table(SILVER).count()}")
display(spark.table(SILVER).select("title", "runtime", "certification",
                                   "document").limit(5))

# COMMAND ----------
# MAGIC %md ## Stage 4 - upsert movies into Lakebase
from movienight.db import connection

rows = spark.table(SILVER).collect()
print(f"upserting {len(rows)} movies")

UPSERT = """
INSERT INTO movies (movie_id, title, release_year, tagline, overview, runtime,
                    certification, genres, keywords, cast_names, director,
                    poster_path, vote_average, vote_count, popularity, updated_at)
VALUES (%(movie_id)s, %(title)s, %(release_year)s, %(tagline)s, %(overview)s,
        %(runtime)s, %(certification)s, %(genres)s, %(keywords)s, %(cast)s,
        %(director)s, %(poster_path)s, %(vote_average)s, %(vote_count)s,
        %(popularity)s, now())
ON CONFLICT (movie_id) DO UPDATE SET
    title=EXCLUDED.title, release_year=EXCLUDED.release_year,
    tagline=EXCLUDED.tagline, overview=EXCLUDED.overview,
    runtime=EXCLUDED.runtime, certification=EXCLUDED.certification,
    genres=EXCLUDED.genres, keywords=EXCLUDED.keywords,
    cast_names=EXCLUDED.cast_names, director=EXCLUDED.director,
    poster_path=EXCLUDED.poster_path, vote_average=EXCLUDED.vote_average,
    vote_count=EXCLUDED.vote_count, popularity=EXCLUDED.popularity,
    updated_at=now()
"""

with connection() as conn:
    for i in range(0, len(rows), 500):
        batch = rows[i:i + 500]
        params = []
        for r in batch:
            d = r.asDict()
            d["genres"] = list(d.get("genres") or [])
            d["keywords"] = list(d.get("keywords") or [])
            d["cast"] = list(d.get("cast") or [])
            params.append(d)
        conn.cursor().executemany(UPSERT, params)
        conn.commit()
        print(f"  {min(i + 500, len(rows))}/{len(rows)}")

# COMMAND ----------
# MAGIC %md ## Stage 5 - embed only what changed
from movienight.embeddings import MAX_BATCH, client_from_workspace

embedder = client_from_workspace(w)

with connection() as conn:
    existing = {r["movie_id"]: r["doc_sha256"] for r in conn.execute(
        "SELECT movie_id, doc_sha256 FROM movie_embeddings").fetchall()}

pending = [r.asDict() for r in rows
           if REEMBED_ALL or existing.get(r["movie_id"]) != r["doc_sha256"]]
print(f"{len(pending)} of {len(rows)} need embedding "
      f"({len(rows) - len(pending)} unchanged)")

EMBED_UPSERT = """
INSERT INTO movie_embeddings (movie_id, document, doc_sha256, embedding, updated_at)
VALUES (%(movie_id)s, %(document)s, %(doc_sha256)s, %(embedding)s::vector, now())
ON CONFLICT (movie_id) DO UPDATE SET
    document=EXCLUDED.document, doc_sha256=EXCLUDED.doc_sha256,
    embedding=EXCLUDED.embedding, updated_at=now()
"""

with connection() as conn:
    for i in range(0, len(pending), MAX_BATCH):
        chunk = pending[i:i + MAX_BATCH]
        vectors = embedder.embed([c["document"] for c in chunk])
        conn.cursor().executemany(EMBED_UPSERT, [
            {"movie_id": c["movie_id"], "document": c["document"],
             "doc_sha256": c["doc_sha256"], "embedding": str(list(v))}
            for c, v in zip(chunk, vectors)])
        conn.commit()
        if i % (MAX_BATCH * 25) == 0:
            print(f"  embedded {min(i + MAX_BATCH, len(pending))}/{len(pending)}")

# COMMAND ----------
# MAGIC %md ## Stage 6 - verify
with connection() as conn:
    for label, sql in [
        ("movies", "SELECT count(*) AS n FROM movies"),
        ("embeddings", "SELECT count(*) AS n FROM movie_embeddings"),
        ("missing vectors",
         "SELECT count(*) AS n FROM movies m LEFT JOIN movie_embeddings e "
         "USING (movie_id) WHERE e.movie_id IS NULL"),
        ("with runtime", "SELECT count(*) AS n FROM movies WHERE runtime IS NOT NULL"),
        ("with certification",
         "SELECT count(*) AS n FROM movies WHERE certification IS NOT NULL"),
    ]:
        print(f"{label:20s} {conn.execute(sql).fetchone()['n']}")
