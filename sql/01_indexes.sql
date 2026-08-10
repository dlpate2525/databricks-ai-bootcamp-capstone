SET search_path TO movie_night, public;

-- Approximate NN search. m/ef_construction are pgvector defaults; at 5k rows
-- the build is seconds.
CREATE INDEX IF NOT EXISTS movie_embeddings_hnsw
    ON movie_embeddings USING hnsw (embedding vector_cosine_ops);

-- Filters applied alongside the vector ordering (see retrieval.py).
CREATE INDEX IF NOT EXISTS movies_runtime_idx      ON movies (runtime);
CREATE INDEX IF NOT EXISTS movies_year_idx         ON movies (release_year);
CREATE INDEX IF NOT EXISTS movies_cert_idx         ON movies (certification);
CREATE INDEX IF NOT EXISTS movies_genres_gin       ON movies USING gin (genres);
CREATE INDEX IF NOT EXISTS ratings_movie_idx       ON ratings (movie_id);
CREATE INDEX IF NOT EXISTS watchlist_group_idx     ON watchlist_items (group_id);
