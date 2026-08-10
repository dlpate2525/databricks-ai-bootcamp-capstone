CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS movie_night;
SET search_path TO movie_night, public;

CREATE TABLE IF NOT EXISTS users (
    user_id     bigserial PRIMARY KEY,
    email       text UNIQUE NOT NULL,
    display_name text NOT NULL,
    is_demo     boolean NOT NULL DEFAULT false,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS groups (
    group_id    bigserial PRIMARY KEY,
    name        text NOT NULL,
    created_by  bigint REFERENCES users(user_id),
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS group_members (
    group_id    bigint NOT NULL REFERENCES groups(group_id) ON DELETE CASCADE,
    user_id     bigint NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    joined_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (group_id, user_id)
);

CREATE TABLE IF NOT EXISTS movies (
    movie_id      bigint PRIMARY KEY,          -- TMDB id, natural key
    title         text NOT NULL,
    release_year  int,
    tagline       text NOT NULL DEFAULT '',
    overview      text NOT NULL DEFAULT '',
    runtime       int,
    certification text,                        -- NULL = unknown, not "safe"
    genres        text[] NOT NULL DEFAULT '{}',
    keywords      text[] NOT NULL DEFAULT '{}',
    cast_names    text[] NOT NULL DEFAULT '{}',
    director      text,
    poster_path   text,
    vote_average  real NOT NULL DEFAULT 0,
    vote_count    int  NOT NULL DEFAULT 0,
    popularity    real NOT NULL DEFAULT 0,
    updated_at    timestamptz NOT NULL DEFAULT now()
);

-- Separate table so re-embedding never rewrites movies, and so a movie can
-- exist before its vector does (the pipeline is resumable).
CREATE TABLE IF NOT EXISTS movie_embeddings (
    movie_id   bigint PRIMARY KEY REFERENCES movies(movie_id) ON DELETE CASCADE,
    document   text NOT NULL,
    doc_sha256 char(64) NOT NULL,     -- gates re-embedding on content change
    embedding  vector(1024) NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ratings (
    user_id    bigint NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    movie_id   bigint NOT NULL REFERENCES movies(movie_id) ON DELETE CASCADE,
    score      int NOT NULL CHECK (score BETWEEN 1 AND 10),
    rated_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, movie_id)   -- re-rating is an UPSERT, never a duplicate
);

CREATE TABLE IF NOT EXISTS watchlist_items (
    item_id    bigserial PRIMARY KEY,
    group_id   bigint NOT NULL REFERENCES groups(group_id) ON DELETE CASCADE,
    movie_id   bigint NOT NULL REFERENCES movies(movie_id) ON DELETE CASCADE,
    status     text NOT NULL DEFAULT 'queued'
               CHECK (status IN ('queued', 'watched', 'skipped')),
    reason     text,
    added_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (group_id, movie_id)       -- makes add_to_watchlist idempotent
);

-- Audit trail of what the agent did and why.
CREATE TABLE IF NOT EXISTS recommendations (
    rec_id            bigserial PRIMARY KEY,
    group_id          bigint NOT NULL REFERENCES groups(group_id) ON DELETE CASCADE,
    user_query        text NOT NULL,
    candidate_ids     bigint[] NOT NULL DEFAULT '{}',
    chosen_movie_id   bigint REFERENCES movies(movie_id),
    rationale         text,
    created_at        timestamptz NOT NULL DEFAULT now()
);
