-- PostgreSQL schema for the knowledge-base assistant.
-- Requires the pgvector extension: https://github.com/pgvector/pgvector

CREATE EXTENSION IF NOT EXISTS vector;

-- Corpus: one row per text chunk. bge-m3 produces 1024-dimensional embeddings.
CREATE TABLE IF NOT EXISTS article_chunks (
    id          BIGSERIAL PRIMARY KEY,
    date        DATE,
    source      TEXT,                 -- e.g. 'product'; the product role filters on it
    chunk_text  TEXT NOT NULL,
    embedding   vector(1024) NOT NULL
);

CREATE INDEX IF NOT EXISTS article_chunks_embedding_idx
    ON article_chunks USING hnsw (embedding vector_l2_ops);

-- System prompts by name; the bot loads 'expert_promt' at start.
CREATE TABLE IF NOT EXISTS promt_data (
    name     TEXT PRIMARY KEY,
    content  TEXT NOT NULL
);

-- Allow-list; created by the bot on start as well, kept here for completeness.
CREATE TABLE IF NOT EXISTS access_control (
    user_id    BIGINT PRIMARY KEY,
    username   VARCHAR(255),
    full_name  VARCHAR(255),
    is_allowed BOOLEAN DEFAULT FALSE
);
