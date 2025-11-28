-- Add migration script here
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TYPE feed_status AS ENUM ('ACTIVE', 'ERROR', 'PAUSED');

CREATE TABLE feeds (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    url TEXT NOT NULL UNIQUE,
    title TEXT,
    description TEXT,
    status feed_status NOT NULL DEFAULT 'ACTIVE',
    last_fetched_at TIMESTAMPTZ,
    error_count INT DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    feed_id UUID NOT NULL REFERENCES feeds(id) ON DELETE CASCADE,
    external_id TEXT NOT NULL, -- ID from the RSS feed (GUID)
    url TEXT NOT NULL,
    title TEXT,
    description TEXT,
    published_at TIMESTAMPTZ,
    content TEXT,
    author TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(feed_id, external_id)
);

CREATE INDEX idx_items_feed_id ON items(feed_id);
CREATE INDEX idx_items_published_at ON items(published_at DESC);
