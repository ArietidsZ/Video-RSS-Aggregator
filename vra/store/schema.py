"""DB schema for VRA v2. FROZEN — repos must not alter these statements."""

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS feeds (
    id TEXT PRIMARY KEY,
    url TEXT NOT NULL UNIQUE,
    title TEXT,
    site_url TEXT,
    etag TEXT,
    last_modified TEXT,
    last_checked TEXT,
    poll_minutes INTEGER NOT NULL DEFAULT 15,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS videos (
    id TEXT PRIMARY KEY,
    canonical_id TEXT NOT NULL UNIQUE,
    source_url TEXT NOT NULL,
    feed_id TEXT REFERENCES feeds(id) ON DELETE SET NULL,
    guid TEXT,
    title TEXT,
    description TEXT,
    duration_s REAL,
    chapters TEXT,
    published_at TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    error TEXT,
    media_path TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_videos_feed ON videos(feed_id);
CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status);

CREATE TABLE IF NOT EXISTS transcripts (
    id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    source TEXT NOT NULL,
    lang TEXT,
    text TEXT NOT NULL,
    segments TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_transcripts_video ON transcripts(video_id);

CREATE TABLE IF NOT EXISTS frames (
    id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    path TEXT NOT NULL,
    ts_s REAL NOT NULL,
    scene_score REAL NOT NULL DEFAULT 0,
    dhash TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_frames_video ON frames(video_id, idx);

CREATE TABLE IF NOT EXISTS summaries (
    id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    summary TEXT NOT NULL,
    key_points TEXT NOT NULL,
    visual_highlights TEXT NOT NULL,
    model TEXT NOT NULL DEFAULT '',
    think TEXT NOT NULL DEFAULT 'off',
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    latency_ms INTEGER NOT NULL DEFAULT 0,
    vram_mb REAL NOT NULL DEFAULT 0,
    transcript_chars INTEGER NOT NULL DEFAULT 0,
    frame_count INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_summaries_video ON summaries(video_id, created_at DESC);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'queued',
    payload TEXT NOT NULL DEFAULT '{}',
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    run_at TEXT,
    video_id TEXT REFERENCES videos(id) ON DELETE CASCADE,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state, run_at);
"""
