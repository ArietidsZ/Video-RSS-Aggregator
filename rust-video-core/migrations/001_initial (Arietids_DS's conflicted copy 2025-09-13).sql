-- Create videos table
CREATE TABLE videos (
    id TEXT PRIMARY KEY,
    platform TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    url TEXT NOT NULL UNIQUE,
    thumbnail_url TEXT,
    author TEXT NOT NULL,
    duration INTEGER NOT NULL DEFAULT 0,
    views INTEGER NOT NULL DEFAULT 0,
    likes INTEGER NOT NULL DEFAULT 0,
    comments INTEGER NOT NULL DEFAULT 0,
    tags TEXT NOT NULL DEFAULT '[]', -- JSON array
    upload_date DATETIME NOT NULL,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL
);

-- Create video_analyses table
CREATE TABLE video_analyses (
    id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    transcript TEXT,
    language TEXT,
    scene_count INTEGER NOT NULL DEFAULT 0,
    detected_objects INTEGER NOT NULL DEFAULT 0,
    dominant_colors TEXT NOT NULL DEFAULT '[]', -- JSON array
    motion_intensity REAL NOT NULL DEFAULT 0.0,
    processing_time_ms INTEGER NOT NULL DEFAULT 0,
    gpu_backend TEXT NOT NULL,
    analyzed_at DATETIME NOT NULL,
    FOREIGN KEY (video_id) REFERENCES videos (id) ON DELETE CASCADE
);

-- Create feeds table
CREATE TABLE feeds (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    platform TEXT,
    url_pattern TEXT,
    last_updated DATETIME NOT NULL,
    created_at DATETIME NOT NULL
);

-- Create indexes for performance
CREATE INDEX idx_videos_platform ON videos (platform);
CREATE INDEX idx_videos_upload_date ON videos (upload_date DESC);
CREATE INDEX idx_videos_created_at ON videos (created_at DESC);
CREATE INDEX idx_videos_author ON videos (author);

CREATE INDEX idx_video_analyses_video_id ON video_analyses (video_id);
CREATE INDEX idx_video_analyses_analyzed_at ON video_analyses (analyzed_at DESC);
CREATE INDEX idx_video_analyses_gpu_backend ON video_analyses (gpu_backend);

CREATE INDEX idx_feeds_platform ON feeds (platform);
CREATE INDEX idx_feeds_last_updated ON feeds (last_updated DESC);

-- Create unique constraint on video_analyses to prevent duplicate analyses
CREATE UNIQUE INDEX idx_video_analyses_unique ON video_analyses (video_id);