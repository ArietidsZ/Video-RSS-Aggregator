-- Initial database schema for Video RSS Aggregator
-- Database tables with proper indexing

-- Videos table - core video metadata
CREATE TABLE videos (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    url TEXT NOT NULL UNIQUE,
    author TEXT NOT NULL,
    platform TEXT NOT NULL CHECK (platform IN ('bilibili', 'youtube', 'douyin', 'kuaishou')),
    upload_date TIMESTAMP NOT NULL,
    duration_seconds INTEGER,
    view_count BIGINT DEFAULT 0,
    like_count BIGINT DEFAULT 0,
    comment_count BIGINT DEFAULT 0,
    thumbnail_url TEXT,

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_checked TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Data quality
    data_source TEXT NOT NULL,
    extraction_method TEXT,

    -- Performance optimizations
    search_vector TEXT, -- For full-text search
    content_hash TEXT,  -- For deduplication

    -- Soft delete
    deleted_at TIMESTAMP
);

-- Tags table - normalized tag storage
CREATE TABLE tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Video tags junction table
CREATE TABLE video_tags (
    video_id TEXT NOT NULL,
    tag_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (video_id, tag_id),
    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE,
    FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
);

-- Transcriptions table - AI transcription data
CREATE TABLE transcriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL UNIQUE,
    paragraph_summary TEXT,
    sentence_subtitle TEXT,
    full_transcript TEXT,
    status TEXT NOT NULL CHECK (status IN ('success', 'pending', 'failed', 'unavailable')),

    -- Model information
    transcriber_model TEXT NOT NULL,
    summarizer_model TEXT NOT NULL,
    processing_time_ms INTEGER,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Quality metrics
    confidence_score REAL,
    language_detected TEXT,

    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
);

-- Cache entries table - application-level caching
CREATE TABLE cache_entries (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- RSS feeds table - generated RSS feed tracking
CREATE TABLE rss_feeds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    platforms TEXT NOT NULL, -- Comma-separated platform list
    content_hash TEXT NOT NULL,
    xml_content TEXT NOT NULL,
    etag TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Performance metrics table
CREATE TABLE performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    metric_type TEXT NOT NULL CHECK (metric_type IN ('counter', 'gauge', 'histogram')),
    labels TEXT, -- JSON string for metric labels
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- API requests table - request logging and analytics
CREATE TABLE api_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    endpoint TEXT NOT NULL,
    method TEXT NOT NULL,
    status_code INTEGER NOT NULL,
    response_time_ms INTEGER NOT NULL,
    user_agent TEXT,
    ip_address TEXT,
    request_size INTEGER,
    response_size INTEGER,
    cache_hit BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Platform status table - track platform availability
CREATE TABLE platform_status (
    platform TEXT PRIMARY KEY CHECK (platform IN ('bilibili', 'youtube', 'douyin', 'kuaishou')),
    is_available BOOLEAN NOT NULL DEFAULT TRUE,
    last_success TIMESTAMP,
    last_failure TIMESTAMP,
    failure_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    avg_response_time_ms REAL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance optimization

-- Videos table indexes
CREATE INDEX idx_videos_platform ON videos(platform);
CREATE INDEX idx_videos_upload_date ON videos(upload_date DESC);
CREATE INDEX idx_videos_view_count ON videos(view_count DESC);
CREATE INDEX idx_videos_created_at ON videos(created_at DESC);
CREATE INDEX idx_videos_updated_at ON videos(updated_at DESC);
CREATE INDEX idx_videos_author ON videos(author);
CREATE INDEX idx_videos_content_hash ON videos(content_hash);
CREATE INDEX idx_videos_search ON videos(title, description);
CREATE INDEX idx_videos_active ON videos(deleted_at) WHERE deleted_at IS NULL;

-- Video tags indexes
CREATE INDEX idx_video_tags_video_id ON video_tags(video_id);
CREATE INDEX idx_video_tags_tag_id ON video_tags(tag_id);

-- Tags indexes
CREATE INDEX idx_tags_name ON tags(name);

-- Transcriptions indexes
CREATE INDEX idx_transcriptions_video_id ON transcriptions(video_id);
CREATE INDEX idx_transcriptions_status ON transcriptions(status);
CREATE INDEX idx_transcriptions_created_at ON transcriptions(created_at DESC);

-- Cache entries indexes
CREATE INDEX idx_cache_expires_at ON cache_entries(expires_at);
CREATE INDEX idx_cache_last_accessed ON cache_entries(last_accessed);

-- RSS feeds indexes
CREATE INDEX idx_rss_platforms ON rss_feeds(platforms);
CREATE INDEX idx_rss_content_hash ON rss_feeds(content_hash);
CREATE INDEX idx_rss_created_at ON rss_feeds(created_at DESC);

-- Performance metrics indexes
CREATE INDEX idx_metrics_name ON performance_metrics(metric_name);
CREATE INDEX idx_metrics_recorded_at ON performance_metrics(recorded_at DESC);
CREATE INDEX idx_metrics_type ON performance_metrics(metric_type);

-- API requests indexes
CREATE INDEX idx_api_requests_endpoint ON api_requests(endpoint);
CREATE INDEX idx_api_requests_created_at ON api_requests(created_at DESC);
CREATE INDEX idx_api_requests_status ON api_requests(status_code);
CREATE INDEX idx_api_requests_cache_hit ON api_requests(cache_hit);

-- Platform status indexes
CREATE INDEX idx_platform_status_available ON platform_status(is_available);
CREATE INDEX idx_platform_status_updated ON platform_status(updated_at DESC);

-- Create triggers for updated_at timestamps
CREATE TRIGGER update_videos_updated_at
    AFTER UPDATE ON videos
BEGIN
    UPDATE videos SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER update_transcriptions_updated_at
    AFTER UPDATE ON transcriptions
BEGIN
    UPDATE transcriptions SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER update_platform_status_updated_at
    AFTER UPDATE ON platform_status
BEGIN
    UPDATE platform_status SET updated_at = CURRENT_TIMESTAMP WHERE platform = NEW.platform;
END;

-- Initialize platform status
INSERT INTO platform_status (platform, is_available) VALUES
('bilibili', TRUE),
('youtube', TRUE),
('douyin', TRUE),
('kuaishou', TRUE);

-- Create views for common queries

-- Active videos view (non-deleted)
CREATE VIEW active_videos AS
SELECT * FROM videos WHERE deleted_at IS NULL;

-- Recent videos view (last 24 hours)
CREATE VIEW recent_videos AS
SELECT * FROM active_videos
WHERE created_at > datetime('now', '-1 day')
ORDER BY created_at DESC;

-- Popular videos view (high engagement)
CREATE VIEW popular_videos AS
SELECT *,
       (view_count + like_count * 10) as engagement_score
FROM active_videos
WHERE view_count > 1000
ORDER BY engagement_score DESC;

-- Video stats view with aggregations
CREATE VIEW video_stats AS
SELECT
    platform,
    COUNT(*) as total_videos,
    AVG(view_count) as avg_views,
    AVG(like_count) as avg_likes,
    MIN(upload_date) as earliest_video,
    MAX(upload_date) as latest_video
FROM active_videos
GROUP BY platform;

-- Cache efficiency view
CREATE VIEW cache_efficiency AS
SELECT
    COUNT(*) as total_entries,
    COUNT(CASE WHEN expires_at > CURRENT_TIMESTAMP THEN 1 END) as active_entries,
    AVG(access_count) as avg_access_count,
    SUM(CASE WHEN expires_at <= CURRENT_TIMESTAMP THEN 1 ELSE 0 END) as expired_entries
FROM cache_entries;