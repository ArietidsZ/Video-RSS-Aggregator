use crate::{error::VideoRssError, types::*, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{migrate::MigrateDatabase, Executor, Pool, Row, Sqlite, SqlitePool};
use std::collections::HashMap;
use std::path::Path;
use tracing::{debug, error, info, warn};

#[derive(Debug, Clone)]
pub struct DatabaseConfig {
    pub url: String,
    pub max_connections: u32,
    pub min_connections: u32,
    pub connect_timeout: std::time::Duration,
    pub idle_timeout: std::time::Duration,
    pub max_lifetime: std::time::Duration,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            url: "sqlite:video_rss.db".to_string(),
            max_connections: 20,
            min_connections: 5,
            connect_timeout: std::time::Duration::from_secs(30),
            idle_timeout: std::time::Duration::from_secs(600),
            max_lifetime: std::time::Duration::from_secs(3600),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct VideoRecord {
    pub id: String,
    pub title: String,
    pub description: Option<String>,
    pub url: String,
    pub author: String,
    pub platform: String,
    pub upload_date: DateTime<Utc>,
    pub duration_seconds: Option<i64>,
    pub view_count: i64,
    pub like_count: i64,
    pub comment_count: i64,
    pub thumbnail_url: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub last_checked: DateTime<Utc>,
    pub data_source: String,
    pub extraction_method: Option<String>,
    pub content_hash: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct TranscriptionRecord {
    pub id: i64,
    pub video_id: String,
    pub paragraph_summary: Option<String>,
    pub sentence_subtitle: Option<String>,
    pub full_transcript: Option<String>,
    pub status: String,
    pub transcriber_model: String,
    pub summarizer_model: String,
    pub processing_time_ms: Option<i64>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub confidence_score: Option<f64>,
    pub language_detected: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct PlatformStatusRecord {
    pub platform: String,
    pub is_available: bool,
    pub last_success: Option<DateTime<Utc>>,
    pub last_failure: Option<DateTime<Utc>>,
    pub failure_count: i64,
    pub success_count: i64,
    pub avg_response_time_ms: Option<f64>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoStats {
    pub platform: String,
    pub total_videos: i64,
    pub avg_views: f64,
    pub avg_likes: f64,
    pub earliest_video: DateTime<Utc>,
    pub latest_video: DateTime<Utc>,
}

pub struct Database {
    pool: Pool<Sqlite>,
}

impl Database {
    pub async fn new(config: DatabaseConfig) -> Result<Self> {
        info!("Initializing database connection pool: {}", config.url);

        // Create database if it doesn't exist
        if !Sqlite::database_exists(&config.url).await.unwrap_or(false) {
            info!("Creating database: {}", config.url);
            Sqlite::create_database(&config.url).await?;
        }

        // Create connection pool
        let pool = sqlx::SqlitePool::builder()
            .max_connections(config.max_connections)
            .min_connections(config.min_connections)
            .acquire_timeout(config.connect_timeout)
            .idle_timeout(config.idle_timeout)
            .max_lifetime(config.max_lifetime)
            .build(&config.url)
            .await?;

        // Run migrations
        info!("Running database migrations");
        sqlx::migrate!("./migrations").run(&pool).await?;

        // Optimize SQLite settings
        pool.execute("PRAGMA journal_mode = WAL").await?;
        pool.execute("PRAGMA synchronous = NORMAL").await?;
        pool.execute("PRAGMA cache_size = -64000").await?; // 64MB cache
        pool.execute("PRAGMA temp_store = MEMORY").await?;
        pool.execute("PRAGMA mmap_size = 268435456").await?; // 256MB mmap

        info!("Database initialized successfully");

        Ok(Self { pool })
    }

    pub fn get_pool(&self) -> &Pool<Sqlite> {
        &self.pool
    }

    // Video operations
    pub async fn insert_video(&self, video: &VideoInfo) -> Result<()> {
        let content_hash = self.calculate_content_hash(video);

        sqlx::query!(
            r#"
            INSERT OR REPLACE INTO videos (
                id, title, description, url, author, platform, upload_date,
                duration_seconds, view_count, like_count, comment_count,
                thumbnail_url, data_source, extraction_method, content_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
            video.id,
            video.title,
            video.description,
            video.url,
            video.author,
            video.platform.as_str(),
            video.upload_date,
            video.duration.map(|d| d as i64),
            video.view_count as i64,
            video.like_count as i64,
            video.comment_count as i64,
            video.thumbnail_url,
            "extractor", // Default data source
            "api", // Default extraction method
            content_hash
        )
        .execute(&self.pool)
        .await?;

        // Insert tags
        self.insert_video_tags(&video.id, &video.tags).await?;

        debug!("Inserted video: {}", video.id);
        Ok(())
    }

    pub async fn get_video(&self, video_id: &str) -> Result<Option<VideoRecord>> {
        let video = sqlx::query_as!(
            VideoRecord,
            r#"
            SELECT id, title, description, url, author, platform, upload_date,
                   duration_seconds, view_count, like_count, comment_count,
                   thumbnail_url, created_at, updated_at, last_checked,
                   data_source, extraction_method, content_hash
            FROM videos WHERE id = ? AND deleted_at IS NULL
            "#,
            video_id
        )
        .fetch_optional(&self.pool)
        .await?;

        Ok(video)
    }

    pub async fn get_videos_by_platform(
        &self,
        platform: Platform,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<VideoRecord>> {
        let videos = sqlx::query_as!(
            VideoRecord,
            r#"
            SELECT id, title, description, url, author, platform, upload_date,
                   duration_seconds, view_count, like_count, comment_count,
                   thumbnail_url, created_at, updated_at, last_checked,
                   data_source, extraction_method, content_hash
            FROM videos
            WHERE platform = ? AND deleted_at IS NULL
            ORDER BY upload_date DESC
            LIMIT ? OFFSET ?
            "#,
            platform.as_str(),
            limit,
            offset
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(videos)
    }

    pub async fn get_recent_videos(&self, hours: i64, limit: i64) -> Result<Vec<VideoRecord>> {
        let videos = sqlx::query_as!(
            VideoRecord,
            r#"
            SELECT id, title, description, url, author, platform, upload_date,
                   duration_seconds, view_count, like_count, comment_count,
                   thumbnail_url, created_at, updated_at, last_checked,
                   data_source, extraction_method, content_hash
            FROM videos
            WHERE created_at > datetime('now', '-' || ? || ' hours') AND deleted_at IS NULL
            ORDER BY created_at DESC
            LIMIT ?
            "#,
            hours,
            limit
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(videos)
    }

    pub async fn search_videos(&self, query: &str, limit: i64) -> Result<Vec<VideoRecord>> {
        let search_query = format!("%{}%", query);

        let videos = sqlx::query_as!(
            VideoRecord,
            r#"
            SELECT id, title, description, url, author, platform, upload_date,
                   duration_seconds, view_count, like_count, comment_count,
                   thumbnail_url, created_at, updated_at, last_checked,
                   data_source, extraction_method, content_hash
            FROM videos
            WHERE (title LIKE ? OR description LIKE ? OR author LIKE ?)
              AND deleted_at IS NULL
            ORDER BY view_count DESC
            LIMIT ?
            "#,
            search_query,
            search_query,
            search_query,
            limit
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(videos)
    }

    pub async fn delete_video(&self, video_id: &str) -> Result<()> {
        sqlx::query!(
            "UPDATE videos SET deleted_at = CURRENT_TIMESTAMP WHERE id = ?",
            video_id
        )
        .execute(&self.pool)
        .await?;

        debug!("Soft deleted video: {}", video_id);
        Ok(())
    }

    // Tag operations
    async fn insert_video_tags(&self, video_id: &str, tags: &[String]) -> Result<()> {
        if tags.is_empty() {
            return Ok(());
        }

        // Insert tags if they don't exist
        for tag in tags {
            sqlx::query!("INSERT OR IGNORE INTO tags (name) VALUES (?)", tag)
                .execute(&self.pool)
                .await?;
        }

        // Get tag IDs
        let tag_ids: Vec<i64> = sqlx::query_scalar!(
            "SELECT id FROM tags WHERE name IN ({})",
            tags.iter().map(|_| "?").collect::<Vec<_>>().join(",")
        )
        .fetch_all(&self.pool)
        .await?;

        // Link video to tags
        for tag_id in tag_ids {
            sqlx::query!(
                "INSERT OR IGNORE INTO video_tags (video_id, tag_id) VALUES (?, ?)",
                video_id,
                tag_id
            )
            .execute(&self.pool)
            .await?;
        }

        Ok(())
    }

    pub async fn get_video_tags(&self, video_id: &str) -> Result<Vec<String>> {
        let tags = sqlx::query_scalar!(
            r#"
            SELECT t.name
            FROM tags t
            JOIN video_tags vt ON t.id = vt.tag_id
            WHERE vt.video_id = ?
            "#,
            video_id
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(tags)
    }

    // Transcription operations
    pub async fn insert_transcription(&self, transcription: &TranscriptionData, video_id: &str) -> Result<()> {
        sqlx::query!(
            r#"
            INSERT OR REPLACE INTO transcriptions (
                video_id, paragraph_summary, sentence_subtitle, full_transcript,
                status, transcriber_model, summarizer_model
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            "#,
            video_id,
            transcription.paragraph_summary,
            transcription.sentence_subtitle,
            transcription.full_transcript,
            transcription.status.to_string(),
            transcription.model_info.transcriber,
            transcription.model_info.summarizer
        )
        .execute(&self.pool)
        .await?;

        debug!("Inserted transcription for video: {}", video_id);
        Ok(())
    }

    pub async fn get_transcription(&self, video_id: &str) -> Result<Option<TranscriptionRecord>> {
        let transcription = sqlx::query_as!(
            TranscriptionRecord,
            r#"
            SELECT id, video_id, paragraph_summary, sentence_subtitle, full_transcript,
                   status, transcriber_model, summarizer_model, processing_time_ms,
                   created_at, updated_at, confidence_score, language_detected
            FROM transcriptions WHERE video_id = ?
            "#,
            video_id
        )
        .fetch_optional(&self.pool)
        .await?;

        Ok(transcription)
    }

    // Platform status operations
    pub async fn update_platform_status(&self, platform: Platform, is_available: bool, response_time_ms: Option<f64>) -> Result<()> {
        if is_available {
            sqlx::query!(
                r#"
                UPDATE platform_status
                SET is_available = ?, last_success = CURRENT_TIMESTAMP,
                    success_count = success_count + 1,
                    avg_response_time_ms = COALESCE(?, avg_response_time_ms)
                WHERE platform = ?
                "#,
                is_available,
                response_time_ms,
                platform.as_str()
            )
            .execute(&self.pool)
            .await?;
        } else {
            sqlx::query!(
                r#"
                UPDATE platform_status
                SET is_available = ?, last_failure = CURRENT_TIMESTAMP,
                    failure_count = failure_count + 1
                WHERE platform = ?
                "#,
                is_available,
                platform.as_str()
            )
            .execute(&self.pool)
            .await?;
        }

        Ok(())
    }

    pub async fn get_platform_status(&self, platform: Platform) -> Result<Option<PlatformStatusRecord>> {
        let status = sqlx::query_as!(
            PlatformStatusRecord,
            r#"
            SELECT platform, is_available, last_success, last_failure,
                   failure_count, success_count, avg_response_time_ms, updated_at
            FROM platform_status WHERE platform = ?
            "#,
            platform.as_str()
        )
        .fetch_optional(&self.pool)
        .await?;

        Ok(status)
    }

    pub async fn get_all_platform_status(&self) -> Result<Vec<PlatformStatusRecord>> {
        let statuses = sqlx::query_as!(
            PlatformStatusRecord,
            r#"
            SELECT platform, is_available, last_success, last_failure,
                   failure_count, success_count, avg_response_time_ms, updated_at
            FROM platform_status
            ORDER BY platform
            "#
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(statuses)
    }

    // Analytics and metrics
    pub async fn get_video_stats(&self) -> Result<Vec<VideoStats>> {
        let stats = sqlx::query!(
            r#"
            SELECT platform,
                   COUNT(*) as total_videos,
                   AVG(view_count) as avg_views,
                   AVG(like_count) as avg_likes,
                   MIN(upload_date) as earliest_video,
                   MAX(upload_date) as latest_video
            FROM active_videos
            GROUP BY platform
            "#
        )
        .fetch_all(&self.pool)
        .await?;

        let video_stats = stats
            .into_iter()
            .map(|row| VideoStats {
                platform: row.platform,
                total_videos: row.total_videos,
                avg_views: row.avg_views.unwrap_or(0.0),
                avg_likes: row.avg_likes.unwrap_or(0.0),
                earliest_video: row.earliest_video,
                latest_video: row.latest_video,
            })
            .collect();

        Ok(video_stats)
    }

    pub async fn record_api_request(&self,
        endpoint: &str,
        method: &str,
        status_code: i32,
        response_time_ms: i64,
        cache_hit: bool,
    ) -> Result<()> {
        sqlx::query!(
            r#"
            INSERT INTO api_requests (endpoint, method, status_code, response_time_ms, cache_hit)
            VALUES (?, ?, ?, ?, ?)
            "#,
            endpoint,
            method,
            status_code,
            response_time_ms,
            cache_hit
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    // Cache operations
    pub async fn cleanup_expired_cache(&self) -> Result<i64> {
        let result = sqlx::query!(
            "DELETE FROM cache_entries WHERE expires_at <= CURRENT_TIMESTAMP"
        )
        .execute(&self.pool)
        .await?;

        let deleted_count = result.rows_affected() as i64;
        if deleted_count > 0 {
            info!("Cleaned up {} expired cache entries", deleted_count);
        }

        Ok(deleted_count)
    }

    pub async fn cleanup_old_data(&self, days: i64) -> Result<HashMap<String, i64>> {
        let mut results = HashMap::new();

        // Clean up old API requests
        let api_result = sqlx::query!(
            "DELETE FROM api_requests WHERE created_at < datetime('now', '-' || ? || ' days')",
            days
        )
        .execute(&self.pool)
        .await?;
        results.insert("api_requests".to_string(), api_result.rows_affected() as i64);

        // Clean up old performance metrics
        let metrics_result = sqlx::query!(
            "DELETE FROM performance_metrics WHERE recorded_at < datetime('now', '-' || ? || ' days')",
            days
        )
        .execute(&self.pool)
        .await?;
        results.insert("performance_metrics".to_string(), metrics_result.rows_affected() as i64);

        Ok(results)
    }

    // Utility methods
    fn calculate_content_hash(&self, video: &VideoInfo) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        video.title.hash(&mut hasher);
        video.url.hash(&mut hasher);
        video.author.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    pub async fn get_database_stats(&self) -> Result<HashMap<String, i64>> {
        let mut stats = HashMap::new();

        // Table row counts
        let tables = ["videos", "tags", "video_tags", "transcriptions", "cache_entries", "rss_feeds"];

        for table in &tables {
            let count: i64 = sqlx::query_scalar(&format!("SELECT COUNT(*) FROM {}", table))
                .fetch_one(&self.pool)
                .await?;
            stats.insert(table.to_string(), count);
        }

        // Database size (SQLite specific)
        let page_count: i64 = sqlx::query_scalar("PRAGMA page_count")
            .fetch_one(&self.pool)
            .await?;
        let page_size: i64 = sqlx::query_scalar("PRAGMA page_size")
            .fetch_one(&self.pool)
            .await?;

        stats.insert("database_size_bytes".to_string(), page_count * page_size);
        stats.insert("page_count".to_string(), page_count);
        stats.insert("page_size".to_string(), page_size);

        Ok(stats)
    }

    pub async fn health_check(&self) -> Result<bool> {
        let result: i64 = sqlx::query_scalar("SELECT 1").fetch_one(&self.pool).await?;
        Ok(result == 1)
    }
}

impl TranscriptionStatus {
    fn to_string(&self) -> String {
        match self {
            TranscriptionStatus::Success => "success".to_string(),
            TranscriptionStatus::Pending => "pending".to_string(),
            TranscriptionStatus::Failed => "failed".to_string(),
            TranscriptionStatus::Unavailable => "unavailable".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_database_initialization() {
        let temp_file = NamedTempFile::new().unwrap();
        let db_url = format!("sqlite:{}", temp_file.path().to_str().unwrap());

        let config = DatabaseConfig {
            url: db_url,
            ..Default::default()
        };

        let db = Database::new(config).await.unwrap();
        assert!(db.health_check().await.unwrap());
    }

    #[tokio::test]
    async fn test_video_operations() {
        let temp_file = NamedTempFile::new().unwrap();
        let db_url = format!("sqlite:{}", temp_file.path().to_str().unwrap());

        let config = DatabaseConfig {
            url: db_url,
            ..Default::default()
        };

        let db = Database::new(config).await.unwrap();

        let video = VideoInfo {
            id: "test123".to_string(),
            title: "Test Video".to_string(),
            description: "Test Description".to_string(),
            url: "https://example.com/test".to_string(),
            author: "Test Author".to_string(),
            upload_date: Utc::now(),
            duration: Some(300),
            view_count: 1000,
            like_count: 100,
            comment_count: 10,
            tags: vec!["test".to_string(), "video".to_string()],
            thumbnail_url: Some("https://example.com/thumb.jpg".to_string()),
            platform: Platform::Bilibili,
            transcription: None,
        };

        // Insert video
        db.insert_video(&video).await.unwrap();

        // Get video
        let retrieved = db.get_video("test123").await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().title, "Test Video");

        // Get tags
        let tags = db.get_video_tags("test123").await.unwrap();
        assert_eq!(tags.len(), 2);
        assert!(tags.contains(&"test".to_string()));
    }
}