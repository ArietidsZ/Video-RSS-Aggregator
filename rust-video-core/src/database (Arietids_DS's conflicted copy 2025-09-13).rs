use anyhow::{Result, Context};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{Pool, Sqlite, Row};
use std::path::Path;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Video {
    pub id: String,
    pub platform: String,
    pub title: String,
    pub description: Option<String>,
    pub url: String,
    pub thumbnail_url: Option<String>,
    pub author: String,
    pub duration: i64,
    pub views: i64,
    pub likes: i64,
    pub comments: i64,
    pub tags: Vec<String>,
    pub upload_date: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoAnalysis {
    pub id: String,
    pub video_id: String,
    pub summary: String,
    pub transcript: Option<String>,
    pub language: Option<String>,
    pub scene_count: i32,
    pub detected_objects: i32,
    pub dominant_colors: Vec<String>,
    pub motion_intensity: f32,
    pub processing_time_ms: i64,
    pub gpu_backend: String,
    pub analyzed_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Feed {
    pub id: String,
    pub name: String,
    pub description: String,
    pub platform: Option<String>,
    pub url_pattern: Option<String>,
    pub last_updated: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
}

#[derive(Clone)]
pub struct Database {
    pool: Pool<Sqlite>,
}

impl Database {
    pub async fn new(database_url: &str) -> Result<Self> {
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(10)
            .connect(database_url)
            .await
            .context("Failed to connect to database")?;

        // Run migrations
        sqlx::migrate!("./migrations")
            .run(&pool)
            .await
            .context("Failed to run database migrations")?;

        Ok(Database { pool })
    }

    pub async fn create_if_not_exists(database_path: &Path) -> Result<Self> {
        let database_url = format!("sqlite://{}", database_path.display());

        // Create directory if it doesn't exist
        if let Some(parent) = database_path.parent() {
            std::fs::create_dir_all(parent)
                .context("Failed to create database directory")?;
        }

        Self::new(&database_url).await
    }

    // Video operations
    pub async fn save_video(&self, video: &Video) -> Result<()> {
        let tags_json = serde_json::to_string(&video.tags)?;

        sqlx::query!(
            r#"
            INSERT OR REPLACE INTO videos (
                id, platform, title, description, url, thumbnail_url, author,
                duration, views, likes, comments, tags, upload_date, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
            video.id,
            video.platform,
            video.title,
            video.description,
            video.url,
            video.thumbnail_url,
            video.author,
            video.duration,
            video.views,
            video.likes,
            video.comments,
            tags_json,
            video.upload_date,
            video.created_at,
            video.updated_at
        )
        .execute(&self.pool)
        .await
        .context("Failed to save video")?;

        Ok(())
    }

    pub async fn get_video(&self, id: &str) -> Result<Option<Video>> {
        let row = sqlx::query!(
            "SELECT * FROM videos WHERE id = ?",
            id
        )
        .fetch_optional(&self.pool)
        .await
        .context("Failed to get video")?;

        if let Some(row) = row {
            let tags: Vec<String> = serde_json::from_str(&row.tags)?;

            Ok(Some(Video {
                id: row.id,
                platform: row.platform,
                title: row.title,
                description: row.description,
                url: row.url,
                thumbnail_url: row.thumbnail_url,
                author: row.author,
                duration: row.duration,
                views: row.views,
                likes: row.likes,
                comments: row.comments,
                tags,
                upload_date: DateTime::parse_from_rfc3339(&row.upload_date)?.with_timezone(&Utc),
                created_at: DateTime::parse_from_rfc3339(&row.created_at)?.with_timezone(&Utc),
                updated_at: DateTime::parse_from_rfc3339(&row.updated_at)?.with_timezone(&Utc),
            }))
        } else {
            Ok(None)
        }
    }

    pub async fn get_videos_by_platform(&self, platform: &str, limit: i64) -> Result<Vec<Video>> {
        let rows = sqlx::query!(
            "SELECT * FROM videos WHERE platform = ? ORDER BY upload_date DESC LIMIT ?",
            platform, limit
        )
        .fetch_all(&self.pool)
        .await
        .context("Failed to get videos by platform")?;

        let mut videos = Vec::new();
        for row in rows {
            let tags: Vec<String> = serde_json::from_str(&row.tags)?;

            videos.push(Video {
                id: row.id,
                platform: row.platform,
                title: row.title,
                description: row.description,
                url: row.url,
                thumbnail_url: row.thumbnail_url,
                author: row.author,
                duration: row.duration,
                views: row.views,
                likes: row.likes,
                comments: row.comments,
                tags,
                upload_date: DateTime::parse_from_rfc3339(&row.upload_date)?.with_timezone(&Utc),
                created_at: DateTime::parse_from_rfc3339(&row.created_at)?.with_timezone(&Utc),
                updated_at: DateTime::parse_from_rfc3339(&row.updated_at)?.with_timezone(&Utc),
            });
        }

        Ok(videos)
    }

    pub async fn get_recent_videos(&self, limit: i64) -> Result<Vec<Video>> {
        let rows = sqlx::query!(
            "SELECT * FROM videos ORDER BY upload_date DESC LIMIT ?",
            limit
        )
        .fetch_all(&self.pool)
        .await
        .context("Failed to get recent videos")?;

        let mut videos = Vec::new();
        for row in rows {
            let tags: Vec<String> = serde_json::from_str(&row.tags)?;

            videos.push(Video {
                id: row.id,
                platform: row.platform,
                title: row.title,
                description: row.description,
                url: row.url,
                thumbnail_url: row.thumbnail_url,
                author: row.author,
                duration: row.duration,
                views: row.views,
                likes: row.likes,
                comments: row.comments,
                tags,
                upload_date: DateTime::parse_from_rfc3339(&row.upload_date)?.with_timezone(&Utc),
                created_at: DateTime::parse_from_rfc3339(&row.created_at)?.with_timezone(&Utc),
                updated_at: DateTime::parse_from_rfc3339(&row.updated_at)?.with_timezone(&Utc),
            });
        }

        Ok(videos)
    }

    // Analysis operations
    pub async fn save_analysis(&self, analysis: &VideoAnalysis) -> Result<()> {
        let colors_json = serde_json::to_string(&analysis.dominant_colors)?;

        sqlx::query!(
            r#"
            INSERT OR REPLACE INTO video_analyses (
                id, video_id, summary, transcript, language, scene_count,
                detected_objects, dominant_colors, motion_intensity,
                processing_time_ms, gpu_backend, analyzed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
            analysis.id,
            analysis.video_id,
            analysis.summary,
            analysis.transcript,
            analysis.language,
            analysis.scene_count,
            analysis.detected_objects,
            colors_json,
            analysis.motion_intensity,
            analysis.processing_time_ms,
            analysis.gpu_backend,
            analysis.analyzed_at
        )
        .execute(&self.pool)
        .await
        .context("Failed to save analysis")?;

        Ok(())
    }

    pub async fn get_analysis(&self, video_id: &str) -> Result<Option<VideoAnalysis>> {
        let row = sqlx::query!(
            "SELECT * FROM video_analyses WHERE video_id = ?",
            video_id
        )
        .fetch_optional(&self.pool)
        .await
        .context("Failed to get analysis")?;

        if let Some(row) = row {
            let dominant_colors: Vec<String> = serde_json::from_str(&row.dominant_colors)?;

            Ok(Some(VideoAnalysis {
                id: row.id,
                video_id: row.video_id,
                summary: row.summary,
                transcript: row.transcript,
                language: row.language,
                scene_count: row.scene_count,
                detected_objects: row.detected_objects,
                dominant_colors,
                motion_intensity: row.motion_intensity,
                processing_time_ms: row.processing_time_ms,
                gpu_backend: row.gpu_backend,
                analyzed_at: DateTime::parse_from_rfc3339(&row.analyzed_at)?.with_timezone(&Utc),
            }))
        } else {
            Ok(None)
        }
    }

    pub async fn get_videos_with_analysis(&self, limit: i64) -> Result<Vec<(Video, VideoAnalysis)>> {
        let rows = sqlx::query!(
            r#"
            SELECT
                v.*,
                a.id as analysis_id,
                a.summary,
                a.transcript,
                a.language,
                a.scene_count,
                a.detected_objects,
                a.dominant_colors,
                a.motion_intensity,
                a.processing_time_ms,
                a.gpu_backend,
                a.analyzed_at
            FROM videos v
            INNER JOIN video_analyses a ON v.id = a.video_id
            ORDER BY v.upload_date DESC
            LIMIT ?
            "#,
            limit
        )
        .fetch_all(&self.pool)
        .await
        .context("Failed to get videos with analysis")?;

        let mut results = Vec::new();
        for row in rows {
            let tags: Vec<String> = serde_json::from_str(&row.tags)?;
            let dominant_colors: Vec<String> = serde_json::from_str(&row.dominant_colors)?;

            let video = Video {
                id: row.id.clone(),
                platform: row.platform,
                title: row.title,
                description: row.description,
                url: row.url,
                thumbnail_url: row.thumbnail_url,
                author: row.author,
                duration: row.duration,
                views: row.views,
                likes: row.likes,
                comments: row.comments,
                tags,
                upload_date: DateTime::parse_from_rfc3339(&row.upload_date)?.with_timezone(&Utc),
                created_at: DateTime::parse_from_rfc3339(&row.created_at)?.with_timezone(&Utc),
                updated_at: DateTime::parse_from_rfc3339(&row.updated_at)?.with_timezone(&Utc),
            };

            let analysis = VideoAnalysis {
                id: row.analysis_id,
                video_id: row.id,
                summary: row.summary,
                transcript: row.transcript,
                language: row.language,
                scene_count: row.scene_count,
                detected_objects: row.detected_objects,
                dominant_colors,
                motion_intensity: row.motion_intensity,
                processing_time_ms: row.processing_time_ms,
                gpu_backend: row.gpu_backend,
                analyzed_at: DateTime::parse_from_rfc3339(&row.analyzed_at)?.with_timezone(&Utc),
            };

            results.push((video, analysis));
        }

        Ok(results)
    }

    // Feed operations
    pub async fn save_feed(&self, feed: &Feed) -> Result<()> {
        sqlx::query!(
            r#"
            INSERT OR REPLACE INTO feeds (
                id, name, description, platform, url_pattern, last_updated, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            "#,
            feed.id,
            feed.name,
            feed.description,
            feed.platform,
            feed.url_pattern,
            feed.last_updated,
            feed.created_at
        )
        .execute(&self.pool)
        .await
        .context("Failed to save feed")?;

        Ok(())
    }

    pub async fn get_feed(&self, id: &str) -> Result<Option<Feed>> {
        let row = sqlx::query!(
            "SELECT * FROM feeds WHERE id = ?",
            id
        )
        .fetch_optional(&self.pool)
        .await
        .context("Failed to get feed")?;

        if let Some(row) = row {
            Ok(Some(Feed {
                id: row.id,
                name: row.name,
                description: row.description,
                platform: row.platform,
                url_pattern: row.url_pattern,
                last_updated: DateTime::parse_from_rfc3339(&row.last_updated)?.with_timezone(&Utc),
                created_at: DateTime::parse_from_rfc3339(&row.created_at)?.with_timezone(&Utc),
            }))
        } else {
            Ok(None)
        }
    }

    pub async fn get_all_feeds(&self) -> Result<Vec<Feed>> {
        let rows = sqlx::query!("SELECT * FROM feeds ORDER BY created_at DESC")
            .fetch_all(&self.pool)
            .await
            .context("Failed to get all feeds")?;

        let mut feeds = Vec::new();
        for row in rows {
            feeds.push(Feed {
                id: row.id,
                name: row.name,
                description: row.description,
                platform: row.platform,
                url_pattern: row.url_pattern,
                last_updated: DateTime::parse_from_rfc3339(&row.last_updated)?.with_timezone(&Utc),
                created_at: DateTime::parse_from_rfc3339(&row.created_at)?.with_timezone(&Utc),
            });
        }

        Ok(feeds)
    }

    // Utility functions
    pub async fn cleanup_old_videos(&self, days: i64) -> Result<u64> {
        let result = sqlx::query!(
            "DELETE FROM videos WHERE created_at < datetime('now', '-' || ? || ' days')",
            days
        )
        .execute(&self.pool)
        .await
        .context("Failed to cleanup old videos")?;

        Ok(result.rows_affected())
    }

    pub async fn get_stats(&self) -> Result<DatabaseStats> {
        let video_count: i64 = sqlx::query_scalar!("SELECT COUNT(*) FROM videos")
            .fetch_one(&self.pool)
            .await
            .context("Failed to get video count")?;

        let analysis_count: i64 = sqlx::query_scalar!("SELECT COUNT(*) FROM video_analyses")
            .fetch_one(&self.pool)
            .await
            .context("Failed to get analysis count")?;

        let feed_count: i64 = sqlx::query_scalar!("SELECT COUNT(*) FROM feeds")
            .fetch_one(&self.pool)
            .await
            .context("Failed to get feed count")?;

        Ok(DatabaseStats {
            total_videos: video_count,
            total_analyses: analysis_count,
            total_feeds: feed_count,
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DatabaseStats {
    pub total_videos: i64,
    pub total_analyses: i64,
    pub total_feeds: i64,
}

// Helper function to generate UUIDs
pub fn generate_id() -> String {
    Uuid::new_v4().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[tokio::test]
    async fn test_database_operations() {
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let db = Database::create_if_not_exists(&db_path).await.unwrap();

        // Test video operations
        let video = Video {
            id: generate_id(),
            platform: "youtube".to_string(),
            title: "Test Video".to_string(),
            description: Some("Test description".to_string()),
            url: "https://youtube.com/watch?v=test".to_string(),
            thumbnail_url: Some("https://youtube.com/thumb.jpg".to_string()),
            author: "Test Author".to_string(),
            duration: 300,
            views: 1000,
            likes: 100,
            comments: 10,
            tags: vec!["test".to_string(), "video".to_string()],
            upload_date: Utc::now(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        db.save_video(&video).await.unwrap();
        let retrieved = db.get_video(&video.id).await.unwrap().unwrap();
        assert_eq!(retrieved.title, video.title);

        // Test analysis operations
        let analysis = VideoAnalysis {
            id: generate_id(),
            video_id: video.id.clone(),
            summary: "Test summary".to_string(),
            transcript: Some("Test transcript".to_string()),
            language: Some("en".to_string()),
            scene_count: 5,
            detected_objects: 10,
            dominant_colors: vec!["#ff0000".to_string(), "#00ff00".to_string()],
            motion_intensity: 0.7,
            processing_time_ms: 5000,
            gpu_backend: "Metal".to_string(),
            analyzed_at: Utc::now(),
        };

        db.save_analysis(&analysis).await.unwrap();
        let retrieved_analysis = db.get_analysis(&video.id).await.unwrap().unwrap();
        assert_eq!(retrieved_analysis.summary, analysis.summary);

        // Test stats
        let stats = db.get_stats().await.unwrap();
        assert_eq!(stats.total_videos, 1);
        assert_eq!(stats.total_analyses, 1);
    }
}