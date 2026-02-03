use anyhow::Result;
use chrono::{DateTime, Utc};
use sqlx::{postgres::PgPoolOptions, PgPool, Row};
use uuid::Uuid;

use crate::summarize::SummaryResult;
use crate::transcribe::TranscriptionResult;

#[derive(Clone)]
pub struct Database {
    pool: PgPool,
}

#[derive(Clone, Debug)]
pub struct SummaryRecord {
    pub title: Option<String>,
    pub source_url: String,
    pub published_at: Option<DateTime<Utc>>,
    pub summary: String,
    pub key_points: Vec<String>,
    pub created_at: DateTime<Utc>,
}

impl Database {
    pub async fn connect(database_url: &str) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(10)
            .connect(database_url)
            .await?;

        Ok(Self { pool })
    }

    pub async fn migrate(&self) -> Result<()> {
        sqlx::query(
            "\
            CREATE TABLE IF NOT EXISTS feeds (
                id UUID PRIMARY KEY,
                url TEXT NOT NULL UNIQUE,
                title TEXT,
                last_checked TIMESTAMPTZ
            );
            ",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "\
            CREATE TABLE IF NOT EXISTS videos (
                id UUID PRIMARY KEY,
                feed_id UUID REFERENCES feeds(id),
                source_url TEXT NOT NULL UNIQUE,
                guid TEXT,
                title TEXT,
                published_at TIMESTAMPTZ
            );
            ",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "\
            CREATE TABLE IF NOT EXISTS transcripts (
                id UUID PRIMARY KEY,
                video_id UUID NOT NULL REFERENCES videos(id),
                language TEXT,
                confidence REAL,
                text TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            ",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "\
            CREATE TABLE IF NOT EXISTS summaries (
                id UUID PRIMARY KEY,
                video_id UUID NOT NULL REFERENCES videos(id),
                summary TEXT NOT NULL,
                key_points JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            ",
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn upsert_feed(&self, url: &str, title: Option<&str>) -> Result<Uuid> {
        let id = Uuid::new_v4();
        let row = sqlx::query(
            "\
            INSERT INTO feeds (id, url, title, last_checked)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (url)
            DO UPDATE SET title = COALESCE(EXCLUDED.title, feeds.title), last_checked = NOW()
            RETURNING id;
            ",
        )
        .bind(id)
        .bind(url)
        .bind(title)
        .fetch_one(&self.pool)
        .await?;

        Ok(row.try_get("id")?)
    }

    pub async fn upsert_video(
        &self,
        feed_id: Option<Uuid>,
        guid: Option<&str>,
        title: Option<&str>,
        source_url: &str,
        published_at: Option<DateTime<Utc>>,
    ) -> Result<Uuid> {
        let id = Uuid::new_v4();
        let row = sqlx::query(
            "\
            INSERT INTO videos (id, feed_id, source_url, guid, title, published_at)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (source_url)
            DO UPDATE SET
                title = COALESCE(EXCLUDED.title, videos.title),
                published_at = COALESCE(EXCLUDED.published_at, videos.published_at),
                guid = COALESCE(EXCLUDED.guid, videos.guid),
                feed_id = COALESCE(EXCLUDED.feed_id, videos.feed_id)
            RETURNING id;
            ",
        )
        .bind(id)
        .bind(feed_id)
        .bind(source_url)
        .bind(guid)
        .bind(title)
        .bind(published_at)
        .fetch_one(&self.pool)
        .await?;

        Ok(row.try_get("id")?)
    }

    pub async fn insert_transcript(
        &self,
        video_id: Uuid,
        transcript: &TranscriptionResult,
    ) -> Result<Uuid> {
        let id = Uuid::new_v4();
        sqlx::query(
            "\
            INSERT INTO transcripts (id, video_id, language, confidence, text)
            VALUES ($1, $2, $3, $4, $5);
            ",
        )
        .bind(id)
        .bind(video_id)
        .bind(&transcript.language)
        .bind(transcript.confidence)
        .bind(&transcript.text)
        .execute(&self.pool)
        .await?;

        Ok(id)
    }

    pub async fn insert_summary(&self, video_id: Uuid, summary: &SummaryResult) -> Result<Uuid> {
        let id = Uuid::new_v4();
        let key_points = serde_json::to_value(&summary.key_points)?;
        sqlx::query(
            "\
            INSERT INTO summaries (id, video_id, summary, key_points)
            VALUES ($1, $2, $3, $4);
            ",
        )
        .bind(id)
        .bind(video_id)
        .bind(&summary.summary)
        .bind(key_points)
        .execute(&self.pool)
        .await?;

        Ok(id)
    }

    pub async fn latest_summaries(&self, limit: i64) -> Result<Vec<SummaryRecord>> {
        let rows = sqlx::query(
            "\
            SELECT v.title, v.source_url, v.published_at, s.summary, s.key_points, s.created_at
            FROM summaries s
            JOIN videos v ON v.id = s.video_id
            ORDER BY s.created_at DESC
            LIMIT $1;
            ",
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        let mut items = Vec::with_capacity(rows.len());
        for row in rows {
            let key_points_value: serde_json::Value = row.try_get("key_points")?;
            let key_points = key_points_value
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|item| item.as_str().map(|value| value.to_string()))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();

            items.push(SummaryRecord {
                title: row.try_get("title")?,
                source_url: row.try_get("source_url")?,
                published_at: row.try_get("published_at")?,
                summary: row.try_get("summary")?,
                key_points,
                created_at: row.try_get("created_at")?,
            });
        }

        Ok(items)
    }
}
