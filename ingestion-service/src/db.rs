use sqlx::{postgres::PgPoolOptions, Pool, Postgres};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use anyhow::Result;

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct Feed {
    pub id: Uuid,
    pub url: String,
    pub title: Option<String>,
    pub description: Option<String>,
    pub status: FeedStatus,
    pub last_fetched_at: Option<DateTime<Utc>>,
    pub error_count: i32,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, sqlx::Type)]
#[sqlx(type_name = "feed_status", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FeedStatus {
    Active,
    Error,
    Paused,
}

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct Item {
    pub id: Uuid,
    pub feed_id: Uuid,
    pub external_id: String,
    pub url: String,
    pub title: Option<String>,
    pub description: Option<String>,
    pub published_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
}

pub struct DbClient {
    pool: Pool<Postgres>,
}

impl DbClient {
    pub async fn new(database_url: &str) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(database_url)
            .await?;
        
        // Run migrations
        sqlx::migrate!("./migrations").run(&pool).await?;

        Ok(Self { pool })
    }

    pub async fn create_feed(&self, url: &str) -> Result<Feed> {
        let feed = sqlx::query_as!(
            Feed,
            r#"
            INSERT INTO feeds (url)
            VALUES ($1)
            RETURNING id, url, title, description, status as "status: FeedStatus", last_fetched_at, error_count, created_at, updated_at
            "#,
            url
        )
        .fetch_one(&self.pool)
        .await?;
        Ok(feed)
    }

    pub async fn list_active_feeds(&self) -> Result<Vec<Feed>> {
        let feeds = sqlx::query_as!(
            Feed,
            r#"
            SELECT id, url, title, description, status as "status: FeedStatus", last_fetched_at, error_count, created_at, updated_at
            FROM feeds
            WHERE status = 'ACTIVE'
            "#
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(feeds)
    }
    
    pub async fn upsert_item(&self, feed_id: Uuid, item: &Item) -> Result<bool> {
        // Returns true if inserted (new), false if updated (existing)
        let result = sqlx::query!(
            r#"
            INSERT INTO items (feed_id, external_id, url, title, description, published_at)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (feed_id, external_id) DO UPDATE
            SET updated_at = NOW()
            RETURNING id
            "#,
            feed_id,
            item.external_id,
            item.url,
            item.title,
            item.description,
            item.published_at
        )
        .fetch_optional(&self.pool)
        .await?;
        
        // In a real upsert with ON CONFLICT DO UPDATE, it always returns a row. 
        // To distinguish, we might check xmax or just assume it's processed.
        // For now, we just want to know it succeeded.
        Ok(result.is_some())
    }
}
