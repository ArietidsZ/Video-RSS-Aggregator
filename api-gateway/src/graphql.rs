use async_graphql::{
    dataloader::{DataLoader, Loader},
    extensions::Tracing,
    http::{playground_source, GraphQLPlaygroundConfig},
    *,
};
use async_graphql_axum::{GraphQLRequest, GraphQLResponse, GraphQLSubscription};
use axum::{
    extract::State,
    response::{Html, IntoResponse},
    routing::get,
    Router,
};
use chrono::{DateTime, Utc};
use futures::stream::{self, StreamExt};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
use tracing::{debug, error, info};
use uuid::Uuid;

// GraphQL Schema Types

#[derive(Debug, Clone, Serialize, Deserialize, SimpleObject)]
pub struct Video {
    pub id: ID,
    pub title: String,
    pub description: String,
    pub url: String,
    pub thumbnail_url: Option<String>,
    pub duration_seconds: i32,
    pub channel_id: ID,
    pub quality_score: f32,
    pub view_count: i64,
    pub like_count: i64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,

    #[graphql(skip)]
    pub _channel: Option<Channel>,
}

#[derive(Debug, Clone, Serialize, Deserialize, SimpleObject)]
pub struct Channel {
    pub id: ID,
    pub name: String,
    pub description: String,
    pub url: String,
    pub subscriber_count: i64,
    pub video_count: i64,
    pub created_at: DateTime<Utc>,

    #[graphql(skip)]
    pub _videos: Vec<Video>,
}

#[derive(Debug, Clone, Serialize, Deserialize, SimpleObject)]
pub struct User {
    pub id: ID,
    pub username: String,
    pub email: String,
    pub role: UserRole,
    pub preferences: UserPreferences,
    pub created_at: DateTime<Utc>,
    pub last_login: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Enum)]
pub enum UserRole {
    Admin,
    User,
    Premium,
    Guest,
}

#[derive(Debug, Clone, Serialize, Deserialize, SimpleObject)]
pub struct UserPreferences {
    pub categories: Vec<String>,
    pub languages: Vec<String>,
    pub min_quality_score: f32,
    pub max_duration_seconds: i32,
    pub notifications_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, SimpleObject)]
pub struct Feed {
    pub id: ID,
    pub url: String,
    pub title: String,
    pub description: String,
    pub category: String,
    pub update_frequency_minutes: i32,
    pub last_fetched: Option<DateTime<Utc>>,
    pub active: bool,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, SimpleObject)]
pub struct Summary {
    pub id: ID,
    pub video_id: ID,
    pub content: String,
    pub key_points: Vec<String>,
    pub sentiment_score: f32,
    pub language: String,
    pub word_count: i32,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, SimpleObject)]
pub struct Recommendation {
    pub id: ID,
    pub user_id: ID,
    pub video_id: ID,
    pub score: f32,
    pub reason: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, SimpleObject)]
pub struct Analytics {
    pub total_videos: i64,
    pub total_channels: i64,
    pub total_users: i64,
    pub total_summaries: i64,
    pub avg_quality_score: f32,
    pub processing_rate: f32,
    pub cache_hit_rate: f32,
    pub error_rate: f32,
}

// Input types

#[derive(Debug, InputObject)]
pub struct VideoFilter {
    pub channel_id: Option<ID>,
    pub min_quality_score: Option<f32>,
    pub max_duration_seconds: Option<i32>,
    pub category: Option<String>,
    pub language: Option<String>,
    pub search: Option<String>,
}

#[derive(Debug, InputObject)]
pub struct CreateVideoInput {
    pub title: String,
    pub description: String,
    pub url: String,
    pub thumbnail_url: Option<String>,
    pub duration_seconds: i32,
    pub channel_id: ID,
}

#[derive(Debug, InputObject)]
pub struct UpdateVideoInput {
    pub id: ID,
    pub title: Option<String>,
    pub description: Option<String>,
    pub quality_score: Option<f32>,
}

#[derive(Debug, InputObject)]
pub struct CreateUserInput {
    pub username: String,
    pub email: String,
    pub password: String,
    pub role: Option<UserRole>,
}

#[derive(Debug, InputObject)]
pub struct UpdateUserPreferencesInput {
    pub categories: Option<Vec<String>>,
    pub languages: Option<Vec<String>>,
    pub min_quality_score: Option<f32>,
    pub max_duration_seconds: Option<i32>,
    pub notifications_enabled: Option<bool>,
}

#[derive(Debug, InputObject)]
pub struct PaginationInput {
    pub limit: Option<i32>,
    pub offset: Option<i32>,
    pub cursor: Option<String>,
}

// Connection types for pagination

#[derive(Debug, Clone, SimpleObject)]
pub struct VideoConnection {
    pub edges: Vec<VideoEdge>,
    pub page_info: PageInfo,
    pub total_count: i32,
}

#[derive(Debug, Clone, SimpleObject)]
pub struct VideoEdge {
    pub node: Video,
    pub cursor: String,
}

#[derive(Debug, Clone, SimpleObject)]
pub struct PageInfo {
    pub has_next_page: bool,
    pub has_previous_page: bool,
    pub start_cursor: Option<String>,
    pub end_cursor: Option<String>,
}

// DataLoader for N+1 query optimization

pub struct ChannelLoader {
    pool: PgPool,
}

#[async_trait::async_trait]
impl Loader<ID> for ChannelLoader {
    type Value = Channel;
    type Error = async_graphql::Error;

    async fn load(&self, keys: &[ID]) -> Result<HashMap<ID, Self::Value>, Self::Error> {
        let ids: Vec<String> = keys.iter().map(|k| k.to_string()).collect();

        // Fetch channels from database
        let channels = sqlx::query_as!(
            Channel,
            r#"
            SELECT id as "id: ID", name, description, url,
                   subscriber_count, video_count, created_at
            FROM channels
            WHERE id = ANY($1)
            "#,
            &ids[..]
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| async_graphql::Error::new(e.to_string()))?;

        let mut map = HashMap::new();
        for channel in channels {
            map.insert(channel.id.clone(), channel);
        }

        Ok(map)
    }
}

// GraphQL Query Root

pub struct QueryRoot {
    pool: PgPool,
}

#[Object]
impl QueryRoot {
    // Video queries
    async fn video(&self, ctx: &Context<'_>, id: ID) -> Result<Option<Video>> {
        let pool = ctx.data::<PgPool>()?;

        let video = sqlx::query_as!(
            Video,
            r#"
            SELECT id as "id: ID", title, description, url, thumbnail_url,
                   duration_seconds, channel_id as "channel_id: ID",
                   quality_score, view_count, like_count,
                   created_at, updated_at
            FROM videos
            WHERE id = $1
            "#,
            id.to_string()
        )
        .fetch_optional(pool)
        .await?;

        Ok(video)
    }

    async fn videos(
        &self,
        ctx: &Context<'_>,
        filter: Option<VideoFilter>,
        pagination: Option<PaginationInput>,
    ) -> Result<VideoConnection> {
        let pool = ctx.data::<PgPool>()?;
        let limit = pagination.as_ref().and_then(|p| p.limit).unwrap_or(20);
        let offset = pagination.as_ref().and_then(|p| p.offset).unwrap_or(0);

        // Build dynamic query based on filters
        let mut query = String::from(
            r#"
            SELECT id, title, description, url, thumbnail_url,
                   duration_seconds, channel_id, quality_score,
                   view_count, like_count, created_at, updated_at
            FROM videos
            WHERE 1=1
            "#,
        );

        let mut params = Vec::new();

        if let Some(filter) = filter {
            if let Some(channel_id) = filter.channel_id {
                query.push_str(&format!(" AND channel_id = ${}", params.len() + 1));
                params.push(channel_id.to_string());
            }

            if let Some(min_score) = filter.min_quality_score {
                query.push_str(&format!(" AND quality_score >= ${}", params.len() + 1));
                params.push(min_score.to_string());
            }

            if let Some(search) = filter.search {
                query.push_str(&format!(
                    " AND (title ILIKE ${}  OR description ILIKE ${})",
                    params.len() + 1,
                    params.len() + 2
                ));
                let search_pattern = format!("%{}%", search);
                params.push(search_pattern.clone());
                params.push(search_pattern);
            }
        }

        query.push_str(&format!(" ORDER BY created_at DESC LIMIT {} OFFSET {}", limit, offset));

        // Execute query (simplified - in production use proper query builder)
        let videos: Vec<Video> = Vec::new(); // Would execute actual query

        // Get total count
        let total_count = 100; // Would get actual count

        // Build connection response
        let edges = videos
            .into_iter()
            .map(|video| VideoEdge {
                cursor: base64::encode(&video.id.to_string()),
                node: video,
            })
            .collect();

        let page_info = PageInfo {
            has_next_page: offset + limit < total_count,
            has_previous_page: offset > 0,
            start_cursor: edges.first().map(|e| e.cursor.clone()),
            end_cursor: edges.last().map(|e| e.cursor.clone()),
        };

        Ok(VideoConnection {
            edges,
            page_info,
            total_count,
        })
    }

    // Channel queries
    async fn channel(&self, ctx: &Context<'_>, id: ID) -> Result<Option<Channel>> {
        let loader = ctx.data::<DataLoader<ChannelLoader>>()?;
        Ok(loader.load_one(id).await?)
    }

    async fn channels(&self, ctx: &Context<'_>) -> Result<Vec<Channel>> {
        let pool = ctx.data::<PgPool>()?;

        let channels = sqlx::query_as!(
            Channel,
            r#"
            SELECT id as "id: ID", name, description, url,
                   subscriber_count, video_count, created_at
            FROM channels
            ORDER BY subscriber_count DESC
            "#
        )
        .fetch_all(pool)
        .await?;

        Ok(channels)
    }

    // User queries
    async fn me(&self, ctx: &Context<'_>) -> Result<Option<User>> {
        // Get current user from context (set by authentication middleware)
        let user_id = ctx.data::<Option<String>>()?;

        if let Some(id) = user_id {
            let pool = ctx.data::<PgPool>()?;

            let user = sqlx::query_as!(
                User,
                r#"
                SELECT id as "id: ID", username, email,
                       role as "role: UserRole",
                       preferences as "preferences: UserPreferences",
                       created_at, last_login
                FROM users
                WHERE id = $1
                "#,
                id
            )
            .fetch_optional(pool)
            .await?;

            Ok(user)
        } else {
            Ok(None)
        }
    }

    // Analytics queries
    async fn analytics(&self, ctx: &Context<'_>) -> Result<Analytics> {
        let pool = ctx.data::<PgPool>()?;

        // Gather analytics data (simplified)
        let analytics = Analytics {
            total_videos: 1000,
            total_channels: 50,
            total_users: 10000,
            total_summaries: 800,
            avg_quality_score: 0.75,
            processing_rate: 0.95,
            cache_hit_rate: 0.85,
            error_rate: 0.02,
        };

        Ok(analytics)
    }

    // Recommendation queries
    async fn recommendations(
        &self,
        ctx: &Context<'_>,
        user_id: ID,
        limit: Option<i32>,
    ) -> Result<Vec<Recommendation>> {
        let pool = ctx.data::<PgPool>()?;
        let limit = limit.unwrap_or(10);

        let recommendations = sqlx::query_as!(
            Recommendation,
            r#"
            SELECT id as "id: ID", user_id as "user_id: ID",
                   video_id as "video_id: ID", score, reason, created_at
            FROM recommendations
            WHERE user_id = $1
            ORDER BY score DESC
            LIMIT $2
            "#,
            user_id.to_string(),
            limit as i64
        )
        .fetch_all(pool)
        .await?;

        Ok(recommendations)
    }
}

// GraphQL Mutation Root

pub struct MutationRoot {
    pool: PgPool,
    event_sender: broadcast::Sender<SubscriptionEvent>,
}

#[Object]
impl MutationRoot {
    // Video mutations
    async fn create_video(&self, ctx: &Context<'_>, input: CreateVideoInput) -> Result<Video> {
        let pool = ctx.data::<PgPool>()?;

        let video_id = Uuid::new_v4().to_string();
        let now = Utc::now();

        let video = sqlx::query_as!(
            Video,
            r#"
            INSERT INTO videos (id, title, description, url, thumbnail_url,
                              duration_seconds, channel_id, quality_score,
                              view_count, like_count, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, 0.5, 0, 0, $8, $9)
            RETURNING id as "id: ID", title, description, url, thumbnail_url,
                      duration_seconds, channel_id as "channel_id: ID",
                      quality_score, view_count, like_count,
                      created_at, updated_at
            "#,
            video_id,
            input.title,
            input.description,
            input.url,
            input.thumbnail_url,
            input.duration_seconds,
            input.channel_id.to_string(),
            now,
            now
        )
        .fetch_one(pool)
        .await?;

        // Send subscription event
        let _ = self.event_sender.send(SubscriptionEvent::VideoCreated(video.clone()));

        Ok(video)
    }

    async fn update_video(&self, ctx: &Context<'_>, input: UpdateVideoInput) -> Result<Video> {
        let pool = ctx.data::<PgPool>()?;

        // Build dynamic update query
        let mut updates = Vec::new();
        let mut params = Vec::new();

        if let Some(title) = input.title {
            updates.push(format!("title = ${}", params.len() + 2));
            params.push(title);
        }

        if let Some(description) = input.description {
            updates.push(format!("description = ${}", params.len() + 2));
            params.push(description);
        }

        if updates.is_empty() {
            return Err(Error::new("No fields to update"));
        }

        // Execute update (simplified - in production use proper query builder)
        let video = sqlx::query_as!(
            Video,
            r#"
            UPDATE videos
            SET updated_at = NOW()
            WHERE id = $1
            RETURNING id as "id: ID", title, description, url, thumbnail_url,
                      duration_seconds, channel_id as "channel_id: ID",
                      quality_score, view_count, like_count,
                      created_at, updated_at
            "#,
            input.id.to_string()
        )
        .fetch_one(pool)
        .await?;

        // Send subscription event
        let _ = self.event_sender.send(SubscriptionEvent::VideoUpdated(video.clone()));

        Ok(video)
    }

    async fn delete_video(&self, ctx: &Context<'_>, id: ID) -> Result<bool> {
        let pool = ctx.data::<PgPool>()?;

        let result = sqlx::query!(
            "DELETE FROM videos WHERE id = $1",
            id.to_string()
        )
        .execute(pool)
        .await?;

        if result.rows_affected() > 0 {
            let _ = self.event_sender.send(SubscriptionEvent::VideoDeleted(id));
            Ok(true)
        } else {
            Ok(false)
        }
    }

    // User mutations
    async fn create_user(&self, ctx: &Context<'_>, input: CreateUserInput) -> Result<User> {
        let pool = ctx.data::<PgPool>()?;

        // Hash password (simplified - use proper password hashing)
        let password_hash = format!("hashed_{}", input.password);
        let user_id = Uuid::new_v4().to_string();
        let now = Utc::now();

        let default_preferences = UserPreferences {
            categories: vec![],
            languages: vec!["en".to_string()],
            min_quality_score: 0.5,
            max_duration_seconds: 3600,
            notifications_enabled: true,
        };

        let user = sqlx::query_as!(
            User,
            r#"
            INSERT INTO users (id, username, email, password_hash,
                             role, preferences, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id as "id: ID", username, email,
                      role as "role: UserRole",
                      preferences as "preferences: UserPreferences",
                      created_at, last_login
            "#,
            user_id,
            input.username,
            input.email,
            password_hash,
            input.role.unwrap_or(UserRole::User) as i32,
            serde_json::to_value(&default_preferences)?,
            now
        )
        .fetch_one(pool)
        .await?;

        Ok(user)
    }

    async fn update_user_preferences(
        &self,
        ctx: &Context<'_>,
        input: UpdateUserPreferencesInput,
    ) -> Result<User> {
        let user_id = ctx.data::<String>()?.clone();
        let pool = ctx.data::<PgPool>()?;

        // Get current preferences
        let current_user = sqlx::query!(
            "SELECT preferences FROM users WHERE id = $1",
            user_id
        )
        .fetch_one(pool)
        .await?;

        let mut preferences: UserPreferences = serde_json::from_value(current_user.preferences)?;

        // Update preferences
        if let Some(categories) = input.categories {
            preferences.categories = categories;
        }
        if let Some(languages) = input.languages {
            preferences.languages = languages;
        }
        if let Some(min_quality) = input.min_quality_score {
            preferences.min_quality_score = min_quality;
        }

        // Save updated preferences
        let user = sqlx::query_as!(
            User,
            r#"
            UPDATE users
            SET preferences = $2
            WHERE id = $1
            RETURNING id as "id: ID", username, email,
                      role as "role: UserRole",
                      preferences as "preferences: UserPreferences",
                      created_at, last_login
            "#,
            user_id,
            serde_json::to_value(&preferences)?
        )
        .fetch_one(pool)
        .await?;

        Ok(user)
    }

    // Feed mutations
    async fn subscribe_to_feed(&self, ctx: &Context<'_>, url: String) -> Result<Feed> {
        let pool = ctx.data::<PgPool>()?;
        let feed_id = Uuid::new_v4().to_string();
        let now = Utc::now();

        let feed = sqlx::query_as!(
            Feed,
            r#"
            INSERT INTO feeds (id, url, title, description, category,
                             update_frequency_minutes, active, created_at)
            VALUES ($1, $2, 'New Feed', 'Pending', 'uncategorized', 60, true, $3)
            RETURNING id as "id: ID", url, title, description, category,
                      update_frequency_minutes, last_fetched, active, created_at
            "#,
            feed_id,
            url,
            now
        )
        .fetch_one(pool)
        .await?;

        Ok(feed)
    }
}

// GraphQL Subscription Root

#[derive(Debug, Clone)]
pub enum SubscriptionEvent {
    VideoCreated(Video),
    VideoUpdated(Video),
    VideoDeleted(ID),
    NewRecommendation(Recommendation),
}

pub struct SubscriptionRoot {
    event_receiver: broadcast::Receiver<SubscriptionEvent>,
}

#[Subscription]
impl SubscriptionRoot {
    async fn video_created(&self) -> impl Stream<Item = Video> {
        let mut receiver = self.event_receiver.resubscribe();

        stream::unfold(receiver, |mut rx| async move {
            loop {
                match rx.recv().await {
                    Ok(SubscriptionEvent::VideoCreated(video)) => {
                        return Some((video, rx));
                    }
                    Ok(_) => continue,
                    Err(_) => return None,
                }
            }
        })
    }

    async fn video_updated(&self) -> impl Stream<Item = Video> {
        let mut receiver = self.event_receiver.resubscribe();

        stream::unfold(receiver, |mut rx| async move {
            loop {
                match rx.recv().await {
                    Ok(SubscriptionEvent::VideoUpdated(video)) => {
                        return Some((video, rx));
                    }
                    Ok(_) => continue,
                    Err(_) => return None,
                }
            }
        })
    }

    async fn recommendations(&self, user_id: ID) -> impl Stream<Item = Recommendation> {
        let mut receiver = self.event_receiver.resubscribe();

        stream::unfold(receiver, move |mut rx| {
            let uid = user_id.clone();
            async move {
                loop {
                    match rx.recv().await {
                        Ok(SubscriptionEvent::NewRecommendation(rec)) => {
                            if rec.user_id == uid {
                                return Some((rec, rx));
                            }
                            continue;
                        }
                        Ok(_) => continue,
                        Err(_) => return None,
                    }
                }
            }
        })
    }
}

// Field resolvers for complex types

#[ComplexObject]
impl Video {
    async fn channel(&self, ctx: &Context<'_>) -> Result<Option<Channel>> {
        let loader = ctx.data::<DataLoader<ChannelLoader>>()?;
        Ok(loader.load_one(self.channel_id.clone()).await?)
    }

    async fn summary(&self, ctx: &Context<'_>) -> Result<Option<Summary>> {
        let pool = ctx.data::<PgPool>()?;

        let summary = sqlx::query_as!(
            Summary,
            r#"
            SELECT id as "id: ID", video_id as "video_id: ID",
                   content, key_points, sentiment_score,
                   language, word_count, created_at
            FROM summaries
            WHERE video_id = $1
            "#,
            self.id.to_string()
        )
        .fetch_optional(pool)
        .await?;

        Ok(summary)
    }
}

#[ComplexObject]
impl Channel {
    async fn videos(
        &self,
        ctx: &Context<'_>,
        limit: Option<i32>,
    ) -> Result<Vec<Video>> {
        let pool = ctx.data::<PgPool>()?;
        let limit = limit.unwrap_or(10);

        let videos = sqlx::query_as!(
            Video,
            r#"
            SELECT id as "id: ID", title, description, url, thumbnail_url,
                   duration_seconds, channel_id as "channel_id: ID",
                   quality_score, view_count, like_count,
                   created_at, updated_at
            FROM videos
            WHERE channel_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            "#,
            self.id.to_string(),
            limit as i64
        )
        .fetch_all(pool)
        .await?;

        Ok(videos)
    }

    async fn latest_video(&self, ctx: &Context<'_>) -> Result<Option<Video>> {
        let videos = self.videos(ctx, Some(1)).await?;
        Ok(videos.into_iter().next())
    }
}

// Schema builder

pub fn build_schema(pool: PgPool) -> Schema<QueryRoot, MutationRoot, SubscriptionRoot> {
    let (tx, rx) = broadcast::channel(1024);

    let channel_loader = DataLoader::new(
        ChannelLoader { pool: pool.clone() },
        tokio::spawn,
    );

    Schema::build(
        QueryRoot { pool: pool.clone() },
        MutationRoot {
            pool: pool.clone(),
            event_sender: tx,
        },
        SubscriptionRoot {
            event_receiver: rx,
        },
    )
    .data(pool)
    .data(channel_loader)
    .extension(Tracing)
    .finish()
}

// Axum handlers

pub async fn graphql_handler(
    State(schema): State<Schema<QueryRoot, MutationRoot, SubscriptionRoot>>,
    req: GraphQLRequest,
) -> GraphQLResponse {
    schema.execute(req.into_inner()).await.into()
}

pub async fn graphql_playground() -> impl IntoResponse {
    Html(playground_source(
        GraphQLPlaygroundConfig::new("/graphql").subscription_endpoint("/ws"),
    ))
}

pub async fn graphql_ws_handler(
    State(schema): State<Schema<QueryRoot, MutationRoot, SubscriptionRoot>>,
    ws: axum::extract::ws::WebSocketUpgrade,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| {
        GraphQLSubscription::new(socket, schema.clone(), ())
            .serve()
    })
}

// Router creation

pub fn create_graphql_router(pool: PgPool) -> Router {
    let schema = build_schema(pool);

    Router::new()
        .route("/graphql", get(graphql_playground).post(graphql_handler))
        .route("/ws", get(graphql_ws_handler))
        .with_state(schema)
}