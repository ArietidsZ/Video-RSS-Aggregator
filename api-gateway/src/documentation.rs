use axum::{
    extract::State,
    http::StatusCode,
    response::{Html, IntoResponse, Json},
    routing::get,
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use utoipa::{
    openapi::{
        security::{ApiKey, ApiKeyValue, HttpAuthScheme, HttpBuilder, SecurityScheme},
        ContactBuilder, InfoBuilder, LicenseBuilder, ServerBuilder,
    },
    Modify, OpenApi, ToSchema,
};
use utoipa_swagger_ui::SwaggerUi;

// OpenAPI Documentation

#[derive(OpenApi)]
#[openapi(
    paths(
        get_videos,
        get_video,
        create_video,
        update_video,
        delete_video,
        get_channels,
        get_channel,
        get_summary,
        create_summary,
        get_recommendations,
        get_analytics,
        create_webhook,
        list_webhooks,
        get_webhook,
        update_webhook,
        delete_webhook,
    ),
    components(
        schemas(
            Video,
            Channel,
            User,
            Summary,
            Recommendation,
            Analytics,
            Webhook,
            WebhookEvent,
            ErrorResponse,
            PaginatedResponse<Video>,
            CreateVideoRequest,
            UpdateVideoRequest,
            CreateSummaryRequest,
            CreateWebhookRequest,
            UpdateWebhookRequest,
        )
    ),
    modifiers(&SecurityAddon),
    tags(
        (name = "videos", description = "Video management endpoints"),
        (name = "channels", description = "Channel management endpoints"),
        (name = "summaries", description = "Summary generation endpoints"),
        (name = "recommendations", description = "Recommendation engine endpoints"),
        (name = "analytics", description = "Analytics and statistics endpoints"),
        (name = "webhooks", description = "Webhook management endpoints"),
        (name = "auth", description = "Authentication endpoints"),
    ),
    info(
        title = "Video RSS Aggregator API",
        version = "3.0.0",
        description = "A high-performance video RSS aggregator with AI-powered summaries",
        contact(
            name = "API Support",
            email = "support@video-aggregator.com",
            url = "https://docs.video-aggregator.com"
        ),
        license(
            name = "MIT",
            url = "https://opensource.org/licenses/MIT"
        )
    ),
    servers(
        (url = "https://api.video-aggregator.com", description = "Production server"),
        (url = "https://staging-api.video-aggregator.com", description = "Staging server"),
        (url = "http://localhost:8080", description = "Local development server")
    )
)]
pub struct ApiDoc;

struct SecurityAddon;

impl Modify for SecurityAddon {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        if let Some(components) = &mut openapi.components {
            components.add_security_scheme(
                "api_key",
                SecurityScheme::ApiKey(ApiKey::Header(ApiKeyValue::new("X-API-Key"))),
            );
            components.add_security_scheme(
                "bearer",
                SecurityScheme::Http(
                    HttpBuilder::new()
                        .scheme(HttpAuthScheme::Bearer)
                        .bearer_format("JWT")
                        .build(),
                ),
            );
        }
    }
}

// Schema definitions

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct Video {
    /// Unique identifier for the video
    #[schema(example = "550e8400-e29b-41d4-a716-446655440000")]
    pub id: String,

    /// Video title
    #[schema(example = "Introduction to Rust Programming")]
    pub title: String,

    /// Video description
    #[schema(example = "Learn the basics of Rust programming language")]
    pub description: String,

    /// Video URL
    #[schema(example = "https://example.com/video.mp4")]
    pub url: String,

    /// Thumbnail URL
    #[schema(example = "https://example.com/thumbnail.jpg")]
    pub thumbnail_url: Option<String>,

    /// Duration in seconds
    #[schema(example = 1800)]
    pub duration_seconds: i32,

    /// Channel ID
    #[schema(example = "channel_123")]
    pub channel_id: String,

    /// Quality score (0.0 to 1.0)
    #[schema(example = 0.85)]
    pub quality_score: f32,

    /// View count
    #[schema(example = 1000000)]
    pub view_count: i64,

    /// Like count
    #[schema(example = 50000)]
    pub like_count: i64,

    /// Creation timestamp
    #[schema(example = "2024-01-01T00:00:00Z")]
    pub created_at: String,

    /// Last update timestamp
    #[schema(example = "2024-01-02T00:00:00Z")]
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct Channel {
    /// Unique identifier for the channel
    #[schema(example = "channel_123")]
    pub id: String,

    /// Channel name
    #[schema(example = "Tech Tutorials")]
    pub name: String,

    /// Channel description
    #[schema(example = "Educational technology content")]
    pub description: String,

    /// Channel URL
    #[schema(example = "https://example.com/channel")]
    pub url: String,

    /// Subscriber count
    #[schema(example = 100000)]
    pub subscriber_count: i64,

    /// Total video count
    #[schema(example = 500)]
    pub video_count: i64,

    /// Creation timestamp
    #[schema(example = "2023-01-01T00:00:00Z")]
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct User {
    /// Unique user identifier
    #[schema(example = "user_456")]
    pub id: String,

    /// Username
    #[schema(example = "john_doe")]
    pub username: String,

    /// Email address
    #[schema(example = "john@example.com")]
    pub email: String,

    /// User role
    #[schema(example = "premium")]
    pub role: String,

    /// User preferences
    pub preferences: serde_json::Value,

    /// Registration timestamp
    #[schema(example = "2023-06-01T00:00:00Z")]
    pub created_at: String,

    /// Last login timestamp
    #[schema(example = "2024-01-15T12:00:00Z")]
    pub last_login: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct Summary {
    /// Unique summary identifier
    #[schema(example = "summary_789")]
    pub id: String,

    /// Associated video ID
    #[schema(example = "550e8400-e29b-41d4-a716-446655440000")]
    pub video_id: String,

    /// Summary content
    #[schema(example = "This video covers the basics of Rust programming...")]
    pub content: String,

    /// Key points extracted
    #[schema(example = json!(["Memory safety", "Ownership system", "Pattern matching"]))]
    pub key_points: Vec<String>,

    /// Sentiment score (-1.0 to 1.0)
    #[schema(example = 0.7)]
    pub sentiment_score: f32,

    /// Language code
    #[schema(example = "en")]
    pub language: String,

    /// Word count
    #[schema(example = 250)]
    pub word_count: i32,

    /// Creation timestamp
    #[schema(example = "2024-01-01T00:00:00Z")]
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct Recommendation {
    /// Unique recommendation identifier
    #[schema(example = "rec_001")]
    pub id: String,

    /// User ID
    #[schema(example = "user_456")]
    pub user_id: String,

    /// Recommended video ID
    #[schema(example = "550e8400-e29b-41d4-a716-446655440000")]
    pub video_id: String,

    /// Recommendation score (0.0 to 1.0)
    #[schema(example = 0.92)]
    pub score: f32,

    /// Reason for recommendation
    #[schema(example = "Based on your viewing history")]
    pub reason: String,

    /// Creation timestamp
    #[schema(example = "2024-01-15T00:00:00Z")]
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct Analytics {
    /// Total number of videos
    #[schema(example = 10000)]
    pub total_videos: i64,

    /// Total number of channels
    #[schema(example = 500)]
    pub total_channels: i64,

    /// Total number of users
    #[schema(example = 50000)]
    pub total_users: i64,

    /// Total number of summaries generated
    #[schema(example = 8000)]
    pub total_summaries: i64,

    /// Average quality score
    #[schema(example = 0.75)]
    pub avg_quality_score: f32,

    /// Processing rate (videos per hour)
    #[schema(example = 100.5)]
    pub processing_rate: f32,

    /// Cache hit rate
    #[schema(example = 0.85)]
    pub cache_hit_rate: f32,

    /// Error rate
    #[schema(example = 0.02)]
    pub error_rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct Webhook {
    /// Unique webhook identifier
    #[schema(example = "webhook_123")]
    pub id: String,

    /// Webhook name
    #[schema(example = "Video Processing Webhook")]
    pub name: String,

    /// Target URL
    #[schema(example = "https://example.com/webhook")]
    pub url: String,

    /// Subscribed events
    #[schema(example = json!(["video.created", "video.processed"]))]
    pub events: Vec<String>,

    /// Whether webhook is active
    #[schema(example = true)]
    pub active: bool,

    /// Creation timestamp
    #[schema(example = "2024-01-01T00:00:00Z")]
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct WebhookEvent {
    /// Event type
    #[schema(example = "video.created")]
    pub event: String,

    /// Event payload
    pub payload: serde_json::Value,

    /// Event timestamp
    #[schema(example = "2024-01-15T12:30:00Z")]
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ErrorResponse {
    /// Error code
    #[schema(example = "VALIDATION_ERROR")]
    pub code: String,

    /// Error message
    #[schema(example = "Invalid request parameters")]
    pub message: String,

    /// Additional error details
    pub details: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct PaginatedResponse<T> {
    /// Data items
    pub data: Vec<T>,

    /// Total count of items
    #[schema(example = 100)]
    pub total: i64,

    /// Current page number
    #[schema(example = 1)]
    pub page: i32,

    /// Items per page
    #[schema(example = 20)]
    pub per_page: i32,

    /// Whether there are more pages
    #[schema(example = true)]
    pub has_next: bool,
}

// Request schemas

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct CreateVideoRequest {
    /// Video title
    #[schema(example = "My Video")]
    pub title: String,

    /// Video description
    #[schema(example = "Video description")]
    pub description: String,

    /// Video URL
    #[schema(example = "https://example.com/video.mp4")]
    pub url: String,

    /// Channel ID
    #[schema(example = "channel_123")]
    pub channel_id: String,

    /// Duration in seconds
    #[schema(example = 300)]
    pub duration_seconds: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct UpdateVideoRequest {
    /// Updated title
    #[schema(example = "Updated Title")]
    pub title: Option<String>,

    /// Updated description
    #[schema(example = "Updated description")]
    pub description: Option<String>,

    /// Updated quality score
    #[schema(example = 0.9)]
    pub quality_score: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct CreateSummaryRequest {
    /// Video ID to summarize
    #[schema(example = "550e8400-e29b-41d4-a716-446655440000")]
    pub video_id: String,

    /// Summary content
    #[schema(example = "This video discusses...")]
    pub content: String,

    /// Key points
    #[schema(example = json!(["Point 1", "Point 2", "Point 3"]))]
    pub key_points: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct CreateWebhookRequest {
    /// Webhook name
    #[schema(example = "My Webhook")]
    pub name: String,

    /// Target URL
    #[schema(example = "https://example.com/webhook")]
    pub url: String,

    /// Events to subscribe to
    #[schema(example = json!(["video.created", "video.updated"]))]
    pub events: Vec<String>,

    /// Webhook secret for signature verification
    #[schema(example = "secret_key_123")]
    pub secret: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct UpdateWebhookRequest {
    /// Updated name
    pub name: Option<String>,

    /// Updated URL
    pub url: Option<String>,

    /// Updated events
    pub events: Option<Vec<String>>,

    /// Whether webhook is active
    pub active: Option<bool>,
}

// API Endpoints with OpenAPI documentation

/// Get all videos
#[utoipa::path(
    get,
    path = "/api/v3/videos",
    tag = "videos",
    params(
        ("limit" = Option<i32>, Query, description = "Number of items to return"),
        ("offset" = Option<i32>, Query, description = "Number of items to skip"),
        ("channel_id" = Option<String>, Query, description = "Filter by channel ID"),
        ("min_quality_score" = Option<f32>, Query, description = "Minimum quality score"),
    ),
    responses(
        (status = 200, description = "List of videos", body = PaginatedResponse<Video>),
        (status = 401, description = "Unauthorized", body = ErrorResponse),
        (status = 429, description = "Rate limit exceeded", body = ErrorResponse),
    ),
    security(
        ("api_key" = []),
        ("bearer" = [])
    )
)]
async fn get_videos() -> Json<PaginatedResponse<Video>> {
    // Implementation
    Json(PaginatedResponse {
        data: vec![],
        total: 0,
        page: 1,
        per_page: 20,
        has_next: false,
    })
}

/// Get a specific video by ID
#[utoipa::path(
    get,
    path = "/api/v3/videos/{id}",
    tag = "videos",
    params(
        ("id" = String, Path, description = "Video ID")
    ),
    responses(
        (status = 200, description = "Video details", body = Video),
        (status = 404, description = "Video not found", body = ErrorResponse),
    ),
    security(
        ("api_key" = []),
        ("bearer" = [])
    )
)]
async fn get_video() -> Json<Video> {
    // Implementation
    Json(Video {
        id: String::new(),
        title: String::new(),
        description: String::new(),
        url: String::new(),
        thumbnail_url: None,
        duration_seconds: 0,
        channel_id: String::new(),
        quality_score: 0.0,
        view_count: 0,
        like_count: 0,
        created_at: String::new(),
        updated_at: String::new(),
    })
}

/// Create a new video
#[utoipa::path(
    post,
    path = "/api/v3/videos",
    tag = "videos",
    request_body = CreateVideoRequest,
    responses(
        (status = 201, description = "Video created", body = Video),
        (status = 400, description = "Invalid request", body = ErrorResponse),
        (status = 401, description = "Unauthorized", body = ErrorResponse),
    ),
    security(
        ("bearer" = [])
    )
)]
async fn create_video() -> Json<Video> {
    // Implementation
    Json(Video {
        id: String::new(),
        title: String::new(),
        description: String::new(),
        url: String::new(),
        thumbnail_url: None,
        duration_seconds: 0,
        channel_id: String::new(),
        quality_score: 0.0,
        view_count: 0,
        like_count: 0,
        created_at: String::new(),
        updated_at: String::new(),
    })
}

/// Update a video
#[utoipa::path(
    patch,
    path = "/api/v3/videos/{id}",
    tag = "videos",
    params(
        ("id" = String, Path, description = "Video ID")
    ),
    request_body = UpdateVideoRequest,
    responses(
        (status = 200, description = "Video updated", body = Video),
        (status = 404, description = "Video not found", body = ErrorResponse),
    ),
    security(
        ("bearer" = [])
    )
)]
async fn update_video() -> Json<Video> {
    // Implementation
    Json(Video {
        id: String::new(),
        title: String::new(),
        description: String::new(),
        url: String::new(),
        thumbnail_url: None,
        duration_seconds: 0,
        channel_id: String::new(),
        quality_score: 0.0,
        view_count: 0,
        like_count: 0,
        created_at: String::new(),
        updated_at: String::new(),
    })
}

/// Delete a video
#[utoipa::path(
    delete,
    path = "/api/v3/videos/{id}",
    tag = "videos",
    params(
        ("id" = String, Path, description = "Video ID")
    ),
    responses(
        (status = 204, description = "Video deleted"),
        (status = 404, description = "Video not found", body = ErrorResponse),
    ),
    security(
        ("bearer" = [])
    )
)]
async fn delete_video() -> StatusCode {
    StatusCode::NO_CONTENT
}

// Additional endpoint definitions (similar pattern for other resources)

/// Get all channels
#[utoipa::path(
    get,
    path = "/api/v3/channels",
    tag = "channels",
    responses(
        (status = 200, description = "List of channels", body = Vec<Channel>),
    ),
    security(
        ("api_key" = [])
    )
)]
async fn get_channels() -> Json<Vec<Channel>> {
    Json(vec![])
}

/// Get a specific channel
#[utoipa::path(
    get,
    path = "/api/v3/channels/{id}",
    tag = "channels",
    params(
        ("id" = String, Path, description = "Channel ID")
    ),
    responses(
        (status = 200, description = "Channel details", body = Channel),
        (status = 404, description = "Channel not found", body = ErrorResponse),
    )
)]
async fn get_channel() -> Json<Channel> {
    Json(Channel {
        id: String::new(),
        name: String::new(),
        description: String::new(),
        url: String::new(),
        subscriber_count: 0,
        video_count: 0,
        created_at: String::new(),
    })
}

/// Get video summary
#[utoipa::path(
    get,
    path = "/api/v3/videos/{id}/summary",
    tag = "summaries",
    params(
        ("id" = String, Path, description = "Video ID")
    ),
    responses(
        (status = 200, description = "Video summary", body = Summary),
        (status = 404, description = "Summary not found", body = ErrorResponse),
    )
)]
async fn get_summary() -> Json<Summary> {
    Json(Summary {
        id: String::new(),
        video_id: String::new(),
        content: String::new(),
        key_points: vec![],
        sentiment_score: 0.0,
        language: String::new(),
        word_count: 0,
        created_at: String::new(),
    })
}

/// Create a summary
#[utoipa::path(
    post,
    path = "/api/v3/summaries",
    tag = "summaries",
    request_body = CreateSummaryRequest,
    responses(
        (status = 201, description = "Summary created", body = Summary),
        (status = 400, description = "Invalid request", body = ErrorResponse),
    ),
    security(
        ("bearer" = [])
    )
)]
async fn create_summary() -> Json<Summary> {
    Json(Summary {
        id: String::new(),
        video_id: String::new(),
        content: String::new(),
        key_points: vec![],
        sentiment_score: 0.0,
        language: String::new(),
        word_count: 0,
        created_at: String::new(),
    })
}

/// Get recommendations
#[utoipa::path(
    get,
    path = "/api/v3/recommendations",
    tag = "recommendations",
    params(
        ("user_id" = Option<String>, Query, description = "User ID for personalized recommendations"),
        ("limit" = Option<i32>, Query, description = "Number of recommendations to return"),
    ),
    responses(
        (status = 200, description = "List of recommendations", body = Vec<Recommendation>),
    ),
    security(
        ("bearer" = [])
    )
)]
async fn get_recommendations() -> Json<Vec<Recommendation>> {
    Json(vec![])
}

/// Get analytics
#[utoipa::path(
    get,
    path = "/api/v3/analytics",
    tag = "analytics",
    responses(
        (status = 200, description = "System analytics", body = Analytics),
    ),
    security(
        ("bearer" = [])
    )
)]
async fn get_analytics() -> Json<Analytics> {
    Json(Analytics {
        total_videos: 0,
        total_channels: 0,
        total_users: 0,
        total_summaries: 0,
        avg_quality_score: 0.0,
        processing_rate: 0.0,
        cache_hit_rate: 0.0,
        error_rate: 0.0,
    })
}

/// Create a webhook
#[utoipa::path(
    post,
    path = "/api/v3/webhooks",
    tag = "webhooks",
    request_body = CreateWebhookRequest,
    responses(
        (status = 201, description = "Webhook created", body = Webhook),
        (status = 400, description = "Invalid request", body = ErrorResponse),
    ),
    security(
        ("bearer" = [])
    )
)]
async fn create_webhook() -> Json<Webhook> {
    Json(Webhook {
        id: String::new(),
        name: String::new(),
        url: String::new(),
        events: vec![],
        active: true,
        created_at: String::new(),
    })
}

/// List webhooks
#[utoipa::path(
    get,
    path = "/api/v3/webhooks",
    tag = "webhooks",
    responses(
        (status = 200, description = "List of webhooks", body = Vec<Webhook>),
    ),
    security(
        ("bearer" = [])
    )
)]
async fn list_webhooks() -> Json<Vec<Webhook>> {
    Json(vec![])
}

/// Get webhook details
#[utoipa::path(
    get,
    path = "/api/v3/webhooks/{id}",
    tag = "webhooks",
    params(
        ("id" = String, Path, description = "Webhook ID")
    ),
    responses(
        (status = 200, description = "Webhook details", body = Webhook),
        (status = 404, description = "Webhook not found", body = ErrorResponse),
    ),
    security(
        ("bearer" = [])
    )
)]
async fn get_webhook() -> Json<Webhook> {
    Json(Webhook {
        id: String::new(),
        name: String::new(),
        url: String::new(),
        events: vec![],
        active: true,
        created_at: String::new(),
    })
}

/// Update webhook
#[utoipa::path(
    patch,
    path = "/api/v3/webhooks/{id}",
    tag = "webhooks",
    params(
        ("id" = String, Path, description = "Webhook ID")
    ),
    request_body = UpdateWebhookRequest,
    responses(
        (status = 200, description = "Webhook updated", body = Webhook),
        (status = 404, description = "Webhook not found", body = ErrorResponse),
    ),
    security(
        ("bearer" = [])
    )
)]
async fn update_webhook() -> Json<Webhook> {
    Json(Webhook {
        id: String::new(),
        name: String::new(),
        url: String::new(),
        events: vec![],
        active: true,
        created_at: String::new(),
    })
}

/// Delete webhook
#[utoipa::path(
    delete,
    path = "/api/v3/webhooks/{id}",
    tag = "webhooks",
    params(
        ("id" = String, Path, description = "Webhook ID")
    ),
    responses(
        (status = 204, description = "Webhook deleted"),
        (status = 404, description = "Webhook not found", body = ErrorResponse),
    ),
    security(
        ("bearer" = [])
    )
)]
async fn delete_webhook() -> StatusCode {
    StatusCode::NO_CONTENT
}

// Documentation UI handlers

pub async fn swagger_ui() -> impl IntoResponse {
    let openapi = ApiDoc::openapi();
    SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", openapi)
}

pub async fn openapi_json() -> impl IntoResponse {
    Json(ApiDoc::openapi())
}

pub async fn redoc() -> impl IntoResponse {
    Html(r#"
    <!DOCTYPE html>
    <html>
      <head>
        <title>Video RSS Aggregator API Documentation</title>
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700" rel="stylesheet">
      </head>
      <body>
        <redoc spec-url='/api-docs/openapi.json'></redoc>
        <script src="https://cdn.jsdelivr.net/npm/redoc@latest/bundles/redoc.standalone.js"> </script>
      </body>
    </html>
    "#)
}

pub async fn rapidoc() -> impl IntoResponse {
    Html(r#"
    <!DOCTYPE html>
    <html>
      <head>
        <title>Video RSS Aggregator API Documentation</title>
        <meta charset="utf-8">
        <script type="module" src="https://unpkg.com/rapidoc/dist/rapidoc-min.js"></script>
      </head>
      <body>
        <rapi-doc
          spec-url="/api-docs/openapi.json"
          theme="dark"
          render-style="view"
          style="height: 100vh; width: 100%"
          nav-bg-color="#2c3e50"
          primary-color="#3498db"
          show-header="true"
          allow-try="true"
          allow-authentication="true"
          allow-server-selection="true"
          show-info="true"
          show-components="true"
        > </rapi-doc>
      </body>
    </html>
    "#)
}

// API documentation router

pub fn create_docs_router() -> Router {
    Router::new()
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .route("/api-docs/openapi.json", get(openapi_json))
        .route("/redoc", get(redoc))
        .route("/rapidoc", get(rapidoc))
}