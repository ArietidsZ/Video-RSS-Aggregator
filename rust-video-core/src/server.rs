use crate::{
    error::VideoRssError,
    extractor::{ExtractedVideo, RealDataExtractor},
    rss::RssGenerator,
    types::*,
    Result,
};
use axum::{
    extract::{Query, State},
    http::{header, StatusCode},
    response::{Html, IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower::ServiceBuilder;
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};
use tracing::{info, warn};

#[derive(Clone)]
pub struct AppState {
    extractor: Arc<RealDataExtractor>,
    cache: Arc<RwLock<HashMap<String, CachedData>>>,
    config: ServerConfig,
}

#[derive(Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub cache_ttl_seconds: u64,
    pub enable_cors: bool,
    pub enable_metrics: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8000,
            cache_ttl_seconds: 300, // 5 minutes
            enable_cors: true,
            enable_metrics: true,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct CachedData {
    data: Vec<ExtractedVideo>,
    timestamp: i64,
}

pub struct VideoRssServer {
    state: AppState,
}

impl VideoRssServer {
    pub fn new(config: ServerConfig) -> Result<Self> {
        let extractor = Arc::new(RealDataExtractor::new()?);
        let cache = Arc::new(RwLock::new(HashMap::new()));

        Ok(Self {
            state: AppState {
                extractor,
                cache,
                config,
            },
        })
    }

    pub async fn run(self) -> Result<()> {
        let app = self.create_app();
        let addr = format!("{}:{}", self.state.config.host, self.state.config.port);
        let listener = tokio::net::TcpListener::bind(&addr).await?;

        info!("Video RSS Server running on http://{}", addr);

        axum::serve(listener, app).await?;

        Ok(())
    }

    fn create_app(self) -> Router {
        let mut app = Router::new()
            .route("/", get(root_handler))
            .route("/health", get(health_handler))
            .route("/api/status", get(status_handler))
            .route("/api/videos", get(get_videos_handler))
            .route("/api/videos/:platform", get(get_platform_videos_handler))
            .route("/api/rss", get(generate_rss_handler))
            .route("/api/transcribe", post(transcribe_handler))
            .route("/api/refresh", post(refresh_cache_handler))
            .route("/metrics", get(metrics_handler))
            .with_state(self.state.clone());

        // Add middleware
        if self.state.config.enable_cors {
            app = app.layer(
                CorsLayer::new()
                    .allow_origin(Any)
                    .allow_methods(Any)
                    .allow_headers(Any),
            );
        }

        app = app.layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .into_inner(),
        );

        app
    }
}

// Handler functions
async fn root_handler() -> Html<&'static str> {
    Html(
        r#"<!DOCTYPE html>
<html>
<head>
    <title>Video RSS Aggregator</title>
    <style>
        body { font-family: system-ui; padding: 2rem; max-width: 800px; margin: 0 auto; }
        h1 { color: #333; }
        .endpoint { background: #f5f5f5; padding: 1rem; margin: 1rem 0; border-radius: 8px; }
        .method { color: #007bff; font-weight: bold; }
        code { background: #e9ecef; padding: 2px 6px; border-radius: 3px; }
    </style>
</head>
<body>
    <h1>🚀 Video RSS Aggregator API</h1>
    <p>High-performance Rust-powered video content aggregation service</p>

    <div class="endpoint">
        <span class="method">GET</span> <code>/api/status</code>
        <p>Get server status and statistics</p>
    </div>

    <div class="endpoint">
        <span class="method">GET</span> <code>/api/videos</code>
        <p>Get all videos from all platforms</p>
    </div>

    <div class="endpoint">
        <span class="method">GET</span> <code>/api/videos/{platform}</code>
        <p>Get videos from specific platform (bilibili, youtube, douyin)</p>
    </div>

    <div class="endpoint">
        <span class="method">GET</span> <code>/api/rss</code>
        <p>Generate RSS feed with all videos</p>
    </div>

    <div class="endpoint">
        <span class="method">POST</span> <code>/api/transcribe</code>
        <p>Transcribe video content (AI-powered)</p>
    </div>

    <div class="endpoint">
        <span class="method">POST</span> <code>/api/refresh</code>
        <p>Refresh video cache</p>
    </div>

    <div class="endpoint">
        <span class="method">GET</span> <code>/metrics</code>
        <p>Prometheus-compatible metrics endpoint</p>
    </div>
</body>
</html>"#,
    )
}

async fn health_handler() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

#[derive(Serialize)]
struct StatusResponse {
    status: String,
    platforms: Vec<PlatformStatus>,
    cache_size: usize,
    uptime_seconds: u64,
    version: String,
}

#[derive(Serialize)]
struct PlatformStatus {
    name: String,
    available: bool,
    video_count: usize,
    last_updated: Option<String>,
}

async fn status_handler(State(state): State<AppState>) -> impl IntoResponse {
    let cache = state.cache.read().await;

    let platforms = vec![
        Platform::Bilibili,
        Platform::YouTube,
        Platform::Douyin,
        Platform::Kuaishou,
    ];

    let platform_statuses: Vec<PlatformStatus> = platforms
        .iter()
        .map(|platform| {
            let key = platform.as_str();
            let cached = cache.get(key);

            PlatformStatus {
                name: key.to_string(),
                available: cached.is_some(),
                video_count: cached.map(|c| c.data.len()).unwrap_or(0),
                last_updated: cached.map(|c| {
                    chrono::DateTime::from_timestamp(c.timestamp, 0)
                        .map(|dt| dt.to_rfc3339())
                        .unwrap_or_default()
                }),
            }
        })
        .collect();

    Json(StatusResponse {
        status: "online".to_string(),
        platforms: platform_statuses,
        cache_size: cache.len(),
        uptime_seconds: 0, // Would need to track start time
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

#[derive(Deserialize)]
struct VideoQuery {
    limit: Option<usize>,
    sort: Option<String>,
}

async fn get_videos_handler(
    State(state): State<AppState>,
    Query(params): Query<VideoQuery>,
) -> impl IntoResponse {
    let videos = fetch_all_videos(&state).await;

    let limit = params.limit.unwrap_or(50);
    let videos: Vec<_> = videos.into_iter().take(limit).collect();

    Json(serde_json::json!({
        "total": videos.len(),
        "videos": videos,
    }))
}

async fn get_platform_videos_handler(
    State(state): State<AppState>,
    axum::extract::Path(platform): axum::extract::Path<String>,
) -> impl IntoResponse {
    let platform_enum = match platform.as_str() {
        "bilibili" => Platform::Bilibili,
        "youtube" => Platform::YouTube,
        "douyin" => Platform::Douyin,
        "kuaishou" => Platform::Kuaishou,
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": "Invalid platform. Use: bilibili, youtube, douyin, or kuaishou"
                })),
            )
                .into_response();
        }
    };

    let videos = fetch_platform_videos(&state, platform_enum).await;

    Json(serde_json::json!({
        "platform": platform,
        "total": videos.len(),
        "videos": videos,
    }))
        .into_response()
}

async fn generate_rss_handler(State(state): State<AppState>) -> Response {
    let videos = fetch_all_videos(&state).await;

    // Convert ExtractedVideo to VideoInfo for RSS generation
    let video_infos: Vec<VideoInfo> = videos
        .into_iter()
        .map(|v| VideoInfo {
            id: v.id,
            title: v.title,
            description: v.description,
            url: v.url,
            author: v.author,
            upload_date: v.extracted_at,
            duration: None,
            view_count: v.view_count.unwrap_or(0),
            like_count: v.like_count.unwrap_or(0),
            comment_count: 0,
            tags: v.tags,
            thumbnail_url: v.thumbnail,
            platform: v.platform,
            transcription: None,
        })
        .collect();

    let rss_gen = RssGenerator::new(RssConfig::default());

    match rss_gen.generate_feed(&video_infos) {
        Ok(rss_content) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/rss+xml; charset=utf-8")],
            rss_content,
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("Failed to generate RSS: {}", e)
            })),
        )
            .into_response(),
    }
}

#[derive(Deserialize)]
struct TranscribeRequest {
    video_url: String,
    platform: Option<String>,
}

async fn transcribe_handler(
    State(_state): State<AppState>,
    Json(req): Json<TranscribeRequest>,
) -> impl IntoResponse {
    // This would integrate with audio transcription service
    // For now, return a placeholder response
    Json(serde_json::json!({
        "status": "pending",
        "message": "Transcription service integration pending",
        "video_url": req.video_url,
    }))
}

async fn refresh_cache_handler(State(state): State<AppState>) -> impl IntoResponse {
    info!("Refreshing video cache...");

    let all_videos = state.extractor.extract_all().await;
    let mut cache = state.cache.write().await;

    let timestamp = chrono::Utc::now().timestamp();
    let mut total_videos = 0;

    for (platform, videos) in all_videos {
        total_videos += videos.len();
        cache.insert(
            platform.as_str().to_string(),
            CachedData {
                data: videos,
                timestamp,
            },
        );
    }

    Json(serde_json::json!({
        "status": "success",
        "message": "Cache refreshed successfully",
        "total_videos": total_videos,
        "timestamp": timestamp,
    }))
}

async fn metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    if !state.config.enable_metrics {
        return (StatusCode::NOT_FOUND, "Metrics disabled").into_response();
    }

    let cache = state.cache.read().await;
    let total_videos: usize = cache.values().map(|c| c.data.len()).sum();

    let metrics = format!(
        "# HELP video_rss_total_videos Total number of videos in cache\n\
         # TYPE video_rss_total_videos gauge\n\
         video_rss_total_videos {}\n\
         \n\
         # HELP video_rss_cache_size Number of cached platforms\n\
         # TYPE video_rss_cache_size gauge\n\
         video_rss_cache_size {}\n",
        total_videos,
        cache.len()
    );

    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        metrics,
    )
        .into_response()
}

// Helper functions
async fn fetch_all_videos(state: &AppState) -> Vec<ExtractedVideo> {
    let mut should_refresh = false;

    {
        let cache = state.cache.read().await;
        if cache.is_empty() {
            should_refresh = true;
        } else {
            // Check if cache is stale
            let now = chrono::Utc::now().timestamp();
            for cached in cache.values() {
                if now - cached.timestamp > state.config.cache_ttl_seconds as i64 {
                    should_refresh = true;
                    break;
                }
            }
        }
    }

    if should_refresh {
        info!("Cache miss or stale, fetching fresh data...");
        let all_videos = state.extractor.extract_all().await;
        let mut cache = state.cache.write().await;

        let timestamp = chrono::Utc::now().timestamp();
        for (platform, videos) in all_videos {
            cache.insert(
                platform.as_str().to_string(),
                CachedData {
                    data: videos,
                    timestamp,
                },
            );
        }
    }

    let cache = state.cache.read().await;
    cache
        .values()
        .flat_map(|c| c.data.clone())
        .collect()
}

async fn fetch_platform_videos(state: &AppState, platform: Platform) -> Vec<ExtractedVideo> {
    let cache = state.cache.read().await;

    cache
        .get(platform.as_str())
        .map(|c| c.data.clone())
        .unwrap_or_default()
}

// Main entry point
pub async fn run_server(config: ServerConfig) -> Result<()> {
    let server = VideoRssServer::new(config)?;
    server.run().await
}