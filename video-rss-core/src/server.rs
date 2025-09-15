use crate::{
    cache::{CacheManager, RedisCache, MemoryCache, cache_key_for_platform, cache_key_for_rss},
    error::VideoRssError,
    extractor::{ExtractedVideo, VideoExtractor},
    rss::RssGenerator,
    types::*,
    Result,
};
use axum::{
    extract::{Query, State, Path, ConnectInfo},
    http::{header, HeaderMap, HeaderValue, StatusCode, Method},
    middleware::{self, Next},
    response::{Html, IntoResponse, Json, Response},
    routing::{get, post},
    Router,
    body::Body,
    extract::Request,
};
use governor::{Quota, RateLimiter, DefaultKeyedRateLimiter};
use metrics::{counter, histogram, gauge};
use nonzero_ext::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};
use tower::ServiceBuilder;
use tower_http::{
    compression::CompressionLayer,
    cors::{Any, CorsLayer},
    limit::RequestBodyLimitLayer,
    timeout::TimeoutLayer,
    trace::TraceLayer,
};
use tracing::{info, warn, debug, error, Span};

type AppRateLimiter = DefaultKeyedRateLimiter<SocketAddr>;

#[derive(Clone)]
pub struct AppState {
    extractor: Arc<VideoExtractor>,
    cache_manager: Arc<CacheManager>,
    config: ServerConfig,
    rate_limiter: Arc<AppRateLimiter>,
    semaphore: Arc<Semaphore>,
    metrics: Arc<ServerMetrics>,
}

#[derive(Default)]
pub struct ServerMetrics {
    pub requests_total: metrics::Counter,
    pub request_duration: metrics::Histogram,
    pub cache_hits: metrics::Counter,
    pub cache_misses: metrics::Counter,
    pub active_connections: metrics::Gauge,
    pub rate_limited_requests: metrics::Counter,
    pub video_fetch_duration: metrics::Histogram,
}

impl ServerMetrics {
    pub fn new() -> Self {
        Self {
            requests_total: counter!("http_requests_total"),
            request_duration: histogram!("http_request_duration_seconds"),
            cache_hits: counter!("cache_hits_total"),
            cache_misses: counter!("cache_misses_total"),
            active_connections: gauge!("active_connections"),
            rate_limited_requests: counter!("rate_limited_requests_total"),
            video_fetch_duration: histogram!("video_fetch_duration_seconds"),
        }
    }
}

#[derive(Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub cache_ttl_seconds: u64,
    pub enable_cors: bool,
    pub enable_metrics: bool,
    pub max_concurrent_requests: usize,
    pub rate_limit_per_second: NonZeroU32,
    pub redis_url: Option<String>,
    pub request_timeout: Duration,
    pub max_request_size: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8000,
            cache_ttl_seconds: 300, // 5 minutes
            enable_cors: true,
            enable_metrics: true,
            max_concurrent_requests: 1000,
            rate_limit_per_second: NonZeroU32::new(100).unwrap(),
            redis_url: None,
            request_timeout: Duration::from_secs(30),
            max_request_size: 1024 * 1024, // 1MB
        }
    }
}

pub struct VideoRssServer {
    state: AppState,
}

impl VideoRssServer {
    pub async fn new(config: ServerConfig) -> Result<Self> {
        let extractor = Arc::new(VideoExtractor::new()?);

        // Initialize cache
        let cache_manager = if let Some(redis_url) = &config.redis_url {
            info!("Initializing Redis cache at: {}", redis_url);
            let redis_cache = RedisCache::new(redis_url, "video_rss".to_string()).await?;
            Arc::new(CacheManager::new(Box::new(redis_cache), config.cache_ttl_seconds))
        } else {
            info!("Using memory cache");
            let memory_cache = MemoryCache::new("video_rss".to_string());
            Arc::new(CacheManager::new(Box::new(memory_cache), config.cache_ttl_seconds))
        };

        // Initialize rate limiter
        let quota = Quota::per_second(config.rate_limit_per_second);
        let rate_limiter = Arc::new(RateLimiter::keyed(quota));

        // Initialize semaphore for concurrent request limiting
        let semaphore = Arc::new(Semaphore::new(config.max_concurrent_requests));

        // Initialize metrics
        let metrics = Arc::new(ServerMetrics::new());

        Ok(Self {
            state: AppState {
                extractor,
                cache_manager,
                config,
                rate_limiter,
                semaphore,
                metrics,
            },
        })
    }

    pub async fn run(self) -> Result<()> {
        let app = self.create_app();
        let addr = format!("{}:{}", self.state.config.host, self.state.config.port);
        let listener = tokio::net::TcpListener::bind(&addr).await?;

        info!("High-Performance Video RSS Server running on http://{}", addr);
        info!("Features enabled:");
        info!("  - Redis caching: {}", self.state.config.redis_url.is_some());
        info!("  - Rate limiting: {} req/sec", self.state.config.rate_limit_per_second);
        info!("  - Max concurrent: {}", self.state.config.max_concurrent_requests);
        info!("  - Request timeout: {:?}", self.state.config.request_timeout);

        axum::serve(listener, app.into_make_service_with_connect_info::<SocketAddr>()).await?;

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
            .route("/api/rss/:platforms", get(generate_custom_rss_handler))
            .route("/api/transcribe", post(transcribe_handler))
            .route("/api/refresh", post(refresh_cache_handler))
            .route("/api/cache/invalidate", post(invalidate_cache_handler));

        if self.state.config.enable_metrics {
            app = app.route("/metrics", get(metrics_handler));
        }

        app = app
            .layer(middleware::from_fn_with_state(
                self.state.clone(),
                rate_limit_middleware,
            ))
            .layer(middleware::from_fn_with_state(
                self.state.clone(),
                metrics_middleware,
            ))
            .layer(middleware::from_fn_with_state(
                self.state.clone(),
                semaphore_middleware,
            ));

        // Add performance layers
        let service_builder = ServiceBuilder::new()
            .layer(TimeoutLayer::new(self.state.config.request_timeout))
            .layer(RequestBodyLimitLayer::new(self.state.config.max_request_size))
            .layer(CompressionLayer::new())
            .layer(TraceLayer::new_for_http());

        if self.state.config.enable_cors {
            app = app.layer(
                CorsLayer::new()
                    .allow_origin(Any)
                    .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
                    .allow_headers(Any)
                    .max_age(Duration::from_secs(3600)),
            );
        }

        app = app.layer(service_builder).with_state(self.state);

        app
    }
}

// Middleware implementations
async fn rate_limit_middleware(
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    match state.rate_limiter.check_key(&addr) {
        Ok(_) => Ok(next.run(request).await),
        Err(_) => {
            state.metrics.rate_limited_requests.increment(1);
            warn!("Rate limit exceeded for {}", addr);
            Err(StatusCode::TOO_MANY_REQUESTS)
        }
    }
}

async fn metrics_middleware(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Response {
    let start = Instant::now();
    let method = request.method().clone();
    let path = request.uri().path().to_string();

    state.metrics.requests_total.increment(1);
    state.metrics.active_connections.increment(1.0);

    let response = next.run(request).await;

    let duration = start.elapsed();
    state.metrics.request_duration.record(duration.as_secs_f64());
    state.metrics.active_connections.decrement(1.0);

    debug!("{} {} - {:.3}ms", method, path, duration.as_millis());

    response
}

async fn semaphore_middleware(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let _permit = state.semaphore.acquire().await.map_err(|_| {
        error!("Failed to acquire semaphore permit");
        StatusCode::SERVICE_UNAVAILABLE
    })?;

    Ok(next.run(request).await)
}

// Handler implementations
async fn root_handler() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

async fn health_handler() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "version": env!("CARGO_PKG_VERSION"),
        "features": {
            "redis": cfg!(feature = "redis"),
            "metrics": cfg!(feature = "metrics"),
            "database": cfg!(feature = "database")
        }
    }))
}

#[derive(Serialize)]
struct StatusResponse {
    status: String,
    platforms: Vec<PlatformStatus>,
    cache_stats: CacheStats,
    performance: PerformanceStats,
    uptime_seconds: u64,
    version: String,
}

#[derive(Serialize)]
struct PlatformStatus {
    name: String,
    available: bool,
    video_count: usize,
    last_updated: Option<String>,
    cache_hit_ratio: f64,
}

#[derive(Serialize)]
struct CacheStats {
    backend: String,
    hit_ratio: f64,
    total_requests: u64,
}

#[derive(Serialize)]
struct PerformanceStats {
    active_connections: f64,
    avg_response_time_ms: f64,
    requests_per_second: f64,
}

async fn status_handler(State(state): State<AppState>) -> impl IntoResponse {
    let platforms = vec![
        Platform::Bilibili,
        Platform::YouTube,
        Platform::Douyin,
        Platform::Kuaishou,
    ];

    let mut platform_statuses = Vec::new();
    for platform in platforms {
        let key = cache_key_for_platform(platform);
        let available = state.cache_manager.backend.exists(&key).await.unwrap_or(false);

        platform_statuses.push(PlatformStatus {
            name: platform.as_str().to_string(),
            available,
            video_count: if available { 10 } else { 0 }, // Placeholder
            last_updated: if available {
                Some(chrono::Utc::now().to_rfc3339())
            } else {
                None
            },
            cache_hit_ratio: 0.85, // Placeholder
        });
    }

    Json(StatusResponse {
        status: "online".to_string(),
        platforms: platform_statuses,
        cache_stats: CacheStats {
            backend: if state.config.redis_url.is_some() { "redis" } else { "memory" }.to_string(),
            hit_ratio: 0.85,
            total_requests: 1000, // Placeholder
        },
        performance: PerformanceStats {
            active_connections: 0.0, // Would need to track this
            avg_response_time_ms: 50.0,
            requests_per_second: 100.0,
        },
        uptime_seconds: 3600, // Placeholder
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

#[derive(Deserialize)]
struct VideoQuery {
    limit: Option<usize>,
    sort: Option<String>,
    fresh: Option<bool>,
}

async fn get_videos_handler(
    State(state): State<AppState>,
    Query(params): Query<VideoQuery>,
) -> impl IntoResponse {
    let start = Instant::now();

    let cache_key = "all_videos";
    let limit = params.limit.unwrap_or(50).min(200); // Cap at 200

    let videos = if params.fresh.unwrap_or(false) {
        // Force fresh fetch
        state.metrics.cache_misses.increment(1);
        fetch_all_videos_fresh(&state).await
    } else {
        // Try cache first
        match state.cache_manager.get_or_set(cache_key, || async {
            fetch_all_videos_fresh(&state).await
        }).await {
            Ok(videos) => {
                state.metrics.cache_hits.increment(1);
                videos
            }
            Err(e) => {
                error!("Failed to fetch videos: {}", e);
                state.metrics.cache_misses.increment(1);
                Vec::new()
            }
        }
    };

    let videos: Vec<_> = videos.into_iter().take(limit).collect();

    state.metrics.video_fetch_duration.record(start.elapsed().as_secs_f64());

    Json(serde_json::json!({
        "total": videos.len(),
        "videos": videos,
        "cached": !params.fresh.unwrap_or(false),
        "fetch_time_ms": start.elapsed().as_millis(),
    }))
}

async fn get_platform_videos_handler(
    State(state): State<AppState>,
    Path(platform_str): Path<String>,
) -> impl IntoResponse {
    let platform = match platform_str.as_str() {
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
            ).into_response();
        }
    };

    let cache_key = cache_key_for_platform(platform);

    let videos = match state.cache_manager.get_or_set(&cache_key, || async {
        fetch_platform_videos_fresh(&state, platform).await
    }).await {
        Ok(videos) => {
            state.metrics.cache_hits.increment(1);
            videos
        }
        Err(e) => {
            error!("Failed to fetch {} videos: {}", platform_str, e);
            state.metrics.cache_misses.increment(1);
            Vec::new()
        }
    };

    Json(serde_json::json!({
        "platform": platform_str,
        "total": videos.len(),
        "videos": videos,
    })).into_response()
}

async fn generate_rss_handler(State(state): State<AppState>) -> Response {
    generate_rss_response(&state, &[Platform::Bilibili, Platform::YouTube, Platform::Douyin]).await
}

async fn generate_custom_rss_handler(
    State(state): State<AppState>,
    Path(platforms_str): Path<String>,
) -> Response {
    let platforms: Vec<Platform> = platforms_str
        .split(',')
        .filter_map(|p| match p {
            "bilibili" => Some(Platform::Bilibili),
            "youtube" => Some(Platform::YouTube),
            "douyin" => Some(Platform::Douyin),
            "kuaishou" => Some(Platform::Kuaishou),
            _ => None,
        })
        .collect();

    if platforms.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "No valid platforms specified"
            })),
        ).into_response();
    }

    generate_rss_response(&state, &platforms).await
}

async fn generate_rss_response(state: &AppState, platforms: &[Platform]) -> Response {
    let cache_key = cache_key_for_rss(platforms);

    match state.cache_manager.get_with_etag::<String>(&cache_key).await {
        Ok(Some((rss_content, etag))) => {
            let mut headers = HeaderMap::new();
            headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("application/rss+xml; charset=utf-8"));
            headers.insert(header::ETAG, HeaderValue::from_str(&etag).unwrap());
            headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("public, max-age=300"));

            (StatusCode::OK, headers, rss_content).into_response()
        }
        _ => {
            // Generate fresh RSS
            match generate_fresh_rss(state, platforms).await {
                Ok(rss_content) => {
                    let etag = format!("\"{}\"", seahash::hash(rss_content.as_bytes()));

                    // Cache the RSS content
                    if let Err(e) = state.cache_manager.set_with_ttl(&cache_key, rss_content.clone(), 300).await {
                        warn!("Failed to cache RSS: {}", e);
                    }

                    let mut headers = HeaderMap::new();
                    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("application/rss+xml; charset=utf-8"));
                    headers.insert(header::ETAG, HeaderValue::from_str(&etag).unwrap());
                    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("public, max-age=300"));

                    (StatusCode::OK, headers, rss_content).into_response()
                }
                Err(e) => {
                    error!("Failed to generate RSS: {}", e);
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({
                            "error": "Failed to generate RSS feed"
                        })),
                    ).into_response()
                }
            }
        }
    }
}

async fn transcribe_handler(
    State(_state): State<AppState>,
    Json(req): Json<serde_json::Value>,
) -> impl IntoResponse {
    // Placeholder for transcription
    Json(serde_json::json!({
        "status": "pending",
        "message": "Transcription service integration pending",
        "request": req,
    }))
}

async fn refresh_cache_handler(State(state): State<AppState>) -> impl IntoResponse {
    info!("Manual cache refresh requested");

    // Invalidate all platform caches
    let platforms = [Platform::Bilibili, Platform::YouTube, Platform::Douyin, Platform::Kuaishou];
    for platform in platforms {
        let cache_key = cache_key_for_platform(platform);
        if let Err(e) = state.cache_manager.invalidate(&cache_key).await {
            warn!("Failed to invalidate cache for {}: {}", platform.as_str(), e);
        }
    }

    // Invalidate RSS caches
    if let Err(e) = state.cache_manager.invalidate_pattern("rss:*").await {
        warn!("Failed to invalidate RSS caches: {}", e);
    }

    Json(serde_json::json!({
        "status": "success",
        "message": "Cache invalidated successfully",
        "timestamp": chrono::Utc::now().to_rfc3339(),
    }))
}

async fn invalidate_cache_handler(
    State(state): State<AppState>,
    Json(req): Json<serde_json::Value>,
) -> impl IntoResponse {
    if let Some(key) = req.get("key").and_then(|k| k.as_str()) {
        match state.cache_manager.invalidate(key).await {
            Ok(_) => Json(serde_json::json!({
                "status": "success",
                "message": format!("Cache key '{}' invalidated", key)
            })),
            Err(e) => Json(serde_json::json!({
                "status": "error",
                "message": format!("Failed to invalidate key '{}': {}", key, e)
            })),
        }
    } else {
        Json(serde_json::json!({
            "status": "error",
            "message": "Missing 'key' field in request"
        }))
    }
}

async fn metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    if !state.config.enable_metrics {
        return (StatusCode::NOT_FOUND, "Metrics disabled").into_response();
    }

    // Export Prometheus metrics
    let encoder = prometheus::TextEncoder::new();
    let metric_families = prometheus::gather();

    match encoder.encode_to_string(&metric_families) {
        Ok(metrics) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
            metrics,
        ).into_response(),
        Err(e) => {
            error!("Failed to encode metrics: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, "Failed to encode metrics").into_response()
        }
    }
}

// Helper functions
async fn fetch_all_videos_fresh(state: &AppState) -> Result<Vec<ExtractedVideo>> {
    let all_videos = state.extractor.extract_all().await;
    Ok(all_videos.into_values().flatten().collect())
}

async fn fetch_platform_videos_fresh(state: &AppState, platform: Platform) -> Result<Vec<ExtractedVideo>> {
    let all_videos = state.extractor.extract_all().await;
    Ok(all_videos.get(&platform).cloned().unwrap_or_default())
}

async fn generate_fresh_rss(state: &AppState, platforms: &[Platform]) -> Result<String> {
    let mut all_videos = Vec::new();

    for &platform in platforms {
        if let Ok(videos) = fetch_platform_videos_fresh(state, platform).await {
            all_videos.extend(videos);
        }
    }

    // Convert to VideoInfo
    let video_infos: Vec<VideoInfo> = all_videos
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
    rss_gen.generate_feed(&video_infos)
}

// Main entry point
pub async fn run_server(config: ServerConfig) -> Result<()> {
    let server = VideoRssServer::new(config).await?;
    server.run().await
}