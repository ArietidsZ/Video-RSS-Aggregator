use crate::{
    embeddings::{EmbeddingGenerator, EmbeddingConfig},
    error::VideoRssError,
    fast_io::FastIO,
    monitoring::{MonitoringSystem, Profiler},
    realtime::{RealtimeManager, ConnectionFilters},
    summarizer::{Summarizer, SummarizerConfig},
    tiered_cache::TieredCache,
    vector_db::{VectorDatabase, SearchFilters},
    whisper_candle::{WhisperTranscriber, WhisperConfig},
    Result,
};
use axum::{
    extract::{Path, Query, State, Multipart},
    http::StatusCode,
    response::{IntoResponse, Response, Json},
    routing::{get, post, put, delete},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::{
    cors::CorsLayer,
    compression::CompressionLayer,
    limit::RequestBodyLimitLayer,
    timeout::TimeoutLayer,
    trace::TraceLayer,
};
use tracing::{debug, error, info, warn};

#[derive(Clone)]
pub struct ApiState {
    pub cache: Arc<TieredCache>,
    pub vector_db: Arc<VectorDatabase>,
    pub transcriber: Arc<WhisperTranscriber>,
    pub summarizer: Arc<Summarizer>,
    pub embeddings: Arc<EmbeddingGenerator>,
    pub realtime: Arc<RealtimeManager>,
    pub monitoring: Arc<MonitoringSystem>,
    pub fast_io: Arc<FastIO>,
}

pub fn create_api_router(state: ApiState) -> Router {
    Router::new()
        // Video endpoints
        .route("/api/v1/videos", get(list_videos).post(add_video))
        .route("/api/v1/videos/:id", get(get_video).put(update_video).delete(delete_video))
        .route("/api/v1/videos/:id/transcribe", post(transcribe_video))
        .route("/api/v1/videos/:id/summarize", post(summarize_video))
        .route("/api/v1/videos/search", post(search_videos))

        // Feed endpoints
        .route("/api/v1/feeds", get(list_feeds).post(create_feed))
        .route("/api/v1/feeds/:id", get(get_feed).put(update_feed).delete(delete_feed))
        .route("/api/v1/feeds/:id/refresh", post(refresh_feed))

        // RSS endpoints
        .route("/api/v1/rss/generate", post(generate_rss))
        .route("/api/v1/rss/:platform", get(get_platform_rss))

        // Real-time endpoints
        .route("/api/v1/events", get(sse_events))
        .route("/api/v1/websub/subscribe", post(websub_subscribe))
        .route("/api/v1/websub/publish", post(websub_publish))

        // Search endpoints
        .route("/api/v1/search", post(unified_search))
        .route("/api/v1/search/semantic", post(semantic_search))
        .route("/api/v1/search/hybrid", post(hybrid_search))

        // Upload endpoints
        .route("/api/v1/upload/video", post(upload_video))
        .route("/api/v1/upload/audio", post(upload_audio))

        // Analytics endpoints
        .route("/api/v1/analytics/stats", get(get_stats))
        .route("/api/v1/analytics/metrics", get(get_metrics))

        // Health and monitoring
        .route("/health", get(health_check))
        .route("/metrics", get(prometheus_metrics))

        // Apply middleware
        .layer(CorsLayer::permissive())
        .layer(CompressionLayer::new())
        .layer(RequestBodyLimitLayer::new(100 * 1024 * 1024))  // 100MB
        .layer(TimeoutLayer::new(std::time::Duration::from_secs(30)))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

// Video endpoints
async fn list_videos(
    State(state): State<ApiState>,
    Query(params): Query<ListVideosParams>,
) -> Result<Json<ListVideosResponse>> {
    let mut profiler = Profiler::new("list_videos");

    let cache_key = format!("videos:list:{:?}", params);

    // Try cache first
    if let Some(cached) = state.cache.get::<ListVideosResponse>(&cache_key).await? {
        state.monitoring.record_cache_hit("api");
        return Ok(Json(cached));
    }

    state.monitoring.record_cache_miss("api");
    profiler.checkpoint("cache_check");

    // Query database
    let videos = vec![];  // Would query actual database

    profiler.checkpoint("database_query");

    let response = ListVideosResponse {
        videos,
        total: 0,
        page: params.page.unwrap_or(1),
        per_page: params.per_page.unwrap_or(20),
    };

    // Cache response
    state.cache.set(&cache_key, &response, Some(std::time::Duration::from_secs(300))).await?;

    let profile = profiler.finish();
    debug!("Profile: {:?}", profile);

    Ok(Json(response))
}

async fn add_video(
    State(state): State<ApiState>,
    Json(video): Json<AddVideoRequest>,
) -> Result<Json<AddVideoResponse>> {
    let span = state.monitoring.start_span("add_video");

    // Generate embedding for the video
    let embedding = state.embeddings.generate(&video.description).await?;

    // Store in vector database
    state.vector_db.insert_document(crate::vector_db::VideoDocument {
        id: uuid::Uuid::new_v4().to_string(),
        title: video.title.clone(),
        description: video.description.clone(),
        transcript: String::new(),
        embedding: embedding.embedding,
        platform: video.platform.clone(),
        author: video.author.clone(),
        url: video.url.clone(),
        upload_timestamp: chrono::Utc::now().timestamp(),
        view_count: 0,
        duration_seconds: 0,
        tags: video.tags.clone(),
    }).await?;

    // Broadcast real-time update
    state.realtime.broadcast_video_update(crate::extractor::ExtractedVideo {
        id: uuid::Uuid::new_v4().to_string(),
        title: video.title,
        platform: crate::types::Platform::YouTube,
        author: video.author,
        url: video.url,
        description: video.description,
        view_count: Some(0),
        like_count: Some(0),
        duration: None,
        upload_date: chrono::Utc::now().to_rfc3339(),
        thumbnail: None,
        tags: video.tags,
        data_source: "api".to_string(),
        legal_compliance: "user_uploaded".to_string(),
        extraction_method: "manual".to_string(),
        extracted_at: chrono::Utc::now(),
    }).await?;

    Ok(Json(AddVideoResponse {
        id: uuid::Uuid::new_v4().to_string(),
        message: "Video added successfully".to_string(),
    }))
}

async fn transcribe_video(
    State(state): State<ApiState>,
    Path(id): Path<String>,
) -> Result<Json<TranscribeResponse>> {
    let start = std::time::Instant::now();

    // Get video file path (simplified)
    let video_path = std::path::PathBuf::from(format!("./videos/{}.mp4", id));

    // Transcribe using Whisper
    let result = state.transcriber.transcribe_audio(&video_path).await?;

    state.monitoring.record_transcription("whisper", start.elapsed().as_millis() as f64, true);

    Ok(Json(TranscribeResponse {
        video_id: id,
        text: result.text,
        segments: result.segments.into_iter().map(|s| TranscriptSegment {
            start: s.start,
            end: s.end,
            text: s.text,
        }).collect(),
        language: result.language,
        processing_time_ms: result.processing_time_ms,
    }))
}

async fn summarize_video(
    State(state): State<ApiState>,
    Path(id): Path<String>,
    Json(req): Json<SummarizeRequest>,
) -> Result<Json<SummarizeResponse>> {
    // Get transcript
    let transcript = req.transcript.unwrap_or_else(|| "Sample transcript".to_string());

    // Generate summary
    let result = state.summarizer.summarize(&transcript).await?;

    Ok(Json(SummarizeResponse {
        video_id: id,
        summary: result.summary,
        key_points: result.key_points,
        topics: result.topics,
        sentiment: format!("{:?}", result.sentiment),
    }))
}

async fn search_videos(
    State(state): State<ApiState>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>> {
    // Generate embedding for query
    let embedding = if req.semantic {
        Some(state.embeddings.generate(&req.query).await?.embedding)
    } else {
        None
    };

    // Perform search
    let results = state.vector_db.hybrid_search(
        &req.query,
        embedding,
        Some(SearchFilters {
            platform: req.platform,
            author: None,
            min_views: req.min_views,
            max_duration_seconds: None,
            uploaded_after: req.uploaded_after,
            tags: req.tags,
        }),
    ).await?;

    Ok(Json(SearchResponse {
        query: req.query,
        results: results.into_iter().map(|r| SearchResult {
            id: r.document.id,
            title: r.document.title,
            description: r.document.description,
            url: r.document.url,
            score: r.score,
            highlights: r.highlights,
        }).collect(),
        total: 0,
    }))
}

// Real-time SSE endpoint
async fn sse_events(
    State(state): State<ApiState>,
    Query(params): Query<SseParams>,
) -> impl IntoResponse {
    let filters = ConnectionFilters::default();
    state.realtime.create_sse_stream(filters).await
}

// Health check
async fn health_check(State(state): State<ApiState>) -> impl IntoResponse {
    let metrics = state.monitoring.get_system_metrics().await;

    Json(serde_json::json!({
        "status": "healthy",
        "version": env!("CARGO_PKG_VERSION"),
        "cpu_usage": metrics.cpu_usage,
        "memory_usage": metrics.memory_usage,
        "uptime": 0,
    }))
}

// Prometheus metrics
async fn prometheus_metrics(State(state): State<ApiState>) -> impl IntoResponse {
    state.monitoring.export_prometheus_metrics()
}

// Request/Response types
#[derive(Deserialize)]
struct ListVideosParams {
    page: Option<u32>,
    per_page: Option<u32>,
    platform: Option<String>,
    author: Option<String>,
}

#[derive(Serialize)]
struct ListVideosResponse {
    videos: Vec<VideoInfo>,
    total: usize,
    page: u32,
    per_page: u32,
}

#[derive(Serialize)]
struct VideoInfo {
    id: String,
    title: String,
    platform: String,
    author: String,
    url: String,
    view_count: i64,
}

#[derive(Deserialize)]
struct AddVideoRequest {
    title: String,
    description: String,
    platform: String,
    author: String,
    url: String,
    tags: Vec<String>,
}

#[derive(Serialize)]
struct AddVideoResponse {
    id: String,
    message: String,
}

#[derive(Serialize)]
struct TranscribeResponse {
    video_id: String,
    text: String,
    segments: Vec<TranscriptSegment>,
    language: String,
    processing_time_ms: u64,
}

#[derive(Serialize)]
struct TranscriptSegment {
    start: f64,
    end: f64,
    text: String,
}

#[derive(Deserialize)]
struct SummarizeRequest {
    transcript: Option<String>,
}

#[derive(Serialize)]
struct SummarizeResponse {
    video_id: String,
    summary: String,
    key_points: Vec<String>,
    topics: Vec<String>,
    sentiment: String,
}

#[derive(Deserialize)]
struct SearchRequest {
    query: String,
    semantic: bool,
    platform: Option<String>,
    min_views: Option<i64>,
    uploaded_after: Option<i64>,
    tags: Option<Vec<String>>,
}

#[derive(Serialize)]
struct SearchResponse {
    query: String,
    results: Vec<SearchResult>,
    total: usize,
}

#[derive(Serialize)]
struct SearchResult {
    id: String,
    title: String,
    description: String,
    url: String,
    score: f32,
    highlights: Vec<String>,
}

#[derive(Deserialize)]
struct SseParams {
    platforms: Option<String>,
    authors: Option<String>,
}

// Stub implementations for remaining endpoints
async fn get_video(Path(id): Path<String>) -> impl IntoResponse {
    Json(serde_json::json!({"id": id}))
}

async fn update_video(Path(id): Path<String>) -> impl IntoResponse {
    Json(serde_json::json!({"id": id, "updated": true}))
}

async fn delete_video(Path(id): Path<String>) -> impl IntoResponse {
    Json(serde_json::json!({"id": id, "deleted": true}))
}

async fn list_feeds() -> impl IntoResponse {
    Json(serde_json::json!({"feeds": []}))
}

async fn create_feed() -> impl IntoResponse {
    Json(serde_json::json!({"created": true}))
}

async fn get_feed(Path(id): Path<String>) -> impl IntoResponse {
    Json(serde_json::json!({"id": id}))
}

async fn update_feed(Path(id): Path<String>) -> impl IntoResponse {
    Json(serde_json::json!({"id": id, "updated": true}))
}

async fn delete_feed(Path(id): Path<String>) -> impl IntoResponse {
    Json(serde_json::json!({"id": id, "deleted": true}))
}

async fn refresh_feed(Path(id): Path<String>) -> impl IntoResponse {
    Json(serde_json::json!({"id": id, "refreshed": true}))
}

async fn generate_rss() -> impl IntoResponse {
    Json(serde_json::json!({"rss": "generated"}))
}

async fn get_platform_rss(Path(platform): Path<String>) -> impl IntoResponse {
    Json(serde_json::json!({"platform": platform}))
}

async fn websub_subscribe() -> impl IntoResponse {
    Json(serde_json::json!({"subscribed": true}))
}

async fn websub_publish() -> impl IntoResponse {
    Json(serde_json::json!({"published": true}))
}

async fn unified_search() -> impl IntoResponse {
    Json(serde_json::json!({"results": []}))
}

async fn semantic_search() -> impl IntoResponse {
    Json(serde_json::json!({"results": []}))
}

async fn hybrid_search() -> impl IntoResponse {
    Json(serde_json::json!({"results": []}))
}

async fn upload_video() -> impl IntoResponse {
    Json(serde_json::json!({"uploaded": true}))
}

async fn upload_audio() -> impl IntoResponse {
    Json(serde_json::json!({"uploaded": true}))
}

async fn get_stats() -> impl IntoResponse {
    Json(serde_json::json!({"stats": {}}))
}

async fn get_metrics() -> impl IntoResponse {
    Json(serde_json::json!({"metrics": {}}))
}