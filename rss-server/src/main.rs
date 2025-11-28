pub mod config;
mod handlers;
mod models;
mod services;
mod cache;
mod dedup;
mod websub;
mod qrcode_gen;
mod platforms;
mod error;
mod metrics;
mod auth;

use axum::{
    extract::Extension,
    http::{header, HeaderValue, Method, StatusCode},
    middleware,
    response::Response,
    routing::{get, post},
    Router,
};
use std::{net::SocketAddr, sync::Arc, time::Duration};
use tower::ServiceBuilder;
use tower_http::{
    compression::CompressionLayer,
    cors::{Any, CorsLayer},
    limit::RequestBodyLimitLayer,
    services::ServeDir,
    timeout::TimeoutLayer,
    trace::{DefaultMakeSpan, DefaultOnRequest, DefaultOnResponse, TraceLayer},
};
use tracing::{info, Level};

use crate::{
    cache::CacheManager,
    config::AppConfig,
    dedup::DeduplicationService,
    services::{FeedService, SummarizationService},
    websub::WebSubHub,
    metrics::MetricsRecorder,
};

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<AppConfig>,
    pub cache: Arc<CacheManager>,
    pub dedup: Arc<DeduplicationService>,
    pub feed_service: Arc<FeedService>,
    pub summarization: Arc<SummarizationService>,
    pub websub: Arc<WebSubHub>,
    pub metrics: Arc<MetricsRecorder>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("rss_server=debug,tower_http=debug,axum=info")
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE)
        .json()
        .init();

    info!("Starting RSS Server");

    // Load configuration
    let config = Arc::new(AppConfig::from_env()?);

    // Initialize services
    let cache = Arc::new(CacheManager::new(&config.redis_url).await?);
    let dedup = Arc::new(DeduplicationService::new(1000000)); // 1M entries
    let feed_service = Arc::new(FeedService::new(config.clone(), cache.clone()).await?);
    let summarization = Arc::new(SummarizationService::new(config.clone()).await?);
    let websub = Arc::new(WebSubHub::new(config.clone()).await?);
    let metrics = Arc::new(MetricsRecorder::new());

    // Create app state
    let state = AppState {
        config: config.clone(),
        cache,
        dedup,
        feed_service,
        summarization,
        websub,
        metrics,
    };

    // Build router
    let app = create_router(state);

    // Start server
    let addr = SocketAddr::from(([0, 0, 0, 0], config.port));
    info!("RSS Server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;

    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await?;

    Ok(())
}

fn create_router(state: AppState) -> Router {
    // CORS configuration
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE])
        .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION])
        .max_age(Duration::from_secs(3600));

    // Build service layers
    let middleware_stack = ServiceBuilder::new()
        .layer(TraceLayer::new_for_http())
        .layer(CompressionLayer::new())
        .layer(cors)
        .layer(TimeoutLayer::new(Duration::from_secs(30)))
        .layer(RequestBodyLimitLayer::new(10 * 1024 * 1024)) // 10MB
        .layer(middleware::from_fn(auth::auth_middleware))
        .layer(Extension(state.clone()));

    // Define routes
    Router::new()
        // RSS endpoints
        .route("/rss/{channel_id}", get(handlers::rss::get_feed))
        .route("/rss/{channel_id}/items", get(handlers::rss::get_feed_items))
        .route("/rss/generate", post(handlers::rss::generate_feed))
        .route("/rss/batch", post(handlers::rss::batch_generate_feeds))

        // Platform-specific RSS
        .route("/rss/youtube/{channel_id}", get(handlers::platform::youtube_feed))
        .route("/rss/bilibili/{uid}", get(handlers::platform::bilibili_feed))
        .route("/rss/douyin/{user_id}", get(handlers::platform::douyin_feed))
        .route("/rss/kuaishou/{user_id}", get(handlers::platform::kuaishou_feed))

        // Summarization endpoints
        .route("/summarize", post(handlers::summary::summarize_content))
        .route("/summarize/batch", post(handlers::summary::batch_summarize))
        .route("/summarize/status/{job_id}", get(handlers::summary::get_job_status))

        // WebSub endpoints
        .route("/websub/subscribe", post(handlers::websub::subscribe))
        .route("/websub/unsubscribe", post(handlers::websub::unsubscribe))
        .route("/websub/publish", post(handlers::websub::publish))
        .route("/websub/callback/{subscription_id}",
            get(handlers::websub::verify_subscription)
            .post(handlers::websub::receive_update))

        // QR code generation
        .route("/qr/{feed_id}", get(handlers::qr::generate_qr_code))

        // Search and discovery
        .route("/search", get(handlers::search::search_feeds))
        .route("/trending", get(handlers::search::trending_feeds))
        .route("/recommendations/{user_id}", get(handlers::search::recommendations))

        // Management endpoints
        .route("/feeds", get(handlers::management::list_feeds))
        .route("/feeds/{feed_id}",
            get(handlers::management::get_feed)
            .put(handlers::management::update_feed)
            .delete(handlers::management::delete_feed))

        // Health and metrics
        .route("/health", get(handlers::health::health_check))
        .route("/ready", get(handlers::health::readiness_check))
        .route("/metrics", get(handlers::metrics::prometheus_metrics))

        // Static files
        .nest_service("/static", ServeDir::new("static"))

        // Apply middleware
        .layer(middleware_stack)

        // Fallback
        .fallback(handlers::not_found)
}

mod handlers {
    use axum::{
        extract::{Path, Query, State},
        http::StatusCode,
        response::{IntoResponse, Response},
        Json,
    };
    use serde::{Deserialize, Serialize};
    use std::collections::HashMap;

    use crate::AppState;

    pub async fn not_found() -> impl IntoResponse {
        (StatusCode::NOT_FOUND, "Not Found")
    }

    pub mod rss {
        use super::*;
        use crate::models::{Feed, FeedItem};

        #[derive(Deserialize)]
        pub struct GenerateFeedRequest {
            pub url: String,
            pub title: Option<String>,
            pub description: Option<String>,
            pub language: Option<String>,
            pub include_summaries: bool,
            pub max_items: Option<usize>,
        }

        #[derive(Serialize)]
        pub struct GenerateFeedResponse {
            pub feed_id: String,
            pub feed_url: String,
            pub qr_code_url: String,
            pub items_count: usize,
            pub cache_ttl: u64,
        }

        pub async fn get_feed(
            State(state): State<AppState>,
            Path(channel_id): Path<String>,
        ) -> Result<Response, StatusCode> {
            // Check cache first
            if let Some(cached) = state.cache.get_feed(&channel_id).await {
                state.metrics.record_cache_hit("feed");

                return Ok((
                    StatusCode::OK,
                    [(header::CONTENT_TYPE, "application/rss+xml")],
                    cached,
                ).into_response());
            }

            state.metrics.record_cache_miss("feed");

            // Generate feed
            match state.feed_service.generate_feed(&channel_id).await {
                Ok(feed_xml) => {
                    // Cache for 5 minutes
                    state.cache.set_feed(&channel_id, &feed_xml, 300).await;

                    state.metrics.record_feed_generation(&channel_id);

                    Ok((
                        StatusCode::OK,
                        [(header::CONTENT_TYPE, "application/rss+xml")],
                        feed_xml,
                    ).into_response())
                }
                Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
            }
        }

        pub async fn get_feed_items(
            State(state): State<AppState>,
            Path(channel_id): Path<String>,
            Query(params): Query<HashMap<String, String>>,
        ) -> Result<Json<Vec<FeedItem>>, StatusCode> {
            let limit = params
                .get("limit")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(20);

            match state.feed_service.get_feed_items(&channel_id, limit).await {
                Ok(items) => Ok(Json(items)),
                Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
            }
        }

        pub async fn generate_feed(
            State(state): State<AppState>,
            Json(request): Json<GenerateFeedRequest>,
        ) -> Result<Json<GenerateFeedResponse>, StatusCode> {
            // Check for duplicate
            let feed_hash = state.dedup.compute_hash(&request.url);

            if let Some(existing_id) = state.dedup.check_duplicate(&feed_hash).await {
                state.metrics.record_dedup_hit();

                return Ok(Json(GenerateFeedResponse {
                    feed_id: existing_id,
                    feed_url: format!("{}/rss/{}", state.config.base_url, existing_id),
                    qr_code_url: format!("{}/qr/{}", state.config.base_url, existing_id),
                    items_count: 0,
                    cache_ttl: 300,
                }));
            }

            // Generate new feed
            match state.feed_service.create_feed(
                &request.url,
                request.title,
                request.description,
                request.include_summaries,
            ).await {
                Ok(feed) => {
                    state.dedup.add_entry(&feed_hash, &feed.id).await;
                    state.metrics.record_feed_creation(&feed.id);

                    Ok(Json(GenerateFeedResponse {
                        feed_id: feed.id.clone(),
                        feed_url: format!("{}/rss/{}", state.config.base_url, feed.id),
                        qr_code_url: format!("{}/qr/{}", state.config.base_url, feed.id),
                        items_count: feed.items.len(),
                        cache_ttl: 300,
                    }))
                }
                Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
            }
        }

        pub async fn batch_generate_feeds(
            State(state): State<AppState>,
            Json(requests): Json<Vec<GenerateFeedRequest>>,
        ) -> Result<Json<Vec<GenerateFeedResponse>>, StatusCode> {
            use futures::future::join_all;

            let futures = requests.into_iter().map(|req| {
                let state = state.clone();
                async move {
                    generate_feed(State(state), Json(req)).await
                }
            });

            let results = join_all(futures).await;

            let responses: Vec<GenerateFeedResponse> = results
                .into_iter()
                .filter_map(|r| r.ok())
                .map(|Json(r)| r)
                .collect();

            Ok(Json(responses))
        }
    }

    pub mod platform {
        use super::*;

        pub async fn youtube_feed(
            State(state): State<AppState>,
            Path(channel_id): Path<String>,
        ) -> Result<Response, StatusCode> {
            state.feed_service
                .generate_youtube_feed(&channel_id)
                .await
                .map(|xml| {
                    (
                        StatusCode::OK,
                        [(header::CONTENT_TYPE, "application/rss+xml")],
                        xml,
                    ).into_response()
                })
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
        }

        pub async fn bilibili_feed(
            State(state): State<AppState>,
            Path(uid): Path<String>,
        ) -> Result<Response, StatusCode> {
            state.feed_service
                .generate_bilibili_feed(&uid)
                .await
                .map(|xml| {
                    (
                        StatusCode::OK,
                        [(header::CONTENT_TYPE, "application/rss+xml")],
                        xml,
                    ).into_response()
                })
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
        }

        pub async fn douyin_feed(
            State(state): State<AppState>,
            Path(user_id): Path<String>,
        ) -> Result<Response, StatusCode> {
            state.feed_service
                .generate_douyin_feed(&user_id)
                .await
                .map(|xml| {
                    (
                        StatusCode::OK,
                        [(header::CONTENT_TYPE, "application/rss+xml")],
                        xml,
                    ).into_response()
                })
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
        }

        pub async fn kuaishou_feed(
            State(state): State<AppState>,
            Path(user_id): Path<String>,
        ) -> Result<Response, StatusCode> {
            state.feed_service
                .generate_kuaishou_feed(&user_id)
                .await
                .map(|xml| {
                    (
                        StatusCode::OK,
                        [(header::CONTENT_TYPE, "application/rss+xml")],
                        xml,
                    ).into_response()
                })
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
        }
    }

    pub mod summary {
        use super::*;

        #[derive(Deserialize)]
        pub struct SummarizeRequest {
            pub content: String,
            pub style: Option<String>,
            pub max_length: Option<usize>,
        }

        #[derive(Serialize)]
        pub struct SummarizeResponse {
            pub summary: String,
            pub word_count: usize,
            pub processing_time_ms: u64,
        }

        pub async fn summarize_content(
            State(state): State<AppState>,
            Json(request): Json<SummarizeRequest>,
        ) -> Result<Json<SummarizeResponse>, StatusCode> {
            let start = std::time::Instant::now();

            match state.summarization.summarize(
                &request.content,
                request.style.as_deref(),
                request.max_length,
            ).await {
                Ok(summary) => {
                    let processing_time = start.elapsed().as_millis() as u64;
                    state.metrics.record_summarization_time(processing_time);

                    Ok(Json(SummarizeResponse {
                        word_count: summary.split_whitespace().count(),
                        summary,
                        processing_time_ms: processing_time,
                    }))
                }
                Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
            }
        }

        pub async fn batch_summarize(
            State(state): State<AppState>,
            Json(requests): Json<Vec<SummarizeRequest>>,
        ) -> Result<Json<Vec<SummarizeResponse>>, StatusCode> {
            use futures::future::join_all;

            let futures = requests.into_iter().map(|req| {
                let state = state.clone();
                async move {
                    summarize_content(State(state), Json(req)).await
                }
            });

            let results = join_all(futures).await;

            let responses: Vec<SummarizeResponse> = results
                .into_iter()
                .filter_map(|r| r.ok())
                .map(|Json(r)| r)
                .collect();

            Ok(Json(responses))
        }

        pub async fn get_job_status(
            State(state): State<AppState>,
            Path(job_id): Path<String>,
        ) -> Result<Json<serde_json::Value>, StatusCode> {
            match state.summarization.get_job_status(&job_id).await {
                Some(status) => Ok(Json(status)),
                None => Err(StatusCode::NOT_FOUND),
            }
        }
    }

    pub mod websub {
        use super::*;

        #[derive(Deserialize)]
        pub struct SubscribeRequest {
            pub hub_url: String,
            pub topic_url: String,
            pub callback_url: String,
        }

        pub async fn subscribe(
            State(state): State<AppState>,
            Json(request): Json<SubscribeRequest>,
        ) -> Result<StatusCode, StatusCode> {
            state.websub
                .subscribe(&request.hub_url, &request.topic_url, &request.callback_url)
                .await
                .map(|_| StatusCode::ACCEPTED)
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
        }

        pub async fn unsubscribe(
            State(state): State<AppState>,
            Json(request): Json<SubscribeRequest>,
        ) -> Result<StatusCode, StatusCode> {
            state.websub
                .unsubscribe(&request.hub_url, &request.topic_url, &request.callback_url)
                .await
                .map(|_| StatusCode::ACCEPTED)
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
        }

        pub async fn publish(
            State(state): State<AppState>,
            Json(content): Json<serde_json::Value>,
        ) -> Result<StatusCode, StatusCode> {
            state.websub
                .publish_update(content)
                .await
                .map(|_| StatusCode::NO_CONTENT)
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
        }

        pub async fn verify_subscription(
            State(state): State<AppState>,
            Path(subscription_id): Path<String>,
            Query(params): Query<HashMap<String, String>>,
        ) -> Result<String, StatusCode> {
            if let Some(challenge) = params.get("hub.challenge") {
                if state.websub.verify_subscription(&subscription_id, challenge).await {
                    return Ok(challenge.clone());
                }
            }
            Err(StatusCode::NOT_FOUND)
        }

        pub async fn receive_update(
            State(state): State<AppState>,
            Path(subscription_id): Path<String>,
            body: String,
        ) -> StatusCode {
            if state.websub.process_update(&subscription_id, body).await.is_ok() {
                StatusCode::NO_CONTENT
            } else {
                StatusCode::BAD_REQUEST
            }
        }
    }

    pub mod qr {
        use super::*;

        pub async fn generate_qr_code(
            State(state): State<AppState>,
            Path(feed_id): Path<String>,
        ) -> Result<Response, StatusCode> {
            let feed_url = format!("{}/rss/{}", state.config.base_url, feed_id);

            match crate::qrcode_gen::generate_qr_code(&feed_url) {
                Ok(image_data) => {
                    state.metrics.record_qr_generation();

                    Ok((
                        StatusCode::OK,
                        [(header::CONTENT_TYPE, "image/png")],
                        image_data,
                    ).into_response())
                }
                Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
            }
        }
    }

    pub mod search {
        use super::*;

        pub async fn search_feeds(
            State(state): State<AppState>,
            Query(params): Query<HashMap<String, String>>,
        ) -> Result<Json<Vec<serde_json::Value>>, StatusCode> {
            let query = params.get("q").map(|s| s.as_str()).unwrap_or("");

            match state.feed_service.search_feeds(query).await {
                Ok(results) => Ok(Json(results)),
                Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
            }
        }

        pub async fn trending_feeds(
            State(state): State<AppState>,
        ) -> Result<Json<Vec<serde_json::Value>>, StatusCode> {
            match state.feed_service.get_trending_feeds().await {
                Ok(feeds) => Ok(Json(feeds)),
                Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
            }
        }

        pub async fn recommendations(
            State(state): State<AppState>,
            Path(user_id): Path<String>,
        ) -> Result<Json<Vec<serde_json::Value>>, StatusCode> {
            match state.feed_service.get_recommendations(&user_id).await {
                Ok(feeds) => Ok(Json(feeds)),
                Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
            }
        }
    }

    pub mod management {
        use super::*;

        pub async fn list_feeds(
            State(state): State<AppState>,
        ) -> Result<Json<Vec<serde_json::Value>>, StatusCode> {
            match state.feed_service.list_all_feeds().await {
                Ok(feeds) => Ok(Json(feeds)),
                Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
            }
        }

        pub async fn get_feed(
            State(state): State<AppState>,
            Path(feed_id): Path<String>,
        ) -> Result<Json<serde_json::Value>, StatusCode> {
            match state.feed_service.get_feed_details(&feed_id).await {
                Ok(Some(feed)) => Ok(Json(feed)),
                Ok(None) => Err(StatusCode::NOT_FOUND),
                Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
            }
        }

        pub async fn update_feed(
            State(state): State<AppState>,
            Path(feed_id): Path<String>,
            Json(updates): Json<serde_json::Value>,
        ) -> Result<StatusCode, StatusCode> {
            match state.feed_service.update_feed(&feed_id, updates).await {
                Ok(_) => Ok(StatusCode::NO_CONTENT),
                Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
            }
        }

        pub async fn delete_feed(
            State(state): State<AppState>,
            Path(feed_id): Path<String>,
        ) -> Result<StatusCode, StatusCode> {
            match state.feed_service.delete_feed(&feed_id).await {
                Ok(_) => {
                    state.cache.invalidate_feed(&feed_id).await;
                    Ok(StatusCode::NO_CONTENT)
                }
                Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
            }
        }
    }

    pub mod health {
        use super::*;

        #[derive(Serialize)]
        pub struct HealthStatus {
            pub status: String,
            pub version: String,
            pub uptime: u64,
            pub services: HashMap<String, bool>,
        }

        pub async fn health_check(
            State(state): State<AppState>,
        ) -> Json<HealthStatus> {
            let mut services = HashMap::new();

            // Check service health
            services.insert("cache".to_string(), state.cache.is_healthy().await);
            services.insert("database".to_string(), state.feed_service.is_healthy().await);
            services.insert("websub".to_string(), state.websub.is_healthy().await);

            Json(HealthStatus {
                status: if services.values().all(|&v| v) { "healthy" } else { "degraded" }.to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                uptime: state.metrics.get_uptime_seconds(),
                services,
            })
        }

        pub async fn readiness_check(
            State(state): State<AppState>,
        ) -> StatusCode {
            if state.cache.is_healthy().await && state.feed_service.is_healthy().await {
                StatusCode::OK
            } else {
                StatusCode::SERVICE_UNAVAILABLE
            }
        }
    }

    pub mod metrics {
        use super::*;

        pub async fn prometheus_metrics(
            State(state): State<AppState>,
        ) -> Result<String, StatusCode> {
            Ok(state.metrics.export_prometheus())
        }
    }
}