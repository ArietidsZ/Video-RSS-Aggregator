mod cache_manager;
mod partitioner;
mod replication;
mod consistency;
mod cache_warmer;
mod expiration;

use anyhow::Result;
use axum::{
    routing::{get, post, put, delete},
    Router,
};
use std::net::SocketAddr;
use tower_http::{compression::CompressionLayer, cors::CorsLayer, trace::TraceLayer};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use cache_manager::CacheManager;
use cache_warmer::CacheWarmer;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .json()
        .init();

    info!("Starting Cache Manager Service");

    // Initialize cache manager
    let cache_manager = CacheManager::new().await?;
    let cache_manager_clone = cache_manager.clone();

    // Initialize cache warmer
    let cache_warmer = CacheWarmer::new(cache_manager.clone());
    tokio::spawn(async move {
        cache_warmer.start().await;
    });

    // Build router
    let app = Router::new()
        // Cache operations
        .route("/cache/{key}", get(handlers::get_cached))
        .route("/cache/{key}", put(handlers::set_cached))
        .route("/cache/{key}", delete(handlers::delete_cached))
        .route("/cache/batch", post(handlers::batch_get))

        // Partitioning info
        .route("/partition/{key}", get(handlers::get_partition_info))
        .route("/shards", get(handlers::list_shards))

        // Cache warming
        .route("/warm", post(handlers::warm_cache))
        .route("/preload", post(handlers::preload_data))

        // Replication
        .route("/replicate", post(handlers::replicate_data))
        .route("/consistency/{key}", get(handlers::check_consistency))

        // Metrics
        .route("/metrics", get(handlers::metrics))
        .route("/health", get(handlers::health))
        .route("/stats", get(handlers::stats))

        // Management
        .route("/expire", post(handlers::expire_keys))
        .route("/cleanup", post(handlers::cleanup))
        .route("/backup", post(handlers::backup))

        .layer(CompressionLayer::new())
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(cache_manager_clone);

    // Start server
    let addr = SocketAddr::from(([0, 0, 0, 0], 8090));
    info!("Cache Manager listening on {}", addr);

    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await?;

    Ok(())
}

mod handlers {
    use super::*;
    use axum::{
        extract::{Path, State, Json},
        response::{IntoResponse, Response},
        http::StatusCode,
    };
    use serde::{Deserialize, Serialize};
    use std::sync::Arc;

    #[derive(Serialize, Deserialize)]
    pub struct CacheEntry {
        pub key: String,
        pub value: Vec<u8>,
        pub ttl: Option<u64>,
        pub metadata: Option<serde_json::Value>,
    }

    #[derive(Serialize, Deserialize)]
    pub struct BatchRequest {
        pub keys: Vec<String>,
    }

    #[derive(Serialize, Deserialize)]
    pub struct WarmRequest {
        pub pattern: String,
        pub source: String, // "cassandra" or "compute"
        pub priority: i32,
    }

    pub async fn get_cached(
        Path(key): Path<String>,
        State(cache): State<Arc<CacheManager>>,
    ) -> Result<Response, StatusCode> {
        match cache.get(&key).await {
            Ok(Some(value)) => Ok((StatusCode::OK, value).into_response()),
            Ok(None) => Err(StatusCode::NOT_FOUND),
            Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
        }
    }

    pub async fn set_cached(
        Path(key): Path<String>,
        State(cache): State<Arc<CacheManager>>,
        Json(entry): Json<CacheEntry>,
    ) -> Result<StatusCode, StatusCode> {
        match cache.set(&key, entry.value, entry.ttl).await {
            Ok(_) => Ok(StatusCode::OK),
            Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
        }
    }

    pub async fn delete_cached(
        Path(key): Path<String>,
        State(cache): State<Arc<CacheManager>>,
    ) -> Result<StatusCode, StatusCode> {
        match cache.delete(&key).await {
            Ok(_) => Ok(StatusCode::OK),
            Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
        }
    }

    pub async fn batch_get(
        State(cache): State<Arc<CacheManager>>,
        Json(request): Json<BatchRequest>,
    ) -> Result<Json<Vec<Option<Vec<u8>>>>, StatusCode> {
        match cache.batch_get(&request.keys).await {
            Ok(values) => Ok(Json(values)),
            Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
        }
    }

    pub async fn get_partition_info(
        Path(key): Path<String>,
        State(cache): State<Arc<CacheManager>>,
    ) -> Result<Json<serde_json::Value>, StatusCode> {
        match cache.get_partition_info(&key).await {
            Ok(info) => Ok(Json(info)),
            Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
        }
    }

    pub async fn list_shards(
        State(cache): State<Arc<CacheManager>>,
    ) -> Result<Json<Vec<serde_json::Value>>, StatusCode> {
        match cache.list_shards().await {
            Ok(shards) => Ok(Json(shards)),
            Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
        }
    }

    pub async fn warm_cache(
        State(cache): State<Arc<CacheManager>>,
        Json(request): Json<WarmRequest>,
    ) -> Result<StatusCode, StatusCode> {
        match cache.warm_cache(&request.pattern, &request.source).await {
            Ok(_) => Ok(StatusCode::OK),
            Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
        }
    }

    pub async fn preload_data(
        State(cache): State<Arc<CacheManager>>,
    ) -> Result<StatusCode, StatusCode> {
        match cache.preload_hot_data().await {
            Ok(_) => Ok(StatusCode::OK),
            Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
        }
    }

    pub async fn replicate_data(
        State(cache): State<Arc<CacheManager>>,
    ) -> Result<StatusCode, StatusCode> {
        match cache.replicate_all().await {
            Ok(_) => Ok(StatusCode::OK),
            Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
        }
    }

    pub async fn check_consistency(
        Path(key): Path<String>,
        State(cache): State<Arc<CacheManager>>,
    ) -> Result<Json<bool>, StatusCode> {
        match cache.check_consistency(&key).await {
            Ok(consistent) => Ok(Json(consistent)),
            Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
        }
    }

    pub async fn metrics(
        State(cache): State<Arc<CacheManager>>,
    ) -> Result<String, StatusCode> {
        match cache.get_metrics().await {
            Ok(metrics) => Ok(metrics),
            Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
        }
    }

    pub async fn health() -> StatusCode {
        StatusCode::OK
    }

    pub async fn stats(
        State(cache): State<Arc<CacheManager>>,
    ) -> Result<Json<serde_json::Value>, StatusCode> {
        match cache.get_stats().await {
            Ok(stats) => Ok(Json(stats)),
            Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
        }
    }

    pub async fn expire_keys(
        State(cache): State<Arc<CacheManager>>,
    ) -> Result<Json<u64>, StatusCode> {
        match cache.expire_old_keys().await {
            Ok(count) => Ok(Json(count)),
            Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
        }
    }

    pub async fn cleanup(
        State(cache): State<Arc<CacheManager>>,
    ) -> Result<StatusCode, StatusCode> {
        match cache.cleanup().await {
            Ok(_) => Ok(StatusCode::OK),
            Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
        }
    }

    pub async fn backup(
        State(cache): State<Arc<CacheManager>>,
    ) -> Result<Json<String>, StatusCode> {
        match cache.backup().await {
            Ok(backup_id) => Ok(Json(backup_id)),
            Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
        }
    }
}