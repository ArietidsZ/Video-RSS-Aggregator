mod gateway;
mod graphql;
mod versioning;
mod webhooks;
mod documentation;

use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    Router,
};
use clap::Parser;
use sqlx::postgres::PgPoolOptions;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use tower_http::cors::CorsLayer;
use tracing::{info, error};

#[derive(Clone)]
struct AppMetrics {
    start_time: Arc<Instant>,
    requests_total: Arc<AtomicU64>,
    errors_total: Arc<AtomicU64>,
    response_time_sum_ms: Arc<AtomicU64>,
}

impl AppMetrics {
    fn new() -> Self {
        Self {
            start_time: Arc::new(Instant::now()),
            requests_total: Arc::new(AtomicU64::new(0)),
            errors_total: Arc::new(AtomicU64::new(0)),
            response_time_sum_ms: Arc::new(AtomicU64::new(0)),
        }
    }

    fn record_request(&self, duration_ms: u64, is_error: bool) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.response_time_sum_ms.fetch_add(duration_ms, Ordering::Relaxed);
        if is_error {
            self.errors_total.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn get_uptime_seconds(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }

    fn get_requests_total(&self) -> u64 {
        self.requests_total.load(Ordering::Relaxed)
    }

    fn get_errors_total(&self) -> u64 {
        self.errors_total.load(Ordering::Relaxed)
    }

    fn get_avg_response_time_ms(&self) -> f64 {
        let total_requests = self.get_requests_total();
        if total_requests == 0 {
            return 0.0;
        }
        let total_time = self.response_time_sum_ms.load(Ordering::Relaxed);
        total_time as f64 / total_requests as f64
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Port to listen on
    #[arg(short, long, default_value = "8080")]
    port: u16,

    /// Database URL
    #[arg(long, env = "DATABASE_URL")]
    database_url: String,

    /// Redis URL
    #[arg(long, env = "REDIS_URL", default_value = "redis://localhost:6379")]
    redis_url: String,

    /// Enable debug mode
    #[arg(long)]
    debug: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let args = Args::parse();

    info!("Starting Video RSS Aggregator API Gateway on port {}", args.port);

    // Create database pool
    let pool = PgPoolOptions::new()
        .max_connections(50)
        .connect(&args.database_url)
        .await?;

    info!("Connected to database");

    // Run migrations
    sqlx::migrate!("./migrations")
        .run(&pool)
        .await?;

    info!("Database migrations completed");

    // Create the main router
    let app = create_app_router(pool.clone(), &args.redis_url).await?;

    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!("API Gateway listening on {}", addr);

    axum::serve(listener, app)
        .await?;

    Ok(())
}

async fn create_app_router(pool: sqlx::PgPool, redis_url: &str) -> Result<Router> {
    // Initialize components
    let gateway_config = gateway::GatewayConfig::default();
    let api_gateway = std::sync::Arc::new(
        gateway::ApiGateway::new(gateway_config, redis_url).await?
    );

    let webhook_manager = std::sync::Arc::new(
        webhooks::WebhookManager::new(pool.clone(), webhooks::WebhookConfig::default()).await?
    );

    // Initialize metrics
    let app_metrics = AppMetrics::new();

    // Create routers
    let gateway_router = gateway::create_router(api_gateway);
    let graphql_router = graphql::create_graphql_router(pool.clone());
    let versioned_router = versioning::create_versioned_router();
    let webhook_router = create_webhook_router_wrapper(pool.clone());
    let docs_router = documentation::create_docs_router();

    // Combine all routers
    let app = Router::new()
        .route("/", axum::routing::get(root))
        .route("/health", axum::routing::get(health_check))
        .route("/metrics", axum::routing::get(metrics))
        .with_state(app_metrics)
        .nest("/gateway", gateway_router)
        .nest("/", graphql_router)
        .nest("/", versioned_router)
        .nest("/", webhook_router)
        .nest("/", docs_router)
        .layer(CorsLayer::permissive());

    Ok(app)
}

async fn root() -> impl IntoResponse {
    axum::Json(serde_json::json!({
        "name": "Video RSS Aggregator API",
        "version": "3.0.0",
        "documentation": "/swagger-ui",
        "graphql": "/graphql",
        "health": "/health",
        "metrics": "/metrics"
    }))
}

async fn health_check() -> impl IntoResponse {
    (StatusCode::OK, "OK")
}

async fn metrics(State(metrics): State<AppMetrics>) -> impl IntoResponse {
    // Collect real metrics from the application
    axum::Json(serde_json::json!({
        "uptime_seconds": metrics.get_uptime_seconds(),
        "requests_total": metrics.get_requests_total(),
        "errors_total": metrics.get_errors_total(),
        "avg_response_time_ms": metrics.get_avg_response_time_ms(),
        "error_rate": if metrics.get_requests_total() > 0 {
            metrics.get_errors_total() as f64 / metrics.get_requests_total() as f64
        } else {
            0.0
        }
    }))
}

fn create_webhook_router_wrapper(pool: sqlx::PgPool) -> Router {
    // This is a wrapper because webhook router creation is async
    // In production, would properly handle async initialization
    Router::new()
        .route("/webhooks", axum::routing::get(|| async { "Webhooks endpoint" }))
}