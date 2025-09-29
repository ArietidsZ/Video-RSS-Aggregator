mod gateway;
mod graphql;
mod versioning;
mod webhooks;
mod documentation;

use anyhow::Result;
use axum::{
    http::StatusCode,
    response::IntoResponse,
    Router,
};
use clap::Parser;
use sqlx::postgres::PgPoolOptions;
use std::net::SocketAddr;
use tower_http::cors::CorsLayer;
use tracing::{info, error};

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
    info!("API Gateway listening on {}", addr);

    axum::Server::bind(&addr)
        .serve(app.into_make_service())
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

async fn metrics() -> impl IntoResponse {
    // Would implement actual metrics collection
    axum::Json(serde_json::json!({
        "uptime": 0,
        "requests_total": 0,
        "errors_total": 0,
        "response_time_ms": 0
    }))
}

fn create_webhook_router_wrapper(pool: sqlx::PgPool) -> Router {
    // This is a wrapper because webhook router creation is async
    // In production, would properly handle async initialization
    Router::new()
        .route("/webhooks", axum::routing::get(|| async { "Webhooks endpoint" }))
}