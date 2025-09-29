use anyhow::Result;
use axum::{
    extract::{State, WebSocketUpgrade},
    response::IntoResponse,
    routing::get,
    Router,
};
use clap::Parser;
use std::{net::SocketAddr, sync::Arc};
use tokio::net::TcpListener;
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing::{info, Level};

mod server;
mod signaling;
mod streaming;
mod types;

use server::WebRTCServer;

#[derive(Parser, Debug)]
#[command(name = "webrtc-server")]
#[command(about = "High-performance WebRTC server for real-time video streaming")]
struct Args {
    #[arg(short, long, default_value = "0.0.0.0:8090")]
    bind: SocketAddr,

    #[arg(short, long, default_value = "kafka-1:29092,kafka-2:29093,kafka-3:29094")]
    kafka_brokers: String,

    #[arg(short, long, default_value = "redis://localhost:6379")]
    redis_url: String,

    #[arg(long, default_value = "200")]
    max_connections: usize,

    #[arg(long, default_value = "1000")]
    max_bitrate_kbps: u32,

    #[arg(long, default_value = "30")]
    target_fps: u32,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    let args = Args::parse();

    info!("Starting WebRTC Server on {}", args.bind);
    info!("Kafka brokers: {}", args.kafka_brokers);
    info!("Redis URL: {}", args.redis_url);

    // Create WebRTC server instance
    let webrtc_server = Arc::new(
        WebRTCServer::new(
            &args.kafka_brokers,
            &args.redis_url,
            args.max_connections,
            args.max_bitrate_kbps,
            args.target_fps,
        )
        .await?,
    );

    // Create router
    let app = Router::new()
        .route("/ws", get(websocket_handler))
        .route("/health", get(health_check))
        .route("/stats", get(stats_handler))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(webrtc_server);

    // Start server
    let listener = TcpListener::bind(args.bind).await?;
    info!("WebRTC server listening on {}", args.bind);

    axum::serve(listener, app).await?;

    Ok(())
}

async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(server): State<Arc<WebRTCServer>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| async move {
        if let Err(e) = server.handle_websocket(socket).await {
            tracing::error!("WebSocket error: {}", e);
        }
    })
}

async fn health_check() -> impl IntoResponse {
    "OK"
}

async fn stats_handler(State(server): State<Arc<WebRTCServer>>) -> impl IntoResponse {
    axum::Json(server.get_stats().await)
}