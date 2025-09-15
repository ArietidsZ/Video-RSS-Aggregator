use std::env;
use tracing_subscriber;
use video_rss_core::server::{run_server, ServerConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            env::var("RUST_LOG")
                .unwrap_or_else(|_| "video_rss_core=info,server=info".to_string()),
        )
        .init();

    // Load environment variables
    dotenv::dotenv().ok();

    // Configure server
    let config = ServerConfig {
        host: env::var("SERVER_HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
        port: env::var("SERVER_PORT")
            .unwrap_or_else(|_| "8000".to_string())
            .parse()
            .unwrap_or(8000),
        cache_ttl_seconds: env::var("CACHE_TTL")
            .unwrap_or_else(|_| "300".to_string())
            .parse()
            .unwrap_or(300),
        enable_cors: env::var("ENABLE_CORS")
            .unwrap_or_else(|_| "true".to_string())
            .parse()
            .unwrap_or(true),
        enable_metrics: env::var("ENABLE_METRICS")
            .unwrap_or_else(|_| "true".to_string())
            .parse()
            .unwrap_or(true),
    };

    tracing::info!(
        "Starting Video RSS Server on {}:{}",
        config.host,
        config.port
    );

    // Run server
    run_server(config).await?;

    Ok(())
}