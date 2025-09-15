use std::env;
use std::sync::Arc;
use tracing_subscriber;
use video_rss_core::monitor::{MonitorConfig, ServiceConfig, ServiceMonitor};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            env::var("RUST_LOG")
                .unwrap_or_else(|_| "video_rss_core=info,monitor=info".to_string()),
        )
        .init();

    // Load environment variables
    dotenv::dotenv().ok();

    // Configure monitoring
    let config = MonitorConfig {
        check_interval_seconds: env::var("MONITOR_INTERVAL")
            .unwrap_or_else(|_| "30".to_string())
            .parse()
            .unwrap_or(30),
        max_restart_attempts: env::var("MAX_RESTART_ATTEMPTS")
            .unwrap_or_else(|_| "3".to_string())
            .parse()
            .unwrap_or(3),
        restart_delay_seconds: env::var("RESTART_DELAY")
            .unwrap_or_else(|_| "5".to_string())
            .parse()
            .unwrap_or(5),
        health_check_timeout_seconds: env::var("HEALTH_CHECK_TIMEOUT")
            .unwrap_or_else(|_| "10".to_string())
            .parse()
            .unwrap_or(10),
        services: vec![
            ServiceConfig {
                name: "video_rss_server".to_string(),
                health_endpoint: env::var("SERVER_HEALTH_ENDPOINT")
                    .unwrap_or_else(|_| "http://localhost:8000/health".to_string()),
                start_command: env::var("SERVER_START_COMMAND")
                    .unwrap_or_else(|_| "cargo run --bin server".to_string()),
                stop_command: env::var("SERVER_STOP_COMMAND").ok(),
                working_directory: env::var("SERVER_WORKING_DIR").ok(),
                env_vars: vec![],
                expected_port: env::var("SERVER_PORT")
                    .unwrap_or_else(|_| "8000".to_string())
                    .parse()
                    .unwrap_or(8000),
            },
        ],
    };

    tracing::info!("Starting Service Monitor");
    tracing::info!("Monitoring {} services", config.services.len());

    // Create and start monitor
    let monitor = Arc::new(ServiceMonitor::new(config));

    // Start monitoring in background
    let monitor_clone = monitor.clone();
    tokio::spawn(async move {
        monitor_clone.start_monitoring().await;
    });

    // Optional: Start HTTP API for monitor control
    if env::var("MONITOR_API_ENABLED")
        .unwrap_or_else(|_| "true".to_string())
        .parse()
        .unwrap_or(true)
    {
        let api_port = env::var("MONITOR_API_PORT")
            .unwrap_or_else(|_| "9000".to_string())
            .parse()
            .unwrap_or(9000);

        tracing::info!("Starting Monitor API on port {}", api_port);

        let app = video_rss_core::monitor::api::create_monitor_router(monitor);
        let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", api_port)).await?;

        axum::serve(listener, app).await?;
    } else {
        // Just keep monitoring forever
        tokio::signal::ctrl_c().await?;
        tracing::info!("Shutting down monitor...");
    }

    Ok(())
}