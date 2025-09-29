use anyhow::Result;
use clap::Parser;
use std::{sync::Arc, time::Duration};
use tokio::{signal, time::interval};
use tracing::{info, Level};

mod adaptive_rate_limiter;
mod circuit_breaker;
mod flow_controller;
mod load_balancer;
mod metrics_collector;
mod queue_manager;

use flow_controller::FlowController;

#[derive(Parser, Debug)]
#[command(name = "backpressure-controller")]
#[command(about = "Intelligent backpressure and flow control system")]
struct Args {
    #[arg(short, long, default_value = "0.0.0.0:8095")]
    bind_address: String,

    #[arg(long, default_value = "1000")]
    max_queue_size: usize,

    #[arg(long, default_value = "100")]
    target_throughput: f64,

    #[arg(long, default_value = "0.8")]
    warning_threshold: f64,

    #[arg(long, default_value = "0.95")]
    critical_threshold: f64,

    #[arg(long, default_value = "30")]
    circuit_breaker_timeout_sec: u64,

    #[arg(long, default_value = "0.1")]
    error_rate_threshold: f64,

    #[arg(long, default_value = "10")]
    metrics_interval_sec: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    let args = Args::parse();

    info!("Starting Backpressure Controller");
    info!("Bind address: {}", args.bind_address);
    info!("Max queue size: {}", args.max_queue_size);
    info!("Target throughput: {:.1} req/s", args.target_throughput);

    // Initialize flow controller
    let flow_controller = Arc::new(
        FlowController::new(
            args.max_queue_size,
            args.target_throughput,
            args.warning_threshold,
            args.critical_threshold,
            args.circuit_breaker_timeout_sec,
            args.error_rate_threshold,
        )
        .await?,
    );

    // Start flow controller
    flow_controller.start().await?;

    // Start metrics reporting
    start_metrics_reporter(Arc::clone(&flow_controller), args.metrics_interval_sec).await;

    // Start HTTP server for metrics and control
    start_http_server(Arc::clone(&flow_controller), &args.bind_address).await?;

    // Wait for shutdown signal
    info!("Backpressure controller started. Press Ctrl+C to shutdown.");
    match signal::ctrl_c().await {
        Ok(()) => {
            info!("Shutdown signal received");
        }
        Err(err) => {
            eprintln!("Unable to listen for shutdown signal: {}", err);
        }
    }

    // Graceful shutdown
    info!("Shutting down flow controller...");
    flow_controller.shutdown().await?;
    info!("Backpressure controller shutdown complete");

    Ok(())
}

async fn start_metrics_reporter(flow_controller: Arc<FlowController>, interval_sec: u64) {
    tokio::spawn(async move {
        let mut interval = interval(Duration::from_secs(interval_sec));

        loop {
            interval.tick().await;

            let metrics = flow_controller.get_metrics().await;
            info!(
                "Flow Control Metrics - Queue: {}/{} ({:.1}%), Throughput: {:.1}/s, Errors: {:.2}%, Circuit: {}",
                metrics.current_queue_size,
                metrics.max_queue_size,
                metrics.queue_utilization * 100.0,
                metrics.current_throughput,
                metrics.error_rate * 100.0,
                if metrics.circuit_breaker_open { "OPEN" } else { "CLOSED" }
            );
        }
    });
}

async fn start_http_server(
    flow_controller: Arc<FlowController>,
    bind_address: &str,
) -> Result<()> {
    use std::convert::Infallible;
    use std::net::SocketAddr;
    use hyper::service::{make_service_fn, service_fn};
    use hyper::{Body, Method, Request, Response, Server, StatusCode};

    let flow_controller = Arc::clone(&flow_controller);

    let make_svc = make_service_fn(move |_conn| {
        let flow_controller = Arc::clone(&flow_controller);
        async move {
            Ok::<_, Infallible>(service_fn(move |req| {
                let flow_controller = Arc::clone(&flow_controller);
                async move {
                    handle_request(req, flow_controller).await
                }
            }))
        }
    });

    let addr: SocketAddr = bind_address.parse()?;
    let server = Server::bind(&addr).serve(make_svc);

    info!("HTTP server listening on http://{}", addr);

    tokio::spawn(async move {
        if let Err(e) = server.await {
            eprintln!("HTTP server error: {}", e);
        }
    });

    Ok(())
}

async fn handle_request(
    req: Request<Body>,
    flow_controller: Arc<FlowController>,
) -> Result<Response<Body>, Infallible> {
    let response = match (req.method(), req.uri().path()) {
        (&Method::GET, "/health") => {
            Response::builder()
                .status(StatusCode::OK)
                .body(Body::from("OK"))
                .unwrap()
        }
        (&Method::GET, "/metrics") => {
            let metrics = flow_controller.get_metrics().await;
            let json = serde_json::to_string_pretty(&metrics).unwrap();
            Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "application/json")
                .body(Body::from(json))
                .unwrap()
        }
        (&Method::POST, "/control/reset") => {
            flow_controller.reset_circuit_breaker().await;
            Response::builder()
                .status(StatusCode::OK)
                .body(Body::from("Circuit breaker reset"))
                .unwrap()
        }
        (&Method::POST, "/control/pause") => {
            flow_controller.pause_processing().await;
            Response::builder()
                .status(StatusCode::OK)
                .body(Body::from("Processing paused"))
                .unwrap()
        }
        (&Method::POST, "/control/resume") => {
            flow_controller.resume_processing().await;
            Response::builder()
                .status(StatusCode::OK)
                .body(Body::from("Processing resumed"))
                .unwrap()
        }
        _ => {
            Response::builder()
                .status(StatusCode::NOT_FOUND)
                .body(Body::from("Not Found"))
                .unwrap()
        }
    };

    Ok(response)
}

// Add hyper dependency to Cargo.toml for HTTP server
// This would be added to the dependencies section:
// hyper = { version = "0.14", features = ["full"] }