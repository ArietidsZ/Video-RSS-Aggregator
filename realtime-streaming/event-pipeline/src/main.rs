use anyhow::Result;
use clap::Parser;
use std::{sync::Arc, time::Duration};
use tokio::{signal, time::interval};
use tracing::{info, Level};

mod event_bus;
mod pipeline;
mod processors;
mod types;

use event_bus::EventBus;
use pipeline::EventPipeline;

#[derive(Parser, Debug)]
#[command(name = "event-pipeline")]
#[command(about = "High-performance event-driven processing pipeline")]
struct Args {
    #[arg(short, long, default_value = "kafka-1:29092,kafka-2:29093,kafka-3:29094")]
    kafka_brokers: String,

    #[arg(short, long, default_value = "redis://localhost:6379")]
    redis_url: String,

    #[arg(short, long, default_value = "4")]
    worker_threads: usize,

    #[arg(long, default_value = "1000")]
    max_queue_size: usize,

    #[arg(long, default_value = "100")]
    batch_size: usize,

    #[arg(long, default_value = "1000")]
    flush_interval_ms: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    let args = Args::parse();

    info!("Starting Event Processing Pipeline");
    info!("Kafka brokers: {}", args.kafka_brokers);
    info!("Worker threads: {}", args.worker_threads);
    info!("Max queue size: {}", args.max_queue_size);

    // Initialize event bus
    let event_bus = Arc::new(
        EventBus::new(
            &args.kafka_brokers,
            &args.redis_url,
            args.max_queue_size,
        )
        .await?,
    );

    // Initialize pipeline
    let pipeline = Arc::new(
        EventPipeline::new(
            Arc::clone(&event_bus),
            args.worker_threads,
            args.batch_size,
            Duration::from_millis(args.flush_interval_ms),
        )
        .await?,
    );

    // Start pipeline
    pipeline.start().await?;

    // Start metrics reporter
    start_metrics_reporter(Arc::clone(&pipeline)).await;

    // Wait for shutdown signal
    info!("Event pipeline started. Press Ctrl+C to shutdown.");
    match signal::ctrl_c().await {
        Ok(()) => {
            info!("Shutdown signal received");
        }
        Err(err) => {
            eprintln!("Unable to listen for shutdown signal: {}", err);
        }
    }

    // Graceful shutdown
    info!("Shutting down pipeline...");
    pipeline.shutdown().await?;
    info!("Pipeline shutdown complete");

    Ok(())
}

async fn start_metrics_reporter(pipeline: Arc<EventPipeline>) {
    tokio::spawn(async move {
        let mut interval = interval(Duration::from_secs(30));

        loop {
            interval.tick().await;

            let metrics = pipeline.get_metrics().await;
            info!(
                "Pipeline Metrics - Processed: {}, Failed: {}, Queue: {}, Throughput: {:.1}/s",
                metrics.total_processed,
                metrics.total_failed,
                metrics.queue_size,
                metrics.throughput_per_second
            );
        }
    });
}