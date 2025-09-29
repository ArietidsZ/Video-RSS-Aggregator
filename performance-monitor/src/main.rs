use anyhow::Result;
use clap::Parser;
use std::{sync::Arc, time::Duration};
use tokio::{signal, time::interval};
use tracing::{info, Level};

mod collector;
mod profiler;
mod predictor;
mod optimizer;
mod dashboard;
mod alerting;
mod regression_tester;

use collector::MetricsCollector;
use profiler::PerformanceProfiler;
use predictor::ScalingPredictor;
use optimizer::ResourceOptimizer;
use dashboard::DashboardServer;

#[derive(Parser, Debug)]
#[command(name = "performance-monitor")]
#[command(about = "Advanced performance monitoring and optimization system")]
struct Args {
    #[arg(short, long, default_value = "0.0.0.0:8010")]
    bind_address: String,

    #[arg(long, default_value = "http://prometheus:9090")]
    prometheus_url: String,

    #[arg(long, default_value = "redis://localhost:6379")]
    redis_url: String,

    #[arg(long, default_value = "postgresql://videorss:password@localhost/videorss")]
    database_url: String,

    #[arg(long, default_value = "30")]
    collection_interval_sec: u64,

    #[arg(long, default_value = "300")]
    optimization_interval_sec: u64,

    #[arg(long)]
    enable_profiling: bool,

    #[arg(long)]
    enable_predictions: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .with_target(false)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .init();

    let args = Args::parse();

    info!("Starting Performance Monitor and Optimizer");
    info!("Bind address: {}", args.bind_address);
    info!("Prometheus URL: {}", args.prometheus_url);
    info!("Collection interval: {}s", args.collection_interval_sec);

    // Initialize monitoring system
    let monitoring_system = Arc::new(
        MonitoringSystem::new(
            &args.prometheus_url,
            &args.redis_url,
            &args.database_url,
            args.collection_interval_sec,
            args.optimization_interval_sec,
            args.enable_profiling,
            args.enable_predictions,
        )
        .await?,
    );

    // Start monitoring system
    monitoring_system.start().await?;

    // Start dashboard server
    let dashboard = DashboardServer::new(Arc::clone(&monitoring_system));
    dashboard.start(&args.bind_address).await?;

    // Wait for shutdown signal
    info!("Performance monitor started. Press Ctrl+C to shutdown.");
    match signal::ctrl_c().await {
        Ok(()) => {
            info!("Shutdown signal received");
        }
        Err(err) => {
            eprintln!("Unable to listen for shutdown signal: {}", err);
        }
    }

    // Graceful shutdown
    info!("Shutting down performance monitor...");
    monitoring_system.shutdown().await?;
    info!("Performance monitor shutdown complete");

    Ok(())
}

pub struct MonitoringSystem {
    metrics_collector: Arc<MetricsCollector>,
    profiler: Option<Arc<PerformanceProfiler>>,
    predictor: Option<Arc<ScalingPredictor>>,
    optimizer: Arc<ResourceOptimizer>,

    collection_interval: Duration,
    optimization_interval: Duration,

    running: Arc<tokio::sync::RwLock<bool>>,
}

impl MonitoringSystem {
    pub async fn new(
        prometheus_url: &str,
        redis_url: &str,
        database_url: &str,
        collection_interval_sec: u64,
        optimization_interval_sec: u64,
        enable_profiling: bool,
        enable_predictions: bool,
    ) -> Result<Self> {
        info!("Initializing monitoring system...");

        // Initialize metrics collector
        let metrics_collector = Arc::new(
            MetricsCollector::new(prometheus_url, redis_url, database_url).await?
        );

        // Initialize profiler if enabled
        let profiler = if enable_profiling {
            Some(Arc::new(PerformanceProfiler::new().await?))
        } else {
            None
        };

        // Initialize predictor if enabled
        let predictor = if enable_predictions {
            Some(Arc::new(ScalingPredictor::new(database_url).await?))
        } else {
            None
        };

        // Initialize resource optimizer
        let optimizer = Arc::new(ResourceOptimizer::new(redis_url).await?);

        Ok(Self {
            metrics_collector,
            profiler,
            predictor,
            optimizer,
            collection_interval: Duration::from_secs(collection_interval_sec),
            optimization_interval: Duration::from_secs(optimization_interval_sec),
            running: Arc::new(tokio::sync::RwLock::new(false)),
        })
    }

    pub async fn start(&self) -> Result<()> {
        *self.running.write().await = true;

        // Start metrics collection
        self.start_metrics_collection().await;

        // Start performance profiling if enabled
        if let Some(ref profiler) = self.profiler {
            self.start_profiling(Arc::clone(profiler)).await;
        }

        // Start predictive scaling if enabled
        if let Some(ref predictor) = self.predictor {
            self.start_predictive_scaling(Arc::clone(predictor)).await;
        }

        // Start resource optimization
        self.start_resource_optimization().await;

        info!("All monitoring components started");
        Ok(())
    }

    async fn start_metrics_collection(&self) {
        let collector = Arc::clone(&self.metrics_collector);
        let interval = self.collection_interval;
        let running = Arc::clone(&self.running);

        tokio::spawn(async move {
            let mut ticker = interval(interval);

            loop {
                ticker.tick().await;

                if !*running.read().await {
                    break;
                }

                if let Err(e) = collector.collect_all_metrics().await {
                    tracing::error!("Metrics collection error: {}", e);
                }
            }
        });
    }

    async fn start_profiling(&self, profiler: Arc<PerformanceProfiler>) {
        let running = Arc::clone(&self.running);

        tokio::spawn(async move {
            let mut ticker = interval(Duration::from_secs(60)); // Profile every minute

            loop {
                ticker.tick().await;

                if !*running.read().await {
                    break;
                }

                if let Err(e) = profiler.run_performance_analysis().await {
                    tracing::error!("Performance profiling error: {}", e);
                }
            }
        });
    }

    async fn start_predictive_scaling(&self, predictor: Arc<ScalingPredictor>) {
        let running = Arc::clone(&self.running);

        tokio::spawn(async move {
            let mut ticker = interval(Duration::from_secs(300)); // Predict every 5 minutes

            loop {
                ticker.tick().await;

                if !*running.read().await {
                    break;
                }

                if let Err(e) = predictor.update_predictions().await {
                    tracing::error!("Scaling prediction error: {}", e);
                }
            }
        });
    }

    async fn start_resource_optimization(&self) {
        let optimizer = Arc::clone(&self.optimizer);
        let interval = self.optimization_interval;
        let running = Arc::clone(&self.running);

        tokio::spawn(async move {
            let mut ticker = interval(interval);

            loop {
                ticker.tick().await;

                if !*running.read().await {
                    break;
                }

                if let Err(e) = optimizer.optimize_resources().await {
                    tracing::error!("Resource optimization error: {}", e);
                }
            }
        });
    }

    pub async fn shutdown(&self) -> Result<()> {
        *self.running.write().await = false;

        // Give tasks time to finish
        tokio::time::sleep(Duration::from_secs(2)).await;

        info!("Monitoring system shutdown complete");
        Ok(())
    }

    pub async fn get_system_status(&self) -> Result<serde_json::Value> {
        let metrics = self.metrics_collector.get_current_metrics().await?;

        let status = serde_json::json!({
            "timestamp": chrono::Utc::now(),
            "status": "healthy",
            "components": {
                "metrics_collector": "active",
                "profiler": if self.profiler.is_some() { "active" } else { "disabled" },
                "predictor": if self.predictor.is_some() { "active" } else { "disabled" },
                "optimizer": "active"
            },
            "metrics": metrics
        });

        Ok(status)
    }
}