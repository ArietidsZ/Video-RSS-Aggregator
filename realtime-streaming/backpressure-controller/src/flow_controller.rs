use crate::{
    adaptive_rate_limiter::AdaptiveRateLimiter,
    circuit_breaker::CircuitBreaker,
    load_balancer::LoadBalancer,
    metrics_collector::MetricsCollector,
    queue_manager::QueueManager,
};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::{
    sync::{RwLock, Semaphore},
    time::interval,
};
use tracing::{debug, error, info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowControlMetrics {
    pub current_queue_size: usize,
    pub max_queue_size: usize,
    pub queue_utilization: f64,
    pub current_throughput: f64,
    pub target_throughput: f64,
    pub error_rate: f64,
    pub circuit_breaker_open: bool,
    pub processing_paused: bool,
    pub adaptive_rate_limit: f64,
    pub active_workers: usize,
    pub total_processed: u64,
    pub total_errors: u64,
    pub average_latency_ms: f64,
    pub p99_latency_ms: f64,
}

pub struct FlowController {
    // Core components
    queue_manager: Arc<QueueManager>,
    rate_limiter: Arc<AdaptiveRateLimiter>,
    circuit_breaker: Arc<CircuitBreaker>,
    load_balancer: Arc<LoadBalancer>,
    metrics_collector: Arc<MetricsCollector>,

    // Control mechanisms
    processing_semaphore: Arc<Semaphore>,
    shutdown_signal: Arc<AtomicBool>,
    processing_paused: Arc<AtomicBool>,

    // Configuration
    max_queue_size: usize,
    target_throughput: f64,
    warning_threshold: f64,
    critical_threshold: f64,

    // Metrics
    total_processed: Arc<AtomicU64>,
    total_errors: Arc<AtomicU64>,
    start_time: Instant,
}

impl FlowController {
    pub async fn new(
        max_queue_size: usize,
        target_throughput: f64,
        warning_threshold: f64,
        critical_threshold: f64,
        circuit_breaker_timeout_sec: u64,
        error_rate_threshold: f64,
    ) -> Result<Self> {
        info!("Initializing Flow Controller...");

        let queue_manager = Arc::new(QueueManager::new(max_queue_size).await?);
        let rate_limiter = Arc::new(AdaptiveRateLimiter::new(target_throughput).await?);
        let circuit_breaker = Arc::new(
            CircuitBreaker::new(
                error_rate_threshold,
                Duration::from_secs(circuit_breaker_timeout_sec),
            )
            .await?,
        );
        let load_balancer = Arc::new(LoadBalancer::new().await?);
        let metrics_collector = Arc::new(MetricsCollector::new().await?);

        Ok(Self {
            queue_manager,
            rate_limiter,
            circuit_breaker,
            load_balancer,
            metrics_collector,
            processing_semaphore: Arc::new(Semaphore::new(100)), // Max 100 concurrent operations
            shutdown_signal: Arc::new(AtomicBool::new(false)),
            processing_paused: Arc::new(AtomicBool::new(false)),
            max_queue_size,
            target_throughput,
            warning_threshold,
            critical_threshold,
            total_processed: Arc::new(AtomicU64::new(0)),
            total_errors: Arc::new(AtomicU64::new(0)),
            start_time: Instant::now(),
        })
    }

    pub async fn start(&self) -> Result<()> {
        info!("Starting Flow Controller...");

        // Start background monitoring tasks
        self.start_queue_monitor().await;
        self.start_throughput_controller().await;
        self.start_adaptive_rate_limiter().await;
        self.start_circuit_breaker_monitor().await;
        self.start_load_balancer().await;

        info!("Flow Controller started successfully");
        Ok(())
    }

    async fn start_queue_monitor(&self) {
        let queue_manager = Arc::clone(&self.queue_manager);
        let warning_threshold = self.warning_threshold;
        let critical_threshold = self.critical_threshold;
        let max_size = self.max_queue_size;

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(1));

            loop {
                interval.tick().await;

                let current_size = queue_manager.size().await;
                let utilization = current_size as f64 / max_size as f64;

                if utilization > critical_threshold {
                    error!(
                        "CRITICAL: Queue utilization at {:.1}% ({}/{})",
                        utilization * 100.0,
                        current_size,
                        max_size
                    );
                    // Emergency shedding
                    queue_manager.shed_load(0.2).await; // Drop 20% of oldest items
                } else if utilization > warning_threshold {
                    warn!(
                        "WARNING: Queue utilization at {:.1}% ({}/{})",
                        utilization * 100.0,
                        current_size,
                        max_size
                    );
                }
            }
        });
    }

    async fn start_throughput_controller(&self) {
        let rate_limiter = Arc::clone(&self.rate_limiter);
        let queue_manager = Arc::clone(&self.queue_manager);
        let target_throughput = self.target_throughput;

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(5));

            loop {
                interval.tick().await;

                let current_throughput = rate_limiter.get_current_rate().await;
                let queue_size = queue_manager.size().await;

                // Adaptive throughput adjustment
                let adjustment_factor = if queue_size > 100 {
                    0.8 // Reduce throughput if queue is growing
                } else if queue_size < 10 {
                    1.2 // Increase throughput if queue is small
                } else {
                    1.0 // Maintain current rate
                };

                let new_target = (target_throughput * adjustment_factor).min(target_throughput * 2.0);
                rate_limiter.adjust_rate(new_target).await;

                debug!(
                    "Throughput adjustment - Current: {:.1}, Target: {:.1}, Queue: {}",
                    current_throughput, new_target, queue_size
                );
            }
        });
    }

    async fn start_adaptive_rate_limiter(&self) {
        let rate_limiter = Arc::clone(&self.rate_limiter);
        let metrics_collector = Arc::clone(&self.metrics_collector);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(2));

            loop {
                interval.tick().await;

                let error_rate = metrics_collector.get_error_rate().await;
                let avg_latency = metrics_collector.get_average_latency().await;

                // Adjust rate based on system health
                if error_rate > 0.1 || avg_latency > Duration::from_millis(1000) {
                    // System under stress - reduce rate
                    rate_limiter.multiply_rate(0.9).await;
                    debug!("Reducing rate due to system stress (error_rate: {:.2}%, latency: {:?})",
                           error_rate * 100.0, avg_latency);
                } else if error_rate < 0.01 && avg_latency < Duration::from_millis(100) {
                    // System healthy - increase rate
                    rate_limiter.multiply_rate(1.05).await;
                    debug!("Increasing rate due to healthy system");
                }
            }
        });
    }

    async fn start_circuit_breaker_monitor(&self) {
        let circuit_breaker = Arc::clone(&self.circuit_breaker);
        let processing_paused = Arc::clone(&self.processing_paused);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(1));

            loop {
                interval.tick().await;

                if circuit_breaker.is_open().await {
                    if !processing_paused.load(Ordering::Relaxed) {
                        warn!("Circuit breaker opened - pausing processing");
                        processing_paused.store(true, Ordering::Relaxed);
                    }
                } else if processing_paused.load(Ordering::Relaxed) {
                    info!("Circuit breaker closed - resuming processing");
                    processing_paused.store(false, Ordering::Relaxed);
                }
            }
        });
    }

    async fn start_load_balancer(&self) {
        let load_balancer = Arc::clone(&self.load_balancer);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(10));

            loop {
                interval.tick().await;

                // Update load balancer with current system metrics
                load_balancer.update_health_metrics().await;

                // Rebalance if needed
                if load_balancer.should_rebalance().await {
                    info!("Rebalancing load distribution");
                    load_balancer.rebalance().await;
                }
            }
        });
    }

    pub async fn process_request<T, F, R>(&self, request: T, handler: F) -> Result<R>
    where
        F: FnOnce(T) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<R>> + Send>>,
        T: Send + 'static,
        R: Send + 'static,
    {
        // Check if processing is paused
        if self.processing_paused.load(Ordering::Relaxed) {
            return Err(anyhow::anyhow!("Processing paused due to system overload"));
        }

        // Check circuit breaker
        if self.circuit_breaker.is_open().await {
            return Err(anyhow::anyhow!("Circuit breaker open - system unavailable"));
        }

        // Rate limiting
        self.rate_limiter.acquire().await?;

        // Queue management
        if !self.queue_manager.try_enqueue().await {
            return Err(anyhow::anyhow!("Queue full - request rejected"));
        }

        // Acquire processing permit
        let _permit = self.processing_semaphore.acquire().await.unwrap();

        let start_time = Instant::now();

        // Process request
        let result = handler(request).await;

        let processing_time = start_time.elapsed();

        // Update metrics
        match &result {
            Ok(_) => {
                self.total_processed.fetch_add(1, Ordering::Relaxed);
                self.circuit_breaker.record_success().await;
                self.metrics_collector.record_success(processing_time).await;
            }
            Err(_) => {
                self.total_errors.fetch_add(1, Ordering::Relaxed);
                self.circuit_breaker.record_failure().await;
                self.metrics_collector.record_error(processing_time).await;
            }
        }

        // Dequeue
        self.queue_manager.dequeue().await;

        result
    }

    pub async fn get_metrics(&self) -> FlowControlMetrics {
        let uptime = self.start_time.elapsed().as_secs();
        let total_processed = self.total_processed.load(Ordering::Relaxed);
        let total_errors = self.total_errors.load(Ordering::Relaxed);

        let current_throughput = if uptime > 0 {
            total_processed as f64 / uptime as f64
        } else {
            0.0
        };

        let error_rate = if total_processed + total_errors > 0 {
            total_errors as f64 / (total_processed + total_errors) as f64
        } else {
            0.0
        };

        let queue_size = self.queue_manager.size().await;
        let queue_utilization = queue_size as f64 / self.max_queue_size as f64;

        FlowControlMetrics {
            current_queue_size: queue_size,
            max_queue_size: self.max_queue_size,
            queue_utilization,
            current_throughput,
            target_throughput: self.target_throughput,
            error_rate,
            circuit_breaker_open: self.circuit_breaker.is_open().await,
            processing_paused: self.processing_paused.load(Ordering::Relaxed),
            adaptive_rate_limit: self.rate_limiter.get_current_limit().await,
            active_workers: self.processing_semaphore.available_permits(),
            total_processed,
            total_errors,
            average_latency_ms: self.metrics_collector.get_average_latency().await.as_millis() as f64,
            p99_latency_ms: self.metrics_collector.get_p99_latency().await.as_millis() as f64,
        }
    }

    pub async fn reset_circuit_breaker(&self) {
        self.circuit_breaker.reset().await;
        info!("Circuit breaker manually reset");
    }

    pub async fn pause_processing(&self) {
        self.processing_paused.store(true, Ordering::Relaxed);
        info!("Processing manually paused");
    }

    pub async fn resume_processing(&self) {
        self.processing_paused.store(false, Ordering::Relaxed);
        info!("Processing manually resumed");
    }

    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down Flow Controller...");
        self.shutdown_signal.store(true, Ordering::Relaxed);

        // Give time for current operations to complete
        tokio::time::sleep(Duration::from_secs(5)).await;

        info!("Flow Controller shutdown complete");
        Ok(())
    }
}