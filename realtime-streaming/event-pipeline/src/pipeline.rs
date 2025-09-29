use crate::{
    event_bus::EventBus,
    processors::{ProcessorRegistry, ProcessorTrait},
    types::{BackpressureConfig, Event, PipelineMetrics, ProcessingResult, ProcessorConfig},
};
use anyhow::Result;
use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant, SystemTime},
};
use tokio::{
    sync::{mpsc, RwLock, Semaphore},
    time::{interval, timeout},
};
use tracing::{debug, error, info, warn};

pub struct EventPipeline {
    event_bus: Arc<EventBus>,
    processor_registry: Arc<ProcessorRegistry>,

    // Processing controls
    processing_semaphore: Arc<Semaphore>,
    shutdown_signal: Arc<AtomicBool>,

    // Backpressure management
    backpressure_config: BackpressureConfig,
    current_queue_size: Arc<AtomicUsize>,
    processing_rate: Arc<AtomicU64>,

    // Pipeline metrics
    total_processed: Arc<AtomicU64>,
    total_failed: Arc<AtomicU64>,
    start_time: Instant,

    // Configuration
    worker_threads: usize,
    batch_size: usize,
    flush_interval: Duration,
}

impl EventPipeline {
    pub async fn new(
        event_bus: Arc<EventBus>,
        worker_threads: usize,
        batch_size: usize,
        flush_interval: Duration,
    ) -> Result<Self> {
        info!("Initializing Event Pipeline with {} worker threads", worker_threads);

        let processor_registry = Arc::new(ProcessorRegistry::new());

        Ok(Self {
            event_bus,
            processor_registry,
            processing_semaphore: Arc::new(Semaphore::new(worker_threads * 2)),
            shutdown_signal: Arc::new(AtomicBool::new(false)),
            backpressure_config: BackpressureConfig::default(),
            current_queue_size: Arc::new(AtomicUsize::new(0)),
            processing_rate: Arc::new(AtomicU64::new(0)),
            total_processed: Arc::new(AtomicU64::new(0)),
            total_failed: Arc::new(AtomicU64::new(0)),
            start_time: Instant::now(),
            worker_threads,
            batch_size,
            flush_interval,
        })
    }

    pub async fn start(&self) -> Result<()> {
        info!("Starting Event Pipeline...");

        // Register default processors
        self.register_default_processors().await?;

        // Start event consumers for different event types
        self.start_event_consumers().await?;

        // Start backpressure monitor
        self.start_backpressure_monitor().await;

        // Start metrics collector
        self.start_metrics_collector().await;

        info!("Event Pipeline started successfully");
        Ok(())
    }

    async fn register_default_processors(&self) -> Result<()> {
        // Register built-in processors
        self.processor_registry.register_video_processor().await?;
        self.processor_registry.register_transcription_processor().await?;
        self.processor_registry.register_summarization_processor().await?;
        self.processor_registry.register_rss_processor().await?;
        self.processor_registry.register_metrics_processor().await?;
        self.processor_registry.register_system_processor().await?;

        info!("Registered {} processors", self.processor_registry.count().await);
        Ok(())
    }

    async fn start_event_consumers(&self) -> Result<()> {
        let event_types = vec![
            crate::types::EventType::VideoUploaded,
            crate::types::EventType::VideoProcessingStarted,
            crate::types::EventType::TranscriptionCompleted,
            crate::types::EventType::SummarizationCompleted,
            crate::types::EventType::RSSFeedUpdated,
            crate::types::EventType::UserRequest,
            crate::types::EventType::SystemEvent,
            crate::types::EventType::MetricsUpdate,
        ];

        // Create consumer for each worker thread
        for worker_id in 0..self.worker_threads {
            let mut receiver = self.event_bus
                .subscribe(event_types.clone(), &format!("pipeline-worker-{}", worker_id))
                .await?;

            let pipeline = Arc::new(self.clone_for_worker());

            tokio::spawn(async move {
                Self::worker_loop(worker_id, pipeline, receiver).await;
            });
        }

        info!("Started {} event consumer workers", self.worker_threads);
        Ok(())
    }

    async fn worker_loop(
        worker_id: usize,
        pipeline: Arc<EventPipeline>,
        mut receiver: mpsc::UnboundedReceiver<Event>,
    ) {
        info!("Worker {} started", worker_id);
        let mut batch = Vec::with_capacity(pipeline.batch_size);
        let mut last_flush = Instant::now();

        while !pipeline.shutdown_signal.load(Ordering::Relaxed) {
            // Try to receive events for batching
            let should_flush = match timeout(Duration::from_millis(100), receiver.recv()).await {
                Ok(Some(event)) => {
                    batch.push(event);
                    pipeline.current_queue_size.fetch_add(1, Ordering::Relaxed);

                    // Flush if batch is full or flush interval reached
                    batch.len() >= pipeline.batch_size
                        || last_flush.elapsed() >= pipeline.flush_interval
                }
                Ok(None) => {
                    debug!("Worker {} receiver closed", worker_id);
                    break;
                }
                Err(_) => {
                    // Timeout - check if we should flush partial batch
                    !batch.is_empty() && last_flush.elapsed() >= pipeline.flush_interval
                }
            };

            if should_flush && !batch.is_empty() {
                pipeline.process_batch(worker_id, &mut batch).await;
                last_flush = Instant::now();
            }

            // Check backpressure
            if pipeline.should_slow_down().await {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }

        // Process remaining batch on shutdown
        if !batch.is_empty() {
            pipeline.process_batch(worker_id, &mut batch).await;
        }

        info!("Worker {} stopped", worker_id);
    }

    async fn process_batch(&self, worker_id: usize, batch: &mut Vec<Event>) {
        let batch_size = batch.len();
        debug!("Worker {} processing batch of {} events", worker_id, batch_size);

        for event in batch.drain(..) {
            self.current_queue_size.fetch_sub(1, Ordering::Relaxed);

            // Acquire semaphore permit for concurrency control
            let _permit = self.processing_semaphore.acquire().await.unwrap();

            // Process event
            match self.process_event(event).await {
                Ok(_) => {
                    self.total_processed.fetch_add(1, Ordering::Relaxed);
                    self.processing_rate.fetch_add(1, Ordering::Relaxed);
                }
                Err(e) => {
                    self.total_failed.fetch_add(1, Ordering::Relaxed);
                    error!("Event processing failed: {}", e);
                }
            }
        }
    }

    async fn process_event(&self, mut event: Event) -> Result<()> {
        let start_time = Instant::now();

        // Check if event is expired
        if event.is_expired() {
            warn!("Dropping expired event: {}", event.id);
            return Ok(());
        }

        // Get appropriate processor
        let processor = self.processor_registry.get_processor(&event.event_type).await
            .ok_or_else(|| anyhow::anyhow!("No processor found for event type: {:?}", event.event_type))?;

        // Process with timeout
        let config = ProcessorConfig::default();
        let result = timeout(
            Duration::from_millis(config.timeout_ms),
            processor.process(event.clone())
        ).await;

        match result {
            Ok(Ok(processing_result)) => {
                if processing_result.success {
                    debug!("Successfully processed event {} in {:?}",
                           event.id, start_time.elapsed());

                    // If there's output, publish it
                    if let Some(output_event) = processing_result.output {
                        self.event_bus.publish(output_event).await?;
                    }
                } else {
                    // Handle processing failure with retry logic
                    self.handle_processing_failure(event, processing_result, &config).await?;
                }
            }
            Ok(Err(e)) => {
                error!("Processor error for event {}: {}", event.id, e);
                self.handle_processing_error(event, e, &config).await?;
            }
            Err(_) => {
                error!("Timeout processing event {}", event.id);
                self.handle_processing_timeout(event, &config).await?;
            }
        }

        Ok(())
    }

    async fn handle_processing_failure(
        &self,
        mut event: Event,
        result: ProcessingResult,
        config: &ProcessorConfig,
    ) -> Result<()> {
        event.increment_retry();

        if event.should_retry(config.retry_attempts) {
            info!("Retrying event {} (attempt {})", event.id, event.retry_count);

            // Exponential backoff
            let delay = Duration::from_millis(
                (1000.0 * config.backoff_multiplier.powi(event.retry_count as i32)) as u64
            );

            tokio::time::sleep(delay).await;
            self.event_bus.publish(event).await?;
        } else if config.enable_dead_letter_queue {
            warn!("Moving event {} to dead letter queue after {} attempts",
                  event.id, event.retry_count);
            self.send_to_dead_letter_queue(event, result.error).await?;
        }

        Ok(())
    }

    async fn handle_processing_error(
        &self,
        event: Event,
        error: anyhow::Error,
        config: &ProcessorConfig,
    ) -> Result<()> {
        if config.enable_dead_letter_queue {
            self.send_to_dead_letter_queue(event, Some(error.to_string())).await?;
        }
        Ok(())
    }

    async fn handle_processing_timeout(
        &self,
        event: Event,
        config: &ProcessorConfig,
    ) -> Result<()> {
        if config.enable_dead_letter_queue {
            self.send_to_dead_letter_queue(event, Some("Processing timeout".to_string())).await?;
        }
        Ok(())
    }

    async fn send_to_dead_letter_queue(&self, event: Event, error: Option<String>) -> Result<()> {
        let dlq_event = Event::new(
            crate::types::EventType::SystemEvent,
            "dead_letter_queue".to_string(),
            crate::types::EventData::System(crate::types::SystemEventData {
                component: "event_pipeline".to_string(),
                message: format!("Dead letter: {}", error.unwrap_or_else(|| "Unknown error".to_string())),
                level: crate::types::LogLevel::Error,
                error_code: Some("DLQ_001".to_string()),
                additional_data: {
                    let mut data = std::collections::HashMap::new();
                    data.insert("original_event".to_string(), serde_json::to_value(&event)?);
                    data
                },
            }),
        );

        // Would publish to deadletter-events topic
        self.event_bus.publish(dlq_event).await?;
        Ok(())
    }

    async fn should_slow_down(&self) -> bool {
        let queue_size = self.current_queue_size.load(Ordering::Relaxed);
        let max_size = self.backpressure_config.max_queue_size;

        let utilization = queue_size as f32 / max_size as f32;
        utilization > self.backpressure_config.warning_threshold
    }

    async fn start_backpressure_monitor(&self) {
        let current_queue_size = Arc::clone(&self.current_queue_size);
        let config = self.backpressure_config.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(5));

            loop {
                interval.tick().await;

                let queue_size = current_queue_size.load(Ordering::Relaxed);
                let utilization = queue_size as f32 / config.max_queue_size as f32;

                if utilization > config.critical_threshold {
                    error!("Critical backpressure: queue {}% full ({}/{})",
                           (utilization * 100.0) as u32, queue_size, config.max_queue_size);
                } else if utilization > config.warning_threshold {
                    warn!("Backpressure warning: queue {}% full ({}/{})",
                          (utilization * 100.0) as u32, queue_size, config.max_queue_size);
                }
            }
        });
    }

    async fn start_metrics_collector(&self) {
        let total_processed = Arc::clone(&self.total_processed);
        let total_failed = Arc::clone(&self.total_failed);
        let processing_rate = Arc::clone(&self.processing_rate);
        let current_queue_size = Arc::clone(&self.current_queue_size);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(10));
            let mut last_processed = 0u64;

            loop {
                interval.tick().await;

                let current_processed = total_processed.load(Ordering::Relaxed);
                let throughput = (current_processed - last_processed) as f64 / 10.0;
                last_processed = current_processed;

                // Reset rate counter
                processing_rate.store(0, Ordering::Relaxed);

                debug!(
                    "Pipeline metrics - Processed: {}, Failed: {}, Queue: {}, Throughput: {:.1}/s",
                    current_processed,
                    total_failed.load(Ordering::Relaxed),
                    current_queue_size.load(Ordering::Relaxed),
                    throughput
                );
            }
        });
    }

    pub async fn get_metrics(&self) -> PipelineMetrics {
        let uptime = self.start_time.elapsed().as_secs();
        let total_processed = self.total_processed.load(Ordering::Relaxed);
        let total_failed = self.total_failed.load(Ordering::Relaxed);
        let throughput = if uptime > 0 {
            total_processed as f64 / uptime as f64
        } else {
            0.0
        };

        PipelineMetrics {
            total_processed,
            total_failed,
            queue_size: self.current_queue_size.load(Ordering::Relaxed),
            throughput_per_second: throughput,
            average_processing_time_ms: 0.0, // Would track this in production
            error_rate: if total_processed + total_failed > 0 {
                total_failed as f32 / (total_processed + total_failed) as f32
            } else {
                0.0
            },
            active_workers: self.worker_threads,
            uptime_seconds: uptime,
        }
    }

    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down Event Pipeline...");
        self.shutdown_signal.store(true, Ordering::Relaxed);

        // Give workers time to finish current batches
        tokio::time::sleep(Duration::from_secs(5)).await;

        info!("Event Pipeline shutdown complete");
        Ok(())
    }

    fn clone_for_worker(&self) -> EventPipeline {
        EventPipeline {
            event_bus: Arc::clone(&self.event_bus),
            processor_registry: Arc::clone(&self.processor_registry),
            processing_semaphore: Arc::clone(&self.processing_semaphore),
            shutdown_signal: Arc::clone(&self.shutdown_signal),
            backpressure_config: self.backpressure_config.clone(),
            current_queue_size: Arc::clone(&self.current_queue_size),
            processing_rate: Arc::clone(&self.processing_rate),
            total_processed: Arc::clone(&self.total_processed),
            total_failed: Arc::clone(&self.total_failed),
            start_time: self.start_time,
            worker_threads: self.worker_threads,
            batch_size: self.batch_size,
            flush_interval: self.flush_interval,
        }
    }
}