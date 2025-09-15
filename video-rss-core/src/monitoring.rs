use crate::{error::VideoRssError, Result};
use opentelemetry::{
    global,
    metrics::{Counter, Histogram, Meter, ObservableGauge},
    trace::{Tracer, TracerProvider, Span, SpanKind, Status},
    KeyValue,
};
use opentelemetry_otlp::{WithExportConfig, SpanExporter};
use opentelemetry_sdk::{
    metrics::{MeterProvider, PeriodicReader},
    trace::{self, RandomIdGenerator, Sampler},
    Resource,
};
use prometheus::{Encoder, TextEncoder, Registry};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use sysinfo::{System, SystemExt, ProcessExt, CpuExt};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub enable_traces: bool,
    pub enable_metrics: bool,
    pub enable_logs: bool,
    pub otlp_endpoint: String,
    pub service_name: String,
    pub service_version: String,
    pub sampling_rate: f64,
    pub metric_interval: Duration,
    pub export_timeout: Duration,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_traces: true,
            enable_metrics: true,
            enable_logs: true,
            otlp_endpoint: "http://localhost:4317".to_string(),
            service_name: "video-rss-core".to_string(),
            service_version: env!("CARGO_PKG_VERSION").to_string(),
            sampling_rate: 1.0,
            metric_interval: Duration::from_secs(10),
            export_timeout: Duration::from_secs(5),
        }
    }
}

pub struct MonitoringSystem {
    config: MonitoringConfig,
    meter: Meter,
    tracer: Box<dyn Tracer>,
    registry: Registry,
    system: Arc<RwLock<System>>,

    // Metrics
    request_counter: Counter<u64>,
    request_duration: Histogram<f64>,
    active_connections: ObservableGauge<i64>,
    transcription_counter: Counter<u64>,
    transcription_duration: Histogram<f64>,
    cache_hits: Counter<u64>,
    cache_misses: Counter<u64>,
    error_counter: Counter<u64>,
}

impl MonitoringSystem {
    pub async fn new(config: MonitoringConfig) -> Result<Self> {
        info!("Initializing monitoring system");

        // Create resource
        let resource = Resource::new(vec![
            KeyValue::new("service.name", config.service_name.clone()),
            KeyValue::new("service.version", config.service_version.clone()),
            KeyValue::new("deployment.environment", "production"),
        ]);

        // Initialize tracer
        let tracer = if config.enable_traces {
            let exporter = SpanExporter::builder()
                .with_endpoint(&config.otlp_endpoint)
                .with_timeout(config.export_timeout)
                .build()
                .map_err(|e| VideoRssError::Config(format!("OTLP exporter error: {}", e)))?;

            let provider = trace::TracerProvider::builder()
                .with_batch_exporter(exporter, trace::runtime::Tokio)
                .with_sampler(Sampler::TraceIdRatioBased(config.sampling_rate))
                .with_id_generator(RandomIdGenerator::default())
                .with_resource(resource.clone())
                .build();

            global::set_tracer_provider(provider.clone());
            provider.tracer("video-rss-core")
        } else {
            global::tracer("noop")
        };

        // Initialize meter
        let meter = if config.enable_metrics {
            let exporter = opentelemetry_otlp::new_exporter()
                .tonic()
                .with_endpoint(&config.otlp_endpoint)
                .with_timeout(config.export_timeout)
                .build_metrics_exporter()
                .map_err(|e| VideoRssError::Config(format!("Metrics exporter error: {}", e)))?;

            let reader = PeriodicReader::builder(exporter, opentelemetry_sdk::runtime::Tokio)
                .with_interval(config.metric_interval)
                .build();

            let provider = MeterProvider::builder()
                .with_reader(reader)
                .with_resource(resource)
                .build();

            global::set_meter_provider(provider.clone());
            provider.meter("video-rss-core")
        } else {
            global::meter("noop")
        };

        // Create metrics
        let request_counter = meter
            .u64_counter("http.request.count")
            .with_description("Total number of HTTP requests")
            .init();

        let request_duration = meter
            .f64_histogram("http.request.duration")
            .with_description("HTTP request duration in milliseconds")
            .with_unit("ms")
            .init();

        let active_connections = meter
            .i64_observable_gauge("connections.active")
            .with_description("Number of active connections")
            .init();

        let transcription_counter = meter
            .u64_counter("transcription.count")
            .with_description("Total number of transcriptions")
            .init();

        let transcription_duration = meter
            .f64_histogram("transcription.duration")
            .with_description("Transcription duration in milliseconds")
            .with_unit("ms")
            .init();

        let cache_hits = meter
            .u64_counter("cache.hits")
            .with_description("Total cache hits")
            .init();

        let cache_misses = meter
            .u64_counter("cache.misses")
            .with_description("Total cache misses")
            .init();

        let error_counter = meter
            .u64_counter("errors.total")
            .with_description("Total number of errors")
            .init();

        // Initialize Prometheus registry
        let registry = Registry::new();

        // Initialize system info
        let system = Arc::new(RwLock::new(System::new_all()));

        // Start system metrics collector
        let system_clone = system.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(5));
            loop {
                interval.tick().await;
                let mut sys = system_clone.write().await;
                sys.refresh_all();
            }
        });

        info!("Monitoring system initialized");

        Ok(Self {
            config,
            meter,
            tracer: Box::new(tracer),
            registry,
            system,
            request_counter,
            request_duration,
            active_connections,
            transcription_counter,
            transcription_duration,
            cache_hits,
            cache_misses,
            error_counter,
        })
    }

    pub fn start_span(&self, name: &str) -> Span {
        self.tracer.span_builder(name)
            .with_kind(SpanKind::Internal)
            .start(&*self.tracer)
    }

    pub fn start_http_span(&self, method: &str, path: &str) -> Span {
        self.tracer.span_builder(format!("{} {}", method, path))
            .with_kind(SpanKind::Server)
            .with_attributes(vec![
                KeyValue::new("http.method", method.to_string()),
                KeyValue::new("http.target", path.to_string()),
            ])
            .start(&*self.tracer)
    }

    pub fn record_request(&self, method: &str, path: &str, status: u16, duration_ms: f64) {
        let attributes = vec![
            KeyValue::new("http.method", method.to_string()),
            KeyValue::new("http.target", path.to_string()),
            KeyValue::new("http.status_code", status as i64),
        ];

        self.request_counter.add(1, &attributes);
        self.request_duration.record(duration_ms, &attributes);
    }

    pub fn record_transcription(&self, model: &str, duration_ms: f64, success: bool) {
        let attributes = vec![
            KeyValue::new("model", model.to_string()),
            KeyValue::new("success", success),
        ];

        self.transcription_counter.add(1, &attributes);
        self.transcription_duration.record(duration_ms, &attributes);
    }

    pub fn record_cache_hit(&self, tier: &str) {
        self.cache_hits.add(1, &[KeyValue::new("tier", tier.to_string())]);
    }

    pub fn record_cache_miss(&self, tier: &str) {
        self.cache_misses.add(1, &[KeyValue::new("tier", tier.to_string())]);
    }

    pub fn record_error(&self, error_type: &str) {
        self.error_counter.add(1, &[KeyValue::new("type", error_type.to_string())]);
    }

    pub async fn get_system_metrics(&self) -> SystemMetrics {
        let system = self.system.read().await;

        let cpu_usage = system.global_cpu_info().cpu_usage();
        let memory_used = system.used_memory();
        let memory_total = system.total_memory();
        let memory_usage = (memory_used as f64 / memory_total as f64) * 100.0;

        let process = std::process::id();
        let process_metrics = system.process(sysinfo::Pid::from(process as usize))
            .map(|p| ProcessMetrics {
                cpu_usage: p.cpu_usage(),
                memory_mb: p.memory() / 1024 / 1024,
                threads: p.tasks.len(),
                open_files: 0,  // Would need platform-specific implementation
            })
            .unwrap_or_default();

        SystemMetrics {
            cpu_usage,
            memory_usage,
            memory_used_mb: memory_used / 1024 / 1024,
            memory_total_mb: memory_total / 1024 / 1024,
            disk_usage: Self::get_disk_usage(),
            network_rx_mbps: 0.0,  // Would need actual network monitoring
            network_tx_mbps: 0.0,
            process: process_metrics,
        }
    }

    fn get_disk_usage() -> f64 {
        // Simplified disk usage calculation
        // In production, would use actual disk monitoring
        45.0
    }

    pub fn export_prometheus_metrics(&self) -> String {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer).unwrap();
        String::from_utf8(buffer).unwrap()
    }

    pub async fn shutdown(&self) {
        info!("Shutting down monitoring system");
        global::shutdown_tracer_provider();
    }
}

// Performance profiler for detailed analysis
pub struct Profiler {
    name: String,
    start_time: Instant,
    checkpoints: Vec<(String, Duration)>,
    tags: Vec<(String, String)>,
}

impl Profiler {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            start_time: Instant::now(),
            checkpoints: Vec::new(),
            tags: Vec::new(),
        }
    }

    pub fn checkpoint(&mut self, name: impl Into<String>) {
        let elapsed = self.start_time.elapsed();
        self.checkpoints.push((name.into(), elapsed));
    }

    pub fn tag(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.tags.push((key.into(), value.into()));
    }

    pub fn finish(self) -> ProfileResult {
        let total_duration = self.start_time.elapsed();

        ProfileResult {
            name: self.name,
            total_duration,
            checkpoints: self.checkpoints,
            tags: self.tags,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileResult {
    pub name: String,
    pub total_duration: Duration,
    pub checkpoints: Vec<(String, Duration)>,
    pub tags: Vec<(String, String)>,
}

impl ProfileResult {
    pub fn to_flame_graph(&self) -> String {
        // Generate flame graph compatible output
        let mut output = String::new();

        for (i, (name, duration)) in self.checkpoints.iter().enumerate() {
            let prev_duration = if i > 0 {
                self.checkpoints[i - 1].1
            } else {
                Duration::ZERO
            };

            let segment_duration = *duration - prev_duration;
            output.push_str(&format!(
                "{};{} {}\n",
                self.name,
                name,
                segment_duration.as_micros()
            ));
        }

        output
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub cpu_usage: f32,
    pub memory_usage: f64,
    pub memory_used_mb: u64,
    pub memory_total_mb: u64,
    pub disk_usage: f64,
    pub network_rx_mbps: f64,
    pub network_tx_mbps: f64,
    pub process: ProcessMetrics,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProcessMetrics {
    pub cpu_usage: f32,
    pub memory_mb: u64,
    pub threads: usize,
    pub open_files: usize,
}

// Distributed tracing context
pub struct TracingContext {
    span: Span,
    attributes: Vec<KeyValue>,
}

impl TracingContext {
    pub fn new(span: Span) -> Self {
        Self {
            span,
            attributes: Vec::new(),
        }
    }

    pub fn add_attribute(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.attributes.push(KeyValue::new(key.into(), value.into()));
    }

    pub fn set_status(&mut self, status: Status) {
        self.span.set_status(status);
    }

    pub fn finish(mut self) {
        for attr in self.attributes {
            self.span.set_attribute(attr);
        }
        self.span.end();
    }
}

// Anomaly detection for automatic alerting
pub struct AnomalyDetector {
    window_size: usize,
    threshold: f64,
    history: Arc<RwLock<Vec<f64>>>,
}

impl AnomalyDetector {
    pub fn new(window_size: usize, threshold: f64) -> Self {
        Self {
            window_size,
            threshold,
            history: Arc::new(RwLock::new(Vec::with_capacity(window_size))),
        }
    }

    pub async fn check(&self, value: f64) -> bool {
        let mut history = self.history.write().await;

        if history.len() >= self.window_size {
            history.remove(0);
        }
        history.push(value);

        if history.len() < 3 {
            return false;
        }

        // Simple z-score based anomaly detection
        let mean: f64 = history.iter().sum::<f64>() / history.len() as f64;
        let variance: f64 = history.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / history.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return false;
        }

        let z_score = (value - mean).abs() / std_dev;
        z_score > self.threshold
    }
}