use anyhow::Result;
use chrono::{DateTime, Utc};
use redis::AsyncCommands;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, Row};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use sysinfo::{System, SystemExt, CpuExt, ProcessExt};
use tracing::{debug, info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub timestamp: DateTime<Utc>,
    pub cpu_usage: f32,
    pub memory_usage: f32,
    pub disk_usage: f32,
    pub network_io: NetworkMetrics,
    pub gpu_metrics: Option<GpuMetrics>,
    pub process_metrics: HashMap<String, ProcessMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    pub bytes_sent_per_sec: u64,
    pub bytes_recv_per_sec: u64,
    pub packets_sent_per_sec: u64,
    pub packets_recv_per_sec: u64,
    pub errors_per_sec: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMetrics {
    pub utilization: f32,
    pub memory_used: u64,
    pub memory_total: u64,
    pub temperature: f32,
    pub power_usage: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessMetrics {
    pub cpu_usage: f32,
    pub memory_usage: u64,
    pub disk_read_bytes: u64,
    pub disk_write_bytes: u64,
    pub thread_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApplicationMetrics {
    pub api_requests_per_sec: f64,
    pub api_response_time_p99: f64,
    pub error_rate: f64,
    pub active_connections: u64,
    pub queue_size: u64,
    pub cache_hit_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentMetrics {
    pub video_processor: ComponentHealth,
    pub transcription_engine: ComponentHealth,
    pub summarization_engine: ComponentHealth,
    pub rss_server: ComponentHealth,
    pub database: ComponentHealth,
    pub redis: ComponentHealth,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub status: String,
    pub response_time_ms: f64,
    pub throughput: f64,
    pub error_rate: f64,
    pub resource_usage: f32,
}

pub struct MetricsCollector {
    http_client: Client,
    prometheus_url: String,
    redis: redis::aio::ConnectionManager,
    database: PgPool,
    system: System,

    // Previous measurements for rate calculations
    last_network_stats: Option<(Instant, sysinfo::Networks)>,
}

impl MetricsCollector {
    pub async fn new(
        prometheus_url: &str,
        redis_url: &str,
        database_url: &str,
    ) -> Result<Self> {
        info!("Initializing metrics collector...");

        // Initialize HTTP client
        let http_client = Client::new();

        // Connect to Redis
        let redis_client = redis::Client::open(redis_url)?;
        let redis = redis::aio::ConnectionManager::new(redis_client).await?;

        // Connect to database
        let database = PgPool::connect(database_url).await?;

        // Initialize system monitoring
        let mut system = System::new_all();
        system.refresh_all();

        Ok(Self {
            http_client,
            prometheus_url: prometheus_url.to_string(),
            redis,
            database,
            system,
            last_network_stats: None,
        })
    }

    pub async fn collect_all_metrics(&mut self) -> Result<()> {
        debug!("Collecting all metrics...");

        // Collect system metrics
        let system_metrics = self.collect_system_metrics().await?;

        // Collect application metrics from Prometheus
        let app_metrics = self.collect_application_metrics().await?;

        // Collect component health metrics
        let component_metrics = self.collect_component_metrics().await?;

        // Store metrics in Redis for real-time access
        self.store_metrics_redis(&system_metrics, &app_metrics, &component_metrics).await?;

        // Store metrics in database for historical analysis
        self.store_metrics_database(&system_metrics, &app_metrics, &component_metrics).await?;

        debug!("Metrics collection completed");
        Ok(())
    }

    async fn collect_system_metrics(&mut self) -> Result<SystemMetrics> {
        // Refresh system information
        self.system.refresh_all();

        // CPU usage
        let cpu_usage = self.system.global_cpu_info().cpu_usage();

        // Memory usage
        let memory_usage = (self.system.used_memory() as f32 / self.system.total_memory() as f32) * 100.0;

        // Disk usage (simplified - would normally check all disks)
        let disk_usage = 0.0; // Placeholder - would calculate actual disk usage

        // Network metrics
        let network_metrics = self.calculate_network_metrics();

        // GPU metrics
        let gpu_metrics = self.collect_gpu_metrics().await;

        // Process metrics for our components
        let process_metrics = self.collect_process_metrics();

        Ok(SystemMetrics {
            timestamp: Utc::now(),
            cpu_usage,
            memory_usage,
            disk_usage,
            network_io: network_metrics,
            gpu_metrics,
            process_metrics,
        })
    }

    fn calculate_network_metrics(&mut self) -> NetworkMetrics {
        let networks = self.system.networks();
        let now = Instant::now();

        let current_stats = networks.clone();

        if let Some((last_time, ref last_stats)) = &self.last_network_stats {
            let duration = now.duration_since(*last_time).as_secs_f64();

            if duration > 0.0 {
                // Calculate rates (simplified - would sum across all interfaces)
                let bytes_sent_per_sec = (current_stats.iter().map(|(_, data)| data.total_transmitted()).sum::<u64>()
                    - last_stats.iter().map(|(_, data)| data.total_transmitted()).sum::<u64>())
                    / duration as u64;

                let bytes_recv_per_sec = (current_stats.iter().map(|(_, data)| data.total_received()).sum::<u64>()
                    - last_stats.iter().map(|(_, data)| data.total_received()).sum::<u64>())
                    / duration as u64;

                self.last_network_stats = Some((now, current_stats));

                return NetworkMetrics {
                    bytes_sent_per_sec,
                    bytes_recv_per_sec,
                    packets_sent_per_sec: 0, // Would calculate from detailed network stats
                    packets_recv_per_sec: 0,
                    errors_per_sec: 0,
                };
            }
        }

        self.last_network_stats = Some((now, current_stats));

        NetworkMetrics {
            bytes_sent_per_sec: 0,
            bytes_recv_per_sec: 0,
            packets_sent_per_sec: 0,
            packets_recv_per_sec: 0,
            errors_per_sec: 0,
        }
    }

    async fn collect_gpu_metrics(&self) -> Option<GpuMetrics> {
        // Try to get GPU metrics via nvidia-ml-py equivalent or nvidia-smi
        match self.query_nvidia_smi().await {
            Ok(metrics) => Some(metrics),
            Err(e) => {
                debug!("GPU metrics unavailable: {}", e);
                None
            }
        }
    }

    async fn query_nvidia_smi(&self) -> Result<GpuMetrics> {
        use tokio::process::Command;

        let output = Command::new("nvidia-smi")
            .args(&[
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits"
            ])
            .output()
            .await?;

        if output.status.success() {
            let stdout = String::from_utf8(output.stdout)?;
            let line = stdout.lines().next().ok_or_else(|| anyhow::anyhow!("No GPU data"))?;
            let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();

            if parts.len() >= 5 {
                Ok(GpuMetrics {
                    utilization: parts[0].parse()?,
                    memory_used: parts[1].parse::<u64>()? * 1024 * 1024, // Convert MB to bytes
                    memory_total: parts[2].parse::<u64>()? * 1024 * 1024,
                    temperature: parts[3].parse()?,
                    power_usage: parts[4].parse()?,
                })
            } else {
                Err(anyhow::anyhow!("Invalid GPU data format"))
            }
        } else {
            Err(anyhow::anyhow!("nvidia-smi failed"))
        }
    }

    fn collect_process_metrics(&self) -> HashMap<String, ProcessMetrics> {
        let mut metrics = HashMap::new();

        let target_processes = [
            "video-processor",
            "transcription-engine",
            "summarization-engine",
            "rss-server",
            "redis-server",
            "postgres"
        ];

        for process in self.system.processes().values() {
            let process_name = process.name();

            if target_processes.iter().any(|&target| process_name.contains(target)) {
                metrics.insert(process_name.to_string(), ProcessMetrics {
                    cpu_usage: process.cpu_usage(),
                    memory_usage: process.memory(),
                    disk_read_bytes: process.disk_usage().read_bytes,
                    disk_write_bytes: process.disk_usage().written_bytes,
                    thread_count: process.tasks.len(),
                });
            }
        }

        metrics
    }

    async fn collect_application_metrics(&self) -> Result<ApplicationMetrics> {
        // Query Prometheus for application metrics
        let queries = vec![
            ("rate(http_requests_total[1m])", "api_requests_per_sec"),
            ("histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))", "api_response_time_p99"),
            ("rate(http_requests_total{status=~\"5..\"}[1m])", "error_rate"),
            ("active_connections", "active_connections"),
            ("queue_size", "queue_size"),
            ("cache_hit_rate", "cache_hit_rate"),
        ];

        let mut metrics = HashMap::new();

        for (query, name) in queries {
            match self.query_prometheus(query).await {
                Ok(value) => {
                    metrics.insert(name.to_string(), value);
                }
                Err(e) => {
                    warn!("Failed to query {}: {}", name, e);
                    metrics.insert(name.to_string(), 0.0);
                }
            }
        }

        Ok(ApplicationMetrics {
            api_requests_per_sec: metrics.get("api_requests_per_sec").unwrap_or(&0.0).clone(),
            api_response_time_p99: metrics.get("api_response_time_p99").unwrap_or(&0.0).clone(),
            error_rate: metrics.get("error_rate").unwrap_or(&0.0).clone(),
            active_connections: metrics.get("active_connections").unwrap_or(&0.0).clone() as u64,
            queue_size: metrics.get("queue_size").unwrap_or(&0.0).clone() as u64,
            cache_hit_rate: metrics.get("cache_hit_rate").unwrap_or(&0.0).clone(),
        })
    }

    async fn query_prometheus(&self, query: &str) -> Result<f64> {
        let url = format!("{}/api/v1/query", self.prometheus_url);
        let response = self.http_client
            .get(&url)
            .query(&[("query", query)])
            .send()
            .await?;

        let json: serde_json::Value = response.json().await?;

        if let Some(result) = json["data"]["result"].as_array() {
            if let Some(first_result) = result.first() {
                if let Some(value_array) = first_result["value"].as_array() {
                    if let Some(value_str) = value_array.get(1).and_then(|v| v.as_str()) {
                        return Ok(value_str.parse()?);
                    }
                }
            }
        }

        Ok(0.0)
    }

    async fn collect_component_metrics(&self) -> Result<ComponentMetrics> {
        // Health check endpoints for each component
        let components = vec![
            ("video_processor", "http://video-processor:8001/health"),
            ("transcription_engine", "http://transcription-engine:8003/health"),
            ("summarization_engine", "http://summarization-engine:8005/health"),
            ("rss_server", "http://rss-server:8006/health"),
        ];

        let mut component_metrics = ComponentMetrics {
            video_processor: ComponentHealth::default(),
            transcription_engine: ComponentHealth::default(),
            summarization_engine: ComponentHealth::default(),
            rss_server: ComponentHealth::default(),
            database: ComponentHealth::default(),
            redis: ComponentHealth::default(),
        };

        for (component, url) in components {
            let health = self.check_component_health(url).await;

            match component {
                "video_processor" => component_metrics.video_processor = health,
                "transcription_engine" => component_metrics.transcription_engine = health,
                "summarization_engine" => component_metrics.summarization_engine = health,
                "rss_server" => component_metrics.rss_server = health,
                _ => {}
            }
        }

        // Check database health
        component_metrics.database = self.check_database_health().await;

        // Check Redis health
        component_metrics.redis = self.check_redis_health().await;

        Ok(component_metrics)
    }

    async fn check_component_health(&self, url: &str) -> ComponentHealth {
        let start = Instant::now();

        match self.http_client.get(url).send().await {
            Ok(response) => {
                let response_time = start.elapsed().as_millis() as f64;
                let status = if response.status().is_success() {
                    "healthy".to_string()
                } else {
                    "unhealthy".to_string()
                };

                ComponentHealth {
                    status,
                    response_time_ms: response_time,
                    throughput: 0.0, // Would be calculated from metrics
                    error_rate: 0.0,
                    resource_usage: 0.0,
                }
            }
            Err(_) => ComponentHealth {
                status: "unhealthy".to_string(),
                response_time_ms: -1.0,
                throughput: 0.0,
                error_rate: 1.0,
                resource_usage: 0.0,
            }
        }
    }

    async fn check_database_health(&self) -> ComponentHealth {
        let start = Instant::now();

        match sqlx::query("SELECT 1").fetch_one(&self.database).await {
            Ok(_) => ComponentHealth {
                status: "healthy".to_string(),
                response_time_ms: start.elapsed().as_millis() as f64,
                throughput: 0.0,
                error_rate: 0.0,
                resource_usage: 0.0,
            },
            Err(_) => ComponentHealth {
                status: "unhealthy".to_string(),
                response_time_ms: -1.0,
                throughput: 0.0,
                error_rate: 1.0,
                resource_usage: 0.0,
            }
        }
    }

    async fn check_redis_health(&self) -> ComponentHealth {
        let start = Instant::now();
        let mut redis = self.redis.clone();

        match redis.ping::<String>().await {
            Ok(_) => ComponentHealth {
                status: "healthy".to_string(),
                response_time_ms: start.elapsed().as_millis() as f64,
                throughput: 0.0,
                error_rate: 0.0,
                resource_usage: 0.0,
            },
            Err(_) => ComponentHealth {
                status: "unhealthy".to_string(),
                response_time_ms: -1.0,
                throughput: 0.0,
                error_rate: 1.0,
                resource_usage: 0.0,
            }
        }
    }

    async fn store_metrics_redis(
        &self,
        system_metrics: &SystemMetrics,
        app_metrics: &ApplicationMetrics,
        component_metrics: &ComponentMetrics,
    ) -> Result<()> {
        let mut redis = self.redis.clone();

        // Store current metrics with 1-hour TTL
        let system_json = serde_json::to_string(system_metrics)?;
        let app_json = serde_json::to_string(app_metrics)?;
        let component_json = serde_json::to_string(component_metrics)?;

        redis.set_ex::<_, _, ()>("metrics:system:current", system_json, 3600).await?;
        redis.set_ex::<_, _, ()>("metrics:application:current", app_json, 3600).await?;
        redis.set_ex::<_, _, ()>("metrics:components:current", component_json, 3600).await?;

        // Store time series data (last 24 hours)
        let timestamp = system_metrics.timestamp.timestamp();
        redis.zadd::<_, _, _, ()>("metrics:system:timeseries", system_json, timestamp).await?;
        redis.zremrangebyscore::<_, _, _>("metrics:system:timeseries", 0, timestamp - 86400).await?;

        debug!("Metrics stored in Redis");
        Ok(())
    }

    async fn store_metrics_database(
        &self,
        system_metrics: &SystemMetrics,
        app_metrics: &ApplicationMetrics,
        component_metrics: &ComponentMetrics,
    ) -> Result<()> {
        // Store in database for long-term analysis
        sqlx::query(
            "INSERT INTO performance_metrics (
                timestamp, cpu_usage, memory_usage, api_requests_per_sec,
                api_response_time_p99, error_rate, active_connections
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)"
        )
        .bind(system_metrics.timestamp)
        .bind(system_metrics.cpu_usage)
        .bind(system_metrics.memory_usage)
        .bind(app_metrics.api_requests_per_sec)
        .bind(app_metrics.api_response_time_p99)
        .bind(app_metrics.error_rate)
        .bind(app_metrics.active_connections as i64)
        .execute(&self.database)
        .await?;

        debug!("Metrics stored in database");
        Ok(())
    }

    pub async fn get_current_metrics(&self) -> Result<serde_json::Value> {
        let mut redis = self.redis.clone();

        let system_json: Option<String> = redis.get("metrics:system:current").await?;
        let app_json: Option<String> = redis.get("metrics:application:current").await?;
        let component_json: Option<String> = redis.get("metrics:components:current").await?;

        let metrics = serde_json::json!({
            "system": system_json.and_then(|s| serde_json::from_str::<SystemMetrics>(&s).ok()),
            "application": app_json.and_then(|s| serde_json::from_str::<ApplicationMetrics>(&s).ok()),
            "components": component_json.and_then(|s| serde_json::from_str::<ComponentMetrics>(&s).ok())
        });

        Ok(metrics)
    }
}

impl Default for ComponentHealth {
    fn default() -> Self {
        Self {
            status: "unknown".to_string(),
            response_time_ms: 0.0,
            throughput: 0.0,
            error_rate: 0.0,
            resource_usage: 0.0,
        }
    }
}