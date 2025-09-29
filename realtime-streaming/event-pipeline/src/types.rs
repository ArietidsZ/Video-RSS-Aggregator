use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub id: Uuid,
    pub event_type: EventType,
    pub timestamp: DateTime<Utc>,
    pub source: String,
    pub data: EventData,
    pub metadata: HashMap<String, String>,
    pub retry_count: u32,
    pub processing_deadline: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum EventType {
    VideoUploaded,
    VideoProcessingStarted,
    VideoProcessingCompleted,
    VideoProcessingFailed,
    TranscriptionCompleted,
    SummarizationCompleted,
    RSSFeedUpdated,
    UserRequest,
    SystemEvent,
    MetricsUpdate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EventData {
    Video(VideoEventData),
    Transcription(TranscriptionEventData),
    Summary(SummaryEventData),
    RSS(RSSEventData),
    User(UserEventData),
    System(SystemEventData),
    Metrics(MetricsEventData),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoEventData {
    pub video_id: String,
    pub url: String,
    pub title: String,
    pub duration: u64,
    pub format: String,
    pub resolution: String,
    pub file_size: u64,
    pub platform: String,
    pub channel: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionEventData {
    pub video_id: String,
    pub transcript: String,
    pub confidence: f32,
    pub language: String,
    pub segments: Vec<TranscriptSegment>,
    pub processing_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptSegment {
    pub start_time: f64,
    pub end_time: f64,
    pub text: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryEventData {
    pub video_id: String,
    pub summary: String,
    pub key_points: Vec<String>,
    pub topics: Vec<String>,
    pub sentiment: String,
    pub length_reduction_ratio: f32,
    pub processing_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RSSEventData {
    pub feed_id: String,
    pub video_id: String,
    pub title: String,
    pub summary: String,
    pub published_at: DateTime<Utc>,
    pub categories: Vec<String>,
    pub feed_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserEventData {
    pub user_id: String,
    pub action: String,
    pub resource: String,
    pub ip_address: String,
    pub user_agent: String,
    pub session_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemEventData {
    pub component: String,
    pub message: String,
    pub level: LogLevel,
    pub error_code: Option<String>,
    pub additional_data: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsEventData {
    pub metric_name: String,
    pub value: f64,
    pub tags: HashMap<String, String>,
    pub unit: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingResult {
    pub event_id: Uuid,
    pub success: bool,
    pub output: Option<Event>,
    pub error: Option<String>,
    pub processing_time_ms: u64,
    pub retry_after: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineMetrics {
    pub total_processed: u64,
    pub total_failed: u64,
    pub queue_size: usize,
    pub throughput_per_second: f64,
    pub average_processing_time_ms: f64,
    pub error_rate: f32,
    pub active_workers: usize,
    pub uptime_seconds: u64,
}

#[derive(Debug, Clone)]
pub enum ProcessorType {
    VideoProcessor,
    TranscriptionProcessor,
    SummarizationProcessor,
    RSSGenerator,
    MetricsCollector,
    UserActionHandler,
    SystemEventHandler,
}

#[derive(Debug, Clone)]
pub struct ProcessorConfig {
    pub processor_type: ProcessorType,
    pub max_concurrency: usize,
    pub timeout_ms: u64,
    pub retry_attempts: u32,
    pub backoff_multiplier: f64,
    pub enable_dead_letter_queue: bool,
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            processor_type: ProcessorType::SystemEventHandler,
            max_concurrency: 10,
            timeout_ms: 30000,
            retry_attempts: 3,
            backoff_multiplier: 2.0,
            enable_dead_letter_queue: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackpressureConfig {
    pub max_queue_size: usize,
    pub warning_threshold: f32,    // 0.8 = warn at 80% capacity
    pub critical_threshold: f32,   // 0.95 = critical at 95% capacity
    pub slow_down_factor: f32,     // 0.5 = reduce processing rate by 50%
    pub enable_circuit_breaker: bool,
    pub circuit_breaker_threshold: f32, // 0.1 = trip at 10% error rate
    pub circuit_breaker_timeout_ms: u64,
}

impl Default for BackpressureConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 10000,
            warning_threshold: 0.8,
            critical_threshold: 0.95,
            slow_down_factor: 0.5,
            enable_circuit_breaker: true,
            circuit_breaker_threshold: 0.1,
            circuit_breaker_timeout_ms: 30000,
        }
    }
}

impl Event {
    pub fn new(event_type: EventType, source: String, data: EventData) -> Self {
        Self {
            id: Uuid::new_v4(),
            event_type,
            timestamp: Utc::now(),
            source,
            data,
            metadata: HashMap::new(),
            retry_count: 0,
            processing_deadline: None,
        }
    }

    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    pub fn with_deadline(mut self, deadline: DateTime<Utc>) -> Self {
        self.processing_deadline = Some(deadline);
        self
    }

    pub fn is_expired(&self) -> bool {
        if let Some(deadline) = self.processing_deadline {
            Utc::now() > deadline
        } else {
            false
        }
    }

    pub fn increment_retry(&mut self) {
        self.retry_count += 1;
    }

    pub fn should_retry(&self, max_retries: u32) -> bool {
        self.retry_count < max_retries && !self.is_expired()
    }
}