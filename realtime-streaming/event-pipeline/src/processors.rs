use crate::types::{Event, EventType, ProcessingResult};
use anyhow::Result;
use async_trait::async_trait;
use dashmap::DashMap;
use std::{sync::Arc, time::Instant};
use tracing::{debug, info};
use uuid::Uuid;

#[async_trait]
pub trait ProcessorTrait: Send + Sync {
    async fn process(&self, event: Event) -> Result<ProcessingResult>;
    fn get_name(&self) -> &str;
    fn get_supported_events(&self) -> Vec<EventType>;
}

pub struct ProcessorRegistry {
    processors: DashMap<EventType, Arc<dyn ProcessorTrait>>,
}

impl ProcessorRegistry {
    pub fn new() -> Self {
        Self {
            processors: DashMap::new(),
        }
    }

    pub async fn register_processor(
        &self,
        event_types: Vec<EventType>,
        processor: Arc<dyn ProcessorTrait>,
    ) -> Result<()> {
        for event_type in event_types {
            self.processors.insert(event_type, Arc::clone(&processor));
        }
        info!("Registered processor: {}", processor.get_name());
        Ok(())
    }

    pub async fn get_processor(&self, event_type: &EventType) -> Option<Arc<dyn ProcessorTrait>> {
        self.processors.get(event_type).map(|entry| Arc::clone(&*entry))
    }

    pub async fn count(&self) -> usize {
        self.processors.len()
    }

    // Register built-in processors
    pub async fn register_video_processor(&self) -> Result<()> {
        let processor = Arc::new(VideoProcessor::new());
        self.register_processor(
            vec![
                EventType::VideoUploaded,
                EventType::VideoProcessingStarted,
                EventType::VideoProcessingCompleted,
                EventType::VideoProcessingFailed,
            ],
            processor,
        ).await
    }

    pub async fn register_transcription_processor(&self) -> Result<()> {
        let processor = Arc::new(TranscriptionProcessor::new());
        self.register_processor(vec![EventType::TranscriptionCompleted], processor).await
    }

    pub async fn register_summarization_processor(&self) -> Result<()> {
        let processor = Arc::new(SummarizationProcessor::new());
        self.register_processor(vec![EventType::SummarizationCompleted], processor).await
    }

    pub async fn register_rss_processor(&self) -> Result<()> {
        let processor = Arc::new(RSSProcessor::new());
        self.register_processor(vec![EventType::RSSFeedUpdated], processor).await
    }

    pub async fn register_metrics_processor(&self) -> Result<()> {
        let processor = Arc::new(MetricsProcessor::new());
        self.register_processor(vec![EventType::MetricsUpdate], processor).await
    }

    pub async fn register_system_processor(&self) -> Result<()> {
        let processor = Arc::new(SystemEventProcessor::new());
        self.register_processor(vec![EventType::SystemEvent], processor).await
    }
}

// Video Processing Handler
pub struct VideoProcessor {
    name: String,
}

impl VideoProcessor {
    pub fn new() -> Self {
        Self {
            name: "VideoProcessor".to_string(),
        }
    }
}

#[async_trait]
impl ProcessorTrait for VideoProcessor {
    async fn process(&self, event: Event) -> Result<ProcessingResult> {
        let start_time = Instant::now();
        debug!("Processing video event: {}", event.id);

        match &event.data {
            crate::types::EventData::Video(video_data) => {
                match event.event_type {
                    EventType::VideoUploaded => {
                        // Trigger video processing pipeline
                        info!("Starting processing for video: {}", video_data.video_id);

                        let processing_event = Event::new(
                            EventType::VideoProcessingStarted,
                            "video_processor".to_string(),
                            crate::types::EventData::Video(video_data.clone()),
                        );

                        Ok(ProcessingResult {
                            event_id: event.id,
                            success: true,
                            output: Some(processing_event),
                            error: None,
                            processing_time_ms: start_time.elapsed().as_millis() as u64,
                            retry_after: None,
                        })
                    }
                    EventType::VideoProcessingStarted => {
                        // Log processing start
                        info!("Video processing started: {}", video_data.video_id);

                        // Simulate processing delay
                        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

                        Ok(ProcessingResult {
                            event_id: event.id,
                            success: true,
                            output: None,
                            error: None,
                            processing_time_ms: start_time.elapsed().as_millis() as u64,
                            retry_after: None,
                        })
                    }
                    EventType::VideoProcessingCompleted => {
                        info!("Video processing completed: {}", video_data.video_id);
                        Ok(ProcessingResult {
                            event_id: event.id,
                            success: true,
                            output: None,
                            error: None,
                            processing_time_ms: start_time.elapsed().as_millis() as u64,
                            retry_after: None,
                        })
                    }
                    EventType::VideoProcessingFailed => {
                        info!("Video processing failed: {}", video_data.video_id);
                        Ok(ProcessingResult {
                            event_id: event.id,
                            success: true,
                            output: None,
                            error: None,
                            processing_time_ms: start_time.elapsed().as_millis() as u64,
                            retry_after: None,
                        })
                    }
                    _ => {
                        Err(anyhow::anyhow!("Unsupported event type for VideoProcessor"))
                    }
                }
            }
            _ => {
                Err(anyhow::anyhow!("Invalid event data for VideoProcessor"))
            }
        }
    }

    fn get_name(&self) -> &str {
        &self.name
    }

    fn get_supported_events(&self) -> Vec<EventType> {
        vec![
            EventType::VideoUploaded,
            EventType::VideoProcessingStarted,
            EventType::VideoProcessingCompleted,
            EventType::VideoProcessingFailed,
        ]
    }
}

// Transcription Processing Handler
pub struct TranscriptionProcessor {
    name: String,
}

impl TranscriptionProcessor {
    pub fn new() -> Self {
        Self {
            name: "TranscriptionProcessor".to_string(),
        }
    }
}

#[async_trait]
impl ProcessorTrait for TranscriptionProcessor {
    async fn process(&self, event: Event) -> Result<ProcessingResult> {
        let start_time = Instant::now();
        debug!("Processing transcription event: {}", event.id);

        match &event.data {
            crate::types::EventData::Transcription(transcription_data) => {
                info!("Processing transcription for video: {}", transcription_data.video_id);

                // Simulate transcription processing
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;

                // Generate summarization trigger event
                let summarization_event = Event::new(
                    EventType::SummarizationCompleted,
                    "transcription_processor".to_string(),
                    crate::types::EventData::Summary(crate::types::SummaryEventData {
                        video_id: transcription_data.video_id.clone(),
                        summary: format!("Auto-generated summary for {}", transcription_data.video_id),
                        key_points: vec![
                            "Key point 1".to_string(),
                            "Key point 2".to_string(),
                        ],
                        topics: vec!["topic1".to_string(), "topic2".to_string()],
                        sentiment: "neutral".to_string(),
                        length_reduction_ratio: 0.1,
                        processing_time_ms: start_time.elapsed().as_millis() as u64,
                    }),
                );

                Ok(ProcessingResult {
                    event_id: event.id,
                    success: true,
                    output: Some(summarization_event),
                    error: None,
                    processing_time_ms: start_time.elapsed().as_millis() as u64,
                    retry_after: None,
                })
            }
            _ => {
                Err(anyhow::anyhow!("Invalid event data for TranscriptionProcessor"))
            }
        }
    }

    fn get_name(&self) -> &str {
        &self.name
    }

    fn get_supported_events(&self) -> Vec<EventType> {
        vec![EventType::TranscriptionCompleted]
    }
}

// Summarization Processing Handler
pub struct SummarizationProcessor {
    name: String,
}

impl SummarizationProcessor {
    pub fn new() -> Self {
        Self {
            name: "SummarizationProcessor".to_string(),
        }
    }
}

#[async_trait]
impl ProcessorTrait for SummarizationProcessor {
    async fn process(&self, event: Event) -> Result<ProcessingResult> {
        let start_time = Instant::now();
        debug!("Processing summarization event: {}", event.id);

        match &event.data {
            crate::types::EventData::Summary(summary_data) => {
                info!("Processing summary for video: {}", summary_data.video_id);

                // Simulate summarization processing
                tokio::time::sleep(std::time::Duration::from_millis(200)).await;

                // Generate RSS update event
                let rss_event = Event::new(
                    EventType::RSSFeedUpdated,
                    "summarization_processor".to_string(),
                    crate::types::EventData::RSS(crate::types::RSSEventData {
                        feed_id: format!("feed-{}", summary_data.video_id),
                        video_id: summary_data.video_id.clone(),
                        title: format!("Summary: {}", summary_data.video_id),
                        summary: summary_data.summary.clone(),
                        published_at: chrono::Utc::now(),
                        categories: summary_data.topics.clone(),
                        feed_url: format!("https://api.video-rss.com/feeds/{}", summary_data.video_id),
                    }),
                );

                Ok(ProcessingResult {
                    event_id: event.id,
                    success: true,
                    output: Some(rss_event),
                    error: None,
                    processing_time_ms: start_time.elapsed().as_millis() as u64,
                    retry_after: None,
                })
            }
            _ => {
                Err(anyhow::anyhow!("Invalid event data for SummarizationProcessor"))
            }
        }
    }

    fn get_name(&self) -> &str {
        &self.name
    }

    fn get_supported_events(&self) -> Vec<EventType> {
        vec![EventType::SummarizationCompleted]
    }
}

// RSS Processing Handler
pub struct RSSProcessor {
    name: String,
}

impl RSSProcessor {
    pub fn new() -> Self {
        Self {
            name: "RSSProcessor".to_string(),
        }
    }
}

#[async_trait]
impl ProcessorTrait for RSSProcessor {
    async fn process(&self, event: Event) -> Result<ProcessingResult> {
        let start_time = Instant::now();
        debug!("Processing RSS event: {}", event.id);

        match &event.data {
            crate::types::EventData::RSS(rss_data) => {
                info!("Updating RSS feed: {}", rss_data.feed_id);

                // Simulate RSS feed update
                tokio::time::sleep(std::time::Duration::from_millis(30)).await;

                Ok(ProcessingResult {
                    event_id: event.id,
                    success: true,
                    output: None,
                    error: None,
                    processing_time_ms: start_time.elapsed().as_millis() as u64,
                    retry_after: None,
                })
            }
            _ => {
                Err(anyhow::anyhow!("Invalid event data for RSSProcessor"))
            }
        }
    }

    fn get_name(&self) -> &str {
        &self.name
    }

    fn get_supported_events(&self) -> Vec<EventType> {
        vec![EventType::RSSFeedUpdated]
    }
}

// Metrics Processing Handler
pub struct MetricsProcessor {
    name: String,
}

impl MetricsProcessor {
    pub fn new() -> Self {
        Self {
            name: "MetricsProcessor".to_string(),
        }
    }
}

#[async_trait]
impl ProcessorTrait for MetricsProcessor {
    async fn process(&self, event: Event) -> Result<ProcessingResult> {
        let start_time = Instant::now();
        debug!("Processing metrics event: {}", event.id);

        match &event.data {
            crate::types::EventData::Metrics(metrics_data) => {
                debug!("Recording metric: {} = {}", metrics_data.metric_name, metrics_data.value);

                // In production, this would send to Prometheus or similar
                Ok(ProcessingResult {
                    event_id: event.id,
                    success: true,
                    output: None,
                    error: None,
                    processing_time_ms: start_time.elapsed().as_millis() as u64,
                    retry_after: None,
                })
            }
            _ => {
                Err(anyhow::anyhow!("Invalid event data for MetricsProcessor"))
            }
        }
    }

    fn get_name(&self) -> &str {
        &self.name
    }

    fn get_supported_events(&self) -> Vec<EventType> {
        vec![EventType::MetricsUpdate]
    }
}

// System Event Processing Handler
pub struct SystemEventProcessor {
    name: String,
}

impl SystemEventProcessor {
    pub fn new() -> Self {
        Self {
            name: "SystemEventProcessor".to_string(),
        }
    }
}

#[async_trait]
impl ProcessorTrait for SystemEventProcessor {
    async fn process(&self, event: Event) -> Result<ProcessingResult> {
        let start_time = Instant::now();
        debug!("Processing system event: {}", event.id);

        match &event.data {
            crate::types::EventData::System(system_data) => {
                match system_data.level {
                    crate::types::LogLevel::Critical | crate::types::LogLevel::Error => {
                        tracing::error!("[{}] {}: {}", system_data.component, system_data.level as u8, system_data.message);
                    }
                    crate::types::LogLevel::Warn => {
                        tracing::warn!("[{}] {}", system_data.component, system_data.message);
                    }
                    crate::types::LogLevel::Info => {
                        tracing::info!("[{}] {}", system_data.component, system_data.message);
                    }
                    crate::types::LogLevel::Debug => {
                        tracing::debug!("[{}] {}", system_data.component, system_data.message);
                    }
                }

                Ok(ProcessingResult {
                    event_id: event.id,
                    success: true,
                    output: None,
                    error: None,
                    processing_time_ms: start_time.elapsed().as_millis() as u64,
                    retry_after: None,
                })
            }
            _ => {
                Err(anyhow::anyhow!("Invalid event data for SystemEventProcessor"))
            }
        }
    }

    fn get_name(&self) -> &str {
        &self.name
    }

    fn get_supported_events(&self) -> Vec<EventType> {
        vec![EventType::SystemEvent]
    }
}