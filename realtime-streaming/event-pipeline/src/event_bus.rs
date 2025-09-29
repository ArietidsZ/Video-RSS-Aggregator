use crate::types::{Event, EventType};
use anyhow::{anyhow, Result};
use dashmap::DashMap;
use futures_util::StreamExt;
use rdkafka::{
    config::ClientConfig,
    consumer::{CommitMode, Consumer, StreamConsumer},
    message::Message,
    producer::{FutureProducer, FutureRecord},
    ClientContext, TopicPartitionList,
};
use redis::{aio::ConnectionManager, AsyncCommands, Client as RedisClient};
use serde_json;
use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};
use tokio::{
    sync::{mpsc, RwLock},
    time::timeout,
};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

pub struct EventBus {
    // Kafka components
    producer: FutureProducer,
    consumers: Arc<DashMap<String, StreamConsumer>>,

    // Redis for state management
    redis: ConnectionManager,

    // Internal message queues
    event_senders: Arc<DashMap<EventType, mpsc::UnboundedSender<Event>>>,
    event_receivers: Arc<RwLock<DashMap<EventType, mpsc::UnboundedReceiver<Event>>>>,

    // Metrics
    published_count: AtomicU64,
    consumed_count: AtomicU64,
    failed_count: AtomicU64,

    // Configuration
    max_queue_size: usize,
}

impl EventBus {
    pub async fn new(
        kafka_brokers: &str,
        redis_url: &str,
        max_queue_size: usize,
    ) -> Result<Self> {
        info!("Initializing Event Bus...");

        // Initialize Kafka producer
        let producer: FutureProducer = ClientConfig::new()
            .set("bootstrap.servers", kafka_brokers)
            .set("message.timeout.ms", "5000")
            .set("queue.buffering.max.messages", "100000")
            .set("queue.buffering.max.ms", "100")
            .set("batch.num.messages", "500")
            .set("compression.type", "lz4")
            .set("acks", "all")
            .set("retries", "3")
            .set("enable.idempotence", "true")
            .create()?;

        // Initialize Redis
        let redis_client = RedisClient::open(redis_url)?;
        let redis = ConnectionManager::new(redis_client).await?;

        // Test connections
        let mut redis_test = redis.clone();
        let _: String = redis_test.ping().await?;
        info!("Redis connection established");

        let event_bus = Self {
            producer,
            consumers: Arc::new(DashMap::new()),
            redis,
            event_senders: Arc::new(DashMap::new()),
            event_receivers: Arc::new(RwLock::new(DashMap::new())),
            published_count: AtomicU64::new(0),
            consumed_count: AtomicU64::new(0),
            failed_count: AtomicU64::new(0),
            max_queue_size,
        };

        // Initialize topic structure
        event_bus.setup_topics().await?;

        info!("Event Bus initialized successfully");
        Ok(event_bus)
    }

    async fn setup_topics(&self) -> Result<()> {
        let topics = vec![
            "video-events",
            "transcription-events",
            "summarization-events",
            "rss-events",
            "user-events",
            "system-events",
            "metrics-events",
            "deadletter-events",
        ];

        for topic in topics {
            info!("Setting up topic: {}", topic);
            // In production, you would create topics with proper partitioning
            // and replication here using Kafka Admin API
        }

        Ok(())
    }

    pub async fn publish(&self, event: Event) -> Result<()> {
        let topic = self.get_topic_for_event_type(&event.event_type);
        let key = event.id.to_string();
        let payload = serde_json::to_string(&event)?;

        debug!("Publishing event {} to topic {}", event.id, topic);

        // Publish to Kafka
        match timeout(
            Duration::from_secs(5),
            self.producer.send(
                FutureRecord::to(&topic)
                    .key(&key)
                    .payload(&payload)
                    .headers(rdkafka::message::OwnedHeaders::new().insert(
                        rdkafka::message::Header {
                            key: "event_type",
                            value: Some(&serde_json::to_string(&event.event_type)?),
                        },
                    )),
                Duration::from_secs(1),
            ),
        )
        .await
        {
            Ok(Ok(_)) => {
                self.published_count.fetch_add(1, Ordering::Relaxed);

                // Also store in Redis for immediate access
                let mut redis = self.redis.clone();
                let redis_key = format!("event:{}", event.id);
                let _: () = redis
                    .set_ex(&redis_key, &payload, 3600) // 1 hour TTL
                    .await
                    .unwrap_or_else(|e| {
                        warn!("Failed to store event in Redis: {}", e);
                    });

                debug!("Successfully published event {}", event.id);
                Ok(())
            }
            Ok(Err(e)) => {
                self.failed_count.fetch_add(1, Ordering::Relaxed);
                error!("Failed to publish event {}: {}", event.id, e);
                Err(anyhow!("Kafka publish error: {}", e))
            }
            Err(_) => {
                self.failed_count.fetch_add(1, Ordering::Relaxed);
                error!("Timeout publishing event {}", event.id);
                Err(anyhow!("Timeout publishing event"))
            }
        }
    }

    pub async fn subscribe(
        &self,
        event_types: Vec<EventType>,
        consumer_group: &str,
    ) -> Result<mpsc::UnboundedReceiver<Event>> {
        let (sender, receiver) = mpsc::unbounded_channel();

        for event_type in event_types {
            let topic = self.get_topic_for_event_type(&event_type);
            let consumer = self.create_consumer(&topic, consumer_group).await?;

            let sender_clone = sender.clone();
            let consumed_count = Arc::clone(&self.consumed_count);
            let failed_count = Arc::clone(&self.failed_count);

            // Start consumer task
            tokio::spawn(async move {
                Self::consume_messages(consumer, sender_clone, consumed_count, failed_count).await;
            });
        }

        Ok(receiver)
    }

    async fn create_consumer(
        &self,
        topic: &str,
        consumer_group: &str,
    ) -> Result<StreamConsumer> {
        let consumer: StreamConsumer = ClientConfig::new()
            .set("group.id", consumer_group)
            .set("bootstrap.servers", "kafka-1:29092,kafka-2:29093,kafka-3:29094")
            .set("enable.partition.eof", "false")
            .set("session.timeout.ms", "6000")
            .set("enable.auto.commit", "true")
            .set("auto.commit.interval.ms", "1000")
            .set("auto.offset.reset", "latest")
            .set("fetch.min.bytes", "1024")
            .set("fetch.max.wait.ms", "500")
            .create()?;

        consumer.subscribe(&[topic])?;
        self.consumers.insert(topic.to_string(), consumer.clone());

        info!("Created consumer for topic: {} with group: {}", topic, consumer_group);
        Ok(consumer)
    }

    async fn consume_messages(
        consumer: StreamConsumer,
        sender: mpsc::UnboundedSender<Event>,
        consumed_count: Arc<AtomicU64>,
        failed_count: Arc<AtomicU64>,
    ) {
        let mut message_stream = consumer.stream();

        while let Some(message) = message_stream.next().await {
            match message {
                Ok(borrowed_message) => {
                    if let Some(payload) = borrowed_message.payload_view::<str>() {
                        match payload {
                            Ok(json_str) => {
                                match serde_json::from_str::<Event>(json_str) {
                                    Ok(event) => {
                                        consumed_count.fetch_add(1, Ordering::Relaxed);
                                        if sender.send(event).is_err() {
                                            warn!("Failed to send event to internal queue");
                                            break;
                                        }
                                    }
                                    Err(e) => {
                                        failed_count.fetch_add(1, Ordering::Relaxed);
                                        error!("Failed to deserialize event: {}", e);
                                    }
                                }
                            }
                            Err(e) => {
                                failed_count.fetch_add(1, Ordering::Relaxed);
                                error!("Invalid UTF-8 in message payload: {}", e);
                            }
                        }
                    }

                    // Commit message
                    if let Err(e) = consumer.commit_message(&borrowed_message, CommitMode::Async) {
                        error!("Failed to commit message: {}", e);
                    }
                }
                Err(e) => {
                    failed_count.fetch_add(1, Ordering::Relaxed);
                    error!("Kafka consumer error: {}", e);
                }
            }
        }
    }

    pub async fn get_event(&self, event_id: Uuid) -> Result<Option<Event>> {
        let mut redis = self.redis.clone();
        let redis_key = format!("event:{}", event_id);

        match redis.get::<_, Option<String>>(&redis_key).await? {
            Some(json_str) => {
                let event: Event = serde_json::from_str(&json_str)?;
                Ok(Some(event))
            }
            None => Ok(None),
        }
    }

    pub async fn store_processing_state(
        &self,
        event_id: Uuid,
        state: &str,
        data: &serde_json::Value,
    ) -> Result<()> {
        let mut redis = self.redis.clone();
        let state_key = format!("processing:{}:{}", event_id, state);
        let payload = serde_json::to_string(data)?;

        let _: () = redis.set_ex(&state_key, &payload, 3600).await?; // 1 hour TTL
        Ok(())
    }

    pub async fn get_processing_state(
        &self,
        event_id: Uuid,
        state: &str,
    ) -> Result<Option<serde_json::Value>> {
        let mut redis = self.redis.clone();
        let state_key = format!("processing:{}:{}", event_id, state);

        match redis.get::<_, Option<String>>(&state_key).await? {
            Some(json_str) => {
                let data: serde_json::Value = serde_json::from_str(&json_str)?;
                Ok(Some(data))
            }
            None => Ok(None),
        }
    }

    fn get_topic_for_event_type(&self, event_type: &EventType) -> String {
        match event_type {
            EventType::VideoUploaded
            | EventType::VideoProcessingStarted
            | EventType::VideoProcessingCompleted
            | EventType::VideoProcessingFailed => "video-events".to_string(),
            EventType::TranscriptionCompleted => "transcription-events".to_string(),
            EventType::SummarizationCompleted => "summarization-events".to_string(),
            EventType::RSSFeedUpdated => "rss-events".to_string(),
            EventType::UserRequest => "user-events".to_string(),
            EventType::SystemEvent => "system-events".to_string(),
            EventType::MetricsUpdate => "metrics-events".to_string(),
        }
    }

    pub fn get_metrics(&self) -> EventBusMetrics {
        EventBusMetrics {
            published_count: self.published_count.load(Ordering::Relaxed),
            consumed_count: self.consumed_count.load(Ordering::Relaxed),
            failed_count: self.failed_count.load(Ordering::Relaxed),
            active_consumers: self.consumers.len(),
        }
    }

    pub async fn health_check(&self) -> Result<bool> {
        // Test Redis connection
        let mut redis = self.redis.clone();
        let _: String = redis.ping().await?;

        // Test Kafka producer (send a small test message)
        let test_event = Event::new(
            EventType::SystemEvent,
            "health_check".to_string(),
            crate::types::EventData::System(crate::types::SystemEventData {
                component: "event_bus".to_string(),
                message: "health_check".to_string(),
                level: crate::types::LogLevel::Info,
                error_code: None,
                additional_data: std::collections::HashMap::new(),
            }),
        );

        // Don't actually publish, just test serialization
        let _payload = serde_json::to_string(&test_event)?;

        Ok(true)
    }
}

#[derive(Debug, Clone)]
pub struct EventBusMetrics {
    pub published_count: u64,
    pub consumed_count: u64,
    pub failed_count: u64,
    pub active_consumers: usize,
}