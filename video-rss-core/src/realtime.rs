use crate::{error::VideoRssError, Result, types::*};
use axum::{
    response::sse::{Event, KeepAlive, Sse},
    extract::{State, Query},
    response::IntoResponse,
};
use futures::stream::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{broadcast, RwLock};
use tokio_stream::wrappers::BroadcastStream;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeConfig {
    pub max_connections: usize,
    pub heartbeat_interval: Duration,
    pub buffer_size: usize,
    pub compression: bool,
    pub delta_encoding: bool,
    pub binary_format: bool,
}

impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            max_connections: 10_000,
            heartbeat_interval: Duration::from_secs(30),
            buffer_size: 1024,
            compression: true,
            delta_encoding: true,
            binary_format: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RealtimeEvent {
    VideoAdded(VideoUpdate),
    VideoUpdated(VideoUpdate),
    VideoRemoved { id: String },
    TranscriptionCompleted(TranscriptionUpdate),
    FeedUpdated(FeedUpdate),
    SystemStatus(SystemStatus),
    Heartbeat,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoUpdate {
    pub id: String,
    pub title: String,
    pub platform: String,
    pub author: String,
    pub url: String,
    pub view_count: i64,
    pub timestamp: i64,
    pub delta: Option<VideoDelta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoDelta {
    pub view_count_change: i64,
    pub like_count_change: i64,
    pub comment_count_change: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionUpdate {
    pub video_id: String,
    pub status: TranscriptionStatus,
    pub progress: f32,
    pub summary: Option<String>,
    pub processing_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TranscriptionStatus {
    Queued,
    Processing,
    Completed,
    Failed(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedUpdate {
    pub feed_id: String,
    pub platform: String,
    pub new_videos: usize,
    pub total_videos: usize,
    pub last_updated: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatus {
    pub connected_clients: usize,
    pub cpu_usage: f32,
    pub memory_usage: f32,
    pub queue_size: usize,
    pub processing_rate: f32,
}

pub struct RealtimeManager {
    config: RealtimeConfig,
    connections: Arc<RwLock<HashMap<Uuid, Connection>>>,
    event_broadcaster: broadcast::Sender<RealtimeEvent>,
    stats: Arc<RwLock<RealtimeStats>>,
}

struct Connection {
    id: Uuid,
    client_id: String,
    filters: ConnectionFilters,
    created_at: std::time::Instant,
    last_activity: std::time::Instant,
}

#[derive(Debug, Clone, Default)]
pub struct ConnectionFilters {
    platforms: Option<Vec<String>>,
    authors: Option<Vec<String>>,
    event_types: Option<Vec<String>>,
}

#[derive(Default)]
struct RealtimeStats {
    total_connections: u64,
    active_connections: usize,
    events_sent: u64,
    events_dropped: u64,
    bytes_sent: u64,
}

impl RealtimeManager {
    pub fn new(config: RealtimeConfig) -> Self {
        let (tx, _) = broadcast::channel(config.buffer_size);

        Self {
            config,
            connections: Arc::new(RwLock::new(HashMap::new())),
            event_broadcaster: tx,
            stats: Arc::new(RwLock::new(RealtimeStats::default())),
        }
    }

    pub async fn create_sse_stream(
        &self,
        filters: ConnectionFilters,
    ) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
        let conn_id = Uuid::new_v4();
        let client_id = format!("client_{}", conn_id);

        // Register connection
        self.register_connection(conn_id, client_id.clone(), filters).await;

        // Create receiver
        let rx = self.event_broadcaster.subscribe();
        let stream = BroadcastStream::new(rx);

        // Convert to SSE events
        let event_stream = stream
            .filter_map(move |result| {
                async move {
                    match result {
                        Ok(event) => {
                            let sse_event = Self::to_sse_event(event);
                            Some(Ok(sse_event))
                        },
                        Err(e) => {
                            warn!("Broadcast error: {}", e);
                            None
                        }
                    }
                }
            });

        // Configure SSE with keep-alive
        Sse::new(event_stream)
            .keep_alive(KeepAlive::new().interval(self.config.heartbeat_interval))
    }

    fn to_sse_event(event: RealtimeEvent) -> Event {
        match event {
            RealtimeEvent::VideoAdded(update) => {
                Event::default()
                    .event("video.added")
                    .json_data(update)
                    .unwrap()
            },
            RealtimeEvent::VideoUpdated(update) => {
                Event::default()
                    .event("video.updated")
                    .json_data(update)
                    .unwrap()
            },
            RealtimeEvent::VideoRemoved { id } => {
                Event::default()
                    .event("video.removed")
                    .data(id)
            },
            RealtimeEvent::TranscriptionCompleted(update) => {
                Event::default()
                    .event("transcription.completed")
                    .json_data(update)
                    .unwrap()
            },
            RealtimeEvent::FeedUpdated(update) => {
                Event::default()
                    .event("feed.updated")
                    .json_data(update)
                    .unwrap()
            },
            RealtimeEvent::SystemStatus(status) => {
                Event::default()
                    .event("system.status")
                    .json_data(status)
                    .unwrap()
            },
            RealtimeEvent::Heartbeat => {
                Event::default()
                    .event("heartbeat")
                    .data("ping")
            },
        }
    }

    async fn register_connection(
        &self,
        id: Uuid,
        client_id: String,
        filters: ConnectionFilters,
    ) {
        let connection = Connection {
            id,
            client_id,
            filters,
            created_at: std::time::Instant::now(),
            last_activity: std::time::Instant::now(),
        };

        let mut connections = self.connections.write().await;
        connections.insert(id, connection);

        let mut stats = self.stats.write().await;
        stats.total_connections += 1;
        stats.active_connections = connections.len();

        info!("New SSE connection registered: {}", id);
    }

    pub async fn broadcast_event(&self, event: RealtimeEvent) -> Result<()> {
        let result = self.event_broadcaster.send(event);

        let mut stats = self.stats.write().await;
        if result.is_ok() {
            stats.events_sent += 1;
        } else {
            stats.events_dropped += 1;
        }

        Ok(())
    }

    pub async fn broadcast_video_update(&self, video: ExtractedVideo) -> Result<()> {
        let update = VideoUpdate {
            id: video.id.clone(),
            title: video.title.clone(),
            platform: format!("{:?}", video.platform),
            author: video.author.clone(),
            url: video.url.clone(),
            view_count: video.view_count.unwrap_or(0) as i64,
            timestamp: chrono::Utc::now().timestamp(),
            delta: None,
        };

        self.broadcast_event(RealtimeEvent::VideoAdded(update)).await
    }

    pub async fn broadcast_transcription_update(
        &self,
        video_id: String,
        status: TranscriptionStatus,
        progress: f32,
    ) -> Result<()> {
        let update = TranscriptionUpdate {
            video_id,
            status,
            progress,
            summary: None,
            processing_time_ms: 0,
        };

        self.broadcast_event(RealtimeEvent::TranscriptionCompleted(update)).await
    }

    pub async fn broadcast_system_status(&self) -> Result<()> {
        let stats = self.stats.read().await;
        let status = SystemStatus {
            connected_clients: stats.active_connections,
            cpu_usage: Self::get_cpu_usage(),
            memory_usage: Self::get_memory_usage(),
            queue_size: 0,
            processing_rate: 0.0,
        };

        self.broadcast_event(RealtimeEvent::SystemStatus(status)).await
    }

    fn get_cpu_usage() -> f32 {
        // Simplified CPU usage calculation
        // In production, would use sysinfo crate
        0.45
    }

    fn get_memory_usage() -> f32 {
        // Simplified memory usage calculation
        0.62
    }

    pub async fn cleanup_stale_connections(&self) {
        let timeout = Duration::from_secs(300);  // 5 minutes
        let now = std::time::Instant::now();

        let mut connections = self.connections.write().await;
        let before_count = connections.len();

        connections.retain(|_, conn| {
            now.duration_since(conn.last_activity) < timeout
        });

        let removed = before_count - connections.len();
        if removed > 0 {
            info!("Cleaned up {} stale SSE connections", removed);

            let mut stats = self.stats.write().await;
            stats.active_connections = connections.len();
        }
    }

    pub async fn get_stats(&self) -> RealtimeStatistics {
        let stats = self.stats.read().await;
        let connections = self.connections.read().await;

        RealtimeStatistics {
            active_connections: stats.active_connections,
            total_connections: stats.total_connections,
            events_sent: stats.events_sent,
            events_dropped: stats.events_dropped,
            bytes_sent: stats.bytes_sent,
            uptime_seconds: 0,  // Would calculate from start time
            connections_by_platform: Self::count_by_platform(&connections),
        }
    }

    fn count_by_platform(connections: &HashMap<Uuid, Connection>) -> HashMap<String, usize> {
        let mut counts = HashMap::new();

        for conn in connections.values() {
            if let Some(platforms) = &conn.filters.platforms {
                for platform in platforms {
                    *counts.entry(platform.clone()).or_insert(0) += 1;
                }
            }
        }

        counts
    }
}

// WebSub/PubSubHubbub implementation for instant updates
pub struct WebSubHub {
    subscribers: Arc<RwLock<HashMap<String, Subscriber>>>,
    topics: Arc<RwLock<HashMap<String, Topic>>>,
}

#[derive(Clone)]
struct Subscriber {
    callback_url: String,
    secret: Option<String>,
    lease_seconds: u64,
    subscribed_at: std::time::Instant,
}

#[derive(Clone)]
struct Topic {
    url: String,
    last_updated: std::time::Instant,
    content_hash: String,
}

impl WebSubHub {
    pub fn new() -> Self {
        Self {
            subscribers: Arc::new(RwLock::new(HashMap::new())),
            topics: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn subscribe(
        &self,
        callback_url: String,
        topic_url: String,
        lease_seconds: u64,
        secret: Option<String>,
    ) -> Result<()> {
        let subscriber = Subscriber {
            callback_url: callback_url.clone(),
            secret,
            lease_seconds,
            subscribed_at: std::time::Instant::now(),
        };

        let mut subscribers = self.subscribers.write().await;
        let key = format!("{}:{}", topic_url, callback_url);
        subscribers.insert(key, subscriber);

        info!("New WebSub subscription: {} -> {}", topic_url, callback_url);

        Ok(())
    }

    pub async fn publish(&self, topic_url: String, content: String) -> Result<()> {
        // Calculate content hash for deduplication
        let mut hasher = blake3::Hasher::new();
        hasher.update(content.as_bytes());
        let content_hash = hasher.finalize().to_hex().to_string();

        // Check if content has changed
        let mut topics = self.topics.write().await;
        if let Some(topic) = topics.get(&topic_url) {
            if topic.content_hash == content_hash {
                debug!("Content unchanged for topic: {}", topic_url);
                return Ok(());
            }
        }

        // Update topic
        topics.insert(topic_url.clone(), Topic {
            url: topic_url.clone(),
            last_updated: std::time::Instant::now(),
            content_hash,
        });

        // Notify subscribers
        let subscribers = self.subscribers.read().await;
        for (key, subscriber) in subscribers.iter() {
            if key.starts_with(&format!("{}:", topic_url)) {
                self.notify_subscriber(subscriber, &content).await?;
            }
        }

        Ok(())
    }

    async fn notify_subscriber(&self, subscriber: &Subscriber, content: &str) -> Result<()> {
        let client = reqwest::Client::new();

        let mut request = client.post(&subscriber.callback_url)
            .body(content.to_string())
            .header("Content-Type", "application/rss+xml");

        // Add HMAC signature if secret is configured
        if let Some(secret) = &subscriber.secret {
            use hmac::{Hmac, Mac};
            use sha2::Sha256;

            let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes())
                .map_err(|e| VideoRssError::Unknown(format!("HMAC error: {}", e)))?;
            mac.update(content.as_bytes());
            let signature = mac.finalize();
            let signature_hex = format!("{:x}", signature.into_bytes());

            request = request.header("X-Hub-Signature", format!("sha256={}", signature_hex));
        }

        request.send().await
            .map_err(|e| VideoRssError::Http(e))?;

        Ok(())
    }

    pub async fn cleanup_expired_subscriptions(&self) {
        let now = std::time::Instant::now();
        let mut subscribers = self.subscribers.write().await;

        subscribers.retain(|_, sub| {
            now.duration_since(sub.subscribed_at).as_secs() < sub.lease_seconds
        });
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeStatistics {
    pub active_connections: usize,
    pub total_connections: u64,
    pub events_sent: u64,
    pub events_dropped: u64,
    pub bytes_sent: u64,
    pub uptime_seconds: u64,
    pub connections_by_platform: HashMap<String, usize>,
}