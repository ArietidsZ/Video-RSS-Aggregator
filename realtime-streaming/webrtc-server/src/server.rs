use crate::signaling::SignalingHandler;
use crate::streaming::StreamManager;
use crate::types::{ServerStats, StreamSession};
use anyhow::Result;
use axum::extract::ws::{Message, WebSocket};
use dashmap::DashMap;
use futures_util::{SinkExt, StreamExt};
use rdkafka::{
    config::ClientConfig,
    producer::{FutureProducer, FutureRecord},
};
use redis::Client as RedisClient;
use std::{
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{sync::RwLock, time::interval};
use tracing::{error, info, warn};
use uuid::Uuid;

pub struct WebRTCServer {
    // Core components
    signaling: Arc<SignalingHandler>,
    stream_manager: Arc<StreamManager>,

    // External services
    kafka_producer: FutureProducer,
    redis_client: RedisClient,

    // Session management
    active_sessions: Arc<DashMap<Uuid, StreamSession>>,

    // Statistics
    stats: Arc<ServerStats>,
    total_sessions: AtomicU64,
    total_bytes_sent: AtomicU64,
    peak_concurrent: AtomicUsize,
    start_time: SystemTime,

    // Configuration
    max_connections: usize,
    max_bitrate_kbps: u32,
    target_fps: u32,
}

impl WebRTCServer {
    pub async fn new(
        kafka_brokers: &str,
        redis_url: &str,
        max_connections: usize,
        max_bitrate_kbps: u32,
        target_fps: u32,
    ) -> Result<Self> {
        info!("Initializing WebRTC server...");

        // Initialize Kafka producer
        let kafka_producer: FutureProducer = ClientConfig::new()
            .set("bootstrap.servers", kafka_brokers)
            .set("message.timeout.ms", "5000")
            .set("queue.buffering.max.messages", "100000")
            .set("queue.buffering.max.ms", "100")
            .set("batch.num.messages", "100")
            .set("compression.type", "lz4")
            .create()?;

        // Initialize Redis client
        let redis_client = RedisClient::open(redis_url)?;

        // Test connections
        let _: redis::RedisResult<()> = redis_client.get_connection()?.ping();
        info!("Redis connection established");

        // Initialize components
        let signaling = Arc::new(SignalingHandler::new());
        let stream_manager = Arc::new(StreamManager::new(max_bitrate_kbps, target_fps).await?);

        let server = Self {
            signaling,
            stream_manager,
            kafka_producer,
            redis_client,
            active_sessions: Arc::new(DashMap::new()),
            stats: Arc::new(ServerStats {
                active_sessions: 0,
                total_sessions: 0,
                total_bytes_sent: 0,
                average_bitrate: 0,
                peak_concurrent_streams: 0,
                uptime_seconds: 0,
                cpu_usage: 0.0,
                memory_usage: 0.0,
                network_io: crate::types::NetworkStats {
                    bytes_in: 0,
                    bytes_out: 0,
                    packets_in: 0,
                    packets_out: 0,
                    errors: 0,
                },
            }),
            total_sessions: AtomicU64::new(0),
            total_bytes_sent: AtomicU64::new(0),
            peak_concurrent: AtomicUsize::new(0),
            start_time: SystemTime::now(),
            max_connections,
            max_bitrate_kbps,
            target_fps,
        };

        // Start background tasks
        server.start_stats_collector().await;
        server.start_session_cleanup().await;

        info!("WebRTC server initialized successfully");
        Ok(server)
    }

    pub async fn handle_websocket(&self, mut socket: WebSocket) -> Result<()> {
        let session_id = Uuid::new_v4();
        info!("New WebSocket connection: {}", session_id);

        // Check connection limits
        if self.active_sessions.len() >= self.max_connections {
            warn!("Connection limit reached, rejecting connection");
            let _ = socket.send(Message::Text(
                serde_json::to_string(&crate::types::SignalingMessage::Error {
                    code: 503,
                    message: "Server at capacity".to_string(),
                })?
            )).await;
            return Ok(());
        }

        // Handle signaling messages
        while let Some(msg) = socket.recv().await {
            match msg? {
                Message::Text(text) => {
                    if let Err(e) = self.handle_signaling_message(&mut socket, &text, session_id).await {
                        error!("Signaling error: {}", e);
                        break;
                    }
                }
                Message::Binary(_) => {
                    warn!("Received unexpected binary message");
                }
                Message::Close(_) => {
                    info!("WebSocket connection closed: {}", session_id);
                    break;
                }
                _ => {}
            }
        }

        // Cleanup session
        self.cleanup_session(session_id).await;
        Ok(())
    }

    async fn handle_signaling_message(
        &self,
        socket: &mut WebSocket,
        message: &str,
        session_id: Uuid,
    ) -> Result<()> {
        let msg: crate::types::SignalingMessage = serde_json::from_str(message)?;

        match self.signaling.handle_message(msg, session_id).await {
            Ok(Some(response)) => {
                let response_json = serde_json::to_string(&response)?;
                socket.send(Message::Text(response_json)).await?;
            }
            Ok(None) => {
                // No response needed
            }
            Err(e) => {
                error!("Signaling handler error: {}", e);
                let error_msg = crate::types::SignalingMessage::Error {
                    code: 500,
                    message: e.to_string(),
                };
                let error_json = serde_json::to_string(&error_msg)?;
                socket.send(Message::Text(error_json)).await?;
            }
        }

        Ok(())
    }

    async fn cleanup_session(&self, session_id: Uuid) {
        self.active_sessions.remove(&session_id);
        self.stream_manager.cleanup_session(session_id).await;

        // Send session end event to Kafka
        let event = serde_json::json!({
            "type": "session_ended",
            "session_id": session_id.to_string(),
            "timestamp": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
        });

        if let Err(e) = self.kafka_producer.send(
            FutureRecord::to("webrtc-events")
                .key(&session_id.to_string())
                .payload(&event.to_string()),
            Duration::from_secs(1),
        ).await {
            error!("Failed to send session end event: {}", e);
        }
    }

    async fn start_stats_collector(&self) {
        let active_sessions = Arc::clone(&self.active_sessions);
        let total_sessions = Arc::clone(&self.total_sessions);
        let total_bytes_sent = Arc::clone(&self.total_bytes_sent);
        let peak_concurrent = Arc::clone(&self.peak_concurrent);
        let start_time = self.start_time;

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(10));

            loop {
                interval.tick().await;

                let current_sessions = active_sessions.len();
                peak_concurrent.fetch_max(current_sessions, Ordering::Relaxed);

                // Log statistics
                info!(
                    "Stats - Active: {}, Total: {}, Peak: {}, Bytes: {}",
                    current_sessions,
                    total_sessions.load(Ordering::Relaxed),
                    peak_concurrent.load(Ordering::Relaxed),
                    total_bytes_sent.load(Ordering::Relaxed)
                );
            }
        });
    }

    async fn start_session_cleanup(&self) {
        let active_sessions = Arc::clone(&self.active_sessions);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60));

            loop {
                interval.tick().await;

                let now = SystemTime::now();
                let mut to_remove = Vec::new();

                // Find stale sessions (older than 5 minutes)
                for entry in active_sessions.iter() {
                    let session = entry.value();
                    if now.duration_since(session.start_time).unwrap_or_default() > Duration::from_secs(300) {
                        to_remove.push(*entry.key());
                    }
                }

                // Remove stale sessions
                for session_id in to_remove {
                    active_sessions.remove(&session_id);
                    info!("Cleaned up stale session: {}", session_id);
                }
            }
        });
    }

    pub async fn get_stats(&self) -> ServerStats {
        let uptime = self.start_time.elapsed().unwrap_or_default().as_secs();

        ServerStats {
            active_sessions: self.active_sessions.len(),
            total_sessions: self.total_sessions.load(Ordering::Relaxed),
            total_bytes_sent: self.total_bytes_sent.load(Ordering::Relaxed),
            average_bitrate: if self.active_sessions.len() > 0 {
                self.max_bitrate_kbps / 2  // Rough estimate
            } else {
                0
            },
            peak_concurrent_streams: self.peak_concurrent.load(Ordering::Relaxed),
            uptime_seconds: uptime,
            cpu_usage: 0.0,  // Would integrate with system monitoring
            memory_usage: 0.0,
            network_io: crate::types::NetworkStats {
                bytes_in: 0,
                bytes_out: self.total_bytes_sent.load(Ordering::Relaxed),
                packets_in: 0,
                packets_out: 0,
                errors: 0,
            },
        }
    }
}