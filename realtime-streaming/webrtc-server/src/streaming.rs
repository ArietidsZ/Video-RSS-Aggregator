use crate::types::{BitrateController, StreamSession, StreamStats, StreamQuality};
use anyhow::Result;
use dashmap::DashMap;
use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, SystemTime},
};
use tokio::{sync::RwLock, time::interval};
use tracing::{info, warn};
use uuid::Uuid;

pub struct StreamManager {
    sessions: Arc<DashMap<Uuid, Arc<RwLock<StreamSession>>>>,
    bitrate_controllers: Arc<DashMap<Uuid, Arc<RwLock<BitrateController>>>>,
    max_bitrate_kbps: u32,
    target_fps: u32,
    total_bytes_sent: AtomicU64,
}

impl StreamManager {
    pub async fn new(max_bitrate_kbps: u32, target_fps: u32) -> Result<Self> {
        let manager = Self {
            sessions: Arc::new(DashMap::new()),
            bitrate_controllers: Arc::new(DashMap::new()),
            max_bitrate_kbps,
            target_fps,
            total_bytes_sent: AtomicU64::new(0),
        };

        // Start background tasks
        manager.start_stats_updater().await;
        manager.start_adaptive_bitrate_controller().await;

        Ok(manager)
    }

    pub async fn create_session(
        &self,
        session_id: Uuid,
        video_id: String,
        quality: StreamQuality,
    ) -> Result<()> {
        let bitrate = self.get_target_bitrate(&quality);

        let session = StreamSession {
            session_id,
            video_id: video_id.clone(),
            peer_id: format!("peer-{}", session_id),
            start_time: SystemTime::now(),
            bitrate,
            target_fps: self.target_fps,
            quality,
            stats: StreamStats::default(),
        };

        let bitrate_controller = BitrateController::new(bitrate, self.max_bitrate_kbps);

        self.sessions.insert(session_id, Arc::new(RwLock::new(session)));
        self.bitrate_controllers.insert(session_id, Arc::new(RwLock::new(bitrate_controller)));

        info!("Created streaming session {} for video {}", session_id, video_id);
        Ok(())
    }

    pub async fn update_stats(&self, session_id: Uuid, stats: StreamStats) -> Result<()> {
        if let Some(session_ref) = self.sessions.get(&session_id) {
            let mut session = session_ref.write().await;
            session.stats = stats.clone();

            // Update bitrate controller
            if let Some(controller_ref) = self.bitrate_controllers.get(&session_id) {
                let mut controller = controller_ref.write().await;
                controller.adjust_bitrate(&stats);
                session.bitrate = controller.current_bitrate;
            }

            // Update global stats
            self.total_bytes_sent.fetch_add(stats.bytes_sent, Ordering::Relaxed);
        }

        Ok(())
    }

    pub async fn get_session_stats(&self, session_id: Uuid) -> Option<StreamStats> {
        self.sessions
            .get(&session_id)?
            .read()
            .await
            .stats
            .clone()
            .into()
    }

    pub async fn cleanup_session(&self, session_id: Uuid) {
        self.sessions.remove(&session_id);
        self.bitrate_controllers.remove(&session_id);
        info!("Cleaned up streaming session: {}", session_id);
    }

    fn get_target_bitrate(&self, quality: &StreamQuality) -> u32 {
        match quality {
            StreamQuality::Low => 1000,      // 1 Mbps
            StreamQuality::Medium => 2500,   // 2.5 Mbps
            StreamQuality::High => 5000,     // 5 Mbps
            StreamQuality::Ultra => 15000,   // 15 Mbps
        }
    }

    async fn start_stats_updater(&self) {
        let sessions = Arc::clone(&self.sessions);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(5));

            loop {
                interval.tick().await;

                for entry in sessions.iter() {
                    let session_id = *entry.key();
                    let session_ref = entry.value();

                    // Simulate stats collection
                    // In production, this would gather real network stats
                    let mut session = session_ref.write().await;
                    let elapsed = session.start_time.elapsed().unwrap_or_default();

                    // Update simulated stats
                    session.stats.packets_sent += 100;
                    session.stats.packets_lost += if rand::random::<f32>() < 0.02 { 1 } else { 0 };
                    session.stats.bytes_sent += (session.bitrate / 8) as u64 * 5; // 5 seconds worth
                    session.stats.current_bitrate = session.bitrate;
                    session.stats.latency_ms = 50 + (rand::random::<u32>() % 100);
                    session.stats.jitter_ms = 5 + (rand::random::<u32>() % 20);
                    session.stats.buffer_health = 0.5 + (rand::random::<f32>() * 0.5);
                    session.stats.fps = session.target_fps as f32 * (0.9 + rand::random::<f32>() * 0.2);

                    if elapsed.as_secs() % 30 == 0 {
                        info!(
                            "Session {} stats - Bitrate: {}kbps, Latency: {}ms, Loss: {:.2}%",
                            session_id,
                            session.stats.current_bitrate,
                            session.stats.latency_ms,
                            if session.stats.packets_sent > 0 {
                                (session.stats.packets_lost as f32 / session.stats.packets_sent as f32) * 100.0
                            } else {
                                0.0
                            }
                        );
                    }
                }
            }
        });
    }

    async fn start_adaptive_bitrate_controller(&self) {
        let sessions = Arc::clone(&self.sessions);
        let controllers = Arc::clone(&self.bitrate_controllers);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(3));

            loop {
                interval.tick().await;

                for entry in sessions.iter() {
                    let session_id = *entry.key();
                    let session_ref = entry.value();

                    if let Some(controller_ref) = controllers.get(&session_id) {
                        let session = session_ref.read().await;
                        let mut controller = controller_ref.write().await;

                        // Adaptive bitrate control
                        let old_bitrate = controller.current_bitrate;
                        controller.adjust_bitrate(&session.stats);

                        if controller.current_bitrate != old_bitrate {
                            info!(
                                "Adjusted bitrate for session {}: {} -> {} kbps (loss: {:.2}%, latency: {}ms)",
                                session_id,
                                old_bitrate,
                                controller.current_bitrate,
                                if session.stats.packets_sent > 0 {
                                    (session.stats.packets_lost as f32 / session.stats.packets_sent as f32) * 100.0
                                } else {
                                    0.0
                                },
                                session.stats.latency_ms
                            );
                        }
                    }
                }
            }
        });
    }

    pub async fn get_global_stats(&self) -> GlobalStreamStats {
        let mut total_bitrate = 0u32;
        let mut total_sessions = 0usize;
        let mut avg_latency = 0u32;
        let mut total_packet_loss = 0f32;

        for entry in self.sessions.iter() {
            let session = entry.value().read().await;
            total_bitrate += session.stats.current_bitrate;
            total_sessions += 1;
            avg_latency += session.stats.latency_ms;

            if session.stats.packets_sent > 0 {
                total_packet_loss += session.stats.packets_lost as f32 / session.stats.packets_sent as f32;
            }
        }

        GlobalStreamStats {
            active_sessions: total_sessions,
            total_bitrate_kbps: total_bitrate,
            average_latency_ms: if total_sessions > 0 { avg_latency / total_sessions as u32 } else { 0 },
            average_packet_loss: if total_sessions > 0 { total_packet_loss / total_sessions as f32 } else { 0.0 },
            total_bytes_sent: self.total_bytes_sent.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GlobalStreamStats {
    pub active_sessions: usize,
    pub total_bitrate_kbps: u32,
    pub average_latency_ms: u32,
    pub average_packet_loss: f32,
    pub total_bytes_sent: u64,
}

// Simple random number generator for demo purposes
mod rand {
    use std::sync::atomic::{AtomicU64, Ordering};

    static SEED: AtomicU64 = AtomicU64::new(12345);

    pub fn random<T>() -> T
    where
        T: From<u64>,
    {
        let current = SEED.load(Ordering::Relaxed);
        let next = current.wrapping_mul(1103515245).wrapping_add(12345);
        SEED.store(next, Ordering::Relaxed);
        T::from(next)
    }
}