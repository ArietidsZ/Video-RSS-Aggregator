use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoMetadata {
    pub video_id: String,
    pub title: String,
    pub duration: Duration,
    pub format: String,
    pub resolution: Resolution,
    pub fps: f32,
    pub bitrate: u32,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resolution {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamSession {
    pub session_id: Uuid,
    pub video_id: String,
    pub peer_id: String,
    pub start_time: SystemTime,
    pub bitrate: u32,
    pub target_fps: u32,
    pub quality: StreamQuality,
    pub stats: StreamStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamQuality {
    Low,     // 480p
    Medium,  // 720p
    High,    // 1080p
    Ultra,   // 4K
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StreamStats {
    pub packets_sent: u64,
    pub packets_lost: u64,
    pub bytes_sent: u64,
    pub current_bitrate: u32,
    pub target_bitrate: u32,
    pub fps: f32,
    pub latency_ms: u32,
    pub jitter_ms: u32,
    pub buffer_health: f32, // 0.0-1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SignalingMessage {
    Offer {
        sdp: String,
        session_id: String,
        video_id: String,
    },
    Answer {
        sdp: String,
        session_id: String,
    },
    IceCandidate {
        candidate: String,
        sdp_mid: Option<String>,
        sdp_mline_index: Option<u16>,
        session_id: String,
    },
    StreamRequest {
        video_id: String,
        quality: StreamQuality,
        start_time: Option<f64>, // seconds
    },
    StreamResponse {
        session_id: String,
        status: StreamStatus,
        message: Option<String>,
    },
    StatsUpdate {
        session_id: String,
        stats: StreamStats,
    },
    Error {
        code: u16,
        message: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamStatus {
    Pending,
    Ready,
    Streaming,
    Paused,
    Ended,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerStats {
    pub active_sessions: usize,
    pub total_sessions: u64,
    pub total_bytes_sent: u64,
    pub average_bitrate: u32,
    pub peak_concurrent_streams: usize,
    pub uptime_seconds: u64,
    pub cpu_usage: f32,
    pub memory_usage: f32,
    pub network_io: NetworkStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStats {
    pub bytes_in: u64,
    pub bytes_out: u64,
    pub packets_in: u64,
    pub packets_out: u64,
    pub errors: u64,
}

#[derive(Debug, Clone)]
pub struct BitrateController {
    pub current_bitrate: u32,
    pub target_bitrate: u32,
    pub min_bitrate: u32,
    pub max_bitrate: u32,
    pub last_adjustment: SystemTime,
    pub packet_loss_rate: f32,
    pub rtt_ms: u32,
}

impl BitrateController {
    pub fn new(initial_bitrate: u32, max_bitrate: u32) -> Self {
        Self {
            current_bitrate: initial_bitrate,
            target_bitrate: initial_bitrate,
            min_bitrate: initial_bitrate / 4,
            max_bitrate,
            last_adjustment: SystemTime::now(),
            packet_loss_rate: 0.0,
            rtt_ms: 0,
        }
    }

    pub fn adjust_bitrate(&mut self, stats: &StreamStats) {
        let now = SystemTime::now();
        let since_last = now.duration_since(self.last_adjustment).unwrap_or_default();

        // Only adjust every 2 seconds
        if since_last < Duration::from_secs(2) {
            return;
        }

        let loss_rate = if stats.packets_sent > 0 {
            stats.packets_lost as f32 / stats.packets_sent as f32
        } else {
            0.0
        };

        // Aggressive bitrate adjustment based on packet loss and latency
        if loss_rate > 0.05 || stats.latency_ms > 200 {
            // High loss or latency - reduce bitrate
            self.target_bitrate = (self.target_bitrate as f32 * 0.8) as u32;
        } else if loss_rate < 0.01 && stats.latency_ms < 100 && stats.buffer_health > 0.8 {
            // Good conditions - increase bitrate
            self.target_bitrate = (self.target_bitrate as f32 * 1.1) as u32;
        }

        // Clamp to bounds
        self.target_bitrate = self.target_bitrate.clamp(self.min_bitrate, self.max_bitrate);
        self.current_bitrate = self.target_bitrate;
        self.last_adjustment = now;
    }
}