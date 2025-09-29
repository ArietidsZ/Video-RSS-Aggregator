use crate::types::{SignalingMessage, StreamQuality, StreamSession, StreamStats, StreamStatus};
use anyhow::{anyhow, Result};
use dashmap::DashMap;
use std::{
    sync::Arc,
    time::SystemTime,
};
use tokio::sync::RwLock;
use tracing::{info, warn};
use uuid::Uuid;
use webrtc::{
    api::{
        interceptor_registry::register_default_interceptors,
        media_engine::{MediaEngine, MIME_TYPE_H264, MIME_TYPE_OPUS},
        APIBuilder,
    },
    ice_transport::ice_server::RTCIceServer,
    interceptor::registry::Registry,
    peer_connection::{
        configuration::RTCConfiguration, peer_connection_state::RTCPeerConnectionState,
        RTCPeerConnection,
    },
    rtp_transceiver::rtp_codec::RTCRtpCodecCapability,
    track::track_local::{track_local_static_rtp::TrackLocalStaticRTP, TrackLocal},
};

pub struct SignalingHandler {
    peer_connections: Arc<DashMap<Uuid, Arc<RTCPeerConnection>>>,
    pending_offers: Arc<DashMap<String, Uuid>>, // video_id -> session_id
    api: webrtc::api::API,
}

impl SignalingHandler {
    pub fn new() -> Self {
        // Create media engine
        let mut media_engine = MediaEngine::default();

        // Add codecs
        media_engine.register_default_codecs().unwrap();

        // Create interceptor registry
        let mut registry = Registry::new();
        registry = register_default_interceptors(registry, &mut media_engine).unwrap();

        // Create WebRTC API
        let api = APIBuilder::new()
            .with_media_engine(media_engine)
            .with_interceptor_registry(registry)
            .build();

        Self {
            peer_connections: Arc::new(DashMap::new()),
            pending_offers: Arc::new(DashMap::new()),
            api,
        }
    }

    pub async fn handle_message(
        &self,
        message: SignalingMessage,
        session_id: Uuid,
    ) -> Result<Option<SignalingMessage>> {
        match message {
            SignalingMessage::StreamRequest { video_id, quality, start_time } => {
                self.handle_stream_request(session_id, video_id, quality, start_time).await
            }
            SignalingMessage::Offer { sdp, session_id: _, video_id } => {
                self.handle_offer(session_id, sdp, video_id).await
            }
            SignalingMessage::Answer { sdp, session_id: _ } => {
                self.handle_answer(session_id, sdp).await
            }
            SignalingMessage::IceCandidate { candidate, sdp_mid, sdp_mline_index, session_id: _ } => {
                self.handle_ice_candidate(session_id, candidate, sdp_mid, sdp_mline_index).await
            }
            _ => {
                warn!("Unhandled signaling message type");
                Ok(None)
            }
        }
    }

    async fn handle_stream_request(
        &self,
        session_id: Uuid,
        video_id: String,
        quality: StreamQuality,
        _start_time: Option<f64>,
    ) -> Result<Option<SignalingMessage>> {
        info!("Stream request for video: {} with quality: {:?}", video_id, quality);

        // Create peer connection
        let config = RTCConfiguration {
            ice_servers: vec![
                RTCIceServer {
                    urls: vec!["stun:stun.l.google.com:19302".to_owned()],
                    ..Default::default()
                },
                RTCIceServer {
                    urls: vec!["turn:your-turn-server.com:3478".to_owned()],
                    username: "username".to_owned(),
                    credential: "password".to_owned(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        let peer_connection = Arc::new(self.api.new_peer_connection(config).await?);

        // Create video track
        let video_track = Arc::new(TrackLocalStaticRTP::new(
            RTCRtpCodecCapability {
                mime_type: MIME_TYPE_H264.to_owned(),
                ..Default::default()
            },
            format!("video-{}", video_id),
            format!("stream-{}", session_id),
        ));

        // Create audio track
        let audio_track = Arc::new(TrackLocalStaticRTP::new(
            RTCRtpCodecCapability {
                mime_type: MIME_TYPE_OPUS.to_owned(),
                ..Default::default()
            },
            format!("audio-{}", video_id),
            format!("stream-{}", session_id),
        ));

        // Add tracks to peer connection
        let _video_sender = peer_connection.add_track(Arc::clone(&video_track) as Arc<dyn TrackLocal + Send + Sync>).await?;
        let _audio_sender = peer_connection.add_track(Arc::clone(&audio_track) as Arc<dyn TrackLocal + Send + Sync>).await?;

        // Set up connection state handler
        let pc_clone = Arc::clone(&peer_connection);
        let session_id_clone = session_id;
        peer_connection.on_peer_connection_state_change(Box::new(move |state: RTCPeerConnectionState| {
            info!("Peer connection state changed for {}: {:?}", session_id_clone, state);
            Box::pin(async move {})
        })).await;

        // Store peer connection
        self.peer_connections.insert(session_id, peer_connection);
        self.pending_offers.insert(video_id.clone(), session_id);

        // Start streaming in background
        tokio::spawn(async move {
            if let Err(e) = Self::start_streaming(video_track, audio_track, quality).await {
                tracing::error!("Streaming error: {}", e);
            }
        });

        Ok(Some(SignalingMessage::StreamResponse {
            session_id: session_id.to_string(),
            status: StreamStatus::Ready,
            message: Some("Stream prepared, send offer".to_string()),
        }))
    }

    async fn handle_offer(
        &self,
        session_id: Uuid,
        sdp: String,
        video_id: String,
    ) -> Result<Option<SignalingMessage>> {
        let peer_connection = self.peer_connections
            .get(&session_id)
            .ok_or_else(|| anyhow!("Peer connection not found"))?
            .clone();

        // Set remote description
        let offer = webrtc::peer_connection::sdp::session_description::RTCSessionDescription {
            sdp_type: webrtc::peer_connection::sdp::sdp_type::RTCSdpType::Offer,
            sdp,
        };

        peer_connection.set_remote_description(offer).await?;

        // Create answer
        let answer = peer_connection.create_answer(None).await?;
        peer_connection.set_local_description(answer.clone()).await?;

        info!("Created answer for session: {}", session_id);

        Ok(Some(SignalingMessage::Answer {
            sdp: answer.sdp,
            session_id: session_id.to_string(),
        }))
    }

    async fn handle_answer(
        &self,
        session_id: Uuid,
        sdp: String,
    ) -> Result<Option<SignalingMessage>> {
        let peer_connection = self.peer_connections
            .get(&session_id)
            .ok_or_else(|| anyhow!("Peer connection not found"))?
            .clone();

        let answer = webrtc::peer_connection::sdp::session_description::RTCSessionDescription {
            sdp_type: webrtc::peer_connection::sdp::sdp_type::RTCSdpType::Answer,
            sdp,
        };

        peer_connection.set_remote_description(answer).await?;

        info!("Set remote description for session: {}", session_id);

        Ok(Some(SignalingMessage::StreamResponse {
            session_id: session_id.to_string(),
            status: StreamStatus::Streaming,
            message: Some("Connection established".to_string()),
        }))
    }

    async fn handle_ice_candidate(
        &self,
        session_id: Uuid,
        candidate: String,
        sdp_mid: Option<String>,
        sdp_mline_index: Option<u16>,
    ) -> Result<Option<SignalingMessage>> {
        let peer_connection = self.peer_connections
            .get(&session_id)
            .ok_or_else(|| anyhow!("Peer connection not found"))?
            .clone();

        let ice_candidate = webrtc::ice_transport::ice_candidate::RTCIceCandidateInit {
            candidate,
            sdp_mid,
            sdp_mline_index,
            username_fragment: None,
        };

        peer_connection.add_ice_candidate(ice_candidate).await?;

        info!("Added ICE candidate for session: {}", session_id);

        Ok(None)
    }

    async fn start_streaming(
        video_track: Arc<TrackLocalStaticRTP>,
        audio_track: Arc<TrackLocalStaticRTP>,
        quality: StreamQuality,
    ) -> Result<()> {
        info!("Starting streaming with quality: {:?}", quality);

        // This would integrate with the actual video pipeline
        // For now, we'll simulate by generating test patterns

        let (width, height, bitrate) = match quality {
            StreamQuality::Low => (854, 480, 1000),      // 1 Mbps
            StreamQuality::Medium => (1280, 720, 2500),  // 2.5 Mbps
            StreamQuality::High => (1920, 1080, 5000),   // 5 Mbps
            StreamQuality::Ultra => (3840, 2160, 15000), // 15 Mbps
        };

        info!("Streaming {}x{} at {} kbps", width, height, bitrate);

        // Simulate streaming loop
        let mut frame_count = 0u64;
        let frame_duration = std::time::Duration::from_millis(33); // ~30 FPS

        loop {
            tokio::time::sleep(frame_duration).await;

            // Generate dummy H.264 frame data
            let video_payload = Self::generate_h264_frame(frame_count, width, height);

            // Send video frame
            if let Err(e) = video_track.write_rtp(&video_payload).await {
                tracing::warn!("Failed to write video RTP: {}", e);
                break;
            }

            // Generate dummy Opus audio data every 20ms (more frequent than video)
            if frame_count % 1 == 0 { // Audio frame every video frame for simplicity
                let audio_payload = Self::generate_opus_frame(frame_count);
                if let Err(e) = audio_track.write_rtp(&audio_payload).await {
                    tracing::warn!("Failed to write audio RTP: {}", e);
                    break;
                }
            }

            frame_count += 1;

            // Log progress every 5 seconds (150 frames at 30fps)
            if frame_count % 150 == 0 {
                info!("Streamed {} frames", frame_count);
            }
        }

        Ok(())
    }

    fn generate_h264_frame(frame_number: u64, _width: u32, _height: u32) -> webrtc::rtp::packet::Packet {
        // Generate minimal H.264 frame data
        // In production, this would come from the actual video encoder
        let payload = vec![0x00, 0x00, 0x00, 0x01, 0x65]; // H.264 IDR frame header

        webrtc::rtp::packet::Packet {
            header: webrtc::rtp::header::Header {
                version: 2,
                padding: false,
                extension: false,
                marker: true,
                payload_type: 96, // H.264
                sequence_number: (frame_number & 0xFFFF) as u16,
                timestamp: (frame_number * 3000) as u32, // 90kHz clock
                ssrc: 0x12345678,
                ..Default::default()
            },
            payload: payload.into(),
        }
    }

    fn generate_opus_frame(frame_number: u64) -> webrtc::rtp::packet::Packet {
        // Generate minimal Opus frame data
        let payload = vec![0xFC]; // Opus silence frame

        webrtc::rtp::packet::Packet {
            header: webrtc::rtp::header::Header {
                version: 2,
                padding: false,
                extension: false,
                marker: false,
                payload_type: 111, // Opus
                sequence_number: (frame_number & 0xFFFF) as u16,
                timestamp: (frame_number * 960) as u32, // 48kHz clock, 20ms frames
                ssrc: 0x87654321,
                ..Default::default()
            },
            payload: payload.into(),
        }
    }

    pub async fn cleanup_session(&self, session_id: Uuid) {
        if let Some((_, peer_connection)) = self.peer_connections.remove(&session_id) {
            if let Err(e) = peer_connection.close().await {
                tracing::warn!("Error closing peer connection: {}", e);
            }
        }

        // Remove from pending offers
        self.pending_offers.retain(|_, &mut v| v != session_id);
    }
}