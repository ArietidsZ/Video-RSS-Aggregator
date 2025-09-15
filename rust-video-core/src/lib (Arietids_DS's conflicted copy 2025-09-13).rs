pub mod gpu;
pub mod video;
pub mod transcription;
pub mod error;
pub mod ffi;
pub mod database;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoAnalysisConfig {
    pub gpu_backend: GpuBackendType,
    pub max_frames: usize,
    pub transcription_model: String,
    pub batch_size: usize,
    pub enable_cache: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum GpuBackendType {
    Auto,
    Cuda,
    Rocm,
    OpenVino,
    Metal,
    Cpu,
}

impl Default for VideoAnalysisConfig {
    fn default() -> Self {
        Self {
            gpu_backend: GpuBackendType::Auto,
            max_frames: 100,
            transcription_model: "whisper-large-v3".to_string(),
            batch_size: 16,
            enable_cache: true,
        }
    }
}

pub struct VideoProcessor {
    config: VideoAnalysisConfig,
    gpu_backend: Box<dyn gpu::GpuBackend>,
    transcriber: Box<dyn transcription::Transcriber>,
}

impl VideoProcessor {
    pub async fn new(config: VideoAnalysisConfig) -> Result<Self> {
        let gpu_backend = gpu::create_backend(config.gpu_backend).await?;
        let transcriber = transcription::create_transcriber(&config, &*gpu_backend).await?;

        Ok(Self {
            config,
            gpu_backend,
            transcriber,
        })
    }

    pub async fn process_video(&self, video_path: &Path) -> Result<VideoAnalysisResult> {
        let video_data = video::extract_video_data(video_path, self.config.max_frames).await?;

        let visual_features = self.gpu_backend.extract_visual_features(&video_data.frames).await?;

        let transcript = if !video_data.audio_buffer.is_empty() {
            Some(self.transcriber.transcribe(&video_data.audio_buffer).await?)
        } else {
            None
        };

        let summary = self.generate_summary(&visual_features, &transcript).await?;

        Ok(VideoAnalysisResult {
            video_path: video_path.to_string_lossy().to_string(),
            duration: video_data.duration,
            fps: video_data.fps,
            resolution: video_data.resolution,
            visual_features,
            transcript,
            summary,
            processing_time_ms: 0, // Will be set by caller
        })
    }

    async fn generate_summary(
        &self,
        visual_features: &VisualFeatures,
        transcript: &Option<Transcript>,
    ) -> Result<String> {
        // AI-powered summary generation
        let summary_prompt = format!(
            "Visual: {} scenes detected, {} objects identified. Audio: {}",
            visual_features.scene_count,
            visual_features.detected_objects.len(),
            transcript.as_ref()
                .map(|t| format!("{} words transcribed", t.text.split_whitespace().count()))
                .unwrap_or_else(|| "No audio".to_string())
        );

        // This would call an LLM for actual summary generation
        Ok(summary_prompt)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoAnalysisResult {
    pub video_path: String,
    pub duration: f64,
    pub fps: f64,
    pub resolution: (u32, u32),
    pub visual_features: VisualFeatures,
    pub transcript: Option<Transcript>,
    pub summary: String,
    pub processing_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualFeatures {
    pub scene_count: usize,
    pub detected_objects: Vec<DetectedObject>,
    pub dominant_colors: Vec<Color>,
    pub motion_intensity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedObject {
    pub label: String,
    pub confidence: f32,
    pub bounding_box: BoundingBox,
    pub frame_index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub percentage: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transcript {
    pub text: String,
    pub segments: Vec<TranscriptSegment>,
    pub language: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptSegment {
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub confidence: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gpu_backend_detection() {
        let config = VideoAnalysisConfig::default();
        let processor = VideoProcessor::new(config).await;
        assert!(processor.is_ok());
    }
}