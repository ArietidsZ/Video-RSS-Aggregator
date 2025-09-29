mod model_selector;

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int};
use std::path::Path;
use std::slice;
use std::sync::Arc;
use std::time::{Duration, Instant};

use thiserror::Error;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

pub use model_selector::{
    ModelSelector, ModelSelectionCriteria, AudioQuality, ContentType,
    HardwareTier, LatencyRequirement, AccuracyRequirement
};

#[derive(Error, Debug)]
pub enum TranscriptionError {
    #[error("Failed to initialize Whisper Turbo: {0}")]
    InitializationError(String),

    #[error("Model loading failed: {0}")]
    ModelLoadError(String),

    #[error("Audio processing error: {0}")]
    AudioError(String),

    #[error("Transcription failed: {0}")]
    TranscriptionError(String),

    #[error("FFI error: {0}")]
    FFIError(String),

    #[error("IO error: {0}")]
    IOError(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, TranscriptionError>;

// FFI bindings to C++ Whisper Turbo
#[repr(C)]
struct WhisperTurboHandle {
    _unused: [u8; 0],
}

extern "C" {
    fn whisper_turbo_create() -> *mut WhisperTurboHandle;
    fn whisper_turbo_destroy(handle: *mut WhisperTurboHandle);
    fn whisper_turbo_init(
        handle: *mut WhisperTurboHandle,
        model_path: *const c_char,
        model_size: c_int,
        quantization: c_int,
    ) -> c_int;
    fn whisper_turbo_transcribe(
        handle: *mut WhisperTurboHandle,
        audio_data: *const c_float,
        num_samples: usize,
        sample_rate: c_int,
    ) -> *const c_char;
    fn whisper_turbo_get_error(handle: *mut WhisperTurboHandle) -> *const c_char;
    fn whisper_turbo_get_rtf(handle: *mut WhisperTurboHandle) -> c_float;
    fn whisper_turbo_get_memory_usage(handle: *mut WhisperTurboHandle) -> usize;
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ModelSize {
    Tiny = 0,
    Base = 1,
    Small = 2,
    Medium = 3,
    Large = 4,
    Turbo = 5,
    Ultra = 6,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum QuantizationType {
    FP32 = 0,
    FP16 = 1,
    BF16 = 2,
    INT8 = 3,
    FP8 = 4,
    INT4 = 5,
    Dynamic = 6,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ModelVariant {
    WhisperTurbo,
    FasterWhisper,
    WhisperStandard,
    Paraformer,
    FunASR,
    Qwen3ASR,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LanguageOptimization {
    AutoDetect,
    English,
    Chinese,
    Multilingual,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionOptions {
    pub model_variant: ModelVariant,
    pub model_size: ModelSize,
    pub quantization: QuantizationType,
    pub language: LanguageOptimization,
    pub batch_size: u32,
    pub beam_size: u32,
    pub temperature: f32,
    pub vad_threshold: f32,
    pub chunk_length_ms: u32,
    pub enable_timestamps: bool,
    pub enable_diarization: bool,
    pub model_path: String,
    pub hotwords: Vec<String>,
}

impl Default for TranscriptionOptions {
    fn default() -> Self {
        Self {
            model_variant: ModelVariant::WhisperTurbo,
            model_size: ModelSize::Large,
            quantization: QuantizationType::INT8,
            language: LanguageOptimization::AutoDetect,
            batch_size: 1,
            beam_size: 5,
            temperature: 0.0,
            vad_threshold: 0.5,
            chunk_length_ms: 5000,
            enable_timestamps: true,
            enable_diarization: false,
            model_path: String::new(),
            hotwords: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionSegment {
    pub text: String,
    pub start_time: f32,
    pub end_time: f32,
    pub confidence: f32,
    pub language: String,
    pub speaker_id: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResult {
    pub segments: Vec<TranscriptionSegment>,
    pub full_text: String,
    pub detected_language: String,
    pub average_confidence: f32,
    pub processing_time: Duration,
    pub rtf: f32,  // Real-time factor
    pub tokens_generated: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub tokens_per_second: f32,
    pub audio_processing_speed: f32,  // xRT
    pub peak_memory_mb: usize,
    pub current_memory_mb: usize,
    pub gpu_utilization: f32,
}

pub struct WhisperTurbo {
    handle: *mut WhisperTurboHandle,
    options: TranscriptionOptions,
    metrics: Arc<RwLock<PerformanceMetrics>>,
    initialized: bool,
}

impl WhisperTurbo {
    pub fn new() -> Self {
        let handle = unsafe { whisper_turbo_create() };

        Self {
            handle,
            options: TranscriptionOptions::default(),
            metrics: Arc::new(RwLock::new(PerformanceMetrics {
                tokens_per_second: 0.0,
                audio_processing_speed: 0.0,
                peak_memory_mb: 0,
                current_memory_mb: 0,
                gpu_utilization: 0.0,
            })),
            initialized: false,
        }
    }

    pub fn initialize(&mut self, options: TranscriptionOptions) -> Result<()> {
        self.options = options.clone();

        let model_path = CString::new(options.model_path.clone())
            .map_err(|e| TranscriptionError::FFIError(format!("Invalid model path: {}", e)))?;

        let result = unsafe {
            whisper_turbo_init(
                self.handle,
                model_path.as_ptr(),
                options.model_size as c_int,
                options.quantization as c_int,
            )
        };

        if result == 0 {
            self.initialized = true;
            info!("Whisper Turbo initialized successfully");
            Ok(())
        } else {
            let error = unsafe {
                let err_ptr = whisper_turbo_get_error(self.handle);
                CStr::from_ptr(err_ptr)
                    .to_string_lossy()
                    .to_string()
            };
            Err(TranscriptionError::InitializationError(error))
        }
    }

    pub fn transcribe(&self, audio_data: &[f32], sample_rate: u32) -> Result<TranscriptionResult> {
        if !self.initialized {
            return Err(TranscriptionError::TranscriptionError(
                "Model not initialized".to_string()
            ));
        }

        let start_time = Instant::now();

        let result_ptr = unsafe {
            whisper_turbo_transcribe(
                self.handle,
                audio_data.as_ptr(),
                audio_data.len(),
                sample_rate as c_int,
            )
        };

        if result_ptr.is_null() {
            let error = unsafe {
                let err_ptr = whisper_turbo_get_error(self.handle);
                CStr::from_ptr(err_ptr)
                    .to_string_lossy()
                    .to_string()
            };
            return Err(TranscriptionError::TranscriptionError(error));
        }

        let full_text = unsafe {
            CStr::from_ptr(result_ptr)
                .to_string_lossy()
                .to_string()
        };

        let processing_time = start_time.elapsed();
        let rtf = unsafe { whisper_turbo_get_rtf(self.handle) };

        // Update metrics
        let memory_usage = unsafe { whisper_turbo_get_memory_usage(self.handle) };

        let result = TranscriptionResult {
            segments: vec![],  // Would parse segments from detailed output
            full_text,
            detected_language: "auto".to_string(),
            average_confidence: 0.95,
            processing_time,
            rtf,
            tokens_generated: 0,
        };

        // Update performance metrics
        tokio::spawn({
            let metrics = self.metrics.clone();
            async move {
                let mut m = metrics.write().await;
                m.audio_processing_speed = rtf;
                m.current_memory_mb = memory_usage;
                m.peak_memory_mb = m.peak_memory_mb.max(memory_usage);
            }
        });

        Ok(result)
    }

    pub async fn transcribe_file(&self, audio_path: &Path) -> Result<TranscriptionResult> {
        let audio_data = load_audio_file(audio_path)?;
        self.transcribe(&audio_data.samples, audio_data.sample_rate)
    }

    pub async fn transcribe_batch(&self, audio_paths: Vec<&Path>) -> Result<Vec<TranscriptionResult>> {
        let mut results = Vec::new();

        for path in audio_paths {
            let result = self.transcribe_file(path).await?;
            results.push(result);
        }

        Ok(results)
    }

    pub fn set_language(&mut self, language: LanguageOptimization) {
        self.options.language = language;
    }

    pub fn add_hotwords(&mut self, hotwords: Vec<String>) {
        self.options.hotwords = hotwords;
    }

    pub async fn get_metrics(&self) -> PerformanceMetrics {
        self.metrics.read().await.clone()
    }

    pub async fn transcribe_intelligently(
        &mut self,
        audio_data: &[f32],
        sample_rate: u32,
        hardware_tier: HardwareTier,
    ) -> Result<TranscriptionResult> {
        let selector = ModelSelector::new();

        // Quick analysis of audio for language and quality detection
        let audio_quality = selector.estimate_audio_quality(audio_data);

        // Get first few seconds for language detection
        let preview_samples = &audio_data[..std::cmp::min(audio_data.len(), sample_rate as usize * 3)];
        let detected_language = self.detect_language_quick(preview_samples).await;

        // Initial transcription of preview for content type detection
        let preview_result = self.transcribe(preview_samples, sample_rate)?;
        let content_type = selector.detect_content_type(preview_samples, &preview_result.full_text);

        // Build selection criteria
        let criteria = ModelSelectionCriteria {
            detected_language,
            audio_duration_seconds: audio_data.len() as f32 / sample_rate as f32,
            audio_quality,
            content_type,
            hardware_tier,
            latency_requirement: LatencyRequirement::Fast,
            accuracy_requirement: AccuracyRequirement::High,
        };

        // Select optimal model
        let optimal_options = selector.select_optimal_model(&criteria);

        info!("Selected model {:?} for language {:?}, content type {:?}",
              optimal_options.model_variant, criteria.detected_language, content_type);

        // Re-initialize with optimal model if different
        if self.options.model_variant != optimal_options.model_variant ||
           self.options.model_size != optimal_options.model_size {
            self.initialize(optimal_options)?;
        }

        // Perform full transcription with optimal model
        self.transcribe(audio_data, sample_rate)
    }

    async fn detect_language_quick(&self, audio_samples: &[f32]) -> Option<String> {
        // Quick language detection using first few seconds
        // This would use a lightweight language ID model

        // For now, return None to trigger auto-detection
        None
    }

    pub fn optimize_for_hardware(&mut self, hardware_config: &HardwareConfig) {
        // Adjust settings based on hardware capabilities
        if hardware_config.has_tensor_cores {
            self.options.quantization = QuantizationType::FP16;
        }

        if hardware_config.gpu_memory_gb < 8.0 {
            self.options.batch_size = 1;
            self.options.quantization = QuantizationType::INT8;
        } else if hardware_config.gpu_memory_gb >= 24.0 {
            self.options.batch_size = 4;
        }

        if hardware_config.cpu_cores >= 16 {
            // Enable more parallel preprocessing
        }

        info!("Optimized Whisper Turbo for hardware: {:?}", hardware_config);
    }
}

impl Drop for WhisperTurbo {
    fn drop(&mut self) {
        unsafe {
            whisper_turbo_destroy(self.handle);
        }
    }
}

unsafe impl Send for WhisperTurbo {}
unsafe impl Sync for WhisperTurbo {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    pub cpu_cores: usize,
    pub gpu_memory_gb: f32,
    pub has_tensor_cores: bool,
    pub has_cuda: bool,
    pub has_rocm: bool,
    pub has_metal: bool,
}

#[derive(Debug)]
struct AudioData {
    samples: Vec<f32>,
    sample_rate: u32,
}

fn load_audio_file(path: &Path) -> Result<AudioData> {
    // Use symphonia for audio decoding
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = std::fs::File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let hint = Hint::new();
    let format_opts: FormatOptions = Default::default();
    let metadata_opts: MetadataOptions = Default::default();
    let decoder_opts: DecoderOptions = Default::default();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &metadata_opts)
        .map_err(|e| TranscriptionError::AudioError(format!("Probe failed: {}", e)))?;

    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| TranscriptionError::AudioError("No supported audio track found".to_string()))?;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &decoder_opts)
        .map_err(|e| TranscriptionError::AudioError(format!("Decoder creation failed: {}", e)))?;

    let track_id = track.id;
    let mut sample_buffer = None;
    let mut samples = Vec::new();

    while let Ok(packet) = format.next_packet() {
        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                if sample_buffer.is_none() {
                    let spec = *decoded.spec();
                    let duration = decoded.capacity() as u64;
                    sample_buffer = Some(SampleBuffer::<f32>::new(duration, spec));
                }

                if let Some(ref mut buffer) = sample_buffer {
                    buffer.copy_interleaved_ref(decoded);
                    samples.extend_from_slice(buffer.samples());
                }
            }
            Err(err) => {
                warn!("Decode error: {}", err);
            }
        }
    }

    let sample_rate = track.codec_params.sample_rate.unwrap_or(16000);

    Ok(AudioData {
        samples,
        sample_rate,
    })
}

// Integration with hardware detector
pub fn create_optimal_transcriber(
    hardware_profile: &crate::hardware_detector::HardwareProfile,
) -> Result<WhisperTurbo> {
    let mut transcriber = WhisperTurbo::new();

    let hardware_config = HardwareConfig {
        cpu_cores: hardware_profile.cpu.cores_logical,
        gpu_memory_gb: hardware_profile.gpus.first()
            .map(|gpu| gpu.memory_total as f32 / (1024.0 * 1024.0 * 1024.0))
            .unwrap_or(0.0),
        has_tensor_cores: hardware_profile.gpus.iter()
            .any(|gpu| gpu.tensor_cores.unwrap_or(0) > 0),
        has_cuda: hardware_profile.gpus.iter()
            .any(|gpu| matches!(gpu.vendor, crate::hardware_detector::GpuVendor::Nvidia)),
        has_rocm: hardware_profile.gpus.iter()
            .any(|gpu| matches!(gpu.vendor, crate::hardware_detector::GpuVendor::AMD)),
        has_metal: hardware_profile.gpus.iter()
            .any(|gpu| matches!(gpu.vendor, crate::hardware_detector::GpuVendor::Apple)),
    };

    transcriber.optimize_for_hardware(&hardware_config);

    Ok(transcriber)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transcriber_creation() {
        let transcriber = WhisperTurbo::new();
        assert!(!transcriber.initialized);
    }

    #[tokio::test]
    async fn test_metrics() {
        let transcriber = WhisperTurbo::new();
        let metrics = transcriber.get_metrics().await;
        assert_eq!(metrics.tokens_per_second, 0.0);
    }
}