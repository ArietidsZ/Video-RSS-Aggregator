use crate::{error::VideoRssError, Result};
use ndarray::{Array1, Array2, ArrayView1};
use ort::{Environment, ExecutionProvider, GraphOptimizationLevel, Session, SessionBuilder, Value};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{Decoder, DecoderOptions};
use symphonia::core::errors::Error as SymphonyError;
use symphonia::core::formats::{FormatOptions, FormatReader};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use tokio::sync::{Mutex, Semaphore};
use tracing::{debug, error, info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionConfig {
    pub model_path: PathBuf,
    pub language: String,
    pub sample_rate: u32,
    pub chunk_length: usize,
    pub beam_size: usize,
    pub max_concurrent: usize,
    pub use_vad: bool,
    pub vad_threshold: f32,
}

impl Default for TranscriptionConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models/sherpa-onnx-streaming-zipformer-ctc-zh-en.onnx"),
            language: "zh-en".to_string(),
            sample_rate: 16000,
            chunk_length: 1600, // 100ms at 16kHz
            beam_size: 4,
            max_concurrent: 4,
            use_vad: true,
            vad_threshold: 0.5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResult {
    pub text: String,
    pub language: String,
    pub confidence: f32,
    pub segments: Vec<TranscriptionSegment>,
    pub processing_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionSegment {
    pub start_time: f32,
    pub end_time: f32,
    pub text: String,
    pub confidence: f32,
}

pub struct NativeTranscriber {
    session: Arc<Session>,
    config: TranscriptionConfig,
    semaphore: Arc<Semaphore>,
    vad_session: Option<Arc<Session>>,
    environment: Arc<Environment>,
}

impl NativeTranscriber {
    pub async fn new(config: TranscriptionConfig) -> Result<Self> {
        info!("Initializing native transcriber with ONNX Runtime");

        let environment = Arc::new(
            Environment::builder()
                .with_name("video_transcription")
                .with_log_level(ort::LoggingLevel::Warning)
                .build()
                .map_err(|e| VideoRssError::Config(format!("Failed to create ONNX environment: {}", e)))?,
        );

        if !config.model_path.exists() {
            return Err(VideoRssError::Config(format!(
                "Transcription model not found: {}",
                config.model_path.display()
            )));
        }

        let session = Arc::new(
            SessionBuilder::new(&environment)
                .map_err(|e| VideoRssError::Config(format!("Failed to create session builder: {}", e)))?
                .with_optimization_level(GraphOptimizationLevel::All)
                .map_err(|e| VideoRssError::Config(format!("Failed to set optimization level: {}", e)))?
                .with_intra_threads(2)
                .map_err(|e| VideoRssError::Config(format!("Failed to set intra threads: {}", e)))?
                .with_model_from_file(&config.model_path)
                .map_err(|e| VideoRssError::Config(format!("Failed to load model: {}", e)))?,
        );

        // Load VAD model if enabled
        let vad_session = if config.use_vad {
            let vad_path = config.model_path.parent()
                .unwrap_or(Path::new("models"))
                .join("silero_vad.onnx");

            if vad_path.exists() {
                Some(Arc::new(
                    SessionBuilder::new(&environment)
                        .map_err(|e| VideoRssError::Config(format!("Failed to create VAD session builder: {}", e)))?
                        .with_optimization_level(GraphOptimizationLevel::All)
                        .map_err(|e| VideoRssError::Config(format!("Failed to set VAD optimization level: {}", e)))?
                        .with_model_from_file(&vad_path)
                        .map_err(|e| VideoRssError::Config(format!("Failed to load VAD model: {}", e)))?,
                ))
            } else {
                warn!("VAD enabled but model not found at {}", vad_path.display());
                None
            }
        } else {
            None
        };

        info!("Native transcriber initialized successfully");

        Ok(Self {
            session,
            config,
            semaphore: Arc::new(Semaphore::new(config.max_concurrent)),
            vad_session,
            environment,
        })
    }

    pub async fn transcribe_audio_file(&self, audio_path: &Path) -> Result<TranscriptionResult> {
        let _permit = self.semaphore.acquire().await.unwrap();
        let start_time = std::time::Instant::now();

        debug!("Starting transcription for: {}", audio_path.display());

        // Load and process audio
        let audio_data = self.load_audio_file(audio_path).await?;

        // Apply VAD if enabled
        let audio_segments = if self.vad_session.is_some() {
            self.apply_vad(&audio_data).await?
        } else {
            vec![(0.0, audio_data.len() as f32 / self.config.sample_rate as f32, audio_data)]
        };

        let mut all_segments = Vec::new();
        let mut full_text = String::new();
        let mut total_confidence = 0.0;

        // Process each VAD segment
        for (start_sec, end_sec, segment_audio) in audio_segments {
            if segment_audio.len() < self.config.chunk_length {
                continue; // Skip very short segments
            }

            let segment_result = self.transcribe_audio_chunk(&segment_audio, start_sec).await?;

            if !segment_result.text.trim().is_empty() {
                all_segments.push(TranscriptionSegment {
                    start_time: start_sec,
                    end_time: end_sec,
                    text: segment_result.text.clone(),
                    confidence: segment_result.confidence,
                });

                if !full_text.is_empty() {
                    full_text.push(' ');
                }
                full_text.push_str(&segment_result.text);
                total_confidence += segment_result.confidence;
            }
        }

        let avg_confidence = if all_segments.is_empty() {
            0.0
        } else {
            total_confidence / all_segments.len() as f32
        };

        let processing_time = start_time.elapsed().as_millis() as u64;

        info!(
            "Transcription completed in {}ms: {} segments, confidence: {:.2}",
            processing_time,
            all_segments.len(),
            avg_confidence
        );

        Ok(TranscriptionResult {
            text: full_text,
            language: self.config.language.clone(),
            confidence: avg_confidence,
            segments: all_segments,
            processing_time_ms: processing_time,
        })
    }

    async fn load_audio_file(&self, audio_path: &Path) -> Result<Vec<f32>> {
        let file = std::fs::File::open(audio_path)
            .map_err(|e| VideoRssError::FileError(format!("Failed to open audio file: {}", e)))?;

        let mss = MediaSourceStream::new(Box::new(file), Default::default());
        let mut hint = Hint::new();

        if let Some(extension) = audio_path.extension() {
            hint.with_extension(&extension.to_string_lossy());
        }

        let meta_opts: MetadataOptions = Default::default();
        let fmt_opts: FormatOptions = Default::default();

        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &fmt_opts, &meta_opts)
            .map_err(|e| VideoRssError::Config(format!("Failed to probe audio format: {}", e)))?;

        let mut format = probed.format;
        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
            .ok_or_else(|| VideoRssError::Config("No audio track found".to_string()))?;

        let dec_opts: DecoderOptions = Default::default();
        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &dec_opts)
            .map_err(|e| VideoRssError::Config(format!("Failed to create decoder: {}", e)))?;

        let track_id = track.id;
        let mut audio_samples = Vec::new();

        loop {
            let packet = match format.next_packet() {
                Ok(packet) => packet,
                Err(SymphonyError::ResetRequired) => {
                    return Err(VideoRssError::Config("Decoder reset required".to_string()));
                }
                Err(SymphonyError::IoError(_)) => break,
                Err(e) => {
                    return Err(VideoRssError::Config(format!("Packet decode error: {}", e)));
                }
            };

            if packet.track_id() != track_id {
                continue;
            }

            match decoder.decode(&packet) {
                Ok(audio_buf) => {
                    self.convert_audio_buffer(&audio_buf, &mut audio_samples)?;
                }
                Err(SymphonyError::IoError(_)) => break,
                Err(SymphonyError::DecodeError(_)) => continue,
                Err(e) => {
                    return Err(VideoRssError::Config(format!("Audio decode error: {}", e)));
                }
            }
        }

        // Resample to target sample rate if needed
        let resampled = self.resample_audio(&audio_samples, &track.codec_params.sample_rate.unwrap_or(44100))?;

        Ok(resampled)
    }

    fn convert_audio_buffer(&self, audio_buf: &AudioBufferRef, output: &mut Vec<f32>) -> Result<()> {
        match audio_buf {
            AudioBufferRef::F32(buf) => {
                // Convert to mono by averaging channels
                let channels = buf.spec().channels.count();
                let frames = buf.frames();

                for frame_idx in 0..frames {
                    let mut sample_sum = 0.0;
                    for ch in 0..channels {
                        sample_sum += buf.chan(ch)[frame_idx];
                    }
                    output.push(sample_sum / channels as f32);
                }
            }
            AudioBufferRef::S16(buf) => {
                let channels = buf.spec().channels.count();
                let frames = buf.frames();

                for frame_idx in 0..frames {
                    let mut sample_sum = 0.0;
                    for ch in 0..channels {
                        sample_sum += buf.chan(ch)[frame_idx] as f32 / i16::MAX as f32;
                    }
                    output.push(sample_sum / channels as f32);
                }
            }
            _ => {
                return Err(VideoRssError::Config("Unsupported audio format".to_string()));
            }
        }
        Ok(())
    }

    fn resample_audio(&self, audio: &[f32], source_rate: &u32) -> Result<Vec<f32>> {
        if *source_rate == self.config.sample_rate {
            return Ok(audio.to_vec());
        }

        // Simple linear interpolation resampling
        let ratio = *source_rate as f64 / self.config.sample_rate as f64;
        let output_len = (audio.len() as f64 / ratio) as usize;
        let mut output = Vec::with_capacity(output_len);

        for i in 0..output_len {
            let src_idx = i as f64 * ratio;
            let src_idx_floor = src_idx.floor() as usize;
            let src_idx_ceil = (src_idx_floor + 1).min(audio.len() - 1);
            let frac = src_idx - src_idx_floor as f64;

            if src_idx_floor < audio.len() {
                let sample = audio[src_idx_floor] * (1.0 - frac) as f32 +
                           audio[src_idx_ceil] * frac as f32;
                output.push(sample);
            }
        }

        Ok(output)
    }

    async fn apply_vad(&self, audio: &[f32]) -> Result<Vec<(f32, f32, Vec<f32>)>> {
        if let Some(vad_session) = &self.vad_session {
            let chunk_size = self.config.sample_rate as usize; // 1 second chunks
            let mut segments = Vec::new();
            let mut current_segment: Option<(usize, Vec<f32>)> = None;

            for (i, chunk) in audio.chunks(chunk_size).enumerate() {
                let chunk_array = Array1::from_vec(chunk.to_vec()).into_dyn();
                let inputs = vec![Value::from_array(vad_session.allocator(), &chunk_array)
                    .map_err(|e| VideoRssError::Config(format!("Failed to create VAD input: {}", e)))?];

                let outputs = vad_session
                    .run(inputs)
                    .map_err(|e| VideoRssError::Config(format!("VAD inference failed: {}", e)))?;

                let probability = outputs[0]
                    .try_extract::<f32>()
                    .map_err(|e| VideoRssError::Config(format!("Failed to extract VAD output: {}", e)))?
                    .view()
                    .iter()
                    .next()
                    .copied()
                    .unwrap_or(0.0);

                let start_sample = i * chunk_size;
                let is_speech = probability > self.config.vad_threshold;

                if is_speech {
                    match &mut current_segment {
                        Some((_, ref mut segment_audio)) => {
                            segment_audio.extend_from_slice(chunk);
                        }
                        None => {
                            current_segment = Some((start_sample, chunk.to_vec()));
                        }
                    }
                } else if let Some((segment_start, segment_audio)) = current_segment.take() {
                    // End of speech segment
                    let start_time = segment_start as f32 / self.config.sample_rate as f32;
                    let end_time = (segment_start + segment_audio.len()) as f32 / self.config.sample_rate as f32;
                    segments.push((start_time, end_time, segment_audio));
                }
            }

            // Handle final segment if still active
            if let Some((segment_start, segment_audio)) = current_segment {
                let start_time = segment_start as f32 / self.config.sample_rate as f32;
                let end_time = (segment_start + segment_audio.len()) as f32 / self.config.sample_rate as f32;
                segments.push((start_time, end_time, segment_audio));
            }

            Ok(segments)
        } else {
            // No VAD, return entire audio as one segment
            Ok(vec![(0.0, audio.len() as f32 / self.config.sample_rate as f32, audio.to_vec())])
        }
    }

    async fn transcribe_audio_chunk(&self, audio: &[f32], start_time: f32) -> Result<TranscriptionResult> {
        // Prepare input tensor
        let audio_array = Array2::from_shape_vec((1, audio.len()), audio.to_vec())
            .map_err(|e| VideoRssError::Config(format!("Failed to create audio array: {}", e)))?;

        let inputs = vec![
            Value::from_array(self.session.allocator(), &audio_array.into_dyn())
                .map_err(|e| VideoRssError::Config(format!("Failed to create input tensor: {}", e)))?,
        ];

        // Run inference
        let outputs = self.session
            .run(inputs)
            .map_err(|e| VideoRssError::Config(format!("Transcription inference failed: {}", e)))?;

        // Extract text and confidence from outputs
        let (text, confidence) = self.decode_outputs(&outputs)?;

        Ok(TranscriptionResult {
            text,
            language: self.config.language.clone(),
            confidence,
            segments: vec![], // Individual chunk doesn't need segments
            processing_time_ms: 0, // Will be set at higher level
        })
    }

    fn decode_outputs(&self, outputs: &[Value]) -> Result<(String, f32)> {
        if outputs.is_empty() {
            return Ok((String::new(), 0.0));
        }

        // For streaming CTC models, output is typically token probabilities
        let logits = outputs[0]
            .try_extract::<f32>()
            .map_err(|e| VideoRssError::Config(format!("Failed to extract output tensor: {}", e)))?;

        // Simple CTC decoding - take argmax and remove blanks/repeats
        let tokens = self.ctc_decode(logits.view())?;
        let text = self.tokens_to_text(&tokens)?;

        // Calculate average confidence from max probabilities
        let confidence = self.calculate_confidence(logits.view());

        Ok((text, confidence))
    }

    fn ctc_decode(&self, logits: ArrayView1<f32>) -> Result<Vec<usize>> {
        let vocab_size = 4096; // Approximate vocab size for Chinese models
        let timesteps = logits.len() / vocab_size;
        let mut tokens = Vec::new();
        let mut last_token = None;

        for t in 0..timesteps {
            let start_idx = t * vocab_size;
            let end_idx = start_idx + vocab_size;

            if end_idx <= logits.len() {
                let frame_logits = &logits[start_idx..end_idx];
                let (max_idx, _) = frame_logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or((0, &0.0));

                // CTC blank token is typically 0
                if max_idx != 0 && Some(max_idx) != last_token {
                    tokens.push(max_idx);
                }
                last_token = Some(max_idx);
            }
        }

        Ok(tokens)
    }

    fn tokens_to_text(&self, tokens: &[usize]) -> Result<String> {
        // Simplified token-to-text conversion
        // In a real implementation, you'd load the model's vocabulary
        let mut text = String::new();

        for &token in tokens {
            // Basic mapping - in practice, load actual vocabulary
            if token < 26 {
                text.push((b'a' + token as u8) as char);
            } else if token < 52 {
                text.push((b'A' + (token - 26) as u8) as char);
            } else if token < 62 {
                text.push((b'0' + (token - 52) as u8) as char);
            } else {
                text.push(' '); // Space or unknown
            }
        }

        Ok(text)
    }

    fn calculate_confidence(&self, logits: ArrayView1<f32>) -> f32 {
        let vocab_size = 4096;
        let timesteps = logits.len() / vocab_size;
        let mut total_confidence = 0.0;

        for t in 0..timesteps {
            let start_idx = t * vocab_size;
            let end_idx = start_idx + vocab_size;

            if end_idx <= logits.len() {
                let frame_logits = &logits[start_idx..end_idx];
                let max_prob = frame_logits
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .copied()
                    .unwrap_or(0.0);

                // Apply softmax to get probability
                let exp_max = max_prob.exp();
                let sum_exp: f32 = frame_logits.iter().map(|x| x.exp()).sum();
                total_confidence += exp_max / sum_exp;
            }
        }

        if timesteps > 0 {
            total_confidence / timesteps as f32
        } else {
            0.0
        }
    }

    pub async fn download_models(&self) -> Result<()> {
        info!("Downloading required models for native transcription");

        let models_dir = self.config.model_path.parent().unwrap_or(Path::new("models"));
        fs::create_dir_all(models_dir)
            .map_err(|e| VideoRssError::FileError(format!("Failed to create models directory: {}", e)))?;

        // Download sherpa-onnx models
        let model_urls = vec![
            (
                "sherpa-onnx-streaming-zipformer-ctc-zh-en.onnx",
                "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-zh-en-2023-10-24.tar.bz2"
            ),
            (
                "silero_vad.onnx",
                "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
            ),
        ];

        for (filename, url) in model_urls {
            let model_path = models_dir.join(filename);
            if !model_path.exists() {
                info!("Downloading model: {}", filename);
                self.download_file(url, &model_path).await?;
            }
        }

        info!("Model download completed");
        Ok(())
    }

    async fn download_file(&self, url: &str, path: &Path) -> Result<()> {
        let client = reqwest::Client::new();
        let response = client
            .get(url)
            .send()
            .await
            .map_err(|e| VideoRssError::Http(e))?;

        if !response.status().is_success() {
            return Err(VideoRssError::Config(format!(
                "Failed to download {}: {}",
                url,
                response.status()
            )));
        }

        let bytes = response
            .bytes()
            .await
            .map_err(|e| VideoRssError::Http(e))?;

        fs::write(path, bytes)
            .map_err(|e| VideoRssError::FileError(format!("Failed to save model: {}", e)))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_audio_loading() {
        let temp_dir = TempDir::new().unwrap();
        let config = TranscriptionConfig {
            model_path: temp_dir.path().join("test.onnx"),
            ..Default::default()
        };

        // Create a simple test WAV file
        let wav_path = temp_dir.path().join("test.wav");
        create_test_wav(&wav_path);

        // This test would require actual ONNX model files
        // For now, just test the configuration
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.language, "zh-en");
    }

    fn create_test_wav(path: &Path) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer = hound::WavWriter::create(path, spec).unwrap();

        // Write 1 second of silence
        for _ in 0..16000 {
            writer.write_sample(0i16).unwrap();
        }

        writer.finalize().unwrap();
    }
}