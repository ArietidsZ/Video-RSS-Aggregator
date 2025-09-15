use crate::{error::VideoRssError, Result};
use anyhow::Error as E;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m, audio, Config};
use hf_hub::{api::tokio::Api, Repo, RepoType};
use rand::{rngs::StdRng, SeedableRng};
use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{Decoder, DecoderOptions};
use symphonia::core::formats::{FormatOptions, FormatReader};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use tokenizers::Tokenizer;
use tokio::sync::{Mutex, Semaphore};
use tracing::{debug, error, info, warn};

const SAMPLE_RATE: usize = 16000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhisperConfig {
    pub model_id: String,
    pub model_type: WhisperModel,
    pub device: DeviceConfig,
    pub language: Option<String>,
    pub task: Task,
    pub timestamps: bool,
    pub temperature: f64,
    pub max_concurrent: usize,
    pub quantized: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WhisperModel {
    Tiny,
    Base,
    Small,
    Medium,
    Large,
    LargeV2,
    LargeV3,
    DistilLargeV3,  // 6x faster
    TurboLargeV3,   // 8x faster
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceConfig {
    Cpu,
    Cuda(usize),
    Metal,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Task {
    Transcribe,
    Translate,
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self {
            model_id: "distil-whisper/distil-large-v3".to_string(),
            model_type: WhisperModel::DistilLargeV3,
            device: DeviceConfig::Cpu,
            language: Some("en".to_string()),
            task: Task::Transcribe,
            timestamps: false,
            temperature: 0.0,
            max_concurrent: 4,
            quantized: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResult {
    pub text: String,
    pub segments: Vec<TranscriptionSegment>,
    pub language: String,
    pub processing_time_ms: u64,
    pub audio_duration_ms: u64,
    pub rtf: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionSegment {
    pub id: usize,
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub tokens: Vec<u32>,
    pub temperature: f64,
    pub avg_logprob: f64,
    pub compression_ratio: f64,
    pub no_speech_prob: f64,
}

pub struct WhisperTranscriber {
    model: Arc<m::model::Whisper>,
    tokenizer: Arc<Tokenizer>,
    config: WhisperConfig,
    device: Device,
    mel_filters: Vec<f32>,
    semaphore: Arc<Semaphore>,
}

impl WhisperTranscriber {
    pub async fn new(config: WhisperConfig) -> Result<Self> {
        info!("Initializing Whisper transcriber with model: {}", config.model_id);

        // Set up device
        let device = match &config.device {
            DeviceConfig::Cpu => Device::Cpu,
            DeviceConfig::Cuda(id) => Device::new_cuda(*id)
                .map_err(|e| VideoRssError::Config(format!("CUDA device error: {}", e)))?,
            DeviceConfig::Metal => Device::new_metal(0)
                .map_err(|e| VideoRssError::Config(format!("Metal device error: {}", e)))?,
        };

        // Download model from HuggingFace
        let api = Api::new()
            .map_err(|e| VideoRssError::Config(format!("Failed to create HF API: {}", e)))?;

        let repo = api.repo(Repo::model(config.model_id.clone()));

        // Determine model files based on quantization
        let (model_file, config_file) = if config.quantized {
            ("model.q4_0.gguf", "config.json")
        } else {
            ("model.safetensors", "config.json")
        };

        // Download files
        let model_path = repo
            .get(model_file)
            .await
            .map_err(|e| VideoRssError::Config(format!("Failed to download model: {}", e)))?;

        let config_path = repo
            .get(config_file)
            .await
            .map_err(|e| VideoRssError::Config(format!("Failed to download config: {}", e)))?;

        let tokenizer_path = repo
            .get("tokenizer.json")
            .await
            .map_err(|e| VideoRssError::Config(format!("Failed to download tokenizer: {}", e)))?;

        // Load configuration
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| VideoRssError::Io(e))?;
        let whisper_config: Config = serde_json::from_str(&config_str)
            .map_err(|e| VideoRssError::Config(format!("Failed to parse config: {}", e)))?;

        // Load model weights
        let vb = if config.quantized {
            // Load quantized model
            VarBuilder::from_gguf(&model_path, &device)
                .map_err(|e| VideoRssError::Config(format!("Failed to load GGUF model: {}", e)))?
        } else {
            // Load SafeTensors model
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[model_path], candle_core::DType::F32, &device)
                    .map_err(|e| VideoRssError::Config(format!("Failed to load model: {}", e)))?
            }
        };

        // Create Whisper model
        let model = m::model::Whisper::load(&vb, whisper_config)
            .map_err(|e| VideoRssError::Config(format!("Failed to create model: {}", e)))?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| VideoRssError::Config(format!("Failed to load tokenizer: {}", e)))?;

        // Load mel filters
        let mel_bytes = match config.model_type {
            WhisperModel::Tiny | WhisperModel::Base | WhisperModel::Small => {
                include_bytes!("../assets/mel_filters.npz")
            }
            _ => include_bytes!("../assets/mel_filters_large.npz"),
        };

        let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
        for (i, chunk) in mel_bytes.chunks_exact(4).enumerate() {
            mel_filters[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }

        let semaphore = Arc::new(Semaphore::new(config.max_concurrent));

        info!("Whisper transcriber initialized successfully");

        Ok(Self {
            model: Arc::new(model),
            tokenizer: Arc::new(tokenizer),
            config,
            device,
            mel_filters,
            semaphore,
        })
    }

    pub async fn transcribe_audio(&self, audio_path: &Path) -> Result<TranscriptionResult> {
        let _permit = self.semaphore.acquire().await
            .map_err(|e| VideoRssError::Unknown(format!("Semaphore error: {}", e)))?;

        let start_time = std::time::Instant::now();

        // Load and preprocess audio
        let pcm_data = self.load_and_preprocess_audio(audio_path).await?;
        let audio_duration_ms = (pcm_data.len() as f64 / SAMPLE_RATE as f64 * 1000.0) as u64;

        // Convert to mel spectrogram
        let mel = audio::pcm_to_mel(&self.config.mel_filters, &pcm_data, &self.mel_filters);
        let mel_len = mel.len();
        let mel = Tensor::from_vec(
            mel,
            (1, self.config.mel_filters.len(), mel_len / self.config.mel_filters.len()),
            &self.device,
        ).map_err(|e| VideoRssError::Unknown(format!("Mel tensor error: {}", e)))?;

        info!("Processing audio of duration {}ms", audio_duration_ms);

        // Detect language if not specified
        let language_token = if let Some(lang) = &self.config.language {
            self.token_id(&format!("<|{lang}|>"))
                .ok_or_else(|| VideoRssError::Config(format!("Language {} not found", lang)))?
        } else {
            // Auto-detect language
            self.detect_language(&mel).await?
        };

        // Set up decoder
        let mut dc = self.create_decoder(language_token);

        // Run decoding
        let segments = self.decode_with_fallback(&mel, &mut dc).await?;

        // Combine segments
        let text = segments.iter()
            .map(|s| s.text.clone())
            .collect::<Vec<_>>()
            .join(" ")
            .trim()
            .to_string();

        let processing_time_ms = start_time.elapsed().as_millis() as u64;
        let rtf = processing_time_ms as f32 / audio_duration_ms as f32;

        info!(
            "Transcription completed in {}ms (RTF: {:.2}x) for {}ms audio",
            processing_time_ms, rtf, audio_duration_ms
        );

        Ok(TranscriptionResult {
            text,
            segments,
            language: self.config.language.clone().unwrap_or_else(|| "auto".to_string()),
            processing_time_ms,
            audio_duration_ms,
            rtf,
        })
    }

    async fn load_and_preprocess_audio(&self, path: &Path) -> Result<Vec<f32>> {
        // Load audio file using symphonia
        let file = std::fs::File::open(path)
            .map_err(|e| VideoRssError::Io(e))?;

        let mss = MediaSourceStream::new(Box::new(file), Default::default());
        let hint = Hint::new();
        let format_opts: FormatOptions = Default::default();
        let metadata_opts: MetadataOptions = Default::default();

        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &format_opts, &metadata_opts)
            .map_err(|e| VideoRssError::Unknown(format!("Probe error: {}", e)))?;

        let mut format = probed.format;
        let track = format.default_track()
            .ok_or_else(|| VideoRssError::Unknown("No audio track found".to_string()))?;

        let sample_rate = track.codec_params.sample_rate
            .ok_or_else(|| VideoRssError::Unknown("No sample rate found".to_string()))?;

        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &DecoderOptions::default())
            .map_err(|e| VideoRssError::Unknown(format!("Decoder error: {}", e)))?;

        let mut audio_data = Vec::new();
        let track_id = track.id;

        // Decode audio
        while let Ok(packet) = format.next_packet() {
            if packet.track_id() != track_id {
                continue;
            }

            match decoder.decode(&packet) {
                Ok(decoded) => {
                    match decoded {
                        AudioBufferRef::F32(buf) => {
                            // Convert to mono by averaging channels
                            let num_channels = buf.spec().channels.count();
                            let num_samples = buf.frames();

                            for i in 0..num_samples {
                                let mut sample = 0.0f32;
                                for ch in 0..num_channels {
                                    sample += buf.chan(ch)[i];
                                }
                                audio_data.push(sample / num_channels as f32);
                            }
                        },
                        _ => {
                            warn!("Non-F32 audio format, converting...");
                            // Handle other audio formats
                        }
                    }
                },
                Err(e) => {
                    error!("Decode error: {}", e);
                }
            }
        }

        // Resample to 16kHz if needed
        if sample_rate != SAMPLE_RATE as u32 {
            audio_data = self.resample_audio(audio_data, sample_rate as usize, SAMPLE_RATE)?;
        }

        Ok(audio_data)
    }

    fn resample_audio(&self, input: Vec<f32>, from_rate: usize, to_rate: usize) -> Result<Vec<f32>> {
        if from_rate == to_rate {
            return Ok(input);
        }

        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };

        let mut resampler = SincFixedIn::<f32>::new(
            to_rate as f64 / from_rate as f64,
            2.0,
            params,
            input.len(),
            1,
        ).map_err(|e| VideoRssError::Unknown(format!("Resampler error: {}", e)))?;

        let waves_in = vec![input];
        let mut waves_out = resampler.process(&waves_in, None)
            .map_err(|e| VideoRssError::Unknown(format!("Resample error: {}", e)))?;

        Ok(waves_out.remove(0))
    }

    async fn detect_language(&self, mel: &Tensor) -> Result<u32> {
        // Language detection using the first 30 seconds
        info!("Auto-detecting language...");

        // Run language detection
        let language_logits = self.model.detect_language(mel)
            .map_err(|e| VideoRssError::Unknown(format!("Language detection error: {}", e)))?;

        // Get the most likely language
        let language_token = language_logits.argmax(candle_core::D::Minus1)
            .map_err(|e| VideoRssError::Unknown(format!("Argmax error: {}", e)))?
            .to_scalar::<u32>()
            .map_err(|e| VideoRssError::Unknown(format!("Scalar conversion error: {}", e)))?;

        info!("Detected language token: {}", language_token);
        Ok(language_token)
    }

    fn create_decoder(&self, language_token: u32) -> DecodingContext {
        DecodingContext {
            tokens: vec![
                self.token_id("<|startoftranscript|>").unwrap(),
                language_token,
                match self.config.task {
                    Task::Transcribe => self.token_id("<|transcribe|>").unwrap(),
                    Task::Translate => self.token_id("<|translate|>").unwrap(),
                },
                if self.config.timestamps {
                    self.token_id("<|notimestamps|>").unwrap()
                } else {
                    self.token_id("<|notimestamps|>").unwrap()
                },
            ],
            temperature: self.config.temperature,
            compression_ratio_threshold: 2.4,
            logprob_threshold: -1.0,
            no_speech_threshold: 0.6,
        }
    }

    async fn decode_with_fallback(
        &self,
        mel: &Tensor,
        dc: &mut DecodingContext,
    ) -> Result<Vec<TranscriptionSegment>> {
        let mut segments = Vec::new();
        let mut offset = 0;

        // Process in 30-second chunks
        let chunk_length = 30 * SAMPLE_RATE; // 30 seconds

        loop {
            // Get chunk of mel spectrogram
            let chunk_mel = if offset + chunk_length < mel.dims()[2] {
                mel.narrow(2, offset, chunk_length)
                    .map_err(|e| VideoRssError::Unknown(format!("Narrow error: {}", e)))?
            } else if offset < mel.dims()[2] {
                mel.narrow(2, offset, mel.dims()[2] - offset)
                    .map_err(|e| VideoRssError::Unknown(format!("Narrow error: {}", e)))?
            } else {
                break;
            };

            // Decode chunk
            let segment = self.decode_chunk(&chunk_mel, dc, offset).await?;
            segments.push(segment);

            offset += chunk_length;
        }

        Ok(segments)
    }

    async fn decode_chunk(
        &self,
        mel: &Tensor,
        dc: &DecodingContext,
        offset: usize,
    ) -> Result<TranscriptionSegment> {
        // Run encoder
        let encoder_output = self.model.encoder.forward(mel, true)
            .map_err(|e| VideoRssError::Unknown(format!("Encoder error: {}", e)))?;

        // Run decoder
        let mut tokens = dc.tokens.clone();
        let mut text_tokens = Vec::new();

        for _ in 0..448 {  // Max tokens per chunk
            let tokens_tensor = Tensor::new(tokens.as_slice(), &self.device)
                .map_err(|e| VideoRssError::Unknown(format!("Token tensor error: {}", e)))?;

            let logits = self.model.decoder.forward(&tokens_tensor, &encoder_output, true)
                .map_err(|e| VideoRssError::Unknown(format!("Decoder error: {}", e)))?;

            let next_token = if dc.temperature > 0.0 {
                // Sample with temperature
                self.sample_with_temperature(&logits, dc.temperature)?
            } else {
                // Greedy decoding
                logits.argmax(candle_core::D::Minus1)
                    .map_err(|e| VideoRssError::Unknown(format!("Argmax error: {}", e)))?
                    .to_scalar::<u32>()
                    .map_err(|e| VideoRssError::Unknown(format!("Scalar error: {}", e)))?
            };

            tokens.push(next_token);
            text_tokens.push(next_token);

            // Check for end token
            if next_token == self.token_id("<|endoftext|>").unwrap() {
                break;
            }
        }

        // Decode tokens to text
        let text = self.tokenizer.decode(&text_tokens, true)
            .map_err(|e| VideoRssError::Unknown(format!("Decode error: {}", e)))?;

        Ok(TranscriptionSegment {
            id: offset / (30 * SAMPLE_RATE),
            start: offset as f64 / SAMPLE_RATE as f64,
            end: (offset + text_tokens.len() * 20) as f64 / SAMPLE_RATE as f64,  // Approximate
            text,
            tokens: text_tokens,
            temperature: dc.temperature,
            avg_logprob: 0.0,  // Would need to calculate
            compression_ratio: 1.0,  // Would need to calculate
            no_speech_prob: 0.0,  // Would need to calculate
        })
    }

    fn sample_with_temperature(&self, logits: &Tensor, temperature: f64) -> Result<u32> {
        let logits = (logits / temperature)
            .map_err(|e| VideoRssError::Unknown(format!("Temperature scaling error: {}", e)))?;

        let probs = candle_nn::ops::softmax_last_dim(&logits)
            .map_err(|e| VideoRssError::Unknown(format!("Softmax error: {}", e)))?;

        // Sample from distribution
        let mut rng = StdRng::from_entropy();
        let probs_vec: Vec<f32> = probs.to_vec1()
            .map_err(|e| VideoRssError::Unknown(format!("Vec conversion error: {}", e)))?;

        // Weighted random sampling
        let mut cumsum = 0.0;
        let rand_val: f32 = rand::Rng::gen(&mut rng);

        for (idx, &prob) in probs_vec.iter().enumerate() {
            cumsum += prob;
            if cumsum > rand_val {
                return Ok(idx as u32);
            }
        }

        Ok((probs_vec.len() - 1) as u32)
    }

    fn token_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }
}

struct DecodingContext {
    tokens: Vec<u32>,
    temperature: f64,
    compression_ratio_threshold: f64,
    logprob_threshold: f64,
    no_speech_threshold: f64,
}