use std::env;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use crate::accel::{AccelBackend, AccelConfig, BackendPurpose};

#[derive(Clone, Debug)]
pub struct TranscriptionConfig {
    pub model_path: String,
    pub language: Option<String>,
    pub beam_size: u32,
    pub temperature: f32,
}

impl TranscriptionConfig {
    pub fn from_env() -> Result<Self> {
        let model_path = env::var("VRA_TRANSCRIBE_MODEL_PATH")
            .map_err(|_| anyhow!("VRA_TRANSCRIBE_MODEL_PATH must be set"))?;
        let language = env::var("VRA_TRANSCRIBE_LANGUAGE").ok();
        let beam_size = env::var("VRA_TRANSCRIBE_BEAM_SIZE")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(5);
        let temperature = env::var("VRA_TRANSCRIBE_TEMPERATURE")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.0);

        Ok(Self {
            model_path,
            language,
            beam_size,
            temperature,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TranscriptionResult {
    pub text: String,
    pub language: String,
    pub confidence: f32,
}

pub struct TranscriptionEngine {
    backend: AccelBackend,
    config: TranscriptionConfig,
}

impl TranscriptionEngine {
    pub fn new(accel: &AccelConfig) -> Result<Self> {
        let config = TranscriptionConfig::from_env()?;
        let extra = serde_json::json!({
            "model_path": config.model_path,
            "language": config.language,
            "beam_size": config.beam_size,
            "temperature": config.temperature,
        });
        let backend = AccelBackend::load_for(BackendPurpose::Transcription, accel, extra)?;

        Ok(Self { backend, config })
    }

    pub fn transcribe_audio(&self, audio_path: &str) -> Result<TranscriptionResult> {
        let output = self.backend.library.transcribe(audio_path)?;
        let value: serde_json::Value = serde_json::from_str(&output)
            .map_err(|err| anyhow!("Invalid transcription output JSON: {}", err))?;

        let text = value
            .get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Transcription output missing 'text'"))?
            .to_string();
        let language = value
            .get("language")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let confidence = value
            .get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;

        Ok(TranscriptionResult {
            text,
            language,
            confidence,
        })
    }
}
