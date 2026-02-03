use std::env;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use crate::accel::{AccelBackend, AccelConfig, BackendPurpose};

#[derive(Clone, Debug)]
pub struct SummarizationConfig {
    pub model_path: String,
    pub max_tokens: u32,
    pub temperature: f32,
}

impl SummarizationConfig {
    pub fn from_env() -> Result<Self> {
        let model_path = env::var("VRA_SUMMARIZE_MODEL_PATH")
            .map_err(|_| anyhow!("VRA_SUMMARIZE_MODEL_PATH must be set"))?;
        let max_tokens = env::var("VRA_SUMMARIZE_MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(256);
        let temperature = env::var("VRA_SUMMARIZE_TEMPERATURE")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.3);

        Ok(Self {
            model_path,
            max_tokens,
            temperature,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SummaryResult {
    pub summary: String,
    pub key_points: Vec<String>,
}

pub struct SummarizationEngine {
    backend: AccelBackend,
    config: SummarizationConfig,
}

impl SummarizationEngine {
    pub fn new(accel: &AccelConfig) -> Result<Self> {
        let config = SummarizationConfig::from_env()?;
        let extra = serde_json::json!({
            "model_path": config.model_path,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
        });
        let backend = AccelBackend::load_for(BackendPurpose::Summarization, accel, extra)?;

        Ok(Self { backend, config })
    }

    pub fn summarize(&self, text: &str) -> Result<SummaryResult> {
        let output = self.backend.library.summarize(text)?;
        let value: serde_json::Value = serde_json::from_str(&output)
            .map_err(|err| anyhow!("Invalid summary output JSON: {}", err))?;

        let summary = value
            .get("summary")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Summary output missing 'summary'"))?
            .to_string();
        let key_points = value
            .get("key_points")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|item| item.as_str().map(|s| s.to_string()))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        Ok(SummaryResult {
            summary,
            key_points,
        })
    }
}
