use std::env;
use std::path::PathBuf;

use anyhow::{anyhow, Result};

use crate::ffi::BackendLibrary;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccelPreference {
    Auto,
    Mps,
    CoreMl,
    Cuda,
    Rocm,
    OneApi,
    Cpu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccelBackendKind {
    Mps,
    CoreMl,
    Cuda,
    Rocm,
    OneApi,
    Cpu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendPurpose {
    Transcription,
    Summarization,
}

#[derive(Clone, Debug)]
pub struct AccelConfig {
    pub preference: AccelPreference,
    pub allow_cpu: bool,
    pub lib_dir: Option<PathBuf>,
    pub lib_name: Option<String>,
    pub device: Option<String>,
    pub transcribe: AccelOverride,
    pub summarize: AccelOverride,
}

#[derive(Clone, Debug, Default)]
pub struct AccelOverride {
    pub preference: Option<AccelPreference>,
    pub lib_dir: Option<PathBuf>,
    pub lib_name: Option<String>,
    pub device: Option<String>,
}

impl AccelConfig {
    pub fn from_env() -> Self {
        let preference =
            parse_preference(&env::var("VRA_ACCEL").unwrap_or_else(|_| "auto".to_string()));

        let allow_cpu = env::var("VRA_ALLOW_CPU")
            .map(|val| val == "1" || val.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        let lib_dir = env::var("VRA_ACCEL_LIB_DIR").ok().map(PathBuf::from);
        let lib_name = env::var("VRA_ACCEL_LIB_NAME").ok();
        let device = env::var("VRA_ACCEL_DEVICE").ok();

        let transcribe = AccelOverride {
            preference: env::var("VRA_TRANSCRIBE_ACCEL")
                .ok()
                .map(|value| parse_preference(&value)),
            lib_dir: env::var("VRA_TRANSCRIBE_LIB_DIR").ok().map(PathBuf::from),
            lib_name: env::var("VRA_TRANSCRIBE_LIB_NAME").ok(),
            device: env::var("VRA_TRANSCRIBE_DEVICE").ok(),
        };

        let summarize = AccelOverride {
            preference: env::var("VRA_SUMMARIZE_ACCEL")
                .ok()
                .map(|value| parse_preference(&value)),
            lib_dir: env::var("VRA_SUMMARIZE_LIB_DIR").ok().map(PathBuf::from),
            lib_name: env::var("VRA_SUMMARIZE_LIB_NAME").ok(),
            device: env::var("VRA_SUMMARIZE_DEVICE").ok(),
        };

        Self {
            preference,
            allow_cpu,
            lib_dir,
            lib_name,
            device,
            transcribe,
            summarize,
        }
    }

    fn preference_for(&self, purpose: BackendPurpose) -> AccelPreference {
        match purpose {
            BackendPurpose::Transcription => self.transcribe.preference.unwrap_or(self.preference),
            BackendPurpose::Summarization => self.summarize.preference.unwrap_or(self.preference),
        }
    }

    fn lib_dir_for(&self, purpose: BackendPurpose) -> Option<PathBuf> {
        match purpose {
            BackendPurpose::Transcription => self
                .transcribe
                .lib_dir
                .clone()
                .or_else(|| self.lib_dir.clone()),
            BackendPurpose::Summarization => self
                .summarize
                .lib_dir
                .clone()
                .or_else(|| self.lib_dir.clone()),
        }
    }

    fn lib_name_for(&self, purpose: BackendPurpose) -> Option<String> {
        match purpose {
            BackendPurpose::Transcription => self
                .transcribe
                .lib_name
                .clone()
                .or_else(|| self.lib_name.clone()),
            BackendPurpose::Summarization => self
                .summarize
                .lib_name
                .clone()
                .or_else(|| self.lib_name.clone()),
        }
    }

    fn device_for(&self, purpose: BackendPurpose) -> Option<String> {
        match purpose {
            BackendPurpose::Transcription => self
                .transcribe
                .device
                .clone()
                .or_else(|| self.device.clone()),
            BackendPurpose::Summarization => self
                .summarize
                .device
                .clone()
                .or_else(|| self.device.clone()),
        }
    }
}

pub struct AccelBackend {
    pub kind: AccelBackendKind,
    pub library: BackendLibrary,
}

impl AccelBackend {
    pub fn load_for(
        purpose: BackendPurpose,
        accel: &AccelConfig,
        extra_config: serde_json::Value,
    ) -> Result<Self> {
        let candidates = backend_candidates(accel.preference_for(purpose), accel.allow_cpu);
        let mut last_error = None;
        let device = accel.device_for(purpose);
        let lib_dir = accel.lib_dir_for(purpose);
        let lib_name = accel.lib_name_for(purpose);

        for kind in candidates {
            let config_json = serde_json::json!({
                "backend": format_backend(kind),
                "purpose": format_purpose(purpose),
                "device": device,
                "extra": extra_config,
            });

            match BackendLibrary::load(kind, lib_dir.clone(), lib_name.clone(), &config_json) {
                Ok(library) => {
                    return Ok(Self { kind, library });
                }
                Err(err) => {
                    last_error = Some(err);
                }
            }
        }

        if let Some(err) = last_error {
            return Err(err);
        }

        Err(anyhow!("No accelerator backend available"))
    }
}

fn backend_candidates(preference: AccelPreference, allow_cpu: bool) -> Vec<AccelBackendKind> {
    if preference != AccelPreference::Auto {
        return vec![match preference {
            AccelPreference::Mps => AccelBackendKind::Mps,
            AccelPreference::CoreMl => AccelBackendKind::CoreMl,
            AccelPreference::Cuda => AccelBackendKind::Cuda,
            AccelPreference::Rocm => AccelBackendKind::Rocm,
            AccelPreference::OneApi => AccelBackendKind::OneApi,
            AccelPreference::Cpu => AccelBackendKind::Cpu,
            AccelPreference::Auto => AccelBackendKind::Cpu,
        }];
    }

    let mut ordered = if cfg!(target_os = "macos") {
        vec![AccelBackendKind::CoreMl, AccelBackendKind::Mps]
    } else if cfg!(target_os = "windows") {
        vec![
            AccelBackendKind::Cuda,
            AccelBackendKind::Rocm,
            AccelBackendKind::OneApi,
        ]
    } else {
        vec![
            AccelBackendKind::Cuda,
            AccelBackendKind::Rocm,
            AccelBackendKind::OneApi,
        ]
    };

    if allow_cpu {
        ordered.push(AccelBackendKind::Cpu);
    }

    ordered
}

fn format_backend(kind: AccelBackendKind) -> &'static str {
    match kind {
        AccelBackendKind::Mps => "mps",
        AccelBackendKind::CoreMl => "coreml",
        AccelBackendKind::Cuda => "cuda",
        AccelBackendKind::Rocm => "rocm",
        AccelBackendKind::OneApi => "oneapi",
        AccelBackendKind::Cpu => "cpu",
    }
}

fn format_purpose(purpose: BackendPurpose) -> &'static str {
    match purpose {
        BackendPurpose::Transcription => "transcription",
        BackendPurpose::Summarization => "summarization",
    }
}

fn parse_preference(value: &str) -> AccelPreference {
    match value.to_lowercase().as_str() {
        "mps" => AccelPreference::Mps,
        "coreml" => AccelPreference::CoreMl,
        "cuda" => AccelPreference::Cuda,
        "rocm" => AccelPreference::Rocm,
        "oneapi" => AccelPreference::OneApi,
        "cpu" => AccelPreference::Cpu,
        _ => AccelPreference::Auto,
    }
}
