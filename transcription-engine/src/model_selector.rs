use crate::{ModelVariant, ModelSize, QuantizationType, LanguageOptimization, TranscriptionOptions};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSelectionCriteria {
    pub detected_language: Option<String>,
    pub audio_duration_seconds: f32,
    pub audio_quality: AudioQuality,
    pub content_type: ContentType,
    pub hardware_tier: HardwareTier,
    pub latency_requirement: LatencyRequirement,
    pub accuracy_requirement: AccuracyRequirement,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AudioQuality {
    High,      // Clean audio, minimal noise
    Medium,    // Some background noise
    Low,       // Heavy noise, poor quality
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ContentType {
    Speech,         // Regular speech/conversation
    Technical,      // Technical content with jargon
    Medical,        // Medical terminology
    Legal,          // Legal terminology
    Educational,    // Lectures, tutorials
    Entertainment,  // Movies, shows, podcasts
    News,          // News broadcasts
    Music,         // Contains music/singing
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum HardwareTier {
    Ultra,    // High-end GPU with Tensor Cores
    High,     // Good GPU
    Medium,   // Entry-level GPU or strong CPU
    Low,      // CPU-only, limited resources
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LatencyRequirement {
    RealTime,      // < 100ms latency
    NearRealTime,  // < 500ms latency
    Fast,          // < 2s latency
    Normal,        // Best quality/speed balance
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AccuracyRequirement {
    Maximum,   // Highest possible accuracy
    High,      // >95% accuracy target
    Normal,    // >90% accuracy target
    Fast,      // Speed prioritized over accuracy
}

pub struct ModelSelector {
    model_registry: HashMap<String, ModelProfile>,
    language_models: HashMap<String, Vec<ModelVariant>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelProfile {
    variant: ModelVariant,
    supported_languages: Vec<String>,
    optimal_content_types: Vec<ContentType>,
    speed_multiplier: f32,  // Relative to baseline
    accuracy_score: f32,     // 0-100
    memory_requirement_mb: usize,
    min_hardware_tier: HardwareTier,
}

impl ModelSelector {
    pub fn new() -> Self {
        let mut selector = Self {
            model_registry: HashMap::new(),
            language_models: HashMap::new(),
        };

        selector.initialize_model_profiles();
        selector
    }

    fn initialize_model_profiles(&mut self) {
        // WhisperTurbo - Fastest for English/European languages
        self.model_registry.insert(
            "whisper_turbo".to_string(),
            ModelProfile {
                variant: ModelVariant::WhisperTurbo,
                supported_languages: vec![
                    "en".to_string(), "es".to_string(), "fr".to_string(),
                    "de".to_string(), "it".to_string(), "pt".to_string(),
                ],
                optimal_content_types: vec![
                    ContentType::Speech,
                    ContentType::News,
                    ContentType::Entertainment,
                ],
                speed_multiplier: 10.0,  // 10x faster than baseline
                accuracy_score: 92.0,
                memory_requirement_mb: 1500,
                min_hardware_tier: HardwareTier::Medium,
            },
        );

        // FasterWhisper - Good balance for most languages
        self.model_registry.insert(
            "faster_whisper".to_string(),
            ModelProfile {
                variant: ModelVariant::FasterWhisper,
                supported_languages: vec!["*".to_string()],  // All languages
                optimal_content_types: vec![
                    ContentType::Speech,
                    ContentType::Educational,
                    ContentType::Technical,
                ],
                speed_multiplier: 8.0,
                accuracy_score: 94.0,
                memory_requirement_mb: 2000,
                min_hardware_tier: HardwareTier::Medium,
            },
        );

        // Paraformer - Optimized for Chinese
        self.model_registry.insert(
            "paraformer".to_string(),
            ModelProfile {
                variant: ModelVariant::Paraformer,
                supported_languages: vec!["zh".to_string(), "zh-CN".to_string(), "zh-TW".to_string()],
                optimal_content_types: vec![
                    ContentType::Speech,
                    ContentType::News,
                    ContentType::Educational,
                    ContentType::Technical,
                ],
                speed_multiplier: 12.0,  // Very fast for Chinese
                accuracy_score: 96.0,    // Excellent Chinese accuracy
                memory_requirement_mb: 1800,
                min_hardware_tier: HardwareTier::Medium,
            },
        );

        // FunASR - Chinese with dialect support
        self.model_registry.insert(
            "funasr".to_string(),
            ModelProfile {
                variant: ModelVariant::FunASR,
                supported_languages: vec![
                    "zh".to_string(), "zh-CN".to_string(), "zh-TW".to_string(),
                    "yue".to_string(),  // Cantonese
                ],
                optimal_content_types: vec![
                    ContentType::Speech,
                    ContentType::Entertainment,
                    ContentType::News,
                ],
                speed_multiplier: 9.0,
                accuracy_score: 95.0,
                memory_requirement_mb: 2200,
                min_hardware_tier: HardwareTier::Medium,
            },
        );

        // Qwen3-ASR - Advanced Chinese model
        self.model_registry.insert(
            "qwen3_asr".to_string(),
            ModelProfile {
                variant: ModelVariant::Qwen3ASR,
                supported_languages: vec!["zh".to_string(), "en".to_string()],
                optimal_content_types: vec![
                    ContentType::Technical,
                    ContentType::Medical,
                    ContentType::Legal,
                    ContentType::Educational,
                ],
                speed_multiplier: 6.0,
                accuracy_score: 97.0,  // Highest accuracy for Chinese
                memory_requirement_mb: 4000,
                min_hardware_tier: HardwareTier::High,
            },
        );

        // WhisperStandard - Baseline, maximum compatibility
        self.model_registry.insert(
            "whisper_standard".to_string(),
            ModelProfile {
                variant: ModelVariant::WhisperStandard,
                supported_languages: vec!["*".to_string()],
                optimal_content_types: vec![
                    ContentType::Speech,
                    ContentType::Music,
                    ContentType::Entertainment,
                ],
                speed_multiplier: 1.0,  // Baseline speed
                accuracy_score: 90.0,
                memory_requirement_mb: 3000,
                min_hardware_tier: HardwareTier::Low,
            },
        );

        // Build language-specific model mappings
        self.language_models.insert("zh".to_string(), vec![
            ModelVariant::Paraformer,
            ModelVariant::FunASR,
            ModelVariant::Qwen3ASR,
        ]);

        self.language_models.insert("en".to_string(), vec![
            ModelVariant::WhisperTurbo,
            ModelVariant::FasterWhisper,
        ]);
    }

    pub fn select_optimal_model(&self, criteria: &ModelSelectionCriteria) -> TranscriptionOptions {
        info!("Selecting optimal model for criteria: {:?}", criteria);

        // First, filter by language if detected
        let language_candidates = if let Some(ref lang) = criteria.detected_language {
            self.get_language_specific_models(lang)
        } else {
            // If no language detected, use all models
            self.model_registry.values()
                .map(|p| p.variant)
                .collect()
        };

        // Filter by hardware capability
        let hardware_candidates: Vec<_> = language_candidates
            .iter()
            .filter(|variant| {
                if let Some(profile) = self.get_model_profile(variant) {
                    self.meets_hardware_requirements(profile, criteria.hardware_tier)
                } else {
                    false
                }
            })
            .cloned()
            .collect();

        if hardware_candidates.is_empty() {
            warn!("No models meet hardware requirements, falling back to WhisperStandard");
            return self.create_fallback_options();
        }

        // Score each candidate based on criteria
        let mut best_model = ModelVariant::WhisperStandard;
        let mut best_score = 0.0;

        for variant in hardware_candidates {
            let score = self.score_model(&variant, criteria);
            debug!("Model {:?} scored: {:.2}", variant, score);

            if score > best_score {
                best_score = score;
                best_model = variant;
            }
        }

        info!("Selected model: {:?} with score: {:.2}", best_model, best_score);

        // Create optimized options for the selected model
        self.create_options_for_model(best_model, criteria)
    }

    fn get_language_specific_models(&self, language_code: &str) -> Vec<ModelVariant> {
        // Check for exact language match
        if let Some(models) = self.language_models.get(language_code) {
            return models.clone();
        }

        // Check for language family (e.g., "zh-CN" -> "zh")
        let lang_family = language_code.split('-').next().unwrap_or(language_code);
        if let Some(models) = self.language_models.get(lang_family) {
            return models.clone();
        }

        // Check which models support this language
        let mut supported_models = Vec::new();
        for profile in self.model_registry.values() {
            if profile.supported_languages.contains(&"*".to_string()) ||
               profile.supported_languages.contains(&language_code.to_string()) ||
               profile.supported_languages.contains(&lang_family.to_string()) {
                supported_models.push(profile.variant);
            }
        }

        supported_models
    }

    fn get_model_profile(&self, variant: &ModelVariant) -> Option<&ModelProfile> {
        self.model_registry.values()
            .find(|p| p.variant == *variant)
    }

    fn meets_hardware_requirements(&self, profile: &ModelProfile, hardware_tier: HardwareTier) -> bool {
        match (profile.min_hardware_tier, hardware_tier) {
            (HardwareTier::Ultra, HardwareTier::Ultra) => true,
            (HardwareTier::High, HardwareTier::High) | (HardwareTier::High, HardwareTier::Ultra) => true,
            (HardwareTier::Medium, HardwareTier::Medium) |
            (HardwareTier::Medium, HardwareTier::High) |
            (HardwareTier::Medium, HardwareTier::Ultra) => true,
            (HardwareTier::Low, _) => true,
            _ => false,
        }
    }

    fn score_model(&self, variant: &ModelVariant, criteria: &ModelSelectionCriteria) -> f32 {
        let profile = match self.get_model_profile(variant) {
            Some(p) => p,
            None => return 0.0,
        };

        let mut score = 0.0;

        // Language support score (40% weight)
        let language_score = if let Some(ref lang) = criteria.detected_language {
            if profile.supported_languages.contains(&lang.to_string()) ||
               profile.supported_languages.contains(&"*".to_string()) {
                1.0
            } else {
                0.0
            }
        } else {
            0.5  // No language preference
        };
        score += language_score * 40.0;

        // Content type optimization (20% weight)
        let content_score = if profile.optimal_content_types.contains(&criteria.content_type) {
            1.0
        } else {
            0.5
        };
        score += content_score * 20.0;

        // Speed vs accuracy trade-off (30% weight)
        let performance_score = match criteria.latency_requirement {
            LatencyRequirement::RealTime => {
                // Prioritize speed
                (profile.speed_multiplier / 12.0).min(1.0)
            }
            LatencyRequirement::NearRealTime => {
                // Balance speed and accuracy
                let speed_component = (profile.speed_multiplier / 10.0).min(1.0) * 0.6;
                let accuracy_component = (profile.accuracy_score / 100.0) * 0.4;
                speed_component + accuracy_component
            }
            LatencyRequirement::Fast => {
                // Slight speed preference
                let speed_component = (profile.speed_multiplier / 8.0).min(1.0) * 0.4;
                let accuracy_component = (profile.accuracy_score / 100.0) * 0.6;
                speed_component + accuracy_component
            }
            LatencyRequirement::Normal => {
                // Prioritize accuracy
                profile.accuracy_score / 100.0
            }
        };
        score += performance_score * 30.0;

        // Audio quality consideration (10% weight)
        let quality_score = match criteria.audio_quality {
            AudioQuality::High => 1.0,  // Any model works
            AudioQuality::Medium => {
                // Prefer models with better noise handling
                if matches!(variant, ModelVariant::FasterWhisper | ModelVariant::Qwen3ASR) {
                    1.0
                } else {
                    0.7
                }
            }
            AudioQuality::Low => {
                // Need robust models
                if matches!(variant, ModelVariant::Qwen3ASR | ModelVariant::WhisperStandard) {
                    1.0
                } else {
                    0.5
                }
            }
        };
        score += quality_score * 10.0;

        score
    }

    fn create_options_for_model(&self, variant: ModelVariant, criteria: &ModelSelectionCriteria) -> TranscriptionOptions {
        let mut options = TranscriptionOptions::default();
        options.model_variant = variant;

        // Select model size based on hardware and requirements
        options.model_size = self.select_model_size(criteria.hardware_tier, criteria.accuracy_requirement);

        // Select quantization based on hardware
        options.quantization = self.select_quantization(criteria.hardware_tier, variant);

        // Language settings
        options.language = if let Some(ref lang) = criteria.detected_language {
            if lang.starts_with("zh") {
                LanguageOptimization::Chinese
            } else if lang == "en" {
                LanguageOptimization::English
            } else {
                LanguageOptimization::Multilingual
            }
        } else {
            LanguageOptimization::AutoDetect
        };

        // Beam search settings based on requirements
        options.beam_size = match criteria.accuracy_requirement {
            AccuracyRequirement::Maximum => 10,
            AccuracyRequirement::High => 5,
            AccuracyRequirement::Normal => 3,
            AccuracyRequirement::Fast => 1,
        };

        // VAD threshold based on audio quality
        options.vad_threshold = match criteria.audio_quality {
            AudioQuality::High => 0.3,
            AudioQuality::Medium => 0.5,
            AudioQuality::Low => 0.7,
        };

        // Chunk length optimization
        options.chunk_length_ms = match criteria.latency_requirement {
            LatencyRequirement::RealTime => 1000,      // 1 second chunks
            LatencyRequirement::NearRealTime => 3000,  // 3 second chunks
            LatencyRequirement::Fast => 5000,          // 5 second chunks
            LatencyRequirement::Normal => 10000,       // 10 second chunks
        };

        // Enable timestamps for certain content types
        options.enable_timestamps = matches!(
            criteria.content_type,
            ContentType::Educational | ContentType::Legal | ContentType::Medical
        );

        // Model-specific path
        options.model_path = self.get_model_path(variant, options.model_size);

        info!("Created options for {:?}: size={:?}, quant={:?}, beam={}, chunk={}ms",
            variant, options.model_size, options.quantization,
            options.beam_size, options.chunk_length_ms);

        options
    }

    fn select_model_size(&self, hardware_tier: HardwareTier, accuracy_req: AccuracyRequirement) -> ModelSize {
        match (hardware_tier, accuracy_req) {
            (HardwareTier::Ultra, AccuracyRequirement::Maximum) => ModelSize::Ultra,
            (HardwareTier::Ultra, _) => ModelSize::Large,
            (HardwareTier::High, AccuracyRequirement::Maximum) => ModelSize::Large,
            (HardwareTier::High, AccuracyRequirement::High) => ModelSize::Large,
            (HardwareTier::High, _) => ModelSize::Medium,
            (HardwareTier::Medium, AccuracyRequirement::Fast) => ModelSize::Small,
            (HardwareTier::Medium, _) => ModelSize::Medium,
            (HardwareTier::Low, AccuracyRequirement::Fast) => ModelSize::Tiny,
            (HardwareTier::Low, _) => ModelSize::Base,
        }
    }

    fn select_quantization(&self, hardware_tier: HardwareTier, variant: ModelVariant) -> QuantizationType {
        match hardware_tier {
            HardwareTier::Ultra => {
                // Can use FP16 or FP8 for best speed with minimal quality loss
                if matches!(variant, ModelVariant::WhisperTurbo) {
                    QuantizationType::FP8  // Aggressive for turbo model
                } else {
                    QuantizationType::FP16
                }
            }
            HardwareTier::High => QuantizationType::INT8,
            HardwareTier::Medium => QuantizationType::INT8,
            HardwareTier::Low => {
                // Most aggressive quantization for low-end hardware
                if matches!(variant, ModelVariant::WhisperTurbo | ModelVariant::Paraformer) {
                    QuantizationType::INT4
                } else {
                    QuantizationType::INT8
                }
            }
        }
    }

    fn get_model_path(&self, variant: ModelVariant, size: ModelSize) -> String {
        let variant_name = match variant {
            ModelVariant::WhisperTurbo => "whisper-turbo",
            ModelVariant::FasterWhisper => "faster-whisper",
            ModelVariant::WhisperStandard => "whisper",
            ModelVariant::Paraformer => "paraformer",
            ModelVariant::FunASR => "funasr",
            ModelVariant::Qwen3ASR => "qwen3-asr",
        };

        let size_name = match size {
            ModelSize::Tiny => "tiny",
            ModelSize::Base => "base",
            ModelSize::Small => "small",
            ModelSize::Medium => "medium",
            ModelSize::Large => "large",
            ModelSize::Turbo => "turbo",
            ModelSize::Ultra => "ultra",
        };

        format!("/models/{}/{}", variant_name, size_name)
    }

    fn create_fallback_options(&self) -> TranscriptionOptions {
        let mut options = TranscriptionOptions::default();
        options.model_variant = ModelVariant::WhisperStandard;
        options.model_size = ModelSize::Base;
        options.quantization = QuantizationType::INT8;
        options.beam_size = 1;
        options.chunk_length_ms = 10000;
        options
    }

    pub fn detect_language_from_audio(&self, audio_features: &[f32]) -> Option<String> {
        // Simple language detection based on audio features
        // In production, this would use a dedicated language ID model

        // Placeholder implementation
        None
    }

    pub fn detect_content_type(&self, audio_features: &[f32], initial_transcript: &str) -> ContentType {
        // Analyze audio characteristics and initial transcript to determine content type

        // Check for technical keywords
        let technical_keywords = ["api", "function", "algorithm", "database", "code", "software"];
        let medical_keywords = ["patient", "diagnosis", "treatment", "medication", "symptom"];
        let legal_keywords = ["court", "law", "legal", "contract", "attorney", "clause"];

        let lower_transcript = initial_transcript.to_lowercase();

        if technical_keywords.iter().any(|k| lower_transcript.contains(k)) {
            return ContentType::Technical;
        }

        if medical_keywords.iter().any(|k| lower_transcript.contains(k)) {
            return ContentType::Medical;
        }

        if legal_keywords.iter().any(|k| lower_transcript.contains(k)) {
            return ContentType::Legal;
        }

        // Default to speech
        ContentType::Speech
    }

    pub fn estimate_audio_quality(&self, audio_data: &[f32]) -> AudioQuality {
        // Analyze audio signal for noise level estimation

        // Calculate signal-to-noise ratio (simplified)
        let mean: f32 = audio_data.iter().sum::<f32>() / audio_data.len() as f32;
        let variance: f32 = audio_data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / audio_data.len() as f32;

        let snr_estimate = variance.sqrt();

        if snr_estimate > 0.5 {
            AudioQuality::High
        } else if snr_estimate > 0.2 {
            AudioQuality::Medium
        } else {
            AudioQuality::Low
        }
    }
}

impl Default for ModelSelector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chinese_model_selection() {
        let selector = ModelSelector::new();

        let criteria = ModelSelectionCriteria {
            detected_language: Some("zh-CN".to_string()),
            audio_duration_seconds: 60.0,
            audio_quality: AudioQuality::High,
            content_type: ContentType::Speech,
            hardware_tier: HardwareTier::High,
            latency_requirement: LatencyRequirement::Fast,
            accuracy_requirement: AccuracyRequirement::High,
        };

        let options = selector.select_optimal_model(&criteria);

        // Should select Paraformer for Chinese
        assert!(matches!(options.model_variant, ModelVariant::Paraformer));
    }

    #[test]
    fn test_english_realtime_selection() {
        let selector = ModelSelector::new();

        let criteria = ModelSelectionCriteria {
            detected_language: Some("en".to_string()),
            audio_duration_seconds: 10.0,
            audio_quality: AudioQuality::High,
            content_type: ContentType::Speech,
            hardware_tier: HardwareTier::Ultra,
            latency_requirement: LatencyRequirement::RealTime,
            accuracy_requirement: AccuracyRequirement::Normal,
        };

        let options = selector.select_optimal_model(&criteria);

        // Should select WhisperTurbo for English real-time
        assert!(matches!(options.model_variant, ModelVariant::WhisperTurbo));
        assert_eq!(options.chunk_length_ms, 1000);  // Short chunks for low latency
    }

    #[test]
    fn test_low_hardware_fallback() {
        let selector = ModelSelector::new();

        let criteria = ModelSelectionCriteria {
            detected_language: None,
            audio_duration_seconds: 300.0,
            audio_quality: AudioQuality::Low,
            content_type: ContentType::Speech,
            hardware_tier: HardwareTier::Low,
            latency_requirement: LatencyRequirement::Normal,
            accuracy_requirement: AccuracyRequirement::Normal,
        };

        let options = selector.select_optimal_model(&criteria);

        // Should use conservative settings for low hardware
        assert!(matches!(options.model_size, ModelSize::Base | ModelSize::Tiny));
        assert!(matches!(options.quantization, QuantizationType::INT8 | QuantizationType::INT4));
    }
}