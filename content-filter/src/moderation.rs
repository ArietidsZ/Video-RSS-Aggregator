use anyhow::{Context, Result};
use dashmap::DashMap;
use ndarray::{Array1, Array2};
use ort::{Environment, Session, SessionBuilder, Value};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModerationResult {
    pub content_id: String,
    pub safe: bool,
    pub categories: ModerationCategories,
    pub confidence_scores: HashMap<String, f32>,
    pub flags: Vec<ModerationFlag>,
    pub action: ModerationAction,
    pub explanation: Option<String>,
    pub model_version: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModerationCategories {
    pub violence: CategoryResult,
    pub adult_content: CategoryResult,
    pub hate_speech: CategoryResult,
    pub self_harm: CategoryResult,
    pub illegal_content: CategoryResult,
    pub harassment: CategoryResult,
    pub misinformation: CategoryResult,
    pub spam: CategoryResult,
    pub copyright: CategoryResult,
    pub privacy_violation: CategoryResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryResult {
    pub detected: bool,
    pub score: f32,
    pub severity: SeverityLevel,
    pub subcategories: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SeverityLevel {
    None,
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModerationFlag {
    pub category: String,
    pub severity: SeverityLevel,
    pub confidence: f32,
    pub details: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModerationAction {
    Allow,
    Review,
    Blur,
    Restrict,
    Block,
    Report,
}

#[derive(Debug, Clone)]
pub struct ContentModerator {
    text_model: Arc<Session>,
    image_model: Arc<Session>,
    video_model: Arc<Session>,
    audio_model: Arc<Session>,

    thresholds: ModerationThresholds,
    rules: Arc<RwLock<Vec<ModerationRule>>>,

    cache: Arc<DashMap<String, ModerationResult>>,
    statistics: Arc<RwLock<ModerationStatistics>>,

    environment: Arc<Environment>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModerationThresholds {
    pub violence: f32,
    pub adult_content: f32,
    pub hate_speech: f32,
    pub self_harm: f32,
    pub illegal_content: f32,
    pub harassment: f32,
    pub misinformation: f32,
    pub spam: f32,
    pub copyright: f32,
    pub privacy_violation: f32,

    pub auto_block_threshold: f32,
    pub auto_review_threshold: f32,
    pub blur_threshold: f32,
}

impl Default for ModerationThresholds {
    fn default() -> Self {
        Self {
            violence: 0.7,
            adult_content: 0.8,
            hate_speech: 0.6,
            self_harm: 0.5,
            illegal_content: 0.4,
            harassment: 0.6,
            misinformation: 0.7,
            spam: 0.8,
            copyright: 0.9,
            privacy_violation: 0.6,

            auto_block_threshold: 0.9,
            auto_review_threshold: 0.5,
            blur_threshold: 0.7,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModerationRule {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    pub pattern: Option<String>,
    pub keywords: Vec<String>,
    pub category: String,
    pub action: ModerationAction,
    pub severity: SeverityLevel,
    pub enabled: bool,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ModerationStatistics {
    pub total_moderated: u64,
    pub blocked_count: u64,
    pub reviewed_count: u64,
    pub allowed_count: u64,
    pub false_positives: u64,
    pub false_negatives: u64,
    pub category_counts: HashMap<String, u64>,
    pub average_processing_time_ms: f64,
}

impl ContentModerator {
    pub async fn new(models_path: &Path) -> Result<Self> {
        let environment = Arc::new(
            Environment::builder()
                .with_name("content_moderator")
                .with_log_level(ort::LoggingLevel::Warning)
                .build()?,
        );

        // Load ML models for different content types
        let text_model = Arc::new(
            SessionBuilder::new(&environment)?
                .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
                .with_model_from_file(models_path.join("text_moderation.onnx"))?
        );

        let image_model = Arc::new(
            SessionBuilder::new(&environment)?
                .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
                .with_model_from_file(models_path.join("image_moderation.onnx"))?
        );

        let video_model = Arc::new(
            SessionBuilder::new(&environment)?
                .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
                .with_model_from_file(models_path.join("video_moderation.onnx"))?
        );

        let audio_model = Arc::new(
            SessionBuilder::new(&environment)?
                .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
                .with_model_from_file(models_path.join("audio_moderation.onnx"))?
        );

        Ok(Self {
            text_model,
            image_model,
            video_model,
            audio_model,
            thresholds: ModerationThresholds::default(),
            rules: Arc::new(RwLock::new(Self::load_default_rules())),
            cache: Arc::new(DashMap::new()),
            statistics: Arc::new(RwLock::new(ModerationStatistics::default())),
            environment,
        })
    }

    pub async fn moderate_text(&self, text: &str) -> Result<ModerationResult> {
        let cache_key = format!("text:{}", blake3::hash(text.as_bytes()));

        // Check cache
        if let Some(cached) = self.cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        let start = std::time::Instant::now();

        // Tokenize and prepare input
        let tokens = self.tokenize_text(text)?;
        let input = Array2::from_shape_vec((1, tokens.len()), tokens)?;
        let input_value = Value::from_array(self.environment.clone(), &input)?;

        // Run inference
        let outputs = self.text_model.run(vec![input_value])?;
        let predictions = outputs[0].try_extract::<f32>()?.view().to_owned();

        // Parse predictions
        let categories = self.parse_text_predictions(&predictions);
        let result = self.generate_moderation_result(
            &cache_key,
            categories,
            "text_model_v1.0",
        );

        // Update statistics
        self.update_statistics(&result, start.elapsed()).await;

        // Cache result
        self.cache.insert(cache_key, result.clone());

        Ok(result)
    }

    pub async fn moderate_image(&self, image_data: &[u8]) -> Result<ModerationResult> {
        let cache_key = format!("image:{}", blake3::hash(image_data));

        if let Some(cached) = self.cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        let start = std::time::Instant::now();

        // Decode and preprocess image
        let img = image::load_from_memory(image_data)?;
        let preprocessed = self.preprocess_image(&img)?;
        let input_value = Value::from_array(self.environment.clone(), &preprocessed)?;

        // Run inference
        let outputs = self.image_model.run(vec![input_value])?;
        let predictions = outputs[0].try_extract::<f32>()?.view().to_owned();

        // Parse predictions
        let categories = self.parse_image_predictions(&predictions);
        let result = self.generate_moderation_result(
            &cache_key,
            categories,
            "image_model_v1.0",
        );

        self.update_statistics(&result, start.elapsed()).await;
        self.cache.insert(cache_key, result.clone());

        Ok(result)
    }

    pub async fn moderate_video(&self, video_frames: Vec<Vec<u8>>) -> Result<ModerationResult> {
        let cache_key = format!("video:{}", Uuid::new_v4());
        let start = std::time::Instant::now();

        // Sample key frames
        let sampled_frames = self.sample_key_frames(&video_frames, 10)?;
        let mut frame_results = Vec::new();

        for frame_data in sampled_frames {
            let img = image::load_from_memory(&frame_data)?;
            let preprocessed = self.preprocess_image(&img)?;
            let input_value = Value::from_array(self.environment.clone(), &preprocessed)?;

            let outputs = self.video_model.run(vec![input_value])?;
            let predictions = outputs[0].try_extract::<f32>()?.view().to_owned();
            frame_results.push(predictions);
        }

        // Aggregate frame results
        let categories = self.aggregate_video_predictions(&frame_results);
        let result = self.generate_moderation_result(
            &cache_key,
            categories,
            "video_model_v1.0",
        );

        self.update_statistics(&result, start.elapsed()).await;

        Ok(result)
    }

    pub async fn moderate_audio(&self, audio_data: &[f32], sample_rate: u32) -> Result<ModerationResult> {
        let cache_key = format!("audio:{}", Uuid::new_v4());
        let start = std::time::Instant::now();

        // Extract audio features
        let features = self.extract_audio_features(audio_data, sample_rate)?;
        let input_value = Value::from_array(self.environment.clone(), &features)?;

        // Run inference
        let outputs = self.audio_model.run(vec![input_value])?;
        let predictions = outputs[0].try_extract::<f32>()?.view().to_owned();

        // Parse predictions
        let categories = self.parse_audio_predictions(&predictions);
        let result = self.generate_moderation_result(
            &cache_key,
            categories,
            "audio_model_v1.0",
        );

        self.update_statistics(&result, start.elapsed()).await;

        Ok(result)
    }

    pub async fn apply_custom_rules(&self, content: &str) -> Vec<ModerationFlag> {
        let mut flags = Vec::new();
        let rules = self.rules.read().await;

        for rule in rules.iter().filter(|r| r.enabled) {
            // Check pattern matching
            if let Some(pattern) = &rule.pattern {
                if let Ok(re) = regex::Regex::new(pattern) {
                    if re.is_match(content) {
                        flags.push(ModerationFlag {
                            category: rule.category.clone(),
                            severity: rule.severity,
                            confidence: 1.0,
                            details: format!("Matched rule: {}", rule.name),
                        });
                    }
                }
            }

            // Check keyword matching
            for keyword in &rule.keywords {
                if content.to_lowercase().contains(&keyword.to_lowercase()) {
                    flags.push(ModerationFlag {
                        category: rule.category.clone(),
                        severity: rule.severity,
                        confidence: 0.9,
                        details: format!("Contains keyword: {}", keyword),
                    });
                    break;
                }
            }
        }

        flags
    }

    fn tokenize_text(&self, text: &str) -> Result<Vec<f32>> {
        // Simplified tokenization - in production, use proper tokenizer
        let words: Vec<&str> = text.split_whitespace().collect();
        let max_length = 512;

        let mut tokens = Vec::with_capacity(max_length);
        for (i, word) in words.iter().enumerate() {
            if i >= max_length {
                break;
            }
            // Simple hash-based token ID (in production, use vocabulary)
            let token_id = (word.len() as f32) * 100.0 + (word.chars().next().unwrap_or(' ') as f32);
            tokens.push(token_id);
        }

        // Pad to max length
        while tokens.len() < max_length {
            tokens.push(0.0);
        }

        Ok(tokens)
    }

    fn preprocess_image(&self, img: &image::DynamicImage) -> Result<Array2<f32>> {
        let resized = img.resize(224, 224, image::imageops::FilterType::Lanczos3);
        let rgb = resized.to_rgb8();

        let mut array = Array2::zeros((3, 224 * 224));
        for (x, y, pixel) in rgb.enumerate_pixels() {
            let idx = (y * 224 + x) as usize;
            array[[0, idx]] = pixel[0] as f32 / 255.0;
            array[[1, idx]] = pixel[1] as f32 / 255.0;
            array[[2, idx]] = pixel[2] as f32 / 255.0;
        }

        Ok(array)
    }

    fn sample_key_frames(&self, frames: &[Vec<u8>], max_frames: usize) -> Result<Vec<Vec<u8>>> {
        if frames.len() <= max_frames {
            return Ok(frames.to_vec());
        }

        let step = frames.len() / max_frames;
        let mut sampled = Vec::with_capacity(max_frames);

        for i in 0..max_frames {
            sampled.push(frames[i * step].clone());
        }

        Ok(sampled)
    }

    fn extract_audio_features(&self, audio: &[f32], sample_rate: u32) -> Result<Array2<f32>> {
        // Simplified feature extraction - in production, use proper audio processing
        let feature_size = 128;
        let frame_size = 512;
        let hop_size = 256;

        let num_frames = (audio.len() - frame_size) / hop_size + 1;
        let mut features = Array2::zeros((num_frames.min(100), feature_size));

        for i in 0..num_frames.min(100) {
            let start = i * hop_size;
            let end = (start + frame_size).min(audio.len());

            if end > start {
                let frame = &audio[start..end];

                // Simple spectral features
                for (j, chunk) in frame.chunks(4).enumerate() {
                    if j < feature_size {
                        features[[i, j]] = chunk.iter().sum::<f32>() / chunk.len() as f32;
                    }
                }
            }
        }

        Ok(features)
    }

    fn parse_text_predictions(&self, predictions: &Array1<f32>) -> ModerationCategories {
        ModerationCategories {
            violence: CategoryResult {
                detected: predictions[0] > self.thresholds.violence,
                score: predictions[0],
                severity: self.score_to_severity(predictions[0]),
                subcategories: vec![],
            },
            adult_content: CategoryResult {
                detected: predictions[1] > self.thresholds.adult_content,
                score: predictions[1],
                severity: self.score_to_severity(predictions[1]),
                subcategories: vec![],
            },
            hate_speech: CategoryResult {
                detected: predictions[2] > self.thresholds.hate_speech,
                score: predictions[2],
                severity: self.score_to_severity(predictions[2]),
                subcategories: vec![],
            },
            self_harm: CategoryResult {
                detected: predictions[3] > self.thresholds.self_harm,
                score: predictions[3],
                severity: self.score_to_severity(predictions[3]),
                subcategories: vec![],
            },
            illegal_content: CategoryResult {
                detected: predictions[4] > self.thresholds.illegal_content,
                score: predictions[4],
                severity: self.score_to_severity(predictions[4]),
                subcategories: vec![],
            },
            harassment: CategoryResult {
                detected: predictions[5] > self.thresholds.harassment,
                score: predictions[5],
                severity: self.score_to_severity(predictions[5]),
                subcategories: vec![],
            },
            misinformation: CategoryResult {
                detected: predictions[6] > self.thresholds.misinformation,
                score: predictions[6],
                severity: self.score_to_severity(predictions[6]),
                subcategories: vec![],
            },
            spam: CategoryResult {
                detected: predictions[7] > self.thresholds.spam,
                score: predictions[7],
                severity: self.score_to_severity(predictions[7]),
                subcategories: vec![],
            },
            copyright: CategoryResult {
                detected: predictions[8] > self.thresholds.copyright,
                score: predictions[8],
                severity: self.score_to_severity(predictions[8]),
                subcategories: vec![],
            },
            privacy_violation: CategoryResult {
                detected: predictions[9] > self.thresholds.privacy_violation,
                score: predictions[9],
                severity: self.score_to_severity(predictions[9]),
                subcategories: vec![],
            },
        }
    }

    fn parse_image_predictions(&self, predictions: &Array1<f32>) -> ModerationCategories {
        // Similar to parse_text_predictions but for image-specific categories
        self.parse_text_predictions(predictions)
    }

    fn parse_audio_predictions(&self, predictions: &Array1<f32>) -> ModerationCategories {
        // Similar to parse_text_predictions but for audio-specific categories
        self.parse_text_predictions(predictions)
    }

    fn aggregate_video_predictions(&self, frame_results: &[Array1<f32>]) -> ModerationCategories {
        // Take maximum score across all frames for each category
        let mut max_scores = Array1::zeros(10);

        for frame in frame_results {
            for i in 0..10 {
                max_scores[i] = max_scores[i].max(frame[i]);
            }
        }

        self.parse_text_predictions(&max_scores)
    }

    fn score_to_severity(&self, score: f32) -> SeverityLevel {
        if score < 0.2 {
            SeverityLevel::None
        } else if score < 0.4 {
            SeverityLevel::Low
        } else if score < 0.6 {
            SeverityLevel::Medium
        } else if score < 0.8 {
            SeverityLevel::High
        } else {
            SeverityLevel::Critical
        }
    }

    fn generate_moderation_result(
        &self,
        content_id: &str,
        categories: ModerationCategories,
        model_version: &str,
    ) -> ModerationResult {
        let mut flags = Vec::new();
        let mut confidence_scores = HashMap::new();

        // Check each category and generate flags
        macro_rules! check_category {
            ($cat:ident, $name:expr) => {
                if categories.$cat.detected {
                    flags.push(ModerationFlag {
                        category: $name.to_string(),
                        severity: categories.$cat.severity,
                        confidence: categories.$cat.score,
                        details: format!("{} detected with {:.2}% confidence", $name, categories.$cat.score * 100.0),
                    });
                }
                confidence_scores.insert($name.to_string(), categories.$cat.score);
            };
        }

        check_category!(violence, "violence");
        check_category!(adult_content, "adult_content");
        check_category!(hate_speech, "hate_speech");
        check_category!(self_harm, "self_harm");
        check_category!(illegal_content, "illegal_content");
        check_category!(harassment, "harassment");
        check_category!(misinformation, "misinformation");
        check_category!(spam, "spam");
        check_category!(copyright, "copyright");
        check_category!(privacy_violation, "privacy_violation");

        // Determine action based on severity
        let max_score = confidence_scores.values().cloned().fold(0.0f32, f32::max);
        let action = if max_score >= self.thresholds.auto_block_threshold {
            ModerationAction::Block
        } else if max_score >= self.thresholds.blur_threshold {
            ModerationAction::Blur
        } else if max_score >= self.thresholds.auto_review_threshold {
            ModerationAction::Review
        } else {
            ModerationAction::Allow
        };

        let safe = flags.is_empty();

        ModerationResult {
            content_id: content_id.to_string(),
            safe,
            categories,
            confidence_scores,
            flags,
            action,
            explanation: if !safe {
                Some(format!("Content flagged for {} issues", flags.len()))
            } else {
                None
            },
            model_version: model_version.to_string(),
            timestamp: chrono::Utc::now(),
        }
    }

    async fn update_statistics(&self, result: &ModerationResult, duration: std::time::Duration) {
        let mut stats = self.statistics.write().await;

        stats.total_moderated += 1;

        match result.action {
            ModerationAction::Block | ModerationAction::Report => stats.blocked_count += 1,
            ModerationAction::Review | ModerationAction::Restrict => stats.reviewed_count += 1,
            ModerationAction::Allow => stats.allowed_count += 1,
            _ => {}
        }

        for flag in &result.flags {
            *stats.category_counts.entry(flag.category.clone()).or_insert(0) += 1;
        }

        // Update average processing time
        let current_avg = stats.average_processing_time_ms;
        let current_count = stats.total_moderated as f64;
        stats.average_processing_time_ms =
            (current_avg * (current_count - 1.0) + duration.as_millis() as f64) / current_count;
    }

    fn load_default_rules() -> Vec<ModerationRule> {
        vec![
            ModerationRule {
                id: Uuid::new_v4(),
                name: "Explicit Violence".to_string(),
                description: "Block content with explicit violence".to_string(),
                pattern: Some(r"(?i)(gore|graphic violence|murder|killing)".to_string()),
                keywords: vec!["gore".to_string(), "graphic".to_string()],
                category: "violence".to_string(),
                action: ModerationAction::Block,
                severity: SeverityLevel::Critical,
                enabled: true,
            },
            ModerationRule {
                id: Uuid::new_v4(),
                name: "Hate Speech".to_string(),
                description: "Block hate speech and discrimination".to_string(),
                pattern: None,
                keywords: vec![], // Would contain actual hate terms
                category: "hate_speech".to_string(),
                action: ModerationAction::Block,
                severity: SeverityLevel::High,
                enabled: true,
            },
            ModerationRule {
                id: Uuid::new_v4(),
                name: "Spam Detection".to_string(),
                description: "Flag potential spam content".to_string(),
                pattern: Some(r"(?i)(click here|limited time|act now|free money)".to_string()),
                keywords: vec!["spam".to_string(), "scam".to_string()],
                category: "spam".to_string(),
                action: ModerationAction::Review,
                severity: SeverityLevel::Medium,
                enabled: true,
            },
        ]
    }

    pub async fn get_statistics(&self) -> ModerationStatistics {
        self.statistics.read().await.clone()
    }

    pub async fn add_rule(&self, rule: ModerationRule) -> Result<()> {
        let mut rules = self.rules.write().await;
        rules.push(rule);
        Ok(())
    }

    pub async fn remove_rule(&self, rule_id: Uuid) -> Result<()> {
        let mut rules = self.rules.write().await;
        rules.retain(|r| r.id != rule_id);
        Ok(())
    }

    pub async fn update_thresholds(&mut self, thresholds: ModerationThresholds) {
        self.thresholds = thresholds;
        // Clear cache when thresholds change
        self.cache.clear();
    }

    pub async fn report_false_positive(&self, content_id: &str) {
        let mut stats = self.statistics.write().await;
        stats.false_positives += 1;

        // Remove from cache to force re-evaluation
        self.cache.remove(content_id);

        info!("False positive reported for content: {}", content_id);
    }

    pub async fn report_false_negative(&self, content_id: &str) {
        let mut stats = self.statistics.write().await;
        stats.false_negatives += 1;

        warn!("False negative reported for content: {}", content_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_text_moderation() {
        // Test would require actual model files
    }

    #[tokio::test]
    async fn test_custom_rules() {
        // Test custom rule application
    }

    #[tokio::test]
    async fn test_threshold_configuration() {
        // Test threshold updates
    }
}