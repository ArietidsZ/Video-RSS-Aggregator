use anyhow::{Context, Result};
use dashmap::DashMap;
use image::{DynamicImage, GenericImageView, Pixel};
use ndarray::{Array3, Array4, s};
use ort::{Environment, Session, SessionBuilder, Value};
use photon_rs::transform;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdultContentResult {
    pub content_id: String,
    pub is_safe: bool,
    pub safety_score: f32,
    pub categories: ContentCategories,
    pub detection_confidence: f32,
    pub requires_blur: bool,
    pub requires_warning: bool,
    pub age_restriction: AgeRestriction,
    pub detected_regions: Vec<DetectedRegion>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentCategories {
    pub adult: f32,
    pub violence: f32,
    pub suggestive: f32,
    pub medical: f32,
    pub artistic: f32,
    pub educational: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgeRestriction {
    AllAges,      // G - General Audiences
    Teen,         // PG-13 - 13+
    Mature,       // R - 17+
    Adult,        // NC-17 - 18+
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedRegion {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub confidence: f32,
    pub category: String,
}

#[derive(Debug, Clone)]
pub struct AdultContentFilter {
    // ML Models
    image_classifier: Arc<Session>,
    text_classifier: Arc<Session>,
    video_classifier: Arc<Session>,

    // Configuration
    config: FilterConfig,

    // Caching
    cache: Arc<DashMap<String, AdultContentResult>>,

    // Statistics
    statistics: Arc<RwLock<FilterStatistics>>,

    // ONNX Runtime Environment
    environment: Arc<Environment>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterConfig {
    pub adult_threshold: f32,
    pub violence_threshold: f32,
    pub suggestive_threshold: f32,
    pub blur_threshold: f32,
    pub warning_threshold: f32,
    pub cache_ttl_seconds: u64,
    pub max_image_size: u32,
    pub enable_region_detection: bool,
    pub strict_mode: bool,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            adult_threshold: 0.8,
            violence_threshold: 0.7,
            suggestive_threshold: 0.6,
            blur_threshold: 0.7,
            warning_threshold: 0.5,
            cache_ttl_seconds: 3600,
            max_image_size: 4096,
            enable_region_detection: true,
            strict_mode: false,
        }
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct FilterStatistics {
    pub total_scanned: u64,
    pub adult_detected: u64,
    pub violence_detected: u64,
    pub suggestive_detected: u64,
    pub false_positives: u64,
    pub false_negatives: u64,
    pub average_processing_time_ms: f64,
    pub category_distribution: HashMap<String, u64>,
}

impl AdultContentFilter {
    pub async fn new(models_path: &Path, config: FilterConfig) -> Result<Self> {
        let environment = Arc::new(
            Environment::builder()
                .with_name("adult_filter")
                .with_log_level(ort::LoggingLevel::Warning)
                .build()?,
        );

        // Load ML models
        let image_classifier = Arc::new(
            SessionBuilder::new(&environment)?
                .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
                .with_model_from_file(models_path.join("nsfw_image_classifier.onnx"))?
        );

        let text_classifier = Arc::new(
            SessionBuilder::new(&environment)?
                .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
                .with_model_from_file(models_path.join("nsfw_text_classifier.onnx"))?
        );

        let video_classifier = Arc::new(
            SessionBuilder::new(&environment)?
                .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
                .with_model_from_file(models_path.join("nsfw_video_classifier.onnx"))?
        );

        Ok(Self {
            image_classifier,
            text_classifier,
            video_classifier,
            config,
            cache: Arc::new(DashMap::new()),
            statistics: Arc::new(RwLock::new(FilterStatistics::default())),
            environment,
        })
    }

    pub async fn check_image(&self, image_data: &[u8]) -> Result<AdultContentResult> {
        let content_id = format!("img_{}", blake3::hash(image_data));

        // Check cache
        if let Some(cached) = self.get_cached_result(&content_id).await {
            return Ok(cached);
        }

        let start = std::time::Instant::now();

        // Load and preprocess image
        let img = image::load_from_memory(image_data)?;
        let preprocessed = self.preprocess_image(&img)?;

        // Run classification
        let input_value = Value::from_array(self.environment.clone(), &preprocessed)?;
        let outputs = self.image_classifier.run(vec![input_value])?;
        let predictions = outputs[0].try_extract::<f32>()?.view().to_owned();

        // Parse predictions
        let categories = self.parse_image_predictions(&predictions);

        // Detect regions if enabled
        let detected_regions = if self.config.enable_region_detection {
            self.detect_sensitive_regions(&img).await?
        } else {
            Vec::new()
        };

        // Generate result
        let result = self.generate_result(content_id, categories, detected_regions);

        // Update statistics
        self.update_statistics(&result, start.elapsed()).await;

        // Cache result
        self.cache_result(&result).await;

        Ok(result)
    }

    pub async fn check_text(&self, text: &str) -> Result<AdultContentResult> {
        let content_id = format!("txt_{}", blake3::hash(text.as_bytes()));

        // Check cache
        if let Some(cached) = self.get_cached_result(&content_id).await {
            return Ok(cached);
        }

        let start = std::time::Instant::now();

        // Tokenize and prepare input
        let tokens = self.tokenize_text(text)?;
        let input = self.prepare_text_input(&tokens)?;
        let input_value = Value::from_array(self.environment.clone(), &input)?;

        // Run classification
        let outputs = self.text_classifier.run(vec![input_value])?;
        let predictions = outputs[0].try_extract::<f32>()?.view().to_owned();

        // Parse predictions
        let categories = self.parse_text_predictions(&predictions);

        // Generate result
        let result = self.generate_result(content_id, categories, Vec::new());

        // Update statistics
        self.update_statistics(&result, start.elapsed()).await;

        // Cache result
        self.cache_result(&result).await;

        Ok(result)
    }

    pub async fn check_video(&self, video_frames: &[Vec<u8>]) -> Result<AdultContentResult> {
        let content_id = format!("vid_{}", Uuid::new_v4());
        let start = std::time::Instant::now();

        // Sample key frames
        let key_frames = self.sample_video_frames(video_frames, 10)?;
        let mut frame_categories = Vec::new();
        let mut all_regions = Vec::new();

        for (i, frame_data) in key_frames.iter().enumerate() {
            let img = image::load_from_memory(frame_data)?;
            let preprocessed = self.preprocess_image(&img)?;
            let input_value = Value::from_array(self.environment.clone(), &preprocessed)?;

            let outputs = self.video_classifier.run(vec![input_value])?;
            let predictions = outputs[0].try_extract::<f32>()?.view().to_owned();

            frame_categories.push(self.parse_image_predictions(&predictions));

            if self.config.enable_region_detection {
                let mut regions = self.detect_sensitive_regions(&img).await?;
                for region in &mut regions {
                    // Add frame number to region for video context
                    region.category = format!("frame_{}: {}", i, region.category);
                }
                all_regions.extend(regions);
            }
        }

        // Aggregate frame results
        let categories = self.aggregate_video_categories(&frame_categories);

        // Generate result
        let result = self.generate_result(content_id, categories, all_regions);

        // Update statistics
        self.update_statistics(&result, start.elapsed()).await;

        Ok(result)
    }

    pub async fn blur_sensitive_regions(
        &self,
        image: &mut DynamicImage,
        regions: &[DetectedRegion],
    ) -> Result<()> {
        for region in regions {
            if region.confidence > self.config.blur_threshold {
                self.apply_blur(image, region)?;
            }
        }
        Ok(())
    }

    fn preprocess_image(&self, img: &DynamicImage) -> Result<Array4<f32>> {
        // Resize to model input size (224x224 for most classifiers)
        let resized = img.resize_exact(224, 224, image::imageops::FilterType::Lanczos3);
        let rgb = resized.to_rgb8();

        let mut array = Array4::zeros((1, 3, 224, 224));

        for (x, y, pixel) in rgb.enumerate_pixels() {
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;

            // Normalize with ImageNet mean and std
            array[[0, 0, y as usize, x as usize]] = (r - 0.485) / 0.229;
            array[[0, 1, y as usize, x as usize]] = (g - 0.456) / 0.224;
            array[[0, 2, y as usize, x as usize]] = (b - 0.406) / 0.225;
        }

        Ok(array)
    }

    fn tokenize_text(&self, text: &str) -> Result<Vec<i64>> {
        // Simplified tokenization - in production, use proper tokenizer
        let words: Vec<&str> = text.split_whitespace().take(512).collect();
        let mut tokens = Vec::with_capacity(512);

        for word in words {
            // Simple hash-based token ID
            let token_id = (word.len() as i64) * 1000 + (word.chars().next().unwrap_or(' ') as i64);
            tokens.push(token_id);
        }

        // Pad to fixed length
        while tokens.len() < 512 {
            tokens.push(0);
        }

        Ok(tokens)
    }

    fn prepare_text_input(&self, tokens: &[i64]) -> Result<Array3<i64>> {
        let batch_size = 1;
        let seq_len = tokens.len();
        let mut array = Array3::zeros((batch_size, seq_len, 1));

        for (i, &token) in tokens.iter().enumerate() {
            array[[0, i, 0]] = token;
        }

        Ok(array)
    }

    fn parse_image_predictions(&self, predictions: &ndarray::ArrayView1<f32>) -> ContentCategories {
        ContentCategories {
            adult: predictions[0],
            violence: predictions[1],
            suggestive: predictions[2],
            medical: predictions[3],
            artistic: predictions[4],
            educational: predictions[5],
        }
    }

    fn parse_text_predictions(&self, predictions: &ndarray::ArrayView1<f32>) -> ContentCategories {
        // Similar structure but may have different indices based on model
        ContentCategories {
            adult: predictions[0],
            violence: predictions[1],
            suggestive: predictions[2],
            medical: 0.0,  // Text models may not detect these
            artistic: 0.0,
            educational: 0.0,
        }
    }

    fn aggregate_video_categories(&self, frame_categories: &[ContentCategories]) -> ContentCategories {
        if frame_categories.is_empty() {
            return ContentCategories {
                adult: 0.0,
                violence: 0.0,
                suggestive: 0.0,
                medical: 0.0,
                artistic: 0.0,
                educational: 0.0,
            };
        }

        // Take maximum score across all frames for safety
        let mut max_categories = ContentCategories {
            adult: 0.0,
            violence: 0.0,
            suggestive: 0.0,
            medical: 0.0,
            artistic: 0.0,
            educational: 0.0,
        };

        for cat in frame_categories {
            max_categories.adult = max_categories.adult.max(cat.adult);
            max_categories.violence = max_categories.violence.max(cat.violence);
            max_categories.suggestive = max_categories.suggestive.max(cat.suggestive);
            max_categories.medical = max_categories.medical.max(cat.medical);
            max_categories.artistic = max_categories.artistic.max(cat.artistic);
            max_categories.educational = max_categories.educational.max(cat.educational);
        }

        max_categories
    }

    async fn detect_sensitive_regions(&self, img: &DynamicImage) -> Result<Vec<DetectedRegion>> {
        // Simplified region detection - in production, use proper object detection model
        let mut regions = Vec::new();

        // Divide image into grid for analysis
        let (width, height) = img.dimensions();
        let grid_size = 64;

        for y in (0..height).step_by(grid_size as usize) {
            for x in (0..width).step_by(grid_size as usize) {
                let region_width = grid_size.min(width - x);
                let region_height = grid_size.min(height - y);

                let sub_image = img.crop_imm(x, y, region_width, region_height);
                let score = self.analyze_region(&sub_image)?;

                if score > self.config.blur_threshold {
                    regions.push(DetectedRegion {
                        x,
                        y,
                        width: region_width,
                        height: region_height,
                        confidence: score,
                        category: "sensitive".to_string(),
                    });
                }
            }
        }

        // Merge adjacent regions
        self.merge_adjacent_regions(&mut regions);

        Ok(regions)
    }

    fn analyze_region(&self, img: &DynamicImage) -> Result<f32> {
        // Simplified skin tone detection as a proxy for sensitivity
        let rgb = img.to_rgb8();
        let mut skin_pixels = 0;
        let total_pixels = rgb.width() * rgb.height();

        for pixel in rgb.pixels() {
            let r = pixel[0] as f32;
            let g = pixel[1] as f32;
            let b = pixel[2] as f32;

            // Simple skin tone detection
            if r > 95.0 && g > 40.0 && b > 20.0
                && r > g && r > b
                && (r - g).abs() > 15.0
                && r.max(g).max(b) - r.min(g).min(b) > 15.0 {
                skin_pixels += 1;
            }
        }

        Ok(skin_pixels as f32 / total_pixels as f32)
    }

    fn merge_adjacent_regions(&self, regions: &mut Vec<DetectedRegion>) {
        // Merge overlapping or adjacent regions
        let mut merged = true;
        while merged {
            merged = false;
            let mut i = 0;
            while i < regions.len() {
                let mut j = i + 1;
                while j < regions.len() {
                    if self.regions_overlap(&regions[i], &regions[j]) {
                        let merged_region = self.merge_regions(&regions[i], &regions[j]);
                        regions[i] = merged_region;
                        regions.remove(j);
                        merged = true;
                    } else {
                        j += 1;
                    }
                }
                i += 1;
            }
        }
    }

    fn regions_overlap(&self, r1: &DetectedRegion, r2: &DetectedRegion) -> bool {
        let x1_end = r1.x + r1.width;
        let y1_end = r1.y + r1.height;
        let x2_end = r2.x + r2.width;
        let y2_end = r2.y + r2.height;

        !(r1.x > x2_end || r2.x > x1_end || r1.y > y2_end || r2.y > y1_end)
    }

    fn merge_regions(&self, r1: &DetectedRegion, r2: &DetectedRegion) -> DetectedRegion {
        let x = r1.x.min(r2.x);
        let y = r1.y.min(r2.y);
        let x_end = (r1.x + r1.width).max(r2.x + r2.width);
        let y_end = (r1.y + r1.height).max(r2.y + r2.height);

        DetectedRegion {
            x,
            y,
            width: x_end - x,
            height: y_end - y,
            confidence: r1.confidence.max(r2.confidence),
            category: r1.category.clone(),
        }
    }

    fn apply_blur(&self, img: &mut DynamicImage, region: &DetectedRegion) -> Result<()> {
        // Extract region
        let sub_img = img.crop(region.x, region.y, region.width, region.height);

        // Apply Gaussian blur
        let blurred = sub_img.blur(20.0);

        // Replace original region with blurred version
        for (x, y, pixel) in blurred.pixels() {
            img.put_pixel(region.x + x, region.y + y, pixel);
        }

        Ok(())
    }

    fn sample_video_frames(&self, frames: &[Vec<u8>], max_frames: usize) -> Result<Vec<Vec<u8>>> {
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

    fn generate_result(
        &self,
        content_id: String,
        categories: ContentCategories,
        detected_regions: Vec<DetectedRegion>,
    ) -> AdultContentResult {
        let is_adult = categories.adult > self.config.adult_threshold;
        let is_violent = categories.violence > self.config.violence_threshold;
        let is_suggestive = categories.suggestive > self.config.suggestive_threshold;

        let is_safe = !is_adult && !is_violent && !is_suggestive;

        let safety_score = 1.0 - categories.adult.max(categories.violence).max(categories.suggestive);

        let requires_blur = categories.adult > self.config.blur_threshold
            || categories.violence > self.config.blur_threshold;

        let requires_warning = categories.suggestive > self.config.warning_threshold
            || categories.violence > self.config.warning_threshold;

        let age_restriction = if is_adult {
            AgeRestriction::Adult
        } else if is_violent || categories.suggestive > 0.7 {
            AgeRestriction::Mature
        } else if is_suggestive {
            AgeRestriction::Teen
        } else {
            AgeRestriction::AllAges
        };

        let detection_confidence = if is_safe { safety_score } else {
            categories.adult.max(categories.violence).max(categories.suggestive)
        };

        AdultContentResult {
            content_id,
            is_safe,
            safety_score,
            categories,
            detection_confidence,
            requires_blur,
            requires_warning,
            age_restriction,
            detected_regions,
            timestamp: chrono::Utc::now(),
        }
    }

    async fn get_cached_result(&self, content_id: &str) -> Option<AdultContentResult> {
        self.cache.get(content_id).map(|entry| entry.clone())
    }

    async fn cache_result(&self, result: &AdultContentResult) {
        self.cache.insert(result.content_id.clone(), result.clone());

        // Clean old cache entries periodically
        if self.cache.len() > 10000 {
            let to_remove: Vec<String> = self.cache
                .iter()
                .filter(|entry| {
                    let age = chrono::Utc::now().timestamp() - entry.value().timestamp.timestamp();
                    age > self.config.cache_ttl_seconds as i64
                })
                .map(|entry| entry.key().clone())
                .collect();

            for key in to_remove {
                self.cache.remove(&key);
            }
        }
    }

    async fn update_statistics(&self, result: &AdultContentResult, duration: std::time::Duration) {
        let mut stats = self.statistics.write().await;

        stats.total_scanned += 1;

        if result.categories.adult > self.config.adult_threshold {
            stats.adult_detected += 1;
            *stats.category_distribution.entry("adult".to_string()).or_insert(0) += 1;
        }

        if result.categories.violence > self.config.violence_threshold {
            stats.violence_detected += 1;
            *stats.category_distribution.entry("violence".to_string()).or_insert(0) += 1;
        }

        if result.categories.suggestive > self.config.suggestive_threshold {
            stats.suggestive_detected += 1;
            *stats.category_distribution.entry("suggestive".to_string()).or_insert(0) += 1;
        }

        // Update average processing time
        let current_avg = stats.average_processing_time_ms;
        let current_count = stats.total_scanned as f64;
        stats.average_processing_time_ms =
            (current_avg * (current_count - 1.0) + duration.as_millis() as f64) / current_count;
    }

    pub async fn report_false_positive(&self, content_id: &str) {
        let mut stats = self.statistics.write().await;
        stats.false_positives += 1;
        self.cache.remove(content_id);
        info!("False positive reported for content: {}", content_id);
    }

    pub async fn report_false_negative(&self, content_id: &str) {
        let mut stats = self.statistics.write().await;
        stats.false_negatives += 1;
        self.cache.remove(content_id);
        warn!("False negative reported for content: {}", content_id);
    }

    pub async fn get_statistics(&self) -> FilterStatistics {
        self.statistics.read().await.clone()
    }

    pub fn update_config(&mut self, config: FilterConfig) {
        self.config = config;
        // Clear cache when config changes
        self.cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_age_restriction_classification() {
        // Test age restriction logic
        let filter = AdultContentFilter {
            image_classifier: Arc::new(unsafe { std::mem::zeroed() }),
            text_classifier: Arc::new(unsafe { std::mem::zeroed() }),
            video_classifier: Arc::new(unsafe { std::mem::zeroed() }),
            config: FilterConfig::default(),
            cache: Arc::new(DashMap::new()),
            statistics: Arc::new(RwLock::new(FilterStatistics::default())),
            environment: Arc::new(unsafe { std::mem::zeroed() }),
        };

        let categories = ContentCategories {
            adult: 0.9,
            violence: 0.1,
            suggestive: 0.2,
            medical: 0.0,
            artistic: 0.0,
            educational: 0.0,
        };

        let result = filter.generate_result(
            "test".to_string(),
            categories,
            Vec::new(),
        );

        assert_eq!(result.age_restriction, AgeRestriction::Adult);
        assert!(!result.is_safe);
        assert!(result.requires_blur);
    }
}