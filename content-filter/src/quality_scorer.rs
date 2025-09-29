use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentQualityScore {
    pub content_id: String,
    pub overall_score: f32,
    pub technical_score: f32,
    pub content_score: f32,
    pub engagement_score: f32,
    pub freshness_score: f32,
    pub authority_score: f32,
    pub relevance_score: f32,
    pub originality_score: f32,
    pub completeness_score: f32,
    pub readability_score: f32,
    pub multimedia_score: f32,
    pub components: QualityComponents,
    pub issues: Vec<QualityIssue>,
    pub recommendations: Vec<String>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityComponents {
    // Video Quality
    pub resolution: VideoResolution,
    pub bitrate: u32,
    pub framerate: f32,
    pub codec: String,
    pub duration: u32,
    pub file_size: u64,

    // Audio Quality
    pub audio_bitrate: u32,
    pub audio_channels: u8,
    pub audio_codec: String,
    pub has_audio: bool,
    pub audio_clarity: f32,

    // Content Metrics
    pub transcript_word_count: u32,
    pub unique_words: u32,
    pub sentence_count: u32,
    pub avg_sentence_length: f32,
    pub vocabulary_richness: f32,
    pub speaking_pace: f32, // words per minute

    // Summary Quality
    pub summary_length: u32,
    pub summary_coherence: f32,
    pub key_points_coverage: f32,
    pub information_density: f32,

    // Metadata Quality
    pub has_title: bool,
    pub has_description: bool,
    pub has_tags: bool,
    pub has_thumbnail: bool,
    pub metadata_completeness: f32,

    // Engagement Indicators
    pub view_count: Option<u64>,
    pub like_ratio: Option<f32>,
    pub comment_count: Option<u32>,
    pub share_count: Option<u32>,
    pub watch_time_ratio: Option<f32>,

    // Technical Indicators
    pub processing_success_rate: f32,
    pub transcription_confidence: f32,
    pub language_detection_confidence: f32,
    pub content_classification_confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoResolution {
    pub width: u32,
    pub height: u32,
    pub quality_tier: QualityTier,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityTier {
    UHD8K,    // 7680x4320
    UHD4K,    // 3840x2160
    QHD,      // 2560x1440
    FHD,      // 1920x1080
    HD,       // 1280x720
    SD,       // 854x480
    LD,       // 426x240
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityIssue {
    pub issue_type: IssueType,
    pub severity: IssueSeverity,
    pub description: String,
    pub impact_on_score: f32,
    pub can_be_fixed: bool,
    pub fix_suggestion: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueType {
    LowResolution,
    PoorAudioQuality,
    MissingTranscript,
    ShortContent,
    LowEngagement,
    IncompleteMetadata,
    PoorReadability,
    LowOriginality,
    OutdatedContent,
    ProcessingError,
    ContentWarning,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueSeverity {
    Critical,
    Major,
    Minor,
    Info,
}

pub struct QualityScorer {
    weights: ScoringWeights,
    thresholds: QualityThresholds,
    language_model: Option<LanguageQualityModel>,
}

#[derive(Debug, Clone)]
struct ScoringWeights {
    technical: f32,
    content: f32,
    engagement: f32,
    freshness: f32,
    authority: f32,
    relevance: f32,
    originality: f32,
    completeness: f32,
    readability: f32,
    multimedia: f32,
}

#[derive(Debug, Clone)]
struct QualityThresholds {
    min_resolution_width: u32,
    min_bitrate: u32,
    min_duration: u32,
    min_word_count: u32,
    min_vocabulary_richness: f32,
    max_speaking_pace: f32,
    min_summary_coherence: f32,
    min_metadata_completeness: f32,
    min_transcription_confidence: f32,
}

struct LanguageQualityModel {
    // Placeholder for ML model
}

impl QualityScorer {
    pub fn new() -> Self {
        Self {
            weights: Self::default_weights(),
            thresholds: Self::default_thresholds(),
            language_model: None,
        }
    }

    fn default_weights() -> ScoringWeights {
        ScoringWeights {
            technical: 0.20,
            content: 0.25,
            engagement: 0.15,
            freshness: 0.10,
            authority: 0.10,
            relevance: 0.05,
            originality: 0.05,
            completeness: 0.05,
            readability: 0.03,
            multimedia: 0.02,
        }
    }

    fn default_thresholds() -> QualityThresholds {
        QualityThresholds {
            min_resolution_width: 1280,
            min_bitrate: 1000000, // 1 Mbps
            min_duration: 60, // 1 minute
            min_word_count: 100,
            min_vocabulary_richness: 0.3,
            max_speaking_pace: 200.0,
            min_summary_coherence: 0.6,
            min_metadata_completeness: 0.7,
            min_transcription_confidence: 0.8,
        }
    }

    pub async fn score_content(&self, content: &ContentMetadata) -> Result<ContentQualityScore> {
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();

        // Calculate component scores
        let technical_score = self.calculate_technical_score(&content.components, &mut issues);
        let content_score = self.calculate_content_score(&content.components, &mut issues);
        let engagement_score = self.calculate_engagement_score(&content.components, &mut issues);
        let freshness_score = self.calculate_freshness_score(&content.published_at);
        let authority_score = self.calculate_authority_score(&content.source);
        let relevance_score = self.calculate_relevance_score(&content.components);
        let originality_score = self.calculate_originality_score(&content.components);
        let completeness_score = self.calculate_completeness_score(&content.components, &mut issues);
        let readability_score = self.calculate_readability_score(&content.components);
        let multimedia_score = self.calculate_multimedia_score(&content.components);

        // Calculate weighted overall score
        let overall_score = self.calculate_overall_score(
            technical_score,
            content_score,
            engagement_score,
            freshness_score,
            authority_score,
            relevance_score,
            originality_score,
            completeness_score,
            readability_score,
            multimedia_score,
        );

        // Generate recommendations based on issues
        self.generate_recommendations(&issues, &mut recommendations);

        Ok(ContentQualityScore {
            content_id: content.id.clone(),
            overall_score,
            technical_score,
            content_score,
            engagement_score,
            freshness_score,
            authority_score,
            relevance_score,
            originality_score,
            completeness_score,
            readability_score,
            multimedia_score,
            components: content.components.clone(),
            issues,
            recommendations,
            timestamp: Utc::now(),
        })
    }

    fn calculate_technical_score(&self, components: &QualityComponents, issues: &mut Vec<QualityIssue>) -> f32 {
        let mut score = 100.0;

        // Resolution score
        let resolution_score = match components.resolution.quality_tier {
            QualityTier::UHD8K => 100.0,
            QualityTier::UHD4K => 95.0,
            QualityTier::QHD => 90.0,
            QualityTier::FHD => 85.0,
            QualityTier::HD => 75.0,
            QualityTier::SD => 60.0,
            QualityTier::LD => 40.0,
        };

        if components.resolution.width < self.thresholds.min_resolution_width {
            issues.push(QualityIssue {
                issue_type: IssueType::LowResolution,
                severity: IssueSeverity::Major,
                description: format!("Video resolution {}x{} is below recommended minimum",
                    components.resolution.width, components.resolution.height),
                impact_on_score: -20.0,
                can_be_fixed: false,
                fix_suggestion: Some("Consider using higher quality source videos".to_string()),
            });
            score -= 20.0;
        }

        // Bitrate score
        if components.bitrate < self.thresholds.min_bitrate {
            let bitrate_penalty = (1.0 - components.bitrate as f32 / self.thresholds.min_bitrate as f32) * 15.0;
            score -= bitrate_penalty;
        }

        // Audio quality
        if !components.has_audio {
            score -= 10.0;
        } else if components.audio_clarity < 0.7 {
            issues.push(QualityIssue {
                issue_type: IssueType::PoorAudioQuality,
                severity: IssueSeverity::Major,
                description: "Audio quality is below acceptable threshold".to_string(),
                impact_on_score: -15.0,
                can_be_fixed: true,
                fix_suggestion: Some("Apply audio enhancement filters during processing".to_string()),
            });
            score -= 15.0;
        }

        // Framerate score
        let framerate_score = if components.framerate >= 60.0 {
            100.0
        } else if components.framerate >= 30.0 {
            90.0
        } else if components.framerate >= 24.0 {
            80.0
        } else {
            60.0
        };

        // Processing quality
        if components.processing_success_rate < 0.95 {
            score -= (1.0 - components.processing_success_rate) * 20.0;
        }

        // Transcription confidence
        if components.transcription_confidence < self.thresholds.min_transcription_confidence {
            issues.push(QualityIssue {
                issue_type: IssueType::ProcessingError,
                severity: IssueSeverity::Minor,
                description: "Low transcription confidence may affect content accuracy".to_string(),
                impact_on_score: -10.0,
                can_be_fixed: true,
                fix_suggestion: Some("Use higher quality audio or manual review".to_string()),
            });
            score -= 10.0;
        }

        // Combine sub-scores
        score = (score * 0.3 + resolution_score * 0.3 + framerate_score * 0.2 +
                (components.audio_clarity * 100.0) * 0.2).max(0.0).min(100.0);

        score / 100.0
    }

    fn calculate_content_score(&self, components: &QualityComponents, issues: &mut Vec<QualityIssue>) -> f32 {
        let mut score = 100.0;

        // Word count score
        if components.transcript_word_count < self.thresholds.min_word_count {
            issues.push(QualityIssue {
                issue_type: IssueType::ShortContent,
                severity: IssueSeverity::Major,
                description: format!("Content has only {} words, below minimum of {}",
                    components.transcript_word_count, self.thresholds.min_word_count),
                impact_on_score: -25.0,
                can_be_fixed: false,
                fix_suggestion: None,
            });
            score -= 25.0;
        }

        // Vocabulary richness
        if components.vocabulary_richness < self.thresholds.min_vocabulary_richness {
            score -= (self.thresholds.min_vocabulary_richness - components.vocabulary_richness) * 50.0;
        }

        // Speaking pace
        if components.speaking_pace > self.thresholds.max_speaking_pace {
            let pace_penalty = ((components.speaking_pace - self.thresholds.max_speaking_pace) / 50.0) * 10.0;
            score -= pace_penalty.min(15.0);
        }

        // Summary quality
        if components.summary_coherence < self.thresholds.min_summary_coherence {
            score -= (self.thresholds.min_summary_coherence - components.summary_coherence) * 30.0;
        }

        // Information density
        let density_score = (components.information_density * 100.0).min(100.0);

        // Key points coverage
        let coverage_score = (components.key_points_coverage * 100.0).min(100.0);

        // Duration score
        let duration_score = if components.duration < self.thresholds.min_duration {
            50.0
        } else if components.duration < 300 { // 5 minutes
            70.0
        } else if components.duration < 1200 { // 20 minutes
            90.0
        } else if components.duration < 3600 { // 1 hour
            100.0
        } else {
            85.0 // Very long content may have lower engagement
        };

        // Combine sub-scores
        score = (score * 0.3 + density_score * 0.2 + coverage_score * 0.2 +
                duration_score * 0.15 + (components.summary_coherence * 100.0) * 0.15)
                .max(0.0).min(100.0);

        score / 100.0
    }

    fn calculate_engagement_score(&self, components: &QualityComponents, issues: &mut Vec<QualityIssue>) -> f32 {
        // If no engagement data is available, return neutral score
        if components.view_count.is_none() {
            return 0.5;
        }

        let mut score = 50.0; // Start with neutral score

        // View count score (logarithmic scale)
        if let Some(views) = components.view_count {
            let view_score = (views as f32).log10().min(6.0) / 6.0 * 30.0;
            score += view_score;
        }

        // Like ratio score
        if let Some(like_ratio) = components.like_ratio {
            if like_ratio < 0.8 {
                issues.push(QualityIssue {
                    issue_type: IssueType::LowEngagement,
                    severity: IssueSeverity::Minor,
                    description: format!("Like ratio of {:.1}% indicates mixed reception", like_ratio * 100.0),
                    impact_on_score: -10.0,
                    can_be_fixed: false,
                    fix_suggestion: Some("Review content quality and relevance to audience".to_string()),
                });
            }
            score += like_ratio * 20.0;
        }

        // Comment engagement
        if let Some(comment_count) = components.comment_count {
            if let Some(view_count) = components.view_count {
                let comment_ratio = comment_count as f32 / view_count.max(1) as f32;
                score += (comment_ratio * 1000.0).min(10.0);
            }
        }

        // Watch time ratio
        if let Some(watch_ratio) = components.watch_time_ratio {
            if watch_ratio < 0.4 {
                issues.push(QualityIssue {
                    issue_type: IssueType::LowEngagement,
                    severity: IssueSeverity::Major,
                    description: "Low average watch time suggests content doesn't hold viewer attention".to_string(),
                    impact_on_score: -15.0,
                    can_be_fixed: true,
                    fix_suggestion: Some("Consider improving pacing or content structure".to_string()),
                });
            }
            score += watch_ratio * 20.0;
        }

        (score / 100.0).max(0.0).min(1.0)
    }

    fn calculate_freshness_score(&self, published_at: &Option<DateTime<Utc>>) -> f32 {
        if let Some(published) = published_at {
            let age_days = (Utc::now() - *published).num_days();

            // Exponential decay with half-life of 30 days
            let half_life_days = 30.0;
            let decay_rate = 0.693 / half_life_days;
            let freshness = (-decay_rate * age_days as f64).exp() as f32;

            // Minimum freshness score of 0.2 for very old content
            freshness.max(0.2)
        } else {
            0.5 // Unknown freshness
        }
    }

    fn calculate_authority_score(&self, source: &ContentSource) -> f32 {
        match source.platform {
            Platform::YouTube => {
                let mut score = 0.7; // Base score for YouTube

                if let Some(subscriber_count) = source.channel_subscribers {
                    // Logarithmic scale for subscriber influence
                    let subscriber_score = (subscriber_count as f32).log10().min(7.0) / 7.0 * 0.2;
                    score += subscriber_score;
                }

                if source.is_verified {
                    score += 0.1;
                }

                score.min(1.0)
            },
            Platform::Bilibili => {
                let mut score = 0.65; // Base score for Bilibili

                if let Some(follower_count) = source.channel_followers {
                    let follower_score = (follower_count as f32).log10().min(6.0) / 6.0 * 0.25;
                    score += follower_score;
                }

                if source.is_official {
                    score += 0.1;
                }

                score.min(1.0)
            },
            Platform::Custom => {
                // For custom sources, use trust score if available
                source.trust_score.unwrap_or(0.5)
            },
            _ => 0.5,
        }
    }

    fn calculate_relevance_score(&self, components: &QualityComponents) -> f32 {
        // This would use ML models to determine relevance to user interests
        // For now, use a combination of factors

        let mut score = 0.5; // Base relevance

        // Higher language confidence indicates clearer topic
        score += components.language_detection_confidence * 0.2;

        // Higher classification confidence indicates well-defined content
        score += components.content_classification_confidence * 0.2;

        // Information density indicates focused content
        score += (components.information_density * 0.1).min(0.1);

        score.min(1.0)
    }

    fn calculate_originality_score(&self, components: &QualityComponents) -> f32 {
        // This would use similarity detection against existing content
        // For now, use vocabulary richness as a proxy

        let mut score = 0.3; // Base score

        // Vocabulary richness indicates unique content
        score += components.vocabulary_richness * 0.4;

        // Unique word ratio
        let unique_ratio = components.unique_words as f32 / components.transcript_word_count.max(1) as f32;
        score += unique_ratio * 0.3;

        score.min(1.0)
    }

    fn calculate_completeness_score(&self, components: &QualityComponents, issues: &mut Vec<QualityIssue>) -> f32 {
        let mut score = 0.0;
        let mut max_score = 0.0;

        // Metadata completeness
        if components.has_title {
            score += 20.0;
        } else {
            issues.push(QualityIssue {
                issue_type: IssueType::IncompleteMetadata,
                severity: IssueSeverity::Major,
                description: "Missing video title".to_string(),
                impact_on_score: -20.0,
                can_be_fixed: true,
                fix_suggestion: Some("Extract title from video metadata or generate from content".to_string()),
            });
        }
        max_score += 20.0;

        if components.has_description {
            score += 15.0;
        } else {
            issues.push(QualityIssue {
                issue_type: IssueType::IncompleteMetadata,
                severity: IssueSeverity::Minor,
                description: "Missing video description".to_string(),
                impact_on_score: -10.0,
                can_be_fixed: true,
                fix_suggestion: Some("Generate description from transcript summary".to_string()),
            });
        }
        max_score += 15.0;

        if components.has_tags {
            score += 10.0;
        }
        max_score += 10.0;

        if components.has_thumbnail {
            score += 15.0;
        }
        max_score += 15.0;

        // Transcript completeness
        if components.transcript_word_count > 0 {
            score += 25.0;
        } else {
            issues.push(QualityIssue {
                issue_type: IssueType::MissingTranscript,
                severity: IssueSeverity::Critical,
                description: "No transcript available for content".to_string(),
                impact_on_score: -30.0,
                can_be_fixed: true,
                fix_suggestion: Some("Process video through transcription service".to_string()),
            });
        }
        max_score += 25.0;

        // Summary availability
        if components.summary_length > 0 {
            score += 15.0;
        }
        max_score += 15.0;

        (score / max_score).max(0.0).min(1.0)
    }

    fn calculate_readability_score(&self, components: &QualityComponents) -> f32 {
        // Flesch Reading Ease approximation
        let avg_sentence_length = components.avg_sentence_length;
        let syllables_per_word = 1.5; // Approximation

        let flesch_score = 206.835 - 1.015 * avg_sentence_length - 84.6 * syllables_per_word;
        let normalized_flesch = (flesch_score.max(0.0).min(100.0) / 100.0) as f32;

        // Adjust for speaking pace
        let pace_factor = if components.speaking_pace > 0.0 {
            let optimal_pace = 150.0;
            1.0 - ((components.speaking_pace - optimal_pace).abs() / optimal_pace * 0.3).min(0.3)
        } else {
            0.7
        };

        (normalized_flesch * pace_factor).max(0.0).min(1.0)
    }

    fn calculate_multimedia_score(&self, components: &QualityComponents) -> f32 {
        let mut score = 0.0;

        // Has video
        score += 0.4;

        // Has audio
        if components.has_audio {
            score += 0.3;
        }

        // Has thumbnail
        if components.has_thumbnail {
            score += 0.15;
        }

        // High resolution
        if matches!(components.resolution.quality_tier,
                   QualityTier::FHD | QualityTier::QHD | QualityTier::UHD4K | QualityTier::UHD8K) {
            score += 0.15;
        }

        score.min(1.0)
    }

    fn calculate_overall_score(
        &self,
        technical: f32,
        content: f32,
        engagement: f32,
        freshness: f32,
        authority: f32,
        relevance: f32,
        originality: f32,
        completeness: f32,
        readability: f32,
        multimedia: f32,
    ) -> f32 {
        let weights = &self.weights;

        let weighted_sum =
            technical * weights.technical +
            content * weights.content +
            engagement * weights.engagement +
            freshness * weights.freshness +
            authority * weights.authority +
            relevance * weights.relevance +
            originality * weights.originality +
            completeness * weights.completeness +
            readability * weights.readability +
            multimedia * weights.multimedia;

        let total_weight =
            weights.technical + weights.content + weights.engagement +
            weights.freshness + weights.authority + weights.relevance +
            weights.originality + weights.completeness + weights.readability +
            weights.multimedia;

        (weighted_sum / total_weight).max(0.0).min(1.0)
    }

    fn generate_recommendations(&self, issues: &[QualityIssue], recommendations: &mut Vec<String>) {
        // Group issues by type and severity
        let critical_issues: Vec<_> = issues.iter()
            .filter(|i| matches!(i.severity, IssueSeverity::Critical))
            .collect();

        let major_issues: Vec<_> = issues.iter()
            .filter(|i| matches!(i.severity, IssueSeverity::Major))
            .collect();

        // Priority recommendations for critical issues
        for issue in &critical_issues {
            if let Some(ref fix) = issue.fix_suggestion {
                recommendations.push(format!("CRITICAL: {}", fix));
            }
        }

        // Important recommendations for major issues
        for issue in &major_issues {
            if let Some(ref fix) = issue.fix_suggestion {
                recommendations.push(format!("Important: {}", fix));
            }
        }

        // General recommendations based on patterns
        if issues.iter().any(|i| matches!(i.issue_type, IssueType::LowResolution | IssueType::PoorAudioQuality)) {
            recommendations.push("Consider upgrading source content quality for better user experience".to_string());
        }

        if issues.iter().any(|i| matches!(i.issue_type, IssueType::LowEngagement)) {
            recommendations.push("Analyze audience preferences and optimize content strategy".to_string());
        }

        if issues.iter().any(|i| matches!(i.issue_type, IssueType::IncompleteMetadata)) {
            recommendations.push("Implement automated metadata extraction and enrichment".to_string());
        }

        // Limit recommendations to top 5
        recommendations.truncate(5);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentMetadata {
    pub id: String,
    pub title: String,
    pub description: Option<String>,
    pub source: ContentSource,
    pub published_at: Option<DateTime<Utc>>,
    pub components: QualityComponents,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentSource {
    pub platform: Platform,
    pub channel_id: String,
    pub channel_name: String,
    pub channel_subscribers: Option<u64>,
    pub channel_followers: Option<u64>,
    pub is_verified: bool,
    pub is_official: bool,
    pub trust_score: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Platform {
    YouTube,
    Bilibili,
    Vimeo,
    Dailymotion,
    Custom,
}

impl QualityScorer {
    pub async fn batch_score(&self, contents: Vec<ContentMetadata>) -> Result<Vec<ContentQualityScore>> {
        let mut scores = Vec::with_capacity(contents.len());

        for content in contents {
            match self.score_content(&content).await {
                Ok(score) => scores.push(score),
                Err(e) => {
                    warn!("Failed to score content {}: {}", content.id, e);
                    // Create a minimal score for failed content
                    scores.push(ContentQualityScore {
                        content_id: content.id,
                        overall_score: 0.0,
                        technical_score: 0.0,
                        content_score: 0.0,
                        engagement_score: 0.0,
                        freshness_score: 0.0,
                        authority_score: 0.0,
                        relevance_score: 0.0,
                        originality_score: 0.0,
                        completeness_score: 0.0,
                        readability_score: 0.0,
                        multimedia_score: 0.0,
                        components: content.components,
                        issues: vec![QualityIssue {
                            issue_type: IssueType::ProcessingError,
                            severity: IssueSeverity::Critical,
                            description: format!("Failed to score content: {}", e),
                            impact_on_score: -100.0,
                            can_be_fixed: true,
                            fix_suggestion: Some("Retry scoring after fixing the error".to_string()),
                        }],
                        recommendations: vec!["Fix scoring error and retry".to_string()],
                        timestamp: Utc::now(),
                    });
                }
            }
        }

        Ok(scores)
    }

    pub fn adjust_weights(&mut self, adjustments: HashMap<String, f32>) {
        for (component, weight) in adjustments {
            match component.as_str() {
                "technical" => self.weights.technical = weight.max(0.0).min(1.0),
                "content" => self.weights.content = weight.max(0.0).min(1.0),
                "engagement" => self.weights.engagement = weight.max(0.0).min(1.0),
                "freshness" => self.weights.freshness = weight.max(0.0).min(1.0),
                "authority" => self.weights.authority = weight.max(0.0).min(1.0),
                "relevance" => self.weights.relevance = weight.max(0.0).min(1.0),
                "originality" => self.weights.originality = weight.max(0.0).min(1.0),
                "completeness" => self.weights.completeness = weight.max(0.0).min(1.0),
                "readability" => self.weights.readability = weight.max(0.0).min(1.0),
                "multimedia" => self.weights.multimedia = weight.max(0.0).min(1.0),
                _ => warn!("Unknown weight component: {}", component),
            }
        }

        // Normalize weights to sum to 1.0
        let total = self.weights.technical + self.weights.content + self.weights.engagement +
                   self.weights.freshness + self.weights.authority + self.weights.relevance +
                   self.weights.originality + self.weights.completeness + self.weights.readability +
                   self.weights.multimedia;

        if total > 0.0 {
            self.weights.technical /= total;
            self.weights.content /= total;
            self.weights.engagement /= total;
            self.weights.freshness /= total;
            self.weights.authority /= total;
            self.weights.relevance /= total;
            self.weights.originality /= total;
            self.weights.completeness /= total;
            self.weights.readability /= total;
            self.weights.multimedia /= total;
        }
    }
}