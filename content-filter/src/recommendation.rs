use anyhow::{Context, Result};
use dashmap::DashMap;
use linfa::prelude::*;
use linfa_clustering::KMeans;
use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentRecommendation {
    pub user_id: String,
    pub recommended_items: Vec<RecommendedItem>,
    pub explanation: String,
    pub confidence_score: f32,
    pub recommendation_type: RecommendationType,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendedItem {
    pub content_id: String,
    pub title: String,
    pub score: f32,
    pub reasons: Vec<RecommendationReason>,
    pub metadata: ContentMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendationReason {
    pub reason_type: ReasonType,
    pub weight: f32,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReasonType {
    Similar,
    Trending,
    PersonalPreference,
    Collaborative,
    Popular,
    New,
    Diverse,
    Serendipitous,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationType {
    ContentBased,
    CollaborativeFiltering,
    Hybrid,
    Trending,
    Personalized,
    ColdStart,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentMetadata {
    pub category: String,
    pub tags: Vec<String>,
    pub duration_seconds: Option<u32>,
    pub quality_score: f32,
    pub engagement_score: f32,
    pub freshness_days: u32,
    pub language: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    pub user_id: String,
    pub interests: Vec<Interest>,
    pub viewing_history: VecDeque<ViewingEvent>,
    pub preferences: UserPreferences,
    pub feature_vector: Array1<f32>,
    pub cluster_id: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Interest {
    pub category: String,
    pub weight: f32,
    pub keywords: Vec<String>,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewingEvent {
    pub content_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub duration_watched: u32,
    pub completed: bool,
    pub rating: Option<f32>,
    pub engagement_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferences {
    pub preferred_categories: Vec<String>,
    pub blocked_categories: Vec<String>,
    pub preferred_duration_range: (u32, u32),
    pub quality_threshold: f32,
    pub diversity_factor: f32,
    pub novelty_factor: f32,
}

impl Default for UserPreferences {
    fn default() -> Self {
        Self {
            preferred_categories: Vec::new(),
            blocked_categories: Vec::new(),
            preferred_duration_range: (300, 3600),  // 5 min to 1 hour
            quality_threshold: 0.6,
            diversity_factor: 0.3,
            novelty_factor: 0.2,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RecommendationEngine {
    // User profiles
    user_profiles: Arc<DashMap<String, UserProfile>>,

    // Content index
    content_index: Arc<DashMap<String, ContentItem>>,
    content_features: Arc<RwLock<HashMap<String, Array1<f32>>>>,

    // Similarity matrices
    user_similarity: Arc<RwLock<HashMap<(String, String), f32>>>,
    content_similarity: Arc<RwLock<HashMap<(String, String), f32>>>,

    // Clustering
    user_clusters: Arc<RwLock<Vec<Vec<String>>>>,
    content_clusters: Arc<RwLock<Vec<Vec<String>>>>,

    // Trending and popular items
    trending_items: Arc<RwLock<Vec<String>>>,
    popular_items: Arc<RwLock<Vec<String>>>,

    // Cache
    cache: Arc<DashMap<String, ContentRecommendation>>,

    // Configuration
    config: RecommendationConfig,

    // Statistics
    statistics: Arc<RwLock<RecommendationStatistics>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendationConfig {
    pub max_recommendations: usize,
    pub min_similarity_threshold: f32,
    pub collaborative_weight: f32,
    pub content_based_weight: f32,
    pub trending_weight: f32,
    pub diversity_penalty: f32,
    pub recency_boost: f32,
    pub cache_ttl_seconds: u64,
    pub enable_clustering: bool,
    pub num_clusters: usize,
}

impl Default for RecommendationConfig {
    fn default() -> Self {
        Self {
            max_recommendations: 20,
            min_similarity_threshold: 0.3,
            collaborative_weight: 0.4,
            content_based_weight: 0.4,
            trending_weight: 0.2,
            diversity_penalty: 0.1,
            recency_boost: 0.15,
            cache_ttl_seconds: 3600,
            enable_clustering: true,
            num_clusters: 10,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentItem {
    pub content_id: String,
    pub title: String,
    pub metadata: ContentMetadata,
    pub feature_vector: Array1<f32>,
    pub view_count: u64,
    pub average_rating: f32,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct RecommendationStatistics {
    pub total_recommendations: u64,
    pub click_through_rate: f32,
    pub average_relevance_score: f32,
    pub diversity_score: f32,
    pub coverage_percentage: f32,
    pub recommendation_type_distribution: HashMap<String, u64>,
    pub average_processing_time_ms: f64,
}

impl RecommendationEngine {
    pub async fn new(config: RecommendationConfig) -> Result<Self> {
        Ok(Self {
            user_profiles: Arc::new(DashMap::new()),
            content_index: Arc::new(DashMap::new()),
            content_features: Arc::new(RwLock::new(HashMap::new())),
            user_similarity: Arc::new(RwLock::new(HashMap::new())),
            content_similarity: Arc::new(RwLock::new(HashMap::new())),
            user_clusters: Arc::new(RwLock::new(Vec::new())),
            content_clusters: Arc::new(RwLock::new(Vec::new())),
            trending_items: Arc::new(RwLock::new(Vec::new())),
            popular_items: Arc::new(RwLock::new(Vec::new())),
            cache: Arc::new(DashMap::new()),
            config,
            statistics: Arc::new(RwLock::new(RecommendationStatistics::default())),
        })
    }

    pub async fn get_recommendations(&self, user_id: &str) -> Result<ContentRecommendation> {
        // Check cache
        if let Some(cached) = self.get_cached_recommendations(user_id).await {
            return Ok(cached);
        }

        let start = std::time::Instant::now();

        // Get or create user profile
        let profile = self.get_or_create_user_profile(user_id).await?;

        // Determine recommendation strategy
        let recommendation_type = self.determine_recommendation_type(&profile);

        let recommended_items = match recommendation_type {
            RecommendationType::ContentBased => {
                self.get_content_based_recommendations(&profile).await?
            }
            RecommendationType::CollaborativeFiltering => {
                self.get_collaborative_recommendations(&profile).await?
            }
            RecommendationType::Hybrid => {
                self.get_hybrid_recommendations(&profile).await?
            }
            RecommendationType::Trending => {
                self.get_trending_recommendations(&profile).await?
            }
            RecommendationType::ColdStart => {
                self.get_cold_start_recommendations(&profile).await?
            }
            RecommendationType::Personalized => {
                self.get_personalized_recommendations(&profile).await?
            }
        };

        // Apply diversity
        let diverse_items = self.apply_diversity(&recommended_items, &profile).await?;

        // Generate explanation
        let explanation = self.generate_explanation(&recommendation_type, &diverse_items);

        // Calculate confidence score
        let confidence_score = self.calculate_confidence(&diverse_items);

        let recommendation = ContentRecommendation {
            user_id: user_id.to_string(),
            recommended_items: diverse_items,
            explanation,
            confidence_score,
            recommendation_type,
            timestamp: chrono::Utc::now(),
        };

        // Update statistics
        self.update_statistics(&recommendation, start.elapsed()).await;

        // Cache recommendations
        self.cache_recommendations(&recommendation).await;

        Ok(recommendation)
    }

    async fn get_content_based_recommendations(
        &self,
        profile: &UserProfile,
    ) -> Result<Vec<RecommendedItem>> {
        let mut recommendations = Vec::new();
        let content_features = self.content_features.read().await;
        let content_similarity = self.content_similarity.read().await;

        // Get recent viewing history
        let recent_items: Vec<_> = profile.viewing_history
            .iter()
            .take(10)
            .map(|e| e.content_id.clone())
            .collect();

        // Find similar content for each recently viewed item
        for recent_id in &recent_items {
            if let Some(recent_features) = content_features.get(recent_id) {
                for (content_id, features) in content_features.iter() {
                    if recent_items.contains(content_id) {
                        continue;
                    }

                    let similarity = self.cosine_similarity(recent_features, features);

                    if similarity > self.config.min_similarity_threshold {
                        if let Some(content) = self.content_index.get(content_id) {
                            recommendations.push(RecommendedItem {
                                content_id: content_id.clone(),
                                title: content.title.clone(),
                                score: similarity,
                                reasons: vec![RecommendationReason {
                                    reason_type: ReasonType::Similar,
                                    weight: similarity,
                                    description: format!("Similar to content you watched"),
                                }],
                                metadata: content.metadata.clone(),
                            });
                        }
                    }
                }
            }
        }

        // Sort and limit recommendations
        recommendations.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        recommendations.truncate(self.config.max_recommendations);

        Ok(recommendations)
    }

    async fn get_collaborative_recommendations(
        &self,
        profile: &UserProfile,
    ) -> Result<Vec<RecommendedItem>> {
        let mut recommendations = Vec::new();
        let user_similarity = self.user_similarity.read().await;

        // Find similar users
        let mut similar_users = Vec::new();
        for ((user1, user2), similarity) in user_similarity.iter() {
            if user1 == &profile.user_id {
                similar_users.push((user2.clone(), *similarity));
            } else if user2 == &profile.user_id {
                similar_users.push((user1.clone(), *similarity));
            }
        }

        similar_users.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similar_users.truncate(10);

        // Get items liked by similar users
        let mut item_scores: HashMap<String, f32> = HashMap::new();

        for (similar_user_id, similarity) in similar_users {
            if let Some(similar_profile) = self.user_profiles.get(&similar_user_id) {
                for event in &similar_profile.viewing_history {
                    if event.completed || event.rating.unwrap_or(0.0) > 3.5 {
                        let score = similarity * event.engagement_score;
                        *item_scores.entry(event.content_id.clone()).or_insert(0.0) += score;
                    }
                }
            }
        }

        // Filter out items already viewed
        let viewed_items: HashSet<_> = profile.viewing_history
            .iter()
            .map(|e| e.content_id.clone())
            .collect();

        for (content_id, score) in item_scores {
            if !viewed_items.contains(&content_id) {
                if let Some(content) = self.content_index.get(&content_id) {
                    recommendations.push(RecommendedItem {
                        content_id: content_id.clone(),
                        title: content.title.clone(),
                        score,
                        reasons: vec![RecommendationReason {
                            reason_type: ReasonType::Collaborative,
                            weight: score,
                            description: "Users like you also watched this".to_string(),
                        }],
                        metadata: content.metadata.clone(),
                    });
                }
            }
        }

        recommendations.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        recommendations.truncate(self.config.max_recommendations);

        Ok(recommendations)
    }

    async fn get_hybrid_recommendations(
        &self,
        profile: &UserProfile,
    ) -> Result<Vec<RecommendedItem>> {
        // Combine content-based and collaborative filtering
        let content_based = self.get_content_based_recommendations(profile).await?;
        let collaborative = self.get_collaborative_recommendations(profile).await?;
        let trending = self.get_trending_recommendations(profile).await?;

        let mut combined_scores: HashMap<String, (f32, Vec<RecommendationReason>)> = HashMap::new();

        // Add content-based scores
        for item in content_based {
            let weighted_score = item.score * self.config.content_based_weight;
            combined_scores.entry(item.content_id.clone())
                .or_insert((0.0, Vec::new()))
                .0 += weighted_score;
            combined_scores.get_mut(&item.content_id).unwrap().1.extend(item.reasons);
        }

        // Add collaborative scores
        for item in collaborative {
            let weighted_score = item.score * self.config.collaborative_weight;
            combined_scores.entry(item.content_id.clone())
                .or_insert((0.0, Vec::new()))
                .0 += weighted_score;
            combined_scores.get_mut(&item.content_id).unwrap().1.extend(item.reasons);
        }

        // Add trending scores
        for item in trending {
            let weighted_score = item.score * self.config.trending_weight;
            combined_scores.entry(item.content_id.clone())
                .or_insert((0.0, Vec::new()))
                .0 += weighted_score;
            combined_scores.get_mut(&item.content_id).unwrap().1.extend(item.reasons);
        }

        // Create final recommendations
        let mut recommendations = Vec::new();
        for (content_id, (score, reasons)) in combined_scores {
            if let Some(content) = self.content_index.get(&content_id) {
                recommendations.push(RecommendedItem {
                    content_id,
                    title: content.title.clone(),
                    score,
                    reasons,
                    metadata: content.metadata.clone(),
                });
            }
        }

        recommendations.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        recommendations.truncate(self.config.max_recommendations);

        Ok(recommendations)
    }

    async fn get_trending_recommendations(
        &self,
        profile: &UserProfile,
    ) -> Result<Vec<RecommendedItem>> {
        let mut recommendations = Vec::new();
        let trending = self.trending_items.read().await;

        for (i, content_id) in trending.iter().enumerate() {
            if let Some(content) = self.content_index.get(content_id) {
                // Check if category matches preferences
                if !profile.preferences.blocked_categories.contains(&content.metadata.category) {
                    let score = 1.0 - (i as f32 / trending.len() as f32);

                    recommendations.push(RecommendedItem {
                        content_id: content_id.clone(),
                        title: content.title.clone(),
                        score,
                        reasons: vec![RecommendationReason {
                            reason_type: ReasonType::Trending,
                            weight: score,
                            description: "Currently trending".to_string(),
                        }],
                        metadata: content.metadata.clone(),
                    });
                }
            }
        }

        recommendations.truncate(self.config.max_recommendations);
        Ok(recommendations)
    }

    async fn get_cold_start_recommendations(
        &self,
        profile: &UserProfile,
    ) -> Result<Vec<RecommendedItem>> {
        // For new users, recommend popular and diverse content
        let mut recommendations = Vec::new();
        let popular = self.popular_items.read().await;

        // Get popular items from different categories
        let mut category_items: HashMap<String, Vec<RecommendedItem>> = HashMap::new();

        for content_id in popular.iter() {
            if let Some(content) = self.content_index.get(content_id) {
                let item = RecommendedItem {
                    content_id: content_id.clone(),
                    title: content.title.clone(),
                    score: content.average_rating * 0.5 + (content.view_count as f32 / 1000.0).min(0.5),
                    reasons: vec![RecommendationReason {
                        reason_type: ReasonType::Popular,
                        weight: 0.8,
                        description: "Popular content to get you started".to_string(),
                    }],
                    metadata: content.metadata.clone(),
                };

                category_items.entry(content.metadata.category.clone())
                    .or_insert_with(Vec::new)
                    .push(item);
            }
        }

        // Take top items from each category for diversity
        for (_, items) in category_items {
            recommendations.extend(items.into_iter().take(3));
        }

        recommendations.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        recommendations.truncate(self.config.max_recommendations);

        Ok(recommendations)
    }

    async fn get_personalized_recommendations(
        &self,
        profile: &UserProfile,
    ) -> Result<Vec<RecommendedItem>> {
        // Deep personalization based on user profile
        let mut recommendations = Vec::new();

        // Get items matching user interests
        for interest in &profile.interests {
            for (content_id, content) in self.content_index.iter() {
                // Check if content matches interest
                let interest_match = content.metadata.tags
                    .iter()
                    .any(|tag| interest.keywords.contains(tag));

                if interest_match {
                    let base_score = interest.weight;

                    // Apply quality threshold
                    if content.metadata.quality_score < profile.preferences.quality_threshold {
                        continue;
                    }

                    // Apply duration preference
                    if let Some(duration) = content.metadata.duration_seconds {
                        let (min_dur, max_dur) = profile.preferences.preferred_duration_range;
                        if duration < min_dur || duration > max_dur {
                            continue;
                        }
                    }

                    // Calculate final score with recency boost
                    let recency_factor = 1.0 / (content.metadata.freshness_days as f32 + 1.0);
                    let score = base_score + recency_factor * self.config.recency_boost;

                    recommendations.push(RecommendedItem {
                        content_id: content_id.clone(),
                        title: content.title.clone(),
                        score,
                        reasons: vec![RecommendationReason {
                            reason_type: ReasonType::PersonalPreference,
                            weight: score,
                            description: format!("Matches your interest in {}", interest.category),
                        }],
                        metadata: content.metadata.clone(),
                    });
                }
            }
        }

        recommendations.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        recommendations.truncate(self.config.max_recommendations * 2);  // Get extra for diversity

        Ok(recommendations)
    }

    async fn apply_diversity(
        &self,
        items: &[RecommendedItem],
        profile: &UserProfile,
    ) -> Result<Vec<RecommendedItem>> {
        if items.len() <= self.config.max_recommendations {
            return Ok(items.to_vec());
        }

        let diversity_factor = profile.preferences.diversity_factor;
        let mut selected = Vec::new();
        let mut category_counts: HashMap<String, usize> = HashMap::new();

        for item in items {
            let category = &item.metadata.category;
            let category_count = category_counts.get(category).unwrap_or(&0);

            // Apply diversity penalty based on how many items from this category are already selected
            let diversity_penalty = (*category_count as f32) * self.config.diversity_penalty;
            let adjusted_score = item.score - diversity_penalty;

            if adjusted_score > 0.0 && selected.len() < self.config.max_recommendations {
                selected.push(item.clone());
                *category_counts.entry(category.clone()).or_insert(0) += 1;
            }
        }

        // Add serendipitous items if needed
        if profile.preferences.novelty_factor > 0.0 && selected.len() < self.config.max_recommendations {
            let serendipitous = self.get_serendipitous_items(items, &selected).await?;
            selected.extend(serendipitous);
        }

        selected.truncate(self.config.max_recommendations);
        Ok(selected)
    }

    async fn get_serendipitous_items(
        &self,
        all_items: &[RecommendedItem],
        selected: &[RecommendedItem],
    ) -> Result<Vec<RecommendedItem>> {
        let selected_categories: HashSet<_> = selected
            .iter()
            .map(|i| i.metadata.category.clone())
            .collect();

        let mut serendipitous = Vec::new();

        for item in all_items {
            if !selected_categories.contains(&item.metadata.category) {
                let mut item_copy = item.clone();
                item_copy.reasons.push(RecommendationReason {
                    reason_type: ReasonType::Serendipitous,
                    weight: 0.3,
                    description: "Something different you might enjoy".to_string(),
                });
                serendipitous.push(item_copy);

                if serendipitous.len() >= 3 {
                    break;
                }
            }
        }

        Ok(serendipitous)
    }

    fn determine_recommendation_type(&self, profile: &UserProfile) -> RecommendationType {
        if profile.viewing_history.len() < 5 {
            RecommendationType::ColdStart
        } else if profile.viewing_history.len() < 20 {
            RecommendationType::Hybrid
        } else {
            RecommendationType::Personalized
        }
    }

    fn cosine_similarity(&self, a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        let dot_product = a.dot(b);
        let norm_a = a.dot(a).sqrt();
        let norm_b = b.dot(b).sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }

    fn generate_explanation(
        &self,
        rec_type: &RecommendationType,
        items: &[RecommendedItem],
    ) -> String {
        match rec_type {
            RecommendationType::ContentBased => {
                "Based on content similar to what you've watched before".to_string()
            }
            RecommendationType::CollaborativeFiltering => {
                "Based on what users with similar preferences enjoyed".to_string()
            }
            RecommendationType::Hybrid => {
                "A personalized mix based on your viewing history and preferences".to_string()
            }
            RecommendationType::Trending => {
                "Currently popular and trending content".to_string()
            }
            RecommendationType::ColdStart => {
                "Popular content to help us learn your preferences".to_string()
            }
            RecommendationType::Personalized => {
                format!("Specially curated based on your interests in {}",
                    items.iter()
                        .flat_map(|i| i.reasons.iter())
                        .filter(|r| r.reason_type == ReasonType::PersonalPreference)
                        .take(3)
                        .map(|r| &r.description)
                        .collect::<Vec<_>>()
                        .join(", "))
            }
        }
    }

    fn calculate_confidence(&self, items: &[RecommendedItem]) -> f32 {
        if items.is_empty() {
            return 0.0;
        }

        let avg_score = items.iter().map(|i| i.score).sum::<f32>() / items.len() as f32;
        let has_multiple_reasons = items.iter().any(|i| i.reasons.len() > 1);

        let confidence = avg_score * if has_multiple_reasons { 1.2 } else { 1.0 };
        confidence.min(1.0)
    }

    async fn get_or_create_user_profile(&self, user_id: &str) -> Result<UserProfile> {
        if let Some(profile) = self.user_profiles.get(user_id) {
            Ok(profile.clone())
        } else {
            let profile = UserProfile {
                user_id: user_id.to_string(),
                interests: Vec::new(),
                viewing_history: VecDeque::new(),
                preferences: UserPreferences::default(),
                feature_vector: Array1::zeros(128),
                cluster_id: None,
            };

            self.user_profiles.insert(user_id.to_string(), profile.clone());
            Ok(profile)
        }
    }

    async fn get_cached_recommendations(&self, user_id: &str) -> Option<ContentRecommendation> {
        self.cache.get(user_id).map(|entry| entry.clone())
    }

    async fn cache_recommendations(&self, recommendation: &ContentRecommendation) {
        self.cache.insert(recommendation.user_id.clone(), recommendation.clone());
    }

    async fn update_statistics(&self, recommendation: &ContentRecommendation, duration: std::time::Duration) {
        let mut stats = self.statistics.write().await;

        stats.total_recommendations += 1;

        let rec_type = format!("{:?}", recommendation.recommendation_type);
        *stats.recommendation_type_distribution.entry(rec_type).or_insert(0) += 1;

        // Update average processing time
        let current_avg = stats.average_processing_time_ms;
        let current_count = stats.total_recommendations as f64;
        stats.average_processing_time_ms =
            (current_avg * (current_count - 1.0) + duration.as_millis() as f64) / current_count;

        // Update average relevance score
        let avg_score = recommendation.recommended_items
            .iter()
            .map(|i| i.score)
            .sum::<f32>() / recommendation.recommended_items.len().max(1) as f32;

        stats.average_relevance_score =
            (stats.average_relevance_score * (current_count - 1.0) as f32 + avg_score) / current_count as f32;
    }

    pub async fn update_user_profile(&self, user_id: &str, event: ViewingEvent) -> Result<()> {
        let mut profile = self.get_or_create_user_profile(user_id).await?;

        // Add to viewing history
        profile.viewing_history.push_front(event.clone());
        if profile.viewing_history.len() > 100 {
            profile.viewing_history.pop_back();
        }

        // Update interests based on viewing patterns
        self.update_user_interests(&mut profile).await?;

        // Update feature vector
        self.update_user_feature_vector(&mut profile).await?;

        self.user_profiles.insert(user_id.to_string(), profile);

        // Clear cache for this user
        self.cache.remove(user_id);

        Ok(())
    }

    async fn update_user_interests(&self, profile: &mut UserProfile) -> Result<()> {
        // Analyze viewing history to extract interests
        let mut category_counts: HashMap<String, f32> = HashMap::new();

        for event in &profile.viewing_history {
            if let Some(content) = self.content_index.get(&event.content_id) {
                let weight = if event.completed { 1.0 } else { 0.5 };
                let score = weight * event.engagement_score;

                *category_counts.entry(content.metadata.category.clone()).or_insert(0.0) += score;
            }
        }

        // Update interests
        profile.interests.clear();
        for (category, weight) in category_counts {
            profile.interests.push(Interest {
                category: category.clone(),
                weight: weight / profile.viewing_history.len() as f32,
                keywords: Vec::new(),  // Would be populated from content analysis
                last_updated: chrono::Utc::now(),
            });
        }

        Ok(())
    }

    async fn update_user_feature_vector(&self, profile: &mut UserProfile) -> Result<()> {
        // Create feature vector from viewing history
        let mut feature_vector = Array1::zeros(128);
        let content_features = self.content_features.read().await;

        for event in &profile.viewing_history {
            if let Some(features) = content_features.get(&event.content_id) {
                feature_vector = feature_vector + features * event.engagement_score;
            }
        }

        if profile.viewing_history.len() > 0 {
            feature_vector = feature_vector / profile.viewing_history.len() as f32;
        }

        profile.feature_vector = feature_vector;

        Ok(())
    }

    pub async fn add_content(&self, content: ContentItem) -> Result<()> {
        let content_id = content.content_id.clone();
        let feature_vector = content.feature_vector.clone();

        self.content_index.insert(content_id.clone(), content);
        self.content_features.write().await.insert(content_id, feature_vector);

        // Update trending/popular if needed
        self.update_trending_items().await?;

        Ok(())
    }

    async fn update_trending_items(&self) -> Result<()> {
        // Simple trending algorithm based on recent views and ratings
        let mut item_scores: Vec<(String, f32)> = Vec::new();

        for entry in self.content_index.iter() {
            let content = entry.value();
            let recency_factor = 1.0 / (content.metadata.freshness_days as f32 + 1.0);
            let score = (content.view_count as f32).log10() * content.average_rating * recency_factor;
            item_scores.push((content.content_id.clone(), score));
        }

        item_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut trending = self.trending_items.write().await;
        trending.clear();
        trending.extend(item_scores.iter().take(100).map(|(id, _)| id.clone()));

        Ok(())
    }

    pub async fn get_statistics(&self) -> RecommendationStatistics {
        self.statistics.read().await.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_recommendation_engine() {
        let config = RecommendationConfig::default();
        let engine = RecommendationEngine::new(config).await.unwrap();

        // Add some test content
        for i in 0..10 {
            let content = ContentItem {
                content_id: format!("content_{}", i),
                title: format!("Content {}", i),
                metadata: ContentMetadata {
                    category: if i % 2 == 0 { "Tech" } else { "Science" }.to_string(),
                    tags: vec![format!("tag_{}", i)],
                    duration_seconds: Some(600),
                    quality_score: 0.8,
                    engagement_score: 0.7,
                    freshness_days: i as u32,
                    language: "English".to_string(),
                },
                feature_vector: Array1::from_elem(128, i as f32 / 10.0),
                view_count: (100 - i * 10) as u64,
                average_rating: 4.0 - (i as f32 / 10.0),
                created_at: chrono::Utc::now(),
            };
            engine.add_content(content).await.unwrap();
        }

        // Get recommendations for a new user
        let recommendations = engine.get_recommendations("user_123").await.unwrap();

        assert!(!recommendations.recommended_items.is_empty());
        assert_eq!(recommendations.recommendation_type, RecommendationType::ColdStart);
    }
}