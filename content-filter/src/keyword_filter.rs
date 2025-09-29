use anyhow::{Context, Result};
use dashmap::DashMap;
use regex::Regex;
use rust_stemmers::{Algorithm, Stemmer};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use stop_words::{Spark, LANGUAGE};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeywordFilterResult {
    pub content_id: String,
    pub allowed: bool,
    pub matched_rules: Vec<MatchedRule>,
    pub topics: Vec<DetectedTopic>,
    pub sentiment: SentimentScore,
    pub language: String,
    pub keywords: Vec<ExtractedKeyword>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchedRule {
    pub rule_id: Uuid,
    pub rule_name: String,
    pub rule_type: RuleType,
    pub action: FilterAction,
    pub matched_terms: Vec<String>,
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuleType {
    Blocklist,
    Allowlist,
    Topic,
    Sentiment,
    Language,
    Custom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FilterAction {
    Allow,
    Block,
    Review,
    Tag,
    Notify,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedTopic {
    pub topic: String,
    pub confidence: f32,
    pub keywords: Vec<String>,
    pub category: TopicCategory,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopicCategory {
    Technology,
    Science,
    Politics,
    Business,
    Entertainment,
    Sports,
    Health,
    Education,
    Gaming,
    Lifestyle,
    News,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentScore {
    pub positive: f32,
    pub negative: f32,
    pub neutral: f32,
    pub overall: f32,  // -1.0 to 1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedKeyword {
    pub term: String,
    pub frequency: u32,
    pub tfidf_score: f32,
    pub is_entity: bool,
    pub entity_type: Option<String>,
}

#[derive(Debug, Clone)]
pub struct KeywordFilter {
    // Filter rules
    blocklist: Arc<RwLock<HashSet<String>>>,
    allowlist: Arc<RwLock<HashSet<String>>>,
    topic_filters: Arc<RwLock<Vec<TopicFilter>>>,
    custom_rules: Arc<RwLock<Vec<CustomRule>>>,

    // NLP components
    stemmer: Stemmer,
    stop_words: HashSet<String>,
    topic_classifiers: Arc<HashMap<TopicCategory, Vec<String>>>,

    // Caching
    cache: Arc<DashMap<String, KeywordFilterResult>>,

    // Configuration
    config: KeywordFilterConfig,

    // Statistics
    statistics: Arc<RwLock<FilterStatistics>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeywordFilterConfig {
    pub enable_stemming: bool,
    pub remove_stop_words: bool,
    pub min_keyword_length: usize,
    pub max_keywords: usize,
    pub sentiment_threshold: f32,
    pub topic_confidence_threshold: f32,
    pub cache_ttl_seconds: u64,
    pub case_sensitive: bool,
    pub fuzzy_matching: bool,
    pub fuzzy_distance: usize,
}

impl Default for KeywordFilterConfig {
    fn default() -> Self {
        Self {
            enable_stemming: true,
            remove_stop_words: true,
            min_keyword_length: 3,
            max_keywords: 50,
            sentiment_threshold: -0.7,
            topic_confidence_threshold: 0.6,
            cache_ttl_seconds: 3600,
            case_sensitive: false,
            fuzzy_matching: true,
            fuzzy_distance: 2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicFilter {
    pub id: Uuid,
    pub name: String,
    pub category: TopicCategory,
    pub keywords: Vec<String>,
    pub action: FilterAction,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomRule {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    pub pattern: Option<Regex>,
    pub keywords: Vec<String>,
    pub action: FilterAction,
    pub priority: i32,
    pub enabled: bool,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct FilterStatistics {
    pub total_filtered: u64,
    pub blocked_count: u64,
    pub allowed_count: u64,
    pub reviewed_count: u64,
    pub topic_distribution: HashMap<String, u64>,
    pub language_distribution: HashMap<String, u64>,
    pub average_processing_time_ms: f64,
}

impl KeywordFilter {
    pub async fn new(config: KeywordFilterConfig) -> Result<Self> {
        let stemmer = Stemmer::create(Algorithm::English);
        let stop_words = stop_words::get(LANGUAGE::English);

        Ok(Self {
            blocklist: Arc::new(RwLock::new(Self::load_default_blocklist())),
            allowlist: Arc::new(RwLock::new(Self::load_default_allowlist())),
            topic_filters: Arc::new(RwLock::new(Self::load_default_topic_filters())),
            custom_rules: Arc::new(RwLock::new(Vec::new())),
            stemmer,
            stop_words,
            topic_classifiers: Arc::new(Self::load_topic_classifiers()),
            cache: Arc::new(DashMap::new()),
            config,
            statistics: Arc::new(RwLock::new(FilterStatistics::default())),
        })
    }

    pub async fn filter_content(&self, content_id: &str, content: &str) -> Result<KeywordFilterResult> {
        // Check cache
        if let Some(cached) = self.get_cached_result(content_id).await {
            return Ok(cached);
        }

        let start = std::time::Instant::now();

        // Detect language
        let language = self.detect_language(content);

        // Extract keywords
        let keywords = self.extract_keywords(content)?;

        // Check filter rules
        let matched_rules = self.check_filter_rules(content, &keywords).await?;

        // Detect topics
        let topics = self.detect_topics(content, &keywords)?;

        // Analyze sentiment
        let sentiment = self.analyze_sentiment(content)?;

        // Determine if content is allowed
        let allowed = self.determine_action(&matched_rules, &sentiment);

        let result = KeywordFilterResult {
            content_id: content_id.to_string(),
            allowed,
            matched_rules,
            topics,
            sentiment,
            language,
            keywords,
            timestamp: chrono::Utc::now(),
        };

        // Update statistics
        self.update_statistics(&result, start.elapsed()).await;

        // Cache result
        self.cache_result(&result).await;

        Ok(result)
    }

    async fn check_filter_rules(
        &self,
        content: &str,
        keywords: &[ExtractedKeyword],
    ) -> Result<Vec<MatchedRule>> {
        let mut matched_rules = Vec::new();
        let content_lower = if !self.config.case_sensitive {
            content.to_lowercase()
        } else {
            content.to_string()
        };

        // Check blocklist
        {
            let blocklist = self.blocklist.read().await;
            let mut matched_terms = Vec::new();

            for blocked_term in blocklist.iter() {
                let term = if !self.config.case_sensitive {
                    blocked_term.to_lowercase()
                } else {
                    blocked_term.clone()
                };

                if content_lower.contains(&term) {
                    matched_terms.push(blocked_term.clone());
                } else if self.config.fuzzy_matching {
                    // Check keywords for fuzzy match
                    for keyword in keywords {
                        if levenshtein::levenshtein(&keyword.term, blocked_term) <= self.config.fuzzy_distance {
                            matched_terms.push(blocked_term.clone());
                            break;
                        }
                    }
                }
            }

            if !matched_terms.is_empty() {
                matched_rules.push(MatchedRule {
                    rule_id: Uuid::new_v4(),
                    rule_name: "Blocklist".to_string(),
                    rule_type: RuleType::Blocklist,
                    action: FilterAction::Block,
                    matched_terms,
                    confidence: 1.0,
                });
            }
        }

        // Check allowlist
        {
            let allowlist = self.allowlist.read().await;
            let mut matched_terms = Vec::new();

            for allowed_term in allowlist.iter() {
                let term = if !self.config.case_sensitive {
                    allowed_term.to_lowercase()
                } else {
                    allowed_term.clone()
                };

                if content_lower.contains(&term) {
                    matched_terms.push(allowed_term.clone());
                }
            }

            if !matched_terms.is_empty() {
                matched_rules.push(MatchedRule {
                    rule_id: Uuid::new_v4(),
                    rule_name: "Allowlist".to_string(),
                    rule_type: RuleType::Allowlist,
                    action: FilterAction::Allow,
                    matched_terms,
                    confidence: 1.0,
                });
            }
        }

        // Check custom rules
        {
            let custom_rules = self.custom_rules.read().await;
            let mut rules_with_priority: Vec<_> = custom_rules
                .iter()
                .filter(|r| r.enabled)
                .collect();

            rules_with_priority.sort_by_key(|r| -r.priority);

            for rule in rules_with_priority {
                let mut matched = false;
                let mut matched_terms = Vec::new();

                // Check pattern
                if let Some(pattern) = &rule.pattern {
                    if pattern.is_match(&content_lower) {
                        matched = true;
                        if let Some(captures) = pattern.find(&content_lower) {
                            matched_terms.push(captures.as_str().to_string());
                        }
                    }
                }

                // Check keywords
                for keyword in &rule.keywords {
                    let kw = if !self.config.case_sensitive {
                        keyword.to_lowercase()
                    } else {
                        keyword.clone()
                    };

                    if content_lower.contains(&kw) {
                        matched = true;
                        matched_terms.push(keyword.clone());
                    }
                }

                if matched {
                    matched_rules.push(MatchedRule {
                        rule_id: rule.id,
                        rule_name: rule.name.clone(),
                        rule_type: RuleType::Custom,
                        action: rule.action,
                        matched_terms,
                        confidence: 0.9,
                    });
                }
            }
        }

        // Check topic filters
        {
            let topic_filters = self.topic_filters.read().await;

            for filter in topic_filters.iter().filter(|f| f.enabled) {
                let mut score = 0.0;
                let mut matched_terms = Vec::new();

                for keyword in &filter.keywords {
                    let kw = if !self.config.case_sensitive {
                        keyword.to_lowercase()
                    } else {
                        keyword.clone()
                    };

                    if content_lower.contains(&kw) {
                        score += 1.0;
                        matched_terms.push(keyword.clone());
                    }
                }

                let confidence = score / filter.keywords.len() as f32;

                if confidence >= self.config.topic_confidence_threshold {
                    matched_rules.push(MatchedRule {
                        rule_id: filter.id,
                        rule_name: filter.name.clone(),
                        rule_type: RuleType::Topic,
                        action: filter.action,
                        matched_terms,
                        confidence,
                    });
                }
            }
        }

        Ok(matched_rules)
    }

    fn extract_keywords(&self, content: &str) -> Result<Vec<ExtractedKeyword>> {
        let mut word_freq: HashMap<String, u32> = HashMap::new();
        let mut keywords = Vec::new();

        // Tokenize and count words
        let words: Vec<&str> = content.split_whitespace().collect();
        let total_words = words.len() as f32;

        for word in words {
            let cleaned = word.chars()
                .filter(|c| c.is_alphabetic() || c.is_numeric())
                .collect::<String>()
                .to_lowercase();

            if cleaned.len() < self.config.min_keyword_length {
                continue;
            }

            if self.config.remove_stop_words && self.stop_words.contains(&cleaned) {
                continue;
            }

            let stemmed = if self.config.enable_stemming {
                self.stemmer.stem(&cleaned).to_string()
            } else {
                cleaned
            };

            *word_freq.entry(stemmed).or_insert(0) += 1;
        }

        // Calculate TF-IDF scores (simplified)
        for (term, freq) in word_freq.iter() {
            let tf = *freq as f32 / total_words;
            // Simplified IDF (in production, use corpus statistics)
            let idf = (10000.0 / (1.0 + *freq as f32)).ln();
            let tfidf_score = tf * idf;

            keywords.push(ExtractedKeyword {
                term: term.clone(),
                frequency: *freq,
                tfidf_score,
                is_entity: self.is_likely_entity(term),
                entity_type: self.detect_entity_type(term),
            });
        }

        // Sort by TF-IDF score and limit
        keywords.sort_by(|a, b| b.tfidf_score.partial_cmp(&a.tfidf_score).unwrap());
        keywords.truncate(self.config.max_keywords);

        Ok(keywords)
    }

    fn detect_topics(
        &self,
        content: &str,
        keywords: &[ExtractedKeyword],
    ) -> Result<Vec<DetectedTopic>> {
        let mut topics = Vec::new();
        let content_lower = content.to_lowercase();

        for (category, category_keywords) in self.topic_classifiers.iter() {
            let mut score = 0.0;
            let mut matched_keywords = Vec::new();

            for category_keyword in category_keywords {
                if content_lower.contains(category_keyword) {
                    score += 1.0;
                    matched_keywords.push(category_keyword.clone());
                }

                // Check extracted keywords
                for keyword in keywords {
                    if keyword.term.contains(category_keyword) || category_keyword.contains(&keyword.term) {
                        score += keyword.tfidf_score;
                        if !matched_keywords.contains(&keyword.term) {
                            matched_keywords.push(keyword.term.clone());
                        }
                    }
                }
            }

            let confidence = (score / category_keywords.len() as f32).min(1.0);

            if confidence >= self.config.topic_confidence_threshold {
                topics.push(DetectedTopic {
                    topic: format!("{:?}", category),
                    confidence,
                    keywords: matched_keywords,
                    category: category.clone(),
                });
            }
        }

        // Sort by confidence
        topics.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        Ok(topics)
    }

    fn analyze_sentiment(&self, content: &str) -> Result<SentimentScore> {
        // Simplified sentiment analysis using keyword approach
        let positive_words = vec![
            "good", "great", "excellent", "amazing", "wonderful", "fantastic",
            "love", "like", "enjoy", "happy", "positive", "best", "awesome",
            "brilliant", "outstanding", "perfect", "beautiful", "exciting",
        ];

        let negative_words = vec![
            "bad", "terrible", "awful", "horrible", "worst", "hate", "dislike",
            "angry", "sad", "negative", "poor", "disappointing", "frustrating",
            "annoying", "useless", "broken", "failed", "wrong", "evil",
        ];

        let content_lower = content.to_lowercase();
        let words: Vec<&str> = content_lower.split_whitespace().collect();
        let total_words = words.len() as f32;

        let mut positive_count = 0;
        let mut negative_count = 0;

        for word in &words {
            if positive_words.contains(word) {
                positive_count += 1;
            }
            if negative_words.contains(word) {
                negative_count += 1;
            }
        }

        // Check for negation patterns
        let negation_patterns = vec!["not", "no", "never", "neither", "nor", "n't"];
        for i in 0..words.len() {
            if negation_patterns.contains(&words[i]) {
                // Look at next 2 words for sentiment reversal
                for j in 1..=2.min(words.len() - i - 1) {
                    if positive_words.contains(&words[i + j]) {
                        positive_count -= 1;
                        negative_count += 1;
                    } else if negative_words.contains(&words[i + j]) {
                        negative_count -= 1;
                        positive_count += 1;
                    }
                }
            }
        }

        let positive = (positive_count as f32 / total_words).min(1.0);
        let negative = (negative_count as f32 / total_words).min(1.0);
        let neutral = 1.0 - positive - negative;

        let overall = positive - negative;  // Range: -1.0 to 1.0

        Ok(SentimentScore {
            positive,
            negative,
            neutral: neutral.max(0.0),
            overall,
        })
    }

    fn detect_language(&self, content: &str) -> String {
        // Use whatlang for language detection
        match whatlang::detect(content) {
            Some(info) => info.lang().eng_name().to_string(),
            None => "Unknown".to_string(),
        }
    }

    fn is_likely_entity(&self, term: &str) -> bool {
        // Simple heuristic: capitalized words are likely entities
        term.chars().next().map_or(false, |c| c.is_uppercase())
    }

    fn detect_entity_type(&self, term: &str) -> Option<String> {
        // Simplified entity type detection
        if term.chars().all(|c| c.is_numeric()) {
            Some("NUMBER".to_string())
        } else if term.contains('@') {
            Some("EMAIL".to_string())
        } else if term.starts_with("http") {
            Some("URL".to_string())
        } else if term.chars().next().map_or(false, |c| c.is_uppercase()) {
            Some("PERSON_OR_ORG".to_string())
        } else {
            None
        }
    }

    fn determine_action(&self, matched_rules: &[MatchedRule], sentiment: &SentimentScore) -> bool {
        // Check for explicit blocks
        for rule in matched_rules {
            if rule.action == FilterAction::Block {
                return false;
            }
        }

        // Check for explicit allows
        for rule in matched_rules {
            if rule.action == FilterAction::Allow && rule.rule_type == RuleType::Allowlist {
                return true;
            }
        }

        // Check sentiment threshold
        if sentiment.overall < self.config.sentiment_threshold {
            return false;
        }

        // Default to allow if no blocking rules matched
        true
    }

    async fn get_cached_result(&self, content_id: &str) -> Option<KeywordFilterResult> {
        self.cache.get(content_id).map(|entry| entry.clone())
    }

    async fn cache_result(&self, result: &KeywordFilterResult) {
        self.cache.insert(result.content_id.clone(), result.clone());

        // Clean old cache entries
        if self.cache.len() > 10000 {
            let to_remove: Vec<String> = self.cache
                .iter()
                .filter(|entry| {
                    let age = chrono::Utc::now().timestamp() - entry.value().timestamp.timestamp();
                    age > self.config.cache_ttl_seconds as i64
                })
                .map(|entry| entry.key().clone())
                .take(1000)
                .collect();

            for key in to_remove {
                self.cache.remove(&key);
            }
        }
    }

    async fn update_statistics(&self, result: &KeywordFilterResult, duration: std::time::Duration) {
        let mut stats = self.statistics.write().await;

        stats.total_filtered += 1;

        if result.allowed {
            stats.allowed_count += 1;
        } else {
            stats.blocked_count += 1;
        }

        // Update topic distribution
        for topic in &result.topics {
            let key = format!("{:?}", topic.category);
            *stats.topic_distribution.entry(key).or_insert(0) += 1;
        }

        // Update language distribution
        *stats.language_distribution.entry(result.language.clone()).or_insert(0) += 1;

        // Update average processing time
        let current_avg = stats.average_processing_time_ms;
        let current_count = stats.total_filtered as f64;
        stats.average_processing_time_ms =
            (current_avg * (current_count - 1.0) + duration.as_millis() as f64) / current_count;
    }

    fn load_default_blocklist() -> HashSet<String> {
        // In production, load from database or file
        let terms = vec![
            "spam", "scam", "fake", "fraud", "phishing",
            "malware", "virus", "trojan", "exploit",
        ];
        terms.into_iter().map(|s| s.to_string()).collect()
    }

    fn load_default_allowlist() -> HashSet<String> {
        // In production, load from database or file
        HashSet::new()
    }

    fn load_default_topic_filters() -> Vec<TopicFilter> {
        vec![
            TopicFilter {
                id: Uuid::new_v4(),
                name: "Technology Filter".to_string(),
                category: TopicCategory::Technology,
                keywords: vec![
                    "software".to_string(),
                    "hardware".to_string(),
                    "programming".to_string(),
                    "computer".to_string(),
                    "technology".to_string(),
                ],
                action: FilterAction::Tag,
                enabled: true,
            },
            TopicFilter {
                id: Uuid::new_v4(),
                name: "Politics Filter".to_string(),
                category: TopicCategory::Politics,
                keywords: vec![
                    "politics".to_string(),
                    "election".to_string(),
                    "government".to_string(),
                    "policy".to_string(),
                    "congress".to_string(),
                ],
                action: FilterAction::Review,
                enabled: true,
            },
        ]
    }

    fn load_topic_classifiers() -> HashMap<TopicCategory, Vec<String>> {
        let mut classifiers = HashMap::new();

        classifiers.insert(
            TopicCategory::Technology,
            vec![
                "software", "hardware", "computer", "programming", "code",
                "algorithm", "database", "network", "internet", "cloud",
                "ai", "machine learning", "blockchain", "cybersecurity",
            ].into_iter().map(String::from).collect(),
        );

        classifiers.insert(
            TopicCategory::Science,
            vec![
                "research", "experiment", "hypothesis", "theory", "discovery",
                "physics", "chemistry", "biology", "astronomy", "mathematics",
                "study", "analysis", "data", "scientist",
            ].into_iter().map(String::from).collect(),
        );

        classifiers.insert(
            TopicCategory::Business,
            vec![
                "business", "company", "market", "economy", "finance",
                "investment", "stock", "revenue", "profit", "startup",
                "entrepreneur", "corporate", "industry",
            ].into_iter().map(String::from).collect(),
        );

        classifiers
    }

    pub async fn add_blocklist_term(&self, term: String) {
        let mut blocklist = self.blocklist.write().await;
        blocklist.insert(term);
        self.cache.clear();  // Clear cache when rules change
    }

    pub async fn add_allowlist_term(&self, term: String) {
        let mut allowlist = self.allowlist.write().await;
        allowlist.insert(term);
        self.cache.clear();
    }

    pub async fn add_custom_rule(&self, rule: CustomRule) {
        let mut rules = self.custom_rules.write().await;
        rules.push(rule);
        self.cache.clear();
    }

    pub async fn get_statistics(&self) -> FilterStatistics {
        self.statistics.read().await.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_keyword_extraction() {
        let config = KeywordFilterConfig::default();
        let filter = KeywordFilter::new(config).await.unwrap();

        let content = "This is a test content about machine learning and artificial intelligence.";
        let keywords = filter.extract_keywords(content).unwrap();

        assert!(!keywords.is_empty());
        assert!(keywords.iter().any(|k| k.term.contains("machine") || k.term.contains("learning")));
    }

    #[tokio::test]
    async fn test_sentiment_analysis() {
        let config = KeywordFilterConfig::default();
        let filter = KeywordFilter::new(config).await.unwrap();

        let positive_content = "This is a great and wonderful product. I love it!";
        let sentiment = filter.analyze_sentiment(positive_content).unwrap();
        assert!(sentiment.overall > 0.0);

        let negative_content = "This is terrible and awful. I hate it completely.";
        let sentiment = filter.analyze_sentiment(negative_content).unwrap();
        assert!(sentiment.overall < 0.0);
    }
}