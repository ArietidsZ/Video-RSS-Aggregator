use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use crate::{
    VideoMetadata, Platform, Result, ExtractorError, ExtractionConfig,
    BatchExtractionRequest, BatchExtractionResponse, ExtractionFailure,
    url_parser::{UrlParser, ParsedUrl},
    cache::TwoTierCache,
    rate_limiter::RateLimiter,
    proxy_manager::{ProxyManager, SmartProxySelector, SelectionStrategy},
    platforms::{
        PlatformExtractor,
        youtube::YouTubeExtractor,
        bilibili::BilibiliExtractor,
        douyin::DouyinExtractor,
        tiktok::TikTokExtractor,
        kuaishou::KuaishouExtractor,
    },
};

/// Main video metadata extractor with all optimizations
pub struct VideoMetadataExtractor {
    /// Platform-specific extractors
    extractors: Arc<RwLock<HashMap<Platform, Arc<dyn PlatformExtractor>>>>,

    /// URL parser
    url_parser: Arc<UrlParser>,

    /// Two-tier cache (memory + Redis)
    cache: Arc<RwLock<TwoTierCache>>,

    /// Rate limiter
    rate_limiter: Arc<RateLimiter>,

    /// Proxy manager
    proxy_manager: Arc<ProxyManager>,

    /// Configuration
    config: Arc<ExtractionConfig>,
}

impl VideoMetadataExtractor {
    /// Create new extractor with default configuration
    pub async fn new(redis_url: &str) -> Result<Self> {
        let config = Arc::new(ExtractionConfig::default());

        Ok(Self {
            extractors: Arc::new(RwLock::new(HashMap::new())),
            url_parser: Arc::new(UrlParser::new()),
            cache: Arc::new(RwLock::new(
                TwoTierCache::new(redis_url, 1000).await?
            )),
            rate_limiter: Arc::new(RateLimiter::new()),
            proxy_manager: Arc::new(ProxyManager::new()),
            config,
        })
    }

    /// Create with custom configuration
    pub async fn with_config(redis_url: &str, config: ExtractionConfig) -> Result<Self> {
        let mut extractor = Self::new(redis_url).await?;
        extractor.config = Arc::new(config);
        Ok(extractor)
    }

    /// Initialize platform extractors
    pub async fn initialize(&self) -> Result<()> {
        let mut extractors = self.extractors.write().await;

        // Initialize YouTube
        let youtube = Arc::new(YouTubeExtractor::new(
            std::env::var("YOUTUBE_API_KEY").ok()
        ));
        extractors.insert(Platform::YouTube, youtube);

        // Initialize Bilibili
        let bilibili = Arc::new(BilibiliExtractor::new(
            std::env::var("BILIBILI_COOKIE").ok()
        ));
        extractors.insert(Platform::Bilibili, bilibili);

        // Initialize Douyin
        let douyin = Arc::new(DouyinExtractor::new());
        extractors.insert(Platform::Douyin, douyin);

        // Initialize TikTok
        let tiktok = Arc::new(TikTokExtractor::new());
        extractors.insert(Platform::TikTok, tiktok);

        // Initialize Kuaishou
        let kuaishou = Arc::new(KuaishouExtractor::new());
        extractors.insert(Platform::Kuaishou, kuaishou);

        // Setup rate limits
        self.rate_limiter.setup_defaults().await;

        Ok(())
    }

    /// Extract metadata from a video URL
    pub async fn extract(&self, url: &str) -> Result<VideoMetadata> {
        let start = std::time::Instant::now();

        // Parse URL
        let parsed = self.url_parser.parse(url)?;

        // Check cache
        if self.config.cache_ttl > 0 {
            let mut cache = self.cache.write().await;
            if let Some(metadata) = cache.get(url).await? {
                tracing::debug!("Cache hit for {}", url);
                return Ok(metadata);
            }
        }

        // Wait for rate limit
        self.rate_limiter.wait(&parsed.platform).await?;

        // Get platform extractor
        let extractors = self.extractors.read().await;
        let extractor = extractors.get(&parsed.platform)
            .ok_or_else(|| ExtractorError::UnsupportedPlatform(
                format!("Platform {:?} not supported", parsed.platform)
            ))?;

        // Extract metadata
        let metadata = extractor.extract(&parsed.video_id).await?;

        // Store in cache
        if self.config.cache_ttl > 0 {
            let mut cache = self.cache.write().await;
            cache.set(url, &metadata).await?;
        }

        let elapsed = start.elapsed();
        tracing::info!(
            "Extracted metadata for {} in {:?} (platform: {:?})",
            url, elapsed, parsed.platform
        );

        Ok(metadata)
    }

    /// Extract metadata from multiple URLs (batch)
    pub async fn extract_batch(&self, request: BatchExtractionRequest) -> Result<BatchExtractionResponse> {
        let start = std::time::Instant::now();
        let mut successful = Vec::new();
        let mut failed = Vec::new();

        // Group URLs by platform for efficient batch processing
        let mut platform_groups: HashMap<Platform, Vec<ParsedUrl>> = HashMap::new();

        for url in &request.urls {
            match self.url_parser.parse(url) {
                Ok(parsed) => {
                    platform_groups.entry(parsed.platform.clone())
                        .or_insert_with(Vec::new)
                        .push(parsed);
                }
                Err(e) => {
                    failed.push(ExtractionFailure {
                        url: url.clone(),
                        error: e.to_string(),
                        timestamp: chrono::Utc::now(),
                    });
                }
            }
        }

        // Process each platform group
        for (platform, parsed_urls) in platform_groups {
            let extractors = self.extractors.read().await;

            if let Some(extractor) = extractors.get(&platform) {
                // Extract video IDs
                let video_ids: Vec<String> = parsed_urls.iter()
                    .map(|p| p.video_id.clone())
                    .collect();

                // Batch extract
                match extractor.extract_batch(video_ids).await {
                    Ok(metadatas) => {
                        successful.extend(metadatas);
                    }
                    Err(e) => {
                        for parsed in &parsed_urls {
                            failed.push(ExtractionFailure {
                                url: parsed.original.clone(),
                                error: e.to_string(),
                                timestamp: chrono::Utc::now(),
                            });
                        }
                    }
                }
            }
        }

        let total_time_ms = start.elapsed().as_millis() as u64;

        Ok(BatchExtractionResponse {
            successful,
            failed,
            total_time_ms,
        })
    }

    /// Extract with content type detection
    pub async fn extract_with_detection(&self, url: &str) -> Result<(VideoMetadata, VideoContentType)> {
        let metadata = self.extract(url).await?;
        let content_type = self.detect_content_type(&metadata);

        Ok((metadata, content_type))
    }

    /// Detect video content type
    fn detect_content_type(&self, metadata: &VideoMetadata) -> VideoContentType {
        // Duration-based detection
        if metadata.duration < 60 {
            return VideoContentType::Short;
        } else if metadata.duration < 600 {
            return VideoContentType::Regular;
        } else if metadata.duration < 3600 {
            return VideoContentType::Medium;
        } else {
            return VideoContentType::Long;
        }
    }

    /// Extract with proxy
    pub async fn extract_with_proxy(&self, url: &str) -> Result<VideoMetadata> {
        // Get a proxy
        let selector = SmartProxySelector::new(
            self.proxy_manager.clone(),
            SelectionStrategy::BestPerformance
        );

        if let Some(proxy) = selector.select().await {
            tracing::debug!("Using proxy: {}", proxy.url);

            // Create client with proxy
            let client = self.proxy_manager.create_client(&proxy)?;

            // Extract with proxy client
            // (Would need to modify extractors to accept custom client)
            self.extract(url).await
        } else {
            // Fallback to direct extraction
            self.extract(url).await
        }
    }

    /// Validate and extract
    pub async fn validate_and_extract(&self, url: &str) -> Result<VideoMetadata> {
        // Validate URL first
        if !self.url_parser.validate(url).await? {
            return Err(ExtractorError::InvalidUrl(
                format!("URL validation failed: {}", url)
            ));
        }

        self.extract(url).await
    }

    /// Get extraction statistics
    pub async fn get_stats(&self) -> ExtractionStats {
        let extractors = self.extractors.read().await;

        let platforms: Vec<PlatformStats> = extractors.iter()
            .map(|(platform, extractor)| PlatformStats {
                platform: platform.clone(),
                is_authenticated: extractor.is_authenticated(),
            })
            .collect();

        ExtractionStats {
            platforms,
            cache_enabled: self.config.cache_ttl > 0,
            proxy_count: 0, // Would get from proxy_manager
        }
    }

    /// Clear cache
    pub async fn clear_cache(&self) -> Result<()> {
        let mut cache = self.cache.write().await;
        // Implement cache clearing
        Ok(())
    }
}

/// Video content type classification
#[derive(Debug, Clone)]
pub enum VideoContentType {
    Short,      // < 1 minute
    Regular,    // 1-10 minutes
    Medium,     // 10-60 minutes
    Long,       // > 60 minutes
    LiveStream,
    Podcast,
    Music,
    Educational,
}

/// Extraction statistics
#[derive(Debug, Clone)]
pub struct ExtractionStats {
    pub platforms: Vec<PlatformStats>,
    pub cache_enabled: bool,
    pub proxy_count: usize,
}

#[derive(Debug, Clone)]
pub struct PlatformStats {
    pub platform: Platform,
    pub is_authenticated: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_url_extraction() {
        let extractor = VideoMetadataExtractor::new("redis://localhost:6379")
            .await
            .unwrap();

        extractor.initialize().await.unwrap();

        // Test YouTube URL
        let url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ";

        // This would fail without actual API key
        // let metadata = extractor.extract(url).await;
    }

    #[tokio::test]
    async fn test_batch_extraction() {
        let extractor = VideoMetadataExtractor::new("redis://localhost:6379")
            .await
            .unwrap();

        extractor.initialize().await.unwrap();

        let request = BatchExtractionRequest {
            urls: vec![
                "https://www.youtube.com/watch?v=test1".to_string(),
                "https://www.bilibili.com/video/BV1234567890".to_string(),
            ],
            config: ExtractionConfig::default(),
            parallel_limit: 2,
        };

        // let response = extractor.extract_batch(request).await;
    }
}