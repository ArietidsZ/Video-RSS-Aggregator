use redis::aio::ConnectionManager;
use redis::{AsyncCommands, Client};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use blake3;
use crate::{VideoMetadata, Result, ExtractorError};

/// Redis-based caching system for video metadata
pub struct MetadataCache {
    connection: ConnectionManager,
    default_ttl: Duration,
    prefix: String,
}

impl MetadataCache {
    /// Create new cache instance
    pub async fn new(redis_url: &str) -> Result<Self> {
        let client = Client::open(redis_url)
            .map_err(|e| ExtractorError::Cache(e))?;

        let connection = ConnectionManager::new(client).await
            .map_err(|e| ExtractorError::Cache(e))?;

        Ok(Self {
            connection,
            default_ttl: Duration::from_secs(3600), // 1 hour default
            prefix: "video_metadata".to_string(),
        })
    }

    /// Create cache with custom TTL
    pub async fn with_ttl(redis_url: &str, ttl_seconds: u64) -> Result<Self> {
        let mut cache = Self::new(redis_url).await?;
        cache.default_ttl = Duration::from_secs(ttl_seconds);
        Ok(cache)
    }

    /// Generate cache key from URL
    fn generate_key(&self, url: &str) -> String {
        let hash = blake3::hash(url.as_bytes());
        format!("{}:{}", self.prefix, hash.to_hex())
    }

    /// Get metadata from cache
    pub async fn get(&mut self, url: &str) -> Result<Option<VideoMetadata>> {
        let key = self.generate_key(url);

        let data: Option<Vec<u8>> = self.connection
            .get(&key)
            .await
            .map_err(|e| ExtractorError::Cache(e))?;

        match data {
            Some(bytes) => {
                let metadata = bincode::deserialize(&bytes)
                    .map_err(|e| ExtractorError::Parsing(format!("Failed to deserialize: {}", e)))?;
                Ok(Some(metadata))
            }
            None => Ok(None)
        }
    }

    /// Set metadata in cache
    pub async fn set(&mut self, url: &str, metadata: &VideoMetadata) -> Result<()> {
        self.set_with_ttl(url, metadata, self.default_ttl.as_secs()).await
    }

    /// Set metadata with custom TTL
    pub async fn set_with_ttl(&mut self, url: &str, metadata: &VideoMetadata, ttl_seconds: u64) -> Result<()> {
        let key = self.generate_key(url);

        let data = bincode::serialize(metadata)
            .map_err(|e| ExtractorError::Parsing(format!("Failed to serialize: {}", e)))?;

        self.connection
            .set_ex(key, data, ttl_seconds)
            .await
            .map_err(|e| ExtractorError::Cache(e))?;

        Ok(())
    }

    /// Get multiple metadata entries
    pub async fn get_batch(&mut self, urls: &[String]) -> Result<Vec<Option<VideoMetadata>>> {
        let keys: Vec<String> = urls.iter()
            .map(|url| self.generate_key(url))
            .collect();

        let values: Vec<Option<Vec<u8>>> = redis::cmd("MGET")
            .arg(&keys)
            .query_async(&mut self.connection)
            .await
            .map_err(|e| ExtractorError::Cache(e))?;

        let mut results = Vec::with_capacity(values.len());
        for value in values {
            match value {
                Some(bytes) => {
                    let metadata = bincode::deserialize(&bytes)
                        .map_err(|e| ExtractorError::Parsing(format!("Failed to deserialize: {}", e)))?;
                    results.push(Some(metadata));
                }
                None => results.push(None)
            }
        }

        Ok(results)
    }

    /// Set multiple metadata entries
    pub async fn set_batch(&mut self, entries: Vec<(&str, &VideoMetadata)>) -> Result<()> {
        use redis::pipe;

        let mut pipeline = pipe();
        pipeline.atomic();

        for (url, metadata) in entries {
            let key = self.generate_key(url);
            let data = bincode::serialize(metadata)
                .map_err(|e| ExtractorError::Parsing(format!("Failed to serialize: {}", e)))?;

            pipeline.set_ex(key, data, self.default_ttl.as_secs());
        }

        pipeline.query_async(&mut self.connection)
            .await
            .map_err(|e| ExtractorError::Cache(e))?;

        Ok(())
    }

    /// Delete metadata from cache
    pub async fn delete(&mut self, url: &str) -> Result<()> {
        let key = self.generate_key(url);

        self.connection
            .del(key)
            .await
            .map_err(|e| ExtractorError::Cache(e))?;

        Ok(())
    }

    /// Check if URL is cached
    pub async fn exists(&mut self, url: &str) -> Result<bool> {
        let key = self.generate_key(url);

        let exists: bool = self.connection
            .exists(key)
            .await
            .map_err(|e| ExtractorError::Cache(e))?;

        Ok(exists)
    }

    /// Get remaining TTL for cached entry
    pub async fn ttl(&mut self, url: &str) -> Result<Option<u64>> {
        let key = self.generate_key(url);

        let ttl: i64 = self.connection
            .ttl(key)
            .await
            .map_err(|e| ExtractorError::Cache(e))?;

        if ttl >= 0 {
            Ok(Some(ttl as u64))
        } else {
            Ok(None)
        }
    }

    /// Clear all cached metadata
    pub async fn clear_all(&mut self) -> Result<()> {
        let pattern = format!("{}:*", self.prefix);

        let keys: Vec<String> = self.connection
            .keys(pattern)
            .await
            .map_err(|e| ExtractorError::Cache(e))?;

        if !keys.is_empty() {
            self.connection
                .del(keys)
                .await
                .map_err(|e| ExtractorError::Cache(e))?;
        }

        Ok(())
    }

    /// Get cache statistics
    pub async fn stats(&mut self) -> Result<CacheStats> {
        let pattern = format!("{}:*", self.prefix);

        let keys: Vec<String> = self.connection
            .keys(&pattern)
            .await
            .map_err(|e| ExtractorError::Cache(e))?;

        let total_entries = keys.len();

        // Get memory usage for all keys
        let mut total_memory = 0u64;
        for key in &keys[..keys.len().min(100)] { // Sample first 100 keys
            let memory: Option<u64> = redis::cmd("MEMORY")
                .arg("USAGE")
                .arg(key)
                .query_async(&mut self.connection)
                .await
                .ok();

            if let Some(mem) = memory {
                total_memory += mem;
            }
        }

        // Estimate total memory if we sampled
        if keys.len() > 100 {
            total_memory = total_memory * keys.len() as u64 / 100;
        }

        Ok(CacheStats {
            total_entries,
            total_memory_bytes: total_memory,
            prefix: self.prefix.clone(),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub total_entries: usize,
    pub total_memory_bytes: u64,
    pub prefix: String,
}

/// In-memory LRU cache for frequently accessed items
pub struct MemoryCache {
    cache: cached::AsyncCache<String, VideoMetadata>,
}

impl MemoryCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: cached::AsyncCache::new(capacity),
        }
    }

    pub async fn get(&self, url: &str) -> Option<VideoMetadata> {
        self.cache.get(&url.to_string()).await
    }

    pub async fn set(&self, url: String, metadata: VideoMetadata) -> Option<VideoMetadata> {
        self.cache.set(url, metadata).await
    }

    pub fn size(&self) -> usize {
        self.cache.size()
    }
}

/// Two-tier caching with memory and Redis
pub struct TwoTierCache {
    memory: MemoryCache,
    redis: MetadataCache,
}

impl TwoTierCache {
    pub async fn new(redis_url: &str, memory_capacity: usize) -> Result<Self> {
        Ok(Self {
            memory: MemoryCache::new(memory_capacity),
            redis: MetadataCache::new(redis_url).await?,
        })
    }

    pub async fn get(&mut self, url: &str) -> Result<Option<VideoMetadata>> {
        // Check memory cache first
        if let Some(metadata) = self.memory.get(url).await {
            return Ok(Some(metadata));
        }

        // Check Redis cache
        if let Some(metadata) = self.redis.get(url).await? {
            // Populate memory cache
            self.memory.set(url.to_string(), metadata.clone()).await;
            return Ok(Some(metadata));
        }

        Ok(None)
    }

    pub async fn set(&mut self, url: &str, metadata: &VideoMetadata) -> Result<()> {
        // Set in both caches
        self.memory.set(url.to_string(), metadata.clone()).await;
        self.redis.set(url, metadata).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_metadata() -> VideoMetadata {
        VideoMetadata {
            id: "test123".to_string(),
            platform: crate::Platform::YouTube,
            title: "Test Video".to_string(),
            description: "Test Description".to_string(),
            author: crate::AuthorInfo {
                id: "author123".to_string(),
                name: "Test Author".to_string(),
                username: None,
                url: "https://example.com".to_string(),
                avatar_url: None,
                subscriber_count: None,
                verified: false,
                description: None,
            },
            duration: 300,
            published_at: Utc::now(),
            updated_at: Utc::now(),
            statistics: crate::VideoStatistics {
                view_count: 1000,
                like_count: Some(100),
                dislike_count: None,
                comment_count: Some(50),
                share_count: None,
                favorite_count: None,
                platform_stats: Default::default(),
            },
            qualities: vec![],
            thumbnails: vec![],
            tags: vec![],
            category: crate::VideoCategory::Education,
            language: crate::LanguageInfo {
                primary: "en".to_string(),
                detected: vec![],
                audio_language: None,
            },
            subtitles: vec![],
            url: "https://example.com/video".to_string(),
            video_urls: Default::default(),
            audio_url: None,
            extra_metadata: Default::default(),
            content_rating: crate::ContentRating {
                age_restricted: false,
                region_blocked: vec![],
                requires_login: false,
                is_private: false,
                is_unlisted: false,
                content_warning: None,
            },
            extracted_at: Utc::now(),
            cache_ttl: 3600,
        }
    }

    #[tokio::test]
    async fn test_memory_cache() {
        let cache = MemoryCache::new(10);
        let metadata = create_test_metadata();

        cache.set("test_url".to_string(), metadata.clone()).await;
        let retrieved = cache.get("test_url").await;

        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, "test123");
    }
}