use crate::{error::VideoRssError, types::*, Result};
use async_trait::async_trait;
use bb8::Pool;
use bb8_redis::RedisConnectionManager;
use chrono::{DateTime, Utc};
use redis::AsyncCommands;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry<T> {
    pub data: T,
    pub expires_at: DateTime<Utc>,
    pub etag: String,
    pub version: u32,
}

impl<T> CacheEntry<T> {
    pub fn new(data: T, ttl_seconds: i64) -> Self {
        let expires_at = Utc::now() + chrono::Duration::seconds(ttl_seconds);
        let etag = format!("{:x}", seahash::hash(expires_at.timestamp().to_string().as_bytes()));

        Self {
            data,
            expires_at,
            etag,
            version: 1,
        }
    }

    pub fn is_expired(&self) -> bool {
        Utc::now() > self.expires_at
    }

    pub fn is_fresh(&self, max_age_seconds: i64) -> bool {
        let age = Utc::now() - (self.expires_at - chrono::Duration::seconds(300)); // Assume 5min default TTL
        age.num_seconds() < max_age_seconds
    }
}

#[async_trait]
pub trait CacheBackend: Send + Sync {
    async fn get<T>(&self, key: &str) -> Result<Option<CacheEntry<T>>>
    where
        T: for<'de> Deserialize<'de> + Send;

    async fn set<T>(&self, key: &str, entry: &CacheEntry<T>, ttl_seconds: u64) -> Result<()>
    where
        T: Serialize + Send + Sync;

    async fn delete(&self, key: &str) -> Result<()>;
    async fn exists(&self, key: &str) -> Result<bool>;
    async fn get_ttl(&self, key: &str) -> Result<Option<u64>>;
}

// Redis Backend Implementation
pub struct RedisCache {
    pool: Pool<RedisConnectionManager>,
    prefix: String,
}

impl RedisCache {
    pub async fn new(redis_url: &str, prefix: String) -> Result<Self> {
        let manager = RedisConnectionManager::new(redis_url)?;
        let pool = Pool::builder()
            .max_size(20)
            .min_idle(Some(5))
            .max_lifetime(Some(Duration::from_secs(3600))) // 1 hour
            .idle_timeout(Some(Duration::from_secs(600))) // 10 minutes
            .connection_timeout(Duration::from_secs(5))
            .build(manager)
            .await
            .map_err(|e| VideoRssError::Config(format!("Redis pool creation failed: {}", e)))?;

        info!("Redis cache initialized with prefix: {}", prefix);
        Ok(Self { pool, prefix })
    }

    fn make_key(&self, key: &str) -> String {
        format!("{}:{}", self.prefix, key)
    }
}

#[async_trait]
impl CacheBackend for RedisCache {
    async fn get<T>(&self, key: &str) -> Result<Option<CacheEntry<T>>>
    where
        T: for<'de> Deserialize<'de> + Send,
    {
        let mut conn = self.pool.get().await
            .map_err(|e| VideoRssError::Config(format!("Redis connection failed: {}", e)))?;

        let redis_key = self.make_key(key);

        match conn.get::<_, Option<String>>(&redis_key).await {
            Ok(Some(data)) => {
                match serde_json::from_str::<CacheEntry<T>>(&data) {
                    Ok(entry) => {
                        if entry.is_expired() {
                            // Clean up expired entry
                            let _ = conn.del::<_, ()>(&redis_key).await;
                            debug!("Removed expired cache entry: {}", key);
                            Ok(None)
                        } else {
                            debug!("Cache hit for key: {}", key);
                            Ok(Some(entry))
                        }
                    }
                    Err(e) => {
                        warn!("Failed to deserialize cache entry {}: {}", key, e);
                        // Clean up corrupted entry
                        let _ = conn.del::<_, ()>(&redis_key).await;
                        Ok(None)
                    }
                }
            }
            Ok(None) => {
                debug!("Cache miss for key: {}", key);
                Ok(None)
            }
            Err(e) => {
                error!("Redis get error for key {}: {}", key, e);
                Ok(None) // Graceful degradation
            }
        }
    }

    async fn set<T>(&self, key: &str, entry: &CacheEntry<T>, ttl_seconds: u64) -> Result<()>
    where
        T: Serialize + Send + Sync,
    {
        let mut conn = self.pool.get().await
            .map_err(|e| VideoRssError::Config(format!("Redis connection failed: {}", e)))?;

        let redis_key = self.make_key(key);
        let serialized = serde_json::to_string(entry)
            .map_err(|e| VideoRssError::Json(e))?;

        conn.set_ex(&redis_key, serialized, ttl_seconds).await
            .map_err(|e| VideoRssError::Config(format!("Redis set failed: {}", e)))?;

        debug!("Cached entry for key: {} with TTL: {}s", key, ttl_seconds);
        Ok(())
    }

    async fn delete(&self, key: &str) -> Result<()> {
        let mut conn = self.pool.get().await
            .map_err(|e| VideoRssError::Config(format!("Redis connection failed: {}", e)))?;

        let redis_key = self.make_key(key);
        conn.del::<_, ()>(&redis_key).await
            .map_err(|e| VideoRssError::Config(format!("Redis delete failed: {}", e)))?;

        debug!("Deleted cache entry: {}", key);
        Ok(())
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        let mut conn = self.pool.get().await
            .map_err(|e| VideoRssError::Config(format!("Redis connection failed: {}", e)))?;

        let redis_key = self.make_key(key);
        let exists: bool = conn.exists(&redis_key).await
            .map_err(|e| VideoRssError::Config(format!("Redis exists failed: {}", e)))?;

        Ok(exists)
    }

    async fn get_ttl(&self, key: &str) -> Result<Option<u64>> {
        let mut conn = self.pool.get().await
            .map_err(|e| VideoRssError::Config(format!("Redis connection failed: {}", e)))?;

        let redis_key = self.make_key(key);
        let ttl: i64 = conn.ttl(&redis_key).await
            .map_err(|e| VideoRssError::Config(format!("Redis TTL failed: {}", e)))?;

        if ttl < 0 {
            Ok(None)
        } else {
            Ok(Some(ttl as u64))
        }
    }
}

// In-Memory Backend for Development/Testing
pub struct MemoryCache {
    data: Arc<RwLock<std::collections::HashMap<String, (String, DateTime<Utc>)>>>,
    prefix: String,
}

impl MemoryCache {
    pub fn new(prefix: String) -> Self {
        info!("Memory cache initialized with prefix: {}", prefix);
        Self {
            data: Arc::new(RwLock::new(std::collections::HashMap::new())),
            prefix,
        }
    }

    fn make_key(&self, key: &str) -> String {
        format!("{}:{}", self.prefix, key)
    }

    async fn cleanup_expired(&self) {
        let mut data = self.data.write().await;
        let now = Utc::now();
        data.retain(|_, (_, expires_at)| now <= *expires_at);
    }
}

#[async_trait]
impl CacheBackend for MemoryCache {
    async fn get<T>(&self, key: &str) -> Result<Option<CacheEntry<T>>>
    where
        T: for<'de> Deserialize<'de> + Send,
    {
        self.cleanup_expired().await;

        let data = self.data.read().await;
        let cache_key = self.make_key(key);

        if let Some((serialized, expires_at)) = data.get(&cache_key) {
            if Utc::now() <= *expires_at {
                match serde_json::from_str::<CacheEntry<T>>(serialized) {
                    Ok(entry) => {
                        debug!("Memory cache hit for key: {}", key);
                        Ok(Some(entry))
                    }
                    Err(e) => {
                        warn!("Failed to deserialize memory cache entry {}: {}", key, e);
                        Ok(None)
                    }
                }
            } else {
                debug!("Memory cache entry expired for key: {}", key);
                Ok(None)
            }
        } else {
            debug!("Memory cache miss for key: {}", key);
            Ok(None)
        }
    }

    async fn set<T>(&self, key: &str, entry: &CacheEntry<T>, ttl_seconds: u64) -> Result<()>
    where
        T: Serialize + Send + Sync,
    {
        let serialized = serde_json::to_string(entry)
            .map_err(|e| VideoRssError::Json(e))?;

        let expires_at = Utc::now() + chrono::Duration::seconds(ttl_seconds as i64);
        let cache_key = self.make_key(key);

        let mut data = self.data.write().await;
        data.insert(cache_key, (serialized, expires_at));

        debug!("Memory cached entry for key: {} with TTL: {}s", key, ttl_seconds);
        Ok(())
    }

    async fn delete(&self, key: &str) -> Result<()> {
        let cache_key = self.make_key(key);
        let mut data = self.data.write().await;
        data.remove(&cache_key);
        debug!("Deleted memory cache entry: {}", key);
        Ok(())
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        self.cleanup_expired().await;
        let data = self.data.read().await;
        let cache_key = self.make_key(key);
        Ok(data.contains_key(&cache_key))
    }

    async fn get_ttl(&self, key: &str) -> Result<Option<u64>> {
        let data = self.data.read().await;
        let cache_key = self.make_key(key);

        if let Some((_, expires_at)) = data.get(&cache_key) {
            let ttl = (*expires_at - Utc::now()).num_seconds();
            if ttl > 0 {
                Ok(Some(ttl as u64))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }
}

// High-level cache manager
pub struct CacheManager {
    backend: Box<dyn CacheBackend>,
    default_ttl: u64,
}

impl CacheManager {
    pub fn new(backend: Box<dyn CacheBackend>, default_ttl: u64) -> Self {
        Self {
            backend,
            default_ttl,
        }
    }

    pub async fn get_or_set<T, F, Fut>(&self, key: &str, fetcher: F) -> Result<T>
    where
        T: Serialize + for<'de> Deserialize<'de> + Send + Sync + Clone,
        F: FnOnce() -> Fut + Send,
        Fut: std::future::Future<Output = Result<T>> + Send,
    {
        // Try cache first
        if let Some(entry) = self.backend.get::<T>(key).await? {
            return Ok(entry.data);
        }

        // Cache miss - fetch and cache
        let data = fetcher().await?;
        let entry = CacheEntry::new(data.clone(), self.default_ttl as i64);

        if let Err(e) = self.backend.set(key, &entry, self.default_ttl).await {
            warn!("Failed to cache entry for key {}: {}", key, e);
        }

        Ok(data)
    }

    pub async fn get_with_etag<T>(&self, key: &str) -> Result<Option<(T, String)>>
    where
        T: for<'de> Deserialize<'de> + Send,
    {
        if let Some(entry) = self.backend.get::<T>(key).await? {
            Ok(Some((entry.data, entry.etag)))
        } else {
            Ok(None)
        }
    }

    pub async fn set_with_ttl<T>(&self, key: &str, data: T, ttl_seconds: u64) -> Result<()>
    where
        T: Serialize + Send + Sync,
    {
        let entry = CacheEntry::new(data, ttl_seconds as i64);
        self.backend.set(key, &entry, ttl_seconds).await
    }

    pub async fn invalidate(&self, key: &str) -> Result<()> {
        self.backend.delete(key).await
    }

    pub async fn invalidate_pattern(&self, pattern: &str) -> Result<()> {
        // For simple implementation, we'll just log this
        // In production, you'd want to implement pattern-based deletion
        warn!("Pattern-based invalidation not implemented: {}", pattern);
        Ok(())
    }
}

// Utility functions
pub fn cache_key_for_platform(platform: Platform) -> String {
    format!("videos:{}", platform.as_str())
}

pub fn cache_key_for_rss(platforms: &[Platform]) -> String {
    let mut platform_names: Vec<_> = platforms.iter().map(|p| p.as_str()).collect();
    platform_names.sort();
    format!("rss:{}", platform_names.join(","))
}

pub fn cache_key_for_transcription(video_id: &str) -> String {
    format!("transcription:{}", video_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_cache() {
        let cache = MemoryCache::new("test".to_string());

        let test_data = vec!["item1".to_string(), "item2".to_string()];
        let entry = CacheEntry::new(test_data.clone(), 3600);

        cache.set("test_key", &entry, 3600).await.unwrap();

        let result: Option<CacheEntry<Vec<String>>> = cache.get("test_key").await.unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().data, test_data);
    }

    #[test]
    fn test_cache_entry_expiration() {
        let entry = CacheEntry::new("test".to_string(), -1); // Expired
        assert!(entry.is_expired());

        let entry = CacheEntry::new("test".to_string(), 3600); // Valid
        assert!(!entry.is_expired());
    }
}