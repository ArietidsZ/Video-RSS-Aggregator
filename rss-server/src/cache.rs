use std::sync::Arc;
use std::time::Duration;
use anyhow::Result;
use moka::future::Cache;
use redis::aio::ConnectionManager;
use redis::{AsyncCommands, Client};
use serde::{Deserialize, Serialize};
use tokio::time::Instant;

/// Multi-tier caching system for RSS feeds
/// L1: In-memory cache (Moka)
/// L2: Redis cache
/// L3: CDN cache (headers)
pub struct CacheManager {
    /// L1 in-memory cache - ultra-fast access
    memory_cache: Arc<Cache<String, CachedItem>>,

    /// L2 Redis cache - distributed and persistent
    redis_conn: ConnectionManager,

    /// Compression for large items
    compression_enabled: bool,

    /// Statistics
    stats: Arc<CacheStats>,
}

impl CacheManager {
    pub async fn new(redis_url: &str) -> Result<Self> {
        // Initialize in-memory cache with 10k items max, 100MB max size
        let memory_cache = Cache::builder()
            .max_capacity(10_000)
            .time_to_live(Duration::from_secs(300)) // 5 minutes default TTL
            .time_to_idle(Duration::from_secs(60))  // Remove if idle for 1 minute
            .build();

        // Connect to Redis
        let client = Client::open(redis_url)?;
        let redis_conn = ConnectionManager::new(client).await?;

        Ok(Self {
            memory_cache: Arc::new(memory_cache),
            redis_conn,
            compression_enabled: true,
            stats: Arc::new(CacheStats::new()),
        })
    }

    /// Get feed from cache (checks L1 then L2)
    pub async fn get_feed(&self, key: &str) -> Option<String> {
        let start = Instant::now();

        // Check L1 (memory cache)
        if let Some(item) = self.memory_cache.get(key).await {
            self.stats.record_l1_hit(start.elapsed());
            return Some(item.data);
        }

        self.stats.record_l1_miss();

        // Check L2 (Redis)
        if let Ok(Some(data)) = self.get_from_redis(key).await {
            // Populate L1 for next time
            self.memory_cache.insert(
                key.to_string(),
                CachedItem {
                    data: data.clone(),
                    timestamp: chrono::Utc::now(),
                }
            ).await;

            self.stats.record_l2_hit(start.elapsed());
            return Some(data);
        }

        self.stats.record_l2_miss();
        None
    }

    /// Set feed in cache (updates both L1 and L2)
    pub async fn set_feed(&self, key: &str, value: &str, ttl_seconds: u64) {
        let item = CachedItem {
            data: value.to_string(),
            timestamp: chrono::Utc::now(),
        };

        // Update L1
        self.memory_cache.insert(key.to_string(), item.clone()).await;

        // Update L2
        let _ = self.set_in_redis(key, &item.data, ttl_seconds).await;

        self.stats.record_set();
    }

    /// Get multiple feeds in a single operation (batch get)
    pub async fn get_feeds_batch(&self, keys: Vec<String>) -> Vec<Option<String>> {
        let mut results = Vec::with_capacity(keys.len());

        // First, try to get all from L1
        let mut l1_misses = Vec::new();
        for key in &keys {
            if let Some(item) = self.memory_cache.get(key).await {
                results.push(Some(item.data));
            } else {
                results.push(None);
                l1_misses.push(key.clone());
            }
        }

        // Batch get L1 misses from Redis
        if !l1_misses.is_empty() {
            if let Ok(redis_results) = self.get_batch_from_redis(&l1_misses).await {
                // Update results and populate L1
                for (key, value) in l1_misses.iter().zip(redis_results) {
                    if let Some(data) = value {
                        // Find index in original keys
                        if let Some(idx) = keys.iter().position(|k| k == key) {
                            results[idx] = Some(data.clone());

                            // Populate L1
                            self.memory_cache.insert(
                                key.clone(),
                                CachedItem {
                                    data,
                                    timestamp: chrono::Utc::now(),
                                }
                            ).await;
                        }
                    }
                }
            }
        }

        results
    }

    /// Set multiple feeds in a single operation (batch set)
    pub async fn set_feeds_batch(&self, items: Vec<(String, String, u64)>) {
        // Update L1
        for (key, value, _) in &items {
            self.memory_cache.insert(
                key.clone(),
                CachedItem {
                    data: value.clone(),
                    timestamp: chrono::Utc::now(),
                }
            ).await;
        }

        // Batch update L2
        let _ = self.set_batch_in_redis(items).await;
    }

    /// Invalidate a specific feed
    pub async fn invalidate_feed(&self, key: &str) {
        // Remove from L1
        self.memory_cache.invalidate(key).await;

        // Remove from L2
        let _: Result<(), _> = self.redis_conn
            .clone()
            .del(format!("rss:{}", key))
            .await;

        self.stats.record_invalidation();
    }

    /// Invalidate multiple feeds
    pub async fn invalidate_feeds_batch(&self, keys: Vec<String>) {
        // Remove from L1
        for key in &keys {
            self.memory_cache.invalidate(key).await;
        }

        // Batch remove from L2
        if !keys.is_empty() {
            let redis_keys: Vec<String> = keys.iter()
                .map(|k| format!("rss:{}", k))
                .collect();

            let _: Result<(), _> = self.redis_conn
                .clone()
                .del(redis_keys)
                .await;
        }

        self.stats.record_invalidation_batch(keys.len());
    }

    /// Clear all caches
    pub async fn clear_all(&self) -> Result<()> {
        // Clear L1
        self.memory_cache.invalidate_all();

        // Clear L2 (with pattern matching)
        let keys: Vec<String> = self.redis_conn
            .clone()
            .keys("rss:*")
            .await?;

        if !keys.is_empty() {
            let _: () = self.redis_conn
                .clone()
                .del(keys)
                .await?;
        }

        Ok(())
    }

    /// Warm up cache with frequently accessed feeds
    pub async fn warm_up(&self, feed_ids: Vec<String>, feed_service: Arc<crate::services::FeedService>) {
        use futures::stream::{self, StreamExt};

        let concurrent_limit = 10;

        stream::iter(feed_ids)
            .map(|feed_id| {
                let service = feed_service.clone();
                async move {
                    if let Ok(feed_xml) = service.generate_feed(&feed_id).await {
                        self.set_feed(&feed_id, &feed_xml, 300).await;
                    }
                }
            })
            .buffer_unordered(concurrent_limit)
            .collect::<()>()
            .await;
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStatsSnapshot {
        self.stats.snapshot()
    }

    /// Check cache health
    pub async fn is_healthy(&self) -> bool {
        // Check Redis connection
        let redis_ok = self.redis_conn
            .clone()
            .ping::<String>()
            .await
            .is_ok();

        // Check memory cache
        let memory_ok = self.memory_cache.entry_count() < 9000; // Below 90% capacity

        redis_ok && memory_ok
    }

    // Private helper methods

    async fn get_from_redis(&self, key: &str) -> Result<Option<String>> {
        let redis_key = format!("rss:{}", key);

        let data: Option<Vec<u8>> = self.redis_conn
            .clone()
            .get(&redis_key)
            .await?;

        match data {
            Some(bytes) => {
                let decompressed = if self.compression_enabled {
                    Self::decompress(&bytes)?
                } else {
                    String::from_utf8(bytes)?
                };
                Ok(Some(decompressed))
            }
            None => Ok(None)
        }
    }

    async fn set_in_redis(&self, key: &str, value: &str, ttl_seconds: u64) -> Result<()> {
        let redis_key = format!("rss:{}", key);

        let data = if self.compression_enabled && value.len() > 1024 {
            Self::compress(value)?
        } else {
            value.as_bytes().to_vec()
        };

        self.redis_conn
            .clone()
            .set_ex(redis_key, data, ttl_seconds)
            .await?;

        Ok(())
    }

    async fn get_batch_from_redis(&self, keys: &[String]) -> Result<Vec<Option<String>>> {
        let redis_keys: Vec<String> = keys.iter()
            .map(|k| format!("rss:{}", k))
            .collect();

        let values: Vec<Option<Vec<u8>>> = redis::cmd("MGET")
            .arg(&redis_keys)
            .query_async(&mut self.redis_conn.clone())
            .await?;

        let mut results = Vec::with_capacity(values.len());
        for value in values {
            match value {
                Some(bytes) => {
                    let decompressed = if self.compression_enabled {
                        Self::decompress(&bytes)?
                    } else {
                        String::from_utf8(bytes)?
                    };
                    results.push(Some(decompressed));
                }
                None => results.push(None)
            }
        }

        Ok(results)
    }

    async fn set_batch_in_redis(&self, items: Vec<(String, String, u64)>) -> Result<()> {
        use redis::pipe;

        let mut pipeline = pipe();
        pipeline.atomic();

        for (key, value, ttl) in items {
            let redis_key = format!("rss:{}", key);

            let data = if self.compression_enabled && value.len() > 1024 {
                Self::compress(&value)?
            } else {
                value.into_bytes()
            };

            pipeline.set_ex(redis_key, data, ttl);
        }

        pipeline.query_async(&mut self.redis_conn.clone()).await?;

        Ok(())
    }

    fn compress(data: &str) -> Result<Vec<u8>> {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;

        let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
        encoder.write_all(data.as_bytes())?;
        Ok(encoder.finish()?)
    }

    fn decompress(data: &[u8]) -> Result<String> {
        use flate2::read::GzDecoder;
        use std::io::Read;

        let mut decoder = GzDecoder::new(data);
        let mut decompressed = String::new();
        decoder.read_to_string(&mut decompressed)?;
        Ok(decompressed)
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct CachedItem {
    data: String,
    timestamp: chrono::DateTime<chrono::Utc>,
}

/// Cache statistics collector
struct CacheStats {
    l1_hits: std::sync::atomic::AtomicU64,
    l1_misses: std::sync::atomic::AtomicU64,
    l2_hits: std::sync::atomic::AtomicU64,
    l2_misses: std::sync::atomic::AtomicU64,
    sets: std::sync::atomic::AtomicU64,
    invalidations: std::sync::atomic::AtomicU64,
    total_l1_latency_us: std::sync::atomic::AtomicU64,
    total_l2_latency_us: std::sync::atomic::AtomicU64,
}

impl CacheStats {
    fn new() -> Self {
        use std::sync::atomic::AtomicU64;

        Self {
            l1_hits: AtomicU64::new(0),
            l1_misses: AtomicU64::new(0),
            l2_hits: AtomicU64::new(0),
            l2_misses: AtomicU64::new(0),
            sets: AtomicU64::new(0),
            invalidations: AtomicU64::new(0),
            total_l1_latency_us: AtomicU64::new(0),
            total_l2_latency_us: AtomicU64::new(0),
        }
    }

    fn record_l1_hit(&self, latency: Duration) {
        use std::sync::atomic::Ordering;

        self.l1_hits.fetch_add(1, Ordering::Relaxed);
        self.total_l1_latency_us.fetch_add(
            latency.as_micros() as u64,
            Ordering::Relaxed
        );
    }

    fn record_l1_miss(&self) {
        use std::sync::atomic::Ordering;
        self.l1_misses.fetch_add(1, Ordering::Relaxed);
    }

    fn record_l2_hit(&self, latency: Duration) {
        use std::sync::atomic::Ordering;

        self.l2_hits.fetch_add(1, Ordering::Relaxed);
        self.total_l2_latency_us.fetch_add(
            latency.as_micros() as u64,
            Ordering::Relaxed
        );
    }

    fn record_l2_miss(&self) {
        use std::sync::atomic::Ordering;
        self.l2_misses.fetch_add(1, Ordering::Relaxed);
    }

    fn record_set(&self) {
        use std::sync::atomic::Ordering;
        self.sets.fetch_add(1, Ordering::Relaxed);
    }

    fn record_invalidation(&self) {
        use std::sync::atomic::Ordering;
        self.invalidations.fetch_add(1, Ordering::Relaxed);
    }

    fn record_invalidation_batch(&self, count: usize) {
        use std::sync::atomic::Ordering;
        self.invalidations.fetch_add(count as u64, Ordering::Relaxed);
    }

    fn snapshot(&self) -> CacheStatsSnapshot {
        use std::sync::atomic::Ordering;

        let l1_hits = self.l1_hits.load(Ordering::Relaxed);
        let l1_misses = self.l1_misses.load(Ordering::Relaxed);
        let l2_hits = self.l2_hits.load(Ordering::Relaxed);
        let l2_misses = self.l2_misses.load(Ordering::Relaxed);

        let l1_hit_rate = if l1_hits + l1_misses > 0 {
            (l1_hits as f64) / ((l1_hits + l1_misses) as f64)
        } else {
            0.0
        };

        let l2_hit_rate = if l2_hits + l2_misses > 0 {
            (l2_hits as f64) / ((l2_hits + l2_misses) as f64)
        } else {
            0.0
        };

        let avg_l1_latency_us = if l1_hits > 0 {
            self.total_l1_latency_us.load(Ordering::Relaxed) / l1_hits
        } else {
            0
        };

        let avg_l2_latency_us = if l2_hits > 0 {
            self.total_l2_latency_us.load(Ordering::Relaxed) / l2_hits
        } else {
            0
        };

        CacheStatsSnapshot {
            l1_hits,
            l1_misses,
            l2_hits,
            l2_misses,
            l1_hit_rate,
            l2_hit_rate,
            avg_l1_latency_us,
            avg_l2_latency_us,
            total_sets: self.sets.load(Ordering::Relaxed),
            total_invalidations: self.invalidations.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CacheStatsSnapshot {
    pub l1_hits: u64,
    pub l1_misses: u64,
    pub l2_hits: u64,
    pub l2_misses: u64,
    pub l1_hit_rate: f64,
    pub l2_hit_rate: f64,
    pub avg_l1_latency_us: u64,
    pub avg_l2_latency_us: u64,
    pub total_sets: u64,
    pub total_invalidations: u64,
}

/// CDN cache headers generator
pub struct CDNHeaders {
    max_age: u32,
    s_maxage: u32,
    stale_while_revalidate: u32,
}

impl CDNHeaders {
    pub fn new() -> Self {
        Self {
            max_age: 300,        // Browser cache: 5 minutes
            s_maxage: 3600,      // CDN cache: 1 hour
            stale_while_revalidate: 86400, // Serve stale for 1 day while revalidating
        }
    }

    pub fn generate(&self, content_type: &str) -> Vec<(String, String)> {
        vec![
            ("Cache-Control".to_string(), format!(
                "public, max-age={}, s-maxage={}, stale-while-revalidate={}",
                self.max_age, self.s_maxage, self.stale_while_revalidate
            )),
            ("Content-Type".to_string(), content_type.to_string()),
            ("Vary".to_string(), "Accept-Encoding".to_string()),
            ("X-Cache-Status".to_string(), "HIT".to_string()),
        ]
    }

    pub fn generate_no_cache() -> Vec<(String, String)> {
        vec![
            ("Cache-Control".to_string(), "no-cache, no-store, must-revalidate".to_string()),
            ("Pragma".to_string(), "no-cache".to_string()),
            ("Expires".to_string(), "0".to_string()),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cache_operations() {
        let cache = CacheManager::new("redis://localhost:6379").await.unwrap();

        // Test set and get
        cache.set_feed("test_key", "test_value", 60).await;
        let result = cache.get_feed("test_key").await;
        assert_eq!(result, Some("test_value".to_string()));

        // Test invalidation
        cache.invalidate_feed("test_key").await;
        let result = cache.get_feed("test_key").await;
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn test_batch_operations() {
        let cache = CacheManager::new("redis://localhost:6379").await.unwrap();

        // Batch set
        let items = vec![
            ("key1".to_string(), "value1".to_string(), 60),
            ("key2".to_string(), "value2".to_string(), 60),
            ("key3".to_string(), "value3".to_string(), 60),
        ];
        cache.set_feeds_batch(items).await;

        // Batch get
        let keys = vec!["key1".to_string(), "key2".to_string(), "key3".to_string()];
        let results = cache.get_feeds_batch(keys).await;

        assert_eq!(results.len(), 3);
        assert_eq!(results[0], Some("value1".to_string()));
        assert_eq!(results[1], Some("value2".to_string()));
        assert_eq!(results[2], Some("value3".to_string()));
    }
}