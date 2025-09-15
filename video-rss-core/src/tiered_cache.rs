use crate::{error::VideoRssError, Result};
use async_trait::async_trait;
use bytes::Bytes;
use moka::future::Cache as MokaCache;
use rocksdb::{DB, Options as RocksOptions, WriteBatch};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use blake3::Hasher;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TieredCacheConfig {
    // L1 Cache (In-memory)
    pub l1_max_items: u64,
    pub l1_time_to_live: Duration,
    pub l1_time_to_idle: Duration,

    // L2 Cache (SSD/RocksDB)
    pub l2_path: PathBuf,
    pub l2_max_size_gb: u64,
    pub l2_compression: bool,
    pub l2_bloom_filter: bool,

    // L3 Cache (Disk/Memory-mapped)
    pub l3_path: PathBuf,
    pub l3_max_size_gb: u64,

    // Eviction and promotion policies
    pub promotion_threshold: u32,  // Access count before promotion
    pub eviction_policy: EvictionPolicy,
    pub prefetch_enabled: bool,
    pub write_through: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvictionPolicy {
    LRU,
    LFU,
    FIFO,
    ARC,  // Adaptive Replacement Cache
    TinyLFU,
}

impl Default for TieredCacheConfig {
    fn default() -> Self {
        Self {
            l1_max_items: 10_000,
            l1_time_to_live: Duration::from_secs(3600),
            l1_time_to_idle: Duration::from_secs(300),
            l2_path: PathBuf::from("./cache/l2"),
            l2_max_size_gb: 10,
            l2_compression: true,
            l2_bloom_filter: true,
            l3_path: PathBuf::from("./cache/l3"),
            l3_max_size_gb: 100,
            promotion_threshold: 3,
            eviction_policy: EvictionPolicy::TinyLFU,
            prefetch_enabled: true,
            write_through: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry<T> {
    pub data: T,
    pub key: String,
    pub size_bytes: usize,
    pub access_count: u32,
    pub created_at: u64,
    pub last_accessed: u64,
    pub ttl: Option<Duration>,
    pub tier: CacheTier,
    pub checksum: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CacheTier {
    L1Memory,
    L2SSD,
    L3Disk,
}

pub struct TieredCache {
    config: TieredCacheConfig,

    // L1: In-memory cache (Moka)
    l1_cache: MokaCache<String, Bytes>,

    // L2: SSD cache (RocksDB)
    l2_cache: Arc<DB>,

    // L3: Disk cache (Custom implementation)
    l3_cache: Arc<RwLock<DiskCache>>,

    // Metadata tracking
    access_stats: Arc<RwLock<AccessStatistics>>,

    // Prefetch queue
    prefetch_queue: Arc<RwLock<Vec<String>>>,
}

impl TieredCache {
    pub async fn new(config: TieredCacheConfig) -> Result<Self> {
        info!("Initializing tiered cache system");

        // Create directories
        std::fs::create_dir_all(&config.l2_path)
            .map_err(|e| VideoRssError::Io(e))?;
        std::fs::create_dir_all(&config.l3_path)
            .map_err(|e| VideoRssError::Io(e))?;

        // Initialize L1 cache
        let l1_cache = MokaCache::builder()
            .max_capacity(config.l1_max_items)
            .time_to_live(config.l1_time_to_live)
            .time_to_idle(config.l1_time_to_idle)
            .build();

        // Initialize L2 cache (RocksDB)
        let mut rocks_opts = RocksOptions::default();
        rocks_opts.create_if_missing(true);
        rocks_opts.set_compression_type(if config.l2_compression {
            rocksdb::DBCompressionType::Lz4
        } else {
            rocksdb::DBCompressionType::None
        });

        if config.l2_bloom_filter {
            rocks_opts.set_bloom_locality(10);
        }

        rocks_opts.set_max_open_files(1000);
        rocks_opts.set_use_direct_io_for_flush_and_compaction(true);

        let l2_cache = Arc::new(
            DB::open(&rocks_opts, &config.l2_path)
                .map_err(|e| VideoRssError::Config(format!("RocksDB error: {}", e)))?
        );

        // Initialize L3 cache
        let l3_cache = Arc::new(RwLock::new(
            DiskCache::new(&config.l3_path, config.l3_max_size_gb)?
        ));

        // Initialize statistics
        let access_stats = Arc::new(RwLock::new(AccessStatistics::default()));

        info!("Tiered cache initialized with {} tiers", 3);

        Ok(Self {
            config,
            l1_cache,
            l2_cache,
            l3_cache,
            access_stats,
            prefetch_queue: Arc::new(RwLock::new(Vec::new())),
        })
    }

    pub async fn get<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        let hash_key = self.hash_key(key);

        // Update access statistics
        self.record_access(&hash_key).await;

        // Try L1 cache first
        if let Some(data) = self.l1_cache.get(&hash_key).await {
            debug!("L1 cache hit for key: {}", key);
            self.update_stats(CacheTier::L1Memory, true).await;
            return Ok(Some(self.deserialize(&data)?));
        }

        // Try L2 cache
        if let Some(data) = self.get_from_l2(&hash_key).await? {
            debug!("L2 cache hit for key: {}", key);
            self.update_stats(CacheTier::L2SSD, true).await;

            // Promote to L1 if accessed frequently
            if self.should_promote(&hash_key).await {
                self.l1_cache.insert(hash_key.clone(), data.clone()).await;
            }

            return Ok(Some(self.deserialize(&data)?));
        }

        // Try L3 cache
        if let Some(data) = self.get_from_l3(&hash_key).await? {
            debug!("L3 cache hit for key: {}", key);
            self.update_stats(CacheTier::L3Disk, true).await;

            // Promote to L2 if accessed frequently
            if self.should_promote(&hash_key).await {
                self.set_l2(&hash_key, &data).await?;
            }

            return Ok(Some(self.deserialize(&data)?));
        }

        debug!("Cache miss for key: {}", key);
        self.update_stats(CacheTier::L1Memory, false).await;

        // Trigger prefetch if enabled
        if self.config.prefetch_enabled {
            self.queue_prefetch(key).await;
        }

        Ok(None)
    }

    pub async fn set<T: Serialize>(&self, key: &str, value: &T, ttl: Option<Duration>) -> Result<()> {
        let hash_key = self.hash_key(key);
        let data = self.serialize(value)?;
        let size = data.len();

        // Determine initial tier based on size and policy
        let tier = self.determine_tier(size);

        match tier {
            CacheTier::L1Memory => {
                self.l1_cache.insert(hash_key.clone(), data.clone()).await;

                if self.config.write_through {
                    self.set_l2(&hash_key, &data).await?;
                }
            },
            CacheTier::L2SSD => {
                self.set_l2(&hash_key, &data).await?;
            },
            CacheTier::L3Disk => {
                self.set_l3(&hash_key, &data).await?;
            },
        }

        Ok(())
    }

    pub async fn delete(&self, key: &str) -> Result<()> {
        let hash_key = self.hash_key(key);

        // Remove from all tiers
        self.l1_cache.remove(&hash_key).await;
        self.l2_cache.delete(&hash_key)
            .map_err(|e| VideoRssError::Unknown(format!("RocksDB delete error: {}", e)))?;
        self.l3_cache.write().await.delete(&hash_key)?;

        Ok(())
    }

    async fn get_from_l2(&self, key: &str) -> Result<Option<Bytes>> {
        match self.l2_cache.get(key) {
            Ok(Some(data)) => Ok(Some(Bytes::from(data))),
            Ok(None) => Ok(None),
            Err(e) => Err(VideoRssError::Unknown(format!("RocksDB get error: {}", e))),
        }
    }

    async fn set_l2(&self, key: &str, data: &Bytes) -> Result<()> {
        self.l2_cache
            .put(key, data)
            .map_err(|e| VideoRssError::Unknown(format!("RocksDB put error: {}", e)))
    }

    async fn get_from_l3(&self, key: &str) -> Result<Option<Bytes>> {
        self.l3_cache.read().await.get(key)
    }

    async fn set_l3(&self, key: &str, data: &Bytes) -> Result<()> {
        self.l3_cache.write().await.set(key, data)
    }

    fn hash_key(&self, key: &str) -> String {
        let mut hasher = Hasher::new();
        hasher.update(key.as_bytes());
        hasher.finalize().to_hex().to_string()
    }

    fn serialize<T: Serialize>(&self, value: &T) -> Result<Bytes> {
        let data = bincode::serialize(value)
            .map_err(|e| VideoRssError::Unknown(format!("Serialization error: {}", e)))?;
        Ok(Bytes::from(data))
    }

    fn deserialize<T: DeserializeOwned>(&self, data: &Bytes) -> Result<T> {
        bincode::deserialize(data)
            .map_err(|e| VideoRssError::Unknown(format!("Deserialization error: {}", e)))
    }

    fn determine_tier(&self, size: usize) -> CacheTier {
        if size < 1024 * 1024 {  // < 1MB
            CacheTier::L1Memory
        } else if size < 100 * 1024 * 1024 {  // < 100MB
            CacheTier::L2SSD
        } else {
            CacheTier::L3Disk
        }
    }

    async fn should_promote(&self, key: &str) -> bool {
        let stats = self.access_stats.read().await;
        stats.get_access_count(key) >= self.config.promotion_threshold
    }

    async fn record_access(&self, key: &str) {
        let mut stats = self.access_stats.write().await;
        stats.record_access(key);
    }

    async fn update_stats(&self, tier: CacheTier, hit: bool) {
        let mut stats = self.access_stats.write().await;
        stats.update(tier, hit);
    }

    async fn queue_prefetch(&self, key: &str) {
        let mut queue = self.prefetch_queue.write().await;
        queue.push(key.to_string());

        // Trigger prefetch if queue is large enough
        if queue.len() >= 10 {
            self.execute_prefetch(queue.clone()).await;
            queue.clear();
        }
    }

    async fn execute_prefetch(&self, keys: Vec<String>) {
        // Implement predictive prefetching based on access patterns
        tokio::spawn(async move {
            debug!("Executing prefetch for {} keys", keys.len());
            // Prefetch logic would go here
        });
    }

    pub async fn get_stats(&self) -> CacheStatistics {
        let stats = self.access_stats.read().await;
        let l1_size = self.l1_cache.entry_count();

        CacheStatistics {
            l1_items: l1_size,
            l1_hit_rate: stats.l1_hit_rate(),
            l2_hit_rate: stats.l2_hit_rate(),
            l3_hit_rate: stats.l3_hit_rate(),
            total_requests: stats.total_requests,
            total_hits: stats.total_hits,
            total_misses: stats.total_misses,
        }
    }

    pub async fn clear_tier(&self, tier: CacheTier) -> Result<()> {
        match tier {
            CacheTier::L1Memory => {
                self.l1_cache.invalidate_all().await;
            },
            CacheTier::L2SSD => {
                // Clear RocksDB
                let batch = WriteBatch::default();
                self.l2_cache.write(batch)
                    .map_err(|e| VideoRssError::Unknown(format!("RocksDB clear error: {}", e)))?;
            },
            CacheTier::L3Disk => {
                self.l3_cache.write().await.clear()?;
            },
        }
        Ok(())
    }
}

struct DiskCache {
    base_path: PathBuf,
    max_size_bytes: u64,
    current_size: Arc<RwLock<u64>>,
    index: Arc<RwLock<std::collections::HashMap<String, DiskCacheEntry>>>,
}

#[derive(Clone, Serialize, Deserialize)]
struct DiskCacheEntry {
    path: PathBuf,
    size: u64,
    created_at: u64,
    last_accessed: u64,
}

impl DiskCache {
    fn new(path: &Path, max_size_gb: u64) -> Result<Self> {
        std::fs::create_dir_all(path)
            .map_err(|e| VideoRssError::Io(e))?;

        Ok(Self {
            base_path: path.to_path_buf(),
            max_size_bytes: max_size_gb * 1024 * 1024 * 1024,
            current_size: Arc::new(RwLock::new(0)),
            index: Arc::new(RwLock::new(std::collections::HashMap::new())),
        })
    }

    fn get(&self, key: &str) -> Result<Option<Bytes>> {
        let path = self.get_path(key);

        if path.exists() {
            let data = std::fs::read(&path)
                .map_err(|e| VideoRssError::Io(e))?;
            Ok(Some(Bytes::from(data)))
        } else {
            Ok(None)
        }
    }

    fn set(&mut self, key: &str, data: &Bytes) -> Result<()> {
        let path = self.get_path(key);

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| VideoRssError::Io(e))?;
        }

        std::fs::write(&path, data)
            .map_err(|e| VideoRssError::Io(e))?;

        Ok(())
    }

    fn delete(&mut self, key: &str) -> Result<()> {
        let path = self.get_path(key);

        if path.exists() {
            std::fs::remove_file(&path)
                .map_err(|e| VideoRssError::Io(e))?;
        }

        Ok(())
    }

    fn clear(&mut self) -> Result<()> {
        std::fs::remove_dir_all(&self.base_path)
            .map_err(|e| VideoRssError::Io(e))?;
        std::fs::create_dir_all(&self.base_path)
            .map_err(|e| VideoRssError::Io(e))?;
        Ok(())
    }

    fn get_path(&self, key: &str) -> PathBuf {
        // Use first 2 chars as subdirectory for better file system performance
        let subdir = &key[..2.min(key.len())];
        self.base_path.join(subdir).join(key)
    }
}

#[derive(Default)]
struct AccessStatistics {
    access_counts: std::collections::HashMap<String, u32>,
    l1_hits: u64,
    l1_misses: u64,
    l2_hits: u64,
    l2_misses: u64,
    l3_hits: u64,
    l3_misses: u64,
    total_requests: u64,
    total_hits: u64,
    total_misses: u64,
}

impl AccessStatistics {
    fn record_access(&mut self, key: &str) {
        *self.access_counts.entry(key.to_string()).or_insert(0) += 1;
        self.total_requests += 1;
    }

    fn get_access_count(&self, key: &str) -> u32 {
        *self.access_counts.get(key).unwrap_or(&0)
    }

    fn update(&mut self, tier: CacheTier, hit: bool) {
        match (tier, hit) {
            (CacheTier::L1Memory, true) => {
                self.l1_hits += 1;
                self.total_hits += 1;
            },
            (CacheTier::L1Memory, false) => {
                self.l1_misses += 1;
                self.total_misses += 1;
            },
            (CacheTier::L2SSD, true) => {
                self.l2_hits += 1;
                self.total_hits += 1;
            },
            (CacheTier::L2SSD, false) => {
                self.l2_misses += 1;
                self.total_misses += 1;
            },
            (CacheTier::L3Disk, true) => {
                self.l3_hits += 1;
                self.total_hits += 1;
            },
            (CacheTier::L3Disk, false) => {
                self.l3_misses += 1;
                self.total_misses += 1;
            },
        }
    }

    fn l1_hit_rate(&self) -> f64 {
        if self.l1_hits + self.l1_misses == 0 {
            0.0
        } else {
            self.l1_hits as f64 / (self.l1_hits + self.l1_misses) as f64
        }
    }

    fn l2_hit_rate(&self) -> f64 {
        if self.l2_hits + self.l2_misses == 0 {
            0.0
        } else {
            self.l2_hits as f64 / (self.l2_hits + self.l2_misses) as f64
        }
    }

    fn l3_hit_rate(&self) -> f64 {
        if self.l3_hits + self.l3_misses == 0 {
            0.0
        } else {
            self.l3_hits as f64 / (self.l3_hits + self.l3_misses) as f64
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStatistics {
    pub l1_items: u64,
    pub l1_hit_rate: f64,
    pub l2_hit_rate: f64,
    pub l3_hit_rate: f64,
    pub total_requests: u64,
    pub total_hits: u64,
    pub total_misses: u64,
}