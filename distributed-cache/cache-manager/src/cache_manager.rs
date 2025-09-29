use anyhow::{anyhow, Result};
use async_trait::async_trait;
use redis::aio::ClusterConnection;
use redis::cluster::ClusterClient;
use redis::{AsyncCommands, Script};
use scylla::{SessionBuilder, Session};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::partitioner::{Partitioner, PartitionStrategy};
use crate::replication::{ReplicationManager, ReplicationStrategy};
use crate::consistency::{ConsistencyLevel, ConsistencyChecker};

#[derive(Clone)]
pub struct CacheManager {
    redis_cluster: Arc<RwLock<ClusterConnection>>,
    cassandra_session: Arc<Session>,
    partitioner: Arc<Partitioner>,
    replication_manager: Arc<ReplicationManager>,
    consistency_checker: Arc<ConsistencyChecker>,
    config: Arc<CacheConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub redis_nodes: Vec<String>,
    pub cassandra_nodes: Vec<String>,
    pub replication_factor: usize,
    pub consistency_level: ConsistencyLevel,
    pub default_ttl: u64,
    pub max_memory_per_node: u64,
    pub eviction_policy: String,
    pub compression_enabled: bool,
    pub batch_size: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            redis_nodes: vec![
                "redis-master-1:7000".to_string(),
                "redis-master-2:7001".to_string(),
                "redis-master-3:7002".to_string(),
            ],
            cassandra_nodes: vec![
                "cassandra-1:9042".to_string(),
                "cassandra-2:9042".to_string(),
                "cassandra-3:9042".to_string(),
            ],
            replication_factor: 3,
            consistency_level: ConsistencyLevel::Quorum,
            default_ttl: 3600, // 1 hour
            max_memory_per_node: 4 * 1024 * 1024 * 1024, // 4GB
            eviction_policy: "allkeys-lru".to_string(),
            compression_enabled: true,
            batch_size: 1000,
        }
    }
}

impl CacheManager {
    pub async fn new() -> Result<Arc<Self>> {
        let config = Arc::new(CacheConfig::default());
        Self::with_config(config).await
    }

    pub async fn with_config(config: Arc<CacheConfig>) -> Result<Arc<Self>> {
        // Initialize Redis cluster
        let redis_cluster = Self::init_redis_cluster(&config).await?;

        // Initialize Cassandra session
        let cassandra_session = Self::init_cassandra_session(&config).await?;

        // Initialize components
        let partitioner = Arc::new(Partitioner::new(
            PartitionStrategy::ConsistentHash,
            config.redis_nodes.len(),
        ));

        let replication_manager = Arc::new(ReplicationManager::new(
            ReplicationStrategy::Quorum,
            config.replication_factor,
        ));

        let consistency_checker = Arc::new(ConsistencyChecker::new(
            config.consistency_level.clone(),
        ));

        Ok(Arc::new(Self {
            redis_cluster: Arc::new(RwLock::new(redis_cluster)),
            cassandra_session: Arc::new(cassandra_session),
            partitioner,
            replication_manager,
            consistency_checker,
            config,
        }))
    }

    async fn init_redis_cluster(config: &CacheConfig) -> Result<ClusterConnection> {
        let nodes: Vec<&str> = config.redis_nodes.iter().map(|s| s.as_str()).collect();
        let client = ClusterClient::open(nodes)?;
        let connection = client.get_async_connection().await?;
        info!("Connected to Redis cluster with {} nodes", config.redis_nodes.len());
        Ok(connection)
    }

    async fn init_cassandra_session(config: &CacheConfig) -> Result<Session> {
        let session = SessionBuilder::new()
            .known_nodes(&config.cassandra_nodes)
            .build()
            .await?;

        // Create keyspace and tables if not exists
        Self::init_cassandra_schema(&session).await?;

        info!("Connected to Cassandra cluster with {} nodes", config.cassandra_nodes.len());
        Ok(session)
    }

    async fn init_cassandra_schema(session: &Session) -> Result<()> {
        // Create keyspace
        session.query(
            "CREATE KEYSPACE IF NOT EXISTS video_rss WITH replication = {
                'class': 'NetworkTopologyStrategy',
                'datacenter1': 3
            } AND durable_writes = true",
            &[],
        ).await?;

        // Use keyspace
        session.use_keyspace("video_rss", false).await?;

        // Create tables
        session.query(
            "CREATE TABLE IF NOT EXISTS video_metadata (
                video_id text PRIMARY KEY,
                platform text,
                title text,
                duration int,
                metadata text,
                created_at timestamp,
                updated_at timestamp
            ) WITH compression = {'class': 'LZ4Compressor'}
              AND caching = {'keys': 'ALL', 'rows_per_partition': 'ALL'}",
            &[],
        ).await?;

        session.query(
            "CREATE TABLE IF NOT EXISTS transcriptions (
                video_id text,
                chunk_id int,
                text text,
                start_time float,
                end_time float,
                confidence float,
                created_at timestamp,
                PRIMARY KEY (video_id, chunk_id)
            ) WITH CLUSTERING ORDER BY (chunk_id ASC)
              AND compression = {'class': 'LZ4Compressor'}",
            &[],
        ).await?;

        session.query(
            "CREATE TABLE IF NOT EXISTS summaries (
                video_id text PRIMARY KEY,
                summary text,
                key_points list<text>,
                model_used text,
                quality_score float,
                created_at timestamp
            ) WITH compression = {'class': 'LZ4Compressor'}",
            &[],
        ).await?;

        Ok(())
    }

    // Core cache operations
    pub async fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        // Determine partition
        let partition = self.partitioner.get_partition(key);

        // Try Redis first
        let redis_result = self.get_from_redis(key, partition).await;

        match redis_result {
            Ok(Some(value)) => {
                debug!("Cache hit for key: {}", key);
                return Ok(Some(value));
            }
            Ok(None) => {
                debug!("Cache miss for key: {}", key);
            }
            Err(e) => {
                warn!("Redis error for key {}: {}", key, e);
            }
        }

        // Fallback to Cassandra
        let cassandra_result = self.get_from_cassandra(key).await?;

        // Write-through to Redis if found
        if let Some(ref value) = cassandra_result {
            let _ = self.set_in_redis(key, value.clone(), partition, Some(self.config.default_ttl)).await;
        }

        Ok(cassandra_result)
    }

    pub async fn set(&self, key: &str, value: Vec<u8>, ttl: Option<u64>) -> Result<()> {
        let partition = self.partitioner.get_partition(key);

        // Write to Redis with replication
        let replicas = self.replication_manager.get_replicas(partition);
        let mut write_futures = vec![];

        for replica in replicas {
            let redis = self.redis_cluster.clone();
            let key = key.to_string();
            let value = value.clone();
            let ttl = ttl.or(Some(self.config.default_ttl));

            write_futures.push(tokio::spawn(async move {
                Self::write_to_redis_replica(redis, &key, value, replica, ttl).await
            }));
        }

        // Wait for quorum writes
        let write_results: Vec<_> = futures::future::join_all(write_futures).await;
        let successful_writes = write_results.iter().filter(|r| r.is_ok()).count();

        if !self.consistency_checker.check_write_consistency(successful_writes, replicas.len()) {
            return Err(anyhow!("Failed to achieve write consistency"));
        }

        // Async write to Cassandra
        let cassandra = self.cassandra_session.clone();
        let key = key.to_string();
        let value = value.clone();

        tokio::spawn(async move {
            if let Err(e) = Self::write_to_cassandra(cassandra, &key, value).await {
                error!("Failed to write to Cassandra: {}", e);
            }
        });

        Ok(())
    }

    pub async fn delete(&self, key: &str) -> Result<()> {
        let partition = self.partitioner.get_partition(key);
        let replicas = self.replication_manager.get_replicas(partition);

        // Delete from all replicas
        for replica in replicas {
            let mut conn = self.redis_cluster.write().await;
            let _: Result<(), _> = conn.del(key).await;
        }

        // Delete from Cassandra
        self.cassandra_session
            .query("DELETE FROM video_rss.cache WHERE key = ?", (key,))
            .await?;

        Ok(())
    }

    pub async fn batch_get(&self, keys: &[String]) -> Result<Vec<Option<Vec<u8>>>> {
        let mut results = Vec::with_capacity(keys.len());

        // Group keys by partition for efficient retrieval
        let mut partitioned_keys: HashMap<usize, Vec<String>> = HashMap::new();
        for key in keys {
            let partition = self.partitioner.get_partition(key);
            partitioned_keys.entry(partition).or_default().push(key.clone());
        }

        // Fetch from each partition in parallel
        let mut fetch_futures = vec![];
        for (partition, partition_keys) in partitioned_keys {
            let redis = self.redis_cluster.clone();
            fetch_futures.push(tokio::spawn(async move {
                Self::batch_get_from_redis(redis, partition_keys).await
            }));
        }

        let fetch_results = futures::future::join_all(fetch_futures).await;

        // Combine results maintaining order
        let mut result_map = HashMap::new();
        for result in fetch_results {
            if let Ok(Ok(partition_results)) = result {
                result_map.extend(partition_results);
            }
        }

        for key in keys {
            results.push(result_map.remove(key));
        }

        Ok(results)
    }

    // Redis operations
    async fn get_from_redis(&self, key: &str, _partition: usize) -> Result<Option<Vec<u8>>> {
        let mut conn = self.redis_cluster.write().await;
        let value: Option<Vec<u8>> = conn.get(key).await?;
        Ok(value)
    }

    async fn set_in_redis(&self, key: &str, value: Vec<u8>, _partition: usize, ttl: Option<u64>) -> Result<()> {
        let mut conn = self.redis_cluster.write().await;

        if let Some(ttl_seconds) = ttl {
            conn.set_ex(key, value, ttl_seconds as usize).await?;
        } else {
            conn.set(key, value).await?;
        }

        Ok(())
    }

    async fn write_to_redis_replica(
        redis: Arc<RwLock<ClusterConnection>>,
        key: &str,
        value: Vec<u8>,
        _replica: usize,
        ttl: Option<u64>,
    ) -> Result<()> {
        let mut conn = redis.write().await;

        if let Some(ttl_seconds) = ttl {
            conn.set_ex(key, value, ttl_seconds as usize).await?;
        } else {
            conn.set(key, value).await?;
        }

        Ok(())
    }

    async fn batch_get_from_redis(
        redis: Arc<RwLock<ClusterConnection>>,
        keys: Vec<String>,
    ) -> Result<HashMap<String, Vec<u8>>> {
        let mut conn = redis.write().await;
        let mut pipe = redis::pipe();

        for key in &keys {
            pipe.get(key);
        }

        let values: Vec<Option<Vec<u8>>> = pipe.query_async(&mut *conn).await?;

        let mut results = HashMap::new();
        for (key, value) in keys.into_iter().zip(values) {
            if let Some(v) = value {
                results.insert(key, v);
            }
        }

        Ok(results)
    }

    // Cassandra operations
    async fn get_from_cassandra(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let rows = self.cassandra_session
            .query("SELECT value FROM video_rss.cache WHERE key = ?", (key,))
            .await?;

        if let Some(row) = rows.rows()?.next() {
            let value: Vec<u8> = row.get(0)?;
            Ok(Some(value))
        } else {
            Ok(None)
        }
    }

    async fn write_to_cassandra(
        session: Arc<Session>,
        key: &str,
        value: Vec<u8>,
    ) -> Result<()> {
        session
            .query(
                "INSERT INTO video_rss.cache (key, value, created_at) VALUES (?, ?, ?)",
                (key, value, SystemTime::now()),
            )
            .await?;
        Ok(())
    }

    // Management operations
    pub async fn get_partition_info(&self, key: &str) -> Result<serde_json::Value> {
        let partition = self.partitioner.get_partition(key);
        let replicas = self.replication_manager.get_replicas(partition);

        Ok(serde_json::json!({
            "key": key,
            "partition": partition,
            "replicas": replicas,
            "strategy": format!("{:?}", self.partitioner.strategy),
        }))
    }

    pub async fn list_shards(&self) -> Result<Vec<serde_json::Value>> {
        let shards = self.partitioner.list_all_shards();
        Ok(shards.into_iter().map(|s| serde_json::json!(s)).collect())
    }

    pub async fn warm_cache(&self, pattern: &str, source: &str) -> Result<()> {
        info!("Warming cache with pattern: {} from source: {}", pattern, source);

        match source {
            "cassandra" => {
                // Implement cache warming from Cassandra
                let rows = self.cassandra_session
                    .query(
                        "SELECT key, value FROM video_rss.cache WHERE key LIKE ? LIMIT 1000",
                        (format!("{}%", pattern),),
                    )
                    .await?;

                for row in rows.rows()? {
                    let key: String = row.get(0)?;
                    let value: Vec<u8> = row.get(1)?;
                    let _ = self.set(&key, value, Some(self.config.default_ttl)).await;
                }
            }
            _ => {
                warn!("Unknown cache warming source: {}", source);
            }
        }

        Ok(())
    }

    pub async fn preload_hot_data(&self) -> Result<()> {
        info!("Preloading hot data into cache");

        // Query for frequently accessed items
        let rows = self.cassandra_session
            .query(
                "SELECT key, value FROM video_rss.cache
                 WHERE access_count > 100
                 ORDER BY access_count DESC
                 LIMIT 1000",
                &[],
            )
            .await?;

        for row in rows.rows()? {
            let key: String = row.get(0)?;
            let value: Vec<u8> = row.get(1)?;
            let _ = self.set(&key, value, Some(self.config.default_ttl * 2)).await;
        }

        Ok(())
    }

    pub async fn replicate_all(&self) -> Result<()> {
        self.replication_manager.replicate_all_partitions().await
    }

    pub async fn check_consistency(&self, key: &str) -> Result<bool> {
        let partition = self.partitioner.get_partition(key);
        let replicas = self.replication_manager.get_replicas(partition);

        let mut values = vec![];
        for _replica in replicas {
            if let Ok(Some(value)) = self.get_from_redis(key, partition).await {
                values.push(value);
            }
        }

        // Check if all values are identical
        if values.is_empty() {
            return Ok(true);
        }

        let first = &values[0];
        Ok(values.iter().all(|v| v == first))
    }

    pub async fn get_metrics(&self) -> Result<String> {
        // Implement metrics collection
        Ok("# HELP cache_hits Total number of cache hits\n# TYPE cache_hits counter\ncache_hits 0\n".to_string())
    }

    pub async fn get_stats(&self) -> Result<serde_json::Value> {
        let mut conn = self.redis_cluster.write().await;
        let info: String = redis::cmd("INFO").query_async(&mut *conn).await?;

        Ok(serde_json::json!({
            "redis_info": info,
            "partitions": self.partitioner.get_partition_count(),
            "replication_factor": self.config.replication_factor,
            "consistency_level": format!("{:?}", self.config.consistency_level),
        }))
    }

    pub async fn expire_old_keys(&self) -> Result<u64> {
        // Implement key expiration logic
        let mut conn = self.redis_cluster.write().await;
        let script = Script::new(
            r"
            local cursor = '0'
            local count = 0
            repeat
                local result = redis.call('SCAN', cursor, 'COUNT', 1000)
                cursor = result[1]
                local keys = result[2]
                for i, key in ipairs(keys) do
                    local ttl = redis.call('TTL', key)
                    if ttl == -1 then
                        redis.call('EXPIRE', key, 3600)
                        count = count + 1
                    end
                end
            until cursor == '0'
            return count
            "
        );

        let expired: u64 = script.invoke_async(&mut *conn, &[]).await?;
        Ok(expired)
    }

    pub async fn cleanup(&self) -> Result<()> {
        info!("Running cache cleanup");

        // Clean up expired keys
        let expired = self.expire_old_keys().await?;
        info!("Expired {} keys", expired);

        // Compact Cassandra
        self.cassandra_session
            .query("TRUNCATE video_rss.cache_expired", &[])
            .await?;

        Ok(())
    }

    pub async fn backup(&self) -> Result<String> {
        let backup_id = format!("backup_{}", chrono::Utc::now().timestamp());
        info!("Creating backup: {}", backup_id);

        // Trigger Redis BGSAVE
        let mut conn = self.redis_cluster.write().await;
        let _: String = redis::cmd("BGSAVE").query_async(&mut *conn).await?;

        // Trigger Cassandra snapshot
        self.cassandra_session
            .query(
                "INSERT INTO video_rss.backups (id, created_at) VALUES (?, ?)",
                (backup_id.clone(), SystemTime::now()),
            )
            .await?;

        Ok(backup_id)
    }
}