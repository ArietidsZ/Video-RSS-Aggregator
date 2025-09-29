use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::time::{interval, Duration};
use tracing::{debug, info, warn};

use crate::cache_manager::CacheManager;

pub struct CacheWarmer {
    cache_manager: Arc<CacheManager>,
    warming_queue: Arc<tokio::sync::RwLock<VecDeque<WarmingTask>>>,
    warming_history: Arc<tokio::sync::RwLock<Vec<WarmingResult>>>,
    config: WarmingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmingConfig {
    pub enable_auto_warming: bool,
    pub warming_interval_secs: u64,
    pub max_concurrent_warmings: usize,
    pub priority_threshold: i32,
    pub ttl_extension_factor: f64,
    pub prefetch_related: bool,
}

impl Default for WarmingConfig {
    fn default() -> Self {
        Self {
            enable_auto_warming: true,
            warming_interval_secs: 300, // 5 minutes
            max_concurrent_warmings: 10,
            priority_threshold: 5,
            ttl_extension_factor: 1.5,
            prefetch_related: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmingTask {
    pub id: String,
    pub pattern: String,
    pub source: WarmingSource,
    pub priority: i32,
    pub created_at: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WarmingSource {
    PredictiveAnalysis,
    AccessPattern,
    ScheduledJob,
    UserRequest,
    RelatedContent,
}

#[derive(Debug, Clone, Serialize)]
pub struct WarmingResult {
    pub task_id: String,
    pub keys_warmed: usize,
    pub bytes_loaded: usize,
    pub duration_ms: u64,
    pub success: bool,
    pub error: Option<String>,
    pub timestamp: DateTime<Utc>,
}

impl CacheWarmer {
    pub fn new(cache_manager: Arc<CacheManager>) -> Self {
        Self {
            cache_manager,
            warming_queue: Arc::new(tokio::sync::RwLock::new(VecDeque::new())),
            warming_history: Arc::new(tokio::sync::RwLock::new(Vec::new())),
            config: WarmingConfig::default(),
        }
    }

    pub async fn start(&self) {
        if !self.config.enable_auto_warming {
            info!("Cache warming disabled");
            return;
        }

        info!("Starting cache warmer with {} second interval",
              self.config.warming_interval_secs);

        let mut ticker = interval(Duration::from_secs(self.config.warming_interval_secs));

        loop {
            ticker.tick().await;

            if let Err(e) = self.run_warming_cycle().await {
                warn!("Warming cycle failed: {}", e);
            }
        }
    }

    async fn run_warming_cycle(&self) -> Result<()> {
        debug!("Running cache warming cycle");

        // Analyze access patterns
        let patterns = self.analyze_access_patterns().await?;

        // Generate warming tasks
        for pattern in patterns {
            self.enqueue_task(pattern).await?;
        }

        // Process warming queue
        self.process_queue().await?;

        // Clean up old history
        self.cleanup_history().await;

        Ok(())
    }

    async fn analyze_access_patterns(&self) -> Result<Vec<WarmingTask>> {
        let mut tasks = Vec::new();

        // Analyze recent access patterns
        // In production, this would query actual access logs
        tasks.push(WarmingTask {
            id: uuid::Uuid::new_v4().to_string(),
            pattern: "video:trending:*".to_string(),
            source: WarmingSource::AccessPattern,
            priority: 10,
            created_at: Utc::now(),
            metadata: HashMap::new(),
        });

        tasks.push(WarmingTask {
            id: uuid::Uuid::new_v4().to_string(),
            pattern: "transcript:recent:*".to_string(),
            source: WarmingSource::PredictiveAnalysis,
            priority: 8,
            created_at: Utc::now(),
            metadata: HashMap::new(),
        });

        // Scheduled warming for known hot data
        tasks.push(WarmingTask {
            id: uuid::Uuid::new_v4().to_string(),
            pattern: "summary:popular:*".to_string(),
            source: WarmingSource::ScheduledJob,
            priority: 7,
            created_at: Utc::now(),
            metadata: HashMap::new(),
        });

        Ok(tasks)
    }

    pub async fn enqueue_task(&self, task: WarmingTask) -> Result<()> {
        let mut queue = self.warming_queue.write().await;

        // Insert based on priority
        let position = queue
            .iter()
            .position(|t| t.priority < task.priority)
            .unwrap_or(queue.len());

        queue.insert(position, task);

        // Limit queue size
        while queue.len() > 1000 {
            queue.pop_back();
        }

        Ok(())
    }

    async fn process_queue(&self) -> Result<()> {
        let tasks_to_process = {
            let mut queue = self.warming_queue.write().await;
            let mut tasks = Vec::new();

            for _ in 0..self.config.max_concurrent_warmings {
                if let Some(task) = queue.pop_front() {
                    if task.priority >= self.config.priority_threshold {
                        tasks.push(task);
                    }
                }
            }

            tasks
        };

        // Process tasks concurrently
        let mut futures = Vec::new();
        for task in tasks_to_process {
            let cache = self.cache_manager.clone();
            let config = self.config.clone();
            futures.push(tokio::spawn(async move {
                Self::warm_cache_for_task(cache, task, config).await
            }));
        }

        // Collect results
        let results = futures::future::join_all(futures).await;

        // Store results in history
        let mut history = self.warming_history.write().await;
        for result in results {
            if let Ok(Ok(warming_result)) = result {
                history.push(warming_result);
            }
        }

        Ok(())
    }

    async fn warm_cache_for_task(
        cache: Arc<CacheManager>,
        task: WarmingTask,
        config: WarmingConfig,
    ) -> Result<WarmingResult> {
        let start = std::time::Instant::now();
        let mut keys_warmed = 0;
        let mut bytes_loaded = 0;

        info!("Warming cache for pattern: {}", task.pattern);

        // Warm cache based on pattern
        match cache.warm_cache(&task.pattern, "cassandra").await {
            Ok(_) => {
                // In production, track actual metrics
                keys_warmed = 100; // Simulated
                bytes_loaded = 1024 * 1024; // 1MB simulated

                // Prefetch related content if enabled
                if config.prefetch_related {
                    Self::prefetch_related_content(cache.clone(), &task.pattern).await?;
                }
            }
            Err(e) => {
                return Ok(WarmingResult {
                    task_id: task.id,
                    keys_warmed: 0,
                    bytes_loaded: 0,
                    duration_ms: start.elapsed().as_millis() as u64,
                    success: false,
                    error: Some(e.to_string()),
                    timestamp: Utc::now(),
                });
            }
        }

        Ok(WarmingResult {
            task_id: task.id,
            keys_warmed,
            bytes_loaded,
            duration_ms: start.elapsed().as_millis() as u64,
            success: true,
            error: None,
            timestamp: Utc::now(),
        })
    }

    async fn prefetch_related_content(cache: Arc<CacheManager>, pattern: &str) -> Result<()> {
        debug!("Prefetching related content for pattern: {}", pattern);

        // Extract base key from pattern
        if pattern.starts_with("video:") {
            // If warming video metadata, also warm related transcripts and summaries
            let video_id = pattern.replace("video:", "").replace("*", "");

            let _ = cache.warm_cache(&format!("transcript:{}:*", video_id), "cassandra").await;
            let _ = cache.warm_cache(&format!("summary:{}", video_id), "cassandra").await;
        } else if pattern.starts_with("transcript:") {
            // If warming transcript, also warm the summary
            let video_id = pattern
                .replace("transcript:", "")
                .split(':')
                .next()
                .unwrap_or("")
                .to_string();

            let _ = cache.warm_cache(&format!("summary:{}", video_id), "cassandra").await;
        }

        Ok(())
    }

    async fn cleanup_history(&self) {
        let mut history = self.warming_history.write().await;

        // Keep only last 24 hours of history
        let cutoff = Utc::now() - chrono::Duration::hours(24);
        history.retain(|result| result.timestamp > cutoff);

        // Limit to 10,000 entries
        while history.len() > 10000 {
            history.remove(0);
        }
    }

    pub async fn get_warming_stats(&self) -> WarmingStats {
        let queue = self.warming_queue.read().await;
        let history = self.warming_history.read().await;

        let successful = history.iter().filter(|r| r.success).count();
        let failed = history.iter().filter(|r| !r.success).count();

        let total_keys_warmed: usize = history.iter().map(|r| r.keys_warmed).sum();
        let total_bytes_loaded: usize = history.iter().map(|r| r.bytes_loaded).sum();
        let avg_duration_ms = if !history.is_empty() {
            history.iter().map(|r| r.duration_ms).sum::<u64>() / history.len() as u64
        } else {
            0
        };

        WarmingStats {
            queue_size: queue.len(),
            successful_warmings: successful,
            failed_warmings: failed,
            total_keys_warmed,
            total_bytes_loaded,
            avg_duration_ms,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct WarmingStats {
    pub queue_size: usize,
    pub successful_warmings: usize,
    pub failed_warmings: usize,
    pub total_keys_warmed: usize,
    pub total_bytes_loaded: usize,
    pub avg_duration_ms: u64,
}

// Use uuid for task IDs
use uuid;