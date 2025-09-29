use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use redis::AsyncCommands;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{debug, info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationAction {
    pub timestamp: DateTime<Utc>,
    pub action_type: ActionType,
    pub component: String,
    pub resource: String,
    pub current_value: f64,
    pub target_value: f64,
    pub expected_improvement: f64,
    pub priority: OptimizationPriority,
    pub status: ActionStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionType {
    ScaleReplicas { from: i32, to: i32 },
    AdjustCpuLimits { from: f64, to: f64 },
    AdjustMemoryLimits { from_gb: f64, to_gb: f64 },
    OptimizeCacheSettings { setting: String, from: String, to: String },
    TuneConnectionPool { pool_name: String, from: i32, to: i32 },
    AdjustBatchSize { component: String, from: i32, to: i32 },
    ModifyTimeout { timeout_type: String, from_ms: i32, to_ms: i32 },
    EnableFeatureFlag { flag: String },
    DisableFeatureFlag { flag: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationPriority {
    Critical,   // Immediate performance impact
    High,       // Significant improvement expected
    Medium,     // Moderate improvement
    Low,        // Minor optimization
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    RolledBack,
}

#[derive(Debug, Clone)]
pub struct ResourceConfiguration {
    pub component: String,
    pub cpu_limit: f64,
    pub memory_limit_gb: f64,
    pub replica_count: i32,
    pub connection_pool_size: i32,
    pub batch_size: i32,
    pub timeout_ms: i32,
    pub cache_settings: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct OptimizationRule {
    pub name: String,
    pub condition: String,
    pub action: ActionType,
    pub priority: OptimizationPriority,
    pub cooldown_minutes: i32,
    pub max_attempts: i32,
}

pub struct ResourceOptimizer {
    redis: redis::aio::ConnectionManager,
    optimization_rules: Vec<OptimizationRule>,
    action_history: HashMap<String, DateTime<Utc>>,
    rollback_threshold: f64, // Performance degradation threshold for rollbacks
}

impl ResourceOptimizer {
    pub async fn new(redis_url: &str) -> Result<Self> {
        let client = redis::Client::open(redis_url)
            .context("Failed to create Redis client for optimizer")?;
        let redis = redis::aio::ConnectionManager::new(client)
            .await
            .context("Failed to create Redis connection manager")?;

        let optimization_rules = Self::load_default_rules();

        Ok(Self {
            redis,
            optimization_rules,
            action_history: HashMap::new(),
            rollback_threshold: -10.0, // 10% performance degradation triggers rollback
        })
    }

    fn load_default_rules() -> Vec<OptimizationRule> {
        vec![
            // CPU optimization rules
            OptimizationRule {
                name: "scale_up_high_cpu".to_string(),
                condition: "cpu_usage > 80 AND duration > 300".to_string(),
                action: ActionType::ScaleReplicas { from: 0, to: 0 }, // Will be dynamically set
                priority: OptimizationPriority::High,
                cooldown_minutes: 10,
                max_attempts: 3,
            },
            OptimizationRule {
                name: "scale_down_low_cpu".to_string(),
                condition: "cpu_usage < 20 AND duration > 600".to_string(),
                action: ActionType::ScaleReplicas { from: 0, to: 0 },
                priority: OptimizationPriority::Medium,
                cooldown_minutes: 30,
                max_attempts: 2,
            },
            OptimizationRule {
                name: "increase_cpu_limits".to_string(),
                condition: "cpu_throttling > 0.1 AND cpu_usage > 90".to_string(),
                action: ActionType::AdjustCpuLimits { from: 0.0, to: 0.0 },
                priority: OptimizationPriority::High,
                cooldown_minutes: 15,
                max_attempts: 3,
            },
            // Memory optimization rules
            OptimizationRule {
                name: "increase_memory_limits".to_string(),
                condition: "memory_usage > 85 AND oom_kills > 0".to_string(),
                action: ActionType::AdjustMemoryLimits { from_gb: 0.0, to_gb: 0.0 },
                priority: OptimizationPriority::Critical,
                cooldown_minutes: 5,
                max_attempts: 2,
            },
            OptimizationRule {
                name: "optimize_connection_pool".to_string(),
                condition: "connection_wait_time > 100 AND active_connections > pool_size * 0.8".to_string(),
                action: ActionType::TuneConnectionPool { pool_name: "".to_string(), from: 0, to: 0 },
                priority: OptimizationPriority::Medium,
                cooldown_minutes: 20,
                max_attempts: 2,
            },
            // Cache optimization rules
            OptimizationRule {
                name: "increase_cache_ttl".to_string(),
                condition: "cache_hit_rate < 70 AND cache_eviction_rate > 0.1".to_string(),
                action: ActionType::OptimizeCacheSettings {
                    setting: "ttl".to_string(),
                    from: "3600".to_string(),
                    to: "7200".to_string()
                },
                priority: OptimizationPriority::Medium,
                cooldown_minutes: 60,
                max_attempts: 2,
            },
            // Batch processing optimization
            OptimizationRule {
                name: "optimize_batch_size".to_string(),
                condition: "processing_latency > target_latency * 1.5 AND batch_utilization < 0.7".to_string(),
                action: ActionType::AdjustBatchSize { component: "".to_string(), from: 0, to: 0 },
                priority: OptimizationPriority::Medium,
                cooldown_minutes: 30,
                max_attempts: 3,
            },
            // Timeout optimization
            OptimizationRule {
                name: "reduce_timeout_errors".to_string(),
                condition: "timeout_error_rate > 0.05 AND avg_response_time < timeout * 0.8".to_string(),
                action: ActionType::ModifyTimeout { timeout_type: "request".to_string(), from_ms: 0, to_ms: 0 },
                priority: OptimizationPriority::High,
                cooldown_minutes: 15,
                max_attempts: 2,
            },
        ]
    }

    pub async fn optimize_resources(&self) -> Result<()> {
        info!("Starting resource optimization cycle");

        let current_metrics = self.get_current_metrics().await?;
        let component_configs = self.get_component_configurations().await?;

        for component_name in component_configs.keys() {
            if let Err(e) = self.optimize_component(component_name, &current_metrics, &component_configs).await {
                warn!("Failed to optimize component {}: {}", component_name, e);
            }
        }

        // Check for rollbacks needed
        self.check_and_rollback_failed_optimizations().await?;

        info!("Resource optimization cycle completed");
        Ok(())
    }

    async fn get_current_metrics(&self) -> Result<HashMap<String, HashMap<String, f64>>> {
        let mut conn = self.redis.clone();
        let keys: Vec<String> = conn.keys("metrics:*").await?;

        let mut metrics = HashMap::new();
        for key in keys {
            if let Ok(value) = conn.hgetall::<_, HashMap<String, String>>(&key).await {
                let component = key.replace("metrics:", "");
                let mut component_metrics = HashMap::new();

                for (metric_name, metric_value) in value {
                    if let Ok(parsed_value) = metric_value.parse::<f64>() {
                        component_metrics.insert(metric_name, parsed_value);
                    }
                }

                if !component_metrics.is_empty() {
                    metrics.insert(component, component_metrics);
                }
            }
        }

        Ok(metrics)
    }

    async fn get_component_configurations(&self) -> Result<HashMap<String, ResourceConfiguration>> {
        let mut conn = self.redis.clone();
        let keys: Vec<String> = conn.keys("config:*").await?;

        let mut configs = HashMap::new();
        for key in keys {
            if let Ok(config_data) = conn.get::<_, String>(&key).await {
                if let Ok(config) = serde_json::from_str::<ResourceConfiguration>(&config_data) {
                    let component = key.replace("config:", "");
                    configs.insert(component, config);
                }
            }
        }

        // Add default configurations for components without explicit config
        let default_components = vec!["api-server", "metadata-extractor", "transcription-service", "summarization-service"];
        for component in default_components {
            if !configs.contains_key(component) {
                configs.insert(component.to_string(), self.get_default_config(component));
            }
        }

        Ok(configs)
    }

    fn get_default_config(&self, component: &str) -> ResourceConfiguration {
        match component {
            "api-server" => ResourceConfiguration {
                component: component.to_string(),
                cpu_limit: 2.0,
                memory_limit_gb: 4.0,
                replica_count: 3,
                connection_pool_size: 50,
                batch_size: 10,
                timeout_ms: 30000,
                cache_settings: [
                    ("ttl".to_string(), "3600".to_string()),
                    ("max_size".to_string(), "10000".to_string()),
                ].into_iter().collect(),
            },
            "metadata-extractor" => ResourceConfiguration {
                component: component.to_string(),
                cpu_limit: 1.0,
                memory_limit_gb: 2.0,
                replica_count: 2,
                connection_pool_size: 20,
                batch_size: 5,
                timeout_ms: 60000,
                cache_settings: HashMap::new(),
            },
            "transcription-service" => ResourceConfiguration {
                component: component.to_string(),
                cpu_limit: 4.0,
                memory_limit_gb: 8.0,
                replica_count: 2,
                connection_pool_size: 10,
                batch_size: 1,
                timeout_ms: 300000,
                cache_settings: HashMap::new(),
            },
            "summarization-service" => ResourceConfiguration {
                component: component.to_string(),
                cpu_limit: 4.0,
                memory_limit_gb: 8.0,
                replica_count: 2,
                connection_pool_size: 10,
                batch_size: 2,
                timeout_ms: 180000,
                cache_settings: HashMap::new(),
            },
            _ => ResourceConfiguration {
                component: component.to_string(),
                cpu_limit: 1.0,
                memory_limit_gb: 2.0,
                replica_count: 1,
                connection_pool_size: 10,
                batch_size: 1,
                timeout_ms: 30000,
                cache_settings: HashMap::new(),
            }
        }
    }

    async fn optimize_component(
        &self,
        component: &str,
        metrics: &HashMap<String, HashMap<String, f64>>,
        configs: &HashMap<String, ResourceConfiguration>,
    ) -> Result<()> {
        let component_metrics = metrics.get(component);
        let component_config = configs.get(component);

        if component_metrics.is_none() || component_config.is_none() {
            debug!("Skipping optimization for {} - missing metrics or config", component);
            return Ok();
        }

        let metrics = component_metrics.unwrap();
        let config = component_config.unwrap();

        // Apply optimization rules
        for rule in &self.optimization_rules {
            if self.should_apply_rule(rule, component, metrics).await? {
                if let Some(action) = self.create_optimization_action(rule, component, config, metrics).await? {
                    self.execute_optimization_action(&action).await?;
                    self.record_action_history(component, &action.action_type).await;
                }
            }
        }

        Ok(())
    }

    async fn should_apply_rule(
        &self,
        rule: &OptimizationRule,
        component: &str,
        metrics: &HashMap<String, f64>,
    ) -> Result<bool> {
        // Check cooldown period
        let action_key = format!("{}_{}", component, rule.name);
        if let Some(last_action) = self.action_history.get(&action_key) {
            let cooldown = chrono::Duration::minutes(rule.cooldown_minutes as i64);
            if Utc::now() - *last_action < cooldown {
                return Ok(false);
            }
        }

        // Evaluate rule condition
        self.evaluate_condition(&rule.condition, metrics).await
    }

    async fn evaluate_condition(&self, condition: &str, metrics: &HashMap<String, f64>) -> Result<bool> {
        // Simple condition evaluator
        // In production, this would be a more sophisticated expression parser

        let conditions: Vec<&str> = condition.split(" AND ").collect();

        for cond in conditions {
            let cond = cond.trim();

            if cond.contains(" > ") {
                let parts: Vec<&str> = cond.split(" > ").collect();
                if parts.len() == 2 {
                    let metric_name = parts[0].trim();
                    let threshold: f64 = parts[1].trim().parse().unwrap_or(0.0);

                    if let Some(&value) = metrics.get(metric_name) {
                        if value <= threshold {
                            return Ok(false);
                        }
                    } else {
                        return Ok(false);
                    }
                }
            } else if cond.contains(" < ") {
                let parts: Vec<&str> = cond.split(" < ").collect();
                if parts.len() == 2 {
                    let metric_name = parts[0].trim();
                    let threshold: f64 = parts[1].trim().parse().unwrap_or(0.0);

                    if let Some(&value) = metrics.get(metric_name) {
                        if value >= threshold {
                            return Ok(false);
                        }
                    } else {
                        return Ok(false);
                    }
                }
            }
        }

        Ok(true)
    }

    async fn create_optimization_action(
        &self,
        rule: &OptimizationRule,
        component: &str,
        config: &ResourceConfiguration,
        metrics: &HashMap<String, f64>,
    ) -> Result<Option<OptimizationAction>> {
        let action_type = match &rule.action {
            ActionType::ScaleReplicas { .. } => {
                let cpu_usage = metrics.get("cpu_usage").unwrap_or(&0.0);
                if *cpu_usage > 80.0 {
                    ActionType::ScaleReplicas {
                        from: config.replica_count,
                        to: (config.replica_count + 1).min(10),
                    }
                } else if *cpu_usage < 20.0 && config.replica_count > 1 {
                    ActionType::ScaleReplicas {
                        from: config.replica_count,
                        to: (config.replica_count - 1).max(1),
                    }
                } else {
                    return Ok(None);
                }
            },
            ActionType::AdjustCpuLimits { .. } => {
                ActionType::AdjustCpuLimits {
                    from: config.cpu_limit,
                    to: (config.cpu_limit * 1.3).min(8.0),
                }
            },
            ActionType::AdjustMemoryLimits { .. } => {
                ActionType::AdjustMemoryLimits {
                    from_gb: config.memory_limit_gb,
                    to_gb: (config.memory_limit_gb * 1.3).min(16.0),
                }
            },
            ActionType::TuneConnectionPool { .. } => {
                ActionType::TuneConnectionPool {
                    pool_name: "database".to_string(),
                    from: config.connection_pool_size,
                    to: (config.connection_pool_size * 1.2) as i32,
                }
            },
            ActionType::AdjustBatchSize { .. } => {
                let processing_latency = metrics.get("processing_latency").unwrap_or(&0.0);
                if *processing_latency > 1000.0 { // High latency - reduce batch size
                    ActionType::AdjustBatchSize {
                        component: component.to_string(),
                        from: config.batch_size,
                        to: (config.batch_size / 2).max(1),
                    }
                } else { // Low latency - can increase batch size
                    ActionType::AdjustBatchSize {
                        component: component.to_string(),
                        from: config.batch_size,
                        to: config.batch_size * 2,
                    }
                }
            },
            ActionType::ModifyTimeout { .. } => {
                ActionType::ModifyTimeout {
                    timeout_type: "request".to_string(),
                    from_ms: config.timeout_ms,
                    to_ms: (config.timeout_ms as f64 * 1.5) as i32,
                }
            },
            other => other.clone(),
        };

        let expected_improvement = self.calculate_expected_improvement(&action_type, metrics);

        Ok(Some(OptimizationAction {
            timestamp: Utc::now(),
            action_type,
            component: component.to_string(),
            resource: self.get_resource_name(&rule.action),
            current_value: self.get_current_resource_value(&rule.action, config),
            target_value: self.get_target_resource_value(&action_type, config),
            expected_improvement,
            priority: rule.priority.clone(),
            status: ActionStatus::Pending,
        }))
    }

    fn calculate_expected_improvement(&self, action: &ActionType, metrics: &HashMap<String, f64>) -> f64 {
        match action {
            ActionType::ScaleReplicas { from, to } => {
                if *to > *from {
                    // Scaling up - expect CPU usage reduction
                    let cpu_usage = metrics.get("cpu_usage").unwrap_or(&50.0);
                    (cpu_usage * 0.2).min(30.0) // Max 30% improvement
                } else {
                    // Scaling down - expect resource savings
                    15.0
                }
            },
            ActionType::AdjustCpuLimits { .. } => 25.0,
            ActionType::AdjustMemoryLimits { .. } => 20.0,
            ActionType::TuneConnectionPool { .. } => 15.0,
            ActionType::AdjustBatchSize { .. } => 30.0,
            ActionType::ModifyTimeout { .. } => 10.0,
            ActionType::OptimizeCacheSettings { .. } => 40.0,
            _ => 10.0,
        }
    }

    fn get_resource_name(&self, action: &ActionType) -> String {
        match action {
            ActionType::ScaleReplicas { .. } => "replicas".to_string(),
            ActionType::AdjustCpuLimits { .. } => "cpu".to_string(),
            ActionType::AdjustMemoryLimits { .. } => "memory".to_string(),
            ActionType::TuneConnectionPool { .. } => "connection_pool".to_string(),
            ActionType::AdjustBatchSize { .. } => "batch_size".to_string(),
            ActionType::ModifyTimeout { .. } => "timeout".to_string(),
            ActionType::OptimizeCacheSettings { .. } => "cache".to_string(),
            _ => "unknown".to_string(),
        }
    }

    fn get_current_resource_value(&self, action: &ActionType, config: &ResourceConfiguration) -> f64 {
        match action {
            ActionType::ScaleReplicas { .. } => config.replica_count as f64,
            ActionType::AdjustCpuLimits { .. } => config.cpu_limit,
            ActionType::AdjustMemoryLimits { .. } => config.memory_limit_gb,
            ActionType::TuneConnectionPool { .. } => config.connection_pool_size as f64,
            ActionType::AdjustBatchSize { .. } => config.batch_size as f64,
            ActionType::ModifyTimeout { .. } => config.timeout_ms as f64,
            _ => 0.0,
        }
    }

    fn get_target_resource_value(&self, action: &ActionType, _config: &ResourceConfiguration) -> f64 {
        match action {
            ActionType::ScaleReplicas { to, .. } => *to as f64,
            ActionType::AdjustCpuLimits { to, .. } => *to,
            ActionType::AdjustMemoryLimits { to_gb, .. } => *to_gb,
            ActionType::TuneConnectionPool { to, .. } => *to as f64,
            ActionType::AdjustBatchSize { to, .. } => *to as f64,
            ActionType::ModifyTimeout { to_ms, .. } => *to_ms as f64,
            _ => 0.0,
        }
    }

    async fn execute_optimization_action(&self, action: &OptimizationAction) -> Result<()> {
        info!("Executing optimization action: {:?} for component {}", action.action_type, action.component);

        match &action.action_type {
            ActionType::ScaleReplicas { from: _, to } => {
                // In production, this would call Kubernetes API
                self.update_replica_count(&action.component, *to).await?;
            },
            ActionType::AdjustCpuLimits { from: _, to } => {
                self.update_cpu_limits(&action.component, *to).await?;
            },
            ActionType::AdjustMemoryLimits { from_gb: _, to_gb } => {
                self.update_memory_limits(&action.component, *to_gb).await?;
            },
            ActionType::TuneConnectionPool { pool_name, from: _, to } => {
                self.update_connection_pool(pool_name, *to).await?;
            },
            ActionType::AdjustBatchSize { component, from: _, to } => {
                self.update_batch_size(component, *to).await?;
            },
            ActionType::ModifyTimeout { timeout_type, from_ms: _, to_ms } => {
                self.update_timeout(timeout_type, *to_ms).await?;
            },
            ActionType::OptimizeCacheSettings { setting, from: _, to } => {
                self.update_cache_setting(&action.component, setting, to).await?;
            },
            ActionType::EnableFeatureFlag { flag } => {
                self.enable_feature_flag(flag).await?;
            },
            ActionType::DisableFeatureFlag { flag } => {
                self.disable_feature_flag(flag).await?;
            },
        }

        // Wait a moment for the change to take effect
        sleep(Duration::from_secs(30)).await;

        // Store action for tracking
        self.store_optimization_action(action).await?;

        Ok(())
    }

    async fn update_replica_count(&self, component: &str, count: i32) -> Result<()> {
        let mut conn = self.redis.clone();
        conn.hset(format!("config:{}", component), "replica_count", count).await?;
        info!("Updated replica count for {} to {}", component, count);
        Ok(())
    }

    async fn update_cpu_limits(&self, component: &str, limit: f64) -> Result<()> {
        let mut conn = self.redis.clone();
        conn.hset(format!("config:{}", component), "cpu_limit", limit).await?;
        info!("Updated CPU limit for {} to {}", component, limit);
        Ok(())
    }

    async fn update_memory_limits(&self, component: &str, limit_gb: f64) -> Result<()> {
        let mut conn = self.redis.clone();
        conn.hset(format!("config:{}", component), "memory_limit_gb", limit_gb).await?;
        info!("Updated memory limit for {} to {}GB", component, limit_gb);
        Ok(())
    }

    async fn update_connection_pool(&self, pool_name: &str, size: i32) -> Result<()> {
        let mut conn = self.redis.clone();
        conn.hset("connection_pools", pool_name, size).await?;
        info!("Updated connection pool {} to size {}", pool_name, size);
        Ok(())
    }

    async fn update_batch_size(&self, component: &str, size: i32) -> Result<()> {
        let mut conn = self.redis.clone();
        conn.hset(format!("config:{}", component), "batch_size", size).await?;
        info!("Updated batch size for {} to {}", component, size);
        Ok(())
    }

    async fn update_timeout(&self, timeout_type: &str, timeout_ms: i32) -> Result<()> {
        let mut conn = self.redis.clone();
        conn.hset("timeouts", timeout_type, timeout_ms).await?;
        info!("Updated {} timeout to {}ms", timeout_type, timeout_ms);
        Ok(())
    }

    async fn update_cache_setting(&self, component: &str, setting: &str, value: &str) -> Result<()> {
        let mut conn = self.redis.clone();
        conn.hset(format!("config:{}:cache", component), setting, value).await?;
        info!("Updated cache setting {} for {} to {}", setting, component, value);
        Ok(())
    }

    async fn enable_feature_flag(&self, flag: &str) -> Result<()> {
        let mut conn = self.redis.clone();
        conn.hset("feature_flags", flag, "enabled").await?;
        info!("Enabled feature flag: {}", flag);
        Ok(())
    }

    async fn disable_feature_flag(&self, flag: &str) -> Result<()> {
        let mut conn = self.redis.clone();
        conn.hset("feature_flags", flag, "disabled").await?;
        info!("Disabled feature flag: {}", flag);
        Ok(())
    }

    async fn store_optimization_action(&self, action: &OptimizationAction) -> Result<()> {
        let mut conn = self.redis.clone();
        let key = format!("optimization_actions:{}", Utc::now().timestamp());
        let serialized = serde_json::to_string(action)?;
        conn.setex(key, 86400, serialized).await?; // Store for 24 hours
        Ok(())
    }

    async fn record_action_history(&mut self, component: &str, action_type: &ActionType) {
        let action_key = format!("{}_{}", component, self.get_resource_name(action_type));
        self.action_history.insert(action_key, Utc::now());
    }

    async fn check_and_rollback_failed_optimizations(&self) -> Result<()> {
        let current_time = Utc::now();
        let check_window = chrono::Duration::minutes(15);

        let mut conn = self.redis.clone();
        let keys: Vec<String> = conn.keys("optimization_actions:*").await?;

        for key in keys {
            if let Ok(action_data) = conn.get::<_, String>(&key).await {
                if let Ok(action) = serde_json::from_str::<OptimizationAction>(&action_data) {
                    if current_time - action.timestamp < check_window {
                        if let Err(_) = self.check_action_effectiveness(&action).await {
                            warn!("Rolling back failed optimization for {}", action.component);
                            self.rollback_action(&action).await?;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    async fn check_action_effectiveness(&self, action: &OptimizationAction) -> Result<()> {
        let current_metrics = self.get_current_metrics().await?;

        if let Some(component_metrics) = current_metrics.get(&action.component) {
            // Check if performance degraded significantly
            let performance_change = self.calculate_performance_change(component_metrics, action);

            if performance_change < self.rollback_threshold {
                return Err(anyhow::anyhow!("Performance degraded by {:.2}%", performance_change.abs()));
            }
        }

        Ok(())
    }

    fn calculate_performance_change(&self, metrics: &HashMap<String, f64>, _action: &OptimizationAction) -> f64 {
        // Simple performance calculation based on key metrics
        let cpu_usage = metrics.get("cpu_usage").unwrap_or(&50.0);
        let memory_usage = metrics.get("memory_usage").unwrap_or(&50.0);
        let error_rate = metrics.get("error_rate").unwrap_or(&0.0);
        let response_time = metrics.get("avg_response_time").unwrap_or(&100.0);

        // Performance score (higher is better)
        let performance_score = 100.0 - (cpu_usage * 0.3 + memory_usage * 0.3 + error_rate * 10.0 + response_time * 0.01);

        // For now, return a positive change (would compare with baseline in production)
        performance_score - 50.0
    }

    async fn rollback_action(&self, action: &OptimizationAction) -> Result<()> {
        info!("Rolling back optimization action for {}", action.component);

        match &action.action_type {
            ActionType::ScaleReplicas { from, to: _ } => {
                self.update_replica_count(&action.component, *from).await?;
            },
            ActionType::AdjustCpuLimits { from, to: _ } => {
                self.update_cpu_limits(&action.component, *from).await?;
            },
            ActionType::AdjustMemoryLimits { from_gb, to_gb: _ } => {
                self.update_memory_limits(&action.component, *from_gb).await?;
            },
            ActionType::TuneConnectionPool { pool_name, from, to: _ } => {
                self.update_connection_pool(pool_name, *from).await?;
            },
            ActionType::AdjustBatchSize { component, from, to: _ } => {
                self.update_batch_size(component, *from).await?;
            },
            ActionType::ModifyTimeout { timeout_type, from_ms, to_ms: _ } => {
                self.update_timeout(timeout_type, *from_ms).await?;
            },
            _ => {}
        }

        Ok(())
    }

    pub async fn get_optimization_history(&self, hours: i32) -> Result<Vec<OptimizationAction>> {
        let mut conn = self.redis.clone();
        let keys: Vec<String> = conn.keys("optimization_actions:*").await?;

        let mut actions = Vec::new();
        let cutoff_time = Utc::now() - chrono::Duration::hours(hours as i64);

        for key in keys {
            if let Ok(action_data) = conn.get::<_, String>(&key).await {
                if let Ok(action) = serde_json::from_str::<OptimizationAction>(&action_data) {
                    if action.timestamp > cutoff_time {
                        actions.push(action);
                    }
                }
            }
        }

        actions.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        Ok(actions)
    }
}