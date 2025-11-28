use anyhow::{Context, Result};
use chrono::{DateTime, Duration, Utc};
use governor::{
    clock::DefaultClock,
    state::{InMemoryState, NotKeyed},
    Quota, RateLimiter,
};
use nonzero_ext::*;
use redis::AsyncCommands;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitRule {
    pub id: Uuid,
    pub name: String,
    pub resource: String,
    pub method: Option<String>,
    pub path_pattern: Option<String>,
    pub limit_type: LimitType,
    pub window_seconds: u32,
    pub max_requests: u32,
    pub burst_size: Option<u32>,
    pub enabled: bool,
    pub priority: i32,
    pub actions: Vec<LimitAction>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LimitType {
    PerIP,
    PerUser,
    PerAPIKey,
    Global,
    PerUserIP,
    PerEndpoint,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LimitAction {
    Block,
    Throttle { delay_ms: u64 },
    RequireCaptcha,
    RequireAuth,
    TemporaryBan { duration_minutes: u32 },
    Alert { webhook_url: String },
    Log { level: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitStatus {
    pub key: String,
    pub limit: u32,
    pub remaining: u32,
    pub reset_time: DateTime<Utc>,
    pub retry_after: Option<u64>,
    pub blocked: bool,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockedEntity {
    pub key: String,
    pub entity_type: String,
    pub reason: String,
    pub blocked_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub block_count: u32,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitViolation {
    pub id: Uuid,
    pub rule_id: Uuid,
    pub key: String,
    pub endpoint: String,
    pub method: String,
    pub ip_address: Option<IpAddr>,
    pub user_id: Option<Uuid>,
    pub violation_count: u32,
    pub action_taken: LimitAction,
    pub timestamp: DateTime<Utc>,
    pub user_agent: Option<String>,
    pub request_headers: Option<serde_json::Value>,
}

pub struct RateLimitService {
    redis: redis::aio::ConnectionManager,
    rules: Arc<RwLock<Vec<RateLimitRule>>>,
    limiters: Arc<RwLock<HashMap<String, Arc<RateLimiter<NotKeyed, InMemoryState, DefaultClock>>>>>,
    blocked_entities: Arc<RwLock<HashMap<String, BlockedEntity>>>,
    violations: Arc<RwLock<Vec<RateLimitViolation>>>,
}

impl RateLimitService {
    pub async fn new(redis_url: &str) -> Result<Self> {
        let redis_client = redis::Client::open(redis_url)
            .context("Failed to create Redis client for rate limiting")?;
        let redis = redis::aio::ConnectionManager::new(redis_client)
            .await
            .context("Failed to create Redis connection for rate limiting")?;

        let rules = Arc::new(RwLock::new(Self::create_default_rules()));
        let limiters = Arc::new(RwLock::new(HashMap::new()));
        let blocked_entities = Arc::new(RwLock::new(HashMap::new()));
        let violations = Arc::new(RwLock::new(Vec::new()));

        let service = Self {
            redis,
            rules,
            limiters,
            blocked_entities,
            violations,
        };

        // Initialize limiters for default rules
        service.initialize_limiters().await?;

        info!("Rate limiting service initialized");

        Ok(service)
    }

    fn create_default_rules() -> Vec<RateLimitRule> {
        vec![
            // Global API rate limit
            RateLimitRule {
                id: Uuid::new_v4(),
                name: "Global API Limit".to_string(),
                resource: "api".to_string(),
                method: None,
                path_pattern: Some("/api/*".to_string()),
                limit_type: LimitType::PerIP,
                window_seconds: 3600, // 1 hour
                max_requests: 10000,
                burst_size: Some(100),
                enabled: true,
                priority: 1000,
                actions: vec![
                    LimitAction::Throttle { delay_ms: 1000 },
                    LimitAction::Log { level: "warn".to_string() },
                ],
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },

            // Authentication endpoints - stricter limits
            RateLimitRule {
                id: Uuid::new_v4(),
                name: "Authentication Rate Limit".to_string(),
                resource: "auth".to_string(),
                method: Some("POST".to_string()),
                path_pattern: Some("/auth/*".to_string()),
                limit_type: LimitType::PerIP,
                window_seconds: 900, // 15 minutes
                max_requests: 10,
                burst_size: Some(3),
                enabled: true,
                priority: 100,
                actions: vec![
                    LimitAction::Block,
                    LimitAction::TemporaryBan { duration_minutes: 15 },
                    LimitAction::Log { level: "error".to_string() },
                ],
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },

            // Login attempts - very strict
            RateLimitRule {
                id: Uuid::new_v4(),
                name: "Login Attempts".to_string(),
                resource: "login".to_string(),
                method: Some("POST".to_string()),
                path_pattern: Some("/auth/login".to_string()),
                limit_type: LimitType::PerIP,
                window_seconds: 300, // 5 minutes
                max_requests: 5,
                burst_size: Some(2),
                enabled: true,
                priority: 50,
                actions: vec![
                    LimitAction::Block,
                    LimitAction::TemporaryBan { duration_minutes: 30 },
                    LimitAction::Alert { webhook_url: "https://alerts.videorss.com/rate-limit".to_string() },
                ],
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },

            // Password reset - prevent abuse
            RateLimitRule {
                id: Uuid::new_v4(),
                name: "Password Reset".to_string(),
                resource: "password_reset".to_string(),
                method: Some("POST".to_string()),
                path_pattern: Some("/auth/reset-password".to_string()),
                limit_type: LimitType::PerIP,
                window_seconds: 3600, // 1 hour
                max_requests: 3,
                burst_size: Some(1),
                enabled: true,
                priority: 75,
                actions: vec![
                    LimitAction::Block,
                    LimitAction::RequireCaptcha,
                    LimitAction::Log { level: "warn".to_string() },
                ],
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },

            // Feed creation/modification
            RateLimitRule {
                id: Uuid::new_v4(),
                name: "Feed Operations".to_string(),
                resource: "feeds".to_string(),
                method: Some("POST".to_string()),
                path_pattern: Some("/api/feeds".to_string()),
                limit_type: LimitType::PerUser,
                window_seconds: 3600, // 1 hour
                max_requests: 100,
                burst_size: Some(10),
                enabled: true,
                priority: 200,
                actions: vec![
                    LimitAction::Throttle { delay_ms: 2000 },
                    LimitAction::Log { level: "info".to_string() },
                ],
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },

            // Video processing requests
            RateLimitRule {
                id: Uuid::new_v4(),
                name: "Video Processing".to_string(),
                resource: "video_processing".to_string(),
                method: Some("POST".to_string()),
                path_pattern: Some("/api/videos/process".to_string()),
                limit_type: LimitType::PerUser,
                window_seconds: 3600, // 1 hour
                max_requests: 50,
                burst_size: Some(5),
                enabled: true,
                priority: 150,
                actions: vec![
                    LimitAction::Throttle { delay_ms: 5000 },
                    LimitAction::RequireAuth,
                    LimitAction::Log { level: "info".to_string() },
                ],
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },

            // Heavy API endpoints
            RateLimitRule {
                id: Uuid::new_v4(),
                name: "Heavy API Operations".to_string(),
                resource: "heavy_api".to_string(),
                method: None,
                path_pattern: Some("/api/heavy/*".to_string()),
                limit_type: LimitType::PerUser,
                window_seconds: 300, // 5 minutes
                max_requests: 10,
                burst_size: Some(2),
                enabled: true,
                priority: 300,
                actions: vec![
                    LimitAction::Throttle { delay_ms: 10000 },
                    LimitAction::RequireAuth,
                ],
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },

            // Public RSS feeds - more permissive
            RateLimitRule {
                id: Uuid::new_v4(),
                name: "RSS Feed Access".to_string(),
                resource: "rss".to_string(),
                method: Some("GET".to_string()),
                path_pattern: Some("/rss/*".to_string()),
                limit_type: LimitType::PerIP,
                window_seconds: 60, // 1 minute
                max_requests: 60,
                burst_size: Some(10),
                enabled: true,
                priority: 500,
                actions: vec![
                    LimitAction::Throttle { delay_ms: 500 },
                ],
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },

            // Aggressive behavior detection
            RateLimitRule {
                id: Uuid::new_v4(),
                name: "Aggressive Behavior".to_string(),
                resource: "aggressive".to_string(),
                method: None,
                path_pattern: None,
                limit_type: LimitType::PerIP,
                window_seconds: 60, // 1 minute
                max_requests: 1000, // Very high threshold
                burst_size: Some(100),
                enabled: true,
                priority: 10,
                actions: vec![
                    LimitAction::Block,
                    LimitAction::TemporaryBan { duration_minutes: 60 },
                    LimitAction::Alert { webhook_url: "https://security.videorss.com/aggressive-behavior".to_string() },
                ],
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },
        ]
    }

    async fn initialize_limiters(&self) -> Result<()> {
        let rules = self.rules.read().await;
        let mut limiters = self.limiters.write().await;

        for rule in rules.iter() {
            if rule.enabled {
                let max_req = std::num::NonZeroU32::new(rule.max_requests).unwrap_or(nonzero!(1u32));
                let burst = std::num::NonZeroU32::new(rule.burst_size.unwrap_or(rule.max_requests / 10)).unwrap_or(nonzero!(1u32));
                let quota = Quota::per_second(max_req)
                    .allow_burst(burst);

                let limiter = Arc::new(RateLimiter::direct(quota));
                limiters.insert(rule.id.to_string(), limiter);
            }
        }

        info!("Initialized {} rate limiters", limiters.len());
        Ok(())
    }

    pub async fn check_rate_limit(&self, request_info: &RateLimitRequest) -> Result<RateLimitStatus> {
        let rules = self.rules.read().await;

        // Find applicable rules (sorted by priority)
        let mut applicable_rules: Vec<&RateLimitRule> = rules
            .iter()
            .filter(|rule| self.rule_matches(rule, request_info))
            .collect();

        applicable_rules.sort_by_key(|rule| rule.priority);

        // Check against each rule
        for rule in applicable_rules {
            let key = self.generate_rate_limit_key(rule, request_info);

            // Check if entity is blocked
            if let Some(blocked) = self.is_blocked(&key).await {
                return Ok(RateLimitStatus {
                    key,
                    limit: rule.max_requests,
                    remaining: 0,
                    reset_time: blocked.expires_at.unwrap_or_else(|| Utc::now() + Duration::hours(1)),
                    retry_after: Some(3600),
                    blocked: true,
                    reason: Some(blocked.reason),
                });
            }

            // Check rate limit
            let status = self.check_rule_limit(rule, &key, request_info).await?;

            if status.blocked {
                // Handle violation
                self.handle_rate_limit_violation(rule, &key, request_info).await?;
                return Ok(status);
            }

            // If we've exceeded this rule's limit, we don't need to check lower priority rules
            if status.remaining == 0 {
                return Ok(status);
            }
        }

        // No limits exceeded
        Ok(RateLimitStatus {
            key: "global".to_string(),
            limit: 10000,
            remaining: 10000,
            reset_time: Utc::now() + Duration::hours(1),
            retry_after: None,
            blocked: false,
            reason: None,
        })
    }

    fn rule_matches(&self, rule: &RateLimitRule, request: &RateLimitRequest) -> bool {
        // Check method
        if let Some(method) = &rule.method {
            if method != &request.method {
                return false;
            }
        }

        // Check path pattern
        if let Some(pattern) = &rule.path_pattern {
            if !self.path_matches_pattern(&request.path, pattern) {
                return false;
            }
        }

        // Check resource
        if rule.resource != "global" && rule.resource != request.resource {
            return false;
        }

        true
    }

    fn path_matches_pattern(&self, path: &str, pattern: &str) -> bool {
        if pattern.ends_with('*') {
            let prefix = &pattern[..pattern.len() - 1];
            path.starts_with(prefix)
        } else {
            path == pattern
        }
    }

    fn generate_rate_limit_key(&self, rule: &RateLimitRule, request: &RateLimitRequest) -> String {
        match rule.limit_type {
            LimitType::PerIP => format!("ip:{}:{}", request.ip_address, rule.id),
            LimitType::PerUser => {
                if let Some(user_id) = request.user_id {
                    format!("user:{}:{}", user_id, rule.id)
                } else {
                    format!("ip:{}:{}", request.ip_address, rule.id)
                }
            },
            LimitType::PerAPIKey => {
                if let Some(api_key) = &request.api_key {
                    format!("apikey:{}:{}", api_key, rule.id)
                } else {
                    format!("ip:{}:{}", request.ip_address, rule.id)
                }
            },
            LimitType::Global => format!("global:{}", rule.id),
            LimitType::PerUserIP => {
                if let Some(user_id) = request.user_id {
                    format!("userip:{}:{}:{}", user_id, request.ip_address, rule.id)
                } else {
                    format!("ip:{}:{}", request.ip_address, rule.id)
                }
            },
            LimitType::PerEndpoint => format!("endpoint:{}:{}:{}", request.method, request.path, rule.id),
        }
    }

    async fn check_rule_limit(&self, rule: &RateLimitRule, key: &str, _request: &RateLimitRequest) -> Result<RateLimitStatus> {
        let mut conn = self.redis.clone();

        // Use Redis for distributed rate limiting
        let current_time = Utc::now().timestamp();
        let window_start = current_time - rule.window_seconds as i64;

        // Clean old entries
        redis::cmd("ZREMRANGEBYSCORE")
            .arg(key)
            .arg(0)
            .arg(window_start)
            .query_async::<()>(&mut conn)
            .await?;

        // Count current requests in window
        let current_count: u32 = redis::cmd("ZCARD")
            .arg(key)
            .query_async(&mut conn)
            .await?;

        let remaining = rule.max_requests.saturating_sub(current_count);
        let reset_time = Utc::now() + Duration::seconds(rule.window_seconds as i64);

        if current_count >= rule.max_requests {
            // Rate limit exceeded
            return Ok(RateLimitStatus {
                key: key.to_string(),
                limit: rule.max_requests,
                remaining: 0,
                reset_time,
                retry_after: Some(rule.window_seconds as u64),
                blocked: true,
                reason: Some(format!("Rate limit exceeded for {}", rule.name)),
            });
        }

        // Record this request
        let request_id = Uuid::new_v4().to_string();
        redis::cmd("ZADD")
            .arg(key)
            .arg(current_time)
            .arg(&request_id)
            .query_async::<()>(&mut conn)
            .await?;
        redis::cmd("EXPIRE")
            .arg(key)
            .arg(rule.window_seconds)
            .query_async::<()>(&mut conn)
            .await?;

        Ok(RateLimitStatus {
            key: key.to_string(),
            limit: rule.max_requests,
            remaining: remaining.saturating_sub(1),
            reset_time,
            retry_after: None,
            blocked: false,
            reason: None,
        })
    }

    async fn handle_rate_limit_violation(&self, rule: &RateLimitRule, key: &str, request: &RateLimitRequest) -> Result<()> {
        // Record violation
        let violation = RateLimitViolation {
            id: Uuid::new_v4(),
            rule_id: rule.id,
            key: key.to_string(),
            endpoint: request.path.clone(),
            method: request.method.clone(),
            ip_address: Some(request.ip_address),
            user_id: request.user_id,
            violation_count: self.get_violation_count(key).await.unwrap_or(0) + 1,
            action_taken: rule.actions.first().cloned().unwrap_or(LimitAction::Block),
            timestamp: Utc::now(),
            user_agent: request.user_agent.clone(),
            request_headers: request.headers.clone(),
        };

        // Store violation
        {
            let mut violations = self.violations.write().await;
            violations.push(violation.clone());

            // Keep only recent violations (last 1000)
            if violations.len() > 1000 {
                violations.remove(0);
            }
        }

        // Execute actions
        for action in &rule.actions {
            self.execute_rate_limit_action(action, key, request, &violation).await?;
        }

        warn!("Rate limit violation: {} for key: {}", rule.name, key);

        Ok(())
    }

    async fn execute_rate_limit_action(&self, action: &LimitAction, key: &str, request: &RateLimitRequest, violation: &RateLimitViolation) -> Result<()> {
        match action {
            LimitAction::Block => {
                // Log the violation with request details
                info!("Rate limit block executed for {} - path: {}, count: {}",
                    key, request.path, violation.violation_count);
            },
            LimitAction::Throttle { delay_ms: _ } => {
                // Delay will be handled by the caller
                info!("Rate limit throttle applied for {} - path: {}",
                    key, request.path);
            },
            LimitAction::RequireCaptcha => {
                // Flag for requiring captcha verification with request context
                let mut conn = self.redis.clone();
                redis::cmd("SETEX")
                    .arg(format!("captcha_required:{}", key))
                    .arg(3600)
                    .arg(serde_json::json!({
                        "path": request.path,
                        "violation_count": violation.violation_count,
                        "timestamp": chrono::Utc::now()
                    }).to_string())
                    .query_async::<()>(&mut conn)
                    .await?;
            },
            LimitAction::RequireAuth => {
                // Flag for requiring authentication with violation details
                let mut conn = self.redis.clone();
                redis::cmd("SETEX")
                    .arg(format!("auth_required:{}", key))
                    .arg(3600)
                    .arg("true")
                    .query_async::<()>(&mut conn)
                    .await?;
            },
            LimitAction::TemporaryBan { duration_minutes } => {
                self.block_entity(key, "temporary_ban", &format!("Rate limit exceeded: {}", violation.rule_id), Some(*duration_minutes)).await?;
            },
            LimitAction::Alert { webhook_url } => {
                self.send_alert(webhook_url, violation).await?;
            },
            LimitAction::Log { level } => {
                match level.as_str() {
                    "error" => tracing::error!("Rate limit violation: {:?}", violation),
                    "warn" => tracing::warn!("Rate limit violation: {:?}", violation),
                    "info" => tracing::info!("Rate limit violation: {:?}", violation),
                    _ => tracing::debug!("Rate limit violation: {:?}", violation),
                }
            },
        }

        Ok(())
    }

    async fn block_entity(&self, key: &str, entity_type: &str, reason: &str, duration_minutes: Option<u32>) -> Result<()> {
        let expires_at = duration_minutes.map(|minutes| Utc::now() + Duration::minutes(minutes as i64));

        let blocked_entity = BlockedEntity {
            key: key.to_string(),
            entity_type: entity_type.to_string(),
            reason: reason.to_string(),
            blocked_at: Utc::now(),
            expires_at,
            block_count: self.get_block_count(key).await.unwrap_or(0) + 1,
            metadata: None,
        };

        // Store in memory
        {
            let mut blocked = self.blocked_entities.write().await;
            blocked.insert(key.to_string(), blocked_entity.clone());
        }

        // Store in Redis
        let mut conn = self.redis.clone();
        let serialized = serde_json::to_string(&blocked_entity)?;

        if let Some(expires) = expires_at {
            let duration_seconds = (expires - Utc::now()).num_seconds().max(1) as u64;
            redis::cmd("SETEX")
                .arg(format!("blocked:{}", key))
                .arg(duration_seconds)
                .arg(serialized)
                .query_async::<()>(&mut conn)
                .await?;
        } else {
            redis::cmd("SET")
                .arg(format!("blocked:{}", key))
                .arg(serialized)
                .query_async::<()>(&mut conn)
                .await?;
        }

        info!("Blocked entity: {} for {} (duration: {:?})", key, reason, duration_minutes);

        Ok(())
    }

    async fn is_blocked(&self, key: &str) -> Option<BlockedEntity> {
        // Check memory cache first
        {
            let blocked = self.blocked_entities.read().await;
            if let Some(entity) = blocked.get(key) {
                if let Some(expires_at) = entity.expires_at {
                    if Utc::now() > expires_at {
                        // Expired, remove from cache
                        drop(blocked);
                        let mut blocked_mut = self.blocked_entities.write().await;
                        blocked_mut.remove(key);
                        return None;
                    }
                }
                return Some(entity.clone());
            }
        }

        // Check Redis
        let mut conn = self.redis.clone();
        if let Ok(Some(serialized)) = conn.get::<_, Option<String>>(format!("blocked:{}", key)).await {
            if let Ok(entity) = serde_json::from_str::<BlockedEntity>(&serialized) {
                if let Some(expires_at) = entity.expires_at {
                    if Utc::now() > expires_at {
                        // Expired, remove from Redis
                        let _: () = conn.del(format!("blocked:{}", key)).await.unwrap_or(());
                        return None;
                    }
                }

                // Cache in memory
                {
                    let mut blocked = self.blocked_entities.write().await;
                    blocked.insert(key.to_string(), entity.clone());
                }

                return Some(entity);
            }
        }

        None
    }

    async fn get_violation_count(&self, key: &str) -> Result<u32> {
        let violations = self.violations.read().await;
        let count = violations.iter()
            .filter(|v| v.key == key && Utc::now() - v.timestamp < Duration::hours(24))
            .count() as u32;
        Ok(count)
    }

    async fn get_block_count(&self, key: &str) -> Result<u32> {
        let mut conn = self.redis.clone();
        let count: u32 = conn.get(format!("block_count:{}", key)).await.unwrap_or(0);
        Ok(count)
    }

    async fn send_alert(&self, webhook_url: &str, violation: &RateLimitViolation) -> Result<()> {
        let client = reqwest::Client::new();
        let payload = serde_json::json!({
            "type": "rate_limit_violation",
            "timestamp": violation.timestamp,
            "rule_id": violation.rule_id,
            "key": violation.key,
            "endpoint": violation.endpoint,
            "method": violation.method,
            "ip_address": violation.ip_address,
            "user_id": violation.user_id,
            "violation_count": violation.violation_count
        });

        match client.post(webhook_url).json(&payload).send().await {
            Ok(_) => debug!("Alert sent successfully for violation: {}", violation.id),
            Err(e) => warn!("Failed to send alert: {}", e),
        }

        Ok(())
    }

    pub async fn unblock_ip(&self, payload: serde_json::Value) -> Result<serde_json::Value> {
        let ip_address = payload.get("ip_address")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("IP address is required"))?;

        let key_pattern = format!("ip:{}", ip_address);

        // Remove from memory cache
        {
            let mut blocked = self.blocked_entities.write().await;
            blocked.retain(|k, _| !k.contains(&key_pattern));
        }

        // Remove from Redis
        let mut conn = self.redis.clone();
        let keys: Vec<String> = conn.keys(format!("blocked:{}:*", key_pattern)).await?;

        for key in keys {
            let _: () = conn.del(&key).await?;
        }

        info!("Unblocked IP address: {}", ip_address);

        Ok(serde_json::json!({
            "message": "IP address unblocked successfully",
            "ip_address": ip_address
        }))
    }

    pub async fn get_status(&self) -> Result<serde_json::Value> {
        let rules_count = self.rules.read().await.len();
        let limiters_count = self.limiters.read().await.len();
        let blocked_count = self.blocked_entities.read().await.len();
        let violations_count = self.violations.read().await.len();

        Ok(serde_json::json!({
            "active_rules": rules_count,
            "active_limiters": limiters_count,
            "blocked_entities": blocked_count,
            "recent_violations": violations_count,
            "status": "healthy"
        }))
    }

    pub async fn get_blocked_ips(&self) -> Result<serde_json::Value> {
        let blocked = self.blocked_entities.read().await;
        let ip_blocks: Vec<&BlockedEntity> = blocked
            .values()
            .filter(|entity| entity.key.starts_with("ip:"))
            .collect();

        let blocked_ips: Vec<serde_json::Value> = ip_blocks
            .into_iter()
            .map(|entity| serde_json::json!({
                "ip_address": entity.key.strip_prefix("ip:").unwrap_or(&entity.key),
                "reason": entity.reason,
                "blocked_at": entity.blocked_at,
                "expires_at": entity.expires_at,
                "block_count": entity.block_count
            }))
            .collect();

        Ok(serde_json::json!({
            "blocked_ips": blocked_ips
        }))
    }

    pub async fn add_rule(&self, rule: RateLimitRule) -> Result<()> {
        // Add to rules
        {
            let mut rules = self.rules.write().await;
            rules.push(rule.clone());
        }

        // Initialize limiter if enabled
        if rule.enabled {
            let max_req = std::num::NonZeroU32::new(rule.max_requests).unwrap_or(nonzero!(1u32));
            let burst = std::num::NonZeroU32::new(rule.burst_size.unwrap_or(rule.max_requests / 10)).unwrap_or(nonzero!(1u32));
            let quota = Quota::per_second(max_req)
                .allow_burst(burst);

            let limiter = Arc::new(RateLimiter::direct(quota));

            let mut limiters = self.limiters.write().await;
            limiters.insert(rule.id.to_string(), limiter);
        }

        info!("Added rate limiting rule: {}", rule.name);
        Ok(())
    }

    pub async fn remove_rule(&self, rule_id: &Uuid) -> Result<()> {
        // Remove from rules
        {
            let mut rules = self.rules.write().await;
            rules.retain(|r| r.id != *rule_id);
        }

        // Remove limiter
        {
            let mut limiters = self.limiters.write().await;
            limiters.remove(&rule_id.to_string());
        }

        info!("Removed rate limiting rule: {}", rule_id);
        Ok(())
    }

    pub async fn get_violations(&self, hours: u32) -> Result<Vec<RateLimitViolation>> {
        let violations = self.violations.read().await;
        let cutoff = Utc::now() - Duration::hours(hours as i64);

        let recent_violations: Vec<RateLimitViolation> = violations
            .iter()
            .filter(|v| v.timestamp > cutoff)
            .cloned()
            .collect();

        Ok(recent_violations)
    }

    pub async fn cleanup_expired(&self) -> Result<()> {
        // Clean up expired blocked entities
        {
            let mut blocked = self.blocked_entities.write().await;
            blocked.retain(|_, entity| {
                if let Some(expires_at) = entity.expires_at {
                    Utc::now() <= expires_at
                } else {
                    true
                }
            });
        }

        // Clean up old violations
        {
            let mut violations = self.violations.write().await;
            let cutoff = Utc::now() - Duration::days(7);
            violations.retain(|v| v.timestamp > cutoff);
        }

        // Clean up Redis keys
        let mut conn = self.redis.clone();
        let keys: Vec<String> = conn.keys("blocked:*").await?;

        for key in keys {
            let exists: bool = conn.exists(&key).await?;
            if !exists {
                continue;
            }

            if let Ok(Some(serialized)) = conn.get::<_, Option<String>>(&key).await {
                if let Ok(entity) = serde_json::from_str::<BlockedEntity>(&serialized) {
                    if let Some(expires_at) = entity.expires_at {
                        if Utc::now() > expires_at {
                            let _: () = conn.del(&key).await?;
                        }
                    }
                }
            }
        }

        debug!("Cleaned up expired rate limiting data");
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct RateLimitRequest {
    pub method: String,
    pub path: String,
    pub resource: String,
    pub ip_address: IpAddr,
    pub user_id: Option<Uuid>,
    pub api_key: Option<String>,
    pub user_agent: Option<String>,
    pub headers: Option<serde_json::Value>,
}

impl RateLimitRequest {
    pub fn new(
        method: String,
        path: String,
        resource: String,
        ip_address: IpAddr,
    ) -> Self {
        Self {
            method,
            path,
            resource,
            ip_address,
            user_id: None,
            api_key: None,
            user_agent: None,
            headers: None,
        }
    }

    pub fn with_user_id(mut self, user_id: Uuid) -> Self {
        self.user_id = Some(user_id);
        self
    }

    pub fn with_api_key(mut self, api_key: String) -> Self {
        self.api_key = Some(api_key);
        self
    }

    pub fn with_user_agent(mut self, user_agent: String) -> Self {
        self.user_agent = Some(user_agent);
        self
    }

    pub fn with_headers(mut self, headers: serde_json::Value) -> Self {
        self.headers = Some(headers);
        self
    }

    pub fn from_http_request(
        method: &str,
        path: &str,
        ip_address: IpAddr,
        user_id: Option<Uuid>,
        api_key: Option<String>,
        user_agent: Option<String>,
        headers: Option<serde_json::Value>,
    ) -> Self {
        // Determine resource from path
        let resource = path.split('/').nth(1).unwrap_or("default").to_string();

        Self {
            method: method.to_string(),
            path: path.to_string(),
            resource,
            ip_address,
            user_id,
            api_key,
            user_agent,
            headers,
        }
    }
}