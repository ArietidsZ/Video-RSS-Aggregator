use anyhow::{Context, Result};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{delete, get, post},
    Router,
};
use chrono::{DateTime, Duration, Utc};
use hmac::{Hmac, Mac};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::Sha256;
use sqlx::PgPool;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tokio::time::sleep;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

// Webhook types and configuration

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Webhook {
    pub id: Uuid,
    pub user_id: String,
    pub name: String,
    pub url: String,
    pub events: Vec<EventType>,
    pub headers: HashMap<String, String>,
    pub secret: Option<String>,
    pub active: bool,
    pub retry_config: RetryConfig,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub last_triggered: Option<DateTime<Utc>>,
    pub failure_count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventType {
    // Video events
    VideoCreated,
    VideoUpdated,
    VideoDeleted,
    VideoProcessed,
    VideoQualityChanged,

    // Channel events
    ChannelCreated,
    ChannelUpdated,
    ChannelDeleted,

    // User events
    UserRegistered,
    UserUpdated,
    UserDeleted,
    UserLoginFailed,

    // Summary events
    SummaryGenerated,
    SummaryFailed,

    // Feed events
    FeedAdded,
    FeedUpdated,
    FeedFailed,
    FeedProcessed,

    // Recommendation events
    RecommendationGenerated,

    // System events
    SystemAlert,
    SystemMaintenance,
    RateLimitExceeded,

    // Custom events
    Custom(String),
}

impl EventType {
    pub fn from_str(s: &str) -> Self {
        match s {
            "video.created" => EventType::VideoCreated,
            "video.updated" => EventType::VideoUpdated,
            "video.deleted" => EventType::VideoDeleted,
            "video.processed" => EventType::VideoProcessed,
            "video.quality_changed" => EventType::VideoQualityChanged,
            "channel.created" => EventType::ChannelCreated,
            "channel.updated" => EventType::ChannelUpdated,
            "channel.deleted" => EventType::ChannelDeleted,
            "user.registered" => EventType::UserRegistered,
            "user.updated" => EventType::UserUpdated,
            "user.deleted" => EventType::UserDeleted,
            "user.login_failed" => EventType::UserLoginFailed,
            "summary.generated" => EventType::SummaryGenerated,
            "summary.failed" => EventType::SummaryFailed,
            "feed.added" => EventType::FeedAdded,
            "feed.updated" => EventType::FeedUpdated,
            "feed.failed" => EventType::FeedFailed,
            "feed.processed" => EventType::FeedProcessed,
            "recommendation.generated" => EventType::RecommendationGenerated,
            "system.alert" => EventType::SystemAlert,
            "system.maintenance" => EventType::SystemMaintenance,
            "rate_limit.exceeded" => EventType::RateLimitExceeded,
            other => EventType::Custom(other.to_string()),
        }
    }

    pub fn to_string(&self) -> String {
        match self {
            EventType::VideoCreated => "video.created".to_string(),
            EventType::VideoUpdated => "video.updated".to_string(),
            EventType::VideoDeleted => "video.deleted".to_string(),
            EventType::VideoProcessed => "video.processed".to_string(),
            EventType::VideoQualityChanged => "video.quality_changed".to_string(),
            EventType::ChannelCreated => "channel.created".to_string(),
            EventType::ChannelUpdated => "channel.updated".to_string(),
            EventType::ChannelDeleted => "channel.deleted".to_string(),
            EventType::UserRegistered => "user.registered".to_string(),
            EventType::UserUpdated => "user.updated".to_string(),
            EventType::UserDeleted => "user.deleted".to_string(),
            EventType::UserLoginFailed => "user.login_failed".to_string(),
            EventType::SummaryGenerated => "summary.generated".to_string(),
            EventType::SummaryFailed => "summary.failed".to_string(),
            EventType::FeedAdded => "feed.added".to_string(),
            EventType::FeedUpdated => "feed.updated".to_string(),
            EventType::FeedFailed => "feed.failed".to_string(),
            EventType::FeedProcessed => "feed.processed".to_string(),
            EventType::RecommendationGenerated => "recommendation.generated".to_string(),
            EventType::SystemAlert => "system.alert".to_string(),
            EventType::SystemMaintenance => "system.maintenance".to_string(),
            EventType::RateLimitExceeded => "rate_limit.exceeded".to_string(),
            EventType::Custom(s) => s.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    pub max_attempts: u32,
    pub initial_delay_ms: u64,
    pub max_delay_ms: u64,
    pub backoff_multiplier: f32,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay_ms: 1000,
            max_delay_ms: 60000,
            backoff_multiplier: 2.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookEvent {
    pub id: Uuid,
    pub webhook_id: Uuid,
    pub event_type: EventType,
    pub payload: Value,
    pub timestamp: DateTime<Utc>,
    pub signature: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookDelivery {
    pub id: Uuid,
    pub webhook_id: Uuid,
    pub event_id: Uuid,
    pub url: String,
    pub request_headers: HashMap<String, String>,
    pub request_body: Value,
    pub response_status: Option<u16>,
    pub response_headers: Option<HashMap<String, String>>,
    pub response_body: Option<String>,
    pub delivered_at: DateTime<Utc>,
    pub duration_ms: u64,
    pub attempt_number: u32,
    pub success: bool,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookStatistics {
    pub webhook_id: Uuid,
    pub total_deliveries: u64,
    pub successful_deliveries: u64,
    pub failed_deliveries: u64,
    pub average_response_time_ms: f64,
    pub events_by_type: HashMap<String, u64>,
    pub last_delivery: Option<DateTime<Utc>>,
}

// Webhook Manager

#[derive(Debug, Clone)]
pub struct WebhookManager {
    pool: PgPool,
    client: Client,
    webhooks: Arc<RwLock<HashMap<Uuid, Webhook>>>,
    event_queue: Arc<Mutex<VecDeque<WebhookEvent>>>,
    delivery_history: Arc<RwLock<Vec<WebhookDelivery>>>,
    statistics: Arc<RwLock<HashMap<Uuid, WebhookStatistics>>>,
    config: WebhookConfig,
}

#[derive(Debug, Clone)]
pub struct WebhookConfig {
    pub max_queue_size: usize,
    pub delivery_timeout_ms: u64,
    pub max_concurrent_deliveries: usize,
    pub signature_algorithm: SignatureAlgorithm,
    pub enable_retry: bool,
    pub enable_signature: bool,
    pub enable_delivery_history: bool,
    pub history_retention_days: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignatureAlgorithm {
    HmacSha256,
    HmacSha512,
}

impl Default for WebhookConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 10000,
            delivery_timeout_ms: 30000,
            max_concurrent_deliveries: 50,
            signature_algorithm: SignatureAlgorithm::HmacSha256,
            enable_retry: true,
            enable_signature: true,
            enable_delivery_history: true,
            history_retention_days: 30,
        }
    }
}

impl WebhookManager {
    pub async fn new(pool: PgPool, config: WebhookConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_millis(config.delivery_timeout_ms))
            .build()?;

        let manager = Self {
            pool,
            client,
            webhooks: Arc::new(RwLock::new(HashMap::new())),
            event_queue: Arc::new(Mutex::new(VecDeque::new())),
            delivery_history: Arc::new(RwLock::new(Vec::new())),
            statistics: Arc::new(RwLock::new(HashMap::new())),
            config,
        };

        // Load webhooks from database
        manager.load_webhooks().await?;

        // Start background workers
        manager.start_delivery_worker();
        manager.start_cleanup_worker();

        Ok(manager)
    }

    async fn load_webhooks(&self) -> Result<()> {
        let webhooks = sqlx::query_as!(
            Webhook,
            r#"
            SELECT id, user_id, name, url, events, headers, secret,
                   active, retry_config, created_at, updated_at,
                   last_triggered, failure_count
            FROM webhooks
            WHERE active = true
            "#
        )
        .fetch_all(&self.pool)
        .await?;

        let mut webhook_map = self.webhooks.write().await;
        for webhook in webhooks {
            webhook_map.insert(webhook.id, webhook);
        }

        info!("Loaded {} active webhooks", webhook_map.len());
        Ok(())
    }

    pub async fn create_webhook(&self, webhook: Webhook) -> Result<Webhook> {
        let webhook = sqlx::query_as!(
            Webhook,
            r#"
            INSERT INTO webhooks (id, user_id, name, url, events, headers,
                                secret, active, retry_config, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            RETURNING *
            "#,
            webhook.id,
            webhook.user_id,
            webhook.name,
            webhook.url,
            serde_json::to_value(&webhook.events)?,
            serde_json::to_value(&webhook.headers)?,
            webhook.secret,
            webhook.active,
            serde_json::to_value(&webhook.retry_config)?,
            webhook.created_at,
            webhook.updated_at,
        )
        .fetch_one(&self.pool)
        .await?;

        self.webhooks.write().await.insert(webhook.id, webhook.clone());

        info!("Created webhook: {} ({})", webhook.name, webhook.id);
        Ok(webhook)
    }

    pub async fn update_webhook(&self, id: Uuid, updates: WebhookUpdate) -> Result<Webhook> {
        let webhook = sqlx::query_as!(
            Webhook,
            r#"
            UPDATE webhooks
            SET name = COALESCE($2, name),
                url = COALESCE($3, url),
                events = COALESCE($4, events),
                headers = COALESCE($5, headers),
                secret = COALESCE($6, secret),
                active = COALESCE($7, active),
                updated_at = NOW()
            WHERE id = $1
            RETURNING *
            "#,
            id,
            updates.name,
            updates.url,
            updates.events.map(|e| serde_json::to_value(&e).unwrap()),
            updates.headers.map(|h| serde_json::to_value(&h).unwrap()),
            updates.secret,
            updates.active,
        )
        .fetch_one(&self.pool)
        .await?;

        self.webhooks.write().await.insert(webhook.id, webhook.clone());

        info!("Updated webhook: {}", id);
        Ok(webhook)
    }

    pub async fn delete_webhook(&self, id: Uuid) -> Result<()> {
        sqlx::query!(
            "UPDATE webhooks SET active = false WHERE id = $1",
            id
        )
        .execute(&self.pool)
        .await?;

        self.webhooks.write().await.remove(&id);

        info!("Deleted webhook: {}", id);
        Ok(())
    }

    pub async fn trigger_event(&self, event_type: EventType, payload: Value) -> Result<()> {
        let event = WebhookEvent {
            id: Uuid::new_v4(),
            webhook_id: Uuid::nil(), // Will be set per webhook
            event_type: event_type.clone(),
            payload: payload.clone(),
            timestamp: Utc::now(),
            signature: None,
        };

        // Find matching webhooks
        let webhooks = self.webhooks.read().await;
        let matching_webhooks: Vec<_> = webhooks
            .values()
            .filter(|w| w.active && w.events.contains(&event_type))
            .cloned()
            .collect();

        drop(webhooks); // Release read lock

        // Queue events for each matching webhook
        for webhook in matching_webhooks {
            let mut webhook_event = event.clone();
            webhook_event.webhook_id = webhook.id;

            // Generate signature if needed
            if self.config.enable_signature && webhook.secret.is_some() {
                webhook_event.signature = Some(self.generate_signature(
                    &webhook.secret.as_ref().unwrap(),
                    &webhook_event.payload,
                )?);
            }

            // Add to queue
            let mut queue = self.event_queue.lock().await;
            if queue.len() < self.config.max_queue_size {
                queue.push_back(webhook_event);
                debug!("Queued event {} for webhook {}", event_type.to_string(), webhook.id);
            } else {
                warn!("Event queue full, dropping event for webhook {}", webhook.id);
            }
        }

        Ok(())
    }

    fn generate_signature(&self, secret: &str, payload: &Value) -> Result<String> {
        let payload_str = serde_json::to_string(payload)?;

        match self.config.signature_algorithm {
            SignatureAlgorithm::HmacSha256 => {
                type HmacSha256 = Hmac<Sha256>;
                let mut mac = HmacSha256::new_from_slice(secret.as_bytes())?;
                mac.update(payload_str.as_bytes());
                let result = mac.finalize();
                Ok(hex::encode(result.into_bytes()))
            }
            SignatureAlgorithm::HmacSha512 => {
                type HmacSha512 = Hmac<sha2::Sha512>;
                let mut mac = HmacSha512::new_from_slice(secret.as_bytes())?;
                mac.update(payload_str.as_bytes());
                let result = mac.finalize();
                Ok(hex::encode(result.into_bytes()))
            }
        }
    }

    fn start_delivery_worker(&self) {
        let manager = self.clone();

        tokio::spawn(async move {
            loop {
                if let Some(event) = manager.get_next_event().await {
                    if let Some(webhook) = manager.webhooks.read().await.get(&event.webhook_id) {
                        let webhook = webhook.clone();
                        let manager_clone = manager.clone();

                        tokio::spawn(async move {
                            if let Err(e) = manager_clone.deliver_event(&webhook, &event).await {
                                error!("Failed to deliver webhook: {}", e);
                            }
                        });
                    }
                }

                sleep(std::time::Duration::from_millis(100)).await;
            }
        });
    }

    async fn get_next_event(&self) -> Option<WebhookEvent> {
        self.event_queue.lock().await.pop_front()
    }

    async fn deliver_event(&self, webhook: &Webhook, event: &WebhookEvent) -> Result<()> {
        let mut attempt = 0;
        let mut last_error = None;

        while attempt < webhook.retry_config.max_attempts {
            attempt += 1;

            let start = std::time::Instant::now();
            let delivery_id = Uuid::new_v4();

            // Prepare request
            let mut headers = webhook.headers.clone();
            headers.insert("Content-Type".to_string(), "application/json".to_string());
            headers.insert("X-Webhook-Event".to_string(), event.event_type.to_string());
            headers.insert("X-Webhook-ID".to_string(), webhook.id.to_string());
            headers.insert("X-Webhook-Delivery".to_string(), delivery_id.to_string());
            headers.insert("X-Webhook-Timestamp".to_string(), event.timestamp.to_rfc3339());

            if let Some(signature) = &event.signature {
                headers.insert("X-Webhook-Signature".to_string(), signature.clone());
            }

            // Build request
            let mut request = self.client.post(&webhook.url);
            for (key, value) in &headers {
                request = request.header(key, value);
            }
            request = request.json(&event.payload);

            // Send request
            match request.send().await {
                Ok(response) => {
                    let status = response.status().as_u16();
                    let duration_ms = start.elapsed().as_millis() as u64;

                    let delivery = WebhookDelivery {
                        id: delivery_id,
                        webhook_id: webhook.id,
                        event_id: event.id,
                        url: webhook.url.clone(),
                        request_headers: headers.clone(),
                        request_body: event.payload.clone(),
                        response_status: Some(status),
                        response_headers: None, // Could extract if needed
                        response_body: response.text().await.ok(),
                        delivered_at: Utc::now(),
                        duration_ms,
                        attempt_number: attempt,
                        success: status >= 200 && status < 300,
                        error_message: None,
                    };

                    if self.config.enable_delivery_history {
                        self.record_delivery(delivery.clone()).await;
                    }

                    self.update_statistics(webhook.id, &event.event_type, delivery.success, duration_ms).await;

                    if delivery.success {
                        info!("Webhook delivered successfully: {} -> {}", webhook.id, webhook.url);
                        return Ok(());
                    } else {
                        last_error = Some(format!("HTTP {}", status));
                    }
                }
                Err(e) => {
                    last_error = Some(e.to_string());

                    let delivery = WebhookDelivery {
                        id: delivery_id,
                        webhook_id: webhook.id,
                        event_id: event.id,
                        url: webhook.url.clone(),
                        request_headers: headers.clone(),
                        request_body: event.payload.clone(),
                        response_status: None,
                        response_headers: None,
                        response_body: None,
                        delivered_at: Utc::now(),
                        duration_ms: start.elapsed().as_millis() as u64,
                        attempt_number: attempt,
                        success: false,
                        error_message: Some(e.to_string()),
                    };

                    if self.config.enable_delivery_history {
                        self.record_delivery(delivery).await;
                    }

                    self.update_statistics(webhook.id, &event.event_type, false, 0).await;
                }
            }

            // Calculate retry delay
            if self.config.enable_retry && attempt < webhook.retry_config.max_attempts {
                let delay = self.calculate_retry_delay(&webhook.retry_config, attempt);
                sleep(std::time::Duration::from_millis(delay)).await;
            }
        }

        // Update failure count
        self.increment_failure_count(webhook.id).await?;

        Err(anyhow::anyhow!(
            "Failed to deliver webhook after {} attempts: {:?}",
            attempt,
            last_error
        ))
    }

    fn calculate_retry_delay(&self, config: &RetryConfig, attempt: u32) -> u64 {
        let delay = config.initial_delay_ms as f32 * config.backoff_multiplier.powi(attempt as i32 - 1);
        delay.min(config.max_delay_ms as f32) as u64
    }

    async fn record_delivery(&self, delivery: WebhookDelivery) {
        self.delivery_history.write().await.push(delivery.clone());

        // Store in database
        let _ = sqlx::query!(
            r#"
            INSERT INTO webhook_deliveries
            (id, webhook_id, event_id, url, request_headers, request_body,
             response_status, response_body, delivered_at, duration_ms,
             attempt_number, success, error_message)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            "#,
            delivery.id,
            delivery.webhook_id,
            delivery.event_id,
            delivery.url,
            serde_json::to_value(&delivery.request_headers).unwrap(),
            delivery.request_body,
            delivery.response_status.map(|s| s as i32),
            delivery.response_body,
            delivery.delivered_at,
            delivery.duration_ms as i64,
            delivery.attempt_number as i32,
            delivery.success,
            delivery.error_message,
        )
        .execute(&self.pool)
        .await;
    }

    async fn update_statistics(
        &self,
        webhook_id: Uuid,
        event_type: &EventType,
        success: bool,
        duration_ms: u64,
    ) {
        let mut statistics = self.statistics.write().await;

        let stats = statistics.entry(webhook_id).or_insert_with(|| {
            WebhookStatistics {
                webhook_id,
                total_deliveries: 0,
                successful_deliveries: 0,
                failed_deliveries: 0,
                average_response_time_ms: 0.0,
                events_by_type: HashMap::new(),
                last_delivery: None,
            }
        });

        stats.total_deliveries += 1;
        if success {
            stats.successful_deliveries += 1;
        } else {
            stats.failed_deliveries += 1;
        }

        // Update average response time
        stats.average_response_time_ms =
            (stats.average_response_time_ms * (stats.total_deliveries - 1) as f64
             + duration_ms as f64) / stats.total_deliveries as f64;

        // Update events by type
        *stats.events_by_type.entry(event_type.to_string()).or_insert(0) += 1;

        stats.last_delivery = Some(Utc::now());
    }

    async fn increment_failure_count(&self, webhook_id: Uuid) -> Result<()> {
        sqlx::query!(
            r#"
            UPDATE webhooks
            SET failure_count = failure_count + 1,
                active = CASE WHEN failure_count >= 10 THEN false ELSE active END
            WHERE id = $1
            "#,
            webhook_id
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    fn start_cleanup_worker(&self) {
        let manager = self.clone();
        let retention_days = self.config.history_retention_days;

        tokio::spawn(async move {
            loop {
                // Clean up old delivery history every hour
                sleep(std::time::Duration::from_secs(3600)).await;

                let cutoff = Utc::now() - Duration::days(retention_days as i64);

                let _ = sqlx::query!(
                    "DELETE FROM webhook_deliveries WHERE delivered_at < $1",
                    cutoff
                )
                .execute(&manager.pool)
                .await;

                // Clean in-memory history
                let mut history = manager.delivery_history.write().await;
                history.retain(|d| d.delivered_at > cutoff);

                info!("Cleaned up webhook delivery history older than {} days", retention_days);
            }
        });
    }

    pub async fn get_webhook(&self, id: Uuid) -> Option<Webhook> {
        self.webhooks.read().await.get(&id).cloned()
    }

    pub async fn list_webhooks(&self, user_id: Option<String>) -> Vec<Webhook> {
        let webhooks = self.webhooks.read().await;

        if let Some(user_id) = user_id {
            webhooks
                .values()
                .filter(|w| w.user_id == user_id)
                .cloned()
                .collect()
        } else {
            webhooks.values().cloned().collect()
        }
    }

    pub async fn get_delivery_history(&self, webhook_id: Uuid, limit: usize) -> Vec<WebhookDelivery> {
        let history = self.delivery_history.read().await;

        history
            .iter()
            .filter(|d| d.webhook_id == webhook_id)
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }

    pub async fn get_statistics(&self, webhook_id: Uuid) -> Option<WebhookStatistics> {
        self.statistics.read().await.get(&webhook_id).cloned()
    }

    pub async fn test_webhook(&self, id: Uuid) -> Result<()> {
        let test_payload = serde_json::json!({
            "test": true,
            "timestamp": Utc::now(),
            "message": "This is a test webhook delivery"
        });

        self.trigger_event(EventType::Custom("test".to_string()), test_payload).await?;

        Ok(())
    }
}

// API handlers

#[derive(Debug, Deserialize)]
pub struct CreateWebhookRequest {
    pub name: String,
    pub url: String,
    pub events: Vec<String>,
    pub headers: Option<HashMap<String, String>>,
    pub secret: Option<String>,
    pub retry_config: Option<RetryConfig>,
}

#[derive(Debug, Deserialize)]
pub struct WebhookUpdate {
    pub name: Option<String>,
    pub url: Option<String>,
    pub events: Option<Vec<EventType>>,
    pub headers: Option<HashMap<String, String>>,
    pub secret: Option<String>,
    pub active: Option<bool>,
}

pub async fn create_webhook_handler(
    State(manager): State<Arc<WebhookManager>>,
    Json(request): Json<CreateWebhookRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    let webhook = Webhook {
        id: Uuid::new_v4(),
        user_id: "current_user".to_string(), // Get from auth context
        name: request.name,
        url: request.url,
        events: request.events.iter().map(|e| EventType::from_str(e)).collect(),
        headers: request.headers.unwrap_or_default(),
        secret: request.secret,
        active: true,
        retry_config: request.retry_config.unwrap_or_default(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        last_triggered: None,
        failure_count: 0,
    };

    match manager.create_webhook(webhook).await {
        Ok(webhook) => Ok(Json(webhook)),
        Err(e) => {
            error!("Failed to create webhook: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

pub async fn list_webhooks_handler(
    State(manager): State<Arc<WebhookManager>>,
) -> impl IntoResponse {
    let webhooks = manager.list_webhooks(Some("current_user".to_string())).await;
    Json(webhooks)
}

pub async fn get_webhook_handler(
    State(manager): State<Arc<WebhookManager>>,
    Path(id): Path<Uuid>,
) -> Result<impl IntoResponse, StatusCode> {
    match manager.get_webhook(id).await {
        Some(webhook) => Ok(Json(webhook)),
        None => Err(StatusCode::NOT_FOUND),
    }
}

pub async fn update_webhook_handler(
    State(manager): State<Arc<WebhookManager>>,
    Path(id): Path<Uuid>,
    Json(updates): Json<WebhookUpdate>,
) -> Result<impl IntoResponse, StatusCode> {
    match manager.update_webhook(id, updates).await {
        Ok(webhook) => Ok(Json(webhook)),
        Err(e) => {
            error!("Failed to update webhook: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

pub async fn delete_webhook_handler(
    State(manager): State<Arc<WebhookManager>>,
    Path(id): Path<Uuid>,
) -> Result<impl IntoResponse, StatusCode> {
    match manager.delete_webhook(id).await {
        Ok(_) => Ok(StatusCode::NO_CONTENT),
        Err(e) => {
            error!("Failed to delete webhook: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

pub async fn test_webhook_handler(
    State(manager): State<Arc<WebhookManager>>,
    Path(id): Path<Uuid>,
) -> Result<impl IntoResponse, StatusCode> {
    match manager.test_webhook(id).await {
        Ok(_) => Ok(Json(serde_json::json!({"message": "Test webhook sent"}))),
        Err(e) => {
            error!("Failed to test webhook: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

pub async fn get_delivery_history_handler(
    State(manager): State<Arc<WebhookManager>>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    let history = manager.get_delivery_history(id, 100).await;
    Json(history)
}

pub async fn get_statistics_handler(
    State(manager): State<Arc<WebhookManager>>,
    Path(id): Path<Uuid>,
) -> Result<impl IntoResponse, StatusCode> {
    match manager.get_statistics(id).await {
        Some(stats) => Ok(Json(stats)),
        None => Err(StatusCode::NOT_FOUND),
    }
}

// Router creation

pub fn create_webhook_router(pool: PgPool) -> Router {
    let manager = Arc::new(
        WebhookManager::new(pool, WebhookConfig::default())
            .await
            .expect("Failed to create webhook manager")
    );

    Router::new()
        .route("/webhooks", post(create_webhook_handler).get(list_webhooks_handler))
        .route("/webhooks/:id", get(get_webhook_handler)
            .patch(update_webhook_handler)
            .delete(delete_webhook_handler))
        .route("/webhooks/:id/test", post(test_webhook_handler))
        .route("/webhooks/:id/deliveries", get(get_delivery_history_handler))
        .route("/webhooks/:id/statistics", get(get_statistics_handler))
        .with_state(manager)
}