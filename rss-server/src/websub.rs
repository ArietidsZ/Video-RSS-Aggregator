use anyhow::{anyhow, Result};
use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use hmac::{Hmac, Mac};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

type HmacSha256 = Hmac<Sha256>;

/// WebSub Hub implementation for real-time RSS feed updates
/// Supports both publisher and subscriber roles
pub struct WebSubHub {
    /// HTTP client for making requests
    client: Client,

    /// Active subscriptions
    subscriptions: Arc<RwLock<HashMap<String, Subscription>>>,

    /// Pending subscription verifications
    pending_verifications: Arc<RwLock<HashMap<String, PendingVerification>>>,

    /// Secret key for HMAC signatures
    secret_key: String,

    /// Base URL for callbacks
    callback_base_url: String,

    /// Statistics
    stats: Arc<WebSubStats>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Subscription {
    pub id: String,
    pub hub_url: String,
    pub topic_url: String,
    pub callback_url: String,
    pub secret: String,
    pub lease_seconds: u64,
    pub expires_at: DateTime<Utc>,
    pub state: SubscriptionState,
    pub created_at: DateTime<Utc>,
    pub last_notification: Option<DateTime<Utc>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SubscriptionState {
    Pending,
    Active,
    Expired,
    Failed,
}

#[derive(Clone, Debug)]
struct PendingVerification {
    subscription_id: String,
    challenge: String,
    expires_at: DateTime<Utc>,
}

impl WebSubHub {
    pub async fn new(config: Arc<crate::config::AppConfig>) -> Result<Self> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .user_agent("RSS-Aggregator-WebSub/1.0")
            .build()?;

        Ok(Self {
            client,
            subscriptions: Arc::new(RwLock::new(HashMap::new())),
            pending_verifications: Arc::new(RwLock::new(HashMap::new())),
            secret_key: config.websub_secret.clone().unwrap_or_else(|| {
                "default_websub_secret_key_change_in_production".to_string()
            }),
            callback_base_url: format!("{}/websub/callback", config.base_url),
            stats: Arc::new(WebSubStats::new()),
        })
    }

    /// Subscribe to a topic at a hub
    pub async fn subscribe(
        &self,
        hub_url: &str,
        topic_url: &str,
        callback_url: &str,
    ) -> Result<String> {
        let subscription_id = Uuid::new_v4().to_string();
        let secret = self.generate_secret();

        // Store pending subscription
        let subscription = Subscription {
            id: subscription_id.clone(),
            hub_url: hub_url.to_string(),
            topic_url: topic_url.to_string(),
            callback_url: callback_url.to_string(),
            secret: secret.clone(),
            lease_seconds: 86400, // 24 hours default
            expires_at: Utc::now() + Duration::days(1),
            state: SubscriptionState::Pending,
            created_at: Utc::now(),
            last_notification: None,
        };

        self.subscriptions.write().await.insert(subscription_id.clone(), subscription.clone());

        // Send subscription request to hub
        let params = [
            ("hub.callback", callback_url),
            ("hub.mode", "subscribe"),
            ("hub.topic", topic_url),
            ("hub.lease_seconds", "86400"),
            ("hub.secret", &secret),
        ];

        let response = self.client
            .post(hub_url)
            .form(&params)
            .send()
            .await?;

        if response.status().is_success() {
            self.stats.record_subscription_success();
            Ok(subscription_id)
        } else {
            self.stats.record_subscription_failure();

            // Mark subscription as failed
            if let Some(sub) = self.subscriptions.write().await.get_mut(&subscription_id) {
                sub.state = SubscriptionState::Failed;
            }

            Err(anyhow!("Hub rejected subscription: {}", response.status()))
        }
    }

    /// Unsubscribe from a topic at a hub
    pub async fn unsubscribe(
        &self,
        hub_url: &str,
        topic_url: &str,
        callback_url: &str,
    ) -> Result<()> {
        let params = [
            ("hub.callback", callback_url),
            ("hub.mode", "unsubscribe"),
            ("hub.topic", topic_url),
        ];

        let response = self.client
            .post(hub_url)
            .form(&params)
            .send()
            .await?;

        if response.status().is_success() {
            // Remove subscription
            let mut subs = self.subscriptions.write().await;
            subs.retain(|_, s| {
                !(s.hub_url == hub_url && s.topic_url == topic_url && s.callback_url == callback_url)
            });

            self.stats.record_unsubscription();
            Ok(())
        } else {
            Err(anyhow!("Hub rejected unsubscription: {}", response.status()))
        }
    }

    /// Verify a subscription challenge from the hub
    pub async fn verify_subscription(&self, subscription_id: &str, challenge: &str) -> bool {
        // Check if we have a pending subscription
        if let Some(subscription) = self.subscriptions.read().await.get(subscription_id) {
            if subscription.state == SubscriptionState::Pending {
                // Store verification challenge
                let verification = PendingVerification {
                    subscription_id: subscription_id.to_string(),
                    challenge: challenge.to_string(),
                    expires_at: Utc::now() + Duration::minutes(5),
                };

                self.pending_verifications.write().await.insert(
                    subscription_id.to_string(),
                    verification,
                );

                // Mark subscription as active
                if let Some(sub) = self.subscriptions.write().await.get_mut(subscription_id) {
                    sub.state = SubscriptionState::Active;
                }

                self.stats.record_verification_success();
                return true;
            }
        }

        self.stats.record_verification_failure();
        false
    }

    /// Process an update notification from the hub
    pub async fn process_update(&self, subscription_id: &str, body: String) -> Result<()> {
        // Get subscription
        let subscription = self.subscriptions.read().await
            .get(subscription_id)
            .cloned()
            .ok_or(anyhow!("Subscription not found"))?;

        // Verify HMAC signature if present
        if !subscription.secret.is_empty() {
            // TODO: Verify X-Hub-Signature header
            // This would require passing headers from the handler
        }

        // Update last notification time
        if let Some(sub) = self.subscriptions.write().await.get_mut(subscription_id) {
            sub.last_notification = Some(Utc::now());
        }

        // Process the update (parse RSS/Atom feed)
        self.process_feed_update(&subscription.topic_url, &body).await?;

        self.stats.record_notification_received();
        Ok(())
    }

    /// Publish an update to all subscribers of a topic
    pub async fn publish_update(&self, content: serde_json::Value) -> Result<()> {
        let topic_url = content.get("topic")
            .and_then(|t| t.as_str())
            .ok_or(anyhow!("Missing topic URL"))?;

        let feed_content = content.get("content")
            .and_then(|c| c.as_str())
            .ok_or(anyhow!("Missing feed content"))?;

        // Find all active subscriptions for this topic
        let subscriptions: Vec<Subscription> = self.subscriptions.read().await
            .values()
            .filter(|s| s.topic_url == topic_url && s.state == SubscriptionState::Active)
            .cloned()
            .collect();

        // Send notifications to all subscribers
        let mut tasks = Vec::new();
        for subscription in subscriptions {
            let client = self.client.clone();
            let content = feed_content.to_string();
            let stats = self.stats.clone();

            let task = tokio::spawn(async move {
                let result = Self::send_notification(
                    &client,
                    &subscription.callback_url,
                    &subscription.secret,
                    &content,
                ).await;

                if result.is_ok() {
                    stats.record_notification_sent();
                } else {
                    stats.record_notification_failed();
                }

                result
            });

            tasks.push(task);
        }

        // Wait for all notifications to complete
        for task in tasks {
            let _ = task.await?;
        }

        Ok(())
    }

    /// Act as a hub: accept subscription requests
    pub async fn handle_subscription_request(
        &self,
        mode: &str,
        topic: &str,
        callback: &str,
        lease_seconds: Option<u64>,
        secret: Option<String>,
    ) -> Result<String> {
        match mode {
            "subscribe" => {
                let subscription_id = Uuid::new_v4().to_string();
                let challenge = Uuid::new_v4().to_string();
                let lease = lease_seconds.unwrap_or(86400);

                // Verify the callback URL
                let verify_url = format!(
                    "{}?hub.mode=subscribe&hub.topic={}&hub.challenge={}&hub.lease_seconds={}",
                    callback, urlencoding::encode(topic), challenge, lease
                );

                let response = self.client.get(&verify_url).send().await?;

                if response.status().is_success() {
                    let body = response.text().await?;

                    if body.trim() == challenge {
                        // Store verified subscription
                        let subscription = Subscription {
                            id: subscription_id.clone(),
                            hub_url: self.callback_base_url.clone(),
                            topic_url: topic.to_string(),
                            callback_url: callback.to_string(),
                            secret: secret.unwrap_or_default(),
                            lease_seconds: lease,
                            expires_at: Utc::now() + Duration::seconds(lease as i64),
                            state: SubscriptionState::Active,
                            created_at: Utc::now(),
                            last_notification: None,
                        };

                        self.subscriptions.write().await.insert(
                            subscription_id.clone(),
                            subscription,
                        );

                        return Ok(subscription_id);
                    }
                }

                Err(anyhow!("Callback verification failed"))
            }
            "unsubscribe" => {
                // Remove subscription
                let mut subs = self.subscriptions.write().await;
                subs.retain(|_, s| {
                    !(s.topic_url == topic && s.callback_url == callback)
                });

                Ok("unsubscribed".to_string())
            }
            _ => Err(anyhow!("Invalid hub.mode: {}", mode))
        }
    }

    /// Distribute content to subscribers (hub mode)
    pub async fn distribute_content(&self, topic_url: &str, content: &str) -> Result<()> {
        let subscriptions: Vec<Subscription> = self.subscriptions.read().await
            .values()
            .filter(|s| {
                s.topic_url == topic_url &&
                s.state == SubscriptionState::Active &&
                s.expires_at > Utc::now()
            })
            .cloned()
            .collect();

        let mut tasks = Vec::new();
        for subscription in subscriptions {
            let client = self.client.clone();
            let content = content.to_string();

            let task = tokio::spawn(async move {
                Self::send_notification(
                    &client,
                    &subscription.callback_url,
                    &subscription.secret,
                    &content,
                ).await
            });

            tasks.push(task);
        }

        // Wait for all distributions
        let mut success_count = 0;
        let mut failure_count = 0;

        for task in tasks {
            match task.await? {
                Ok(_) => success_count += 1,
                Err(_) => failure_count += 1,
            }
        }

        self.stats.record_distribution(success_count, failure_count);

        if failure_count > 0 {
            tracing::warn!(
                "Content distribution completed with {} successes and {} failures",
                success_count, failure_count
            );
        }

        Ok(())
    }

    /// Clean up expired subscriptions
    pub async fn cleanup_expired(&self) {
        let now = Utc::now();
        let mut subs = self.subscriptions.write().await;

        let expired_count = subs.len();
        subs.retain(|_, s| s.expires_at > now);
        let removed = expired_count - subs.len();

        if removed > 0 {
            tracing::info!("Cleaned up {} expired subscriptions", removed);
            self.stats.record_cleanup(removed);
        }

        // Clean up old pending verifications
        let mut verifications = self.pending_verifications.write().await;
        verifications.retain(|_, v| v.expires_at > now);
    }

    /// Get subscription statistics
    pub fn get_stats(&self) -> WebSubStatsSnapshot {
        self.stats.snapshot()
    }

    /// Check if the hub is healthy
    pub async fn is_healthy(&self) -> bool {
        // Check if we can access subscriptions
        self.subscriptions.read().await.len() < 100000 // Arbitrary limit
    }

    // Private helper methods

    fn generate_secret(&self) -> String {
        Uuid::new_v4().to_string()
    }

    async fn send_notification(
        client: &Client,
        callback_url: &str,
        secret: &str,
        content: &str,
    ) -> Result<()> {
        let mut request = client.post(callback_url)
            .header("Content-Type", "application/rss+xml")
            .body(content.to_string());

        // Add HMAC signature if secret is present
        if !secret.is_empty() {
            let signature = Self::compute_signature(secret, content)?;
            request = request.header("X-Hub-Signature", format!("sha256={}", signature));
        }

        let response = request.send().await?;

        if response.status().is_success() {
            Ok(())
        } else {
            Err(anyhow!("Notification delivery failed: {}", response.status()))
        }
    }

    fn compute_signature(secret: &str, content: &str) -> Result<String> {
        let mut mac = HmacSha256::new_from_slice(secret.as_bytes())
            .map_err(|e| anyhow!("Invalid HMAC key: {}", e))?;

        mac.update(content.as_bytes());
        let result = mac.finalize();

        Ok(hex::encode(result.into_bytes()))
    }

    async fn process_feed_update(&self, topic_url: &str, content: &str) -> Result<()> {
        // Parse RSS/Atom feed and process updates
        // This would integrate with the feed service to update cached feeds

        tracing::info!("Processing feed update for topic: {}", topic_url);

        // TODO: Integrate with FeedService to update cached content
        // For now, just log the update

        Ok(())
    }
}

/// Statistics collector for WebSub operations
struct WebSubStats {
    subscriptions_success: std::sync::atomic::AtomicU64,
    subscriptions_failed: std::sync::atomic::AtomicU64,
    unsubscriptions: std::sync::atomic::AtomicU64,
    verifications_success: std::sync::atomic::AtomicU64,
    verifications_failed: std::sync::atomic::AtomicU64,
    notifications_sent: std::sync::atomic::AtomicU64,
    notifications_failed: std::sync::atomic::AtomicU64,
    notifications_received: std::sync::atomic::AtomicU64,
    distributions_success: std::sync::atomic::AtomicU64,
    distributions_failed: std::sync::atomic::AtomicU64,
    cleanups_performed: std::sync::atomic::AtomicU64,
}

impl WebSubStats {
    fn new() -> Self {
        use std::sync::atomic::AtomicU64;

        Self {
            subscriptions_success: AtomicU64::new(0),
            subscriptions_failed: AtomicU64::new(0),
            unsubscriptions: AtomicU64::new(0),
            verifications_success: AtomicU64::new(0),
            verifications_failed: AtomicU64::new(0),
            notifications_sent: AtomicU64::new(0),
            notifications_failed: AtomicU64::new(0),
            notifications_received: AtomicU64::new(0),
            distributions_success: AtomicU64::new(0),
            distributions_failed: AtomicU64::new(0),
            cleanups_performed: AtomicU64::new(0),
        }
    }

    fn record_subscription_success(&self) {
        use std::sync::atomic::Ordering;
        self.subscriptions_success.fetch_add(1, Ordering::Relaxed);
    }

    fn record_subscription_failure(&self) {
        use std::sync::atomic::Ordering;
        self.subscriptions_failed.fetch_add(1, Ordering::Relaxed);
    }

    fn record_unsubscription(&self) {
        use std::sync::atomic::Ordering;
        self.unsubscriptions.fetch_add(1, Ordering::Relaxed);
    }

    fn record_verification_success(&self) {
        use std::sync::atomic::Ordering;
        self.verifications_success.fetch_add(1, Ordering::Relaxed);
    }

    fn record_verification_failure(&self) {
        use std::sync::atomic::Ordering;
        self.verifications_failed.fetch_add(1, Ordering::Relaxed);
    }

    fn record_notification_sent(&self) {
        use std::sync::atomic::Ordering;
        self.notifications_sent.fetch_add(1, Ordering::Relaxed);
    }

    fn record_notification_failed(&self) {
        use std::sync::atomic::Ordering;
        self.notifications_failed.fetch_add(1, Ordering::Relaxed);
    }

    fn record_notification_received(&self) {
        use std::sync::atomic::Ordering;
        self.notifications_received.fetch_add(1, Ordering::Relaxed);
    }

    fn record_distribution(&self, success: usize, failed: usize) {
        use std::sync::atomic::Ordering;
        self.distributions_success.fetch_add(success as u64, Ordering::Relaxed);
        self.distributions_failed.fetch_add(failed as u64, Ordering::Relaxed);
    }

    fn record_cleanup(&self, count: usize) {
        use std::sync::atomic::Ordering;
        self.cleanups_performed.fetch_add(count as u64, Ordering::Relaxed);
    }

    fn snapshot(&self) -> WebSubStatsSnapshot {
        use std::sync::atomic::Ordering;

        WebSubStatsSnapshot {
            subscriptions_success: self.subscriptions_success.load(Ordering::Relaxed),
            subscriptions_failed: self.subscriptions_failed.load(Ordering::Relaxed),
            unsubscriptions: self.unsubscriptions.load(Ordering::Relaxed),
            verifications_success: self.verifications_success.load(Ordering::Relaxed),
            verifications_failed: self.verifications_failed.load(Ordering::Relaxed),
            notifications_sent: self.notifications_sent.load(Ordering::Relaxed),
            notifications_failed: self.notifications_failed.load(Ordering::Relaxed),
            notifications_received: self.notifications_received.load(Ordering::Relaxed),
            distributions_success: self.distributions_success.load(Ordering::Relaxed),
            distributions_failed: self.distributions_failed.load(Ordering::Relaxed),
            cleanups_performed: self.cleanups_performed.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct WebSubStatsSnapshot {
    pub subscriptions_success: u64,
    pub subscriptions_failed: u64,
    pub unsubscriptions: u64,
    pub verifications_success: u64,
    pub verifications_failed: u64,
    pub notifications_sent: u64,
    pub notifications_failed: u64,
    pub notifications_received: u64,
    pub distributions_success: u64,
    pub distributions_failed: u64,
    pub cleanups_performed: u64,
}

/// Background task to periodically clean up expired subscriptions
pub async fn cleanup_task(hub: Arc<WebSubHub>) {
    let mut interval = tokio::time::interval(std::time::Duration::from_secs(3600)); // Every hour

    loop {
        interval.tick().await;
        hub.cleanup_expired().await;
    }
}

/// Background task to renew expiring subscriptions
pub async fn renewal_task(hub: Arc<WebSubHub>) {
    let mut interval = tokio::time::interval(std::time::Duration::from_secs(3600)); // Every hour

    loop {
        interval.tick().await;

        let now = Utc::now();
        let renewal_threshold = now + Duration::hours(6); // Renew if expiring in 6 hours

        let subscriptions: Vec<Subscription> = hub.subscriptions.read().await
            .values()
            .filter(|s| {
                s.state == SubscriptionState::Active &&
                s.expires_at < renewal_threshold
            })
            .cloned()
            .collect();

        for subscription in subscriptions {
            // Re-subscribe
            let _ = hub.subscribe(
                &subscription.hub_url,
                &subscription.topic_url,
                &subscription.callback_url,
            ).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_subscription_flow() {
        let config = Arc::new(crate::config::AppConfig {
            base_url: "http://localhost:3000".to_string(),
            websub_secret: Some("test_secret".to_string()),
            ..Default::default()
        });

        let hub = WebSubHub::new(config).await.unwrap();

        // Test subscription
        let sub_id = hub.subscribe(
            "https://hub.example.com",
            "https://example.com/feed.xml",
            "http://localhost:3000/callback",
        ).await;

        assert!(sub_id.is_ok());

        // Test verification
        let verified = hub.verify_subscription(&sub_id.unwrap(), "test_challenge").await;
        assert!(verified);
    }

    #[test]
    fn test_signature_computation() {
        let secret = "test_secret";
        let content = "test content";

        let signature = WebSubHub::compute_signature(secret, content).unwrap();
        assert!(!signature.is_empty());

        // Verify signature is consistent
        let signature2 = WebSubHub::compute_signature(secret, content).unwrap();
        assert_eq!(signature, signature2);
    }
}