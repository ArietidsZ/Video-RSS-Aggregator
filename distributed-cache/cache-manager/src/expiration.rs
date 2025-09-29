use anyhow::Result;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::time::interval;
use tracing::{debug, info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpirationPolicy {
    pub ttl_seconds: u64,
    pub max_age_seconds: u64,
    pub idle_time_seconds: u64,
    pub scan_interval_seconds: u64,
    pub batch_size: usize,
}

impl Default for ExpirationPolicy {
    fn default() -> Self {
        Self {
            ttl_seconds: 3600,        // 1 hour default TTL
            max_age_seconds: 86400,   // 24 hours max age
            idle_time_seconds: 7200,  // 2 hours idle time
            scan_interval_seconds: 300, // 5 minutes scan interval
            batch_size: 1000,
        }
    }
}

pub struct ExpirationManager {
    policies: HashMap<String, ExpirationPolicy>,
}

impl ExpirationManager {
    pub fn new() -> Self {
        let mut policies = HashMap::new();

        // Define different policies for different data types
        policies.insert("video:*".to_string(), ExpirationPolicy {
            ttl_seconds: 7200,      // 2 hours for video metadata
            max_age_seconds: 86400, // 24 hours max
            idle_time_seconds: 3600,
            scan_interval_seconds: 600,
            batch_size: 500,
        });

        policies.insert("transcript:*".to_string(), ExpirationPolicy {
            ttl_seconds: 3600,      // 1 hour for transcripts
            max_age_seconds: 43200, // 12 hours max
            idle_time_seconds: 1800,
            scan_interval_seconds: 300,
            batch_size: 1000,
        });

        policies.insert("summary:*".to_string(), ExpirationPolicy {
            ttl_seconds: 14400,     // 4 hours for summaries
            max_age_seconds: 172800, // 48 hours max
            idle_time_seconds: 7200,
            scan_interval_seconds: 900,
            batch_size: 200,
        });

        policies.insert("default".to_string(), ExpirationPolicy::default());

        Self { policies }
    }

    pub fn get_policy(&self, key: &str) -> &ExpirationPolicy {
        // Find matching policy based on pattern
        for (pattern, policy) in &self.policies {
            if Self::matches_pattern(key, pattern) {
                return policy;
            }
        }

        // Return default policy
        self.policies.get("default").unwrap()
    }

    fn matches_pattern(key: &str, pattern: &str) -> bool {
        if pattern == "default" {
            return true;
        }

        if pattern.ends_with("*") {
            let prefix = &pattern[..pattern.len() - 1];
            return key.starts_with(prefix);
        }

        key == pattern
    }

    pub fn calculate_ttl(&self, key: &str, access_count: usize) -> u64 {
        let base_policy = self.get_policy(key);
        let mut ttl = base_policy.ttl_seconds;

        // Adaptive TTL based on access frequency
        if access_count > 100 {
            ttl = (ttl as f64 * 2.0) as u64; // Double TTL for hot data
        } else if access_count > 50 {
            ttl = (ttl as f64 * 1.5) as u64;
        } else if access_count < 5 {
            ttl = (ttl as f64 * 0.5) as u64; // Halve TTL for cold data
        }

        // Cap at max age
        ttl.min(base_policy.max_age_seconds)
    }

    pub async fn start_expiration_scanner(&self) {
        info!("Starting expiration scanner");

        let mut ticker = interval(tokio::time::Duration::from_secs(300)); // 5 minutes

        loop {
            ticker.tick().await;

            if let Err(e) = self.scan_and_expire().await {
                warn!("Expiration scan failed: {}", e);
            }
        }
    }

    async fn scan_and_expire(&self) -> Result<()> {
        debug!("Running expiration scan");

        let expired_count = 0;
        // In production, this would scan Redis and Cassandra for expired items

        if expired_count > 0 {
            info!("Expired {} items", expired_count);
        }

        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct ExpirationStats {
    pub total_expired: u64,
    pub expired_by_ttl: u64,
    pub expired_by_idle: u64,
    pub expired_by_max_age: u64,
    pub last_scan_time: DateTime<Utc>,
    pub next_scan_time: DateTime<Utc>,
}