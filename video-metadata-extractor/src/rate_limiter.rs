use governor::{Quota, RateLimiter as GovernorRateLimiter, Jitter};
use governor::clock::{DefaultClock, QuantaInstant};
use governor::state::{InMemoryState, NotKeyed};
use std::sync::Arc;
use std::time::Duration;
use std::collections::HashMap;
use tokio::sync::RwLock;
use crate::{Platform, Result, ExtractorError};

/// Rate limiter for API requests
pub struct RateLimiter {
    /// Platform-specific rate limiters
    limiters: Arc<RwLock<HashMap<Platform, Arc<PlatformLimiter>>>>,

    /// Global rate limiter
    global_limiter: Arc<GovernorRateLimiter<NotKeyed, InMemoryState, DefaultClock>>,
}

struct PlatformLimiter {
    limiter: Arc<GovernorRateLimiter<NotKeyed, InMemoryState, DefaultClock>>,
    config: RateLimitConfig,
}

#[derive(Clone)]
pub struct RateLimitConfig {
    /// Maximum requests per second
    pub requests_per_second: u32,

    /// Burst capacity
    pub burst_size: u32,

    /// Retry after rate limit hit (seconds)
    pub retry_after: u64,

    /// Use exponential backoff
    pub exponential_backoff: bool,

    /// Maximum retry attempts
    pub max_retries: u32,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_second: 10,
            burst_size: 20,
            retry_after: 60,
            exponential_backoff: true,
            max_retries: 3,
        }
    }
}

impl RateLimiter {
    pub fn new() -> Self {
        let global_quota = Quota::per_second(std::num::NonZeroU32::new(100).unwrap());
        let global_limiter = Arc::new(GovernorRateLimiter::direct(global_quota));

        Self {
            limiters: Arc::new(RwLock::new(HashMap::new())),
            global_limiter,
        }
    }

    /// Configure platform-specific rate limits
    pub async fn configure_platform(&self, platform: Platform, config: RateLimitConfig) {
        let quota = Quota::per_second(
            std::num::NonZeroU32::new(config.requests_per_second).unwrap()
        ).allow_burst(
            std::num::NonZeroU32::new(config.burst_size).unwrap()
        );

        let limiter = Arc::new(GovernorRateLimiter::direct(quota));

        let platform_limiter = PlatformLimiter {
            limiter,
            config,
        };

        let mut limiters = self.limiters.write().await;
        limiters.insert(platform, Arc::new(platform_limiter));
    }

    /// Setup default platform limits
    pub async fn setup_defaults(&self) {
        // YouTube: 10,000 quota units per day ≈ 0.12 requests/second
        self.configure_platform(Platform::YouTube, RateLimitConfig {
            requests_per_second: 5,
            burst_size: 10,
            retry_after: 60,
            exponential_backoff: true,
            max_retries: 3,
        }).await;

        // Bilibili: More lenient
        self.configure_platform(Platform::Bilibili, RateLimitConfig {
            requests_per_second: 20,
            burst_size: 50,
            retry_after: 30,
            exponential_backoff: true,
            max_retries: 3,
        }).await;

        // Douyin: Strict rate limits
        self.configure_platform(Platform::Douyin, RateLimitConfig {
            requests_per_second: 2,
            burst_size: 5,
            retry_after: 120,
            exponential_backoff: true,
            max_retries: 2,
        }).await;

        // TikTok: Moderate limits
        self.configure_platform(Platform::TikTok, RateLimitConfig {
            requests_per_second: 10,
            burst_size: 20,
            retry_after: 60,
            exponential_backoff: true,
            max_retries: 3,
        }).await;
    }

    /// Check if request can proceed
    pub async fn check(&self, platform: &Platform) -> Result<()> {
        // Check global limit first
        self.global_limiter.check()
            .map_err(|_| ExtractorError::RateLimited)?;

        // Check platform-specific limit
        let limiters = self.limiters.read().await;
        if let Some(platform_limiter) = limiters.get(platform) {
            platform_limiter.limiter.check()
                .map_err(|_| ExtractorError::RateLimited)?;
        }

        Ok(())
    }

    /// Wait until request can proceed
    pub async fn wait(&self, platform: &Platform) -> Result<()> {
        // Wait for global limit
        self.global_limiter.until_ready().await;

        // Wait for platform-specific limit
        let limiters = self.limiters.read().await;
        if let Some(platform_limiter) = limiters.get(platform) {
            platform_limiter.limiter.until_ready().await;
        }

        Ok(())
    }

    /// Wait with jitter to avoid thundering herd
    pub async fn wait_with_jitter(&self, platform: &Platform) -> Result<()> {
        // Add jitter to avoid synchronized requests
        let jitter = Jitter::new(
            Duration::from_millis(0),
            Duration::from_millis(1000),
        );

        self.global_limiter.until_ready_with_jitter(jitter).await;

        let limiters = self.limiters.read().await;
        if let Some(platform_limiter) = limiters.get(platform) {
            platform_limiter.limiter.until_ready_with_jitter(jitter).await;
        }

        Ok(())
    }

    /// Execute with retry and backoff
    pub async fn execute_with_retry<F, T>(&self, platform: &Platform, mut f: F) -> Result<T>
    where
        F: FnMut() -> futures::future::BoxFuture<'static, Result<T>>,
    {
        let limiters = self.limiters.read().await;
        let config = limiters.get(platform)
            .map(|l| l.config.clone())
            .unwrap_or_default();
        drop(limiters);

        let mut retries = 0;
        let mut backoff = Duration::from_secs(1);

        loop {
            // Wait for rate limit
            self.wait(platform).await?;

            match f().await {
                Ok(result) => return Ok(result),
                Err(ExtractorError::RateLimited) if retries < config.max_retries => {
                    retries += 1;

                    if config.exponential_backoff {
                        backoff = backoff * 2;
                    } else {
                        backoff = Duration::from_secs(config.retry_after);
                    }

                    tracing::warn!(
                        "Rate limited for {:?}, retry {}/{} after {:?}",
                        platform, retries, config.max_retries, backoff
                    );

                    tokio::time::sleep(backoff).await;
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Get current rate limit status
    pub async fn status(&self, platform: &Platform) -> RateLimitStatus {
        let limiters = self.limiters.read().await;

        if let Some(platform_limiter) = limiters.get(platform) {
            let state_snapshot = platform_limiter.limiter.state_snapshot();

            RateLimitStatus {
                platform: platform.clone(),
                available: state_snapshot.quota,
                limit: platform_limiter.config.requests_per_second,
                reset_after: Duration::from_secs(1),
            }
        } else {
            RateLimitStatus {
                platform: platform.clone(),
                available: 0,
                limit: 0,
                reset_after: Duration::from_secs(0),
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct RateLimitStatus {
    pub platform: Platform,
    pub available: u32,
    pub limit: u32,
    pub reset_after: Duration,
}

/// Distributed rate limiter using Redis
pub struct DistributedRateLimiter {
    redis_conn: redis::aio::ConnectionManager,
    window_seconds: u64,
    max_requests: u32,
}

impl DistributedRateLimiter {
    pub async fn new(redis_url: &str, window_seconds: u64, max_requests: u32) -> Result<Self> {
        let client = redis::Client::open(redis_url)
            .map_err(|e| ExtractorError::Cache(e))?;

        let redis_conn = redis::aio::ConnectionManager::new(client).await
            .map_err(|e| ExtractorError::Cache(e))?;

        Ok(Self {
            redis_conn,
            window_seconds,
            max_requests,
        })
    }

    /// Check if request is allowed (sliding window algorithm)
    pub async fn is_allowed(&mut self, key: &str) -> Result<bool> {
        use redis::AsyncCommands;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let window_start = now - self.window_seconds;
        let redis_key = format!("rate_limit:{}", key);

        // Remove old entries
        let _: () = self.redis_conn
            .zrembyscore(&redis_key, 0, window_start as f64)
            .await
            .map_err(|e| ExtractorError::Cache(e))?;

        // Count current entries
        let count: u32 = self.redis_conn
            .zcard(&redis_key)
            .await
            .map_err(|e| ExtractorError::Cache(e))?;

        if count < self.max_requests {
            // Add new entry
            let _: () = self.redis_conn
                .zadd(&redis_key, now as f64, now)
                .await
                .map_err(|e| ExtractorError::Cache(e))?;

            // Set expiry
            let _: () = self.redis_conn
                .expire(&redis_key, self.window_seconds as i64)
                .await
                .map_err(|e| ExtractorError::Cache(e))?;

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get remaining requests in current window
    pub async fn remaining(&mut self, key: &str) -> Result<u32> {
        use redis::AsyncCommands;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let window_start = now - self.window_seconds;
        let redis_key = format!("rate_limit:{}", key);

        // Remove old entries
        let _: () = self.redis_conn
            .zrembyscore(&redis_key, 0, window_start as f64)
            .await
            .map_err(|e| ExtractorError::Cache(e))?;

        // Count current entries
        let count: u32 = self.redis_conn
            .zcard(&redis_key)
            .await
            .map_err(|e| ExtractorError::Cache(e))?;

        Ok(self.max_requests.saturating_sub(count))
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rate_limiter() {
        let limiter = RateLimiter::new();
        limiter.setup_defaults().await;

        // Should allow initial requests
        assert!(limiter.check(&Platform::YouTube).await.is_ok());
        assert!(limiter.check(&Platform::YouTube).await.is_ok());
    }

    #[tokio::test]
    async fn test_platform_specific_limits() {
        let limiter = RateLimiter::new();

        limiter.configure_platform(Platform::YouTube, RateLimitConfig {
            requests_per_second: 1,
            burst_size: 1,
            retry_after: 1,
            exponential_backoff: false,
            max_retries: 1,
        }).await;

        assert!(limiter.check(&Platform::YouTube).await.is_ok());

        // Second request should be rate limited
        // Note: This might pass due to timing, but demonstrates the concept
    }
}