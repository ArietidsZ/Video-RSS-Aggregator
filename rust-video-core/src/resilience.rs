use crate::{error::VideoRssError, Result};
use async_trait::async_trait;
use backoff::{future::retry, ExponentialBackoff};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: usize,
    pub recovery_timeout: Duration,
    pub success_threshold: usize,
    pub timeout: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            recovery_timeout: Duration::from_secs(60),
            success_threshold: 3,
            timeout: Duration::from_secs(30),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CircuitState {
    Closed,   // Normal operation
    Open,     // Failing fast
    HalfOpen, // Testing recovery
}

#[derive(Debug)]
pub struct CircuitBreakerStats {
    pub state: CircuitState,
    pub failure_count: usize,
    pub success_count: usize,
    pub last_failure_time: Option<Instant>,
    pub total_requests: u64,
    pub total_failures: u64,
    pub total_successes: u64,
}

pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    state: Arc<RwLock<CircuitState>>,
    failure_count: Arc<AtomicUsize>,
    success_count: Arc<AtomicUsize>,
    last_failure_time: Arc<RwLock<Option<Instant>>>,
    total_requests: Arc<AtomicU64>,
    total_failures: Arc<AtomicU64>,
    total_successes: Arc<AtomicU64>,
    name: String,
}

impl CircuitBreaker {
    pub fn new(name: String, config: CircuitBreakerConfig) -> Self {
        info!("Creating circuit breaker '{}' with config: {:?}", name, config);

        Self {
            config,
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            failure_count: Arc::new(AtomicUsize::new(0)),
            success_count: Arc::new(AtomicUsize::new(0)),
            last_failure_time: Arc::new(RwLock::new(None)),
            total_requests: Arc::new(AtomicU64::new(0)),
            total_failures: Arc::new(AtomicU64::new(0)),
            total_successes: Arc::new(AtomicU64::new(0)),
            name,
        }
    }

    pub async fn call<F, Fut, T>(&self, operation: F) -> Result<T>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        self.total_requests.fetch_add(1, Ordering::Relaxed);

        // Check if circuit is open
        if self.is_open().await {
            self.total_failures.fetch_add(1, Ordering::Relaxed);
            return Err(VideoRssError::Config(format!(
                "Circuit breaker '{}' is OPEN",
                self.name
            )));
        }

        // Execute operation with timeout
        let operation_result = tokio::time::timeout(self.config.timeout, operation()).await;

        match operation_result {
            Ok(Ok(result)) => {
                self.on_success().await;
                self.total_successes.fetch_add(1, Ordering::Relaxed);
                Ok(result)
            }
            Ok(Err(e)) => {
                self.on_failure().await;
                self.total_failures.fetch_add(1, Ordering::Relaxed);
                Err(e)
            }
            Err(_) => {
                // Timeout
                self.on_failure().await;
                self.total_failures.fetch_add(1, Ordering::Relaxed);
                Err(VideoRssError::Timeout)
            }
        }
    }

    async fn is_open(&self) -> bool {
        let state = self.state.read().await;
        match *state {
            CircuitState::Open => {
                // Check if we should transition to half-open
                if let Some(last_failure) = *self.last_failure_time.read().await {
                    if last_failure.elapsed() >= self.config.recovery_timeout {
                        drop(state);
                        self.transition_to_half_open().await;
                        return false;
                    }
                }
                true
            }
            CircuitState::HalfOpen => {
                // Allow limited traffic through
                false
            }
            CircuitState::Closed => false,
        }
    }

    async fn on_success(&self) {
        let state = self.state.read().await;
        match *state {
            CircuitState::HalfOpen => {
                let success_count = self.success_count.fetch_add(1, Ordering::Relaxed) + 1;
                if success_count >= self.config.success_threshold {
                    drop(state);
                    self.transition_to_closed().await;
                }
            }
            CircuitState::Closed => {
                // Reset failure count on success
                self.failure_count.store(0, Ordering::Relaxed);
            }
            CircuitState::Open => {
                // Should not happen, but reset counts
                self.failure_count.store(0, Ordering::Relaxed);
                self.success_count.store(0, Ordering::Relaxed);
            }
        }
    }

    async fn on_failure(&self) {
        let mut last_failure_time = self.last_failure_time.write().await;
        *last_failure_time = Some(Instant::now());
        drop(last_failure_time);

        let state = self.state.read().await;
        match *state {
            CircuitState::Closed => {
                let failure_count = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
                if failure_count >= self.config.failure_threshold {
                    drop(state);
                    self.transition_to_open().await;
                }
            }
            CircuitState::HalfOpen => {
                // Any failure in half-open should go back to open
                drop(state);
                self.transition_to_open().await;
            }
            CircuitState::Open => {
                // Already open, just increment failure count
                self.failure_count.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    async fn transition_to_open(&self) {
        let mut state = self.state.write().await;
        *state = CircuitState::Open;
        self.success_count.store(0, Ordering::Relaxed);
        warn!("Circuit breaker '{}' transitioned to OPEN", self.name);
    }

    async fn transition_to_half_open(&self) {
        let mut state = self.state.write().await;
        *state = CircuitState::HalfOpen;
        self.success_count.store(0, Ordering::Relaxed);
        info!("Circuit breaker '{}' transitioned to HALF-OPEN", self.name);
    }

    async fn transition_to_closed(&self) {
        let mut state = self.state.write().await;
        *state = CircuitState::Closed;
        self.failure_count.store(0, Ordering::Relaxed);
        self.success_count.store(0, Ordering::Relaxed);
        info!("Circuit breaker '{}' transitioned to CLOSED", self.name);
    }

    pub async fn get_stats(&self) -> CircuitBreakerStats {
        let state = self.state.read().await;
        let last_failure_time = self.last_failure_time.read().await;

        CircuitBreakerStats {
            state: state.clone(),
            failure_count: self.failure_count.load(Ordering::Relaxed),
            success_count: self.success_count.load(Ordering::Relaxed),
            last_failure_time: *last_failure_time,
            total_requests: self.total_requests.load(Ordering::Relaxed),
            total_failures: self.total_failures.load(Ordering::Relaxed),
            total_successes: self.total_successes.load(Ordering::Relaxed),
        }
    }

    pub async fn force_open(&self) {
        self.transition_to_open().await;
    }

    pub async fn force_closed(&self) {
        self.transition_to_closed().await;
    }
}

#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_retries: usize,
    pub initial_interval: Duration,
    pub max_interval: Duration,
    pub multiplier: f64,
    pub max_elapsed_time: Option<Duration>,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_interval: Duration::from_millis(100),
            max_interval: Duration::from_secs(10),
            multiplier: 2.0,
            max_elapsed_time: Some(Duration::from_secs(60)),
        }
    }
}

pub struct RetryableOperation;

impl RetryableOperation {
    pub async fn execute_with_retry<F, Fut, T>(
        operation: F,
        config: RetryConfig,
        operation_name: &str,
    ) -> Result<T>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let mut backoff = ExponentialBackoff {
            initial_interval: config.initial_interval,
            max_interval: config.max_interval,
            multiplier: config.multiplier,
            max_elapsed_time: config.max_elapsed_time,
            ..Default::default()
        };

        let mut attempt = 0;
        let start_time = Instant::now();

        let result = retry(backoff.clone(), || async {
            attempt += 1;
            debug!(
                "Executing {} (attempt {}/{})",
                operation_name,
                attempt,
                config.max_retries + 1
            );

            match operation().await {
                Ok(result) => {
                    if attempt > 1 {
                        info!(
                            "Operation '{}' succeeded after {} attempts in {:.2}s",
                            operation_name,
                            attempt,
                            start_time.elapsed().as_secs_f64()
                        );
                    }
                    Ok(result)
                }
                Err(e) => {
                    if attempt <= config.max_retries {
                        warn!(
                            "Operation '{}' failed (attempt {}): {}",
                            operation_name, attempt, e
                        );
                        Err(backoff::Error::transient(e))
                    } else {
                        error!(
                            "Operation '{}' failed permanently after {} attempts: {}",
                            operation_name, attempt, e
                        );
                        Err(backoff::Error::permanent(e))
                    }
                }
            }
        })
        .await;

        match result {
            Ok(value) => Ok(value),
            Err(e) => Err(e),
        }
    }

    pub async fn execute_with_circuit_breaker_and_retry<F, Fut, T>(
        operation: F,
        circuit_breaker: &CircuitBreaker,
        retry_config: RetryConfig,
        operation_name: &str,
    ) -> Result<T>
    where
        F: Fn() -> Fut + Clone,
        Fut: std::future::Future<Output = Result<T>>,
    {
        Self::execute_with_retry(
            || circuit_breaker.call(operation.clone()),
            retry_config,
            operation_name,
        )
        .await
    }
}

// Resilient HTTP client wrapper
pub struct ResilientHttpClient {
    client: reqwest::Client,
    circuit_breaker: CircuitBreaker,
    retry_config: RetryConfig,
}

impl ResilientHttpClient {
    pub fn new(
        client: reqwest::Client,
        circuit_breaker_config: CircuitBreakerConfig,
        retry_config: RetryConfig,
        service_name: String,
    ) -> Self {
        Self {
            client,
            circuit_breaker: CircuitBreaker::new(service_name, circuit_breaker_config),
            retry_config,
        }
    }

    pub async fn get(&self, url: &str) -> Result<reqwest::Response> {
        let url = url.to_string();
        let client = self.client.clone();

        RetryableOperation::execute_with_circuit_breaker_and_retry(
            move || {
                let client = client.clone();
                let url = url.clone();
                async move {
                    client
                        .get(&url)
                        .send()
                        .await
                        .map_err(|e| VideoRssError::Http(e))
                }
            },
            &self.circuit_breaker,
            self.retry_config.clone(),
            &format!("GET {}", url),
        )
        .await
    }

    pub async fn post(&self, url: &str) -> Result<reqwest::RequestBuilder> {
        // For POST requests, we return a builder since bodies can't be cloned easily
        // The actual retry logic would need to be implemented at a higher level
        Ok(self.client.post(url))
    }

    pub async fn get_stats(&self) -> CircuitBreakerStats {
        self.circuit_breaker.get_stats().await
    }

    pub async fn reset_circuit_breaker(&self) {
        self.circuit_breaker.force_closed().await;
    }
}

// Service health checker
pub struct HealthChecker {
    services: Vec<(String, ResilientHttpClient)>,
}

impl HealthChecker {
    pub fn new() -> Self {
        Self {
            services: Vec::new(),
        }
    }

    pub fn add_service(&mut self, name: String, client: ResilientHttpClient) {
        self.services.push((name, client));
    }

    pub async fn check_all_services(&self) -> Vec<(String, bool, Option<String>)> {
        let mut results = Vec::new();

        for (name, client) in &self.services {
            let health_url = format!("https://{}/health", name); // Adjust as needed

            match client.get(&health_url).await {
                Ok(response) => {
                    let is_healthy = response.status().is_success();
                    results.push((name.clone(), is_healthy, None));
                }
                Err(e) => {
                    results.push((name.clone(), false, Some(e.to_string())));
                }
            }
        }

        results
    }

    pub async fn get_service_stats(&self) -> Vec<(String, CircuitBreakerStats)> {
        let mut stats = Vec::new();

        for (name, client) in &self.services {
            let service_stats = client.get_stats().await;
            stats.push((name.clone(), service_stats));
        }

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    #[tokio::test]
    async fn test_circuit_breaker_success() {
        let cb = CircuitBreaker::new(
            "test".to_string(),
            CircuitBreakerConfig {
                failure_threshold: 2,
                ..Default::default()
            },
        );

        let result = cb
            .call(|| async { Ok::<i32, VideoRssError>(42) })
            .await
            .unwrap();

        assert_eq!(result, 42);

        let stats = cb.get_stats().await;
        assert_eq!(stats.state, CircuitState::Closed);
        assert_eq!(stats.total_successes, 1);
    }

    #[tokio::test]
    async fn test_circuit_breaker_failure_threshold() {
        let cb = CircuitBreaker::new(
            "test".to_string(),
            CircuitBreakerConfig {
                failure_threshold: 2,
                ..Default::default()
            },
        );

        // First failure
        let _ = cb
            .call(|| async { Err::<i32, VideoRssError>(VideoRssError::Unknown("test".to_string())) })
            .await;

        let stats = cb.get_stats().await;
        assert_eq!(stats.state, CircuitState::Closed);

        // Second failure - should trigger open
        let _ = cb
            .call(|| async { Err::<i32, VideoRssError>(VideoRssError::Unknown("test".to_string())) })
            .await;

        let stats = cb.get_stats().await;
        assert_eq!(stats.state, CircuitState::Open);
    }

    #[tokio::test]
    async fn test_retry_operation() {
        let attempt_count = Arc::new(AtomicUsize::new(0));
        let should_fail = Arc::new(AtomicBool::new(true));

        let attempt_count_clone = attempt_count.clone();
        let should_fail_clone = should_fail.clone();

        let config = RetryConfig {
            max_retries: 2,
            initial_interval: Duration::from_millis(10),
            ..Default::default()
        };

        let result = RetryableOperation::execute_with_retry(
            move || {
                let attempt_count = attempt_count_clone.clone();
                let should_fail = should_fail_clone.clone();
                async move {
                    let attempt = attempt_count.fetch_add(1, Ordering::Relaxed) + 1;

                    if attempt >= 2 {
                        should_fail.store(false, Ordering::Relaxed);
                    }

                    if should_fail.load(Ordering::Relaxed) {
                        Err(VideoRssError::Unknown("test failure".to_string()))
                    } else {
                        Ok(42)
                    }
                }
            },
            config,
            "test_operation",
        )
        .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
        assert_eq!(attempt_count.load(Ordering::Relaxed), 2);
    }
}