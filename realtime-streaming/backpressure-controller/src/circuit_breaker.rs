use anyhow::Result;
use ringbuffer::{AllocRingBuffer, RingBuffer};
use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::sync::RwLock;
use tracing::{info, warn};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CircuitBreakerState {
    Closed,   // Normal operation
    Open,     // Blocking requests
    HalfOpen, // Testing if system recovered
}

pub struct CircuitBreaker {
    state: Arc<RwLock<CircuitBreakerState>>,
    error_threshold: f64,
    timeout: Duration,
    last_failure_time: Arc<RwLock<Option<Instant>>>,

    // Rolling window for tracking success/failure rates
    success_count: Arc<AtomicU64>,
    failure_count: Arc<AtomicU64>,
    total_requests: Arc<AtomicU64>,

    // Recent results for more granular tracking
    recent_results: Arc<RwLock<AllocRingBuffer<bool>>>, // true = success, false = failure

    // Configuration
    min_requests_before_trip: u64,
    half_open_max_requests: u64,
    reset_timeout: Duration,
}

impl CircuitBreaker {
    pub async fn new(error_threshold: f64, timeout: Duration) -> Result<Self> {
        Ok(Self {
            state: Arc::new(RwLock::new(CircuitBreakerState::Closed)),
            error_threshold,
            timeout,
            last_failure_time: Arc::new(RwLock::new(None)),
            success_count: Arc::new(AtomicU64::new(0)),
            failure_count: Arc::new(AtomicU64::new(0)),
            total_requests: Arc::new(AtomicU64::new(0)),
            recent_results: Arc::new(RwLock::new(AllocRingBuffer::new(1000))), // Track last 1000 requests
            min_requests_before_trip: 20, // Need at least 20 requests before considering tripping
            half_open_max_requests: 5,    // Allow max 5 requests in half-open state
            reset_timeout: timeout,
        })
    }

    pub async fn record_success(&self) {
        self.success_count.fetch_add(1, Ordering::Relaxed);
        self.total_requests.fetch_add(1, Ordering::Relaxed);

        // Add to recent results
        {
            let mut recent = self.recent_results.write().await;
            recent.push(true);
        }

        // Check if we should transition from half-open to closed
        let current_state = *self.state.read().await;
        if current_state == CircuitBreakerState::HalfOpen {
            let success_rate = self.calculate_recent_success_rate().await;
            if success_rate > (1.0 - self.error_threshold) {
                self.transition_to_closed().await;
            }
        }
    }

    pub async fn record_failure(&self) {
        self.failure_count.fetch_add(1, Ordering::Relaxed);
        self.total_requests.fetch_add(1, Ordering::Relaxed);

        // Add to recent results
        {
            let mut recent = self.recent_results.write().await;
            recent.push(false);
        }

        // Update last failure time
        {
            let mut last_failure = self.last_failure_time.write().await;
            *last_failure = Some(Instant::now());
        }

        // Check if we should trip the circuit breaker
        let current_state = *self.state.read().await;
        match current_state {
            CircuitBreakerState::Closed => {
                if self.should_trip().await {
                    self.transition_to_open().await;
                }
            }
            CircuitBreakerState::HalfOpen => {
                // Any failure in half-open state trips the breaker
                self.transition_to_open().await;
            }
            CircuitBreakerState::Open => {
                // Already open, just update timestamp
            }
        }
    }

    pub async fn is_open(&self) -> bool {
        let state = *self.state.read().await;

        match state {
            CircuitBreakerState::Open => {
                // Check if enough time has passed to try half-open
                if let Some(last_failure) = *self.last_failure_time.read().await {
                    if last_failure.elapsed() >= self.reset_timeout {
                        self.transition_to_half_open().await;
                        false // Allow the request to proceed
                    } else {
                        true // Still in timeout period
                    }
                } else {
                    true
                }
            }
            CircuitBreakerState::HalfOpen => {
                // Check if we've exceeded max requests in half-open
                let recent_count = self.get_recent_request_count().await;
                recent_count >= self.half_open_max_requests
            }
            CircuitBreakerState::Closed => false,
        }
    }

    async fn should_trip(&self) -> bool {
        let total = self.total_requests.load(Ordering::Relaxed);

        // Don't trip unless we have enough data
        if total < self.min_requests_before_trip {
            return false;
        }

        let error_rate = self.calculate_recent_error_rate().await;

        // Trip if error rate exceeds threshold
        error_rate > self.error_threshold
    }

    async fn calculate_recent_error_rate(&self) -> f64 {
        let recent = self.recent_results.read().await;
        let total_recent = recent.len();

        if total_recent == 0 {
            return 0.0;
        }

        let failures = recent.iter().filter(|&&result| !result).count();
        failures as f64 / total_recent as f64
    }

    async fn calculate_recent_success_rate(&self) -> f64 {
        1.0 - self.calculate_recent_error_rate().await
    }

    async fn get_recent_request_count(&self) -> u64 {
        let recent = self.recent_results.read().await;

        // Count requests in the last 10 seconds (approximate)
        // This is a simplified implementation - in production you'd track timestamps
        let recent_window_size = std::cmp::min(recent.len(), 100); // Last 100 requests as proxy
        recent_window_size as u64
    }

    async fn transition_to_open(&self) {
        let mut state = self.state.write().await;
        if *state != CircuitBreakerState::Open {
            *state = CircuitBreakerState::Open;
            warn!("Circuit breaker OPENED - blocking requests");
        }
    }

    async fn transition_to_half_open(&self) {
        let mut state = self.state.write().await;
        *state = CircuitBreakerState::HalfOpen;
        info!("Circuit breaker HALF-OPEN - testing system recovery");
    }

    async fn transition_to_closed(&self) {
        let mut state = self.state.write().await;
        *state = CircuitBreakerState::Closed;
        info!("Circuit breaker CLOSED - normal operation resumed");

        // Reset failure timestamp
        {
            let mut last_failure = self.last_failure_time.write().await;
            *last_failure = None;
        }
    }

    pub async fn reset(&self) {
        self.transition_to_closed().await;

        // Reset counters
        self.success_count.store(0, Ordering::Relaxed);
        self.failure_count.store(0, Ordering::Relaxed);
        self.total_requests.store(0, Ordering::Relaxed);

        // Clear recent results
        {
            let mut recent = self.recent_results.write().await;
            recent.clear();
        }

        info!("Circuit breaker manually reset");
    }

    pub async fn get_state(&self) -> CircuitBreakerState {
        *self.state.read().await
    }

    pub async fn get_error_rate(&self) -> f64 {
        self.calculate_recent_error_rate().await
    }

    pub async fn get_stats(&self) -> CircuitBreakerStats {
        let state = *self.state.read().await;
        let total = self.total_requests.load(Ordering::Relaxed);
        let successes = self.success_count.load(Ordering::Relaxed);
        let failures = self.failure_count.load(Ordering::Relaxed);

        let error_rate = if total > 0 {
            failures as f64 / total as f64
        } else {
            0.0
        };

        let last_failure = *self.last_failure_time.read().await;
        let time_since_last_failure = last_failure.map(|t| t.elapsed());

        CircuitBreakerStats {
            state,
            total_requests: total,
            successful_requests: successes,
            failed_requests: failures,
            error_rate,
            error_threshold: self.error_threshold,
            time_since_last_failure,
            timeout: self.timeout,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CircuitBreakerStats {
    pub state: CircuitBreakerState,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub error_rate: f64,
    pub error_threshold: f64,
    pub time_since_last_failure: Option<Duration>,
    pub timeout: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_circuit_breaker_normal_operation() {
        let cb = CircuitBreaker::new(0.5, Duration::from_secs(60)).await.unwrap();

        assert_eq!(cb.get_state().await, CircuitBreakerState::Closed);
        assert!(!cb.is_open().await);

        // Record some successes
        for _ in 0..10 {
            cb.record_success().await;
        }

        assert_eq!(cb.get_state().await, CircuitBreakerState::Closed);
        assert!(!cb.is_open().await);
    }

    #[tokio::test]
    async fn test_circuit_breaker_trips_on_high_error_rate() {
        let cb = CircuitBreaker::new(0.3, Duration::from_secs(60)).await.unwrap();

        // Generate enough requests to trigger evaluation
        for _ in 0..15 {
            cb.record_success().await;
        }

        // Add failures to push error rate over threshold
        for _ in 0..10 {
            cb.record_failure().await;
        }

        assert_eq!(cb.get_state().await, CircuitBreakerState::Open);
        assert!(cb.is_open().await);
    }

    #[tokio::test]
    async fn test_circuit_breaker_recovery() {
        let cb = CircuitBreaker::new(0.3, Duration::from_millis(100)).await.unwrap();

        // Trip the circuit breaker
        for _ in 0..25 {
            if 25 % 3 == 0 {
                cb.record_success().await;
            } else {
                cb.record_failure().await;
            }
        }

        assert_eq!(cb.get_state().await, CircuitBreakerState::Open);

        // Wait for timeout
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Should allow one request (half-open)
        assert!(!cb.is_open().await);

        // Record success to close the circuit
        cb.record_success().await;
        assert_eq!(cb.get_state().await, CircuitBreakerState::Closed);
    }
}