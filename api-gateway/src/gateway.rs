use anyhow::{Context, Result};
use axum::{
    body::Body,
    extract::{Path, Query, State},
    http::{HeaderMap, Method, Request, Response, StatusCode, Uri},
    middleware::{self, Next},
    response::IntoResponse,
    routing::{any, get},
    Router,
};
use bytes::Bytes;
use failsafe::{CircuitBreaker, Config as CircuitBreakerConfig, Error as CircuitError};
use futures::future::BoxFuture;
use governor::{Quota, RateLimiter};
use moka::future::Cache;
use redis::aio::MultiplexedConnection;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tower::ServiceBuilder;
use tower_http::{
    compression::CompressionLayer,
    cors::CorsLayer,
    timeout::TimeoutLayer,
    trace::{DefaultMakeSpan, TraceLayer},
};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct ApiGateway {
    routes: Arc<RwLock<RouteRegistry>>,
    services: Arc<ServiceRegistry>,
    rate_limiters: Arc<RwLock<HashMap<String, Arc<RateLimiter<String, governor::state::InMemoryState, governor::clock::DefaultClock>>>>>,
    circuit_breakers: Arc<RwLock<HashMap<String, Arc<CircuitBreaker>>>>,
    cache: Arc<Cache<String, CachedResponse>>,
    redis: Arc<MultiplexedConnection>,
    config: GatewayConfig,
    metrics: Arc<GatewayMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayConfig {
    pub enable_rate_limiting: bool,
    pub enable_circuit_breaker: bool,
    pub enable_caching: bool,
    pub enable_compression: bool,
    pub enable_cors: bool,
    pub request_timeout_ms: u64,
    pub max_request_body_size: usize,
    pub cache_ttl_seconds: u64,
    pub rate_limit_requests_per_second: u32,
    pub circuit_breaker_failure_threshold: u32,
    pub circuit_breaker_recovery_timeout_ms: u64,
    pub enable_request_logging: bool,
    pub enable_response_caching: bool,
    pub load_balancing_strategy: LoadBalancingStrategy,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            enable_rate_limiting: true,
            enable_circuit_breaker: true,
            enable_caching: true,
            enable_compression: true,
            enable_cors: true,
            request_timeout_ms: 30000,
            max_request_body_size: 10 * 1024 * 1024, // 10MB
            cache_ttl_seconds: 300,
            rate_limit_requests_per_second: 100,
            circuit_breaker_failure_threshold: 5,
            circuit_breaker_recovery_timeout_ms: 60000,
            enable_request_logging: true,
            enable_response_caching: true,
            load_balancing_strategy: LoadBalancingStrategy::RoundRobin,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastConnections,
    WeightedRoundRobin,
    Random,
    IpHash,
}

#[derive(Debug, Clone)]
struct RouteRegistry {
    routes: HashMap<String, Route>,
    patterns: Vec<(regex::Regex, String)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Route {
    pub id: String,
    pub path: String,
    pub service: String,
    pub methods: Vec<Method>,
    pub strip_prefix: bool,
    pub rewrite_path: Option<String>,
    pub timeout_ms: Option<u64>,
    pub rate_limit: Option<u32>,
    pub authentication_required: bool,
    pub roles: Vec<String>,
    pub cache_ttl: Option<u64>,
    pub retry_policy: RetryPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub backoff_ms: u64,
    pub backoff_multiplier: f32,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            backoff_ms: 100,
            backoff_multiplier: 2.0,
        }
    }
}

#[derive(Debug, Clone)]
struct ServiceRegistry {
    services: Arc<RwLock<HashMap<String, ServiceInstance>>>,
    health_checks: Arc<RwLock<HashMap<String, HealthStatus>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceInstance {
    pub id: String,
    pub name: String,
    pub endpoints: Vec<ServiceEndpoint>,
    pub health_check_path: String,
    pub health_check_interval_ms: u64,
    pub weight: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceEndpoint {
    pub url: String,
    pub weight: u32,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub service_id: String,
    pub endpoint: String,
    pub healthy: bool,
    pub last_check: chrono::DateTime<chrono::Utc>,
    pub consecutive_failures: u32,
    pub response_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedResponse {
    pub status: u16,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
    pub cached_at: Instant,
}

#[derive(Debug, Default, Clone)]
pub struct GatewayMetrics {
    pub total_requests: Arc<RwLock<u64>>,
    pub successful_requests: Arc<RwLock<u64>>,
    pub failed_requests: Arc<RwLock<u64>>,
    pub cached_responses: Arc<RwLock<u64>>,
    pub rate_limited_requests: Arc<RwLock<u64>>,
    pub circuit_breaker_opens: Arc<RwLock<u64>>,
    pub average_response_time_ms: Arc<RwLock<f64>>,
    pub service_request_counts: Arc<RwLock<HashMap<String, u64>>>,
}

impl ApiGateway {
    pub async fn new(config: GatewayConfig, redis_url: &str) -> Result<Self> {
        let redis_client = redis::Client::open(redis_url)?;
        let redis = Arc::new(redis_client.get_connection_manager().await?);

        let cache = Arc::new(
            Cache::builder()
                .max_capacity(10_000)
                .time_to_live(Duration::from_secs(config.cache_ttl_seconds))
                .build(),
        );

        Ok(Self {
            routes: Arc::new(RwLock::new(RouteRegistry {
                routes: HashMap::new(),
                patterns: Vec::new(),
            })),
            services: Arc::new(ServiceRegistry {
                services: Arc::new(RwLock::new(HashMap::new())),
                health_checks: Arc::new(RwLock::new(HashMap::new())),
            }),
            rate_limiters: Arc::new(RwLock::new(HashMap::new())),
            circuit_breakers: Arc::new(RwLock::new(HashMap::new())),
            cache,
            redis,
            config,
            metrics: Arc::new(GatewayMetrics::default()),
        })
    }

    pub async fn register_route(&self, route: Route) -> Result<()> {
        let mut registry = self.routes.write().await;

        // Compile regex pattern for the route
        let pattern = regex::Regex::new(&route.path.replace("{", "(?P<").replace("}", ">[^/]+)"))?;
        registry.patterns.push((pattern, route.id.clone()));
        registry.routes.insert(route.id.clone(), route.clone());

        // Initialize rate limiter for this route if needed
        if let Some(rate_limit) = route.rate_limit {
            let quota = Quota::per_second(std::num::NonZeroU32::new(rate_limit).unwrap());
            let limiter = Arc::new(RateLimiter::direct(quota));
            self.rate_limiters.write().await.insert(route.id.clone(), limiter);
        }

        // Initialize circuit breaker for this service
        if self.config.enable_circuit_breaker {
            let cb_config = CircuitBreakerConfig::new()
                .failure_threshold(self.config.circuit_breaker_failure_threshold as usize)
                .recovery_timeout(Duration::from_millis(self.config.circuit_breaker_recovery_timeout_ms));

            let circuit_breaker = Arc::new(CircuitBreaker::new(cb_config));
            self.circuit_breakers.write().await.insert(route.service.clone(), circuit_breaker);
        }

        info!("Registered route: {} -> {}", route.path, route.service);
        Ok(())
    }

    pub async fn register_service(&self, service: ServiceInstance) -> Result<()> {
        self.services.services.write().await.insert(service.id.clone(), service.clone());

        // Start health checking for this service
        self.start_health_checking(service).await?;

        Ok(())
    }

    async fn start_health_checking(&self, service: ServiceInstance) -> Result<()> {
        let services = Arc::clone(&self.services);
        let interval = Duration::from_millis(service.health_check_interval_ms);

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                interval_timer.tick().await;

                for endpoint in &service.endpoints {
                    let start = Instant::now();
                    let client = reqwest::Client::new();

                    let health_url = format!("{}{}", endpoint.url, service.health_check_path);
                    let result = client
                        .get(&health_url)
                        .timeout(Duration::from_secs(5))
                        .send()
                        .await;

                    let response_time_ms = start.elapsed().as_millis() as u64;

                    let mut health_checks = services.health_checks.write().await;
                    let key = format!("{}:{}", service.id, endpoint.url);

                    let healthy = result.map(|r| r.status().is_success()).unwrap_or(false);

                    let status = health_checks.entry(key.clone()).or_insert(HealthStatus {
                        service_id: service.id.clone(),
                        endpoint: endpoint.url.clone(),
                        healthy,
                        last_check: chrono::Utc::now(),
                        consecutive_failures: 0,
                        response_time_ms,
                    });

                    status.healthy = healthy;
                    status.last_check = chrono::Utc::now();
                    status.response_time_ms = response_time_ms;

                    if !healthy {
                        status.consecutive_failures += 1;
                        warn!(
                            "Health check failed for {} (failures: {})",
                            endpoint.url, status.consecutive_failures
                        );
                    } else {
                        status.consecutive_failures = 0;
                    }
                }
            }
        });

        Ok(())
    }

    pub async fn handle_request(
        &self,
        method: Method,
        uri: Uri,
        headers: HeaderMap,
        body: Bytes,
    ) -> Result<Response<Body>> {
        let start = Instant::now();

        // Update metrics
        {
            let mut total = self.metrics.total_requests.write().await;
            *total += 1;
        }

        // Find matching route
        let route = self.find_route(&uri).await?;

        // Check authentication if required
        if route.authentication_required {
            self.verify_authentication(&headers, &route.roles).await?;
        }

        // Apply rate limiting
        if self.config.enable_rate_limiting {
            if let Some(rate_limit) = route.rate_limit {
                self.apply_rate_limiting(&route.id).await?;
            }
        }

        // Check cache
        if self.config.enable_caching && method == Method::GET {
            let cache_key = format!("{}:{}", method, uri);
            if let Some(cached) = self.cache.get(&cache_key).await {
                let mut cached_responses = self.metrics.cached_responses.write().await;
                *cached_responses += 1;

                return self.build_response_from_cache(cached);
            }
        }

        // Get service endpoint
        let endpoint = self.select_endpoint(&route.service).await?;

        // Apply circuit breaker
        let response = if self.config.enable_circuit_breaker {
            self.call_with_circuit_breaker(&route.service, || {
                self.proxy_request(method.clone(), uri.clone(), headers.clone(), body.clone(), &endpoint, &route)
            }).await?
        } else {
            self.proxy_request(method.clone(), uri.clone(), headers.clone(), body.clone(), &endpoint, &route).await?
        };

        // Cache response if applicable
        if self.config.enable_response_caching && method == Method::GET && response.status().is_success() {
            let cache_ttl = route.cache_ttl.unwrap_or(self.config.cache_ttl_seconds);
            self.cache_response(&format!("{}:{}", method, uri), &response, cache_ttl).await?;
        }

        // Update metrics
        let duration = start.elapsed();
        self.update_metrics(response.status().is_success(), duration).await;

        Ok(response)
    }

    async fn find_route(&self, uri: &Uri) -> Result<Route> {
        let registry = self.routes.read().await;
        let path = uri.path();

        for (pattern, route_id) in &registry.patterns {
            if pattern.is_match(path) {
                if let Some(route) = registry.routes.get(route_id) {
                    return Ok(route.clone());
                }
            }
        }

        Err(anyhow::anyhow!("No route found for path: {}", path))
    }

    async fn verify_authentication(&self, headers: &HeaderMap, required_roles: &[String]) -> Result<()> {
        // Extract JWT token from Authorization header
        let auth_header = headers
            .get("Authorization")
            .and_then(|h| h.to_str().ok())
            .ok_or_else(|| anyhow::anyhow!("Missing Authorization header"))?;

        if !auth_header.starts_with("Bearer ") {
            return Err(anyhow::anyhow!("Invalid Authorization header format"));
        }

        let token = &auth_header[7..];

        // Verify token and check roles (simplified - use proper JWT library)
        // In production, validate JWT signature and claims
        if token.is_empty() {
            return Err(anyhow::anyhow!("Invalid token"));
        }

        // Check if user has required roles
        // This would be extracted from JWT claims in production

        Ok(())
    }

    async fn apply_rate_limiting(&self, route_id: &str) -> Result<()> {
        let limiters = self.rate_limiters.read().await;

        if let Some(limiter) = limiters.get(route_id) {
            match limiter.check_key(&route_id.to_string()) {
                Ok(_) => Ok(()),
                Err(_) => {
                    let mut rate_limited = self.metrics.rate_limited_requests.write().await;
                    *rate_limited += 1;

                    Err(anyhow::anyhow!("Rate limit exceeded"))
                }
            }
        } else {
            Ok(())
        }
    }

    async fn select_endpoint(&self, service_name: &str) -> Result<String> {
        let services = self.services.services.read().await;
        let health_checks = self.services.health_checks.read().await;

        let service = services
            .get(service_name)
            .ok_or_else(|| anyhow::anyhow!("Service not found: {}", service_name))?;

        // Filter healthy endpoints
        let healthy_endpoints: Vec<_> = service
            .endpoints
            .iter()
            .filter(|ep| {
                let key = format!("{}:{}", service.id, ep.url);
                health_checks
                    .get(&key)
                    .map(|h| h.healthy)
                    .unwrap_or(true) // Default to healthy if no health check data
            })
            .collect();

        if healthy_endpoints.is_empty() {
            return Err(anyhow::anyhow!("No healthy endpoints for service: {}", service_name));
        }

        // Select endpoint based on load balancing strategy
        let endpoint = match self.config.load_balancing_strategy {
            LoadBalancingStrategy::RoundRobin => {
                // Simple round-robin (in production, maintain counter)
                healthy_endpoints[0].url.clone()
            }
            LoadBalancingStrategy::Random => {
                use rand::seq::SliceRandom;
                healthy_endpoints
                    .choose(&mut rand::thread_rng())
                    .map(|ep| ep.url.clone())
                    .unwrap()
            }
            LoadBalancingStrategy::WeightedRoundRobin => {
                healthy_endpoints[0].url.clone()
            }
            LoadBalancingStrategy::LeastConnections => {
                // In production, track active connections per endpoint
                healthy_endpoints[0].url.clone()
            }
            LoadBalancingStrategy::IpHash => {
                // In production, hash client IP to select endpoint
                healthy_endpoints[0].url.clone()
            }
        };

        Ok(endpoint)
    }

    async fn call_with_circuit_breaker<F, Fut>(&self, service: &str, f: F) -> Result<Response<Body>>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<Response<Body>>>,
    {
        let breakers = self.circuit_breakers.read().await;

        if let Some(breaker) = breakers.get(service) {
            match breaker.call(|| async { f().await.map_err(|_| CircuitError::Inner(std::io::Error::new(std::io::ErrorKind::Other, "error"))) }).await {
                Ok(response) => Ok(response),
                Err(_) => {
                    let mut opens = self.metrics.circuit_breaker_opens.write().await;
                    *opens += 1;

                    Err(anyhow::anyhow!("Circuit breaker open for service: {}", service))
                }
            }
        } else {
            f().await
        }
    }

    async fn proxy_request(
        &self,
        method: Method,
        uri: Uri,
        headers: HeaderMap,
        body: Bytes,
        endpoint: &str,
        route: &Route,
    ) -> Result<Response<Body>> {
        let client = reqwest::Client::new();

        // Build target URL
        let path = if route.strip_prefix {
            uri.path().strip_prefix(&route.path).unwrap_or(uri.path())
        } else {
            uri.path()
        };

        let target_path = if let Some(rewrite) = &route.rewrite_path {
            rewrite.replace("{path}", path)
        } else {
            path.to_string()
        };

        let target_url = format!("{}{}", endpoint, target_path);

        // Build request
        let mut req = client.request(method, &target_url);

        // Copy headers
        for (name, value) in headers.iter() {
            if name != "host" {
                req = req.header(name, value);
            }
        }

        // Set body
        req = req.body(body);

        // Set timeout
        let timeout = route
            .timeout_ms
            .unwrap_or(self.config.request_timeout_ms);
        req = req.timeout(Duration::from_millis(timeout));

        // Execute request with retry
        let mut attempt = 0;
        let mut last_error = None;

        while attempt < route.retry_policy.max_attempts {
            match req.try_clone().unwrap().send().await {
                Ok(resp) => {
                    let status = resp.status();
                    let headers = resp.headers().clone();
                    let body_bytes = resp.bytes().await?;

                    let mut response = Response::builder().status(status);

                    for (name, value) in headers.iter() {
                        response = response.header(name, value);
                    }

                    return Ok(response.body(Body::from(body_bytes))?);
                }
                Err(e) => {
                    last_error = Some(e);
                    attempt += 1;

                    if attempt < route.retry_policy.max_attempts {
                        let backoff = route.retry_policy.backoff_ms as f32
                            * route.retry_policy.backoff_multiplier.powi(attempt as i32 - 1);
                        tokio::time::sleep(Duration::from_millis(backoff as u64)).await;
                    }
                }
            }
        }

        Err(anyhow::anyhow!(
            "Failed after {} attempts: {:?}",
            attempt,
            last_error
        ))
    }

    fn build_response_from_cache(&self, cached: CachedResponse) -> Result<Response<Body>> {
        let mut response = Response::builder().status(cached.status);

        for (name, value) in cached.headers {
            response = response.header(name, value);
        }

        response = response.header("X-Cache", "HIT");

        Ok(response.body(Body::from(cached.body))?)
    }

    async fn cache_response(&self, key: &str, response: &Response<Body>, ttl_seconds: u64) -> Result<()> {
        // Note: This is simplified - in production, we'd need to properly consume and reconstruct the body
        let cached = CachedResponse {
            status: response.status().as_u16(),
            headers: HashMap::new(), // Would extract headers properly
            body: Vec::new(), // Would extract body properly
            cached_at: Instant::now(),
        };

        self.cache.insert(key.to_string(), cached).await;

        Ok(())
    }

    async fn update_metrics(&self, success: bool, duration: Duration) {
        if success {
            let mut successful = self.metrics.successful_requests.write().await;
            *successful += 1;
        } else {
            let mut failed = self.metrics.failed_requests.write().await;
            *failed += 1;
        }

        // Update average response time
        let mut avg_time = self.metrics.average_response_time_ms.write().await;
        let total = *self.metrics.total_requests.read().await as f64;
        *avg_time = (*avg_time * (total - 1.0) + duration.as_millis() as f64) / total;
    }

    pub async fn get_metrics(&self) -> GatewayMetrics {
        GatewayMetrics {
            total_requests: Arc::clone(&self.metrics.total_requests),
            successful_requests: Arc::clone(&self.metrics.successful_requests),
            failed_requests: Arc::clone(&self.metrics.failed_requests),
            cached_responses: Arc::clone(&self.metrics.cached_responses),
            rate_limited_requests: Arc::clone(&self.metrics.rate_limited_requests),
            circuit_breaker_opens: Arc::clone(&self.metrics.circuit_breaker_opens),
            average_response_time_ms: Arc::clone(&self.metrics.average_response_time_ms),
            service_request_counts: Arc::clone(&self.metrics.service_request_counts),
        }
    }
}

pub fn create_router(gateway: Arc<ApiGateway>) -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/metrics", get(get_metrics))
        .route("/{*path}", any(handle_gateway_request))
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(CompressionLayer::new())
                .layer(CorsLayer::permissive())
                .layer(TimeoutLayer::new(Duration::from_secs(30))),
        )
        .with_state(gateway)
}

async fn health_check() -> impl IntoResponse {
    (StatusCode::OK, "OK")
}

async fn get_metrics(State(gateway): State<Arc<ApiGateway>>) -> impl IntoResponse {
    let metrics = gateway.get_metrics().await;

    let response = serde_json::json!({
        "total_requests": *metrics.total_requests.read().await,
        "successful_requests": *metrics.successful_requests.read().await,
        "failed_requests": *metrics.failed_requests.read().await,
        "cached_responses": *metrics.cached_responses.read().await,
        "rate_limited_requests": *metrics.rate_limited_requests.read().await,
        "circuit_breaker_opens": *metrics.circuit_breaker_opens.read().await,
        "average_response_time_ms": *metrics.average_response_time_ms.read().await,
        "service_request_counts": *metrics.service_request_counts.read().await,
    });

    (StatusCode::OK, axum::Json(response))
}

async fn handle_gateway_request(
    State(gateway): State<Arc<ApiGateway>>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    match gateway.handle_request(method, uri, headers, body).await {
        Ok(response) => response,
        Err(e) => {
            error!("Gateway error: {}", e);
            Response::builder()
                .status(StatusCode::BAD_GATEWAY)
                .body(Body::from(format!("Gateway error: {}", e)))
                .unwrap()
        }
    }
}