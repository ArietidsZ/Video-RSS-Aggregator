use reqwest::{Client, Proxy};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::VecDeque;
use crate::{Result, ExtractorError};

/// Proxy manager for rotating proxy servers
pub struct ProxyManager {
    proxies: Arc<RwLock<VecDeque<ProxyConfig>>>,
    current_index: Arc<RwLock<usize>>,
    health_check_interval: std::time::Duration,
}

#[derive(Clone, Debug)]
pub struct ProxyConfig {
    pub url: String,
    pub username: Option<String>,
    pub password: Option<String>,
    pub proxy_type: ProxyType,
    pub country: Option<String>,
    pub is_healthy: bool,
    pub last_used: std::time::Instant,
    pub success_count: u64,
    pub failure_count: u64,
}

#[derive(Clone, Debug)]
pub enum ProxyType {
    HTTP,
    HTTPS,
    SOCKS5,
}

impl ProxyManager {
    pub fn new() -> Self {
        Self {
            proxies: Arc::new(RwLock::new(VecDeque::new())),
            current_index: Arc::new(RwLock::new(0)),
            health_check_interval: std::time::Duration::from_secs(300), // 5 minutes
        }
    }

    /// Add a proxy to the rotation pool
    pub async fn add_proxy(&self, config: ProxyConfig) {
        let mut proxies = self.proxies.write().await;
        proxies.push_back(config);
    }

    /// Add multiple proxies
    pub async fn add_proxies(&self, configs: Vec<ProxyConfig>) {
        let mut proxies = self.proxies.write().await;
        for config in configs {
            proxies.push_back(config);
        }
    }

    /// Get the next healthy proxy in rotation
    pub async fn get_next(&self) -> Option<ProxyConfig> {
        let mut proxies = self.proxies.write().await;
        let mut current_index = self.current_index.write().await;

        if proxies.is_empty() {
            return None;
        }

        let mut attempts = 0;
        let max_attempts = proxies.len();

        while attempts < max_attempts {
            *current_index = (*current_index + 1) % proxies.len();

            if let Some(proxy) = proxies.get(*current_index) {
                if proxy.is_healthy {
                    let mut proxy = proxy.clone();
                    proxy.last_used = std::time::Instant::now();
                    proxies[*current_index] = proxy.clone();
                    return Some(proxy);
                }
            }

            attempts += 1;
        }

        None
    }

    /// Get a proxy for a specific country
    pub async fn get_by_country(&self, country: &str) -> Option<ProxyConfig> {
        let proxies = self.proxies.read().await;

        for proxy in proxies.iter() {
            if proxy.is_healthy && proxy.country.as_deref() == Some(country) {
                return Some(proxy.clone());
            }
        }

        None
    }

    /// Mark proxy as failed
    pub async fn mark_failed(&self, proxy_url: &str) {
        let mut proxies = self.proxies.write().await;

        for proxy in proxies.iter_mut() {
            if proxy.url == proxy_url {
                proxy.failure_count += 1;

                // Mark unhealthy after 3 consecutive failures
                if proxy.failure_count > 3 && proxy.success_count == 0 {
                    proxy.is_healthy = false;
                }

                break;
            }
        }
    }

    /// Mark proxy as successful
    pub async fn mark_success(&self, proxy_url: &str) {
        let mut proxies = self.proxies.write().await;

        for proxy in proxies.iter_mut() {
            if proxy.url == proxy_url {
                proxy.success_count += 1;
                proxy.failure_count = 0; // Reset failure count on success
                proxy.is_healthy = true;
                break;
            }
        }
    }

    /// Create HTTP client with proxy
    pub fn create_client(&self, proxy: &ProxyConfig) -> Result<Client> {
        let mut builder = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36");

        let reqwest_proxy = match proxy.proxy_type {
            ProxyType::HTTP => Proxy::http(&proxy.url),
            ProxyType::HTTPS => Proxy::https(&proxy.url),
            ProxyType::SOCKS5 => Proxy::all(&proxy.url),
        }.map_err(|e| ExtractorError::ProxyError(format!("Invalid proxy URL: {}", e)))?;

        let reqwest_proxy = if let (Some(username), Some(password)) = (&proxy.username, &proxy.password) {
            reqwest_proxy.basic_auth(username, password)
        } else {
            reqwest_proxy
        };

        builder = builder.proxy(reqwest_proxy);

        builder.build()
            .map_err(|e| ExtractorError::ProxyError(format!("Failed to build client: {}", e)))
    }

    /// Health check all proxies
    pub async fn health_check_all(&self) {
        let proxies = self.proxies.read().await.clone();
        drop(self.proxies.read().await);

        for proxy in proxies {
            let is_healthy = self.check_proxy_health(&proxy).await;

            let mut proxies = self.proxies.write().await;
            for p in proxies.iter_mut() {
                if p.url == proxy.url {
                    p.is_healthy = is_healthy;
                    break;
                }
            }
        }
    }

    /// Check if a specific proxy is healthy
    async fn check_proxy_health(&self, proxy: &ProxyConfig) -> bool {
        match self.create_client(proxy) {
            Ok(client) => {
                // Try to connect to a reliable endpoint
                let result = client
                    .get("https://httpbin.org/ip")
                    .send()
                    .await;

                result.is_ok()
            }
            Err(_) => false,
        }
    }

    /// Start background health check task
    pub fn start_health_check_task(self: Arc<Self>) {
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(self.health_check_interval).await;
                self.health_check_all().await;
            }
        });
    }

    /// Get statistics for all proxies
    pub async fn get_stats(&self) -> Vec<ProxyStats> {
        let proxies = self.proxies.read().await;

        proxies.iter().map(|p| ProxyStats {
            url: p.url.clone(),
            country: p.country.clone(),
            is_healthy: p.is_healthy,
            success_rate: if p.success_count + p.failure_count > 0 {
                p.success_count as f64 / (p.success_count + p.failure_count) as f64
            } else {
                0.0
            },
            total_requests: p.success_count + p.failure_count,
        }).collect()
    }

    /// Remove unhealthy proxies
    pub async fn remove_unhealthy(&self) {
        let mut proxies = self.proxies.write().await;
        proxies.retain(|p| p.is_healthy || p.failure_count < 10);
    }

    /// Load proxies from file
    pub async fn load_from_file(&self, path: &str) -> Result<()> {
        let content = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| ExtractorError::Io(e))?;

        let mut configs = Vec::new();

        for line in content.lines() {
            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }

            // Format: type://user:pass@host:port#country
            if let Some(config) = Self::parse_proxy_line(line) {
                configs.push(config);
            }
        }

        self.add_proxies(configs).await;
        Ok(())
    }

    /// Parse proxy configuration from string
    fn parse_proxy_line(line: &str) -> Option<ProxyConfig> {
        // Simple parser for proxy format
        let parts: Vec<&str> = line.split('#').collect();
        let url = parts[0];
        let country = parts.get(1).map(|s| s.to_string());

        let proxy_type = if url.starts_with("http://") {
            ProxyType::HTTP
        } else if url.starts_with("https://") {
            ProxyType::HTTPS
        } else if url.starts_with("socks5://") {
            ProxyType::SOCKS5
        } else {
            return None;
        };

        // Extract credentials if present
        let (username, password) = if url.contains('@') {
            let parts: Vec<&str> = url.split("://").collect();
            if parts.len() >= 2 {
                let auth_parts: Vec<&str> = parts[1].split('@').collect();
                if auth_parts.len() >= 2 {
                    let creds: Vec<&str> = auth_parts[0].split(':').collect();
                    if creds.len() == 2 {
                        (Some(creds[0].to_string()), Some(creds[1].to_string()))
                    } else {
                        (None, None)
                    }
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        Some(ProxyConfig {
            url: url.to_string(),
            username,
            password,
            proxy_type,
            country,
            is_healthy: true,
            last_used: std::time::Instant::now(),
            success_count: 0,
            failure_count: 0,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ProxyStats {
    pub url: String,
    pub country: Option<String>,
    pub is_healthy: bool,
    pub success_rate: f64,
    pub total_requests: u64,
}

/// Smart proxy selector based on performance
pub struct SmartProxySelector {
    manager: Arc<ProxyManager>,
    strategy: SelectionStrategy,
}

#[derive(Clone)]
pub enum SelectionStrategy {
    RoundRobin,
    LeastUsed,
    BestPerformance,
    Random,
}

impl SmartProxySelector {
    pub fn new(manager: Arc<ProxyManager>, strategy: SelectionStrategy) -> Self {
        Self { manager, strategy }
    }

    pub async fn select(&self) -> Option<ProxyConfig> {
        match self.strategy {
            SelectionStrategy::RoundRobin => self.manager.get_next().await,
            SelectionStrategy::LeastUsed => self.select_least_used().await,
            SelectionStrategy::BestPerformance => self.select_best_performance().await,
            SelectionStrategy::Random => self.select_random().await,
        }
    }

    async fn select_least_used(&self) -> Option<ProxyConfig> {
        let proxies = self.manager.proxies.read().await;

        proxies.iter()
            .filter(|p| p.is_healthy)
            .min_by_key(|p| p.success_count + p.failure_count)
            .cloned()
    }

    async fn select_best_performance(&self) -> Option<ProxyConfig> {
        let proxies = self.manager.proxies.read().await;

        proxies.iter()
            .filter(|p| p.is_healthy && p.success_count > 0)
            .max_by(|a, b| {
                let a_rate = a.success_count as f64 / (a.success_count + a.failure_count) as f64;
                let b_rate = b.success_count as f64 / (b.success_count + b.failure_count) as f64;
                a_rate.partial_cmp(&b_rate).unwrap()
            })
            .cloned()
    }

    async fn select_random(&self) -> Option<ProxyConfig> {
        use rand::seq::SliceRandom;

        let proxies = self.manager.proxies.read().await;
        let healthy: Vec<_> = proxies.iter()
            .filter(|p| p.is_healthy)
            .collect();

        healthy.choose(&mut rand::thread_rng())
            .map(|p| (*p).clone())
    }
}

impl Default for ProxyManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_proxy_rotation() {
        let manager = ProxyManager::new();

        manager.add_proxy(ProxyConfig {
            url: "http://proxy1.example.com:8080".to_string(),
            username: None,
            password: None,
            proxy_type: ProxyType::HTTP,
            country: Some("US".to_string()),
            is_healthy: true,
            last_used: std::time::Instant::now(),
            success_count: 0,
            failure_count: 0,
        }).await;

        manager.add_proxy(ProxyConfig {
            url: "http://proxy2.example.com:8080".to_string(),
            username: None,
            password: None,
            proxy_type: ProxyType::HTTP,
            country: Some("UK".to_string()),
            is_healthy: true,
            last_used: std::time::Instant::now(),
            success_count: 0,
            failure_count: 0,
        }).await;

        let proxy1 = manager.get_next().await;
        assert!(proxy1.is_some());

        let proxy2 = manager.get_next().await;
        assert!(proxy2.is_some());

        assert_ne!(proxy1.unwrap().url, proxy2.unwrap().url);
    }

    #[test]
    fn test_proxy_line_parsing() {
        let line = "http://user:pass@proxy.example.com:8080#US";
        let config = ProxyManager::parse_proxy_line(line);

        assert!(config.is_some());
        let config = config.unwrap();
        assert_eq!(config.username, Some("user".to_string()));
        assert_eq!(config.password, Some("pass".to_string()));
        assert_eq!(config.country, Some("US".to_string()));
    }
}