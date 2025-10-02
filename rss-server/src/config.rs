use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub base_url: String,
    pub websub_secret: Option<String>,
    pub database_url: String,
    pub redis_url: String,
    pub port: u16,

    // Platform API configurations
    pub youtube_api_key: Option<String>,
    pub bilibili_cookies: Option<String>,

    // Feature flags
    pub enable_websub: bool,
    pub enable_qrcode: bool,
    pub enable_concurrent_fetching: bool,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:8010".to_string(),
            websub_secret: Some("change_this_secret_in_production".to_string()),
            database_url: "postgresql://videorss:password@localhost/videorss".to_string(),
            redis_url: "redis://localhost:6379".to_string(),
            port: 8010,
            youtube_api_key: None,
            bilibili_cookies: None,
            enable_websub: true,
            enable_qrcode: true,
            enable_concurrent_fetching: true,
        }
    }
}

impl AppConfig {
    pub fn from_env() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            base_url: std::env::var("BASE_URL")
                .unwrap_or_else(|_| "http://localhost:8010".to_string()),
            websub_secret: std::env::var("WEBSUB_SECRET").ok(),
            database_url: std::env::var("DATABASE_URL")
                .unwrap_or_else(|_| "postgresql://videorss:password@localhost/videorss".to_string()),
            redis_url: std::env::var("REDIS_URL")
                .unwrap_or_else(|_| "redis://localhost:6379".to_string()),
            port: std::env::var("PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(8010),
            youtube_api_key: std::env::var("YOUTUBE_API_KEY").ok(),
            bilibili_cookies: std::env::var("BILIBILI_COOKIES").ok(),
            enable_websub: std::env::var("ENABLE_WEBSUB")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(true),
            enable_qrcode: std::env::var("ENABLE_QRCODE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(true),
            enable_concurrent_fetching: std::env::var("ENABLE_CONCURRENT_FETCHING")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(true),
        })
    }
}