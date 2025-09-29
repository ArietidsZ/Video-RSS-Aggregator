use thiserror::Error;

#[derive(Error, Debug)]
pub enum ExtractorError {
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("Parsing error: {0}")]
    Parsing(String),

    #[error("Invalid URL: {0}")]
    InvalidUrl(String),

    #[error("Platform not supported: {0}")]
    UnsupportedPlatform(String),

    #[error("Rate limit exceeded")]
    RateLimited,

    #[error("Authentication required")]
    AuthRequired,

    #[error("Video not found: {0}")]
    VideoNotFound(String),

    #[error("Access denied: {0}")]
    AccessDenied(String),

    #[error("Cache error: {0}")]
    Cache(#[from] redis::RedisError),

    #[error("Timeout after {0} seconds")]
    Timeout(u64),

    #[error("Proxy error: {0}")]
    ProxyError(String),

    #[error("API error: {status}: {message}")]
    ApiError { status: u16, message: String },

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Other error: {0}")]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, ExtractorError>;