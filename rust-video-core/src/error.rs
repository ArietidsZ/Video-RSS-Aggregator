use thiserror::Error;

#[derive(Error, Debug)]
pub enum VideoRssError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON parsing failed: {0}")]
    Json(#[from] serde_json::Error),

    #[error("RSS generation failed: {0}")]
    Rss(#[from] rss::Error),

    #[error("URL parsing failed: {0}")]
    Url(#[from] url::ParseError),

    #[error("Regex error: {0}")]
    Regex(#[from] regex::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Video not found: {0}")]
    VideoNotFound(String),

    #[error("Invalid credentials")]
    InvalidCredentials,

    #[error("Rate limit exceeded")]
    RateLimit,

    #[error("Timeout occurred")]
    Timeout,

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Content analysis failed: {0}")]
    ContentAnalysis(String),

    #[error("Unknown error: {0}")]
    Unknown(String),
}