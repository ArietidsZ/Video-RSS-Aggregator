pub mod extractor;
pub mod platforms;
pub mod cache;
pub mod rate_limiter;
pub mod url_parser;
pub mod proxy_manager;
pub mod error;
pub mod models;

pub use extractor::VideoMetadataExtractor;
pub use models::{VideoMetadata, Platform, VideoQuality, VideoCategory};
pub use error::{ExtractorError, Result};

/// High-performance video metadata extraction library
///
/// Features:
/// - Platform-specific API integrations
/// - Intelligent URL processing and validation
/// - Redis caching with configurable TTL
/// - Rate limiting and proxy rotation
/// - Content type detection and filtering
///
/// Target Performance: <2 seconds per video URL
pub fn initialize() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("video_metadata_extractor=debug")
        .init();

    Ok(())
}