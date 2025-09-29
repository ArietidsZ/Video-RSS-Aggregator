pub mod youtube;
pub mod bilibili;
pub mod douyin;
pub mod tiktok;
pub mod kuaishou;

use async_trait::async_trait;
use crate::{VideoMetadata, Platform, Result};

/// Common trait for all platform extractors
#[async_trait]
pub trait PlatformExtractor: Send + Sync {
    /// Extract metadata for a single video
    async fn extract(&self, video_id: &str) -> Result<VideoMetadata>;

    /// Extract metadata for multiple videos (batch)
    async fn extract_batch(&self, video_ids: Vec<String>) -> Result<Vec<VideoMetadata>>;

    /// Get platform identifier
    fn platform(&self) -> Platform;

    /// Check if authentication is configured
    fn is_authenticated(&self) -> bool;
}