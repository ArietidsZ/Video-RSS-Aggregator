use async_trait::async_trait;
use crate::{VideoMetadata, Platform, Result};
use super::{PlatformExtractor, douyin::DouyinExtractor};

/// TikTok extractor (similar to Douyin)
pub struct TikTokExtractor {
    douyin_extractor: DouyinExtractor,
}

impl TikTokExtractor {
    pub fn new() -> Self {
        Self {
            douyin_extractor: DouyinExtractor::new(),
        }
    }
}

#[async_trait]
impl PlatformExtractor for TikTokExtractor {
    async fn extract(&self, video_id: &str) -> Result<VideoMetadata> {
        let mut metadata = self.douyin_extractor.extract(video_id).await?;
        metadata.platform = Platform::TikTok;
        metadata.url = format!("https://www.tiktok.com/@user/video/{}", video_id);
        Ok(metadata)
    }

    async fn extract_batch(&self, video_ids: Vec<String>) -> Result<Vec<VideoMetadata>> {
        let mut results = self.douyin_extractor.extract_batch(video_ids).await?;
        for metadata in &mut results {
            metadata.platform = Platform::TikTok;
        }
        Ok(results)
    }

    fn platform(&self) -> Platform {
        Platform::TikTok
    }

    fn is_authenticated(&self) -> bool {
        false
    }
}