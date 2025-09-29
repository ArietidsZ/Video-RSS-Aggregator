use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Comprehensive video metadata structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoMetadata {
    /// Unique video identifier
    pub id: String,

    /// Platform where the video is hosted
    pub platform: Platform,

    /// Video title
    pub title: String,

    /// Video description
    pub description: String,

    /// Video author/channel information
    pub author: AuthorInfo,

    /// Video duration in seconds
    pub duration: u32,

    /// Video upload/publish date
    pub published_at: DateTime<Utc>,

    /// Last updated timestamp
    pub updated_at: DateTime<Utc>,

    /// View statistics
    pub statistics: VideoStatistics,

    /// Available video qualities
    pub qualities: Vec<VideoQuality>,

    /// Thumbnail URLs at different resolutions
    pub thumbnails: Vec<Thumbnail>,

    /// Video tags/keywords
    pub tags: Vec<String>,

    /// Video category
    pub category: VideoCategory,

    /// Language information
    pub language: LanguageInfo,

    /// Available subtitles/captions
    pub subtitles: Vec<SubtitleInfo>,

    /// Video URL
    pub url: String,

    /// Direct video file URLs (if available)
    pub video_urls: HashMap<VideoQuality, String>,

    /// Audio-only URL (if available)
    pub audio_url: Option<String>,

    /// Additional platform-specific metadata
    pub extra_metadata: HashMap<String, serde_json::Value>,

    /// Content rating/restrictions
    pub content_rating: ContentRating,

    /// Extraction timestamp
    pub extracted_at: DateTime<Utc>,

    /// Cache TTL in seconds
    pub cache_ttl: u64,
}

/// Video hosting platform
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Platform {
    YouTube,
    Bilibili,
    Douyin,
    TikTok,
    Kuaishou,
    Vimeo,
    Dailymotion,
    Twitch,
    Instagram,
    Twitter,
    Reddit,
    Unknown(String),
}

impl Platform {
    pub fn from_url(url: &str) -> Self {
        if url.contains("youtube.com") || url.contains("youtu.be") {
            Platform::YouTube
        } else if url.contains("bilibili.com") || url.contains("b23.tv") {
            Platform::Bilibili
        } else if url.contains("douyin.com") {
            Platform::Douyin
        } else if url.contains("tiktok.com") {
            Platform::TikTok
        } else if url.contains("kuaishou.com") || url.contains("kwai.com") {
            Platform::Kuaishou
        } else if url.contains("vimeo.com") {
            Platform::Vimeo
        } else if url.contains("dailymotion.com") {
            Platform::Dailymotion
        } else if url.contains("twitch.tv") {
            Platform::Twitch
        } else if url.contains("instagram.com") {
            Platform::Instagram
        } else if url.contains("twitter.com") || url.contains("x.com") {
            Platform::Twitter
        } else if url.contains("reddit.com") {
            Platform::Reddit
        } else {
            Platform::Unknown(url.to_string())
        }
    }

    pub fn api_endpoint(&self) -> Option<&str> {
        match self {
            Platform::YouTube => Some("https://www.googleapis.com/youtube/v3"),
            Platform::Bilibili => Some("https://api.bilibili.com"),
            Platform::Douyin => Some("https://www.iesdouyin.com/web/api/v2"),
            Platform::TikTok => Some("https://api.tiktok.com"),
            _ => None,
        }
    }
}

/// Video quality/resolution
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum VideoQuality {
    /// 8K resolution (7680x4320)
    UHD8K,
    /// 4K resolution (3840x2160)
    UHD4K,
    /// 2K resolution (2560x1440)
    QHD2K,
    /// Full HD (1920x1080)
    FHD1080p,
    /// HD (1280x720)
    HD720p,
    /// Standard Definition (854x480)
    SD480p,
    /// Low Quality (640x360)
    LQ360p,
    /// Mobile Quality (426x240)
    MQ240p,
    /// Lowest Quality (256x144)
    LQ144p,
    /// Audio only
    AudioOnly,
    /// Custom resolution
    Custom(u32, u32),
}

impl VideoQuality {
    pub fn bitrate_estimate(&self) -> u32 {
        match self {
            VideoQuality::UHD8K => 50_000_000,      // 50 Mbps
            VideoQuality::UHD4K => 25_000_000,      // 25 Mbps
            VideoQuality::QHD2K => 16_000_000,      // 16 Mbps
            VideoQuality::FHD1080p => 8_000_000,    // 8 Mbps
            VideoQuality::HD720p => 5_000_000,      // 5 Mbps
            VideoQuality::SD480p => 2_500_000,      // 2.5 Mbps
            VideoQuality::LQ360p => 1_000_000,      // 1 Mbps
            VideoQuality::MQ240p => 500_000,        // 500 Kbps
            VideoQuality::LQ144p => 250_000,        // 250 Kbps
            VideoQuality::AudioOnly => 128_000,     // 128 Kbps
            VideoQuality::Custom(w, h) => (w * h * 30 * 12) / 1000, // Rough estimate
        }
    }
}

/// Video category/genre
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum VideoCategory {
    Education,
    Entertainment,
    Gaming,
    Music,
    News,
    Sports,
    Technology,
    Science,
    HowTo,
    Travel,
    Food,
    Fashion,
    Comedy,
    Documentary,
    Animation,
    Film,
    Lifestyle,
    Vlog,
    Tutorial,
    Review,
    Podcast,
    LiveStream,
    Short,
    Other(String),
}

/// Author/Channel information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorInfo {
    pub id: String,
    pub name: String,
    pub username: Option<String>,
    pub url: String,
    pub avatar_url: Option<String>,
    pub subscriber_count: Option<u64>,
    pub verified: bool,
    pub description: Option<String>,
}

/// Video statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoStatistics {
    pub view_count: u64,
    pub like_count: Option<u64>,
    pub dislike_count: Option<u64>,
    pub comment_count: Option<u64>,
    pub share_count: Option<u64>,
    pub favorite_count: Option<u64>,

    /// Platform-specific statistics
    pub platform_stats: HashMap<String, u64>,
}

/// Thumbnail information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thumbnail {
    pub url: String,
    pub width: u32,
    pub height: u32,
    pub quality: ThumbnailQuality,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThumbnailQuality {
    Default,
    Medium,
    High,
    Standard,
    MaxRes,
}

/// Language information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageInfo {
    pub primary: String,
    pub detected: Vec<String>,
    pub audio_language: Option<String>,
}

/// Subtitle/Caption information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtitleInfo {
    pub language: String,
    pub language_code: String,
    pub auto_generated: bool,
    pub url: Option<String>,
    pub format: SubtitleFormat,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SubtitleFormat {
    VTT,
    SRT,
    ASS,
    TTML,
    JSON,
}

/// Content rating and restrictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentRating {
    pub age_restricted: bool,
    pub region_blocked: Vec<String>,
    pub requires_login: bool,
    pub is_private: bool,
    pub is_unlisted: bool,
    pub content_warning: Option<String>,
}

/// Extraction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionConfig {
    /// Include video file URLs
    pub extract_video_urls: bool,

    /// Include subtitle URLs
    pub extract_subtitles: bool,

    /// Maximum qualities to extract
    pub max_quality: VideoQuality,

    /// Use authentication if available
    pub use_auth: bool,

    /// Proxy configuration
    pub proxy: Option<String>,

    /// Request timeout in seconds
    pub timeout: u64,

    /// User agent string
    pub user_agent: String,

    /// Additional headers
    pub headers: HashMap<String, String>,

    /// Cache TTL in seconds (0 to disable caching)
    pub cache_ttl: u64,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            extract_video_urls: false,
            extract_subtitles: true,
            max_quality: VideoQuality::FHD1080p,
            use_auth: false,
            proxy: None,
            timeout: 10,
            user_agent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36".to_string(),
            headers: HashMap::new(),
            cache_ttl: 3600, // 1 hour default
        }
    }
}

/// Batch extraction request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchExtractionRequest {
    pub urls: Vec<String>,
    pub config: ExtractionConfig,
    pub parallel_limit: usize,
}

/// Batch extraction response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchExtractionResponse {
    pub successful: Vec<VideoMetadata>,
    pub failed: Vec<ExtractionFailure>,
    pub total_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionFailure {
    pub url: String,
    pub error: String,
    pub timestamp: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_detection() {
        assert_eq!(Platform::from_url("https://www.youtube.com/watch?v=123"), Platform::YouTube);
        assert_eq!(Platform::from_url("https://www.bilibili.com/video/BV123"), Platform::Bilibili);
        assert_eq!(Platform::from_url("https://www.douyin.com/video/123"), Platform::Douyin);
    }

    #[test]
    fn test_quality_bitrate() {
        assert_eq!(VideoQuality::FHD1080p.bitrate_estimate(), 8_000_000);
        assert_eq!(VideoQuality::HD720p.bitrate_estimate(), 5_000_000);
        assert_eq!(VideoQuality::AudioOnly.bitrate_estimate(), 128_000);
    }
}