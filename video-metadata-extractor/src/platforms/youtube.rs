use async_trait::async_trait;
use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::{
    VideoMetadata, Platform, Result, ExtractorError,
    AuthorInfo, VideoStatistics, VideoQuality, VideoCategory,
    Thumbnail, ThumbnailQuality, LanguageInfo, SubtitleInfo,
    SubtitleFormat, ContentRating, models,
};
use super::PlatformExtractor;

/// YouTube Data API v3 extractor
pub struct YouTubeExtractor {
    client: Client,
    api_key: Option<String>,
    use_invidious: bool,
    invidious_instance: String,
}

impl YouTubeExtractor {
    pub fn new(api_key: Option<String>) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap();

        Self {
            client,
            api_key,
            use_invidious: false,
            invidious_instance: "https://invidious.io".to_string(),
        }
    }

    /// Enable Invidious as fallback
    pub fn with_invidious(mut self, instance: &str) -> Self {
        self.use_invidious = true;
        self.invidious_instance = instance.to_string();
        self
    }

    /// Extract using YouTube Data API v3
    async fn extract_via_api(&self, video_id: &str) -> Result<VideoMetadata> {
        let api_key = self.api_key.as_ref()
            .ok_or(ExtractorError::AuthRequired)?;

        let url = format!(
            "https://www.googleapis.com/youtube/v3/videos?id={}&key={}&part=snippet,contentDetails,statistics,status",
            video_id, api_key
        );

        let response = self.client.get(&url)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(ExtractorError::ApiError {
                status: response.status().as_u16(),
                message: response.text().await.unwrap_or_default(),
            });
        }

        let data: YouTubeApiResponse = response.json().await?;

        if data.items.is_empty() {
            return Err(ExtractorError::VideoNotFound(video_id.to_string()));
        }

        let item = &data.items[0];
        self.parse_api_response(item).await
    }

    /// Parse YouTube API response
    async fn parse_api_response(&self, item: &YouTubeVideoItem) -> Result<VideoMetadata> {
        let snippet = &item.snippet;
        let statistics = &item.statistics;
        let content_details = &item.content_details;
        let status = &item.status;

        // Parse duration
        let duration = self.parse_duration(&content_details.duration);

        // Parse thumbnails
        let mut thumbnails = Vec::new();
        if let Some(t) = &snippet.thumbnails.default {
            thumbnails.push(Thumbnail {
                url: t.url.clone(),
                width: t.width.unwrap_or(120),
                height: t.height.unwrap_or(90),
                quality: ThumbnailQuality::Default,
            });
        }
        if let Some(t) = &snippet.thumbnails.medium {
            thumbnails.push(Thumbnail {
                url: t.url.clone(),
                width: t.width.unwrap_or(320),
                height: t.height.unwrap_or(180),
                quality: ThumbnailQuality::Medium,
            });
        }
        if let Some(t) = &snippet.thumbnails.high {
            thumbnails.push(Thumbnail {
                url: t.url.clone(),
                width: t.width.unwrap_or(480),
                height: t.height.unwrap_or(360),
                quality: ThumbnailQuality::High,
            });
        }

        // Get channel info
        let author = AuthorInfo {
            id: snippet.channel_id.clone(),
            name: snippet.channel_title.clone(),
            username: None,
            url: format!("https://www.youtube.com/channel/{}", snippet.channel_id),
            avatar_url: None,
            subscriber_count: None,
            verified: false,
            description: None,
        };

        // Parse statistics
        let stats = VideoStatistics {
            view_count: statistics.view_count.parse().unwrap_or(0),
            like_count: statistics.like_count.as_ref().and_then(|s| s.parse().ok()),
            dislike_count: None, // YouTube removed public dislike counts
            comment_count: statistics.comment_count.as_ref().and_then(|s| s.parse().ok()),
            share_count: None,
            favorite_count: statistics.favorite_count.as_ref().and_then(|s| s.parse().ok()),
            platform_stats: HashMap::new(),
        };

        // Parse category
        let category = self.map_category(snippet.category_id.as_ref().and_then(|c| c.parse().ok()));

        // Parse content rating
        let content_rating = ContentRating {
            age_restricted: content_details.content_rating
                .as_ref()
                .map(|r| r.yt_rating.as_ref() == Some(&"ytAgeRestricted".to_string()))
                .unwrap_or(false),
            region_blocked: content_details.region_restriction
                .as_ref()
                .and_then(|r| r.blocked.clone())
                .unwrap_or_default(),
            requires_login: status.privacy_status == "private",
            is_private: status.privacy_status == "private",
            is_unlisted: status.privacy_status == "unlisted",
            content_warning: None,
        };

        // Build metadata
        Ok(VideoMetadata {
            id: item.id.clone(),
            platform: Platform::YouTube,
            title: snippet.title.clone(),
            description: snippet.description.clone(),
            author,
            duration,
            published_at: DateTime::parse_from_rfc3339(&snippet.published_at)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            updated_at: Utc::now(),
            statistics: stats,
            qualities: self.get_available_qualities(),
            thumbnails,
            tags: snippet.tags.clone().unwrap_or_default(),
            category,
            language: LanguageInfo {
                primary: snippet.default_language.clone().unwrap_or_else(|| "en".to_string()),
                detected: vec![],
                audio_language: snippet.default_audio_language.clone(),
            },
            subtitles: vec![], // Would need separate API call
            url: format!("https://www.youtube.com/watch?v={}", item.id),
            video_urls: HashMap::new(),
            audio_url: None,
            extra_metadata: HashMap::new(),
            content_rating,
            extracted_at: Utc::now(),
            cache_ttl: 3600,
        })
    }

    /// Extract using Invidious API (privacy-friendly)
    async fn extract_via_invidious(&self, video_id: &str) -> Result<VideoMetadata> {
        let url = format!("{}/api/v1/videos/{}", self.invidious_instance, video_id);

        let response = self.client.get(&url)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(ExtractorError::ApiError {
                status: response.status().as_u16(),
                message: response.text().await.unwrap_or_default(),
            });
        }

        let data: InvidiousResponse = response.json().await?;
        self.parse_invidious_response(data)
    }

    /// Parse Invidious response
    fn parse_invidious_response(&self, data: InvidiousResponse) -> Result<VideoMetadata> {
        let author = AuthorInfo {
            id: data.author_id.clone(),
            name: data.author.clone(),
            username: None,
            url: data.author_url.clone(),
            avatar_url: data.author_thumbnails.first().map(|t| t.url.clone()),
            subscriber_count: Some(data.sub_count_text.parse().unwrap_or(0)),
            verified: data.author_verified,
            description: None,
        };

        let stats = VideoStatistics {
            view_count: data.view_count,
            like_count: Some(data.like_count),
            dislike_count: Some(data.dislike_count),
            comment_count: None,
            share_count: None,
            favorite_count: None,
            platform_stats: HashMap::new(),
        };

        let thumbnails = data.video_thumbnails.into_iter().map(|t| Thumbnail {
            url: t.url,
            width: t.width,
            height: t.height,
            quality: match t.quality.as_str() {
                "default" => ThumbnailQuality::Default,
                "medium" => ThumbnailQuality::Medium,
                "high" => ThumbnailQuality::High,
                "sddefault" => ThumbnailQuality::Standard,
                "maxresdefault" => ThumbnailQuality::MaxRes,
                _ => ThumbnailQuality::Default,
            },
        }).collect();

        let subtitles = data.captions.into_iter().map(|c| SubtitleInfo {
            language: c.label,
            language_code: c.language_code,
            auto_generated: false,
            url: Some(c.url),
            format: SubtitleFormat::VTT,
        }).collect();

        Ok(VideoMetadata {
            id: data.video_id,
            platform: Platform::YouTube,
            title: data.title,
            description: data.description,
            author,
            duration: data.length_seconds as u32,
            published_at: DateTime::from_timestamp(data.published, 0).unwrap_or_else(|| Utc::now()),
            updated_at: Utc::now(),
            statistics: stats,
            qualities: self.get_available_qualities(),
            thumbnails,
            tags: data.keywords,
            category: VideoCategory::Other(data.genre.unwrap_or_default()),
            language: LanguageInfo {
                primary: "en".to_string(),
                detected: vec![],
                audio_language: None,
            },
            subtitles,
            url: format!("https://www.youtube.com/watch?v={}", data.video_id),
            video_urls: HashMap::new(),
            audio_url: None,
            extra_metadata: HashMap::new(),
            content_rating: ContentRating {
                age_restricted: data.is_family_friendly == false,
                region_blocked: vec![],
                requires_login: false,
                is_private: false,
                is_unlisted: data.is_listed == false,
                content_warning: None,
            },
            extracted_at: Utc::now(),
            cache_ttl: 3600,
        })
    }

    /// Parse ISO 8601 duration to seconds
    fn parse_duration(&self, duration: &str) -> u32 {
        let re = regex::Regex::new(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?").unwrap();

        if let Some(captures) = re.captures(duration) {
            let hours = captures.get(1).and_then(|m| m.as_str().parse::<u32>().ok()).unwrap_or(0);
            let minutes = captures.get(2).and_then(|m| m.as_str().parse::<u32>().ok()).unwrap_or(0);
            let seconds = captures.get(3).and_then(|m| m.as_str().parse::<u32>().ok()).unwrap_or(0);

            hours * 3600 + minutes * 60 + seconds
        } else {
            0
        }
    }

    /// Map YouTube category ID to VideoCategory
    fn map_category(&self, category_id: Option<u32>) -> VideoCategory {
        match category_id {
            Some(1) => VideoCategory::Film,
            Some(2) => VideoCategory::Animation,
            Some(10) => VideoCategory::Music,
            Some(17) => VideoCategory::Sports,
            Some(19) => VideoCategory::Travel,
            Some(20) => VideoCategory::Gaming,
            Some(22) => VideoCategory::Vlog,
            Some(23) => VideoCategory::Comedy,
            Some(24) => VideoCategory::Entertainment,
            Some(25) => VideoCategory::News,
            Some(26) => VideoCategory::HowTo,
            Some(27) => VideoCategory::Education,
            Some(28) => VideoCategory::Science,
            _ => VideoCategory::Other("Unknown".to_string()),
        }
    }

    /// Get typical YouTube quality options
    fn get_available_qualities(&self) -> Vec<VideoQuality> {
        vec![
            VideoQuality::LQ144p,
            VideoQuality::MQ240p,
            VideoQuality::LQ360p,
            VideoQuality::SD480p,
            VideoQuality::HD720p,
            VideoQuality::FHD1080p,
            VideoQuality::QHD2K,
            VideoQuality::UHD4K,
            VideoQuality::AudioOnly,
        ]
    }
}

#[async_trait]
impl PlatformExtractor for YouTubeExtractor {
    async fn extract(&self, video_id: &str) -> Result<VideoMetadata> {
        // Try API first if available
        if self.api_key.is_some() && !self.use_invidious {
            match self.extract_via_api(video_id).await {
                Ok(metadata) => return Ok(metadata),
                Err(e) => {
                    tracing::warn!("YouTube API failed: {}, trying Invidious", e);
                }
            }
        }

        // Fallback to Invidious
        if self.use_invidious {
            self.extract_via_invidious(video_id).await
        } else {
            Err(ExtractorError::AuthRequired)
        }
    }

    async fn extract_batch(&self, video_ids: Vec<String>) -> Result<Vec<VideoMetadata>> {
        if video_ids.is_empty() {
            return Ok(vec![]);
        }

        // YouTube API supports batch requests (up to 50 IDs)
        if let Some(api_key) = &self.api_key {
            let chunks: Vec<Vec<String>> = video_ids.chunks(50)
                .map(|chunk| chunk.to_vec())
                .collect();

            let mut all_metadata = Vec::new();

            for chunk in chunks {
                let ids = chunk.join(",");
                let url = format!(
                    "https://www.googleapis.com/youtube/v3/videos?id={}&key={}&part=snippet,contentDetails,statistics,status",
                    ids, api_key
                );

                let response = self.client.get(&url).send().await?;

                if response.status().is_success() {
                    let data: YouTubeApiResponse = response.json().await?;

                    for item in data.items {
                        if let Ok(metadata) = self.parse_api_response(&item).await {
                            all_metadata.push(metadata);
                        }
                    }
                }
            }

            Ok(all_metadata)
        } else {
            // Fallback to sequential extraction
            let mut results = Vec::new();
            for id in video_ids {
                if let Ok(metadata) = self.extract(&id).await {
                    results.push(metadata);
                }
            }
            Ok(results)
        }
    }

    fn platform(&self) -> Platform {
        Platform::YouTube
    }

    fn is_authenticated(&self) -> bool {
        self.api_key.is_some()
    }
}

// YouTube API response structures
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct YouTubeApiResponse {
    items: Vec<YouTubeVideoItem>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct YouTubeVideoItem {
    id: String,
    snippet: YouTubeSnippet,
    statistics: YouTubeStatistics,
    content_details: YouTubeContentDetails,
    status: YouTubeStatus,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct YouTubeSnippet {
    title: String,
    description: String,
    published_at: String,
    channel_id: String,
    channel_title: String,
    thumbnails: YouTubeThumbnails,
    tags: Option<Vec<String>>,
    category_id: Option<String>,
    default_language: Option<String>,
    default_audio_language: Option<String>,
}

#[derive(Debug, Deserialize)]
struct YouTubeThumbnails {
    default: Option<YouTubeThumbnail>,
    medium: Option<YouTubeThumbnail>,
    high: Option<YouTubeThumbnail>,
    standard: Option<YouTubeThumbnail>,
    maxres: Option<YouTubeThumbnail>,
}

#[derive(Debug, Deserialize)]
struct YouTubeThumbnail {
    url: String,
    width: Option<u32>,
    height: Option<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct YouTubeStatistics {
    view_count: String,
    like_count: Option<String>,
    dislike_count: Option<String>,
    comment_count: Option<String>,
    favorite_count: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct YouTubeContentDetails {
    duration: String,
    dimension: Option<String>,
    definition: Option<String>,
    caption: Option<String>,
    licensed_content: Option<bool>,
    content_rating: Option<YouTubeContentRating>,
    region_restriction: Option<YouTubeRegionRestriction>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct YouTubeContentRating {
    yt_rating: Option<String>,
}

#[derive(Debug, Deserialize)]
struct YouTubeRegionRestriction {
    allowed: Option<Vec<String>>,
    blocked: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct YouTubeStatus {
    privacy_status: String,
    embeddable: Option<bool>,
    public_stats_viewable: Option<bool>,
}

// Invidious API structures
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct InvidiousResponse {
    video_id: String,
    title: String,
    description: String,
    author: String,
    author_id: String,
    author_url: String,
    author_verified: bool,
    author_thumbnails: Vec<InvidiousThumbnail>,
    sub_count_text: String,
    length_seconds: i64,
    view_count: u64,
    like_count: u64,
    dislike_count: u64,
    published: i64,
    keywords: Vec<String>,
    video_thumbnails: Vec<InvidiousThumbnail>,
    captions: Vec<InvidiousCaption>,
    is_listed: bool,
    is_family_friendly: bool,
    genre: Option<String>,
}

#[derive(Debug, Deserialize)]
struct InvidiousThumbnail {
    url: String,
    width: u32,
    height: u32,
    quality: String,
}

#[derive(Debug, Deserialize)]
struct InvidiousCaption {
    label: String,
    language_code: String,
    url: String,
}