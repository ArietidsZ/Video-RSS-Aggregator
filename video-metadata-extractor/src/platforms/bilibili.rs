use async_trait::async_trait;
use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::{
    VideoMetadata, Platform, Result, ExtractorError,
    AuthorInfo, VideoStatistics, VideoQuality, VideoCategory,
    Thumbnail, ThumbnailQuality, LanguageInfo, SubtitleInfo,
    SubtitleFormat, ContentRating,
};
use super::PlatformExtractor;

/// Bilibili API extractor
pub struct BilibiliExtractor {
    client: Client,
    cookie: Option<String>,
    use_web_api: bool,
}

impl BilibiliExtractor {
    pub fn new(cookie: Option<String>) -> Self {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36".parse().unwrap());
        headers.insert("Referer", "https://www.bilibili.com".parse().unwrap());

        if let Some(cookie_value) = &cookie {
            headers.insert("Cookie", cookie_value.parse().unwrap());
        }

        let client = Client::builder()
            .default_headers(headers)
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap();

        Self {
            client,
            cookie,
            use_web_api: true,
        }
    }

    /// Extract video info using Web API
    async fn extract_via_web_api(&self, video_id: &str) -> Result<VideoMetadata> {
        // Convert BV to AV if needed
        let aid = if video_id.starts_with("BV") {
            self.bv_to_av(video_id)?
        } else if video_id.starts_with("av") {
            video_id.trim_start_matches("av").to_string()
        } else {
            video_id.to_string()
        };

        // Get video info
        let url = format!("https://api.bilibili.com/x/web-interface/view?aid={}", aid);
        let response = self.client.get(&url).send().await?;

        if !response.status().is_success() {
            return Err(ExtractorError::ApiError {
                status: response.status().as_u16(),
                message: response.text().await.unwrap_or_default(),
            });
        }

        let data: BilibiliApiResponse = response.json().await?;

        if data.code != 0 {
            return Err(ExtractorError::ApiError {
                status: data.code as u16,
                message: data.message.unwrap_or_default(),
            });
        }

        let video_data = data.data.ok_or_else(|| {
            ExtractorError::VideoNotFound(video_id.to_string())
        })?;

        self.parse_video_data(video_data).await
    }

    /// Parse Bilibili video data
    async fn parse_video_data(&self, data: BilibiliVideoData) -> Result<VideoMetadata> {
        // Get author info
        let author = AuthorInfo {
            id: data.owner.mid.to_string(),
            name: data.owner.name.clone(),
            username: Some(data.owner.name.clone()),
            url: format!("https://space.bilibili.com/{}", data.owner.mid),
            avatar_url: Some(data.owner.face.clone()),
            subscriber_count: None,
            verified: false,
            description: None,
        };

        // Parse statistics
        let mut platform_stats = HashMap::new();
        platform_stats.insert("coin".to_string(), data.stat.coin as u64);
        platform_stats.insert("danmaku".to_string(), data.stat.danmaku as u64);
        platform_stats.insert("reply".to_string(), data.stat.reply as u64);
        platform_stats.insert("share".to_string(), data.stat.share as u64);

        let stats = VideoStatistics {
            view_count: data.stat.view as u64,
            like_count: Some(data.stat.like as u64),
            dislike_count: Some(data.stat.dislike as u64),
            comment_count: Some(data.stat.reply as u64),
            share_count: Some(data.stat.share as u64),
            favorite_count: Some(data.stat.favorite as u64),
            platform_stats,
        };

        // Parse thumbnails
        let thumbnails = vec![
            Thumbnail {
                url: data.pic.clone(),
                width: 480,
                height: 360,
                quality: ThumbnailQuality::High,
            }
        ];

        // Map category
        let category = self.map_category(data.tid);

        // Parse pages (multi-part videos)
        let mut extra_metadata = HashMap::new();
        if data.pages.len() > 1 {
            extra_metadata.insert(
                "pages".to_string(),
                serde_json::to_value(&data.pages).unwrap(),
            );
        }

        // Get subtitles if available
        let subtitles = if let Some(subtitle_data) = data.subtitle {
            subtitle_data.list.into_iter().map(|s| SubtitleInfo {
                language: s.lan_doc,
                language_code: s.lan,
                auto_generated: s.ai_type == 1,
                url: Some(format!("https:{}", s.subtitle_url)),
                format: SubtitleFormat::JSON,
            }).collect()
        } else {
            vec![]
        };

        Ok(VideoMetadata {
            id: data.bvid.clone(),
            platform: Platform::Bilibili,
            title: data.title.clone(),
            description: data.desc.clone(),
            author,
            duration: data.duration as u32,
            published_at: DateTime::from_timestamp(data.pubdate, 0)
                .unwrap_or_else(|| Utc::now()),
            updated_at: DateTime::from_timestamp(data.ctime, 0)
                .unwrap_or_else(|| Utc::now()),
            statistics: stats,
            qualities: self.get_available_qualities(),
            thumbnails,
            tags: vec![data.tname.clone()],
            category,
            language: LanguageInfo {
                primary: "zh".to_string(),
                detected: vec!["zh".to_string()],
                audio_language: Some("zh".to_string()),
            },
            subtitles,
            url: format!("https://www.bilibili.com/video/{}", data.bvid),
            video_urls: HashMap::new(),
            audio_url: None,
            extra_metadata,
            content_rating: ContentRating {
                age_restricted: data.teenage_mode == 1,
                region_blocked: vec![],
                requires_login: data.is_upower_exclusive,
                is_private: false,
                is_unlisted: false,
                content_warning: data.argue_msg.clone(),
            },
            extracted_at: Utc::now(),
            cache_ttl: 3600,
        })
    }

    /// Get user videos
    pub async fn get_user_videos(&self, uid: &str, limit: usize) -> Result<Vec<VideoMetadata>> {
        let url = format!(
            "https://api.bilibili.com/x/space/arc/search?mid={}&ps={}&pn=1",
            uid, limit.min(50)
        );

        let response = self.client.get(&url).send().await?;

        if !response.status().is_success() {
            return Err(ExtractorError::ApiError {
                status: response.status().as_u16(),
                message: response.text().await.unwrap_or_default(),
            });
        }

        let data: BilibiliUserVideosResponse = response.json().await?;

        if data.code != 0 {
            return Err(ExtractorError::ApiError {
                status: data.code as u16,
                message: data.message.unwrap_or_default(),
            });
        }

        let mut videos = Vec::new();

        if let Some(list_data) = data.data {
            if let Some(vlist) = list_data.list {
                for video in vlist.vlist {
                    // Extract each video
                    if let Ok(metadata) = self.extract(&video.bvid).await {
                        videos.push(metadata);
                    }
                }
            }
        }

        Ok(videos)
    }

    /// Convert BV ID to AV ID
    fn bv_to_av(&self, bvid: &str) -> Result<String> {
        // Bilibili's BV to AV conversion algorithm
        const TABLE: &str = "fZodR9XQDSUm21yCkr6zBqiveYah8bt4xsWpHnJE7jL5VG3guMTKNPAwcF";
        const XOR: i64 = 177451812;
        const ADD: i64 = 8728348608;
        const S: [usize; 6] = [11, 10, 3, 8, 4, 6];

        let bv = bvid.trim_start_matches("BV");
        let mut r: i64 = 0;

        for (i, &pos) in S.iter().enumerate() {
            if let Some(c) = bv.chars().nth(pos - 1) {
                if let Some(idx) = TABLE.find(c) {
                    r += idx as i64 * 58_i64.pow(i as u32);
                }
            }
        }

        Ok(((r - ADD) ^ XOR).to_string())
    }

    /// Map Bilibili category ID to VideoCategory
    fn map_category(&self, tid: i64) -> VideoCategory {
        match tid {
            1 | 24 | 25 | 47 | 210 | 86 | 253 => VideoCategory::Animation,
            3 | 28 | 31 | 30 | 194 | 59 | 193 | 29 | 130 => VideoCategory::Music,
            129 | 20 | 198 | 199 | 200 | 154 | 156 => VideoCategory::Entertainment,
            36 | 202 | 124 | 228 | 207 | 208 | 209 | 122 | 201 => VideoCategory::Technology,
            160 | 138 | 250 | 251 | 239 | 161 | 248 | 240 | 164 => VideoCategory::Lifestyle,
            17 | 171 | 172 | 65 | 173 | 121 | 136 | 19 => VideoCategory::Gaming,
            181 | 182 | 183 | 85 | 184 | 86 => VideoCategory::Film,
            234 | 235 | 249 | 237 => VideoCategory::Sports,
            223 | 245 | 246 | 247 | 248 | 240 | 227 => VideoCategory::Vlog,
            211 | 212 | 213 | 214 | 215 => VideoCategory::Food,
            155 | 157 | 252 | 158 | 159 => VideoCategory::Fashion,
            202 | 203 | 204 | 205 | 206 => VideoCategory::News,
            _ => VideoCategory::Other(format!("tid_{}", tid)),
        }
    }

    /// Get typical Bilibili quality options
    fn get_available_qualities(&self) -> Vec<VideoQuality> {
        vec![
            VideoQuality::LQ360p,
            VideoQuality::SD480p,
            VideoQuality::HD720p,
            VideoQuality::FHD1080p,
            VideoQuality::UHD4K,
            VideoQuality::AudioOnly,
        ]
    }
}

#[async_trait]
impl PlatformExtractor for BilibiliExtractor {
    async fn extract(&self, video_id: &str) -> Result<VideoMetadata> {
        self.extract_via_web_api(video_id).await
    }

    async fn extract_batch(&self, video_ids: Vec<String>) -> Result<Vec<VideoMetadata>> {
        // Bilibili doesn't have a batch API, so we extract sequentially
        let mut results = Vec::new();

        for id in video_ids {
            match self.extract(&id).await {
                Ok(metadata) => results.push(metadata),
                Err(e) => {
                    tracing::warn!("Failed to extract {}: {}", id, e);
                }
            }
        }

        Ok(results)
    }

    fn platform(&self) -> Platform {
        Platform::Bilibili
    }

    fn is_authenticated(&self) -> bool {
        self.cookie.is_some()
    }
}

// Bilibili API response structures
#[derive(Debug, Deserialize)]
struct BilibiliApiResponse {
    code: i32,
    message: Option<String>,
    data: Option<BilibiliVideoData>,
}

#[derive(Debug, Deserialize)]
struct BilibiliVideoData {
    bvid: String,
    aid: i64,
    pic: String,
    title: String,
    pubdate: i64,
    ctime: i64,
    desc: String,
    duration: i64,
    owner: BilibiliOwner,
    stat: BilibiliStat,
    tid: i64,
    tname: String,
    pages: Vec<BilibiliPage>,
    teenage_mode: i32,
    is_upower_exclusive: bool,
    argue_msg: Option<String>,
    subtitle: Option<BilibiliSubtitle>,
}

#[derive(Debug, Deserialize)]
struct BilibiliOwner {
    mid: i64,
    name: String,
    face: String,
}

#[derive(Debug, Deserialize)]
struct BilibiliStat {
    view: i64,
    danmaku: i64,
    reply: i64,
    favorite: i64,
    coin: i64,
    share: i64,
    like: i64,
    dislike: i64,
}

#[derive(Debug, Deserialize, Serialize)]
struct BilibiliPage {
    cid: i64,
    page: i32,
    part: String,
    duration: i64,
}

#[derive(Debug, Deserialize)]
struct BilibiliSubtitle {
    list: Vec<BilibiliSubtitleItem>,
}

#[derive(Debug, Deserialize)]
struct BilibiliSubtitleItem {
    lan: String,
    lan_doc: String,
    subtitle_url: String,
    ai_type: i32,
}

#[derive(Debug, Deserialize)]
struct BilibiliUserVideosResponse {
    code: i32,
    message: Option<String>,
    data: Option<BilibiliUserVideosData>,
}

#[derive(Debug, Deserialize)]
struct BilibiliUserVideosData {
    list: Option<BilibiliUserVideosList>,
}

#[derive(Debug, Deserialize)]
struct BilibiliUserVideosList {
    vlist: Vec<BilibiliUserVideo>,
}

#[derive(Debug, Deserialize)]
struct BilibiliUserVideo {
    bvid: String,
    aid: i64,
    title: String,
    pic: String,
    play: i64,
    created: i64,
}