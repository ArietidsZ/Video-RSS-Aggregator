use anyhow::{anyhow, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use regex::Regex;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformConfig {
    pub bilibili: BilibiliConfig,
    pub douyin: DouyinConfig,
    pub kuaishou: KuaishouConfig,
    pub youtube: YouTubeConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BilibiliConfig {
    pub include_danmaku: bool,
    pub include_stats: bool,
    pub quality_preference: VideoQuality,
    pub subtitle_lang: String,
    pub cookie: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DouyinConfig {
    pub include_comments: bool,
    pub include_music: bool,
    pub watermark_removal: bool,
    pub quality_preference: VideoQuality,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KuaishouConfig {
    pub include_gifts: bool,
    pub include_location: bool,
    pub quality_preference: VideoQuality,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YouTubeConfig {
    pub include_captions: bool,
    pub include_chapters: bool,
    pub quality_preference: VideoQuality,
    pub api_key: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VideoQuality {
    Auto,
    Highest,
    HD1080p,
    HD720p,
    SD480p,
    SD360p,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoMetadata {
    pub id: String,
    pub title: String,
    pub author: String,
    pub author_id: String,
    pub description: String,
    pub duration: u32,
    pub views: u64,
    pub likes: u64,
    pub comments: u64,
    pub published_at: DateTime<Utc>,
    pub thumbnail_url: String,
    pub video_url: String,
    pub tags: Vec<String>,
    pub platform: Platform,
    pub extra_data: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Platform {
    Bilibili,
    Douyin,
    Kuaishou,
    YouTube,
}

#[async_trait]
pub trait PlatformAdapter: Send + Sync {
    async fn fetch_video_metadata(&self, video_id: &str) -> Result<VideoMetadata>;
    async fn fetch_channel_videos(&self, channel_id: &str, limit: usize) -> Result<Vec<VideoMetadata>>;
    async fn search_videos(&self, query: &str, limit: usize) -> Result<Vec<VideoMetadata>>;
    async fn get_trending_videos(&self, limit: usize) -> Result<Vec<VideoMetadata>>;
    fn customize_rss_item(&self, metadata: &VideoMetadata) -> Result<String>;
    fn get_platform(&self) -> Platform;
}

pub struct BilibiliAdapter {
    config: BilibiliConfig,
    client: reqwest::Client,
    cache: Arc<RwLock<HashMap<String, (VideoMetadata, std::time::Instant)>>>,
}

impl BilibiliAdapter {
    pub fn new(config: BilibiliConfig) -> Self {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36".parse().unwrap());
        headers.insert("Referer", "https://www.bilibili.com".parse().unwrap());

        if let Some(cookie) = &config.cookie {
            headers.insert("Cookie", cookie.parse().unwrap());
        }

        let client = reqwest::Client::builder()
            .default_headers(headers)
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .unwrap();

        Self {
            config,
            client,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    async fn parse_bilibili_response(&self, json: serde_json::Value) -> Result<VideoMetadata> {
        let data = json.get("data").ok_or(anyhow!("Missing data field"))?;

        let mut extra_data = HashMap::new();

        if self.config.include_danmaku {
            if let Some(danmaku_count) = data.get("stat").and_then(|s| s.get("danmaku")) {
                extra_data.insert("danmaku_count".to_string(), danmaku_count.clone());
            }
        }

        if self.config.include_stats {
            if let Some(stat) = data.get("stat") {
                extra_data.insert("stat".to_string(), stat.clone());
            }
        }

        let published_timestamp = data.get("pubdate")
            .and_then(|p| p.as_i64())
            .unwrap_or(0);

        Ok(VideoMetadata {
            id: data.get("bvid").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            title: data.get("title").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            author: data.get("owner").and_then(|o| o.get("name")).and_then(|v| v.as_str()).unwrap_or("").to_string(),
            author_id: data.get("owner").and_then(|o| o.get("mid")).and_then(|v| v.as_i64()).map(|v| v.to_string()).unwrap_or_default(),
            description: data.get("desc").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            duration: data.get("duration").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            views: data.get("stat").and_then(|s| s.get("view")).and_then(|v| v.as_u64()).unwrap_or(0),
            likes: data.get("stat").and_then(|s| s.get("like")).and_then(|v| v.as_u64()).unwrap_or(0),
            comments: data.get("stat").and_then(|s| s.get("reply")).and_then(|v| v.as_u64()).unwrap_or(0),
            published_at: DateTime::from_timestamp(published_timestamp, 0).unwrap_or_else(|| Utc::now()),
            thumbnail_url: data.get("pic").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            video_url: format!("https://www.bilibili.com/video/{}", data.get("bvid").and_then(|v| v.as_str()).unwrap_or("")),
            tags: data.get("tag").and_then(|t| t.as_array()).map(|arr| {
                arr.iter().filter_map(|t| t.get("tag_name").and_then(|v| v.as_str()).map(|s| s.to_string())).collect()
            }).unwrap_or_default(),
            platform: Platform::Bilibili,
            extra_data,
        })
    }
}

#[async_trait]
impl PlatformAdapter for BilibiliAdapter {
    async fn fetch_video_metadata(&self, video_id: &str) -> Result<VideoMetadata> {
        let cache = self.cache.read().await;
        if let Some((metadata, instant)) = cache.get(video_id) {
            if instant.elapsed() < std::time::Duration::from_secs(300) {
                return Ok(metadata.clone());
            }
        }
        drop(cache);

        let url = format!("https://api.bilibili.com/x/web-interface/view?bvid={}", video_id);
        let response = self.client.get(&url).send().await?;
        let json: serde_json::Value = response.json().await?;

        let metadata = self.parse_bilibili_response(json).await?;

        let mut cache = self.cache.write().await;
        cache.insert(video_id.to_string(), (metadata.clone(), std::time::Instant::now()));

        Ok(metadata)
    }

    async fn fetch_channel_videos(&self, channel_id: &str, limit: usize) -> Result<Vec<VideoMetadata>> {
        let url = format!(
            "https://api.bilibili.com/x/space/arc/search?mid={}&ps={}&pn=1",
            channel_id, limit.min(50)
        );

        let response = self.client.get(&url).send().await?;
        let json: serde_json::Value = response.json().await?;

        let mut videos = Vec::new();
        if let Some(vlist) = json.get("data").and_then(|d| d.get("list")).and_then(|l| l.get("vlist")).and_then(|v| v.as_array()) {
            for video_json in vlist {
                if let Ok(metadata) = self.parse_bilibili_response(serde_json::json!({"data": video_json})).await {
                    videos.push(metadata);
                }
            }
        }

        Ok(videos)
    }

    async fn search_videos(&self, query: &str, limit: usize) -> Result<Vec<VideoMetadata>> {
        let url = format!(
            "https://api.bilibili.com/x/web-interface/search/type?search_type=video&keyword={}&page=1&page_size={}",
            urlencoding::encode(query), limit.min(50)
        );

        let response = self.client.get(&url).send().await?;
        let json: serde_json::Value = response.json().await?;

        let mut videos = Vec::new();
        if let Some(results) = json.get("data").and_then(|d| d.get("result")).and_then(|r| r.as_array()) {
            for result in results {
                let metadata = VideoMetadata {
                    id: result.get("bvid").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    title: result.get("title").and_then(|v| v.as_str()).unwrap_or("").to_string().replace("<em class=\"keyword\">", "").replace("</em>", ""),
                    author: result.get("author").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    author_id: result.get("mid").and_then(|v| v.as_i64()).map(|v| v.to_string()).unwrap_or_default(),
                    description: result.get("description").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    duration: result.get("duration").and_then(|v| v.as_str()).and_then(|d| {
                        let parts: Vec<&str> = d.split(':').collect();
                        if parts.len() == 2 {
                            let minutes = parts[0].parse::<u32>().unwrap_or(0);
                            let seconds = parts[1].parse::<u32>().unwrap_or(0);
                            Some(minutes * 60 + seconds)
                        } else {
                            None
                        }
                    }).unwrap_or(0),
                    views: result.get("play").and_then(|v| v.as_u64()).unwrap_or(0),
                    likes: 0,
                    comments: result.get("review").and_then(|v| v.as_u64()).unwrap_or(0),
                    published_at: Utc::now(),
                    thumbnail_url: format!("https:{}", result.get("pic").and_then(|v| v.as_str()).unwrap_or("")),
                    video_url: format!("https://www.bilibili.com/video/{}", result.get("bvid").and_then(|v| v.as_str()).unwrap_or("")),
                    tags: vec![],
                    platform: Platform::Bilibili,
                    extra_data: HashMap::new(),
                };
                videos.push(metadata);
            }
        }

        Ok(videos)
    }

    async fn get_trending_videos(&self, limit: usize) -> Result<Vec<VideoMetadata>> {
        let url = format!(
            "https://api.bilibili.com/x/web-interface/ranking/v2?rid=0&type=all",
        );

        let response = self.client.get(&url).send().await?;
        let json: serde_json::Value = response.json().await?;

        let mut videos = Vec::new();
        if let Some(list) = json.get("data").and_then(|d| d.get("list")).and_then(|l| l.as_array()) {
            for (i, video_json) in list.iter().enumerate() {
                if i >= limit {
                    break;
                }
                if let Ok(metadata) = self.parse_bilibili_response(serde_json::json!({"data": video_json})).await {
                    videos.push(metadata);
                }
            }
        }

        Ok(videos)
    }

    fn customize_rss_item(&self, metadata: &VideoMetadata) -> Result<String> {
        let mut content = format!(
            r#"<item>
    <title>{}</title>
    <link>{}</link>
    <description><![CDATA[{}]]></description>
    <author>{}</author>
    <pubDate>{}</pubDate>
    <guid isPermaLink="true">{}</guid>"#,
            xmlescape::escape(&metadata.title),
            xmlescape::escape(&metadata.video_url),
            xmlescape::escape(&metadata.description),
            xmlescape::escape(&metadata.author),
            metadata.published_at.to_rfc2822(),
            xmlescape::escape(&metadata.video_url)
        );

        content.push_str(&format!(
            r#"
    <enclosure url="{}" type="image/jpeg"/>
    <bilibili:bvid>{}</bilibili:bvid>
    <bilibili:duration>{}</bilibili:duration>
    <bilibili:views>{}</bilibili:views>
    <bilibili:likes>{}</bilibili:likes>
    <bilibili:comments>{}</bilibili:comments>"#,
            xmlescape::escape(&metadata.thumbnail_url),
            xmlescape::escape(&metadata.id),
            metadata.duration,
            metadata.views,
            metadata.likes,
            metadata.comments
        ));

        if self.config.include_danmaku {
            if let Some(danmaku_count) = metadata.extra_data.get("danmaku_count") {
                content.push_str(&format!(
                    "\n    <bilibili:danmaku>{}</bilibili:danmaku>",
                    danmaku_count
                ));
            }
        }

        if self.config.include_stats {
            if let Some(stat) = metadata.extra_data.get("stat") {
                if let Some(coin) = stat.get("coin") {
                    content.push_str(&format!(
                        "\n    <bilibili:coin>{}</bilibili:coin>",
                        coin
                    ));
                }
                if let Some(share) = stat.get("share") {
                    content.push_str(&format!(
                        "\n    <bilibili:share>{}</bilibili:share>",
                        share
                    ));
                }
            }
        }

        if !metadata.tags.is_empty() {
            content.push_str("\n    <category>");
            content.push_str(&xmlescape::escape(&metadata.tags.join(", ")));
            content.push_str("</category>");
        }

        content.push_str("\n</item>");
        Ok(content)
    }

    fn get_platform(&self) -> Platform {
        Platform::Bilibili
    }
}

pub struct DouyinAdapter {
    config: DouyinConfig,
    client: reqwest::Client,
    cache: Arc<RwLock<HashMap<String, (VideoMetadata, std::time::Instant)>>>,
}

impl DouyinAdapter {
    pub fn new(config: DouyinConfig) -> Self {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("User-Agent", "Mozilla/5.0 (iPhone; CPU iPhone OS 13_2_3 like Mac OS X) AppleWebKit/605.1.15".parse().unwrap());
        headers.insert("Referer", "https://www.douyin.com".parse().unwrap());

        let client = reqwest::Client::builder()
            .default_headers(headers)
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .unwrap();

        Self {
            config,
            client,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    fn extract_video_id(&self, url: &str) -> Option<String> {
        let re = Regex::new(r"video/(\d+)").ok()?;
        re.captures(url).and_then(|cap| cap.get(1).map(|m| m.as_str().to_string()))
    }
}

#[async_trait]
impl PlatformAdapter for DouyinAdapter {
    async fn fetch_video_metadata(&self, video_id: &str) -> Result<VideoMetadata> {
        let cache = self.cache.read().await;
        if let Some((metadata, instant)) = cache.get(video_id) {
            if instant.elapsed() < std::time::Duration::from_secs(300) {
                return Ok(metadata.clone());
            }
        }
        drop(cache);

        let url = format!("https://www.iesdouyin.com/web/api/v2/aweme/iteminfo/?item_ids={}", video_id);
        let response = self.client.get(&url).send().await?;
        let json: serde_json::Value = response.json().await?;

        let item = json.get("item_list")
            .and_then(|list| list.as_array())
            .and_then(|arr| arr.first())
            .ok_or(anyhow!("No video data found"))?;

        let mut extra_data = HashMap::new();

        if self.config.include_comments {
            if let Some(comment_count) = item.get("statistics").and_then(|s| s.get("comment_count")) {
                extra_data.insert("comment_count".to_string(), comment_count.clone());
            }
        }

        if self.config.include_music {
            if let Some(music) = item.get("music") {
                extra_data.insert("music".to_string(), music.clone());
            }
        }

        let metadata = VideoMetadata {
            id: video_id.to_string(),
            title: item.get("desc").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            author: item.get("author").and_then(|a| a.get("nickname")).and_then(|v| v.as_str()).unwrap_or("").to_string(),
            author_id: item.get("author").and_then(|a| a.get("uid")).and_then(|v| v.as_str()).unwrap_or("").to_string(),
            description: item.get("desc").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            duration: item.get("video").and_then(|v| v.get("duration")).and_then(|d| d.as_u64()).unwrap_or(0) as u32 / 1000,
            views: item.get("statistics").and_then(|s| s.get("play_count")).and_then(|v| v.as_u64()).unwrap_or(0),
            likes: item.get("statistics").and_then(|s| s.get("digg_count")).and_then(|v| v.as_u64()).unwrap_or(0),
            comments: item.get("statistics").and_then(|s| s.get("comment_count")).and_then(|v| v.as_u64()).unwrap_or(0),
            published_at: DateTime::from_timestamp(
                item.get("create_time").and_then(|t| t.as_i64()).unwrap_or(0),
                0
            ).unwrap_or_else(|| Utc::now()),
            thumbnail_url: item.get("video").and_then(|v| v.get("cover")).and_then(|c| c.get("url_list"))
                .and_then(|list| list.as_array()).and_then(|arr| arr.first())
                .and_then(|v| v.as_str()).unwrap_or("").to_string(),
            video_url: format!("https://www.douyin.com/video/{}", video_id),
            tags: item.get("text_extra").and_then(|t| t.as_array()).map(|arr| {
                arr.iter().filter_map(|t| t.get("hashtag_name").and_then(|v| v.as_str()).map(|s| s.to_string())).collect()
            }).unwrap_or_default(),
            platform: Platform::Douyin,
            extra_data,
        };

        let mut cache = self.cache.write().await;
        cache.insert(video_id.to_string(), (metadata.clone(), std::time::Instant::now()));

        Ok(metadata)
    }

    async fn fetch_channel_videos(&self, channel_id: &str, limit: usize) -> Result<Vec<VideoMetadata>> {
        let url = format!(
            "https://www.iesdouyin.com/web/api/v2/aweme/post/?sec_uid={}&count={}&max_cursor=0",
            channel_id, limit.min(30)
        );

        let response = self.client.get(&url).send().await?;
        let json: serde_json::Value = response.json().await?;

        let mut videos = Vec::new();
        if let Some(aweme_list) = json.get("aweme_list").and_then(|l| l.as_array()) {
            for item in aweme_list {
                let video_id = item.get("aweme_id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                if !video_id.is_empty() {
                    if let Ok(metadata) = self.fetch_video_metadata(&video_id).await {
                        videos.push(metadata);
                    }
                }
            }
        }

        Ok(videos)
    }

    async fn search_videos(&self, query: &str, limit: usize) -> Result<Vec<VideoMetadata>> {
        Err(anyhow!("Douyin search API requires authentication"))
    }

    async fn get_trending_videos(&self, limit: usize) -> Result<Vec<VideoMetadata>> {
        let url = "https://www.douyin.com/aweme/v1/web/hot/search/list/";
        let response = self.client.get(url).send().await?;
        let json: serde_json::Value = response.json().await?;

        let mut videos = Vec::new();
        if let Some(data_list) = json.get("data").and_then(|d| d.get("word_list")).and_then(|w| w.as_array()) {
            for (i, item) in data_list.iter().enumerate() {
                if i >= limit {
                    break;
                }

                let metadata = VideoMetadata {
                    id: format!("trend_{}", i),
                    title: item.get("word").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    author: "Trending".to_string(),
                    author_id: "0".to_string(),
                    description: format!("Hot search rank: {}", item.get("hot_value").and_then(|v| v.as_u64()).unwrap_or(0)),
                    duration: 0,
                    views: item.get("hot_value").and_then(|v| v.as_u64()).unwrap_or(0),
                    likes: 0,
                    comments: 0,
                    published_at: Utc::now(),
                    thumbnail_url: String::new(),
                    video_url: format!("https://www.douyin.com/search/{}", urlencoding::encode(item.get("word").and_then(|v| v.as_str()).unwrap_or(""))),
                    tags: vec!["trending".to_string()],
                    platform: Platform::Douyin,
                    extra_data: HashMap::new(),
                };
                videos.push(metadata);
            }
        }

        Ok(videos)
    }

    fn customize_rss_item(&self, metadata: &VideoMetadata) -> Result<String> {
        let mut content = format!(
            r#"<item>
    <title>{}</title>
    <link>{}</link>
    <description><![CDATA[{}]]></description>
    <author>{}</author>
    <pubDate>{}</pubDate>
    <guid isPermaLink="true">{}</guid>"#,
            xmlescape::escape(&metadata.title),
            xmlescape::escape(&metadata.video_url),
            xmlescape::escape(&metadata.description),
            xmlescape::escape(&metadata.author),
            metadata.published_at.to_rfc2822(),
            xmlescape::escape(&metadata.video_url)
        );

        if !metadata.thumbnail_url.is_empty() {
            content.push_str(&format!(
                r#"
    <enclosure url="{}" type="image/jpeg"/>"#,
                xmlescape::escape(&metadata.thumbnail_url)
            ));
        }

        content.push_str(&format!(
            r#"
    <douyin:id>{}</douyin:id>
    <douyin:duration>{}</douyin:duration>
    <douyin:views>{}</douyin:views>
    <douyin:likes>{}</douyin:likes>"#,
            xmlescape::escape(&metadata.id),
            metadata.duration,
            metadata.views,
            metadata.likes
        ));

        if self.config.include_comments {
            content.push_str(&format!(
                "\n    <douyin:comments>{}</douyin:comments>",
                metadata.comments
            ));
        }

        if self.config.include_music {
            if let Some(music) = metadata.extra_data.get("music") {
                if let Some(title) = music.get("title").and_then(|t| t.as_str()) {
                    content.push_str(&format!(
                        "\n    <douyin:music>{}</douyin:music>",
                        xmlescape::escape(title)
                    ));
                }
            }
        }

        if !metadata.tags.is_empty() {
            for tag in &metadata.tags {
                content.push_str(&format!(
                    "\n    <category>{}</category>",
                    xmlescape::escape(tag)
                ));
            }
        }

        content.push_str("\n</item>");
        Ok(content)
    }

    fn get_platform(&self) -> Platform {
        Platform::Douyin
    }
}

pub struct KuaishouAdapter {
    config: KuaishouConfig,
    client: reqwest::Client,
    cache: Arc<RwLock<HashMap<String, (VideoMetadata, std::time::Instant)>>>,
}

impl KuaishouAdapter {
    pub fn new(config: KuaishouConfig) -> Self {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36".parse().unwrap());
        headers.insert("Referer", "https://www.kuaishou.com".parse().unwrap());

        let client = reqwest::Client::builder()
            .default_headers(headers)
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .unwrap();

        Self {
            config,
            client,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl PlatformAdapter for KuaishouAdapter {
    async fn fetch_video_metadata(&self, video_id: &str) -> Result<VideoMetadata> {
        let cache = self.cache.read().await;
        if let Some((metadata, instant)) = cache.get(video_id) {
            if instant.elapsed() < std::time::Duration::from_secs(300) {
                return Ok(metadata.clone());
            }
        }
        drop(cache);

        let url = format!("https://www.kuaishou.com/short-video/{}", video_id);
        let response = self.client.get(&url).send().await?;
        let html = response.text().await?;

        let document = Html::parse_document(&html);
        let script_selector = Selector::parse("script").unwrap();

        let mut metadata_json = None;
        for element in document.select(&script_selector) {
            let text = element.text().collect::<String>();
            if text.contains("window.__APOLLO_STATE__") {
                let start = text.find('{').unwrap_or(0);
                let end = text.rfind('}').unwrap_or(text.len()) + 1;
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text[start..end]) {
                    metadata_json = Some(json);
                    break;
                }
            }
        }

        let json = metadata_json.ok_or(anyhow!("Failed to extract video metadata"))?;

        let mut extra_data = HashMap::new();

        if self.config.include_gifts {
            extra_data.insert("gifts_enabled".to_string(), serde_json::Value::Bool(true));
        }

        if self.config.include_location {
            if let Some(location) = json.get("location") {
                extra_data.insert("location".to_string(), location.clone());
            }
        }

        let metadata = VideoMetadata {
            id: video_id.to_string(),
            title: json.get("caption").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            author: json.get("userName").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            author_id: json.get("userId").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            description: json.get("caption").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            duration: json.get("duration").and_then(|v| v.as_u64()).unwrap_or(0) as u32 / 1000,
            views: json.get("viewCount").and_then(|v| v.as_u64()).unwrap_or(0),
            likes: json.get("likeCount").and_then(|v| v.as_u64()).unwrap_or(0),
            comments: json.get("commentCount").and_then(|v| v.as_u64()).unwrap_or(0),
            published_at: DateTime::from_timestamp(
                json.get("timestamp").and_then(|t| t.as_i64()).unwrap_or(0) / 1000,
                0
            ).unwrap_or_else(|| Utc::now()),
            thumbnail_url: json.get("poster").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            video_url: format!("https://www.kuaishou.com/short-video/{}", video_id),
            tags: json.get("tags").and_then(|t| t.as_array()).map(|arr| {
                arr.iter().filter_map(|t| t.as_str().map(|s| s.to_string())).collect()
            }).unwrap_or_default(),
            platform: Platform::Kuaishou,
            extra_data,
        };

        let mut cache = self.cache.write().await;
        cache.insert(video_id.to_string(), (metadata.clone(), std::time::Instant::now()));

        Ok(metadata)
    }

    async fn fetch_channel_videos(&self, channel_id: &str, limit: usize) -> Result<Vec<VideoMetadata>> {
        let url = format!("https://www.kuaishou.com/profile/{}", channel_id);
        let response = self.client.get(&url).send().await?;
        let html = response.text().await?;

        let document = Html::parse_document(&html);
        let video_selector = Selector::parse("div.video-card").unwrap();

        let mut videos = Vec::new();
        for (i, element) in document.select(&video_selector).enumerate() {
            if i >= limit {
                break;
            }

            if let Some(video_id) = element.value().attr("data-video-id") {
                if let Ok(metadata) = self.fetch_video_metadata(video_id).await {
                    videos.push(metadata);
                }
            }
        }

        Ok(videos)
    }

    async fn search_videos(&self, query: &str, limit: usize) -> Result<Vec<VideoMetadata>> {
        let url = format!("https://www.kuaishou.com/search/video?searchKey={}", urlencoding::encode(query));
        let response = self.client.get(&url).send().await?;
        let html = response.text().await?;

        let document = Html::parse_document(&html);
        let video_selector = Selector::parse("div.video-card").unwrap();

        let mut videos = Vec::new();
        for (i, element) in document.select(&video_selector).enumerate() {
            if i >= limit {
                break;
            }

            if let Some(video_id) = element.value().attr("data-video-id") {
                if let Ok(metadata) = self.fetch_video_metadata(video_id).await {
                    videos.push(metadata);
                }
            }
        }

        Ok(videos)
    }

    async fn get_trending_videos(&self, limit: usize) -> Result<Vec<VideoMetadata>> {
        let url = "https://www.kuaishou.com/hot";
        let response = self.client.get(url).send().await?;
        let html = response.text().await?;

        let document = Html::parse_document(&html);
        let video_selector = Selector::parse("div.video-card").unwrap();

        let mut videos = Vec::new();
        for (i, element) in document.select(&video_selector).enumerate() {
            if i >= limit {
                break;
            }

            if let Some(video_id) = element.value().attr("data-video-id") {
                if let Ok(metadata) = self.fetch_video_metadata(video_id).await {
                    videos.push(metadata);
                }
            }
        }

        Ok(videos)
    }

    fn customize_rss_item(&self, metadata: &VideoMetadata) -> Result<String> {
        let mut content = format!(
            r#"<item>
    <title>{}</title>
    <link>{}</link>
    <description><![CDATA[{}]]></description>
    <author>{}</author>
    <pubDate>{}</pubDate>
    <guid isPermaLink="true">{}</guid>"#,
            xmlescape::escape(&metadata.title),
            xmlescape::escape(&metadata.video_url),
            xmlescape::escape(&metadata.description),
            xmlescape::escape(&metadata.author),
            metadata.published_at.to_rfc2822(),
            xmlescape::escape(&metadata.video_url)
        );

        if !metadata.thumbnail_url.is_empty() {
            content.push_str(&format!(
                r#"
    <enclosure url="{}" type="image/jpeg"/>"#,
                xmlescape::escape(&metadata.thumbnail_url)
            ));
        }

        content.push_str(&format!(
            r#"
    <kuaishou:id>{}</kuaishou:id>
    <kuaishou:duration>{}</kuaishou:duration>
    <kuaishou:views>{}</kuaishou:views>
    <kuaishou:likes>{}</kuaishou:likes>
    <kuaishou:comments>{}</kuaishou:comments>"#,
            xmlescape::escape(&metadata.id),
            metadata.duration,
            metadata.views,
            metadata.likes,
            metadata.comments
        ));

        if self.config.include_location {
            if let Some(location) = metadata.extra_data.get("location") {
                if let Some(loc_str) = location.as_str() {
                    content.push_str(&format!(
                        "\n    <kuaishou:location>{}</kuaishou:location>",
                        xmlescape::escape(loc_str)
                    ));
                }
            }
        }

        if self.config.include_gifts {
            content.push_str("\n    <kuaishou:gifts>enabled</kuaishou:gifts>");
        }

        if !metadata.tags.is_empty() {
            for tag in &metadata.tags {
                content.push_str(&format!(
                    "\n    <category>{}</category>",
                    xmlescape::escape(tag)
                ));
            }
        }

        content.push_str("\n</item>");
        Ok(content)
    }

    fn get_platform(&self) -> Platform {
        Platform::Kuaishou
    }
}

pub struct YouTubeAdapter {
    config: YouTubeConfig,
    client: reqwest::Client,
    cache: Arc<RwLock<HashMap<String, (VideoMetadata, std::time::Instant)>>>,
}

impl YouTubeAdapter {
    pub fn new(config: YouTubeConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .unwrap();

        Self {
            config,
            client,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    fn parse_duration(&self, duration: &str) -> u32 {
        let re = Regex::new(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?").unwrap();
        if let Some(captures) = re.captures(duration) {
            let hours = captures.get(1).and_then(|m| m.as_str().parse::<u32>().ok()).unwrap_or(0);
            let minutes = captures.get(2).and_then(|m| m.as_str().parse::<u32>().ok()).unwrap_or(0);
            let seconds = captures.get(3).and_then(|m| m.as_str().parse::<u32>().ok()).unwrap_or(0);
            hours * 3600 + minutes * 60 + seconds
        } else {
            0
        }
    }
}

#[async_trait]
impl PlatformAdapter for YouTubeAdapter {
    async fn fetch_video_metadata(&self, video_id: &str) -> Result<VideoMetadata> {
        let cache = self.cache.read().await;
        if let Some((metadata, instant)) = cache.get(video_id) {
            if instant.elapsed() < std::time::Duration::from_secs(300) {
                return Ok(metadata.clone());
            }
        }
        drop(cache);

        if let Some(api_key) = &self.config.api_key {
            let url = format!(
                "https://www.googleapis.com/youtube/v3/videos?id={}&key={}&part=snippet,contentDetails,statistics",
                video_id, api_key
            );

            let response = self.client.get(&url).send().await?;
            let json: serde_json::Value = response.json().await?;

            let item = json.get("items")
                .and_then(|items| items.as_array())
                .and_then(|arr| arr.first())
                .ok_or(anyhow!("No video data found"))?;

            let snippet = item.get("snippet").ok_or(anyhow!("No snippet data"))?;
            let statistics = item.get("statistics").ok_or(anyhow!("No statistics data"))?;
            let content_details = item.get("contentDetails").ok_or(anyhow!("No content details"))?;

            let mut extra_data = HashMap::new();

            if self.config.include_captions {
                extra_data.insert("captions_available".to_string(), serde_json::Value::Bool(true));
            }

            if self.config.include_chapters {
                extra_data.insert("chapters_enabled".to_string(), serde_json::Value::Bool(true));
            }

            let metadata = VideoMetadata {
                id: video_id.to_string(),
                title: snippet.get("title").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                author: snippet.get("channelTitle").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                author_id: snippet.get("channelId").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                description: snippet.get("description").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                duration: self.parse_duration(content_details.get("duration").and_then(|v| v.as_str()).unwrap_or("")),
                views: statistics.get("viewCount").and_then(|v| v.as_str()).and_then(|s| s.parse().ok()).unwrap_or(0),
                likes: statistics.get("likeCount").and_then(|v| v.as_str()).and_then(|s| s.parse().ok()).unwrap_or(0),
                comments: statistics.get("commentCount").and_then(|v| v.as_str()).and_then(|s| s.parse().ok()).unwrap_or(0),
                published_at: DateTime::parse_from_rfc3339(
                    snippet.get("publishedAt").and_then(|v| v.as_str()).unwrap_or("")
                ).map(|dt| dt.with_timezone(&Utc)).unwrap_or_else(|_| Utc::now()),
                thumbnail_url: snippet.get("thumbnails")
                    .and_then(|t| t.get("high"))
                    .and_then(|h| h.get("url"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("").to_string(),
                video_url: format!("https://www.youtube.com/watch?v={}", video_id),
                tags: snippet.get("tags").and_then(|t| t.as_array()).map(|arr| {
                    arr.iter().filter_map(|t| t.as_str().map(|s| s.to_string())).collect()
                }).unwrap_or_default(),
                platform: Platform::YouTube,
                extra_data,
            };

            let mut cache = self.cache.write().await;
            cache.insert(video_id.to_string(), (metadata.clone(), std::time::Instant::now()));

            Ok(metadata)
        } else {
            Err(anyhow!("YouTube API key not configured"))
        }
    }

    async fn fetch_channel_videos(&self, channel_id: &str, limit: usize) -> Result<Vec<VideoMetadata>> {
        if let Some(api_key) = &self.config.api_key {
            let url = format!(
                "https://www.googleapis.com/youtube/v3/search?channelId={}&key={}&part=id&type=video&maxResults={}&order=date",
                channel_id, api_key, limit.min(50)
            );

            let response = self.client.get(&url).send().await?;
            let json: serde_json::Value = response.json().await?;

            let mut videos = Vec::new();
            if let Some(items) = json.get("items").and_then(|i| i.as_array()) {
                for item in items {
                    if let Some(video_id) = item.get("id").and_then(|id| id.get("videoId")).and_then(|v| v.as_str()) {
                        if let Ok(metadata) = self.fetch_video_metadata(video_id).await {
                            videos.push(metadata);
                        }
                    }
                }
            }

            Ok(videos)
        } else {
            Err(anyhow!("YouTube API key not configured"))
        }
    }

    async fn search_videos(&self, query: &str, limit: usize) -> Result<Vec<VideoMetadata>> {
        if let Some(api_key) = &self.config.api_key {
            let url = format!(
                "https://www.googleapis.com/youtube/v3/search?q={}&key={}&part=id&type=video&maxResults={}",
                urlencoding::encode(query), api_key, limit.min(50)
            );

            let response = self.client.get(&url).send().await?;
            let json: serde_json::Value = response.json().await?;

            let mut videos = Vec::new();
            if let Some(items) = json.get("items").and_then(|i| i.as_array()) {
                for item in items {
                    if let Some(video_id) = item.get("id").and_then(|id| id.get("videoId")).and_then(|v| v.as_str()) {
                        if let Ok(metadata) = self.fetch_video_metadata(video_id).await {
                            videos.push(metadata);
                        }
                    }
                }
            }

            Ok(videos)
        } else {
            Err(anyhow!("YouTube API key not configured"))
        }
    }

    async fn get_trending_videos(&self, limit: usize) -> Result<Vec<VideoMetadata>> {
        if let Some(api_key) = &self.config.api_key {
            let url = format!(
                "https://www.googleapis.com/youtube/v3/videos?chart=mostPopular&key={}&part=id&maxResults={}&regionCode=US",
                api_key, limit.min(50)
            );

            let response = self.client.get(&url).send().await?;
            let json: serde_json::Value = response.json().await?;

            let mut videos = Vec::new();
            if let Some(items) = json.get("items").and_then(|i| i.as_array()) {
                for item in items {
                    if let Some(video_id) = item.get("id").and_then(|v| v.as_str()) {
                        if let Ok(metadata) = self.fetch_video_metadata(video_id).await {
                            videos.push(metadata);
                        }
                    }
                }
            }

            Ok(videos)
        } else {
            Err(anyhow!("YouTube API key not configured"))
        }
    }

    fn customize_rss_item(&self, metadata: &VideoMetadata) -> Result<String> {
        let mut content = format!(
            r#"<item>
    <title>{}</title>
    <link>{}</link>
    <description><![CDATA[{}]]></description>
    <author>{}</author>
    <pubDate>{}</pubDate>
    <guid isPermaLink="true">{}</guid>"#,
            xmlescape::escape(&metadata.title),
            xmlescape::escape(&metadata.video_url),
            xmlescape::escape(&metadata.description),
            xmlescape::escape(&metadata.author),
            metadata.published_at.to_rfc2822(),
            xmlescape::escape(&metadata.video_url)
        );

        content.push_str(&format!(
            r#"
    <enclosure url="{}" type="image/jpeg"/>
    <media:content url="{}" type="video/mp4"/>
    <yt:videoId>{}</yt:videoId>
    <yt:channelId>{}</yt:channelId>
    <media:statistics views="{}"/>
    <media:community starRating="{}" statistics="views: {}, likes: {}, comments: {}"/>"#,
            xmlescape::escape(&metadata.thumbnail_url),
            xmlescape::escape(&metadata.video_url),
            xmlescape::escape(&metadata.id),
            xmlescape::escape(&metadata.author_id),
            metadata.views,
            metadata.likes,
            metadata.views,
            metadata.likes,
            metadata.comments
        ));

        if self.config.include_captions {
            content.push_str("\n    <media:subTitle type=\"application/ttml+xml\" lang=\"en\"/>");
        }

        if self.config.include_chapters {
            content.push_str("\n    <yt:hasChapters>true</yt:hasChapters>");
        }

        if !metadata.tags.is_empty() {
            content.push_str("\n    <media:keywords>");
            content.push_str(&xmlescape::escape(&metadata.tags.join(", ")));
            content.push_str("</media:keywords>");
        }

        content.push_str("\n</item>");
        Ok(content)
    }

    fn get_platform(&self) -> Platform {
        Platform::YouTube
    }
}

pub struct PlatformManager {
    adapters: HashMap<Platform, Arc<dyn PlatformAdapter>>,
}

impl PlatformManager {
    pub fn new(config: PlatformConfig) -> Self {
        let mut adapters = HashMap::new();

        adapters.insert(
            Platform::Bilibili,
            Arc::new(BilibiliAdapter::new(config.bilibili)) as Arc<dyn PlatformAdapter>
        );

        adapters.insert(
            Platform::Douyin,
            Arc::new(DouyinAdapter::new(config.douyin)) as Arc<dyn PlatformAdapter>
        );

        adapters.insert(
            Platform::Kuaishou,
            Arc::new(KuaishouAdapter::new(config.kuaishou)) as Arc<dyn PlatformAdapter>
        );

        adapters.insert(
            Platform::YouTube,
            Arc::new(YouTubeAdapter::new(config.youtube)) as Arc<dyn PlatformAdapter>
        );

        Self { adapters }
    }

    pub fn get_adapter(&self, platform: Platform) -> Option<Arc<dyn PlatformAdapter>> {
        self.adapters.get(&platform).cloned()
    }

    pub async fn generate_unified_rss(&self, videos: Vec<VideoMetadata>) -> Result<String> {
        let mut rss = String::from(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"
     xmlns:media="http://search.yahoo.com/mrss/"
     xmlns:yt="http://www.youtube.com/xml/schemas/2015"
     xmlns:bilibili="https://www.bilibili.com/xml/schemas/2024"
     xmlns:douyin="https://www.douyin.com/xml/schemas/2024"
     xmlns:kuaishou="https://www.kuaishou.com/xml/schemas/2024">
<channel>
    <title>Unified Video Feed</title>
    <link>https://localhost/feed</link>
    <description>Aggregated video content from multiple platforms</description>
    <lastBuildDate>"#
        );

        rss.push_str(&Utc::now().to_rfc2822());
        rss.push_str("</lastBuildDate>\n");

        for video in videos {
            if let Some(adapter) = self.get_adapter(video.platform.clone()) {
                if let Ok(item) = adapter.customize_rss_item(&video) {
                    rss.push_str(&item);
                    rss.push('\n');
                }
            }
        }

        rss.push_str("</channel>\n</rss>");
        Ok(rss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_config() {
        let config = PlatformConfig {
            bilibili: BilibiliConfig {
                include_danmaku: true,
                include_stats: true,
                quality_preference: VideoQuality::HD1080p,
                subtitle_lang: "zh-CN".to_string(),
                cookie: None,
            },
            douyin: DouyinConfig {
                include_comments: true,
                include_music: true,
                watermark_removal: false,
                quality_preference: VideoQuality::HD720p,
            },
            kuaishou: KuaishouConfig {
                include_gifts: false,
                include_location: true,
                quality_preference: VideoQuality::Auto,
            },
            youtube: YouTubeConfig {
                include_captions: true,
                include_chapters: true,
                quality_preference: VideoQuality::Highest,
                api_key: Some("test_key".to_string()),
            },
        };

        let manager = PlatformManager::new(config);
        assert!(manager.get_adapter(Platform::Bilibili).is_some());
        assert!(manager.get_adapter(Platform::Douyin).is_some());
        assert!(manager.get_adapter(Platform::Kuaishou).is_some());
        assert!(manager.get_adapter(Platform::YouTube).is_some());
    }

    #[test]
    fn test_video_metadata_serialization() {
        let metadata = VideoMetadata {
            id: "test123".to_string(),
            title: "Test Video".to_string(),
            author: "Test Author".to_string(),
            author_id: "author123".to_string(),
            description: "Test description".to_string(),
            duration: 300,
            views: 1000,
            likes: 100,
            comments: 50,
            published_at: Utc::now(),
            thumbnail_url: "https://example.com/thumb.jpg".to_string(),
            video_url: "https://example.com/video".to_string(),
            tags: vec!["test".to_string(), "video".to_string()],
            platform: Platform::Bilibili,
            extra_data: HashMap::new(),
        };

        let serialized = serde_json::to_string(&metadata).unwrap();
        let deserialized: VideoMetadata = serde_json::from_str(&serialized).unwrap();

        assert_eq!(metadata.id, deserialized.id);
        assert_eq!(metadata.title, deserialized.title);
        assert_eq!(metadata.platform, deserialized.platform);
    }
}