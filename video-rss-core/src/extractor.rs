use crate::{error::VideoRssError, types::*, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use regex::Regex;
use reqwest::{Client, ClientBuilder};
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::env;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedVideo {
    pub id: String,
    pub title: String,
    pub platform: Platform,
    pub author: String,
    pub url: String,
    pub description: String,
    pub view_count: Option<u64>,
    pub like_count: Option<u64>,
    pub duration: Option<String>,
    pub upload_date: String,
    pub thumbnail: Option<String>,
    pub tags: Vec<String>,

    // Legal compliance metadata
    pub data_source: String,
    pub legal_compliance: String,
    pub extraction_method: String,
    pub extracted_at: DateTime<Utc>,
}

#[async_trait]
pub trait DataExtractor: Send + Sync {
    async fn extract(&self) -> Result<Vec<ExtractedVideo>>;
    fn platform(&self) -> Platform;
}

pub struct VideoExtractor {
    client: Client,
    extractors: Vec<Box<dyn DataExtractor>>,
}

impl VideoExtractor {
    pub fn new() -> Result<Self> {
        dotenv::dotenv().ok();

        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::USER_AGENT,
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36".parse().unwrap(),
        );
        headers.insert(
            reqwest::header::ACCEPT,
            "application/json, text/html".parse().unwrap(),
        );
        headers.insert(
            reqwest::header::ACCEPT_LANGUAGE,
            "zh-CN,zh;q=0.9,en;q=0.8".parse().unwrap(),
        );

        let client = ClientBuilder::new()
            .default_headers(headers)
            .timeout(std::time::Duration::from_secs(30))
            .build()?;

        let mut extractors: Vec<Box<dyn DataExtractor>> = Vec::new();

        // Add Bilibili extractor
        extractors.push(Box::new(BilibiliExtractor::new(client.clone())));

        // Add YouTube extractor
        extractors.push(Box::new(YouTubeExtractor::new(client.clone())));

        // Add Douyin extractor
        extractors.push(Box::new(DouyinExtractor::new(client.clone())));

        Ok(Self { client, extractors })
    }

    pub async fn extract_all(&self) -> HashMap<Platform, Vec<ExtractedVideo>> {
        let mut results = HashMap::new();

        let tasks: Vec<_> = self.extractors.iter().map(|extractor| {
            async move {
                let platform = extractor.platform();
                let videos = extractor.extract().await.unwrap_or_default();
                (platform, videos)
            }
        }).collect();

        let futures = futures::future::join_all(tasks).await;

        for (platform, videos) in futures {
            results.insert(platform, videos);
        }

        results
    }
}

// Bilibili Extractor
pub struct BilibiliExtractor {
    client: Client,
}

impl BilibiliExtractor {
    pub fn new(client: Client) -> Self {
        Self { client }
    }

    async fn extract_personalized(&self) -> Result<Vec<ExtractedVideo>> {
        let mut videos = Vec::new();

        // Check for credentials
        let sessdata = env::var("BILIBILI_SESSDATA").ok();
        let bili_jct = env::var("BILIBILI_BILI_JCT").ok();
        let buvid3 = env::var("BILIBILI_BUVID3").ok();

        if sessdata.is_none() {
            return Ok(videos);
        }

        // Build request with cookies
        let mut request = self.client
            .get("https://api.bilibili.com/x/web-interface/history/cursor")
            .query(&[("ps", "20")]);

        if let Some(sessdata) = sessdata {
            request = request.header("Cookie", format!("SESSDATA={}", sessdata));
        }

        let response = request.send().await?;

        if response.status().is_success() {
            let data: Value = response.json().await?;

            if data["code"].as_i64() == Some(0) {
                if let Some(list) = data["data"]["list"].as_array() {
                    for item in list.iter().take(10) {
                        if item["history"]["business"].as_str() == Some("archive") {
                            let video = ExtractedVideo {
                                id: item["history"]["bvid"].as_str().unwrap_or("").to_string(),
                                title: item["title"].as_str().unwrap_or("").to_string(),
                                platform: Platform::Bilibili,
                                author: item["author_name"].as_str().unwrap_or("Unknown").to_string(),
                                url: format!(
                                    "https://www.bilibili.com/video/{}",
                                    item["history"]["bvid"].as_str().unwrap_or("")
                                ),
                                description: item["new_desc"].as_str().unwrap_or("").to_string(),
                                view_count: item["stat"]["view"].as_u64(),
                                like_count: item["stat"]["like"].as_u64(),
                                duration: item["duration"].as_str().map(|s| s.to_string()),
                                upload_date: Utc::now().format("%Y-%m-%d").to_string(),
                                thumbnail: item["cover"].as_str().map(|s| s.to_string()),
                                tags: item["tag_name"]
                                    .as_str()
                                    .map(|s| s.split(',').map(|t| t.to_string()).collect())
                                    .unwrap_or_default(),
                                data_source: "bilibili_personalized_history".to_string(),
                                legal_compliance: "Authenticated User Data - Fair Use".to_string(),
                                extraction_method: "Personal watch history - No downloads".to_string(),
                                extracted_at: Utc::now(),
                            };
                            videos.push(video);
                        }
                    }
                }
            }
        }

        Ok(videos)
    }

    async fn extract_public(&self) -> Result<Vec<ExtractedVideo>> {
        let mut videos = Vec::new();

        let response = self.client
            .get("https://api.bilibili.com/x/web-interface/search/all")
            .query(&[
                ("keyword", "热门"),
                ("page", "1"),
                ("pagesize", "20"),
            ])
            .send()
            .await?;

        if response.status().is_success() {
            let data: Value = response.json().await?;

            if data["code"].as_i64() == Some(0) {
                if let Some(results) = data["data"]["result"]["video"].as_array() {
                    for item in results.iter().take(10) {
                        let title = item["title"]
                            .as_str()
                            .unwrap_or("")
                            .replace("<em class=\"keyword\">", "")
                            .replace("</em>", "");

                        let video = ExtractedVideo {
                            id: item["bvid"].as_str().unwrap_or("").to_string(),
                            title: html_escape::decode_html_entities(&title).to_string(),
                            platform: Platform::Bilibili,
                            author: item["author"].as_str().unwrap_or("Unknown").to_string(),
                            url: format!(
                                "https://www.bilibili.com/video/{}",
                                item["bvid"].as_str().unwrap_or("")
                            ),
                            description: item["description"].as_str().unwrap_or("").to_string(),
                            view_count: item["play"].as_u64(),
                            like_count: item["favorites"].as_u64(),
                            duration: item["duration"].as_str().map(|s| s.to_string()),
                            upload_date: item["pubdate"]
                                .as_i64()
                                .map(|ts| {
                                    DateTime::from_timestamp(ts, 0)
                                        .unwrap_or_else(|| Utc::now())
                                        .format("%Y-%m-%d")
                                        .to_string()
                                })
                                .unwrap_or_else(|| Utc::now().format("%Y-%m-%d").to_string()),
                            thumbnail: item["pic"]
                                .as_str()
                                .map(|s| s.replace("//", "https://"))
                                .or_else(|| Some(String::new())),
                            tags: item["tag"]
                                .as_str()
                                .map(|s| s.split(',').map(|t| t.to_string()).collect())
                                .unwrap_or_default(),
                            data_source: "bilibili_search_api".to_string(),
                            legal_compliance: "Public API - Fair Use Academic Research".to_string(),
                            extraction_method: "Streaming metadata only - No downloads".to_string(),
                            extracted_at: Utc::now(),
                        };
                        videos.push(video);
                    }
                }
            }
        }

        Ok(videos)
    }
}

#[async_trait]
impl DataExtractor for BilibiliExtractor {
    async fn extract(&self) -> Result<Vec<ExtractedVideo>> {
        // Try personalized first, fallback to public
        let mut videos = self.extract_personalized().await.unwrap_or_default();

        if videos.is_empty() {
            videos = self.extract_public().await?;
        }

        Ok(videos)
    }

    fn platform(&self) -> Platform {
        Platform::Bilibili
    }
}

// YouTube Extractor
pub struct YouTubeExtractor {
    client: Client,
}

impl YouTubeExtractor {
    pub fn new(client: Client) -> Self {
        Self { client }
    }
}

#[async_trait]
impl DataExtractor for YouTubeExtractor {
    async fn extract(&self) -> Result<Vec<ExtractedVideo>> {
        let mut videos = Vec::new();
        let channel_id = "UCBa659QWEk1AI4Tg--mrJ2A"; // Tom Scott

        let rss_url = format!(
            "https://www.youtube.com/feeds/videos.xml?channel_id={}",
            channel_id
        );

        let response = self.client.get(&rss_url).send().await?;

        if response.status().is_success() {
            let rss_content = response.text().await?;

            // Parse RSS using regex (simplified)
            let entry_re = Regex::new(r"<entry>(.*?)</entry>").unwrap();
            let video_id_re = Regex::new(r"<yt:videoId>(.*?)</yt:videoId>").unwrap();
            let title_re = Regex::new(r"<title>(.*?)</title>").unwrap();
            let author_re = Regex::new(r"<name>(.*?)</name>").unwrap();
            let published_re = Regex::new(r"<published>(.*?)</published>").unwrap();

            for entry_match in entry_re.captures_iter(&rss_content).take(10) {
                let entry = &entry_match[1];

                if let (Some(video_id), Some(title), Some(author), Some(published)) = (
                    video_id_re.captures(entry).map(|c| c[1].to_string()),
                    title_re.captures(entry).map(|c| c[1].to_string()),
                    author_re.captures(entry).map(|c| c[1].to_string()),
                    published_re.captures(entry).map(|c| c[1].to_string()),
                ) {
                    let video = ExtractedVideo {
                        id: video_id.clone(),
                        title: html_escape::decode_html_entities(&title).to_string(),
                        platform: Platform::YouTube,
                        author,
                        url: format!("https://www.youtube.com/watch?v={}", video_id),
                        description: "Real video from YouTube RSS feed".to_string(),
                        view_count: None,
                        like_count: None,
                        duration: None,
                        upload_date: published.split('T').next().unwrap_or("").to_string(),
                        thumbnail: Some(format!(
                            "https://img.youtube.com/vi/{}/maxresdefault.jpg",
                            video_id
                        )),
                        tags: Vec::new(),
                        data_source: "youtube_rss_feed".to_string(),
                        legal_compliance: "Official RSS - Fair Use Academic Research".to_string(),
                        extraction_method: "RSS feed parsing - No downloads".to_string(),
                        extracted_at: Utc::now(),
                    };
                    videos.push(video);
                }
            }
        }

        Ok(videos)
    }

    fn platform(&self) -> Platform {
        Platform::YouTube
    }
}

// Douyin Extractor
pub struct DouyinExtractor {
    client: Client,
}

impl DouyinExtractor {
    pub fn new(client: Client) -> Self {
        Self { client }
    }
}

#[async_trait]
impl DataExtractor for DouyinExtractor {
    async fn extract(&self) -> Result<Vec<ExtractedVideo>> {
        let mut videos = Vec::new();

        let urls = vec![
            "https://www.douyin.com/discover",
            "https://www.douyin.com/hot",
        ];

        for url in urls {
            match self.client.get(url).send().await {
                Ok(response) if response.status().is_success() => {
                    let content = response.text().await?;

                    // Try to extract video data from page
                    let patterns = vec![
                        r#""aweme_id":"(\d+)".*?"desc":"([^"]*)".*?"nickname":"([^"]*)"#,
                        r#""id":"(\d+)".*?"title":"([^"]*)".*?"author":"([^"]*)"#,
                    ];

                    for pattern in patterns {
                        let re = Regex::new(pattern).unwrap();

                        for (i, cap) in re.captures_iter(&content).enumerate() {
                            if i >= 3 || videos.len() >= 5 {
                                break;
                            }

                            let video_id = cap[1].to_string();
                            let desc = cap[2].to_string();
                            let author = cap[3].to_string();

                            let video = ExtractedVideo {
                                id: video_id.clone(),
                                title: if desc.len() > 100 {
                                    format!("{}...", &desc[..100])
                                } else if desc.is_empty() {
                                    format!("Douyin Video {}", i + 1)
                                } else {
                                    desc.clone()
                                },
                                platform: Platform::Douyin,
                                author: if author.is_empty() {
                                    "Unknown".to_string()
                                } else {
                                    author
                                },
                                url: format!("https://www.douyin.com/video/{}", video_id),
                                description: if desc.is_empty() {
                                    "Real content from Douyin".to_string()
                                } else {
                                    desc
                                },
                                view_count: None,
                                like_count: None,
                                duration: None,
                                upload_date: Utc::now().format("%Y-%m-%d").to_string(),
                                thumbnail: None,
                                tags: Vec::new(),
                                data_source: "douyin_public_page".to_string(),
                                legal_compliance: "Public Data - Fair Use Academic Research".to_string(),
                                extraction_method: "Page streaming analysis - No downloads".to_string(),
                                extracted_at: Utc::now(),
                            };
                            videos.push(video);
                        }
                    }

                    if !videos.is_empty() {
                        break;
                    }
                }
                _ => continue,
            }
        }

        Ok(videos)
    }

    fn platform(&self) -> Platform {
        Platform::Douyin
    }
}

// Add html_escape module if not available
mod html_escape {
    pub fn decode_html_entities(s: &str) -> String {
        s.replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&#39;", "'")
            .replace("&nbsp;", " ")
    }
}