use crate::{
    error::VideoRssError,
    types::*,
    utils::{extract_video_id, parse_duration},
    Result,
};
use chrono::{DateTime, Utc};
use regex::Regex;
use reqwest::{Client, header::HeaderMap};
use serde_json::Value;
use std::collections::HashMap;
use tracing::{debug, info, warn};

pub struct BilibiliClient {
    client: Client,
    credentials: Option<BilibiliCredentials>,
}

impl BilibiliClient {
    pub fn new(credentials: Option<BilibiliCredentials>) -> Result<Self> {
        let mut headers = HeaderMap::new();
        headers.insert("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36".parse().unwrap());
        headers.insert("Accept", "application/json, text/html".parse().unwrap());
        headers.insert("Accept-Language", "zh-CN,zh;q=0.9,en;q=0.8".parse().unwrap());
        headers.insert("Referer", "https://www.bilibili.com".parse().unwrap());

        let mut client_builder = Client::builder()
            .default_headers(headers)
            .timeout(std::time::Duration::from_secs(30));

        // Add cookies if credentials are provided
        if let Some(ref creds) = credentials {
            let cookie_jar = reqwest::cookie::Jar::default();
            let url = "https://www.bilibili.com".parse::<url::Url>()?;

            cookie_jar.add_cookie_str(&format!("SESSDATA={}", creds.sessdata), &url);
            cookie_jar.add_cookie_str(&format!("bili_jct={}", creds.bili_jct), &url);
            cookie_jar.add_cookie_str(&format!("buvid3={}", creds.buvid3), &url);

            client_builder = client_builder.cookie_provider(std::sync::Arc::new(cookie_jar));
            info!("Initialized Bilibili client with authentication credentials");
        } else {
            info!("Initialized Bilibili client without authentication");
        }

        let client = client_builder.build()?;

        Ok(Self {
            client,
            credentials,
        })
    }

    pub async fn fetch_recommendations(&self, options: &FetchOptions) -> Result<Vec<VideoInfo>> {
        info!("Fetching {} Bilibili recommendations", options.limit);

        if options.personalized && self.credentials.is_none() {
            warn!("Personalized recommendations requested but no credentials provided");
            return Err(VideoRssError::InvalidCredentials);
        }

        let api_url = if options.personalized {
            "https://api.bilibili.com/x/web-interface/index/top/rcmd"
        } else {
            "https://api.bilibili.com/x/web-interface/ranking/v2"
        };

        let mut params = HashMap::new();
        params.insert("rid", "0"); // All categories
        params.insert("day", "3"); // Last 3 days
        params.insert("arc_type", "0"); // All types

        debug!("Making request to Bilibili API: {}", api_url);

        let response = self.client
            .get(api_url)
            .query(&params)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(VideoRssError::Parse(format!(
                "Failed to fetch Bilibili API: status {}",
                response.status()
            )));
        }

        let json: Value = response.json().await?;

        if json["code"].as_i64() != Some(0) {
            let error_msg = json["message"].as_str().unwrap_or("Unknown API error");
            return Err(VideoRssError::ContentAnalysis(error_msg.to_string()));
        }

        let data = &json["data"];
        let videos_array = if options.personalized {
            &data["item"]
        } else {
            &data["list"]
        };

        let videos = videos_array
            .as_array()
            .ok_or_else(|| VideoRssError::Parse("No videos array found".to_string()))?
            .iter()
            .take(options.limit)
            .filter_map(|video| self.parse_video_info(video).ok())
            .collect();

        info!("Successfully fetched {} videos", videos.len());
        Ok(videos)
    }

    pub async fn fetch_video_info(&self, video_url: &str) -> Result<VideoInfo> {
        let video_id = extract_video_id(video_url, Platform::Bilibili)
            .ok_or_else(|| VideoRssError::Parse("Invalid Bilibili URL".to_string()))?;

        debug!("Fetching video info for ID: {}", video_id);

        let api_url = "https://api.bilibili.com/x/web-interface/view";
        let response = self.client
            .get(api_url)
            .query(&[("bvid", &video_id)])
            .send()
            .await?;

        let json: Value = response.json().await?;

        if json["code"].as_i64() != Some(0) {
            return Err(VideoRssError::VideoNotFound(video_id));
        }

        let video_data = &json["data"];
        self.parse_video_info(video_data)
    }

    fn parse_video_info(&self, video: &Value) -> Result<VideoInfo> {
        let id = video["bvid"]
            .as_str()
            .or_else(|| video["aid"].as_i64().map(|aid| format!("av{}", aid)).as_deref())
            .ok_or_else(|| VideoRssError::Parse("Missing video ID".to_string()))?
            .to_string();

        let title = video["title"]
            .as_str()
            .unwrap_or("Untitled")
            .to_string();

        let description = video["desc"]
            .as_str()
            .unwrap_or("")
            .to_string();

        let url = format!("https://www.bilibili.com/video/{}", id);

        let author = video["owner"]["name"]
            .as_str()
            .unwrap_or("Unknown")
            .to_string();

        let upload_timestamp = video["pubdate"]
            .as_i64()
            .unwrap_or(0);
        let upload_date = DateTime::from_timestamp(upload_timestamp, 0)
            .unwrap_or_else(|| Utc::now());

        let duration = video["duration"]
            .as_u64()
            .or_else(|| {
                video["duration"]
                    .as_str()
                    .and_then(|s| parse_duration(s))
            });

        let stats = &video["stat"];
        let view_count = stats["view"].as_u64().unwrap_or(0);
        let like_count = stats["like"].as_u64().unwrap_or(0);
        let comment_count = stats["reply"].as_u64().unwrap_or(0);

        let tags = video["tag"]
            .as_array()
            .map(|tags| {
                tags.iter()
                    .filter_map(|tag| tag["tag_name"].as_str())
                    .map(|s| s.to_string())
                    .collect()
            })
            .unwrap_or_default();

        let thumbnail_url = video["pic"]
            .as_str()
            .map(|s| {
                if s.starts_with("//") {
                    format!("https:{}", s)
                } else {
                    s.to_string()
                }
            });

        Ok(VideoInfo {
            id,
            title,
            description,
            url,
            author,
            upload_date,
            duration,
            view_count,
            like_count,
            comment_count,
            tags,
            thumbnail_url,
            platform: Platform::Bilibili,
            transcription: None,
        })
    }

    pub async fn get_video_subtitles(&self, video_id: &str) -> Result<Option<String>> {
        debug!("Fetching subtitles for video: {}", video_id);

        let subtitle_url = "https://api.bilibili.com/x/web-interface/view/subtitle";
        let response = self.client
            .get(subtitle_url)
            .query(&[("bvid", video_id)])
            .send()
            .await?;

        let json: Value = response.json().await?;

        if json["code"].as_i64() != Some(0) {
            return Ok(None);
        }

        let data = &json["data"];
        let subtitles = data["subtitles"].as_array();

        if let Some(subtitles) = subtitles {
            if let Some(subtitle) = subtitles.first() {
                if let Some(subtitle_url) = subtitle["subtitle_url"].as_str() {
                    let full_url = if subtitle_url.starts_with("//") {
                        format!("https:{}", subtitle_url)
                    } else {
                        subtitle_url.to_string()
                    };

                    debug!("Fetching subtitle content from: {}", full_url);

                    let subtitle_response = self.client.get(&full_url).send().await?;
                    let subtitle_data: Value = subtitle_response.json().await?;

                    if let Some(body) = subtitle_data["body"].as_array() {
                        let text: String = body
                            .iter()
                            .filter_map(|item| item["content"].as_str())
                            .collect::<Vec<_>>()
                            .join(" ");

                        return Ok(Some(text));
                    }
                }
            }
        }

        Ok(None)
    }
}