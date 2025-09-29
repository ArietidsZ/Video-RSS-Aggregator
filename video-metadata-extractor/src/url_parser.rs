use regex::Regex;
use url::Url;
use crate::{Platform, Result, ExtractorError};
use std::collections::HashMap;

/// Intelligent URL parser with pattern recognition and validation
pub struct UrlParser {
    patterns: HashMap<Platform, Vec<Regex>>,
}

impl UrlParser {
    pub fn new() -> Self {
        let mut patterns = HashMap::new();

        // YouTube patterns
        patterns.insert(Platform::YouTube, vec![
            Regex::new(r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/)([a-zA-Z0-9_-]{11})").unwrap(),
            Regex::new(r"youtube\.com/shorts/([a-zA-Z0-9_-]{11})").unwrap(),
            Regex::new(r"youtube\.com/live/([a-zA-Z0-9_-]{11})").unwrap(),
        ]);

        // Bilibili patterns
        patterns.insert(Platform::Bilibili, vec![
            Regex::new(r"bilibili\.com/video/(BV[a-zA-Z0-9]+)").unwrap(),
            Regex::new(r"bilibili\.com/video/(av\d+)").unwrap(),
            Regex::new(r"b23\.tv/([a-zA-Z0-9]+)").unwrap(),
            Regex::new(r"bilibili\.com/bangumi/play/(ep\d+)").unwrap(),
        ]);

        // Douyin patterns
        patterns.insert(Platform::Douyin, vec![
            Regex::new(r"douyin\.com/video/(\d+)").unwrap(),
            Regex::new(r"douyin\.com/user/[^/]+\?modal_id=(\d+)").unwrap(),
            Regex::new(r"v\.douyin\.com/([a-zA-Z0-9]+)").unwrap(),
        ]);

        // TikTok patterns
        patterns.insert(Platform::TikTok, vec![
            Regex::new(r"tiktok\.com/@[^/]+/video/(\d+)").unwrap(),
            Regex::new(r"tiktok\.com/v/(\d+)").unwrap(),
            Regex::new(r"vm\.tiktok\.com/([a-zA-Z0-9]+)").unwrap(),
        ]);

        // Kuaishou patterns
        patterns.insert(Platform::Kuaishou, vec![
            Regex::new(r"kuaishou\.com/short-video/([a-zA-Z0-9]+)").unwrap(),
            Regex::new(r"kuaishou\.com/f/([a-zA-Z0-9]+)").unwrap(),
            Regex::new(r"kwai\.com/([a-zA-Z0-9]+)").unwrap(),
        ]);

        Self { patterns }
    }

    /// Parse and validate a video URL
    pub fn parse(&self, url: &str) -> Result<ParsedUrl> {
        // Clean and normalize URL
        let cleaned_url = self.clean_url(url)?;

        // Parse URL
        let parsed = Url::parse(&cleaned_url)
            .map_err(|e| ExtractorError::InvalidUrl(format!("Failed to parse URL: {}", e)))?;

        // Detect platform
        let platform = Platform::from_url(&cleaned_url);

        // Extract video ID
        let video_id = self.extract_video_id(&cleaned_url, &platform)?;

        // Extract additional parameters
        let params = self.extract_params(&parsed);

        Ok(ParsedUrl {
            original: url.to_string(),
            normalized: cleaned_url,
            platform,
            video_id,
            params,
            host: parsed.host_str().unwrap_or("").to_string(),
            path: parsed.path().to_string(),
        })
    }

    /// Clean and normalize URL
    fn clean_url(&self, url: &str) -> Result<String> {
        let mut cleaned = url.trim().to_string();

        // Add protocol if missing
        if !cleaned.starts_with("http://") && !cleaned.starts_with("https://") {
            cleaned = format!("https://{}", cleaned);
        }

        // Remove tracking parameters
        if let Ok(mut parsed) = Url::parse(&cleaned) {
            let mut params: Vec<(String, String)> = parsed.query_pairs()
                .filter(|(k, _)| !self.is_tracking_param(k))
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();

            // Keep essential parameters
            params.retain(|(k, _)| self.is_essential_param(k));

            parsed.set_query(None);
            if !params.is_empty() {
                let query = params.iter()
                    .map(|(k, v)| format!("{}={}", k, v))
                    .collect::<Vec<_>>()
                    .join("&");
                parsed.set_query(Some(&query));
            }

            cleaned = parsed.to_string();
        }

        Ok(cleaned)
    }

    /// Extract video ID based on platform patterns
    fn extract_video_id(&self, url: &str, platform: &Platform) -> Result<String> {
        if let Some(patterns) = self.patterns.get(platform) {
            for pattern in patterns {
                if let Some(captures) = pattern.captures(url) {
                    if let Some(id) = captures.get(1) {
                        return Ok(id.as_str().to_string());
                    }
                }
            }
        }

        // Fallback: try to extract from URL path
        if let Ok(parsed) = Url::parse(url) {
            let path_segments: Vec<&str> = parsed.path_segments()
                .map(|segments| segments.collect())
                .unwrap_or_default();

            if !path_segments.is_empty() {
                return Ok(path_segments.last().unwrap().to_string());
            }
        }

        Err(ExtractorError::InvalidUrl(format!("Could not extract video ID from URL: {}", url)))
    }

    /// Extract URL parameters
    fn extract_params(&self, url: &Url) -> HashMap<String, String> {
        url.query_pairs()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    /// Check if parameter is tracking-related
    fn is_tracking_param(&self, param: &str) -> bool {
        matches!(param,
            "utm_source" | "utm_medium" | "utm_campaign" | "utm_term" | "utm_content" |
            "fbclid" | "gclid" | "dclid" | "ref" | "referrer" | "share_source" |
            "share_medium" | "share_campaign" | "share_id"
        )
    }

    /// Check if parameter is essential for video playback
    fn is_essential_param(&self, param: &str) -> bool {
        matches!(param,
            "v" | "video_id" | "id" | "t" | "time" | "start" | "end" |
            "list" | "index" | "quality" | "lang" | "cc" | "subtitle"
        )
    }

    /// Validate URL accessibility
    pub async fn validate(&self, url: &str) -> Result<bool> {
        let parsed = self.parse(url)?;

        // Check if platform is supported
        match parsed.platform {
            Platform::Unknown(_) => {
                return Err(ExtractorError::UnsupportedPlatform(
                    format!("Platform not supported for URL: {}", url)
                ));
            }
            _ => {}
        }

        // Quick HEAD request to check if URL is accessible
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .map_err(|e| ExtractorError::Network(e))?;

        let response = client.head(&parsed.normalized)
            .send()
            .await
            .map_err(|e| ExtractorError::Network(e))?;

        Ok(response.status().is_success())
    }

    /// Expand shortened URLs
    pub async fn expand_short_url(&self, short_url: &str) -> Result<String> {
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .map_err(|e| ExtractorError::Network(e))?;

        let response = client.get(short_url)
            .send()
            .await
            .map_err(|e| ExtractorError::Network(e))?;

        if let Some(location) = response.headers().get("location") {
            let expanded = location.to_str()
                .map_err(|_| ExtractorError::InvalidUrl("Invalid redirect URL".to_string()))?;
            return Ok(expanded.to_string());
        }

        Ok(short_url.to_string())
    }

    /// Extract playlist information if present
    pub fn extract_playlist_info(&self, url: &str) -> Option<PlaylistInfo> {
        let parsed = Url::parse(url).ok()?;
        let params: HashMap<String, String> = parsed.query_pairs()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();

        // YouTube playlist
        if url.contains("youtube.com") && params.contains_key("list") {
            return Some(PlaylistInfo {
                platform: Platform::YouTube,
                playlist_id: params.get("list")?.clone(),
                index: params.get("index").and_then(|i| i.parse().ok()),
            });
        }

        // Bilibili playlist
        if url.contains("bilibili.com") {
            if let Some(p) = params.get("p") {
                return Some(PlaylistInfo {
                    platform: Platform::Bilibili,
                    playlist_id: String::new(),
                    index: p.parse().ok(),
                });
            }
        }

        None
    }
}

#[derive(Debug, Clone)]
pub struct ParsedUrl {
    pub original: String,
    pub normalized: String,
    pub platform: Platform,
    pub video_id: String,
    pub params: HashMap<String, String>,
    pub host: String,
    pub path: String,
}

#[derive(Debug, Clone)]
pub struct PlaylistInfo {
    pub platform: Platform,
    pub playlist_id: String,
    pub index: Option<usize>,
}

impl Default for UrlParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_youtube_url_parsing() {
        let parser = UrlParser::new();

        let urls = vec![
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "youtube.com/watch?v=dQw4w9WgXcQ&t=42s",
            "https://www.youtube.com/shorts/abc123defgh",
        ];

        for url in urls {
            let parsed = parser.parse(url).unwrap();
            assert_eq!(parsed.platform, Platform::YouTube);
            assert!(!parsed.video_id.is_empty());
        }
    }

    #[tokio::test]
    async fn test_bilibili_url_parsing() {
        let parser = UrlParser::new();

        let urls = vec![
            "https://www.bilibili.com/video/BV1234567890",
            "https://www.bilibili.com/video/av12345678",
            "https://b23.tv/abcdef",
        ];

        for url in urls {
            let parsed = parser.parse(url).unwrap();
            assert_eq!(parsed.platform, Platform::Bilibili);
            assert!(!parsed.video_id.is_empty());
        }
    }

    #[test]
    fn test_tracking_param_removal() {
        let parser = UrlParser::new();

        let url = "https://www.youtube.com/watch?v=test&utm_source=share&fbclid=123";
        let cleaned = parser.clean_url(url).unwrap();

        assert!(cleaned.contains("v=test"));
        assert!(!cleaned.contains("utm_source"));
        assert!(!cleaned.contains("fbclid"));
    }
}