use crate::{
    types::*,
    utils::{clean_html, extract_video_id},
    Result,
};
use regex::Regex;
use reqwest::Client;
use scraper::{Html, Selector};
use serde_json::Value;
use std::collections::HashMap;
use tracing::{debug, info, warn};

pub struct ContentAnalyzer {
    client: Client,
}

impl ContentAnalyzer {
    pub fn new() -> Result<Self> {
        let client = Client::builder()
            .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
            .timeout(std::time::Duration::from_secs(30))
            .build()?;

        Ok(Self { client })
    }

    pub async fn analyze_video(&self, video: &VideoInfo) -> Result<ContentSummary> {
        info!("Analyzing video content: {}", video.title);

        let keywords = self.extract_keywords(&video.title, &video.description, &video.tags);
        let content_type = self.classify_content(video);
        let sentiment = self.analyze_sentiment(&video.title, &video.description);

        let ai_summary = self.generate_content_summary(video).await?;

        Ok(ContentSummary {
            ai_summary,
            keywords,
            sentiment,
            content_type,
        })
    }

    pub async fn batch_analyze(&self, videos: &[VideoInfo]) -> Result<Vec<AnalysisResult>> {
        info!("Batch analyzing {} videos", videos.len());

        let mut results = Vec::new();

        for video in videos {
            let start_time = std::time::Instant::now();

            match self.analyze_video(video).await {
                Ok(summary) => {
                    let processing_time = start_time.elapsed().as_millis() as u64;

                    results.push(AnalysisResult {
                        video: video.clone(),
                        summary,
                        processing_time_ms: processing_time,
                    });
                }
                Err(e) => {
                    warn!("Failed to analyze video {}: {}", video.id, e);
                    // Continue with other videos
                }
            }

            // Rate limiting
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }

        info!("Completed batch analysis: {} results", results.len());
        Ok(results)
    }

    async fn generate_content_summary(&self, video: &VideoInfo) -> Result<String> {
        // Try to get actual video content
        if let Ok(content_data) = self.fetch_video_page_content(&video.url).await {
            return Ok(self.analyze_page_content(&content_data, video));
        }

        // Fallback to metadata-based summary
        Ok(self.generate_metadata_summary(video))
    }

    async fn fetch_video_page_content(&self, url: &str) -> Result<String> {
        debug!("Fetching video page content from: {}", url);

        let response = self.client.get(url).send().await?;

        if !response.status().is_success() {
            return Err(crate::error::VideoRssError::Http(reqwest::Error::from(
                reqwest::ErrorKind::Request,
            )));
        }

        let content = response.text().await?;
        Ok(content)
    }

    fn analyze_page_content(&self, html: &str, video: &VideoInfo) -> String {
        let document = Html::parse_document(html);
        let mut summary_parts = Vec::new();

        // Try to extract meaningful content from the page
        if let Ok(selector) = Selector::parse("meta[name='description']") {
            if let Some(element) = document.select(&selector).next() {
                if let Some(content) = element.value().attr("content") {
                    if !content.is_empty() && content.len() > 10 {
                        summary_parts.push(format!("📌 核心内容：{}", content));
                    }
                }
            }
        }

        // Extract JSON-LD data if available
        if let Ok(selector) = Selector::parse("script[type='application/ld+json']") {
            for element in document.select(&selector) {
                if let Some(json_text) = element.text().next() {
                    if let Ok(json_data) = serde_json::from_str::<Value>(json_text) {
                        if let Some(description) = json_data["description"].as_str() {
                            summary_parts.push(format!("💬 详细描述：{}", description));
                            break;
                        }
                    }
                }
            }
        }

        // Add engagement metrics
        if video.view_count > 0 {
            summary_parts.push(format!(
                "[STATS] 播放{:,}次，互动良好",
                video.view_count
            ));
        }

        // Add hashtags/topics
        if !video.tags.is_empty() {
            let topics = video.tags
                .iter()
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ");
            summary_parts.push(format!("🏷️ 主要话题：{}", topics));
        }

        if summary_parts.is_empty() {
            self.generate_metadata_summary(video)
        } else {
            summary_parts.join(" | ")
        }
    }

    fn generate_metadata_summary(&self, video: &VideoInfo) -> String {
        let mut summary_parts = Vec::new();

        // Content type classification
        let title_lower = video.title.to_lowercase();
        if title_lower.contains("合集") || title_lower.contains("集合") || title_lower.contains("精选") {
            summary_parts.push(format!("📦 内容合集：{}", video.title));
        } else if title_lower.contains("教程") || title_lower.contains("攻略") || title_lower.contains("方法") {
            summary_parts.push(format!("📚 实用教程：{}", video.title));
        } else if title_lower.contains("音乐") || title_lower.contains("歌曲") || title_lower.contains("歌单") {
            summary_parts.push(format!("🎵 音乐内容：{}", video.title));
        } else if title_lower.contains("游戏") || title_lower.contains("实况") || title_lower.contains("解说") {
            summary_parts.push(format!("🎮 游戏内容：{}", video.title));
        } else {
            summary_parts.push(format!("📹 视频内容：{}", video.title));
        }

        // Duration and engagement
        if let Some(duration) = video.duration {
            let minutes = duration / 60;
            summary_parts.push(format!("⏱️ 时长{}分钟，已有{:,}人观看", minutes, video.view_count));
        } else if video.view_count > 0 {
            summary_parts.push(format!("[STATS] 已有{:,}人观看", video.view_count));
        }

        // Description summary
        if !video.description.is_empty() && video.description != "-" && video.description.len() > 10 {
            let desc_summary = if video.description.len() > 80 {
                format!("{}...", &video.description[..80])
            } else {
                video.description.clone()
            };
            summary_parts.push(format!("📝 简介：{}", desc_summary));
        }

        // Tags
        if !video.tags.is_empty() {
            let main_tags = video.tags
                .iter()
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ");
            summary_parts.push(format!("🏷️ 相关话题：{}", main_tags));
        }

        // Creator info
        if !video.author.is_empty() {
            summary_parts.push(format!("👤 创作者：{}", video.author));
        }

        summary_parts.join(" | ")
    }

    fn extract_keywords(&self, title: &str, description: &str, tags: &[String]) -> Vec<String> {
        let mut keywords = Vec::new();

        // Extract from tags first (highest relevance)
        keywords.extend(tags.iter().take(5).cloned());

        // Extract from title
        keywords.extend(self.extract_keywords_from_text(title, 3));

        // Extract from description
        if !description.is_empty() {
            keywords.extend(self.extract_keywords_from_text(description, 5));
        }

        // Remove duplicates and limit
        keywords.sort();
        keywords.dedup();
        keywords.truncate(10);

        keywords
    }

    fn extract_keywords_from_text(&self, text: &str, limit: usize) -> Vec<String> {
        // Simple keyword extraction based on Chinese text patterns
        let chinese_word_pattern = Regex::new(r"[\u4e00-\u9fff]{2,}").unwrap();
        let mut keywords: Vec<String> = chinese_word_pattern
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .filter(|word| word.len() >= 2 && word.len() <= 6) // Reasonable word length
            .collect();

        // Remove common stop words
        let stop_words = ["这个", "那个", "可以", "我们", "他们", "什么", "怎么", "非常", "真的", "还是"];
        keywords.retain(|word| !stop_words.contains(&word.as_str()));

        keywords.truncate(limit);
        keywords
    }

    fn classify_content(&self, video: &VideoInfo) -> ContentType {
        let title_lower = video.title.to_lowercase();
        let desc_lower = video.description.to_lowercase();

        // Check tags first for more accurate classification
        for tag in &video.tags {
            let tag_lower = tag.to_lowercase();
            match tag_lower.as_str() {
                t if t.contains("教程") || t.contains("学习") || t.contains("教育") => {
                    return ContentType::Educational;
                }
                t if t.contains("游戏") || t.contains("电竞") => {
                    return ContentType::Gaming;
                }
                t if t.contains("音乐") || t.contains("歌曲") => {
                    return ContentType::Music;
                }
                t if t.contains("科技") || t.contains("技术") => {
                    return ContentType::Technology;
                }
                t if t.contains("新闻") || t.contains("时事") => {
                    return ContentType::News;
                }
                _ => {}
            }
        }

        // Fallback to title and description analysis
        let combined_text = format!("{} {}", title_lower, desc_lower);

        if combined_text.contains("教程") || combined_text.contains("学习") || combined_text.contains("教育") {
            ContentType::Educational
        } else if combined_text.contains("游戏") || combined_text.contains("电竞") || combined_text.contains("实况") {
            ContentType::Gaming
        } else if combined_text.contains("音乐") || combined_text.contains("歌曲") || combined_text.contains("MV") {
            ContentType::Music
        } else if combined_text.contains("科技") || combined_text.contains("技术") || combined_text.contains("编程") {
            ContentType::Technology
        } else if combined_text.contains("新闻") || combined_text.contains("时事") || combined_text.contains("报道") {
            ContentType::News
        } else {
            ContentType::Entertainment // Default
        }
    }

    fn analyze_sentiment(&self, title: &str, description: &str) -> Option<f32> {
        // Simple sentiment analysis based on keyword presence
        let positive_words = ["好", "棒", "优秀", "精彩", "完美", "推荐", "喜欢", "爱", "美", "帅"];
        let negative_words = ["差", "烂", "糟糕", "失望", "讨厌", "垃圾", "无聊", "坏", "错"];

        let combined_text = format!("{} {}", title, description).to_lowercase();

        let positive_count = positive_words
            .iter()
            .map(|word| combined_text.matches(word).count())
            .sum::<usize>() as f32;

        let negative_count = negative_words
            .iter()
            .map(|word| combined_text.matches(word).count())
            .sum::<usize>() as f32;

        if positive_count + negative_count == 0.0 {
            return None; // Neutral or insufficient data
        }

        let sentiment = (positive_count - negative_count) / (positive_count + negative_count);
        Some(sentiment.clamp(-1.0, 1.0))
    }
}

impl Default for ContentAnalyzer {
    fn default() -> Self {
        Self::new().unwrap()
    }
}