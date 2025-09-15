use crate::{
    types::*,
    utils::{format_duration, sanitize_xml, truncate_text},
    Result,
};
use chrono::{DateTime, Utc};
use rss::{Channel, ChannelBuilder, Item, ItemBuilder};
use std::collections::HashMap;
use tracing::{debug, info};

pub struct RssGenerator {
    config: RssConfig,
}

impl RssGenerator {
    pub fn new(config: RssConfig) -> Self {
        Self { config }
    }

    pub fn generate_feed(&self, videos: &[VideoInfo]) -> Result<String> {
        info!("Generating RSS feed for {} videos", videos.len());

        let mut channel = ChannelBuilder::default()
            .title(&self.config.title)
            .description(&self.config.description)
            .link(&self.config.link)
            .language(&self.config.language)
            .generator(&self.config.generator)
            .last_build_date(Some(Utc::now().to_rfc2822()))
            .build();

        let items: Vec<Item> = videos
            .iter()
            .map(|video| self.create_rss_item(video))
            .collect::<Result<Vec<_>>>()?;

        channel.set_items(items);

        debug!("RSS feed generated successfully");
        Ok(channel.to_string())
    }

    fn create_rss_item(&self, video: &VideoInfo) -> Result<Item> {
        let title = sanitize_xml(&video.title);
        let description = self.generate_item_description(video);
        let link = video.url.clone();
        let pub_date = video.upload_date.to_rfc2822();
        let guid = video.url.clone();

        // Create categories from tags
        let categories: Vec<rss::Category> = video
            .tags
            .iter()
            .take(5) // Limit to 5 tags
            .map(|tag| rss::Category {
                name: sanitize_xml(tag),
                domain: None,
            })
            .collect();

        let item = ItemBuilder::default()
            .title(Some(title))
            .description(Some(description))
            .link(Some(link))
            .pub_date(Some(pub_date))
            .guid(Some(rss::Guid {
                value: guid,
                permalink: true,
            }))
            .author(Some(sanitize_xml(&video.author)))
            .categories(categories)
            .build();

        Ok(item)
    }

    fn generate_item_description(&self, video: &VideoInfo) -> String {
        let mut parts = Vec::new();

        // Basic video info
        parts.push(format!("<p><strong>👤 作者：</strong>{}</p>", sanitize_xml(&video.author)));
        parts.push(format!(
            "<p><strong>👁️ 观看：</strong>{} | <strong>👍 点赞：</strong>{}</p>",
            format_number(video.view_count), format_number(video.like_count)
        ));

        if let Some(duration) = video.duration {
            let formatted_duration = format_duration(duration);
            parts.push(format!("<p><strong>⏱️ 时长：</strong>{}</p>", formatted_duration));
        }

        // Tags
        if !video.tags.is_empty() {
            let tags_str = video.tags
                .iter()
                .take(5)
                .map(|tag| sanitize_xml(tag))
                .collect::<Vec<_>>()
                .join(", ");
            parts.push(format!("<p><strong>🏷️ 标签：</strong>{}</p>", tags_str));
        }

        // Description
        if !video.description.is_empty() {
            let clean_desc = truncate_text(&sanitize_xml(&video.description), 500);
            parts.push(format!("<p><strong>📝 简介：</strong>{}</p>", clean_desc));
        }

        // Transcription data if available
        if let Some(ref transcription) = video.transcription {
            parts.extend(self.format_transcription_content(transcription));
        }

        // Platform and source info
        parts.push(format!(
            "<p><strong>[SOURCE] 数据来源：</strong>{} - Real Data Only</p>",
            video.platform.as_str().to_uppercase()
        ));

        parts.join("\n                ")
    }

    fn format_transcription_content(&self, transcription: &TranscriptionData) -> Vec<String> {
        let mut parts = Vec::new();

        // Add paragraph summary
        if !transcription.paragraph_summary.is_empty() {
            parts.push(format!(
                "<p><strong>📄 完整段落摘要：</strong>{}</p>",
                sanitize_xml(&transcription.paragraph_summary)
            ));
        }

        // Add sentence subtitle
        if !transcription.sentence_subtitle.is_empty() {
            parts.push(format!(
                "<p><strong>📝 字幕句子：</strong>{}</p>",
                sanitize_xml(&transcription.sentence_subtitle)
            ));
        }

        // Add truncated transcript preview
        if !transcription.full_transcript.is_empty() {
            let transcript_preview = truncate_text(&transcription.full_transcript, 200);
            parts.push(format!(
                "<p><strong>📰 转录内容：</strong>{}</p>",
                sanitize_xml(&transcript_preview)
            ));
        }

        // Add model information
        if !transcription.model_info.transcriber.is_empty() || !transcription.model_info.summarizer.is_empty() {
            parts.push(format!(
                "<p><strong>🤖 AI模型信息：</strong>状态: {:?} | 转录器: {} | 摘要器: {}</p>",
                transcription.status,
                sanitize_xml(&transcription.model_info.transcriber),
                sanitize_xml(&transcription.model_info.summarizer)
            ));
        }

        parts
    }

    pub fn generate_feed_with_summary(&self, videos: &[VideoInfo], include_ai_summary: bool) -> Result<String> {
        info!("Generating RSS feed with AI summary for {} videos", videos.len());

        let enhanced_videos: Vec<VideoInfo> = if include_ai_summary {
            videos
                .iter()
                .map(|video| {
                    let mut enhanced_video = video.clone();
                    // Generate AI summary if not present
                    if enhanced_video.transcription.is_none() {
                        enhanced_video.transcription = Some(TranscriptionData {
                            paragraph_summary: self.generate_ai_summary(video),
                            sentence_subtitle: self.generate_sentence_subtitle(video),
                            full_transcript: String::new(),
                            status: TranscriptionStatus::Success,
                            model_info: ModelInfo {
                                transcriber: "rust-analyzer".to_string(),
                                summarizer: "rust-summarizer".to_string(),
                            },
                            source_types: vec!["metadata".to_string()],
                        });
                    }
                    enhanced_video
                })
                .collect()
        } else {
            videos.to_vec()
        };

        self.generate_feed(&enhanced_videos)
    }

    fn generate_ai_summary(&self, video: &VideoInfo) -> String {
        // Simple content-based summary generation
        let mut summary_parts = Vec::new();

        // Analyze title for content type
        let title_lower = video.title.to_lowercase();
        if title_lower.contains("教程") || title_lower.contains("攻略") || title_lower.contains("方法") {
            summary_parts.push(format!("📚 实用教程：{}", video.title));
        } else if title_lower.contains("音乐") || title_lower.contains("歌曲") {
            summary_parts.push(format!("🎵 音乐内容：{}", video.title));
        } else if title_lower.contains("游戏") || title_lower.contains("实况") {
            summary_parts.push(format!("🎮 游戏内容：{}", video.title));
        } else {
            summary_parts.push(format!("📹 视频内容：{}", video.title));
        }

        // Add engagement metrics
        if video.view_count > 0 {
            summary_parts.push(format!("已有{}人观看", format_number(video.view_count)));
        }

        // Add description summary if available
        if !video.description.is_empty() && video.description.len() > 10 {
            let desc_summary = truncate_text(&video.description, 80);
            summary_parts.push(format!("📝 简介：{}", desc_summary));
        }

        // Add tags if available
        if !video.tags.is_empty() {
            let main_tags = video.tags
                .iter()
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ");
            summary_parts.push(format!("🏷️ 相关话题：{}", main_tags));
        }

        summary_parts.join(" | ")
    }

    fn generate_sentence_subtitle(&self, video: &VideoInfo) -> String {
        // Generate a concise one-sentence summary
        if !video.description.is_empty() {
            let first_sentence = video.description
                .split('。')
                .next()
                .unwrap_or(&video.title)
                .trim();

            if first_sentence.len() > 5 {
                return truncate_text(first_sentence, 50);
            }
        }

        // Fallback to title-based sentence
        truncate_text(&video.title, 50)
    }
}

fn format_number(num: u64) -> String {
    if num >= 100_000_000 {
        format!("{:.1}亿", num as f64 / 100_000_000.0)
    } else if num >= 10_000 {
        format!("{:.1}万", num as f64 / 10_000.0)
    } else {
        format!("{}", num)
    }
}

impl Default for RssGenerator {
    fn default() -> Self {
        Self::new(RssConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_video() -> VideoInfo {
        VideoInfo {
            id: "BV1234567890".to_string(),
            title: "测试视频标题".to_string(),
            description: "这是一个测试视频的描述".to_string(),
            url: "https://www.bilibili.com/video/BV1234567890".to_string(),
            author: "测试作者".to_string(),
            upload_date: Utc::now(),
            duration: Some(300), // 5 minutes
            view_count: 12345,
            like_count: 678,
            comment_count: 90,
            tags: vec!["测试".to_string(), "视频".to_string()],
            thumbnail_url: None,
            platform: Platform::Bilibili,
            transcription: None,
        }
    }

    #[test]
    fn test_rss_generation() {
        let generator = RssGenerator::default();
        let videos = vec![create_test_video()];

        let rss_feed = generator.generate_feed(&videos).unwrap();
        assert!(rss_feed.contains("测试视频标题"));
        assert!(rss_feed.contains("测试作者"));
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(1234), "1234");
        assert_eq!(format_number(12345), "1.2万");
        assert_eq!(format_number(123456789), "1.2亿");
    }
}