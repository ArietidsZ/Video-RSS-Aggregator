use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoInfo {
    pub id: String,
    pub title: String,
    pub description: String,
    pub url: String,
    pub author: String,
    pub upload_date: DateTime<Utc>,
    pub duration: Option<u64>, // seconds
    pub view_count: u64,
    pub like_count: u64,
    pub comment_count: u64,
    pub tags: Vec<String>,
    pub thumbnail_url: Option<String>,
    pub platform: Platform,
    pub transcription: Option<TranscriptionData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionData {
    pub paragraph_summary: String,
    pub sentence_subtitle: String,
    pub full_transcript: String,
    pub status: TranscriptionStatus,
    pub model_info: ModelInfo,
    pub source_types: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TranscriptionStatus {
    Success,
    Pending,
    Failed,
    Unavailable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub transcriber: String,
    pub summarizer: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Platform {
    Bilibili,
    Douyin,
    Kuaishou,
    YouTube,
}

impl Platform {
    pub fn as_str(&self) -> &'static str {
        match self {
            Platform::Bilibili => "bilibili",
            Platform::Douyin => "douyin",
            Platform::Kuaishou => "kuaishou",
            Platform::YouTube => "youtube",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BilibiliCredentials {
    pub sessdata: String,
    pub bili_jct: String,
    pub buvid3: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FetchOptions {
    pub limit: usize,
    pub include_transcription: bool,
    pub personalized: bool,
    pub credentials: Option<BilibiliCredentials>,
}

impl Default for FetchOptions {
    fn default() -> Self {
        Self {
            limit: 10,
            include_transcription: false,
            personalized: false,
            credentials: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RssConfig {
    pub title: String,
    pub description: String,
    pub link: String,
    pub language: String,
    pub generator: String,
}

impl Default for RssConfig {
    fn default() -> Self {
        Self {
            title: "AI智能内容摘要 - 精选视频".to_string(),
            description: "基于人工智能技术的中文视频平台内容聚合与智能分析".to_string(),
            link: "http://localhost:3000".to_string(),
            language: "zh-CN".to_string(),
            generator: "Video RSS Core v1.0".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentSummary {
    pub ai_summary: String,
    pub keywords: Vec<String>,
    pub sentiment: Option<f32>, // -1.0 to 1.0
    pub content_type: ContentType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentType {
    Educational,
    Entertainment,
    News,
    Gaming,
    Technology,
    Music,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub video: VideoInfo,
    pub summary: ContentSummary,
    pub processing_time_ms: u64,
}