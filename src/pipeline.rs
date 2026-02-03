use std::io::Cursor;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use feed_rs::parser;
use reqwest::Client;

use crate::accel::AccelConfig;
use crate::config::AppConfig;
use crate::media;
use crate::rss::{render_feed, RssItemData};
use crate::storage::{Database, SummaryRecord};
use crate::summarize::{SummaryResult, SummarizationEngine};
use crate::transcribe::{TranscriptionEngine, TranscriptionResult};

#[derive(Clone, Debug)]
pub struct IngestReport {
    pub feed_title: Option<String>,
    pub item_count: usize,
    pub processed_count: usize,
}

#[derive(Clone, Debug)]
pub struct ProcessReport {
    pub source_url: String,
    pub title: Option<String>,
    pub transcription: TranscriptionResult,
    pub summary: SummaryResult,
}

pub struct Pipeline {
    db: Database,
    client: Client,
    transcriber: Arc<Mutex<TranscriptionEngine>>,
    summarizer: Arc<Mutex<SummarizationEngine>>,
    storage_dir: PathBuf,
}

impl Pipeline {
    pub async fn new(config: &AppConfig, accel: &AccelConfig, db: Database) -> Result<Self> {
        media::ensure_storage_dirs(PathBuf::from(&config.storage_dir).as_path()).await?;

        let transcriber = TranscriptionEngine::new(accel)?;
        let summarizer = SummarizationEngine::new(accel)?;

        Ok(Self {
            db,
            client: Client::new(),
            transcriber: Arc::new(Mutex::new(transcriber)),
            summarizer: Arc::new(Mutex::new(summarizer)),
            storage_dir: PathBuf::from(&config.storage_dir),
        })
    }

    pub async fn ingest_feed(
        &self,
        feed_url: &str,
        process: bool,
        max_items: Option<usize>,
    ) -> Result<IngestReport> {
        let response = self.client.get(feed_url).send().await?.error_for_status()?;
        let bytes = response.bytes().await?;
        let feed = parser::parse(Cursor::new(bytes))?;

        let feed_title = feed.title.as_ref().map(|title| title.content.clone());
        let feed_id = self.db.upsert_feed(feed_url, feed_title.as_deref()).await?;

        let mut item_count = 0;
        let mut processed_count = 0;

        let entries = feed.entries.iter();
        let limit = max_items.unwrap_or(feed.entries.len());

        for entry in entries.take(limit) {
            let title = entry.title.as_ref().map(|value| value.content.clone());
            let guid = if entry.id.is_empty() {
                None
            } else {
                Some(entry.id.clone())
            };
            let published_at = entry.published.or(entry.updated).map(to_utc);
            let source_url = entry
                .enclosures
                .first()
                .map(|enclosure| enclosure.url.clone())
                .or_else(|| entry.links.first().map(|link| link.href.clone()));

            let Some(source_url) = source_url else {
                continue;
            };

            let video_id = self
                .db
                .upsert_video(
                    Some(feed_id),
                    guid.as_deref(),
                    title.as_deref(),
                    &source_url,
                    published_at,
                )
                .await?;
            item_count += 1;

            if process {
                let report = self.process_source_with_video(video_id, &source_url, title.clone()).await?;
                if !report.summary.summary.is_empty() {
                    processed_count += 1;
                }
            }
        }

        Ok(IngestReport {
            feed_title,
            item_count,
            processed_count,
        })
    }

    pub async fn process_source(&self, source_url: &str, title: Option<String>) -> Result<ProcessReport> {
        let video_id = self
            .db
            .upsert_video(None, None, title.as_deref(), source_url, None)
            .await?;

        self.process_source_with_video(video_id, source_url, title).await
    }

    pub async fn rss_feed(
        &self,
        title: &str,
        link: &str,
        description: &str,
        limit: usize,
    ) -> Result<String> {
        let records = self.db.latest_summaries(limit as i64).await?;
        let items = records.into_iter().map(to_rss_item).collect::<Vec<_>>();
        render_feed(title, link, description, &items)
    }

    async fn process_source_with_video(
        &self,
        video_id: uuid::Uuid,
        source_url: &str,
        title: Option<String>,
    ) -> Result<ProcessReport> {
        let audio_path = media::prepare_audio_from_source(source_url, &self.storage_dir).await?;
        let audio_path_str = audio_path
            .to_str()
            .ok_or_else(|| anyhow!("Invalid audio path"))?
            .to_string();

        let transcription = self.transcribe_blocking(&audio_path_str).await?;
        let summary = self.summarize_blocking(&transcription.text).await?;

        self.db.insert_transcript(video_id, &transcription).await?;
        self.db.insert_summary(video_id, &summary).await?;

        Ok(ProcessReport {
            source_url: source_url.to_string(),
            title,
            transcription,
            summary,
        })
    }

    async fn transcribe_blocking(&self, audio_path: &str) -> Result<TranscriptionResult> {
        let transcriber = Arc::clone(&self.transcriber);
        let audio_path = audio_path.to_string();
        tokio::task::spawn_blocking(move || {
            let guard = transcriber
                .lock()
                .map_err(|_| anyhow!("Transcription engine lock poisoned"))?;
            guard.transcribe_audio(&audio_path)
        })
        .await?
    }

    async fn summarize_blocking(&self, text: &str) -> Result<SummaryResult> {
        let summarizer = Arc::clone(&self.summarizer);
        let text = text.to_string();
        tokio::task::spawn_blocking(move || {
            let guard = summarizer
                .lock()
                .map_err(|_| anyhow!("Summarization engine lock poisoned"))?;
            guard.summarize(&text)
        })
        .await?
    }
}

fn to_utc(dt: DateTime<Utc>) -> DateTime<Utc> {
    dt
}

fn to_rss_item(record: SummaryRecord) -> RssItemData {
    RssItemData {
        title: record
            .title
            .unwrap_or_else(|| "Untitled video".to_string()),
        link: record.source_url,
        summary: record.summary,
        key_points: record.key_points,
        published_at: record.published_at,
        guid: None,
    }
}
