use std::env;
use std::time::Instant;

use anyhow::{anyhow, Result};
use serde::Serialize;

use crate::accel::AccelConfig;
use crate::config::AppConfig;
use crate::pipeline::Pipeline;
use crate::storage::Database;

#[derive(Debug, Serialize)]
pub struct VerificationReport {
    pub feed_url: String,
    pub feed_items: usize,
    pub processed_source: String,
    pub transcription_chars: usize,
    pub summary_chars: usize,
    pub total_ms: u128,
    pub feed_ms: u128,
    pub process_ms: u128,
    pub rss_ms: u128,
}

pub async fn run() -> Result<()> {
    let total_start = Instant::now();
    let config = AppConfig::from_env()?;
    let accel = AccelConfig::from_env();

    let db = Database::connect(&config.database_url).await?;
    db.migrate().await?;

    let pipeline = Pipeline::new(&config, &accel, db).await?;

    let feed_url = env::var("VRA_VERIFY_FEED_URL")
        .map_err(|_| anyhow!("VRA_VERIFY_FEED_URL must be set for verification"))?;

    let feed_start = Instant::now();
    let feed_report = pipeline.ingest_feed(&feed_url, false, Some(5)).await?;
    let feed_ms = feed_start.elapsed().as_millis();

    if feed_report.item_count == 0 {
        return Err(anyhow!("Feed returned no entries"));
    }

    let processed_source = pick_real_source()?;

    let process_start = Instant::now();
    let process_report = pipeline.process_source(&processed_source, None).await?;
    let process_ms = process_start.elapsed().as_millis();

    let rss_start = Instant::now();
    let rss = pipeline
        .rss_feed("Verification Feed", &feed_url, "Verification output", 10)
        .await?;
    let rss_ms = rss_start.elapsed().as_millis();

    if rss.trim().is_empty() {
        return Err(anyhow!("RSS output is empty"));
    }

    let report = VerificationReport {
        feed_url,
        feed_items: feed_report.item_count,
        processed_source,
        transcription_chars: process_report.transcription.text.len(),
        summary_chars: process_report.summary.summary.len(),
        total_ms: total_start.elapsed().as_millis(),
        feed_ms,
        process_ms,
        rss_ms,
    };

    println!("{}", serde_json::to_string_pretty(&report)?);

    Ok(())
}

fn pick_real_source() -> Result<String> {
    if let Ok(path) = env::var("VRA_VERIFY_AUDIO_PATH") {
        if path.trim().is_empty() {
            return Err(anyhow!("VRA_VERIFY_AUDIO_PATH is empty"));
        }
        return Ok(path);
    }

    if let Ok(url) = env::var("VRA_VERIFY_AUDIO_URL") {
        if url.trim().is_empty() {
            return Err(anyhow!("VRA_VERIFY_AUDIO_URL is empty"));
        }
        return Ok(url);
    }

    if let Ok(url) = env::var("VRA_VERIFY_VIDEO_URL") {
        if url.trim().is_empty() {
            return Err(anyhow!("VRA_VERIFY_VIDEO_URL is empty"));
        }
        return Ok(url);
    }

    Err(anyhow!(
        "Set VRA_VERIFY_AUDIO_PATH, VRA_VERIFY_AUDIO_URL, or VRA_VERIFY_VIDEO_URL"
    ))
}
