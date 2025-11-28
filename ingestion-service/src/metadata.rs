use anyhow::{Context, Result};
use serde::Deserialize;
use tokio::process::Command;

#[derive(Debug, Deserialize)]
pub struct VideoMetadata {
    pub id: String,
    pub title: String,
    pub description: String,
    pub duration: Option<f64>,
    pub view_count: Option<u64>,
    pub upload_date: Option<String>,
    pub uploader: Option<String>,
    pub tags: Option<Vec<String>>,
}

pub struct MetadataExtractor;

impl MetadataExtractor {
    pub async fn extract(url: &str) -> Result<VideoMetadata> {
        let output = Command::new("yt-dlp")
            .arg("--dump-json")
            .arg("--no-playlist")
            .arg("--skip-download")
            .arg(url)
            .output()
            .await
            .context("Failed to execute yt-dlp")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow::anyhow!("yt-dlp failed: {}", stderr));
        }

        let json = String::from_utf8(output.stdout).context("Invalid UTF-8 from yt-dlp")?;
        let metadata: VideoMetadata = serde_json::from_str(&json).context("Failed to parse yt-dlp JSON")?;

        Ok(metadata)
    }
}
