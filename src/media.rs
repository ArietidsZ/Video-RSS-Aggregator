use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use futures::StreamExt;
use reqwest::Client;
use tokio::fs::{self, File};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use uuid::Uuid;

pub async fn ensure_storage_dirs(storage_dir: &Path) -> Result<()> {
    fs::create_dir_all(storage_dir).await?;
    fs::create_dir_all(storage_dir.join("raw")).await?;
    fs::create_dir_all(storage_dir.join("audio")).await?;
    Ok(())
}

pub async fn prepare_audio_from_source(
    client: &Client,
    source: &str,
    storage_dir: &Path,
) -> Result<PathBuf> {
    ensure_storage_dirs(storage_dir).await?;

    let id = Uuid::new_v4();
    let raw_path = storage_dir.join("raw").join(id.to_string());
    let input_path = if is_url(source) {
        download_to_file(client, source, &raw_path).await?;
        raw_path
    } else {
        let path = PathBuf::from(source);
        if !path.exists() {
            return Err(anyhow!("Source file not found: {}", source));
        }
        path
    };

    let output_path = storage_dir
        .join("audio")
        .join(format!("{}.wav", id));
    extract_audio_ffmpeg(&input_path, &output_path).await?;

    Ok(output_path)
}

pub async fn download_to_file(client: &Client, url: &str, destination: &Path) -> Result<()> {
    let response = client.get(url).send().await?.error_for_status()?;
    let mut stream = response.bytes_stream();

    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).await?;
    }

    let mut file = File::create(destination).await?;
    while let Some(chunk) = stream.next().await {
        let bytes = chunk?;
        file.write_all(&bytes).await?;
    }

    Ok(())
}

pub async fn extract_audio_ffmpeg(input: &Path, output: &Path) -> Result<()> {
    let status = Command::new("ffmpeg")
        .arg("-y")
        .arg("-i")
        .arg(input)
        .arg("-vn")
        .arg("-acodec")
        .arg("pcm_s16le")
        .arg("-ar")
        .arg("16000")
        .arg("-ac")
        .arg("1")
        .arg(output)
        .status()
        .await?;

    if !status.success() {
        return Err(anyhow!("ffmpeg failed with status {}", status));
    }

    Ok(())
}

fn is_url(value: &str) -> bool {
    value.starts_with("http://") || value.starts_with("https://")
}
