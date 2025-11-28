use anyhow::{Context, Result};
use feed_rs::parser;
use reqwest::Client;
use std::time::Duration;
use bytes::Bytes;

pub struct RssClient {
    http_client: Client,
}

impl RssClient {
    pub fn new() -> Self {
        let http_client = Client::builder()
            .timeout(Duration::from_secs(10))
            .user_agent("VideoRSSAggregator/1.0")
            .build()
            .expect("Failed to build HTTP client");
        
        Self { http_client }
    }

    pub async fn fetch_feed(&self, url: &str) -> Result<feed_rs::model::Feed> {
        let response = self.http_client.get(url)
            .send()
            .await
            .context("Failed to send request")?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Request failed with status: {}", response.status()));
        }

        let content = response.bytes().await.context("Failed to read response body")?;
        let feed = parser::parse(content.as_ref()).context("Failed to parse RSS feed")?;

        Ok(feed)
    }
}
