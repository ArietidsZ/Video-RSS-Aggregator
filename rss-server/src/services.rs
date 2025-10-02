use std::sync::Arc;
use anyhow::Result;
use chrono::{DateTime, Utc};
use rss::{Channel, ChannelBuilder, Item, ItemBuilder, Guid, GuidBuilder};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use regex::Regex;

use crate::{
    cache::CacheManager,
    config::AppConfig,
    models::{Feed, FeedItem},
    platforms::{Platform, PlatformManager, PlatformConfig, VideoMetadata,
               BilibiliConfig, DouyinConfig, KuaishouConfig, YouTubeConfig, VideoQuality},
};

#[derive(Clone)]
pub struct FeedService {
    config: Arc<AppConfig>,
    cache: Arc<CacheManager>,
    db_pool: sqlx::PgPool,
    http_client: reqwest::Client,
    platform_manager: Arc<PlatformManager>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChannelInfo {
    title: String,
    description: String,
    url: String,
}

impl FeedService {
    pub async fn new(config: Arc<AppConfig>, cache: Arc<CacheManager>) -> Result<Self> {
        let db_pool = sqlx::PgPool::connect(&config.database_url).await?;

        // Run migrations
        sqlx::migrate!("./migrations").run(&db_pool).await?;

        let http_client = reqwest::Client::builder()
            .user_agent("RSS-Aggregator/1.0")
            .timeout(std::time::Duration::from_secs(30))
            .gzip(true)
            .build()?;

        // Configure platform adapters
        let platform_config = PlatformConfig {
            bilibili: BilibiliConfig {
                include_danmaku: true,
                include_stats: true,
                quality_preference: VideoQuality::HD1080p,
                subtitle_lang: "zh-CN".to_string(),
                cookie: config.bilibili_cookies.clone(),
            },
            douyin: DouyinConfig {
                include_comments: true,
                include_music: false,
                watermark_removal: true,
                quality_preference: VideoQuality::HD720p,
            },
            kuaishou: KuaishouConfig {
                include_gifts: false,
                include_location: true,
                quality_preference: VideoQuality::Auto,
            },
            youtube: YouTubeConfig {
                include_captions: true,
                include_chapters: true,
                quality_preference: VideoQuality::Highest,
                api_key: config.youtube_api_key.clone(),
            },
        };

        let platform_manager = Arc::new(PlatformManager::new(platform_config));

        Ok(Self {
            config: config.clone(),
            cache: cache.clone(),
            db_pool,
            http_client: http_client.clone(),
            platform_manager,
        })
    }

    pub async fn generate_feed(&self, channel_id: &str) -> Result<String> {
        // Try to get feed from database
        let feed = sqlx::query_as::<_, Feed>(
            "SELECT * FROM feeds WHERE id = $1"
        )
        .bind(channel_id)
        .fetch_optional(&self.db_pool)
        .await?;

        if let Some(feed) = feed {
            self.build_rss_xml(feed).await
        } else {
            Err(anyhow::anyhow!("Feed not found"))
        }
    }

    pub async fn create_feed(
        &self,
        url: &str,
        title: Option<String>,
        description: Option<String>,
        include_summaries: bool,
    ) -> Result<Feed> {
        let feed_id = Uuid::new_v4().to_string();

        // Detect platform from URL
        let platform = self.detect_platform(url);

        // Extract channel/user ID based on platform
        let (channel_id, platform_name) = match platform {
            Platform::YouTube => {
                let id = self.extract_youtube_channel_id(url)?;
                (id, "youtube")
            }
            Platform::Bilibili => {
                let id = self.extract_bilibili_uid(url)?;
                (id, "bilibili")
            }
            Platform::Douyin => {
                let id = self.extract_douyin_user_id(url)?;
                (id, "douyin")
            }
            Platform::Kuaishou => {
                let id = self.extract_kuaishou_user_id(url)?;
                (id, "kuaishou")
            }
            _ => {
                (url.to_string(), "unknown")
            }
        };

        // Fetch channel information
        let channel_info = self.fetch_channel_info(&channel_id, platform).await?;

        // Create feed in database
        let feed = sqlx::query_as::<_, Feed>(
            "INSERT INTO feeds (id, channel_id, platform, title, description, url, include_summaries, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING *"
        )
        .bind(&feed_id)
        .bind(&channel_id)
        .bind(platform_name)
        .bind(title.unwrap_or_else(|| channel_info.title.clone()))
        .bind(description.unwrap_or_else(|| channel_info.description.clone()))
        .bind(url)
        .bind(include_summaries)
        .bind(Utc::now())
        .bind(Utc::now())
        .fetch_one(&self.db_pool)
        .await?;

        // Fetch initial items
        self.update_feed_items(&feed.id).await?;

        Ok(feed)
    }

    pub async fn get_feed_items(&self, channel_id: &str, limit: usize) -> Result<Vec<FeedItem>> {
        let items = sqlx::query_as::<_, FeedItem>(
            "SELECT * FROM feed_items
            WHERE feed_id = $1
            ORDER BY published_at DESC
            LIMIT $2"
        )
        .bind(channel_id)
        .bind(limit as i64)
        .fetch_all(&self.db_pool)
        .await?;

        Ok(items)
    }

    pub async fn update_feed_items(&self, feed_id: &str) -> Result<()> {
        let feed = sqlx::query_as::<_, Feed>(
            "SELECT * FROM feeds WHERE id = $1"
        )
        .bind(feed_id)
        .fetch_one(&self.db_pool)
        .await?;

        // Fetch new items based on platform
        let platform = match feed.platform.as_str() {
            "youtube" => Platform::YouTube,
            "bilibili" => Platform::Bilibili,
            "douyin" => Platform::Douyin,
            "kuaishou" => Platform::Kuaishou,
            _ => return Ok(()),
        };

        let adapter = self.platform_manager.get_adapter(platform)
            .ok_or(anyhow::anyhow!("Platform adapter not found"))?;

        let new_items = adapter.fetch_channel_videos(&feed.channel_id, 20).await?;

        // Insert new items
        for item in new_items {
            let item_id = Uuid::new_v4().to_string();

            // Generate summary if requested
            let summary = if feed.include_summaries {
                self.generate_summary(&item.description).await.ok()
            } else {
                None
            };

            sqlx::query(
                "INSERT INTO feed_items (id, feed_id, title, description, summary, url, guid, published_at, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (guid) DO UPDATE
                SET title = EXCLUDED.title,
                    description = EXCLUDED.description,
                    summary = EXCLUDED.summary,
                    updated_at = NOW()"
            )
            .bind(&item_id)
            .bind(feed_id)
            .bind(&item.title)
            .bind(&item.description)
            .bind(&summary)
            .bind(&item.video_url)
            .bind(&item.id)
            .bind(item.published_at)
            .bind(Utc::now())
            .execute(&self.db_pool)
            .await?;
        }

        // Update feed's last_updated timestamp
        sqlx::query(
            "UPDATE feeds SET updated_at = NOW() WHERE id = $1"
        )
        .bind(feed_id)
        .execute(&self.db_pool)
        .await?;

        Ok(())
    }

    async fn build_rss_xml(&self, feed: Feed) -> Result<String> {
        // Get feed items
        let items = self.get_feed_items(&feed.id, 50).await?;

        // Build RSS channel
        let mut channel = ChannelBuilder::default()
            .title(&feed.title)
            .link(&feed.url)
            .description(&feed.description)
            .language(feed.language.unwrap_or_else(|| "en".to_string()))
            .pub_date(feed.updated_at.to_rfc2822())
            .last_build_date(Utc::now().to_rfc2822())
            .generator("RSS Aggregator v1.0".to_string())
            .build();

        // Add items
        let rss_items: Vec<Item> = items
            .into_iter()
            .map(|item| {
                let guid = GuidBuilder::default()
                    .value(item.guid.clone())
                    .permalink(false)
                    .build();

                let description = if feed.include_summaries && item.summary.is_some() {
                    format!("<p><strong>Summary:</strong></p><p>{}</p><hr/><p><strong>Original:</strong></p><p>{}</p>",
                        item.summary.unwrap(),
                        item.description
                    )
                } else {
                    item.description.clone()
                };

                ItemBuilder::default()
                    .title(Some(item.title))
                    .link(Some(item.url))
                    .description(Some(description))
                    .guid(Some(guid))
                    .pub_date(Some(item.published_at.to_rfc2822()))
                    .build()
            })
            .collect();

        channel.set_items(rss_items);

        // Convert to XML string
        let xml = channel.to_string();
        Ok(xml)
    }

    pub async fn generate_youtube_feed(&self, channel_id: &str) -> Result<String> {
        // Check cache
        let cache_key = format!("youtube:{}", channel_id);
        if let Some(cached) = self.cache.get_feed(&cache_key).await {
            return Ok(cached);
        }

        // Fetch from YouTube using platform adapter
        let adapter = self.platform_manager.get_adapter(Platform::YouTube)
            .ok_or(anyhow::anyhow!("YouTube adapter not found"))?;

        let videos = adapter.fetch_channel_videos(channel_id, 50).await?;

        // Generate customized RSS
        let xml = self.platform_manager.generate_unified_rss(videos).await?;

        // Cache for 5 minutes
        self.cache.set_feed(&cache_key, &xml, 300).await;

        Ok(xml)
    }

    pub async fn generate_bilibili_feed(&self, uid: &str) -> Result<String> {
        let cache_key = format!("bilibili:{}", uid);
        if let Some(cached) = self.cache.get_feed(&cache_key).await {
            return Ok(cached);
        }

        let adapter = self.platform_manager.get_adapter(Platform::Bilibili)
            .ok_or(anyhow::anyhow!("Bilibili adapter not found"))?;

        let videos = adapter.fetch_channel_videos(uid, 50).await?;
        let xml = self.platform_manager.generate_unified_rss(videos).await?;

        self.cache.set_feed(&cache_key, &xml, 300).await;

        Ok(xml)
    }

    pub async fn generate_douyin_feed(&self, user_id: &str) -> Result<String> {
        let cache_key = format!("douyin:{}", user_id);
        if let Some(cached) = self.cache.get_feed(&cache_key).await {
            return Ok(cached);
        }

        let adapter = self.platform_manager.get_adapter(Platform::Douyin)
            .ok_or(anyhow::anyhow!("Douyin adapter not found"))?;

        let videos = adapter.fetch_channel_videos(user_id, 50).await?;
        let xml = self.platform_manager.generate_unified_rss(videos).await?;

        self.cache.set_feed(&cache_key, &xml, 300).await;

        Ok(xml)
    }

    pub async fn generate_kuaishou_feed(&self, user_id: &str) -> Result<String> {
        let cache_key = format!("kuaishou:{}", user_id);
        if let Some(cached) = self.cache.get_feed(&cache_key).await {
            return Ok(cached);
        }

        let adapter = self.platform_manager.get_adapter(Platform::Kuaishou)
            .ok_or(anyhow::anyhow!("Kuaishou adapter not found"))?;

        let videos = adapter.fetch_channel_videos(user_id, 50).await?;
        let xml = self.platform_manager.generate_unified_rss(videos).await?;

        self.cache.set_feed(&cache_key, &xml, 300).await;

        Ok(xml)
    }

    pub async fn search_feeds(&self, query: &str) -> Result<Vec<serde_json::Value>> {
        let results = sqlx::query_as::<_, (String, String, String, String, String)>(
            "SELECT id, title, description, platform, url
            FROM feeds
            WHERE title ILIKE $1 OR description ILIKE $1
            ORDER BY created_at DESC
            LIMIT 20"
        )
        .bind(format!("%{}%", query))
        .fetch_all(&self.db_pool)
        .await?;

        Ok(results.into_iter().map(|r| serde_json::json!({
            "id": r.0,
            "title": r.1,
            "description": r.2,
            "platform": r.3,
            "url": r.4
        })).collect())
    }

    pub async fn get_trending_feeds(&self) -> Result<Vec<serde_json::Value>> {
        // Get feeds with most recent activity
        let results = sqlx::query_as::<_, (String, String, String, String, i64)>(
            "SELECT f.id, f.title, f.description, f.platform, COUNT(fi.id) as item_count
            FROM feeds f
            LEFT JOIN feed_items fi ON f.id = fi.feed_id
            WHERE fi.published_at > NOW() - INTERVAL '7 days'
            GROUP BY f.id
            ORDER BY item_count DESC
            LIMIT 20"
        )
        .fetch_all(&self.db_pool)
        .await?;

        Ok(results.into_iter().map(|r| serde_json::json!({
            "id": r.0,
            "title": r.1,
            "description": r.2,
            "platform": r.3,
            "item_count": r.4
        })).collect())
    }

    pub async fn get_recommendations(&self, user_id: &str) -> Result<Vec<serde_json::Value>> {
        // Simple recommendation based on user's subscribed feeds
        let results = sqlx::query_as::<_, (String, String, String, String)>(
            "SELECT f.id, f.title, f.description, f.platform
            FROM feeds f
            WHERE f.id NOT IN (
                SELECT feed_id FROM user_subscriptions WHERE user_id = $1
            )
            ORDER BY RANDOM()
            LIMIT 10"
        )
        .bind(user_id)
        .fetch_all(&self.db_pool)
        .await?;

        Ok(results.into_iter().map(|r| serde_json::json!({
            "id": r.0,
            "title": r.1,
            "description": r.2,
            "platform": r.3
        })).collect())
    }

    pub async fn list_all_feeds(&self) -> Result<Vec<serde_json::Value>> {
        let results = sqlx::query_as::<_, (String, String, String, String, String)>(
            "SELECT id, title, description, platform, url FROM feeds ORDER BY created_at DESC"
        )
        .fetch_all(&self.db_pool)
        .await?;

        Ok(results.into_iter().map(|r| serde_json::json!({
            "id": r.0,
            "title": r.1,
            "description": r.2,
            "platform": r.3,
            "url": r.4
        })).collect())
    }

    pub async fn get_feed_details(&self, feed_id: &str) -> Result<Option<serde_json::Value>> {
        let result = sqlx::query_as::<_, Feed>(
            "SELECT * FROM feeds WHERE id = $1"
        )
        .bind(feed_id)
        .fetch_optional(&self.db_pool)
        .await?;

        Ok(result.map(|r| serde_json::json!({
            "id": r.id,
            "channel_id": r.channel_id,
            "platform": r.platform,
            "title": r.title,
            "description": r.description,
            "url": r.url,
            "include_summaries": r.include_summaries,
            "created_at": r.created_at,
            "updated_at": r.updated_at
        })))
    }

    pub async fn update_feed(&self, feed_id: &str, updates: serde_json::Value) -> Result<()> {
        if let Some(title) = updates.get("title").and_then(|v| v.as_str()) {
            sqlx::query(
                "UPDATE feeds SET title = $1, updated_at = NOW() WHERE id = $2"
            )
            .bind(title)
            .bind(feed_id)
            .execute(&self.db_pool)
            .await?;
        }

        if let Some(description) = updates.get("description").and_then(|v| v.as_str()) {
            sqlx::query(
                "UPDATE feeds SET description = $1, updated_at = NOW() WHERE id = $2"
            )
            .bind(description)
            .bind(feed_id)
            .execute(&self.db_pool)
            .await?;
        }

        if let Some(include_summaries) = updates.get("include_summaries").and_then(|v| v.as_bool()) {
            sqlx::query(
                "UPDATE feeds SET include_summaries = $1, updated_at = NOW() WHERE id = $2"
            )
            .bind(include_summaries)
            .bind(feed_id)
            .execute(&self.db_pool)
            .await?;
        }

        Ok(())
    }

    pub async fn delete_feed(&self, feed_id: &str) -> Result<()> {
        // Delete feed items first
        sqlx::query(
            "DELETE FROM feed_items WHERE feed_id = $1"
        )
        .bind(feed_id)
        .execute(&self.db_pool)
        .await?;

        // Delete feed
        sqlx::query(
            "DELETE FROM feeds WHERE id = $1"
        )
        .bind(feed_id)
        .execute(&self.db_pool)
        .await?;

        Ok(())
    }

    pub async fn is_healthy(&self) -> bool {
        sqlx::query_scalar::<_, i32>("SELECT 1")
            .fetch_one(&self.db_pool)
            .await
            .is_ok()
    }

    async fn generate_summary(&self, content: &str) -> Result<String> {
        // TODO: Integrate with summarization engine
        // For now, return a simple truncation
        Ok(content.chars().take(200).collect::<String>() + "...")
    }

    fn detect_platform(&self, url: &str) -> Platform {
        if url.contains("youtube.com") || url.contains("youtu.be") {
            Platform::YouTube
        } else if url.contains("bilibili.com") || url.contains("b23.tv") {
            Platform::Bilibili
        } else if url.contains("douyin.com") || url.contains("iesdouyin.com") {
            Platform::Douyin
        } else if url.contains("kuaishou.com") || url.contains("kwai.com") {
            Platform::Kuaishou
        } else {
            Platform::YouTube // Default fallback
        }
    }

    fn extract_youtube_channel_id(&self, url: &str) -> Result<String> {
        let re = Regex::new(r"(?:youtube\.com/(?:c/|channel/|user/)|youtu\.be/)([a-zA-Z0-9_-]+)")?;
        re.captures(url)
            .and_then(|cap| cap.get(1).map(|m| m.as_str().to_string()))
            .ok_or(anyhow::anyhow!("Invalid YouTube URL"))
    }

    fn extract_bilibili_uid(&self, url: &str) -> Result<String> {
        let re = Regex::new(r"space\.bilibili\.com/(\d+)")?;
        re.captures(url)
            .and_then(|cap| cap.get(1).map(|m| m.as_str().to_string()))
            .ok_or(anyhow::anyhow!("Invalid Bilibili URL"))
    }

    fn extract_douyin_user_id(&self, url: &str) -> Result<String> {
        let re = Regex::new(r"douyin\.com/user/([A-Za-z0-9_-]+)")?;
        re.captures(url)
            .and_then(|cap| cap.get(1).map(|m| m.as_str().to_string()))
            .ok_or(anyhow::anyhow!("Invalid Douyin URL"))
    }

    fn extract_kuaishou_user_id(&self, url: &str) -> Result<String> {
        let re = Regex::new(r"kuaishou\.com/profile/([A-Za-z0-9_-]+)")?;
        re.captures(url)
            .and_then(|cap| cap.get(1).map(|m| m.as_str().to_string()))
            .ok_or(anyhow::anyhow!("Invalid Kuaishou URL"))
    }

    async fn fetch_channel_info(&self, channel_id: &str, platform: Platform) -> Result<ChannelInfo> {
        match platform {
            Platform::YouTube => Ok(ChannelInfo {
                title: format!("YouTube Channel {}", channel_id),
                description: "YouTube channel feed".to_string(),
                url: format!("https://www.youtube.com/channel/{}", channel_id),
            }),
            Platform::Bilibili => Ok(ChannelInfo {
                title: format!("Bilibili User {}", channel_id),
                description: "Bilibili user feed".to_string(),
                url: format!("https://space.bilibili.com/{}", channel_id),
            }),
            Platform::Douyin => Ok(ChannelInfo {
                title: format!("Douyin User {}", channel_id),
                description: "Douyin user feed".to_string(),
                url: format!("https://www.douyin.com/user/{}", channel_id),
            }),
            Platform::Kuaishou => Ok(ChannelInfo {
                title: format!("Kuaishou User {}", channel_id),
                description: "Kuaishou user feed".to_string(),
                url: format!("https://www.kuaishou.com/profile/{}", channel_id),
            }),
        }
    }
}

pub struct SummarizationService {
    config: Arc<AppConfig>,
    client: reqwest::Client,
}

impl SummarizationService {
    pub async fn new(config: Arc<AppConfig>) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()?;

        Ok(Self { config, client })
    }

    pub async fn summarize(
        &self,
        content: &str,
        style: Option<&str>,
        max_length: Option<usize>,
    ) -> Result<String> {
        // TODO: Integrate with actual summarization engine
        let summary = content
            .chars()
            .take(max_length.unwrap_or(500))
            .collect::<String>();

        Ok(format!("[{}] {}", style.unwrap_or("default"), summary))
    }

    pub async fn get_job_status(&self, job_id: &str) -> Option<serde_json::Value> {
        Some(serde_json::json!({
            "job_id": job_id,
            "status": "completed",
            "progress": 100
        }))
    }
}