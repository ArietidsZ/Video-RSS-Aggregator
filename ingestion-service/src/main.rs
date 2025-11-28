mod rss;
mod metadata;
mod nats;

use anyhow::Result;
use db::DbClient;
use rss::RssClient;
use nats::NatsClient;
use std::env;
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() -> Result<()> {
    // Load env vars (in real app use dotenv)
    let database_url = env::var("DATABASE_URL").unwrap_or_else(|_| "postgres://user:password@localhost:5432/video_rss".to_string());
    let nats_url = env::var("NATS_URL").unwrap_or_else(|_| "nats://localhost:4222".to_string());

    println!("Connecting to database...");
    let db = DbClient::new(&database_url).await?;
    println!("Database connected and migrations applied.");

    println!("Connecting to NATS...");
    let nats = NatsClient::new(&nats_url).await?;
    println!("NATS connected.");

    let rss_client = RssClient::new();

    // Seed a feed for testing if empty
    if let Ok(feeds) = db.list_active_feeds().await {
        if feeds.is_empty() {
            println!("Seeding test feed...");
            let _ = db.create_feed("https://feeds.feedburner.com/tedtalks_video").await;
        }
    }

    loop {
        println!("Starting ingestion cycle...");
        match db.list_active_feeds().await {
            Ok(feeds) => {
                for feed in feeds {
                    println!("Processing feed: {}", feed.url);
                    match rss_client.fetch_feed(&feed.url).await {
                        Ok(parsed_feed) => {
                            println!("Fetched {} items from {}", parsed_feed.entries.len(), feed.url);
                            for entry in parsed_feed.entries {
                                let item = db::Item {
                                    id: uuid::Uuid::new_v4(), // Ignored on upsert
                                    feed_id: feed.id,
                                    external_id: entry.id.clone(),
                                    url: entry.links.first().map(|l| l.href.clone()).unwrap_or_default(),
                                    title: entry.title.as_ref().map(|t| t.content.clone()),
                                    description: entry.summary.as_ref().map(|s| s.content.clone()),
                                    published_at: entry.published.map(|d| d.into()),
                                    created_at: chrono::Utc::now(),
                                };

                                match db.upsert_item(feed.id, &item).await {
                                    Ok(true) => {
                                        println!("New item: {:?}", item.title);
                                        // Publish event
                                        let event = nats::events::VideoDiscoveredEvent {
                                            feed_id: feed.id.to_string(),
                                            video_id: item.external_id, // Use external ID for now
                                            url: item.url,
                                            title: item.title.unwrap_or_default(),
                                            published_at: Some(prost_types::Timestamp::from(std::time::SystemTime::now())), // Simplified
                                            provider: "RSS".to_string(),
                                        };
                                        if let Err(e) = nats.publish_video_discovered(event).await {
                                            eprintln!("Failed to publish event: {}", e);
                                        }
                                    },
                                    Ok(false) => {}, // Existing item
                                    Err(e) => eprintln!("Failed to upsert item: {}", e),
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("Error fetching feed {}: {}", feed.url, e);
                            // TODO: Update error count
                        }
                    }
                }
            }
            Err(e) => eprintln!("Failed to list feeds: {}", e),
        }
        
        println!("Cycle complete. Sleeping...");
        sleep(Duration::from_secs(60)).await;
    }
}
