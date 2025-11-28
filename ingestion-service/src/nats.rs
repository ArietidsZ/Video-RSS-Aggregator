use anyhow::{Context, Result};
use async_nats::jetstream::{self, Context as JetStreamContext};
use prost::Message;

// Include generated protos
pub mod events {
    tonic::include_proto!("video_rss.events.v1");
}

pub struct NatsClient {
    js: JetStreamContext,
}

impl NatsClient {
    pub async fn new(url: &str) -> Result<Self> {
        let client = async_nats::connect(url).await.context("Failed to connect to NATS")?;
        let js = jetstream::new(client);

        // Ensure stream exists
        let _ = js.get_or_create_stream(jetstream::stream::Config {
            name: "VIDEO_EVENTS".to_string(),
            subjects: vec!["video.discovered".to_string(), "video.transcribed".to_string(), "video.summarized".to_string()],
            ..Default::default()
        }).await.context("Failed to create JetStream stream")?;

        Ok(Self { js })
    }

    pub async fn publish_video_discovered(&self, event: events::VideoDiscoveredEvent) -> Result<()> {
        let mut payload = Vec::new();
        event.encode(&mut payload)?;
        
        self.js.publish("video.discovered", payload.into()).await
            .context("Failed to publish video.discovered event")?;
        
        Ok(())
    }
}
