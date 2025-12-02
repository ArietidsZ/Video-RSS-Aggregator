use anyhow::{Context, Result};
use async_nats::jetstream::{self, Context as JetStreamContext};
use futures::StreamExt;
use prost::Message;
use std::env;
use whisper_turbo::WhisperTurbo;

pub mod events {
    tonic::include_proto!("video_rss.events.v1");
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing/logging
    tracing_subscriber::fmt::init();

    let nats_url = env::var("NATS_URL").unwrap_or_else(|_| "nats://localhost:4222".to_string());
    println!("Connecting to NATS at {}", nats_url);

    let client = async_nats::connect(&nats_url).await?;
    let js = jetstream::new(client);

    // Get stream (expecting it to be created by ingestion-service, or we can create it)
    let stream = js.get_or_create_stream(jetstream::stream::Config {
        name: "VIDEO_EVENTS".to_string(),
        subjects: vec!["video.discovered".to_string(), "video.transcribed".to_string(), "video.summarized".to_string()],
        ..Default::default()
    }).await.context("Failed to get/create JetStream stream")?;

    // Create consumer
    let consumer = stream.create_consumer(jetstream::consumer::pull::Config {
        durable_name: Some("transcription-service".to_string()),
        filter_subject: "video.discovered".to_string(),
        ..Default::default()
    }).await?;

    // Initialize Whisper
    let model_path = env::var("WHISPER_MODEL_PATH").unwrap_or_else(|_| "models/whisper-large-v3".to_string());
    println!("Initializing Whisper with model: {}", model_path);
    
    let mut transcriber = WhisperTurbo::new();
    // In a real app, we might want to handle initialization failure gracefully or retry
    if let Err(e) = transcriber.initialize(whisper_turbo::TranscriptionOptions {
        model_path: model_path.clone(),
        ..Default::default()
    }) {
        eprintln!("Warning: Failed to initialize Whisper (is the model path correct?): {}", e);
        // We continue, but transcription will fail
    }

    println!("Listening for video.discovered events...");

    let mut messages = consumer.messages().await?;

    while let Some(msg) = messages.next().await {
        match msg {
            Ok(msg) => {
                match events::VideoDiscoveredEvent::decode(msg.payload.clone()) {
                    Ok(event) => {
                        println!("Received video: {} ({})", event.title, event.url);
                        
                        match download_audio(&event.url).await {
                            Ok(path) => {
                                println!("Downloaded to {:?}", path);
                                
                                // Transcribe
                                match transcriber.transcribe_file(&path).await {
                                    Ok(result) => {
                                        println!("Transcribed: {} chars", result.full_text.len());
                                        
                                        // Publish event
                                        let completed_event = events::TranscriptionCompletedEvent {
                                            video_id: event.video_id.clone(),
                                            transcription_text: result.full_text,
                                            language: result.detected_language,
                                        };
                                        
                                        let mut payload = Vec::new();
                                        if let Ok(()) = completed_event.encode(&mut payload) {
                                            if let Err(e) = js.publish("video.transcribed", payload.into()).await {
                                                eprintln!("Failed to publish video.transcribed: {}", e);
                                            } else {
                                                println!("Published transcription event for {}", event.video_id);
                                            }
                                        }
                                    },
                                    Err(e) => eprintln!("Transcription failed: {}", e),
                                }
                                
                                // Clean up
                                let _ = tokio::fs::remove_file(path).await;
                            },
                            Err(e) => eprintln!("Failed to download audio: {}", e),
                        }
                        
                        if let Err(e) = msg.ack().await {
                            eprintln!("Failed to ack message: {}", e);
                        }
                    },
                    Err(e) => {
                        eprintln!("Failed to decode event: {}", e);
                        // Ack to remove bad message? Or Nack?
                        let _ = msg.ack().await;
                    }
                }
            },
            Err(e) => eprintln!("Error receiving message: {}", e),
        }
    }

    Ok(())
}

async fn download_audio(url: &str) -> Result<std::path::PathBuf> {
    println!("Downloading {}", url);
    let mut response = reqwest::get(url).await?.error_for_status()?;
    let tmp_dir = std::env::temp_dir();
    let file_name = format!("video_rss_{}.mp4", uuid::Uuid::new_v4());
    let file_path = tmp_dir.join(file_name);
    
    let mut file = tokio::fs::File::create(&file_path).await?;
    
    use tokio::io::AsyncWriteExt;
    while let Some(chunk) = response.chunk().await? {
        file.write_all(&chunk).await?;
    }
    
    Ok(file_path)
}
