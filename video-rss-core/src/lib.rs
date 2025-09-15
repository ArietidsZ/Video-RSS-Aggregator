/*!
# Video RSS Core

Rust library for video RSS aggregation with AI transcription.

## Features

- RSS feed generation
- Multi-platform video content fetching (Bilibili, Douyin, Kuaishou)
- Concurrent video processing
- Python bindings
- Content analysis
*/

pub mod api;
pub mod bilibili;
pub mod cache;
pub mod caiman_asr;
pub mod content;
pub mod crdt_sync;
pub mod database;
pub mod embeddings;
pub mod error;
pub mod extractor;
pub mod fast_io;
pub mod monitor;
pub mod monitoring;
pub mod moonshine;
pub mod neural_compression;
pub mod quantum_search;
pub mod realtime;
pub mod resilience;
pub mod rss;
pub mod server;
pub mod simd_optimizations;
pub mod summarizer;
pub mod tiered_cache;
pub mod types;
pub mod utils;
pub mod vector_db;
pub mod wasm_component;
pub mod whisper_candle;

use error::VideoRssError;
use types::*;

pub type Result<T> = std::result::Result<T, VideoRssError>;

// Re-export main types for convenience
pub use bilibili::BilibiliClient;
pub use content::ContentAnalyzer;
pub use extractor::VideoExtractor;
pub use rss::RssGenerator;