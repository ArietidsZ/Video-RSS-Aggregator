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

pub mod bilibili;
pub mod cache;
pub mod content;
pub mod database;
pub mod error;
pub mod extractor;
pub mod monitor;
pub mod resilience;
pub mod rss;
pub mod server;
pub mod transcription;
pub mod types;
pub mod utils;

#[cfg(feature = "python")]
pub mod python;

use error::VideoRssError;
use types::*;

pub type Result<T> = std::result::Result<T, VideoRssError>;

// Re-export main types for convenience
pub use bilibili::BilibiliClient;
pub use content::ContentAnalyzer;
pub use extractor::VideoExtractor;
pub use rss::RssGenerator;