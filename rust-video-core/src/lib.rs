/*!
# Video RSS Core

High-performance Rust library for video RSS aggregation with AI transcription.

## Features

- Fast RSS feed generation
- Multi-platform video content fetching (Bilibili, Douyin, Kuaishou)
- Concurrent video processing
- Python bindings for seamless integration
- Memory-efficient content analysis
*/

pub mod bilibili;
pub mod content;
pub mod error;
pub mod extractor;
pub mod monitor;
pub mod rss;
pub mod server;
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
pub use extractor::RealDataExtractor;
pub use rss::RssGenerator;