use thiserror::Error;

#[derive(Error, Debug)]
pub enum VideoProcessingError {
    #[error("GPU backend error: {0}")]
    GpuBackendError(String),

    #[error("Video decoding error: {0}")]
    VideoDecodingError(String),

    #[error("Audio processing error: {0}")]
    AudioProcessingError(String),

    #[error("Transcription error: {0}")]
    TranscriptionError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("FFmpeg error: {0}")]
    FfmpegError(#[from] ffmpeg_next::Error),

    #[error("Database error: {0}")]
    DatabaseError(#[from] sqlx::Error),

    #[error("Redis error: {0}")]
    RedisError(#[from] redis::RedisError),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Unknown error: {0}")]
    Unknown(String),
}

impl From<anyhow::Error> for VideoProcessingError {
    fn from(err: anyhow::Error) -> Self {
        VideoProcessingError::Unknown(err.to_string())
    }
}