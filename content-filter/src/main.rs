mod quality_scorer;
mod moderation;
mod spam_detector;
mod adult_filter;
mod keyword_filter;
mod recommendation;

use anyhow::Result;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::{info, warn};

use quality_scorer::{ContentQualityScorer, QualityScorerConfig};
use moderation::{ContentModerator, ModerationThresholds};
use spam_detector::{SpamDetector, SpamDetectorConfig};
use adult_filter::{AdultContentFilter, FilterConfig};
use keyword_filter::{KeywordFilter, KeywordFilterConfig};
use recommendation::{RecommendationEngine, RecommendationConfig, ViewingEvent};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Port to listen on
    #[arg(short, long, default_value = "8090")]
    port: u16,

    /// Path to ML models directory
    #[arg(short, long, default_value = "./models")]
    models_path: PathBuf,

    /// Enable strict filtering mode
    #[arg(long)]
    strict_mode: bool,

    /// Configuration file path
    #[arg(short, long)]
    config: Option<PathBuf>,
}

#[derive(Clone)]
struct AppState {
    quality_scorer: Arc<ContentQualityScorer>,
    content_moderator: Arc<ContentModerator>,
    spam_detector: Arc<SpamDetector>,
    adult_filter: Arc<AdultContentFilter>,
    keyword_filter: Arc<KeywordFilter>,
    recommendation_engine: Arc<RecommendationEngine>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ContentAnalysisRequest {
    content_id: String,
    content_type: ContentType,
    text: Option<String>,
    image_data: Option<Vec<u8>>,
    video_frames: Option<Vec<Vec<u8>>>,
    audio_data: Option<Vec<f32>>,
    metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
enum ContentType {
    Text,
    Image,
    Video,
    Audio,
    Mixed,
}

#[derive(Debug, Serialize, Deserialize)]
struct ContentAnalysisResponse {
    content_id: String,
    quality_score: f32,
    moderation_result: serde_json::Value,
    spam_detected: bool,
    adult_content: bool,
    keyword_filter_result: serde_json::Value,
    action: ContentAction,
    reasons: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
enum ContentAction {
    Allow,
    Review,
    Blur,
    Block,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let args = Args::parse();

    info!("Starting Content Filter Service on port {}", args.port);

    // Initialize components
    let quality_scorer = Arc::new(ContentQualityScorer::new(QualityScorerConfig::default()));

    let content_moderator = Arc::new(
        ContentModerator::new(&args.models_path)
            .await
            .expect("Failed to initialize content moderator"),
    );

    let spam_detector = Arc::new(
        SpamDetector::new(SpamDetectorConfig {
            strict_mode: args.strict_mode,
            ..Default::default()
        })
        .await
        .expect("Failed to initialize spam detector"),
    );

    let adult_filter = Arc::new(
        AdultContentFilter::new(&args.models_path, FilterConfig {
            strict_mode: args.strict_mode,
            ..Default::default()
        })
        .await
        .expect("Failed to initialize adult content filter"),
    );

    let keyword_filter = Arc::new(
        KeywordFilter::new(KeywordFilterConfig::default())
            .await
            .expect("Failed to initialize keyword filter"),
    );

    let recommendation_engine = Arc::new(
        RecommendationEngine::new(RecommendationConfig::default())
            .await
            .expect("Failed to initialize recommendation engine"),
    );

    let app_state = AppState {
        quality_scorer,
        content_moderator,
        spam_detector,
        adult_filter,
        keyword_filter,
        recommendation_engine,
    };

    // Build router
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/analyze", post(analyze_content))
        .route("/quality/:content_id", get(get_quality_score))
        .route("/moderate", post(moderate_content))
        .route("/spam/check", post(check_spam))
        .route("/adult/check", post(check_adult_content))
        .route("/keywords/filter", post(filter_keywords))
        .route("/recommendations/:user_id", get(get_recommendations))
        .route("/recommendations/update", post(update_user_activity))
        .route("/stats", get(get_statistics))
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(CorsLayer::permissive()),
        )
        .with_state(app_state);

    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    info!("Content Filter Service listening on {}", addr);

    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await?;

    Ok(())
}

async fn health_check() -> &'static str {
    "OK"
}

async fn analyze_content(
    State(state): State<AppState>,
    Json(request): Json<ContentAnalysisRequest>,
) -> Result<Json<ContentAnalysisResponse>, StatusCode> {
    let mut reasons = Vec::new();
    let mut action = ContentAction::Allow;

    // Quality scoring
    let quality_result = if let Some(text) = &request.text {
        state.quality_scorer.score_content(text).await
    } else {
        state.quality_scorer.score_content("").await
    };

    let quality_score = match quality_result {
        Ok(score) => score.overall_score,
        Err(_) => {
            reasons.push("Failed to calculate quality score".to_string());
            0.5
        }
    };

    // Moderation
    let moderation_result = if let Some(text) = &request.text {
        match state.content_moderator.moderate_text(text).await {
            Ok(result) => {
                if !result.safe {
                    action = ContentAction::Review;
                    reasons.push(format!("Content flagged for moderation: {:?}", result.flags));
                }
                serde_json::to_value(result).unwrap_or(serde_json::Value::Null)
            }
            Err(_) => {
                reasons.push("Moderation check failed".to_string());
                serde_json::Value::Null
            }
        }
    } else {
        serde_json::Value::Null
    };

    // Spam detection
    let spam_detected = if let Some(text) = &request.text {
        match state.spam_detector.check_content(&request.content_id, text).await {
            Ok(result) => {
                if result.is_spam {
                    action = ContentAction::Block;
                    reasons.push(format!("Spam detected: {:?}", result.spam_indicators));
                }
                result.is_spam
            }
            Err(_) => {
                reasons.push("Spam check failed".to_string());
                false
            }
        }
    } else {
        false
    };

    // Adult content filtering
    let adult_content = if let Some(image_data) = &request.image_data {
        match state.adult_filter.check_image(image_data).await {
            Ok(result) => {
                if !result.is_safe {
                    if result.requires_blur {
                        action = ContentAction::Blur;
                    } else {
                        action = ContentAction::Block;
                    }
                    reasons.push(format!("Adult content detected: {:?}", result.categories));
                }
                !result.is_safe
            }
            Err(_) => {
                reasons.push("Adult content check failed".to_string());
                false
            }
        }
    } else {
        false
    };

    // Keyword filtering
    let keyword_filter_result = if let Some(text) = &request.text {
        match state.keyword_filter.filter_content(&request.content_id, text).await {
            Ok(result) => {
                if !result.allowed {
                    action = ContentAction::Block;
                    reasons.push(format!("Blocked by keyword filter: {:?}", result.matched_rules));
                }
                serde_json::to_value(result).unwrap_or(serde_json::Value::Null)
            }
            Err(_) => {
                reasons.push("Keyword filter check failed".to_string());
                serde_json::Value::Null
            }
        }
    } else {
        serde_json::Value::Null
    };

    let response = ContentAnalysisResponse {
        content_id: request.content_id,
        quality_score,
        moderation_result,
        spam_detected,
        adult_content,
        keyword_filter_result,
        action,
        reasons,
    };

    Ok(Json(response))
}

async fn get_quality_score(
    State(state): State<AppState>,
    Path(content_id): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    // In production, fetch content from database
    let sample_content = "Sample content for quality scoring";

    match state.quality_scorer.score_content(sample_content).await {
        Ok(score) => Ok(Json(serde_json::to_value(score).unwrap())),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

async fn moderate_content(
    State(state): State<AppState>,
    Json(request): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let text = request["text"].as_str().unwrap_or("");

    match state.content_moderator.moderate_text(text).await {
        Ok(result) => Ok(Json(serde_json::to_value(result).unwrap())),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

async fn check_spam(
    State(state): State<AppState>,
    Json(request): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let content_id = request["content_id"].as_str().unwrap_or("unknown");
    let content = request["content"].as_str().unwrap_or("");

    match state.spam_detector.check_content(content_id, content).await {
        Ok(result) => Ok(Json(serde_json::to_value(result).unwrap())),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

async fn check_adult_content(
    State(state): State<AppState>,
    Json(request): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    // For demo purposes, checking text instead of image
    let text = request["text"].as_str().unwrap_or("");

    match state.adult_filter.check_text(text).await {
        Ok(result) => Ok(Json(serde_json::to_value(result).unwrap())),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

async fn filter_keywords(
    State(state): State<AppState>,
    Json(request): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let content_id = request["content_id"].as_str().unwrap_or("unknown");
    let content = request["content"].as_str().unwrap_or("");

    match state.keyword_filter.filter_content(content_id, content).await {
        Ok(result) => Ok(Json(serde_json::to_value(result).unwrap())),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

async fn get_recommendations(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match state.recommendation_engine.get_recommendations(&user_id).await {
        Ok(recommendations) => Ok(Json(serde_json::to_value(recommendations).unwrap())),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

async fn update_user_activity(
    State(state): State<AppState>,
    Json(request): Json<serde_json::Value>,
) -> Result<StatusCode, StatusCode> {
    let user_id = request["user_id"].as_str().unwrap_or("unknown");
    let content_id = request["content_id"].as_str().unwrap_or("unknown");
    let duration_watched = request["duration_watched"].as_u64().unwrap_or(0) as u32;
    let completed = request["completed"].as_bool().unwrap_or(false);
    let rating = request["rating"].as_f64().map(|r| r as f32);
    let engagement_score = request["engagement_score"].as_f64().unwrap_or(0.5) as f32;

    let event = ViewingEvent {
        content_id: content_id.to_string(),
        timestamp: chrono::Utc::now(),
        duration_watched,
        completed,
        rating,
        engagement_score,
    };

    match state.recommendation_engine.update_user_profile(user_id, event).await {
        Ok(_) => Ok(StatusCode::OK),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

async fn get_statistics(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let mut stats = serde_json::json!({});

    // Get statistics from each component
    let spam_stats = state.spam_detector.get_statistics().await;
    let adult_stats = state.adult_filter.get_statistics().await;
    let keyword_stats = state.keyword_filter.get_statistics().await;
    let recommendation_stats = state.recommendation_engine.get_statistics().await;

    stats["spam_detector"] = serde_json::to_value(spam_stats).unwrap();
    stats["adult_filter"] = serde_json::to_value(adult_stats).unwrap();
    stats["keyword_filter"] = serde_json::to_value(keyword_stats).unwrap();
    stats["recommendation_engine"] = serde_json::to_value(recommendation_stats).unwrap();

    Ok(Json(stats))
}