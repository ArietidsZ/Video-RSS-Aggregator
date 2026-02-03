use std::env;
use std::sync::Arc;

use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::auth::Auth;
use crate::pipeline::{IngestReport, Pipeline, ProcessReport};

#[derive(Clone)]
pub struct AppState {
    pipeline: Arc<Pipeline>,
    auth: Auth,
    rss_title: String,
    rss_link: String,
    rss_description: String,
}

impl AppState {
    pub fn new(pipeline: Pipeline, api_key: Option<String>) -> Self {
        let rss_title = env::var("VRA_RSS_TITLE")
            .unwrap_or_else(|_| "Video RSS Aggregator".to_string());
        let rss_link = env::var("VRA_RSS_LINK")
            .unwrap_or_else(|_| "http://localhost:8080/rss".to_string());
        let rss_description = env::var("VRA_RSS_DESCRIPTION")
            .unwrap_or_else(|_| "Video summaries".to_string());

        Self {
            pipeline: Arc::new(pipeline),
            auth: Auth::new(api_key),
            rss_title,
            rss_link,
            rss_description,
        }
    }
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/ingest", post(ingest))
        .route("/process", post(process))
        .route("/rss", get(rss_feed))
        .with_state(state)
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        timestamp: Utc::now(),
    })
}

async fn ingest(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<IngestRequest>,
) -> Result<Json<IngestReport>, ApiError> {
    if !state.auth.is_authorized(&headers) {
        return Err(ApiError::Unauthorized);
    }

    let report = state
        .pipeline
        .ingest_feed(
            &request.feed_url,
            request.process.unwrap_or(false),
            request.max_items,
        )
        .await?;

    Ok(Json(report))
}

async fn process(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<ProcessRequest>,
) -> Result<Json<ProcessReport>, ApiError> {
    if !state.auth.is_authorized(&headers) {
        return Err(ApiError::Unauthorized);
    }

    let report = state
        .pipeline
        .process_source(&request.source_url, request.title)
        .await?;

    Ok(Json(report))
}

async fn rss_feed(
    State(state): State<AppState>,
    Query(query): Query<RssQuery>,
) -> Result<Response, ApiError> {
    let limit = query.limit.unwrap_or(20).max(1).min(200) as usize;
    let feed = state
        .pipeline
        .rss_feed(
            &state.rss_title,
            &state.rss_link,
            &state.rss_description,
            limit,
        )
        .await?;

    Ok((StatusCode::OK, feed).into_response())
}

#[derive(Debug, Deserialize)]
pub struct IngestRequest {
    pub feed_url: String,
    pub process: Option<bool>,
    pub max_items: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct ProcessRequest {
    pub source_url: String,
    pub title: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct RssQuery {
    pub limit: Option<i64>,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("unauthorized")]
    Unauthorized,
    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        match self {
            ApiError::Unauthorized => (StatusCode::UNAUTHORIZED, "unauthorized").into_response(),
            ApiError::Internal(err) => {
                (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response()
            }
        }
    }
}
