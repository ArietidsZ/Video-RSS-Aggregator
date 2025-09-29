use anyhow::{Context, Result};
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::{Html, Json},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tower::ServiceBuilder;
use tower_http::cors::{Any, CorsLayer};
use tracing::{info, warn};

use crate::MonitoringSystem;

#[derive(Debug, Deserialize)]
pub struct DashboardQuery {
    component: Option<String>,
    hours: Option<i32>,
    metric: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct DashboardData {
    pub system_status: serde_json::Value,
    pub recent_metrics: Vec<MetricPoint>,
    pub predictions: Vec<serde_json::Value>,
    pub optimizations: Vec<serde_json::Value>,
    pub alerts: Vec<AlertInfo>,
}

#[derive(Debug, Serialize)]
pub struct MetricPoint {
    pub timestamp: String,
    pub component: String,
    pub metric: String,
    pub value: f64,
}

#[derive(Debug, Serialize)]
pub struct AlertInfo {
    pub timestamp: String,
    pub severity: String,
    pub component: String,
    pub message: String,
    pub status: String,
}

#[derive(Clone)]
pub struct DashboardServer {
    monitoring_system: Arc<MonitoringSystem>,
}

impl DashboardServer {
    pub fn new(monitoring_system: Arc<MonitoringSystem>) -> Self {
        Self { monitoring_system }
    }

    pub async fn start(&self, bind_address: &str) -> Result<()> {
        let app = Router::new()
            .route("/", get(dashboard_home))
            .route("/api/status", get(get_system_status))
            .route("/api/metrics", get(get_metrics))
            .route("/api/predictions", get(get_predictions))
            .route("/api/optimizations", get(get_optimizations))
            .route("/api/alerts", get(get_alerts))
            .route("/api/trigger-optimization", post(trigger_optimization))
            .layer(
                ServiceBuilder::new()
                    .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any))
            )
            .with_state(self.clone());

        let listener = tokio::net::TcpListener::bind(bind_address)
            .await
            .context("Failed to bind dashboard server")?;

        info!("Dashboard server starting on {}", bind_address);

        axum::serve(listener, app)
            .await
            .context("Dashboard server error")?;

        Ok(())
    }
}

async fn dashboard_home() -> Html<&'static str> {
    Html(include_str!("dashboard.html"))
}

async fn get_system_status(State(dashboard): State<DashboardServer>) -> Result<Json<serde_json::Value>, StatusCode> {
    match dashboard.monitoring_system.get_system_status().await {
        Ok(status) => Ok(Json(status)),
        Err(e) => {
            warn!("Failed to get system status: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn get_metrics(
    Query(params): Query<DashboardQuery>,
    State(_dashboard): State<DashboardServer>,
) -> Result<Json<Vec<MetricPoint>>, StatusCode> {
    let hours = params.hours.unwrap_or(24);

    // In a real implementation, this would query the metrics database
    let mock_metrics = vec![
        MetricPoint {
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            component: params.component.unwrap_or("api-server".to_string()),
            metric: params.metric.unwrap_or("cpu_usage".to_string()),
            value: 45.2,
        },
        MetricPoint {
            timestamp: "2024-01-01T01:00:00Z".to_string(),
            component: "api-server".to_string(),
            metric: "cpu_usage".to_string(),
            value: 52.1,
        },
    ];

    Ok(Json(mock_metrics))
}

async fn get_predictions(
    Query(_params): Query<DashboardQuery>,
    State(_dashboard): State<DashboardServer>,
) -> Result<Json<Vec<serde_json::Value>>, StatusCode> {
    // Mock predictions data
    let predictions = vec![
        serde_json::json!({
            "timestamp": "2024-01-01T12:00:00Z",
            "component": "api-server",
            "metric": "cpu_usage",
            "current_value": 45.0,
            "predicted_value": 78.5,
            "prediction_horizon_minutes": 30,
            "confidence": 0.85,
            "urgency": "High"
        })
    ];

    Ok(Json(predictions))
}

async fn get_optimizations(
    Query(_params): Query<DashboardQuery>,
    State(_dashboard): State<DashboardServer>,
) -> Result<Json<Vec<serde_json::Value>>, StatusCode> {
    // Mock optimization data
    let optimizations = vec![
        serde_json::json!({
            "timestamp": "2024-01-01T11:30:00Z",
            "component": "api-server",
            "action": "ScaleUp",
            "from_replicas": 3,
            "to_replicas": 4,
            "expected_improvement": 25.0,
            "status": "Completed"
        })
    ];

    Ok(Json(optimizations))
}

async fn get_alerts(
    Query(_params): Query<DashboardQuery>,
    State(_dashboard): State<DashboardServer>,
) -> Result<Json<Vec<AlertInfo>>, StatusCode> {
    // Mock alerts data
    let alerts = vec![
        AlertInfo {
            timestamp: "2024-01-01T12:15:00Z".to_string(),
            severity: "Warning".to_string(),
            component: "transcription-service".to_string(),
            message: "High memory usage detected (85%)".to_string(),
            status: "Active".to_string(),
        }
    ];

    Ok(Json(alerts))
}

#[derive(Deserialize)]
struct OptimizationRequest {
    component: String,
    action: String,
}

async fn trigger_optimization(
    State(_dashboard): State<DashboardServer>,
    Json(_payload): Json<OptimizationRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    // In a real implementation, this would trigger the optimizer
    Ok(Json(serde_json::json!({
        "success": true,
        "message": "Optimization triggered successfully"
    })))
}