use anyhow::{Context, Result};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::collections::HashMap;
use tracing::{debug, info, warn};

use linfa::prelude::*;
use linfa_linear::LinearRegression;
use ndarray::prelude::*;
use statrs::statistics::{Statistics, Data};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingPrediction {
    pub timestamp: DateTime<Utc>,
    pub component: String,
    pub metric: String,
    pub current_value: f64,
    pub predicted_value: f64,
    pub prediction_horizon_minutes: i32,
    pub confidence: f64,
    pub recommended_action: ScalingAction,
    pub urgency: PredictionUrgency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingAction {
    ScaleUp { target_replicas: i32, resource_increase: f64 },
    ScaleDown { target_replicas: i32, resource_decrease: f64 },
    Maintain { current_replicas: i32 },
    Alert { message: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PredictionUrgency {
    Critical,   // Action needed within 5 minutes
    High,       // Action needed within 15 minutes
    Medium,     // Action needed within 1 hour
    Low,        // Monitor for trends
}

#[derive(Debug, Clone)]
pub struct MetricHistory {
    pub timestamps: Vec<DateTime<Utc>>,
    pub values: Vec<f64>,
    pub component: String,
    pub metric: String,
}

#[derive(Debug, Clone)]
pub struct PredictionModel {
    pub component: String,
    pub metric: String,
    pub model_type: ModelType,
    pub accuracy: f64,
    pub last_trained: DateTime<Utc>,
    pub min_samples: usize,
}

#[derive(Debug, Clone)]
pub enum ModelType {
    LinearRegression,
    MovingAverage { window_size: usize },
    ExponentialSmoothing { alpha: f64 },
    SeasonalDecomposition,
}

pub struct ScalingPredictor {
    database: PgPool,
    models: HashMap<String, PredictionModel>,
    prediction_horizons: Vec<i32>, // Minutes: 5, 15, 30, 60, 120
    min_prediction_samples: usize,
    confidence_threshold: f64,
}

impl ScalingPredictor {
    pub async fn new(database_url: &str) -> Result<Self> {
        let database = PgPool::connect(database_url)
            .await
            .context("Failed to connect to database for predictor")?;

        // Initialize prediction tables
        Self::init_database(&database).await?;

        Ok(Self {
            database,
            models: HashMap::new(),
            prediction_horizons: vec![5, 15, 30, 60, 120],
            min_prediction_samples: 20,
            confidence_threshold: 0.7,
        })
    }

    async fn init_database(database: &PgPool) -> Result<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS scaling_predictions (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                component TEXT NOT NULL,
                metric TEXT NOT NULL,
                current_value DOUBLE PRECISION NOT NULL,
                predicted_value DOUBLE PRECISION NOT NULL,
                prediction_horizon_minutes INTEGER NOT NULL,
                confidence DOUBLE PRECISION NOT NULL,
                recommended_action JSONB NOT NULL,
                urgency TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_scaling_predictions_component_metric
            ON scaling_predictions(component, metric, timestamp);

            CREATE TABLE IF NOT EXISTS prediction_accuracy (
                id SERIAL PRIMARY KEY,
                component TEXT NOT NULL,
                metric TEXT NOT NULL,
                model_type TEXT NOT NULL,
                predicted_value DOUBLE PRECISION NOT NULL,
                actual_value DOUBLE PRECISION NOT NULL,
                prediction_horizon_minutes INTEGER NOT NULL,
                error_percentage DOUBLE PRECISION NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_prediction_accuracy_component_metric
            ON prediction_accuracy(component, metric, timestamp);
            "#,
        )
        .execute(database)
        .await
        .context("Failed to initialize prediction database tables")?;

        Ok(())
    }

    pub async fn update_predictions(&self) -> Result<()> {
        info!("Starting prediction update cycle");

        let components = self.get_monitored_components().await?;

        for component in components {
            if let Err(e) = self.update_component_predictions(&component).await {
                warn!("Failed to update predictions for component {}: {}", component, e);
            }
        }

        // Clean up old predictions
        self.cleanup_old_predictions().await?;

        info!("Prediction update cycle completed");
        Ok(())
    }

    async fn get_monitored_components(&self) -> Result<Vec<String>> {
        let components = sqlx::query_scalar!(
            "SELECT DISTINCT component FROM performance_metrics WHERE timestamp > NOW() - INTERVAL '1 hour'"
        )
        .fetch_all(&self.database)
        .await
        .context("Failed to get monitored components")?;

        Ok(components.into_iter().flatten().collect())
    }

    async fn update_component_predictions(&self, component: &str) -> Result<()> {
        let metrics = self.get_component_metrics(component).await?;

        for metric in &["cpu_usage", "memory_usage", "request_rate", "error_rate", "response_time"] {
            if let Some(history) = metrics.get(metric) {
                if history.values.len() >= self.min_prediction_samples {
                    if let Err(e) = self.generate_predictions_for_metric(history).await {
                        warn!("Failed to generate predictions for {}.{}: {}", component, metric, e);
                    }
                }
            }
        }

        Ok(())
    }

    async fn get_component_metrics(&self, component: &str) -> Result<HashMap<String, MetricHistory>> {
        let mut metrics = HashMap::new();

        let rows = sqlx::query!(
            r#"
            SELECT metric_name, metric_value, timestamp
            FROM performance_metrics
            WHERE component = $1
            AND timestamp > NOW() - INTERVAL '4 hours'
            ORDER BY metric_name, timestamp
            "#,
            component
        )
        .fetch_all(&self.database)
        .await
        .context("Failed to fetch component metrics")?;

        for row in rows {
            let entry = metrics.entry(row.metric_name.clone()).or_insert_with(|| MetricHistory {
                timestamps: Vec::new(),
                values: Vec::new(),
                component: component.to_string(),
                metric: row.metric_name.clone(),
            });

            entry.timestamps.push(row.timestamp);
            entry.values.push(row.metric_value);
        }

        Ok(metrics)
    }

    async fn generate_predictions_for_metric(&self, history: &MetricHistory) -> Result<()> {
        debug!("Generating predictions for {}.{}", history.component, history.metric);

        let model = self.get_or_train_model(history)?;

        for &horizon_minutes in &self.prediction_horizons {
            if let Ok(prediction) = self.predict_value(history, &model, horizon_minutes) {
                if prediction.confidence >= self.confidence_threshold {
                    self.store_prediction(&prediction).await?;

                    // Check if immediate action is needed
                    if matches!(prediction.urgency, PredictionUrgency::Critical | PredictionUrgency::High) {
                        self.trigger_scaling_action(&prediction).await?;
                    }
                }
            }
        }

        Ok(())
    }

    fn get_or_train_model(&self, history: &MetricHistory) -> Result<PredictionModel> {
        let model_key = format!("{}_{}", history.component, history.metric);

        if let Some(existing_model) = self.models.get(&model_key) {
            // Check if model needs retraining (older than 1 hour)
            if Utc::now() - existing_model.last_trained < Duration::hours(1) {
                return Ok(existing_model.clone());
            }
        }

        // Train new model
        self.train_model(history)
    }

    fn train_model(&self, history: &MetricHistory) -> Result<PredictionModel> {
        let n = history.values.len();
        if n < self.min_prediction_samples {
            return Err(anyhow::anyhow!("Insufficient data for training"));
        }

        // Choose best model based on metric characteristics
        let model_type = self.select_best_model_type(history)?;
        let accuracy = self.evaluate_model_accuracy(history, &model_type)?;

        Ok(PredictionModel {
            component: history.component.clone(),
            metric: history.metric.clone(),
            model_type,
            accuracy,
            last_trained: Utc::now(),
            min_samples: self.min_prediction_samples,
        })
    }

    fn select_best_model_type(&self, history: &MetricHistory) -> Result<ModelType> {
        let values = &history.values;
        let data = Data::new(values.clone());

        // Calculate trend and seasonality indicators
        let variance = data.variance();
        let mean = data.mean();
        let cv = if mean > 0.0 { (variance.sqrt() / mean) } else { 0.0 };

        // Detect seasonality (simple check for recurring patterns)
        let has_seasonality = self.detect_seasonality(values);

        // Choose model based on data characteristics
        if has_seasonality {
            Ok(ModelType::SeasonalDecomposition)
        } else if cv < 0.2 {
            // Low variability - use moving average
            Ok(ModelType::MovingAverage { window_size: 10 })
        } else if cv < 0.5 {
            // Medium variability - use exponential smoothing
            Ok(ModelType::ExponentialSmoothing { alpha: 0.3 })
        } else {
            // High variability - use linear regression
            Ok(ModelType::LinearRegression)
        }
    }

    fn detect_seasonality(&self, values: &[f64]) -> bool {
        if values.len() < 24 { // Need at least 24 data points
            return false;
        }

        // Simple autocorrelation check for hourly patterns
        let autocorr = self.calculate_autocorrelation(values, 12); // Check 12-point lag
        autocorr.abs() > 0.3
    }

    fn calculate_autocorrelation(&self, values: &[f64], lag: usize) -> f64 {
        if values.len() <= lag {
            return 0.0;
        }

        let n = values.len() - lag;
        let data = Data::new(values.to_vec());
        let mean = data.mean();

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in 0..n {
            let x_dev = values[i] - mean;
            let y_dev = values[i + lag] - mean;
            numerator += x_dev * y_dev;
            denominator += x_dev * x_dev;
        }

        if denominator == 0.0 { 0.0 } else { numerator / denominator }
    }

    fn evaluate_model_accuracy(&self, history: &MetricHistory, model_type: &ModelType) -> Result<f64> {
        let values = &history.values;
        let n = values.len();
        let test_size = std::cmp::min(n / 4, 20); // Use last 25% or 20 points for testing
        let train_size = n - test_size;

        if train_size < self.min_prediction_samples {
            return Ok(0.5); // Default accuracy for insufficient data
        }

        let train_data = &values[..train_size];
        let test_data = &values[train_size..];

        let mut errors = Vec::new();

        for i in 0..test_data.len() {
            let predicted = self.predict_with_model(train_data, model_type, 1)?;
            let actual = test_data[i];
            let error = ((predicted - actual) / actual).abs();
            errors.push(error);
        }

        let data = Data::new(errors);
        let mean_error = data.mean();
        let accuracy = (1.0 - mean_error).max(0.0).min(1.0);

        Ok(accuracy)
    }

    fn predict_with_model(&self, data: &[f64], model_type: &ModelType, steps_ahead: usize) -> Result<f64> {
        match model_type {
            ModelType::LinearRegression => self.linear_regression_predict(data, steps_ahead),
            ModelType::MovingAverage { window_size } => self.moving_average_predict(data, *window_size),
            ModelType::ExponentialSmoothing { alpha } => self.exponential_smoothing_predict(data, *alpha, steps_ahead),
            ModelType::SeasonalDecomposition => self.seasonal_predict(data, steps_ahead),
        }
    }

    fn linear_regression_predict(&self, data: &[f64], steps_ahead: usize) -> Result<f64> {
        let n = data.len();
        let x: Array1<f64> = Array1::from_iter((0..n).map(|i| i as f64));
        let y: Array1<f64> = Array1::from_vec(data.to_vec());

        let x_matrix = x.view().insert_axis(Axis(1));
        let dataset = Dataset::new(x_matrix, y);

        let model = LinearRegression::default().fit(&dataset)?;

        let future_x = Array1::from_vec(vec![(n + steps_ahead - 1) as f64]);
        let future_x_matrix = future_x.view().insert_axis(Axis(1));
        let prediction = model.predict(&future_x_matrix);

        Ok(prediction[0])
    }

    fn moving_average_predict(&self, data: &[f64], window_size: usize) -> Result<f64> {
        let n = data.len();
        let start = if n >= window_size { n - window_size } else { 0 };
        let window = &data[start..];

        let sum: f64 = window.iter().sum();
        Ok(sum / window.len() as f64)
    }

    fn exponential_smoothing_predict(&self, data: &[f64], alpha: f64, _steps_ahead: usize) -> Result<f64> {
        let mut smoothed = data[0];

        for &value in &data[1..] {
            smoothed = alpha * value + (1.0 - alpha) * smoothed;
        }

        Ok(smoothed)
    }

    fn seasonal_predict(&self, data: &[f64], steps_ahead: usize) -> Result<f64> {
        // Simple seasonal prediction using 24-hour cycle
        let cycle_length = 24;
        let n = data.len();

        if n < cycle_length {
            return self.moving_average_predict(data, data.len());
        }

        let seasonal_index = (n + steps_ahead - 1) % cycle_length;
        let cycles_back = std::cmp::min(3, n / cycle_length);

        let mut seasonal_values = Vec::new();
        for i in 0..cycles_back {
            let idx = n - 1 - (i * cycle_length) + seasonal_index;
            if idx < n {
                seasonal_values.push(data[idx]);
            }
        }

        if seasonal_values.is_empty() {
            self.moving_average_predict(data, std::cmp::min(10, data.len()))
        } else {
            let sum: f64 = seasonal_values.iter().sum();
            Ok(sum / seasonal_values.len() as f64)
        }
    }

    fn predict_value(&self, history: &MetricHistory, model: &PredictionModel, horizon_minutes: i32) -> Result<ScalingPrediction> {
        let steps_ahead = (horizon_minutes / 5).max(1) as usize; // Assuming 5-minute intervals
        let predicted_value = self.predict_with_model(&history.values, &model.model_type, steps_ahead)?;
        let current_value = *history.values.last().unwrap();

        let recommended_action = self.determine_scaling_action(
            &history.component,
            &history.metric,
            current_value,
            predicted_value,
        );

        let urgency = self.determine_urgency(&history.metric, current_value, predicted_value, horizon_minutes);

        Ok(ScalingPrediction {
            timestamp: Utc::now(),
            component: history.component.clone(),
            metric: history.metric.clone(),
            current_value,
            predicted_value,
            prediction_horizon_minutes: horizon_minutes,
            confidence: model.accuracy,
            recommended_action,
            urgency,
        })
    }

    fn determine_scaling_action(&self, component: &str, metric: &str, current: f64, predicted: f64) -> ScalingAction {
        let change_percent = ((predicted - current) / current) * 100.0;

        match metric {
            "cpu_usage" | "memory_usage" => {
                if predicted > 80.0 && change_percent > 10.0 {
                    ScalingAction::ScaleUp {
                        target_replicas: self.calculate_target_replicas(component, 1.5),
                        resource_increase: 0.3
                    }
                } else if predicted < 30.0 && change_percent < -20.0 {
                    ScalingAction::ScaleDown {
                        target_replicas: self.calculate_target_replicas(component, 0.7),
                        resource_decrease: 0.2
                    }
                } else {
                    ScalingAction::Maintain { current_replicas: self.get_current_replicas(component) }
                }
            },
            "request_rate" => {
                if change_percent > 50.0 {
                    ScalingAction::ScaleUp {
                        target_replicas: self.calculate_target_replicas(component, 1.3),
                        resource_increase: 0.2
                    }
                } else if change_percent < -40.0 {
                    ScalingAction::ScaleDown {
                        target_replicas: self.calculate_target_replicas(component, 0.8),
                        resource_decrease: 0.15
                    }
                } else {
                    ScalingAction::Maintain { current_replicas: self.get_current_replicas(component) }
                }
            },
            "error_rate" => {
                if predicted > 5.0 {
                    ScalingAction::Alert {
                        message: format!("High error rate predicted for {}: {:.2}%", component, predicted)
                    }
                } else {
                    ScalingAction::Maintain { current_replicas: self.get_current_replicas(component) }
                }
            },
            _ => ScalingAction::Maintain { current_replicas: self.get_current_replicas(component) }
        }
    }

    fn calculate_target_replicas(&self, _component: &str, scale_factor: f64) -> i32 {
        let current = self.get_current_replicas(_component);
        ((current as f64 * scale_factor).round() as i32).max(1).min(20)
    }

    fn get_current_replicas(&self, _component: &str) -> i32 {
        // In a real implementation, this would query Kubernetes or container orchestrator
        3 // Default replica count
    }

    fn determine_urgency(&self, metric: &str, current: f64, predicted: f64, horizon_minutes: i32) -> PredictionUrgency {
        let change_percent = ((predicted - current) / current).abs() * 100.0;

        match metric {
            "cpu_usage" | "memory_usage" => {
                if predicted > 90.0 || change_percent > 30.0 {
                    if horizon_minutes <= 5 {
                        PredictionUrgency::Critical
                    } else if horizon_minutes <= 15 {
                        PredictionUrgency::High
                    } else {
                        PredictionUrgency::Medium
                    }
                } else if predicted > 70.0 || change_percent > 20.0 {
                    if horizon_minutes <= 15 {
                        PredictionUrgency::Medium
                    } else {
                        PredictionUrgency::Low
                    }
                } else {
                    PredictionUrgency::Low
                }
            },
            "error_rate" => {
                if predicted > 10.0 {
                    PredictionUrgency::Critical
                } else if predicted > 5.0 {
                    PredictionUrgency::High
                } else {
                    PredictionUrgency::Low
                }
            },
            _ => {
                if change_percent > 50.0 {
                    PredictionUrgency::Medium
                } else {
                    PredictionUrgency::Low
                }
            }
        }
    }

    async fn store_prediction(&self, prediction: &ScalingPrediction) -> Result<()> {
        let action_json = serde_json::to_value(&prediction.recommended_action)?;

        sqlx::query!(
            r#"
            INSERT INTO scaling_predictions
            (component, metric, current_value, predicted_value, prediction_horizon_minutes,
             confidence, recommended_action, urgency)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            "#,
            prediction.component,
            prediction.metric,
            prediction.current_value,
            prediction.predicted_value,
            prediction.prediction_horizon_minutes,
            prediction.confidence,
            action_json,
            format!("{:?}", prediction.urgency)
        )
        .execute(&self.database)
        .await
        .context("Failed to store prediction")?;

        Ok(())
    }

    async fn trigger_scaling_action(&self, prediction: &ScalingPrediction) -> Result<()> {
        match &prediction.recommended_action {
            ScalingAction::ScaleUp { target_replicas, .. } => {
                info!("Triggering scale up for {} to {} replicas", prediction.component, target_replicas);
                // In production, this would call Kubernetes API or container orchestrator
            },
            ScalingAction::ScaleDown { target_replicas, .. } => {
                info!("Triggering scale down for {} to {} replicas", prediction.component, target_replicas);
                // In production, this would call Kubernetes API or container orchestrator
            },
            ScalingAction::Alert { message } => {
                warn!("Scaling alert: {}", message);
                // In production, this would send notifications via Slack, email, etc.
            },
            ScalingAction::Maintain { .. } => {
                debug!("Maintaining current scale for {}", prediction.component);
            }
        }

        Ok(())
    }

    async fn cleanup_old_predictions(&self) -> Result<()> {
        sqlx::query!(
            "DELETE FROM scaling_predictions WHERE created_at < NOW() - INTERVAL '24 hours'"
        )
        .execute(&self.database)
        .await
        .context("Failed to cleanup old predictions")?;

        Ok(())
    }

    pub async fn get_recent_predictions(&self, component: Option<&str>, hours: i32) -> Result<Vec<ScalingPrediction>> {
        let query = if let Some(comp) = component {
            sqlx::query!(
                r#"
                SELECT component, metric, current_value, predicted_value,
                       prediction_horizon_minutes, confidence, recommended_action, urgency, timestamp
                FROM scaling_predictions
                WHERE component = $1 AND timestamp > NOW() - INTERVAL '%s hours'
                ORDER BY timestamp DESC
                "#,
                comp,
                hours.to_string()
            )
        } else {
            sqlx::query!(
                r#"
                SELECT component, metric, current_value, predicted_value,
                       prediction_horizon_minutes, confidence, recommended_action, urgency, timestamp
                FROM scaling_predictions
                WHERE timestamp > NOW() - INTERVAL '%s hours'
                ORDER BY timestamp DESC
                "#,
                hours.to_string()
            )
        };

        let rows = query.fetch_all(&self.database).await?;

        let mut predictions = Vec::new();
        for row in rows {
            let action: ScalingAction = serde_json::from_value(row.recommended_action)?;
            let urgency: PredictionUrgency = match row.urgency.as_str() {
                "Critical" => PredictionUrgency::Critical,
                "High" => PredictionUrgency::High,
                "Medium" => PredictionUrgency::Medium,
                _ => PredictionUrgency::Low,
            };

            predictions.push(ScalingPrediction {
                timestamp: row.timestamp,
                component: row.component,
                metric: row.metric,
                current_value: row.current_value,
                predicted_value: row.predicted_value,
                prediction_horizon_minutes: row.prediction_horizon_minutes,
                confidence: row.confidence,
                recommended_action: action,
                urgency,
            });
        }

        Ok(predictions)
    }
}