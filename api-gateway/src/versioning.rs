use anyhow::{Context, Result};
use axum::{
    extract::{Path, Query, State},
    headers::{self, HeaderName, HeaderValue},
    http::{header, HeaderMap, Method, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Json, Response},
    routing::{get, post, put, delete},
    Router, TypedHeader,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

// API Version Management

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ApiVersion {
    V1,
    V2,
    V3,
}

impl ApiVersion {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "v1" | "1" | "1.0" => Some(ApiVersion::V1),
            "v2" | "2" | "2.0" => Some(ApiVersion::V2),
            "v3" | "3" | "3.0" => Some(ApiVersion::V3),
            _ => None,
        }
    }

    pub fn to_string(&self) -> String {
        match self {
            ApiVersion::V1 => "v1".to_string(),
            ApiVersion::V2 => "v2".to_string(),
            ApiVersion::V3 => "v3".to_string(),
        }
    }

    pub fn is_deprecated(&self) -> bool {
        matches!(self, ApiVersion::V1)
    }

    pub fn sunset_date(&self) -> Option<DateTime<Utc>> {
        match self {
            ApiVersion::V1 => Some(chrono::Utc::now() + chrono::Duration::days(180)),
            _ => None,
        }
    }
}

impl Default for ApiVersion {
    fn default() -> Self {
        ApiVersion::V3  // Latest version as default
    }
}

// Version Strategy

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VersionStrategy {
    UrlPath,        // /api/v1/resource
    QueryParam,     // /api/resource?version=1
    Header,         // X-API-Version: 1
    Accept,         // Accept: application/vnd.api+json;version=1
    ContentType,    // Content-Type: application/vnd.api.v1+json
}

// Version Configuration

#[derive(Debug, Clone)]
pub struct VersionConfig {
    pub strategy: VersionStrategy,
    pub default_version: ApiVersion,
    pub supported_versions: Vec<ApiVersion>,
    pub deprecated_versions: Vec<ApiVersion>,
    pub enable_version_discovery: bool,
    pub enable_deprecation_warnings: bool,
    pub enforce_version_requirement: bool,
}

impl Default for VersionConfig {
    fn default() -> Self {
        Self {
            strategy: VersionStrategy::UrlPath,
            default_version: ApiVersion::V3,
            supported_versions: vec![ApiVersion::V1, ApiVersion::V2, ApiVersion::V3],
            deprecated_versions: vec![ApiVersion::V1],
            enable_version_discovery: true,
            enable_deprecation_warnings: true,
            enforce_version_requirement: false,
        }
    }
}

// Version Manager

#[derive(Debug, Clone)]
pub struct VersionManager {
    config: VersionConfig,
    handlers: Arc<RwLock<HashMap<(ApiVersion, String), HandlerInfo>>>,
    transformers: Arc<RwLock<HashMap<(ApiVersion, ApiVersion), TransformFn>>>,
    metrics: Arc<RwLock<VersionMetrics>>,
}

#[derive(Debug, Clone)]
struct HandlerInfo {
    pub version: ApiVersion,
    pub path: String,
    pub method: Method,
    pub handler_name: String,
    pub deprecated: bool,
    pub replacement: Option<String>,
}

type TransformFn = Arc<dyn Fn(Value) -> Result<Value> + Send + Sync>;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct VersionMetrics {
    pub requests_by_version: HashMap<ApiVersion, u64>,
    pub deprecated_version_usage: HashMap<ApiVersion, u64>,
    pub version_negotiation_failures: u64,
    pub transformation_count: u64,
}

impl VersionManager {
    pub fn new(config: VersionConfig) -> Self {
        Self {
            config,
            handlers: Arc::new(RwLock::new(HashMap::new())),
            transformers: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(VersionMetrics::default())),
        }
    }

    pub async fn register_handler(
        &self,
        version: ApiVersion,
        path: &str,
        method: Method,
        handler_name: &str,
    ) -> Result<()> {
        let mut handlers = self.handlers.write().await;

        let key = (version, format!("{}:{}", method, path));
        let info = HandlerInfo {
            version,
            path: path.to_string(),
            method,
            handler_name: handler_name.to_string(),
            deprecated: self.config.deprecated_versions.contains(&version),
            replacement: if version == ApiVersion::V1 {
                Some(format!("v2{}", path))
            } else {
                None
            },
        };

        handlers.insert(key, info);

        info!("Registered handler: {} {} {} -> {}",
              version.to_string(), method, path, handler_name);

        Ok(())
    }

    pub async fn register_transformer(
        &self,
        from_version: ApiVersion,
        to_version: ApiVersion,
        transformer: TransformFn,
    ) -> Result<()> {
        let mut transformers = self.transformers.write().await;
        transformers.insert((from_version, to_version), transformer);

        info!("Registered transformer: {} -> {}",
              from_version.to_string(), to_version.to_string());

        Ok(())
    }

    pub fn extract_version(&self, headers: &HeaderMap, uri: &str, query: &str) -> ApiVersion {
        match self.config.strategy {
            VersionStrategy::UrlPath => {
                // Extract version from URL path: /api/v1/resource
                if let Some(captures) = regex::Regex::new(r"/v(\d+)/")
                    .unwrap()
                    .captures(uri) {
                    if let Some(version_str) = captures.get(1) {
                        return ApiVersion::from_str(version_str.as_str())
                            .unwrap_or(self.config.default_version);
                    }
                }
            }
            VersionStrategy::QueryParam => {
                // Extract version from query parameter: ?version=1
                if let Ok(params) = serde_urlencoded::from_str::<HashMap<String, String>>(query) {
                    if let Some(version_str) = params.get("version") {
                        return ApiVersion::from_str(version_str)
                            .unwrap_or(self.config.default_version);
                    }
                }
            }
            VersionStrategy::Header => {
                // Extract version from custom header: X-API-Version
                if let Some(version_header) = headers.get("x-api-version") {
                    if let Ok(version_str) = version_header.to_str() {
                        return ApiVersion::from_str(version_str)
                            .unwrap_or(self.config.default_version);
                    }
                }
            }
            VersionStrategy::Accept => {
                // Extract version from Accept header
                if let Some(accept_header) = headers.get("accept") {
                    if let Ok(accept_str) = accept_header.to_str() {
                        if let Some(captures) = regex::Regex::new(r"version=(\d+)")
                            .unwrap()
                            .captures(accept_str) {
                            if let Some(version_str) = captures.get(1) {
                                return ApiVersion::from_str(version_str.as_str())
                                    .unwrap_or(self.config.default_version);
                            }
                        }
                    }
                }
            }
            VersionStrategy::ContentType => {
                // Extract version from Content-Type header
                if let Some(content_type) = headers.get("content-type") {
                    if let Ok(ct_str) = content_type.to_str() {
                        if let Some(captures) = regex::Regex::new(r"\.v(\d+)\+")
                            .unwrap()
                            .captures(ct_str) {
                            if let Some(version_str) = captures.get(1) {
                                return ApiVersion::from_str(version_str.as_str())
                                    .unwrap_or(self.config.default_version);
                            }
                        }
                    }
                }
            }
        }

        self.config.default_version
    }

    pub async fn validate_version(&self, version: ApiVersion) -> Result<()> {
        if !self.config.supported_versions.contains(&version) {
            let mut metrics = self.metrics.write().await;
            metrics.version_negotiation_failures += 1;

            return Err(anyhow::anyhow!(
                "API version {} is not supported. Supported versions: {:?}",
                version.to_string(),
                self.config.supported_versions
            ));
        }

        // Update metrics
        let mut metrics = self.metrics.write().await;
        *metrics.requests_by_version.entry(version).or_insert(0) += 1;

        if self.config.deprecated_versions.contains(&version) {
            *metrics.deprecated_version_usage.entry(version).or_insert(0) += 1;
        }

        Ok(())
    }

    pub async fn transform_request(
        &self,
        from_version: ApiVersion,
        to_version: ApiVersion,
        data: Value,
    ) -> Result<Value> {
        if from_version == to_version {
            return Ok(data);
        }

        let transformers = self.transformers.read().await;

        if let Some(transformer) = transformers.get(&(from_version, to_version)) {
            let mut metrics = self.metrics.write().await;
            metrics.transformation_count += 1;

            transformer(data)
        } else {
            // Try to find a chain of transformations
            self.chain_transform(from_version, to_version, data).await
        }
    }

    async fn chain_transform(
        &self,
        from_version: ApiVersion,
        to_version: ApiVersion,
        mut data: Value,
    ) -> Result<Value> {
        // Simple chaining: V1 -> V2 -> V3
        let versions = [ApiVersion::V1, ApiVersion::V2, ApiVersion::V3];

        let from_idx = versions.iter().position(|v| *v == from_version)
            .ok_or_else(|| anyhow::anyhow!("Unknown version"))?;

        let to_idx = versions.iter().position(|v| *v == to_version)
            .ok_or_else(|| anyhow::anyhow!("Unknown version"))?;

        if from_idx < to_idx {
            // Transform forward
            for i in from_idx..to_idx {
                data = self.transform_request(versions[i], versions[i + 1], data).await?;
            }
        } else {
            // Transform backward (if supported)
            for i in (to_idx + 1..=from_idx).rev() {
                data = self.transform_request(versions[i], versions[i - 1], data).await?;
            }
        }

        Ok(data)
    }

    pub fn add_deprecation_headers(&self, version: ApiVersion, headers: &mut HeaderMap) {
        if !self.config.enable_deprecation_warnings {
            return;
        }

        if version.is_deprecated() {
            headers.insert(
                HeaderName::from_static("x-api-deprecated"),
                HeaderValue::from_static("true"),
            );

            if let Some(sunset_date) = version.sunset_date() {
                headers.insert(
                    HeaderName::from_static("sunset"),
                    HeaderValue::from_str(&sunset_date.to_rfc2822()).unwrap(),
                );
            }

            headers.insert(
                HeaderName::from_static("x-api-deprecation-message"),
                HeaderValue::from_str(&format!(
                    "Version {} is deprecated. Please upgrade to {}",
                    version.to_string(),
                    self.config.default_version.to_string()
                )).unwrap(),
            );
        }
    }

    pub async fn get_metrics(&self) -> VersionMetrics {
        self.metrics.read().await.clone()
    }
}

// Middleware for version handling

pub async fn version_middleware<B>(
    State(manager): State<Arc<VersionManager>>,
    headers: HeaderMap,
    uri: axum::http::Uri,
    request: axum::http::Request<B>,
    next: Next<B>,
) -> Result<Response, StatusCode> {
    let query = uri.query().unwrap_or("");
    let version = manager.extract_version(&headers, uri.path(), query);

    // Validate version
    if let Err(e) = manager.validate_version(version).await {
        return Ok(Json(serde_json::json!({
            "error": "Invalid API Version",
            "message": e.to_string(),
            "supported_versions": manager.config.supported_versions,
        })).into_response());
    }

    // Add version to request extensions for handlers
    let mut request = request;
    request.extensions_mut().insert(version);

    // Process request
    let mut response = next.run(request).await;

    // Add version headers to response
    response.headers_mut().insert(
        HeaderName::from_static("x-api-version"),
        HeaderValue::from_str(&version.to_string()).unwrap(),
    );

    // Add deprecation headers if applicable
    manager.add_deprecation_headers(version, response.headers_mut());

    Ok(response)
}

// Version-specific handlers

pub mod v1 {
    use super::*;

    #[derive(Debug, Serialize, Deserialize)]
    pub struct VideoV1 {
        pub id: String,
        pub title: String,
        pub url: String,
    }

    pub async fn get_videos() -> Json<Vec<VideoV1>> {
        Json(vec![
            VideoV1 {
                id: "1".to_string(),
                title: "Video 1".to_string(),
                url: "https://example.com/1".to_string(),
            },
        ])
    }
}

pub mod v2 {
    use super::*;

    #[derive(Debug, Serialize, Deserialize)]
    pub struct VideoV2 {
        pub id: String,
        pub title: String,
        pub description: String,
        pub url: String,
        pub duration_seconds: i32,
    }

    pub async fn get_videos() -> Json<Vec<VideoV2>> {
        Json(vec![
            VideoV2 {
                id: "1".to_string(),
                title: "Video 1".to_string(),
                description: "Description".to_string(),
                url: "https://example.com/1".to_string(),
                duration_seconds: 300,
            },
        ])
    }
}

pub mod v3 {
    use super::*;

    #[derive(Debug, Serialize, Deserialize)]
    pub struct VideoV3 {
        pub id: Uuid,
        pub title: String,
        pub description: String,
        pub url: String,
        pub duration_seconds: i32,
        pub quality_score: f32,
        pub metadata: HashMap<String, String>,
    }

    pub async fn get_videos() -> Json<Vec<VideoV3>> {
        Json(vec![
            VideoV3 {
                id: Uuid::new_v4(),
                title: "Video 1".to_string(),
                description: "Description".to_string(),
                url: "https://example.com/1".to_string(),
                duration_seconds: 300,
                quality_score: 0.95,
                metadata: HashMap::new(),
            },
        ])
    }
}

// Transformers between versions

pub fn create_v1_to_v2_transformer() -> TransformFn {
    Arc::new(|data: Value| {
        let mut transformed = data.clone();

        // Add default fields for V2
        if let Some(obj) = transformed.as_object_mut() {
            if !obj.contains_key("description") {
                obj.insert("description".to_string(), Value::String("".to_string()));
            }
            if !obj.contains_key("duration_seconds") {
                obj.insert("duration_seconds".to_string(), Value::Number(0.into()));
            }
        }

        Ok(transformed)
    })
}

pub fn create_v2_to_v3_transformer() -> TransformFn {
    Arc::new(|data: Value| {
        let mut transformed = data.clone();

        // Add default fields for V3
        if let Some(obj) = transformed.as_object_mut() {
            if !obj.contains_key("quality_score") {
                obj.insert("quality_score".to_string(),
                          Value::Number(serde_json::Number::from_f64(0.5).unwrap()));
            }
            if !obj.contains_key("metadata") {
                obj.insert("metadata".to_string(), Value::Object(serde_json::Map::new()));
            }

            // Convert string ID to UUID format
            if let Some(Value::String(id)) = obj.get("id") {
                if let Ok(uuid) = Uuid::parse_str(id) {
                    obj.insert("id".to_string(), Value::String(uuid.to_string()));
                } else {
                    // Generate new UUID if ID is not valid
                    obj.insert("id".to_string(), Value::String(Uuid::new_v4().to_string()));
                }
            }
        }

        Ok(transformed)
    })
}

// Version Discovery Endpoint

#[derive(Debug, Serialize, Deserialize)]
pub struct VersionInfo {
    pub version: String,
    pub status: VersionStatus,
    pub deprecated: bool,
    pub sunset_date: Option<DateTime<Utc>>,
    pub endpoints: Vec<EndpointInfo>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum VersionStatus {
    Stable,
    Beta,
    Deprecated,
    Unsupported,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EndpointInfo {
    pub path: String,
    pub methods: Vec<String>,
    pub description: String,
}

pub async fn get_version_info(
    State(manager): State<Arc<VersionManager>>,
) -> Json<Vec<VersionInfo>> {
    let mut versions = Vec::new();

    for version in &manager.config.supported_versions {
        let status = if manager.config.deprecated_versions.contains(version) {
            VersionStatus::Deprecated
        } else if *version == manager.config.default_version {
            VersionStatus::Stable
        } else {
            VersionStatus::Stable
        };

        versions.push(VersionInfo {
            version: version.to_string(),
            status,
            deprecated: version.is_deprecated(),
            sunset_date: version.sunset_date(),
            endpoints: vec![
                EndpointInfo {
                    path: format!("/api/{}/videos", version.to_string()),
                    methods: vec!["GET".to_string(), "POST".to_string()],
                    description: "Video management".to_string(),
                },
                EndpointInfo {
                    path: format!("/api/{}/channels", version.to_string()),
                    methods: vec!["GET".to_string()],
                    description: "Channel management".to_string(),
                },
            ],
        });
    }

    Json(versions)
}

// Router creation with versioning

pub fn create_versioned_router() -> Router {
    let config = VersionConfig::default();
    let manager = Arc::new(VersionManager::new(config));

    // Register transformers
    tokio::spawn({
        let manager = manager.clone();
        async move {
            let _ = manager.register_transformer(
                ApiVersion::V1,
                ApiVersion::V2,
                create_v1_to_v2_transformer(),
            ).await;

            let _ = manager.register_transformer(
                ApiVersion::V2,
                ApiVersion::V3,
                create_v2_to_v3_transformer(),
            ).await;
        }
    });

    Router::new()
        // Version discovery
        .route("/api/versions", get(get_version_info))

        // V1 endpoints
        .route("/api/v1/videos", get(v1::get_videos))

        // V2 endpoints
        .route("/api/v2/videos", get(v2::get_videos))

        // V3 endpoints (latest)
        .route("/api/v3/videos", get(v3::get_videos))

        // Default (latest version)
        .route("/api/videos", get(v3::get_videos))

        // Apply version middleware
        .layer(middleware::from_fn_with_state(
            manager.clone(),
            version_middleware,
        ))
        .with_state(manager)
}