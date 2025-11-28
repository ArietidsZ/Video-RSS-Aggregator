use anyhow::Result;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post, put, delete},
    Router,
};
use clap::Parser;
use std::sync::Arc;
use uuid::Uuid;
use tracing::{info, Level};

mod auth;
mod rbac;
mod rate_limit;
mod validation;
mod audit;
mod oauth;
mod middleware;

use auth::AuthService;
use rbac::RBACService;
use rate_limit::RateLimitService;
use validation::ValidationService;
use audit::AuditService;
use oauth::OAuthService;

#[derive(Parser, Debug)]
#[command(name = "security-service")]
#[command(about = "Advanced security and authentication service for Video RSS Aggregator")]
struct Args {
    #[arg(short, long, default_value = "0.0.0.0:8020")]
    bind_address: String,

    #[arg(long, default_value = "redis://localhost:6379")]
    redis_url: String,

    #[arg(long, default_value = "postgresql://videorss:password@localhost/videorss")]
    database_url: String,

    #[arg(long)]
    jwt_secret: String,

    #[arg(long)]
    encryption_key: String,

    #[arg(long, default_value = "86400")]
    token_expiry_seconds: u64,

    #[arg(long, default_value = "604800")]
    refresh_token_expiry_seconds: u64,

    #[arg(long)]
    enable_oauth: bool,

    #[arg(long)]
    oauth_client_id: Option<String>,

    #[arg(long)]
    oauth_client_secret: Option<String>,
}

pub struct SecuritySystem {
    auth_service: Arc<AuthService>,
    rbac_service: Arc<RBACService>,
    rate_limit_service: Arc<RateLimitService>,
    validation_service: Arc<ValidationService>,
    audit_service: Arc<AuditService>,
    oauth_service: Option<Arc<OAuthService>>,
}

impl SecuritySystem {
    pub async fn new(
        redis_url: &str,
        database_url: &str,
        jwt_secret: &str,
        encryption_key: &str,
        token_expiry: u64,
        refresh_token_expiry: u64,
        oauth_config: Option<(&str, &str)>,
    ) -> Result<Self> {
        info!("Initializing security system...");

        let auth_service = Arc::new(
            AuthService::new(database_url, redis_url, jwt_secret, encryption_key, token_expiry, refresh_token_expiry).await?
        );

        let rbac_service = Arc::new(RBACService::new(database_url).await?);
        let rate_limit_service = Arc::new(RateLimitService::new(redis_url).await?);
        let validation_service = Arc::new(ValidationService::new());
        let audit_service = Arc::new(AuditService::new(database_url).await?);

        let oauth_service = if let Some((client_id, client_secret)) = oauth_config {
            Some(Arc::new(OAuthService::new(client_id, client_secret).await?))
        } else {
            None
        };

        Ok(Self {
            auth_service,
            rbac_service,
            rate_limit_service,
            validation_service,
            audit_service,
            oauth_service,
        })
    }

    pub async fn create_router(self: Arc<Self>) -> Router {
        use axum::middleware::{from_fn_with_state, from_fn};
        use crate::middleware::{
            rate_limit_middleware,
            audit_middleware,
            authentication_middleware,
            authorization_middleware,
            security_headers_middleware,
        };

        // Create separate state objects for middleware
        let rate_limit_state = Arc::clone(&self.rate_limit_service);
        let audit_state = Arc::clone(&self.audit_service);
        let auth_state = Arc::clone(&self.auth_service);
        let rbac_state = Arc::clone(&self.rbac_service);

        Router::new()
            // Authentication endpoints (public - no auth middleware)
            .route("/auth/register", post(register))
            .route("/auth/login", post(login))
            .route("/auth/refresh", post(refresh_token))
            .route("/auth/logout", post(logout))
            .route("/auth/verify", get(verify_token))
            .route("/auth/change-password", put(change_password))
            .route("/auth/reset-password", post(reset_password))
            .route("/auth/confirm-reset", post(confirm_password_reset))

            // OAuth endpoints (public)
            .route("/oauth/authorize/{provider}", get(oauth_authorize))
            .route("/oauth/callback/{provider}", get(oauth_callback))
            .route("/oauth/accounts/{user_id}", get(get_user_oauth_accounts))
            .route("/oauth/unlink/{user_id}/{provider}", delete(unlink_oauth_account))
            .route("/oauth/providers", get(get_available_providers))
            .route("/oauth/refresh/{user_id}/{provider}", post(refresh_oauth_token))

            // Health check (public)
            .route("/health", get(health_check))

            // Protected endpoints - apply authentication and authorization middleware
            .route("/users", get(list_users))
            .route("/users/{user_id}", get(get_user))
            .route("/users/{user_id}", put(update_user))
            .route("/users/{user_id}", delete(delete_user))
            .route("/users/{user_id}/roles", get(get_user_roles))
            .route("/users/{user_id}/roles", put(update_user_roles))
            .route("/users/{user_id}/roles/json", get(get_user_roles_json_handler))
            .route("/roles", get(list_roles))
            .route("/roles", post(create_role))
            .route("/roles/{role_id}", get(get_role))
            .route("/roles/{role_id}", put(update_role))
            .route("/roles/{role_id}", delete(delete_role))
            .route("/permissions", get(list_permissions))
            .route("/permissions/check", post(check_permissions_bulk))
            .route("/permissions/by-resource/{user_id}/{resource}", get(get_permissions_by_resource))
            .route("/audit/logs", get(get_audit_logs))
            .route("/audit/stats", get(get_audit_stats))
            .route("/audit/log-event", post(log_event))
            .route("/audit/search", get(search_audit_events))
            .route("/audit/export", get(export_audit_logs_handler))
            .route("/audit/cleanup", post(cleanup_old_audit_events))
            .route("/audit/dashboard", get(get_security_dashboard))
            .route("/security/rate-limits", get(get_rate_limit_status))
            .route("/security/blocked-ips", get(get_blocked_ips))
            .route("/security/unblock-ip", post(unblock_ip))
            .route("/security/rate-limit-rules", post(add_rate_limit_rule))
            .route("/security/rate-limit-rules/{rule_id}", delete(remove_rate_limit_rule))
            .route("/security/violations", get(get_rate_limit_violations))
            .route("/security/cleanup-expired", post(cleanup_expired_blocks))
            .route("/validate", post(validate_input))
            .route("/encryption/encrypt", post(encrypt_data))
            .route("/encryption/decrypt", post(decrypt_data))
            .layer(from_fn_with_state(auth_state.clone(), authentication_middleware))
            .layer(from_fn_with_state(rbac_state.clone(), authorization_middleware))

            // Apply global middleware to all routes
            .layer(from_fn_with_state(audit_state, audit_middleware))
            .layer(from_fn_with_state(rate_limit_state, rate_limit_middleware))
            .layer(from_fn(security_headers_middleware))

            .with_state(self)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .with_target(false)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .init();

    let args = Args::parse();

    info!("Starting Security Service");
    info!("Bind address: {}", args.bind_address);

    let oauth_config = if args.enable_oauth && args.oauth_client_id.is_some() && args.oauth_client_secret.is_some() {
        Some((
            args.oauth_client_id.as_ref().unwrap().as_str(),
            args.oauth_client_secret.as_ref().unwrap().as_str(),
        ))
    } else {
        None
    };

    // Initialize security system
    let security_system = Arc::new(
        SecuritySystem::new(
            &args.redis_url,
            &args.database_url,
            &args.jwt_secret,
            &args.encryption_key,
            args.token_expiry_seconds,
            args.refresh_token_expiry_seconds,
            oauth_config,
        )
        .await?,
    );

    // Create router
    let app = security_system.create_router().await;

    let listener = tokio::net::TcpListener::bind(&args.bind_address).await?;

    info!("Security service started on {}", args.bind_address);

    axum::serve(listener, app).await?;

    Ok(())
}

// Handler functions (stubs - will be implemented in respective modules)
async fn register(
    State(security): State<Arc<SecuritySystem>>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    security.auth_service.register_user(payload).await
        .map(Json)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

async fn login(
    State(security): State<Arc<SecuritySystem>>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    security.auth_service.login_user(payload).await
        .map(Json)
        .map_err(|_| StatusCode::UNAUTHORIZED)
}

async fn refresh_token(
    State(security): State<Arc<SecuritySystem>>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    security.auth_service.refresh_token(payload).await
        .map(Json)
        .map_err(|_| StatusCode::UNAUTHORIZED)
}

async fn logout(
    State(security): State<Arc<SecuritySystem>>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    security.auth_service.logout_user(payload).await
        .map(Json)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

async fn verify_token(
    State(security): State<Arc<SecuritySystem>>,
    headers: axum::http::HeaderMap,
) -> Result<Json<serde_json::Value>, StatusCode> {
    security.auth_service.verify_token(headers).await
        .map(Json)
        .map_err(|_| StatusCode::UNAUTHORIZED)
}

async fn change_password(
    State(security): State<Arc<SecuritySystem>>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    security.auth_service.change_password(payload).await
        .map(Json)
        .map_err(|_| StatusCode::BAD_REQUEST)
}

async fn reset_password(
    State(security): State<Arc<SecuritySystem>>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    security.auth_service.reset_password(payload).await
        .map(Json)
        .map_err(|_| StatusCode::BAD_REQUEST)
}

async fn confirm_password_reset(
    State(security): State<Arc<SecuritySystem>>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    security.auth_service.confirm_password_reset(payload).await
        .map(Json)
        .map_err(|_| StatusCode::BAD_REQUEST)
}

async fn oauth_authorize(
    State(security): State<Arc<SecuritySystem>>,
    Path(provider): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    if let Some(oauth_service) = &security.oauth_service {
        oauth_service.authorize(&provider).await
            .map(Json)
            .map_err(|_| StatusCode::BAD_REQUEST)
    } else {
        Err(StatusCode::NOT_IMPLEMENTED)
    }
}

async fn oauth_callback(
    State(security): State<Arc<SecuritySystem>>,
    Path(provider): Path<String>,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    if let Some(oauth_service) = &security.oauth_service {
        oauth_service.handle_callback(&provider, params).await
            .map(Json)
            .map_err(|_| StatusCode::BAD_REQUEST)
    } else {
        Err(StatusCode::NOT_IMPLEMENTED)
    }
}

// User management handlers
async fn list_users(
    State(security): State<Arc<SecuritySystem>>,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    security.rbac_service.list_users(params).await
        .map(Json)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

async fn get_user(
    State(security): State<Arc<SecuritySystem>>,
    Path(user_id): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    security.rbac_service.get_user(&user_id).await
        .map(Json)
        .map_err(|_| StatusCode::NOT_FOUND)
}

async fn update_user(
    State(security): State<Arc<SecuritySystem>>,
    Path(user_id): Path<String>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    security.rbac_service.update_user(&user_id, payload).await
        .map(Json)
        .map_err(|_| StatusCode::BAD_REQUEST)
}

async fn delete_user(
    State(security): State<Arc<SecuritySystem>>,
    Path(user_id): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    security.rbac_service.delete_user(&user_id).await
        .map(Json)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

async fn get_user_roles(
    State(security): State<Arc<SecuritySystem>>,
    Path(user_id): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let user_uuid = Uuid::parse_str(&user_id).map_err(|_| StatusCode::BAD_REQUEST)?;
    let roles = security.rbac_service.get_user_roles(&user_uuid).await
        .map_err(|_| StatusCode::NOT_FOUND)?;
    Ok(Json(serde_json::to_value(roles).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?))
}

async fn update_user_roles(
    State(security): State<Arc<SecuritySystem>>,
    Path(user_id): Path<String>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    security.rbac_service.update_user_roles(&user_id, payload).await
        .map(Json)
        .map_err(|_| StatusCode::BAD_REQUEST)
}

async fn get_user_roles_json_handler(
    State(security): State<Arc<SecuritySystem>>,
    Path(user_id): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    security.rbac_service.get_user_roles_json(&user_id).await
        .map(Json)
        .map_err(|_| StatusCode::NOT_FOUND)
}

// Role management handlers
async fn list_roles(
    State(security): State<Arc<SecuritySystem>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    security.rbac_service.list_roles().await
        .map(Json)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

async fn create_role(
    State(security): State<Arc<SecuritySystem>>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    security.rbac_service.create_role(payload).await
        .map(Json)
        .map_err(|_| StatusCode::BAD_REQUEST)
}

async fn get_role(
    State(security): State<Arc<SecuritySystem>>,
    Path(role_id): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    security.rbac_service.get_role(&role_id).await
        .map(Json)
        .map_err(|_| StatusCode::NOT_FOUND)
}

async fn update_role(
    State(security): State<Arc<SecuritySystem>>,
    Path(role_id): Path<String>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    security.rbac_service.update_role(&role_id, payload).await
        .map(Json)
        .map_err(|_| StatusCode::BAD_REQUEST)
}

async fn delete_role(
    State(security): State<Arc<SecuritySystem>>,
    Path(role_id): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    security.rbac_service.delete_role(&role_id).await
        .map(Json)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

async fn list_permissions(
    State(security): State<Arc<SecuritySystem>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    security.rbac_service.list_permissions().await
        .map(Json)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

// Audit and security monitoring handlers
async fn get_audit_logs(
    State(security): State<Arc<SecuritySystem>>,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    security.audit_service.get_audit_logs(params).await
        .map(Json)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

async fn get_audit_stats(
    State(security): State<Arc<SecuritySystem>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    security.audit_service.get_audit_stats().await
        .map(Json)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

async fn get_rate_limit_status(
    State(security): State<Arc<SecuritySystem>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    security.rate_limit_service.get_status().await
        .map(Json)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

async fn get_blocked_ips(
    State(security): State<Arc<SecuritySystem>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    security.rate_limit_service.get_blocked_ips().await
        .map(Json)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

async fn unblock_ip(
    State(security): State<Arc<SecuritySystem>>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    security.rate_limit_service.unblock_ip(payload).await
        .map(Json)
        .map_err(|_| StatusCode::BAD_REQUEST)
}

async fn validate_input(
    State(security): State<Arc<SecuritySystem>>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let endpoint = payload.get("endpoint")
        .and_then(|v| v.as_str())
        .unwrap_or("/");
    let data = payload.get("data").cloned().unwrap_or(serde_json::json!({}));

    security.validation_service
        .validate_request_data(data, endpoint)
        .await
        .map(|result| Json(serde_json::to_value(result).unwrap()))
        .map_err(|_| StatusCode::BAD_REQUEST)
}

// OAuth handlers
async fn get_user_oauth_accounts(
    State(security): State<Arc<SecuritySystem>>,
    Path(user_id): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    if let Some(oauth_service) = &security.oauth_service {
        let user_uuid = Uuid::parse_str(&user_id).map_err(|_| StatusCode::BAD_REQUEST)?;
        oauth_service.get_user_oauth_accounts(&user_uuid).await
            .map(|accounts| Json(serde_json::to_value(accounts).unwrap()))
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
    } else {
        Err(StatusCode::NOT_IMPLEMENTED)
    }
}

async fn unlink_oauth_account(
    State(security): State<Arc<SecuritySystem>>,
    Path((user_id, provider)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    if let Some(oauth_service) = &security.oauth_service {
        let user_uuid = Uuid::parse_str(&user_id).map_err(|_| StatusCode::BAD_REQUEST)?;
        oauth_service.unlink_oauth_account(&user_uuid, &provider).await
            .map(|_| Json(serde_json::json!({"success": true})))
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
    } else {
        Err(StatusCode::NOT_IMPLEMENTED)
    }
}

async fn get_available_providers(
    State(security): State<Arc<SecuritySystem>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    if let Some(oauth_service) = &security.oauth_service {
        let providers = oauth_service.get_available_providers();
        Ok(Json(serde_json::to_value(providers).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?))
    } else {
        Err(StatusCode::NOT_IMPLEMENTED)
    }
}

async fn refresh_oauth_token(
    State(security): State<Arc<SecuritySystem>>,
    Path((user_id, provider)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    if let Some(oauth_service) = &security.oauth_service {
        let user_uuid = Uuid::parse_str(&user_id).map_err(|_| StatusCode::BAD_REQUEST)?;
        oauth_service.refresh_oauth_token(&user_uuid, &provider).await
            .map(|_| Json(serde_json::json!({"success": true})))
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
    } else {
        Err(StatusCode::NOT_IMPLEMENTED)
    }
}

// Audit handlers
async fn log_event(
    State(security): State<Arc<SecuritySystem>>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    use crate::audit::AuditEvent;
    let event: AuditEvent = serde_json::from_value(payload).map_err(|_| StatusCode::BAD_REQUEST)?;
    security.audit_service.log_event(event).await
        .map(|_| Json(serde_json::json!({"success": true})))
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

async fn search_audit_events(
    State(security): State<Arc<SecuritySystem>>,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let query = params.get("query").map(|s| s.as_str()).unwrap_or("");
    let limit = params.get("limit")
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(100);

    security.audit_service.search_events(query, limit).await
        .map(|events| Json(serde_json::to_value(events).unwrap()))
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

async fn export_audit_logs_handler(
    State(security): State<Arc<SecuritySystem>>,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    use crate::audit::AuditFilter;
    use base64::{Engine, engine::general_purpose};

    let format = params.get("format").map(|s| s.as_str()).unwrap_or("json");
    let start_time = params.get("start_time")
        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&chrono::Utc));
    let end_time = params.get("end_time")
        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&chrono::Utc));

    let limit = params.get("limit")
        .and_then(|s| s.parse::<u32>().ok());
    let offset = params.get("offset")
        .and_then(|s| s.parse::<u32>().ok());

    let filter = AuditFilter {
        event_types: None,
        user_id: None,
        ip_address: None,
        resource: None,
        action: None,
        result: None,
        severity: None,
        start_time,
        end_time,
        limit,
        offset,
    };

    security.audit_service.export_audit_logs(filter, format).await
        .map(|data| Json(serde_json::json!({"data": general_purpose::STANDARD.encode(data)})))
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

async fn cleanup_old_audit_events(
    State(security): State<Arc<SecuritySystem>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    security.audit_service.cleanup_old_events().await
        .map(|_| Json(serde_json::json!({"success": true})))
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

async fn get_security_dashboard(
    State(security): State<Arc<SecuritySystem>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    security.audit_service.get_security_dashboard_data().await
        .map(Json)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

// RBAC handlers
async fn check_permissions_bulk(
    State(security): State<Arc<SecuritySystem>>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    use crate::rbac::PermissionCheck;

    let user_id = payload.get("user_id")
        .and_then(|v| v.as_str())
        .and_then(|s| Uuid::parse_str(s).ok())
        .ok_or(StatusCode::BAD_REQUEST)?;

    let checks: Vec<PermissionCheck> = serde_json::from_value(
        payload.get("checks").cloned().unwrap_or(serde_json::json!([]))
    ).map_err(|_| StatusCode::BAD_REQUEST)?;

    security.rbac_service.bulk_check_permissions(&user_id, &checks).await
        .map(|results| Json(serde_json::to_value(results).unwrap()))
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

async fn get_permissions_by_resource(
    State(security): State<Arc<SecuritySystem>>,
    Path((user_id, resource)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let user_uuid = Uuid::parse_str(&user_id).map_err(|_| StatusCode::BAD_REQUEST)?;

    security.rbac_service.get_user_permissions_by_resource(&user_uuid, &resource).await
        .map(|perms| Json(serde_json::to_value(perms).unwrap()))
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

// Rate limiting handlers
async fn add_rate_limit_rule(
    State(security): State<Arc<SecuritySystem>>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    use crate::rate_limit::RateLimitRule;

    let rule: RateLimitRule = serde_json::from_value(payload).map_err(|_| StatusCode::BAD_REQUEST)?;

    security.rate_limit_service.add_rule(rule).await
        .map(|_| Json(serde_json::json!({"success": true})))
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

async fn remove_rate_limit_rule(
    State(security): State<Arc<SecuritySystem>>,
    Path(rule_id): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let rule_uuid = Uuid::parse_str(&rule_id).map_err(|_| StatusCode::BAD_REQUEST)?;

    security.rate_limit_service.remove_rule(&rule_uuid).await
        .map(|_| Json(serde_json::json!({"success": true})))
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

async fn get_rate_limit_violations(
    State(security): State<Arc<SecuritySystem>>,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let hours = params.get("hours")
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(24);

    security.rate_limit_service.get_violations(hours).await
        .map(|violations| Json(serde_json::to_value(violations).unwrap()))
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

async fn cleanup_expired_blocks(
    State(security): State<Arc<SecuritySystem>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    security.rate_limit_service.cleanup_expired().await
        .map(|_| Json(serde_json::json!({"success": true})))
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

// Encryption handlers
async fn encrypt_data(
    State(security): State<Arc<SecuritySystem>>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let data = payload.get("data")
        .and_then(|v| v.as_str())
        .ok_or(StatusCode::BAD_REQUEST)?;

    security.auth_service.encrypt_sensitive_data(data)
        .map(|encrypted| Json(serde_json::json!({"encrypted": encrypted})))
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

async fn decrypt_data(
    State(security): State<Arc<SecuritySystem>>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let encrypted = payload.get("encrypted")
        .and_then(|v| v.as_str())
        .ok_or(StatusCode::BAD_REQUEST)?;

    security.auth_service.decrypt_sensitive_data(encrypted)
        .map(|decrypted| Json(serde_json::json!({"data": decrypted})))
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

async fn health_check() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "security-service",
        "timestamp": chrono::Utc::now()
    }))
}