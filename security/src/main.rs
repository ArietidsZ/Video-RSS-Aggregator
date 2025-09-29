use anyhow::Result;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    middleware,
    response::Json,
    routing::{get, post, put, delete},
    Router,
};
use clap::Parser;
use std::sync::Arc;
use tower::ServiceBuilder;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::{info, Level};

mod auth;
mod rbac;
mod rate_limit;
mod validation;
mod audit;
mod oauth;
mod middleware as auth_middleware;

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

    #[arg(long, env = "JWT_SECRET")]
    jwt_secret: String,

    #[arg(long, env = "ENCRYPTION_KEY")]
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
        Router::new()
            // Authentication endpoints
            .route("/auth/register", post(register))
            .route("/auth/login", post(login))
            .route("/auth/refresh", post(refresh_token))
            .route("/auth/logout", post(logout))
            .route("/auth/verify", get(verify_token))
            .route("/auth/change-password", put(change_password))
            .route("/auth/reset-password", post(reset_password))
            .route("/auth/confirm-reset", post(confirm_password_reset))

            // OAuth endpoints
            .route("/oauth/authorize/:provider", get(oauth_authorize))
            .route("/oauth/callback/:provider", get(oauth_callback))

            // User management
            .route("/users", get(list_users))
            .route("/users/:user_id", get(get_user))
            .route("/users/:user_id", put(update_user))
            .route("/users/:user_id", delete(delete_user))
            .route("/users/:user_id/roles", get(get_user_roles))
            .route("/users/:user_id/roles", put(update_user_roles))

            // Role and permission management
            .route("/roles", get(list_roles))
            .route("/roles", post(create_role))
            .route("/roles/:role_id", get(get_role))
            .route("/roles/:role_id", put(update_role))
            .route("/roles/:role_id", delete(delete_role))
            .route("/permissions", get(list_permissions))

            // Audit and security monitoring
            .route("/audit/logs", get(get_audit_logs))
            .route("/audit/stats", get(get_audit_stats))
            .route("/security/rate-limits", get(get_rate_limit_status))
            .route("/security/blocked-ips", get(get_blocked_ips))
            .route("/security/unblock-ip", post(unblock_ip))

            // Health check
            .route("/health", get(health_check))

            .layer(
                ServiceBuilder::new()
                    .layer(TraceLayer::new_for_http())
                    .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any))
                    .layer(middleware::from_fn_with_state(
                        Arc::clone(&self.rate_limit_service),
                        auth_middleware::rate_limit_middleware
                    ))
                    .layer(middleware::from_fn_with_state(
                        Arc::clone(&self.audit_service),
                        auth_middleware::audit_middleware
                    ))
            )
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
    security.rbac_service.get_user_roles(&user_id).await
        .map(Json)
        .map_err(|_| StatusCode::NOT_FOUND)
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

async fn health_check() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "security-service",
        "timestamp": chrono::Utc::now()
    }))
}