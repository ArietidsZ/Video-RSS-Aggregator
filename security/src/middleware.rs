use axum::{
    extract::{ConnectInfo, Request, State},
    http::{HeaderMap, StatusCode},
    middleware::Next,
    response::Response,
};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;
use tracing::{info, warn};
use uuid::Uuid;

use crate::audit::{AuditEvent, AuditEventType, AuditResult, AuditService, AuditSeverity};
use crate::rate_limit::{RateLimitRequest, RateLimitService};

pub async fn rate_limit_middleware(
    State(rate_limit_service): State<Arc<RateLimitService>>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    mut request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let start_time = Instant::now();

    // Extract request information
    let method = request.method().to_string();
    let path = request.uri().path().to_string();
    let ip_address = addr.ip();
    let user_agent = headers
        .get("user-agent")
        .and_then(|h| h.to_str().ok())
        .map(|s| s.to_string());

    // Extract user ID from JWT if present
    let user_id = extract_user_id_from_headers(&headers);

    // Extract API key if present
    let api_key = headers
        .get("x-api-key")
        .and_then(|h| h.to_str().ok())
        .map(|s| s.to_string());

    // Determine resource based on path
    let resource = determine_resource_from_path(&path);

    let rate_limit_request = RateLimitRequest {
        method: method.clone(),
        path: path.clone(),
        resource,
        ip_address,
        user_id,
        api_key,
        user_agent: user_agent.clone(),
        headers: Some(headers_to_json(&headers)),
    };

    // Check rate limits
    let rate_limit_status = rate_limit_service
        .check_rate_limit(&rate_limit_request)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    if rate_limit_status.blocked {
        warn!(
            "Rate limit exceeded for {} {} from {}",
            method, path, ip_address
        );

        // Add rate limit headers
        let mut response = Response::builder()
            .status(StatusCode::TOO_MANY_REQUESTS)
            .header("X-RateLimit-Limit", rate_limit_status.limit.to_string())
            .header("X-RateLimit-Remaining", "0")
            .header(
                "X-RateLimit-Reset",
                rate_limit_status.reset_time.timestamp().to_string(),
            );

        if let Some(retry_after) = rate_limit_status.retry_after {
            response = response.header("Retry-After", retry_after.to_string());
        }

        return response
            .body("Rate limit exceeded".into())
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR);
    }

    // Add rate limit headers to successful requests
    request.headers_mut().insert(
        "X-RateLimit-Limit",
        rate_limit_status.limit.to_string().parse().unwrap(),
    );
    request.headers_mut().insert(
        "X-RateLimit-Remaining",
        rate_limit_status.remaining.to_string().parse().unwrap(),
    );

    // Continue to next middleware/handler
    let response = next.run(request).await;

    // Log performance metrics
    let duration = start_time.elapsed();
    if duration.as_millis() > 1000 {
        warn!(
            "Slow request: {} {} took {}ms",
            method,
            path,
            duration.as_millis()
        );
    }

    Ok(response)
}

pub async fn audit_middleware(
    State(audit_service): State<Arc<AuditService>>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    mut request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let start_time = Instant::now();
    let request_id = Uuid::new_v4().to_string();

    // Extract request information
    let method = request.method().to_string();
    let path = request.uri().path().to_string();
    let ip_address = addr.ip();
    let user_agent = headers
        .get("user-agent")
        .and_then(|h| h.to_str().ok())
        .map(|s| s.to_string());

    let user_id = extract_user_id_from_headers(&headers);

    // Add request ID to headers for tracing
    request
        .headers_mut()
        .insert("X-Request-ID", request_id.parse().unwrap());

    // Determine if this is an API access event
    let is_api_request = path.starts_with("/api/");
    let resource = determine_resource_from_path(&path);

    // Continue to next middleware/handler
    let response = next.run(request).await;

    let duration = start_time.elapsed();
    let status_code = response.status().as_u16();

    // Determine audit event type and action
    let (event_type, action) = classify_request(&method, &path, status_code);

    // Determine result
    let result = if status_code >= 200 && status_code < 300 {
        AuditResult::Success
    } else if status_code >= 400 && status_code < 500 {
        AuditResult::Failure
    } else if status_code >= 500 {
        AuditResult::Failure
    } else {
        AuditResult::Success
    };

    // Determine severity
    let severity = determine_audit_severity(&event_type, status_code, &path);

    // Create audit event
    let audit_event = AuditEvent {
        id: Uuid::new_v4(),
        event_type,
        user_id,
        session_id: extract_session_id_from_headers(&headers),
        ip_address: Some(ip_address),
        user_agent,
        resource,
        action,
        details: serde_json::json!({
            "path": path,
            "query_string": request.uri().query(),
            "user_agent": headers.get("user-agent").and_then(|h| h.to_str().ok()),
            "referer": headers.get("referer").and_then(|h| h.to_str().ok()),
        }),
        result,
        severity,
        timestamp: chrono::Utc::now(),
        request_id: Some(request_id),
        endpoint: Some(path),
        method: Some(method),
        status_code: Some(status_code),
        response_time_ms: Some(duration.as_millis() as u64),
        error_message: if status_code >= 400 {
            Some(format!("HTTP {}", status_code))
        } else {
            None
        },
    };

    // Log audit event (fire and forget to avoid blocking response)
    let audit_service_clone = Arc::clone(&audit_service);
    tokio::spawn(async move {
        if let Err(e) = audit_service_clone.log_event(audit_event).await {
            tracing::error!("Failed to log audit event: {}", e);
        }
    });

    // Log request for immediate visibility
    info!(
        request_id = request_id,
        method = method,
        path = path,
        status = status_code,
        duration_ms = duration.as_millis(),
        user_id = ?user_id,
        ip = %ip_address,
        "HTTP request completed"
    );

    Ok(response)
}

pub async fn authentication_middleware(
    headers: HeaderMap,
    mut request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Check if the endpoint requires authentication
    let path = request.uri().path();

    if is_public_endpoint(path) {
        return Ok(next.run(request).await);
    }

    // Extract and validate JWT token
    let auth_header = headers
        .get("Authorization")
        .and_then(|header| header.to_str().ok())
        .and_then(|header| header.strip_prefix("Bearer "));

    let token = match auth_header {
        Some(token) => token,
        None => {
            return Err(StatusCode::UNAUTHORIZED);
        }
    };

    // Validate token (simplified - in real implementation, use auth service)
    if !is_valid_token(token) {
        return Err(StatusCode::UNAUTHORIZED);
    }

    // Add user information to request headers for downstream handlers
    if let Some(user_id) = extract_user_from_token(token) {
        request
            .headers_mut()
            .insert("X-User-ID", user_id.parse().unwrap());
    }

    Ok(next.run(request).await)
}

pub async fn authorization_middleware(
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let path = request.uri().path();
    let method = request.method().as_str();

    // Skip authorization for public endpoints
    if is_public_endpoint(path) {
        return Ok(next.run(request).await);
    }

    // Extract user ID from headers (set by authentication middleware)
    let user_id = headers
        .get("X-User-ID")
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.parse::<Uuid>().ok());

    if user_id.is_none() {
        return Err(StatusCode::UNAUTHORIZED);
    }

    // Check permissions (simplified - in real implementation, use RBAC service)
    let required_permission = determine_required_permission(method, path);

    if !user_has_permission(user_id.unwrap(), &required_permission).await {
        return Err(StatusCode::FORBIDDEN);
    }

    Ok(next.run(request).await)
}

pub async fn security_headers_middleware(
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let mut response = next.run(request).await;

    let headers = response.headers_mut();

    // Add security headers
    headers.insert("X-Content-Type-Options", "nosniff".parse().unwrap());
    headers.insert("X-Frame-Options", "DENY".parse().unwrap());
    headers.insert("X-XSS-Protection", "1; mode=block".parse().unwrap());
    headers.insert("Referrer-Policy", "strict-origin-when-cross-origin".parse().unwrap());
    headers.insert(
        "Strict-Transport-Security",
        "max-age=31536000; includeSubDomains".parse().unwrap(),
    );
    headers.insert(
        "Content-Security-Policy",
        "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
            .parse()
            .unwrap(),
    );
    headers.insert("Permissions-Policy", "geolocation=(), microphone=(), camera=()".parse().unwrap());

    Ok(response)
}

// Helper functions

fn extract_user_id_from_headers(headers: &HeaderMap) -> Option<Uuid> {
    headers
        .get("Authorization")
        .and_then(|h| h.to_str().ok())
        .and_then(|h| h.strip_prefix("Bearer "))
        .and_then(extract_user_from_token)
        .and_then(|s| s.parse().ok())
}

fn extract_session_id_from_headers(headers: &HeaderMap) -> Option<String> {
    headers
        .get("X-Session-ID")
        .and_then(|h| h.to_str().ok())
        .map(|s| s.to_string())
}

fn headers_to_json(headers: &HeaderMap) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for (name, value) in headers {
        if let Ok(value_str) = value.to_str() {
            map.insert(name.to_string(), serde_json::Value::String(value_str.to_string()));
        }
    }
    serde_json::Value::Object(map)
}

fn determine_resource_from_path(path: &str) -> String {
    if path.starts_with("/auth") {
        "authentication".to_string()
    } else if path.starts_with("/api/users") {
        "users".to_string()
    } else if path.starts_with("/api/feeds") {
        "feeds".to_string()
    } else if path.starts_with("/api/videos") {
        "videos".to_string()
    } else if path.starts_with("/api/roles") {
        "roles".to_string()
    } else if path.starts_with("/oauth") {
        "oauth".to_string()
    } else if path.starts_with("/api") {
        "api".to_string()
    } else {
        "web".to_string()
    }
}

fn classify_request(method: &str, path: &str, status_code: u16) -> (AuditEventType, String) {
    let event_type = if path.starts_with("/auth") {
        AuditEventType::Authentication
    } else if path.starts_with("/oauth") {
        AuditEventType::Authentication
    } else if path.contains("/users") || path.contains("/roles") {
        AuditEventType::UserManagement
    } else if method == "GET" {
        AuditEventType::DataAccess
    } else if method == "POST" || method == "PUT" || method == "PATCH" {
        AuditEventType::DataModification
    } else if method == "DELETE" {
        AuditEventType::DataModification
    } else {
        AuditEventType::APIAccess
    };

    let action = if path.contains("/login") {
        "login".to_string()
    } else if path.contains("/logout") {
        "logout".to_string()
    } else if path.contains("/register") {
        "register".to_string()
    } else {
        format!("{}_request", method.to_lowercase())
    };

    (event_type, action)
}

fn determine_audit_severity(event_type: &AuditEventType, status_code: u16, path: &str) -> AuditSeverity {
    match event_type {
        AuditEventType::Authentication => {
            if status_code >= 400 {
                AuditSeverity::Medium
            } else {
                AuditSeverity::Low
            }
        },
        AuditEventType::UserManagement | AuditEventType::RoleManagement => {
            if status_code >= 400 {
                AuditSeverity::High
            } else {
                AuditSeverity::Medium
            }
        },
        AuditEventType::SecurityEvent | AuditEventType::SuspiciousActivity => {
            AuditSeverity::High
        },
        AuditEventType::DataModification => {
            if path.contains("/admin") || path.contains("/system") {
                AuditSeverity::Medium
            } else {
                AuditSeverity::Low
            }
        },
        _ => {
            if status_code >= 500 {
                AuditSeverity::Medium
            } else if status_code >= 400 {
                AuditSeverity::Low
            } else {
                AuditSeverity::Low
            }
        }
    }
}

fn is_public_endpoint(path: &str) -> bool {
    let public_paths = [
        "/health",
        "/metrics",
        "/auth/login",
        "/auth/register",
        "/auth/reset-password",
        "/oauth/authorize",
        "/oauth/callback",
        "/rss",
    ];

    public_paths.iter().any(|&public_path| path.starts_with(public_path))
}

fn is_valid_token(token: &str) -> bool {
    // Simplified token validation
    // In real implementation, this would use the auth service to validate JWT
    !token.is_empty() && token.len() > 20
}

fn extract_user_from_token(token: &str) -> Option<String> {
    // Simplified user extraction from token
    // In real implementation, this would decode JWT and extract user ID
    if is_valid_token(token) {
        Some("user-id-from-token".to_string())
    } else {
        None
    }
}

fn determine_required_permission(method: &str, path: &str) -> String {
    let resource = determine_resource_from_path(path);
    let action = match method {
        "GET" => "read",
        "POST" => "create",
        "PUT" | "PATCH" => "update",
        "DELETE" => "delete",
        _ => "access",
    };

    format!("{}:{}", resource, action)
}

async fn user_has_permission(user_id: Uuid, permission: &str) -> bool {
    // Simplified permission check
    // In real implementation, this would use the RBAC service
    tracing::debug!("Checking permission {} for user {}", permission, user_id);
    true // Allow all for now
}