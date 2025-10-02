use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::collections::HashMap;
use std::net::IpAddr;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub id: Uuid,
    pub event_type: AuditEventType,
    pub user_id: Option<Uuid>,
    pub session_id: Option<String>,
    pub ip_address: Option<IpAddr>,
    pub user_agent: Option<String>,
    pub resource: String,
    pub action: String,
    pub details: serde_json::Value,
    pub result: AuditResult,
    pub severity: AuditSeverity,
    pub timestamp: DateTime<Utc>,
    pub request_id: Option<String>,
    pub endpoint: Option<String>,
    pub method: Option<String>,
    pub status_code: Option<u16>,
    pub response_time_ms: Option<u64>,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditEventType {
    Authentication,
    Authorization,
    DataAccess,
    DataModification,
    SystemAccess,
    SecurityEvent,
    SecurityIncident,
    AdminAction,
    ApiAccess,
    APIAccess,
    FileAccess,
    ConfigurationChange,
    UserManagement,
    RoleManagement,
    PermissionChange,
    PasswordChange,
    AccountLockout,
    SuspiciousActivity,
    RateLimitViolation,
    ValidationFailure,
    SystemError,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditResult {
    Success,
    Failure,
    Partial,
    Blocked,
    Warning,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityMetrics {
    pub failed_logins_last_hour: u32,
    pub failed_logins_last_24h: u32,
    pub suspicious_activities_last_hour: u32,
    pub rate_limit_violations_last_hour: u32,
    pub blocked_ips_count: u32,
    pub active_sessions_count: u32,
    pub unique_users_last_24h: u32,
    pub api_requests_last_hour: u32,
    pub error_rate_last_hour: f64,
    pub top_error_endpoints: Vec<(String, u32)>,
    pub top_attacking_ips: Vec<(String, u32)>,
    pub geographic_distribution: HashMap<String, u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditFilter {
    pub event_types: Option<Vec<AuditEventType>>,
    pub user_id: Option<Uuid>,
    pub ip_address: Option<IpAddr>,
    pub resource: Option<String>,
    pub action: Option<String>,
    pub result: Option<AuditResult>,
    pub severity: Option<AuditSeverity>,
    pub start_time: Option<DateTime<Utc>>,
    pub end_time: Option<DateTime<Utc>>,
    pub limit: Option<u32>,
    pub offset: Option<u32>,
}

pub struct AuditService {
    database: PgPool,
    buffer: tokio::sync::RwLock<Vec<AuditEvent>>,
    buffer_size: usize,
}

impl AuditService {
    pub async fn new(database_url: &str) -> Result<Self> {
        let database = PgPool::connect(database_url)
            .await
            .context("Failed to connect to database for audit service")?;

        Self::init_database(&database).await?;

        let service = Self {
            database,
            buffer: tokio::sync::RwLock::new(Vec::new()),
            buffer_size: 1000,
        };

        // Start background task to flush buffer periodically
        let service_clone = std::sync::Arc::new(service);
        let service_for_task = std::sync::Arc::clone(&service_clone);

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));
            loop {
                interval.tick().await;
                if let Err(e) = service_for_task.flush_buffer().await {
                    error!("Failed to flush audit buffer: {}", e);
                }
            }
        });

        Ok(std::sync::Arc::try_unwrap(service_clone).unwrap_or_else(|_| panic!("Failed to unwrap Arc")))
    }

    async fn init_database(database: &PgPool) -> Result<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS audit_events (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                event_type VARCHAR(50) NOT NULL,
                user_id UUID REFERENCES users(id),
                session_id VARCHAR(255),
                ip_address INET,
                user_agent TEXT,
                resource VARCHAR(255) NOT NULL,
                action VARCHAR(100) NOT NULL,
                details JSONB NOT NULL DEFAULT '{}',
                result VARCHAR(20) NOT NULL,
                severity VARCHAR(20) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                request_id VARCHAR(255),
                endpoint VARCHAR(500),
                method VARCHAR(10),
                status_code INTEGER,
                response_time_ms BIGINT,
                error_message TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp ON audit_events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_audit_events_user_id ON audit_events(user_id);
            CREATE INDEX IF NOT EXISTS idx_audit_events_ip_address ON audit_events(ip_address);
            CREATE INDEX IF NOT EXISTS idx_audit_events_event_type ON audit_events(event_type);
            CREATE INDEX IF NOT EXISTS idx_audit_events_resource_action ON audit_events(resource, action);
            CREATE INDEX IF NOT EXISTS idx_audit_events_result ON audit_events(result);
            CREATE INDEX IF NOT EXISTS idx_audit_events_severity ON audit_events(severity);

            -- Partition table by month for better performance (PostgreSQL 10+)
            -- This would be implemented in a real production system

            CREATE TABLE IF NOT EXISTS security_incidents (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                incident_type VARCHAR(50) NOT NULL,
                severity VARCHAR(20) NOT NULL,
                title VARCHAR(255) NOT NULL,
                description TEXT,
                source_ip INET,
                affected_user_id UUID REFERENCES users(id),
                detection_method VARCHAR(100),
                status VARCHAR(20) NOT NULL DEFAULT 'open',
                assigned_to UUID REFERENCES users(id),
                resolved_at TIMESTAMPTZ,
                resolution_notes TEXT,
                related_events JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_security_incidents_severity ON security_incidents(severity);
            CREATE INDEX IF NOT EXISTS idx_security_incidents_status ON security_incidents(status);
            CREATE INDEX IF NOT EXISTS idx_security_incidents_created_at ON security_incidents(created_at);

            CREATE TABLE IF NOT EXISTS audit_retention_policies (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                event_type VARCHAR(50) NOT NULL,
                retention_days INTEGER NOT NULL,
                compression_enabled BOOLEAN NOT NULL DEFAULT false,
                archive_enabled BOOLEAN NOT NULL DEFAULT false,
                archive_location TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(event_type)
            );

            -- Insert default retention policies
            INSERT INTO audit_retention_policies (event_type, retention_days, compression_enabled, archive_enabled)
            VALUES
                ('Authentication', 365, true, true),
                ('Authorization', 180, true, false),
                ('DataAccess', 90, false, false),
                ('SecurityEvent', 1095, true, true), -- 3 years
                ('AdminAction', 730, true, true), -- 2 years
                ('SuspiciousActivity', 1095, true, true)
            ON CONFLICT (event_type) DO NOTHING;
            "#,
        )
        .execute(database)
        .await
        .context("Failed to initialize audit database tables")?;

        Ok(())
    }

    pub async fn log_event(&self, event: AuditEvent) -> Result<()> {
        // Add to buffer
        {
            let mut buffer = self.buffer.write().await;
            buffer.push(event.clone());

            // Flush buffer if it's full
            if buffer.len() >= self.buffer_size {
                drop(buffer);
                self.flush_buffer().await?;
            }
        }

        // Log to tracing for immediate visibility
        match event.severity {
            AuditSeverity::Critical => error!("AUDIT CRITICAL: {:?}", event),
            AuditSeverity::High => warn!("AUDIT HIGH: {:?}", event),
            AuditSeverity::Medium => info!("AUDIT MEDIUM: {:?}", event),
            AuditSeverity::Low => debug!("AUDIT LOW: {:?}", event),
        }

        Ok(())
    }

    async fn flush_buffer(&self) -> Result<()> {
        let events = {
            let mut buffer = self.buffer.write().await;
            if buffer.is_empty() {
                return Ok(());
            }
            let events = buffer.clone();
            buffer.clear();
            events
        };

        self.store_events_batch(&events).await?;
        debug!("Flushed {} audit events to database", events.len());

        Ok(())
    }

    async fn store_events_batch(&self, events: &[AuditEvent]) -> Result<()> {
        if events.is_empty() {
            return Ok(());
        }

        let mut tx = self.database.begin().await?;

        for event in events {
            sqlx::query(
                "INSERT INTO audit_events (
                     id, event_type, user_id, session_id, ip_address, user_agent,
                     resource, action, details, result, severity, timestamp,
                     request_id, endpoint, method, status_code, response_time_ms, error_message
                 )
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)"
            )
            .bind(&event.id)
            .bind(format!("{:?}", event.event_type))
            .bind(&event.user_id)
            .bind(&event.session_id)
            .bind(event.ip_address.map(|ip| ip.to_string()))
            .bind(&event.user_agent)
            .bind(&event.resource)
            .bind(&event.action)
            .bind(&event.details)
            .bind(format!("{:?}", event.result))
            .bind(format!("{:?}", event.severity))
            .bind(event.timestamp)
            .bind(&event.request_id)
            .bind(&event.endpoint)
            .bind(&event.method)
            .bind(event.status_code.map(|sc| sc as i32))
            .bind(event.response_time_ms.map(|rt| rt as i64))
            .bind(&event.error_message)
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;

        Ok(())
    }

    pub async fn log_authentication_event(
        &self,
        user_id: Option<Uuid>,
        action: &str,
        result: AuditResult,
        ip_address: Option<IpAddr>,
        user_agent: Option<String>,
        details: serde_json::Value,
    ) -> Result<()> {
        let event = AuditEvent {
            id: Uuid::new_v4(),
            event_type: AuditEventType::Authentication,
            user_id,
            session_id: None,
            ip_address,
            user_agent,
            resource: "authentication".to_string(),
            action: action.to_string(),
            details,
            result: result.clone(),
            severity: match result {
                AuditResult::Failure => AuditSeverity::Medium,
                AuditResult::Blocked => AuditSeverity::High,
                _ => AuditSeverity::Low,
            },
            timestamp: Utc::now(),
            request_id: None,
            endpoint: None,
            method: None,
            status_code: None,
            response_time_ms: None,
            error_message: None,
        };

        self.log_event(event).await
    }

    pub async fn log_authorization_event(
        &self,
        user_id: Uuid,
        resource: &str,
        action: &str,
        result: AuditResult,
        details: serde_json::Value,
    ) -> Result<()> {
        let event = AuditEvent {
            id: Uuid::new_v4(),
            event_type: AuditEventType::Authorization,
            user_id: Some(user_id),
            session_id: None,
            ip_address: None,
            user_agent: None,
            resource: resource.to_string(),
            action: action.to_string(),
            details,
            result: result.clone(),
            severity: match result {
                AuditResult::Failure | AuditResult::Blocked => AuditSeverity::Medium,
                _ => AuditSeverity::Low,
            },
            timestamp: Utc::now(),
            request_id: None,
            endpoint: None,
            method: None,
            status_code: None,
            response_time_ms: None,
            error_message: None,
        };

        self.log_event(event).await
    }

    pub async fn log_data_access(
        &self,
        user_id: Option<Uuid>,
        resource: &str,
        action: &str,
        details: serde_json::Value,
    ) -> Result<()> {
        let event = AuditEvent {
            id: Uuid::new_v4(),
            event_type: AuditEventType::DataAccess,
            user_id,
            session_id: None,
            ip_address: None,
            user_agent: None,
            resource: resource.to_string(),
            action: action.to_string(),
            details,
            result: AuditResult::Success,
            severity: AuditSeverity::Low,
            timestamp: Utc::now(),
            request_id: None,
            endpoint: None,
            method: None,
            status_code: None,
            response_time_ms: None,
            error_message: None,
        };

        self.log_event(event).await
    }

    pub async fn log_security_event(
        &self,
        event_type: AuditEventType,
        resource: &str,
        action: &str,
        severity: AuditSeverity,
        ip_address: Option<IpAddr>,
        user_id: Option<Uuid>,
        details: serde_json::Value,
    ) -> Result<()> {
        let event = AuditEvent {
            id: Uuid::new_v4(),
            event_type,
            user_id,
            session_id: None,
            ip_address,
            user_agent: None,
            resource: resource.to_string(),
            action: action.to_string(),
            details,
            result: AuditResult::Warning,
            severity: severity.clone(),
            timestamp: Utc::now(),
            request_id: None,
            endpoint: None,
            method: None,
            status_code: None,
            response_time_ms: None,
            error_message: None,
        };

        // Create security incident for high/critical events before logging
        if matches!(severity, AuditSeverity::High | AuditSeverity::Critical) {
            self.create_security_incident(&event).await?;
        }

        self.log_event(event).await?;

        Ok(())
    }

    async fn create_security_incident(&self, event: &AuditEvent) -> Result<()> {
        let title = format!("{:?}: {} on {}", event.event_type, event.action, event.resource);
        let description = format!("Security event detected: {:?}", event.details);

        sqlx::query(
            "INSERT INTO security_incidents (
                 incident_type, severity, title, description, source_ip,
                 affected_user_id, detection_method, related_events
             )
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8)"
        )
        .bind(format!("{:?}", event.event_type))
        .bind(format!("{:?}", event.severity))
        .bind(title)
        .bind(description)
        .bind(event.ip_address.map(|ip| ip.to_string()))
        .bind(event.user_id)
        .bind("audit_system")
        .bind(serde_json::json!([event.id]))
        .execute(&self.database)
        .await?;

        info!("Created security incident for event: {}", event.id);

        Ok(())
    }

    pub async fn get_audit_logs(&self, params: HashMap<String, String>) -> Result<serde_json::Value> {
        let limit = params.get("limit")
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(100)
            .min(1000);

        let offset = params.get("offset")
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(0);

        let event_type_filter = params.get("event_type");
        let severity_filter = params.get("severity");
        let user_id_filter = params.get("user_id")
            .and_then(|s| Uuid::parse_str(s).ok());

        let start_time = params.get("start_time")
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Utc));

        let end_time = params.get("end_time")
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(Utc::now);

        let mut query = r#"
            SELECT id, event_type, user_id, session_id, ip_address, user_agent,
                   resource, action, details, result, severity, timestamp,
                   request_id, endpoint, method, status_code, response_time_ms, error_message
            FROM audit_events
            WHERE timestamp <= $1
        "#.to_string();

        let mut param_count = 1;
        let mut query_params: Vec<Box<dyn sqlx::Encode<'_, sqlx::Postgres> + Send + Sync>> = vec![Box::new(end_time)];

        if let Some(start) = start_time {
            param_count += 1;
            query.push_str(&format!(" AND timestamp >= ${}", param_count));
            query_params.push(Box::new(start));
        }

        if let Some(event_type) = event_type_filter {
            param_count += 1;
            query.push_str(&format!(" AND event_type = ${}", param_count));
            query_params.push(Box::new(event_type.clone()));
        }

        if let Some(severity) = severity_filter {
            param_count += 1;
            query.push_str(&format!(" AND severity = ${}", param_count));
            query_params.push(Box::new(severity.clone()));
        }

        if let Some(user_id) = user_id_filter {
            param_count += 1;
            query.push_str(&format!(" AND user_id = ${}", param_count));
            query_params.push(Box::new(user_id));
        }

        query.push_str(" ORDER BY timestamp DESC");
        query.push_str(&format!(" LIMIT {} OFFSET {}", limit, offset));

        // For simplicity, returning a mock response
        // In a real implementation, you'd execute the dynamic query
        let events: Vec<(String, String, Option<String>, Option<String>, String, String, serde_json::Value, String, String, chrono::DateTime<Utc>)> = sqlx::query_as(
            "SELECT id, event_type, user_id, ip_address, resource, action,
                    details, result, severity, timestamp
             FROM audit_events
             ORDER BY timestamp DESC
             LIMIT $1 OFFSET $2"
        )
        .bind(limit)
        .bind(offset)
        .fetch_all(&self.database)
        .await?;

        let audit_events: Vec<serde_json::Value> = events
            .into_iter()
            .map(|e| serde_json::json!({
                "id": e.0,
                "event_type": e.1,
                "user_id": e.2,
                "ip_address": e.3,
                "resource": e.4,
                "action": e.5,
                "details": e.6,
                "result": e.7,
                "severity": e.8,
                "timestamp": e.9
            }))
            .collect();

        Ok(serde_json::json!({
            "events": audit_events,
            "total": audit_events.len(),
            "limit": limit,
            "offset": offset
        }))
    }

    pub async fn get_audit_stats(&self) -> Result<serde_json::Value> {
        let metrics = self.calculate_security_metrics().await?;

        Ok(serde_json::json!({
            "failed_logins_last_hour": metrics.failed_logins_last_hour,
            "failed_logins_last_24h": metrics.failed_logins_last_24h,
            "suspicious_activities_last_hour": metrics.suspicious_activities_last_hour,
            "rate_limit_violations_last_hour": metrics.rate_limit_violations_last_hour,
            "blocked_ips_count": metrics.blocked_ips_count,
            "active_sessions_count": metrics.active_sessions_count,
            "unique_users_last_24h": metrics.unique_users_last_24h,
            "api_requests_last_hour": metrics.api_requests_last_hour,
            "error_rate_last_hour": metrics.error_rate_last_hour,
            "top_error_endpoints": metrics.top_error_endpoints,
            "top_attacking_ips": metrics.top_attacking_ips,
            "geographic_distribution": metrics.geographic_distribution
        }))
    }

    async fn calculate_security_metrics(&self) -> Result<SecurityMetrics> {
        let one_hour_ago = Utc::now() - chrono::Duration::hours(1);
        let twenty_four_hours_ago = Utc::now() - chrono::Duration::hours(24);

        // Failed logins last hour
        let failed_logins_1h = sqlx::query_scalar::<_, Option<i32>>(
            "SELECT COUNT(*)::INTEGER
             FROM audit_events
             WHERE event_type = 'Authentication'
             AND action = 'login'
             AND result = 'Failure'
             AND timestamp > $1"
        )
        .bind(one_hour_ago)
        .fetch_one(&self.database)
        .await?
        .unwrap_or(0) as u32;

        // Failed logins last 24 hours
        let failed_logins_24h = sqlx::query_scalar::<_, Option<i32>>(
            "SELECT COUNT(*)::INTEGER
             FROM audit_events
             WHERE event_type = 'Authentication'
             AND action = 'login'
             AND result = 'Failure'
             AND timestamp > $1"
        )
        .bind(twenty_four_hours_ago)
        .fetch_one(&self.database)
        .await?
        .unwrap_or(0) as u32;

        // Suspicious activities last hour
        let suspicious_activities_1h = sqlx::query_scalar::<_, Option<i32>>(
            "SELECT COUNT(*)::INTEGER
             FROM audit_events
             WHERE event_type = 'SuspiciousActivity'
             AND timestamp > $1"
        )
        .bind(one_hour_ago)
        .fetch_one(&self.database)
        .await?
        .unwrap_or(0) as u32;

        // Rate limit violations last hour
        let rate_limit_violations_1h = sqlx::query_scalar::<_, Option<i32>>(
            "SELECT COUNT(*)::INTEGER
             FROM audit_events
             WHERE event_type = 'RateLimitViolation'
             AND timestamp > $1"
        )
        .bind(one_hour_ago)
        .fetch_one(&self.database)
        .await?
        .unwrap_or(0) as u32;

        // API requests last hour
        let api_requests_1h = sqlx::query_scalar::<_, Option<i32>>(
            "SELECT COUNT(*)::INTEGER
             FROM audit_events
             WHERE event_type = 'APIAccess'
             AND timestamp > $1"
        )
        .bind(one_hour_ago)
        .fetch_one(&self.database)
        .await?
        .unwrap_or(0) as u32;

        // Unique users last 24 hours
        let unique_users_24h = sqlx::query_scalar::<_, Option<i32>>(
            "SELECT COUNT(DISTINCT user_id)::INTEGER
             FROM audit_events
             WHERE user_id IS NOT NULL
             AND timestamp > $1"
        )
        .bind(twenty_four_hours_ago)
        .fetch_one(&self.database)
        .await?
        .unwrap_or(0) as u32;

        // Error rate calculation (mock for now)
        let error_rate_1h = if api_requests_1h > 0 {
            (failed_logins_1h as f64 / api_requests_1h as f64) * 100.0
        } else {
            0.0
        };

        // Calculate top error endpoints
        let top_error_endpoints: Vec<(String, u32)> = sqlx::query_as(
            "SELECT endpoint, COUNT(*) as count
             FROM audit_events
             WHERE result = 'Failure'
             AND timestamp > $1
             AND endpoint IS NOT NULL
             GROUP BY endpoint
             ORDER BY count DESC
             LIMIT 10"
        )
        .bind(one_hour_ago)
        .fetch_all(&self.database)
        .await?
        .into_iter()
        .map(|(endpoint, count): (String, i64)| (endpoint, count as u32))
        .collect();

        // Calculate top attacking IPs
        let top_attacking_ips: Vec<(String, u32)> = sqlx::query_as(
            "SELECT ip_address, COUNT(*) as count
             FROM audit_events
             WHERE result = 'Failure'
             AND timestamp > $1
             AND ip_address IS NOT NULL
             AND (event_type = 'ApiAccess' OR event_type = 'Authentication')
             GROUP BY ip_address
             ORDER BY count DESC
             LIMIT 10"
        )
        .bind(one_hour_ago)
        .fetch_all(&self.database)
        .await?
        .into_iter()
        .map(|(ip, count): (String, i64)| (ip, count as u32))
        .collect();

        // Count blocked IPs from recent rate limit violations
        let blocked_ips_count = sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(DISTINCT ip_address)
             FROM audit_events
             WHERE event_type = 'RateLimitViolation'
             AND timestamp > $1"
        )
        .bind(one_hour_ago)
        .fetch_one(&self.database)
        .await? as u32;

        // Count active sessions
        let active_sessions_count = sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(DISTINCT session_id)
             FROM audit_events
             WHERE session_id IS NOT NULL
             AND timestamp > $1"
        )
        .bind(one_hour_ago)
        .fetch_one(&self.database)
        .await? as u32;

        // Geographic distribution (simplified - based on IP prefixes)
        let geographic_distribution = HashMap::new(); // Would require IP geolocation database

        Ok(SecurityMetrics {
            failed_logins_last_hour: failed_logins_1h,
            failed_logins_last_24h: failed_logins_24h,
            suspicious_activities_last_hour: suspicious_activities_1h,
            rate_limit_violations_last_hour: rate_limit_violations_1h,
            blocked_ips_count,
            active_sessions_count,
            unique_users_last_24h: unique_users_24h,
            api_requests_last_hour: api_requests_1h,
            error_rate_last_hour: error_rate_1h,
            top_error_endpoints,
            top_attacking_ips,
            geographic_distribution,
        })
    }

    pub async fn cleanup_old_events(&self) -> Result<()> {
        // Get retention policies
        let retention_policies = sqlx::query_as::<_, (String, i32)>(
            "SELECT event_type, retention_days FROM audit_retention_policies"
        )
        .fetch_all(&self.database)
        .await?;

        for policy in retention_policies {
            let cutoff_date = Utc::now() - chrono::Duration::days(policy.1 as i64);

            let deleted_count = sqlx::query(
                "DELETE FROM audit_events WHERE event_type = $1 AND timestamp < $2"
            )
            .bind(&policy.0)
            .bind(cutoff_date)
            .execute(&self.database)
            .await?
            .rows_affected();

            if deleted_count > 0 {
                info!(
                    "Cleaned up {} old audit events of type {}",
                    deleted_count, policy.0
                );
            }
        }

        Ok(())
    }

    async fn get_events(&self, filter: AuditFilter) -> Result<Vec<AuditEvent>> {
        let mut query = sqlx::QueryBuilder::new(
            "SELECT id, event_type, user_id, session_id, ip_address, user_agent,
                    resource, action, details, result, severity, timestamp,
                    request_id, endpoint, method, status_code, response_time_ms, error_message
             FROM audit_events WHERE 1=1"
        );

        // Apply filters dynamically
        if let Some(user_id) = filter.user_id {
            query.push(" AND user_id = ");
            query.push_bind(user_id);
        }

        if let Some(resource) = filter.resource {
            query.push(" AND resource = ");
            query.push_bind(resource);
        }

        if let Some(action) = filter.action {
            query.push(" AND action = ");
            query.push_bind(action);
        }

        if let Some(start_time) = filter.start_time {
            query.push(" AND timestamp >= ");
            query.push_bind(start_time);
        }

        if let Some(end_time) = filter.end_time {
            query.push(" AND timestamp <= ");
            query.push_bind(end_time);
        }

        if let Some(ip_address) = filter.ip_address {
            query.push(" AND ip_address = ");
            query.push_bind(ip_address.to_string());
        }

        query.push(" ORDER BY timestamp DESC LIMIT ");
        query.push_bind(filter.limit.unwrap_or(100) as i64);

        if let Some(offset) = filter.offset {
            query.push(" OFFSET ");
            query.push_bind(offset as i64);
        }

        // Execute query and parse results
        let rows = query.build()
            .fetch_all(&self.database)
            .await?;

        let mut events = Vec::new();
        for row in rows {
            use sqlx::Row;
            let event = AuditEvent {
                id: row.get("id"),
                event_type: self.parse_event_type(row.get("event_type")),
                user_id: row.get("user_id"),
                session_id: row.get("session_id"),
                ip_address: row.get::<Option<String>, _>("ip_address")
                    .and_then(|s| s.parse().ok()),
                user_agent: row.get("user_agent"),
                resource: row.get("resource"),
                action: row.get("action"),
                details: row.get("details"),
                result: self.parse_result(row.get("result")),
                severity: self.parse_severity(row.get("severity")),
                timestamp: row.get("timestamp"),
                request_id: row.get("request_id"),
                endpoint: row.get("endpoint"),
                method: row.get("method"),
                status_code: row.get::<Option<i32>, _>("status_code").map(|s| s as u16),
                response_time_ms: row.get::<Option<i64>, _>("response_time_ms").map(|r| r as u64),
                error_message: row.get("error_message"),
            };
            events.push(event);
        }

        Ok(events)
    }

    fn parse_event_type(&self, s: String) -> AuditEventType {
        match s.as_str() {
            "Authentication" => AuditEventType::Authentication,
            "Authorization" => AuditEventType::Authorization,
            "DataAccess" => AuditEventType::DataAccess,
            "DataModification" => AuditEventType::DataModification,
            "UserManagement" => AuditEventType::UserManagement,
            "RoleManagement" => AuditEventType::RoleManagement,
            "ConfigurationChange" => AuditEventType::ConfigurationChange,
            "ApiAccess" => AuditEventType::ApiAccess,
            "SecurityIncident" => AuditEventType::SecurityIncident,
            "RateLimitViolation" => AuditEventType::RateLimitViolation,
            "ValidationFailure" => AuditEventType::ValidationFailure,
            _ => AuditEventType::SystemError,
        }
    }

    fn parse_result(&self, s: String) -> AuditResult {
        match s.as_str() {
            "Success" => AuditResult::Success,
            "Failure" => AuditResult::Failure,
            "Partial" => AuditResult::Partial,
            "Blocked" => AuditResult::Blocked,
            _ => AuditResult::Warning,
        }
    }

    fn parse_severity(&self, s: String) -> AuditSeverity {
        match s.as_str() {
            "Low" => AuditSeverity::Low,
            "Medium" => AuditSeverity::Medium,
            "High" => AuditSeverity::High,
            _ => AuditSeverity::Critical,
        }
    }

    pub async fn export_audit_logs(
        &self,
        filter: AuditFilter,
        format: &str,
    ) -> Result<Vec<u8>> {
        // Fetch events based on filter
        let events = self.get_events(filter).await?;

        match format {
            "csv" => self.export_as_csv(&events),
            "json" => self.export_as_json(&events),
            "xml" => self.export_as_xml(&events),
            _ => Err(anyhow::anyhow!("Unsupported export format: {}", format)),
        }
    }

    fn export_as_csv(&self, events: &[AuditEvent]) -> Result<Vec<u8>> {
        let mut csv = String::from("id,timestamp,event_type,user_id,session_id,ip_address,resource,action,result,severity,endpoint,method,status_code,response_time_ms,error_message\n");

        for event in events {
            csv.push_str(&format!(
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                event.id,
                event.timestamp.to_rfc3339(),
                format!("{:?}", event.event_type),
                event.user_id.map(|id| id.to_string()).unwrap_or_default(),
                event.session_id.clone().unwrap_or_default(),
                event.ip_address.map(|ip| ip.to_string()).unwrap_or_default(),
                event.resource,
                event.action,
                format!("{:?}", event.result),
                format!("{:?}", event.severity),
                event.endpoint.clone().unwrap_or_default(),
                event.method.clone().unwrap_or_default(),
                event.status_code.map(|s| s.to_string()).unwrap_or_default(),
                event.response_time_ms.map(|t| t.to_string()).unwrap_or_default(),
                event.error_message.clone().unwrap_or_default().replace(",", ";")
            ));
        }

        Ok(csv.into_bytes())
    }

    fn export_as_json(&self, events: &[AuditEvent]) -> Result<Vec<u8>> {
        let json = serde_json::to_string_pretty(events)?;
        Ok(json.into_bytes())
    }

    fn export_as_xml(&self, events: &[AuditEvent]) -> Result<Vec<u8>> {
        let mut xml = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<audit_events>\n");

        for event in events {
            xml.push_str("  <event>\n");
            xml.push_str(&format!("    <id>{}</id>\n", event.id));
            xml.push_str(&format!("    <timestamp>{}</timestamp>\n", event.timestamp.to_rfc3339()));
            xml.push_str(&format!("    <event_type>{:?}</event_type>\n", event.event_type));
            if let Some(user_id) = event.user_id {
                xml.push_str(&format!("    <user_id>{}</user_id>\n", user_id));
            }
            xml.push_str(&format!("    <resource>{}</resource>\n", event.resource));
            xml.push_str(&format!("    <action>{}</action>\n", event.action));
            xml.push_str(&format!("    <result>{:?}</result>\n", event.result));
            xml.push_str(&format!("    <severity>{:?}</severity>\n", event.severity));
            if let Some(endpoint) = &event.endpoint {
                xml.push_str(&format!("    <endpoint>{}</endpoint>\n", endpoint));
            }
            if let Some(method) = &event.method {
                xml.push_str(&format!("    <method>{}</method>\n", method));
            }
            if let Some(status_code) = event.status_code {
                xml.push_str(&format!("    <status_code>{}</status_code>\n", status_code));
            }
            xml.push_str("  </event>\n");
        }

        xml.push_str("</audit_events>\n");
        Ok(xml.into_bytes())
    }

    pub async fn search_events(&self, query: &str, limit: u32) -> Result<Vec<AuditEvent>> {
        // Text search across resource, action, and error_message fields
        let search_pattern = format!("%{}%", query);

        let mut query_builder = sqlx::QueryBuilder::new(
            "SELECT id, event_type, user_id, session_id, ip_address, user_agent,
                    resource, action, details, result, severity, timestamp,
                    request_id, endpoint, method, status_code, response_time_ms, error_message
             FROM audit_events
             WHERE resource ILIKE "
        );
        query_builder.push_bind(&search_pattern);
        query_builder.push(" OR action ILIKE ");
        query_builder.push_bind(&search_pattern);
        query_builder.push(" OR error_message ILIKE ");
        query_builder.push_bind(&search_pattern);
        query_builder.push(" ORDER BY timestamp DESC LIMIT ");
        query_builder.push_bind(limit as i64);

        let rows = query_builder.build().fetch_all(&self.database).await?;

        let mut events = Vec::new();
        for row in rows {
            use sqlx::Row;
            let event = AuditEvent {
                id: row.get("id"),
                event_type: self.parse_event_type(row.get("event_type")),
                user_id: row.get("user_id"),
                session_id: row.get("session_id"),
                ip_address: row.get::<Option<String>, _>("ip_address")
                    .and_then(|s| s.parse().ok()),
                user_agent: row.get("user_agent"),
                resource: row.get("resource"),
                action: row.get("action"),
                details: row.get("details"),
                result: self.parse_result(row.get("result")),
                severity: self.parse_severity(row.get("severity")),
                timestamp: row.get("timestamp"),
                request_id: row.get("request_id"),
                endpoint: row.get("endpoint"),
                method: row.get("method"),
                status_code: row.get::<Option<i32>, _>("status_code").map(|s| s as u16),
                response_time_ms: row.get::<Option<i64>, _>("response_time_ms").map(|r| r as u64),
                error_message: row.get("error_message"),
            };
            events.push(event);
        }

        Ok(events)
    }

    pub async fn get_security_dashboard_data(&self) -> Result<serde_json::Value> {
        let metrics = self.calculate_security_metrics().await?;

        // Get recent critical events
        let critical_events = sqlx::query_as::<_, (i32, String, Option<String>, Option<String>, chrono::DateTime<Utc>, Option<serde_json::Value>)>(
            "SELECT id, event_type, resource, action, timestamp, details
            FROM audit_events
            WHERE severity = 'Critical'
            AND timestamp > NOW() - INTERVAL '24 hours'
            ORDER BY timestamp DESC
            LIMIT 10"
        )
        .fetch_all(&self.database)
        .await?;

        let critical_events_json: Vec<serde_json::Value> = critical_events
            .into_iter()
            .map(|e| serde_json::json!({
                "id": e.0,
                "event_type": e.1,
                "resource": e.2,
                "action": e.3,
                "timestamp": e.4,
                "details": e.5
            }))
            .collect();

        Ok(serde_json::json!({
            "metrics": metrics,
            "critical_events": critical_events_json,
            "security_score": self.calculate_security_score(&metrics),
            "recommendations": self.generate_security_recommendations(&metrics)
        }))
    }

    fn calculate_security_score(&self, metrics: &SecurityMetrics) -> u8 {
        let mut score = 100u8;

        // Deduct points for security issues
        if metrics.failed_logins_last_hour > 10 {
            score = score.saturating_sub(20);
        }
        if metrics.suspicious_activities_last_hour > 5 {
            score = score.saturating_sub(15);
        }
        if metrics.rate_limit_violations_last_hour > 20 {
            score = score.saturating_sub(10);
        }
        if metrics.error_rate_last_hour > 5.0 {
            score = score.saturating_sub(10);
        }

        score
    }

    fn generate_security_recommendations(&self, metrics: &SecurityMetrics) -> Vec<String> {
        let mut recommendations = Vec::new();

        if metrics.failed_logins_last_hour > 10 {
            recommendations.push("High number of failed logins detected. Consider implementing additional authentication measures.".to_string());
        }

        if metrics.suspicious_activities_last_hour > 5 {
            recommendations.push("Suspicious activities detected. Review and investigate potential security threats.".to_string());
        }

        if metrics.rate_limit_violations_last_hour > 20 {
            recommendations.push("High rate limit violations. Consider adjusting rate limits or implementing IP blocking.".to_string());
        }

        if metrics.error_rate_last_hour > 5.0 {
            recommendations.push("High error rate detected. Investigate application stability and security.".to_string());
        }

        if recommendations.is_empty() {
            recommendations.push("Security posture is good. Continue monitoring.".to_string());
        }

        recommendations
    }
}