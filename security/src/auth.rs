use anyhow::{Context, Result};
use argon2::{Argon2, PasswordHash, PasswordHasher, PasswordVerifier};
use argon2::password_hash::{rand_core::OsRng, SaltString};
use chrono::{DateTime, Duration, Utc};
use jsonwebtoken::{decode, encode, Algorithm, DecodingKey, EncodingKey, Header, Validation};
use redis::AsyncCommands;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use tracing::info;
use uuid::Uuid;
use validator::{Validate, ValidationError};

#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
pub struct User {
    pub id: Uuid,
    pub email: String,
    pub username: String,
    pub password_hash: String,
    pub first_name: Option<String>,
    pub last_name: Option<String>,
    pub is_active: bool,
    pub is_verified: bool,
    pub failed_login_attempts: i32,
    pub locked_until: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub last_login: Option<DateTime<Utc>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,        // User ID
    pub email: String,
    pub username: String,
    pub roles: Vec<String>,
    pub permissions: Vec<String>,
    pub exp: usize,         // Expiration timestamp
    pub iat: usize,         // Issued at timestamp
    pub jti: String,        // JWT ID for revocation
    pub session_id: String, // Session tracking
}

#[derive(Debug, Serialize, Deserialize, Validate)]
pub struct RegisterRequest {
    #[validate(email(message = "Invalid email format"))]
    pub email: String,

    #[validate(length(min = 3, max = 50, message = "Username must be 3-50 characters"))]
    #[validate(regex(path = "USERNAME_REGEX", message = "Username can only contain alphanumeric characters and underscores"))]
    pub username: String,

    #[validate(length(min = 8, message = "Password must be at least 8 characters"))]
    #[validate(custom = "validate_password_strength")]
    pub password: String,

    pub first_name: Option<String>,
    pub last_name: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Validate)]
pub struct LoginRequest {
    #[validate(length(min = 1, message = "Email or username is required"))]
    pub login: String, // Can be email or username

    #[validate(length(min = 1, message = "Password is required"))]
    pub password: String,

    pub remember_me: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TokenResponse {
    pub access_token: String,
    pub refresh_token: String,
    pub token_type: String,
    pub expires_in: u64,
    pub user: UserProfile,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UserProfile {
    pub id: Uuid,
    pub email: String,
    pub username: String,
    pub first_name: Option<String>,
    pub last_name: Option<String>,
    pub roles: Vec<String>,
    pub permissions: Vec<String>,
    pub is_verified: bool,
    pub last_login: Option<DateTime<Utc>>,
}

#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
pub struct RefreshToken {
    pub token: String,
    pub user_id: Uuid,
    pub expires_at: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
    pub is_revoked: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PasswordResetToken {
    pub token: String,
    pub user_id: Uuid,
    pub expires_at: DateTime<Utc>,
    pub used: bool,
}

lazy_static::lazy_static! {
    static ref USERNAME_REGEX: regex::Regex = regex::Regex::new(r"^[a-zA-Z0-9_]+$").unwrap();
}

fn validate_password_strength(password: &str) -> Result<(), ValidationError> {
    let mut score = 0;

    if password.len() >= 12 { score += 1; }
    if password.chars().any(|c| c.is_lowercase()) { score += 1; }
    if password.chars().any(|c| c.is_uppercase()) { score += 1; }
    if password.chars().any(|c| c.is_numeric()) { score += 1; }
    if password.chars().any(|c| !c.is_alphanumeric()) { score += 1; }

    if score < 3 {
        return Err(ValidationError::new("Password must contain at least 3 of: lowercase, uppercase, numbers, special characters"));
    }

    // Check for common weak patterns
    let weak_patterns = ["password", "123456", "qwerty", "admin"];
    let password_lower = password.to_lowercase();

    for pattern in weak_patterns {
        if password_lower.contains(pattern) {
            return Err(ValidationError::new("Password contains common weak patterns"));
        }
    }

    Ok(())
}

pub struct AuthService {
    database: PgPool,
    redis: redis::aio::ConnectionManager,
    jwt_secret: String,
    encryption_key: String,
    token_expiry: Duration,
    refresh_token_expiry: Duration,
    argon2: Argon2<'static>,
}

impl AuthService {
    pub async fn new(
        database_url: &str,
        redis_url: &str,
        jwt_secret: &str,
        encryption_key: &str,
        token_expiry_seconds: u64,
        refresh_token_expiry_seconds: u64,
    ) -> Result<Self> {
        let database = PgPool::connect(database_url)
            .await
            .context("Failed to connect to database")?;

        let redis_client = redis::Client::open(redis_url)
            .context("Failed to create Redis client")?;
        let redis = redis::aio::ConnectionManager::new(redis_client)
            .await
            .context("Failed to create Redis connection")?;

        Self::init_database(&database).await?;

        Ok(Self {
            database,
            redis,
            jwt_secret: jwt_secret.to_string(),
            encryption_key: encryption_key.to_string(),
            token_expiry: Duration::seconds(token_expiry_seconds as i64),
            refresh_token_expiry: Duration::seconds(refresh_token_expiry_seconds as i64),
            argon2: Argon2::default(),
        })
    }

    async fn init_database(database: &PgPool) -> Result<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                email VARCHAR(255) UNIQUE NOT NULL,
                username VARCHAR(50) UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                first_name VARCHAR(100),
                last_name VARCHAR(100),
                is_active BOOLEAN NOT NULL DEFAULT true,
                is_verified BOOLEAN NOT NULL DEFAULT false,
                failed_login_attempts INTEGER NOT NULL DEFAULT 0,
                locked_until TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                last_login TIMESTAMPTZ
            );

            CREATE TABLE IF NOT EXISTS refresh_tokens (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                token TEXT UNIQUE NOT NULL,
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                expires_at TIMESTAMPTZ NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                is_revoked BOOLEAN NOT NULL DEFAULT false,
                revoked_at TIMESTAMPTZ,
                device_info JSONB
            );

            CREATE TABLE IF NOT EXISTS password_reset_tokens (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                token TEXT UNIQUE NOT NULL,
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                expires_at TIMESTAMPTZ NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                used BOOLEAN NOT NULL DEFAULT false,
                used_at TIMESTAMPTZ
            );

            CREATE TABLE IF NOT EXISTS user_sessions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                session_token TEXT UNIQUE NOT NULL,
                ip_address INET,
                user_agent TEXT,
                expires_at TIMESTAMPTZ NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                last_activity TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                is_active BOOLEAN NOT NULL DEFAULT true
            );

            CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
            CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
            CREATE INDEX IF NOT EXISTS idx_refresh_tokens_user_id ON refresh_tokens(user_id);
            CREATE INDEX IF NOT EXISTS idx_refresh_tokens_expires_at ON refresh_tokens(expires_at);
            CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
            CREATE INDEX IF NOT EXISTS idx_user_sessions_expires_at ON user_sessions(expires_at);
            "#,
        )
        .execute(database)
        .await
        .context("Failed to initialize auth database tables")?;

        Ok(())
    }

    pub async fn register_user(&self, payload: serde_json::Value) -> Result<serde_json::Value> {
        let request: RegisterRequest = serde_json::from_value(payload)
            .context("Invalid registration request format")?;

        request.validate()
            .context("Registration validation failed")?;

        // Check if user already exists
        let existing_user = sqlx::query_scalar::<_, i32>(
            "SELECT id FROM users WHERE email = $1 OR username = $2"
        )
        .bind(&request.email)
        .bind(&request.username)
        .fetch_optional(&self.database)
        .await?;

        if existing_user.is_some() {
            return Err(anyhow::anyhow!("User with this email or username already exists"));
        }

        // Hash password
        let salt = SaltString::generate(&mut OsRng);
        let password_hash = self.argon2
            .hash_password(request.password.as_bytes(), &salt)
            .map_err(|e| anyhow::anyhow!("Failed to hash password: {:?}", e))?
            .to_string();

        // Create user
        let user_id: Uuid = sqlx::query_scalar(
            "INSERT INTO users (email, username, password_hash, first_name, last_name)
             VALUES ($1, $2, $3, $4, $5)
             RETURNING id"
        )
        .bind(&request.email)
        .bind(&request.username)
        .bind(&password_hash)
        .bind(&request.first_name)
        .bind(&request.last_name)
        .fetch_one(&self.database)
        .await
        .context("Failed to create user")?;

        info!("User registered successfully: {} ({})", request.username, user_id);

        // Generate verification token (in production, send verification email)
        let verification_token = self.generate_verification_token(&user_id).await?;

        Ok(serde_json::json!({
            "message": "User registered successfully",
            "user_id": user_id,
            "verification_required": true,
            "verification_token": verification_token // Remove in production
        }))
    }

    pub async fn login_user(&self, payload: serde_json::Value) -> Result<serde_json::Value> {
        let request: LoginRequest = serde_json::from_value(payload)
            .context("Invalid login request format")?;

        request.validate()
            .context("Login validation failed")?;

        // Get user by email or username
        let user = sqlx::query_as::<_, User>(
            "SELECT id, email, username, password_hash, first_name, last_name,
                    is_active, is_verified, failed_login_attempts, locked_until,
                    created_at, updated_at, last_login
             FROM users
             WHERE email = $1 OR username = $1"
        )
        .bind(&request.login)
        .fetch_optional(&self.database)
        .await?;

        let mut user = user.ok_or_else(|| anyhow::anyhow!("Invalid credentials"))?;

        // Check if account is locked
        if let Some(locked_until) = user.locked_until {
            if Utc::now() < locked_until {
                return Err(anyhow::anyhow!("Account is temporarily locked due to too many failed login attempts"));
            } else {
                // Unlock account
                sqlx::query(
                    "UPDATE users SET locked_until = NULL, failed_login_attempts = 0 WHERE id = $1"
                )
                .bind(user.id)
                .execute(&self.database)
                .await?;
                user.locked_until = None;
                user.failed_login_attempts = 0;
            }
        }

        // Check if account is active
        if !user.is_active {
            return Err(anyhow::anyhow!("Account is deactivated"));
        }

        // Verify password
        let parsed_hash = PasswordHash::new(&user.password_hash)
            .map_err(|e| anyhow::anyhow!("Failed to parse password hash: {:?}", e))?;

        if self.argon2.verify_password(request.password.as_bytes(), &parsed_hash).is_err() {
            // Increment failed login attempts
            let new_attempts = user.failed_login_attempts + 1;
            let locked_until = if new_attempts >= 5 {
                Some(Utc::now() + Duration::minutes(15)) // Lock for 15 minutes after 5 failed attempts
            } else {
                None
            };

            sqlx::query(
                "UPDATE users SET failed_login_attempts = $1, locked_until = $2 WHERE id = $3"
            )
            .bind(new_attempts)
            .bind(locked_until)
            .bind(user.id)
            .execute(&self.database)
            .await?;

            return Err(anyhow::anyhow!("Invalid credentials"));
        }

        // Reset failed login attempts and update last login
        sqlx::query(
            "UPDATE users SET failed_login_attempts = 0, locked_until = NULL, last_login = NOW() WHERE id = $1"
        )
        .bind(user.id)
        .execute(&self.database)
        .await?;

        // Get user roles and permissions
        let (roles, permissions) = self.get_user_roles_and_permissions(&user.id).await?;

        // Generate tokens
        let session_id = Uuid::new_v4().to_string();
        let access_token = self.generate_access_token(&user, &roles, &permissions, &session_id)?;
        let refresh_token = self.generate_refresh_token(&user.id, request.remember_me.unwrap_or(false)).await?;

        // Store session
        self.store_user_session(&user.id, &session_id, &access_token).await?;

        info!("User logged in successfully: {} ({})", user.username, user.id);

        let response = TokenResponse {
            access_token,
            refresh_token: refresh_token.token,
            token_type: "Bearer".to_string(),
            expires_in: self.token_expiry.num_seconds() as u64,
            user: UserProfile {
                id: user.id,
                email: user.email,
                username: user.username,
                first_name: user.first_name,
                last_name: user.last_name,
                roles,
                permissions,
                is_verified: user.is_verified,
                last_login: user.last_login,
            },
        };

        Ok(serde_json::to_value(response)?)
    }

    pub async fn refresh_token(&self, payload: serde_json::Value) -> Result<serde_json::Value> {
        let refresh_token_str = payload.get("refresh_token")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Refresh token is required"))?;

        // Validate refresh token
        let refresh_token = sqlx::query_as::<_, RefreshToken>(
            "SELECT token, user_id, expires_at, created_at, is_revoked
             FROM refresh_tokens
             WHERE token = $1 AND is_revoked = false AND expires_at > NOW()"
        )
        .bind(refresh_token_str)
        .fetch_optional(&self.database)
        .await?
        .ok_or_else(|| anyhow::anyhow!("Invalid or expired refresh token"))?;

        // Get user
        let user = sqlx::query_as::<_, User>(
            "SELECT id, email, username, password_hash, first_name, last_name,
                    is_active, is_verified, failed_login_attempts, locked_until,
                    created_at, updated_at, last_login
             FROM users
             WHERE id = $1 AND is_active = true"
        )
        .bind(refresh_token.user_id)
        .fetch_optional(&self.database)
        .await?
        .ok_or_else(|| anyhow::anyhow!("User not found or inactive"))?;

        // Get roles and permissions
        let (roles, permissions) = self.get_user_roles_and_permissions(&user.id).await?;

        // Generate new tokens
        let session_id = Uuid::new_v4().to_string();
        let new_access_token = self.generate_access_token(&user, &roles, &permissions, &session_id)?;
        let new_refresh_token = self.generate_refresh_token(&user.id, true).await?;

        // Revoke old refresh token
        sqlx::query(
            "UPDATE refresh_tokens SET is_revoked = true, revoked_at = NOW() WHERE token = $1"
        )
        .bind(refresh_token_str)
        .execute(&self.database)
        .await?;

        // Store new session
        self.store_user_session(&user.id, &session_id, &new_access_token).await?;

        let response = TokenResponse {
            access_token: new_access_token,
            refresh_token: new_refresh_token.token,
            token_type: "Bearer".to_string(),
            expires_in: self.token_expiry.num_seconds() as u64,
            user: UserProfile {
                id: user.id,
                email: user.email,
                username: user.username,
                first_name: user.first_name,
                last_name: user.last_name,
                roles,
                permissions,
                is_verified: user.is_verified,
                last_login: user.last_login,
            },
        };

        Ok(serde_json::to_value(response)?)
    }

    pub async fn logout_user(&self, payload: serde_json::Value) -> Result<serde_json::Value> {
        let token = payload.get("token")
            .or_else(|| payload.get("access_token"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Token is required"))?;

        // Decode token to get claims
        let claims = self.decode_token(token)?;

        // Revoke refresh tokens
        sqlx::query(
            "UPDATE refresh_tokens SET is_revoked = true, revoked_at = NOW() WHERE user_id = $1 AND is_revoked = false"
        )
        .bind(Uuid::parse_str(&claims.sub)?)
        .execute(&self.database)
        .await?;

        // Deactivate session
        sqlx::query(
            "UPDATE user_sessions SET is_active = false WHERE session_token = $1"
        )
        .bind(&claims.jti)
        .execute(&self.database)
        .await?;

        // Add token to blacklist in Redis
        let mut conn = self.redis.clone();
        redis::cmd("SETEX")
            .arg(format!("blacklist:{}", claims.jti))
            .arg(self.token_expiry.num_seconds())
            .arg("revoked")
            .query_async::<()>(&mut conn)
            .await?;

        info!("User logged out successfully: {} ({})", claims.username, claims.sub);

        Ok(serde_json::json!({
            "message": "Logged out successfully"
        }))
    }

    pub async fn verify_token(&self, headers: axum::http::HeaderMap) -> Result<serde_json::Value> {
        let auth_header = headers.get("Authorization")
            .and_then(|header| header.to_str().ok())
            .and_then(|header| header.strip_prefix("Bearer "))
            .ok_or_else(|| anyhow::anyhow!("Missing or invalid Authorization header"))?;

        let claims = self.decode_and_validate_token(auth_header).await?;

        Ok(serde_json::json!({
            "valid": true,
            "claims": claims
        }))
    }

    pub async fn change_password(&self, payload: serde_json::Value) -> Result<serde_json::Value> {
        let user_id = payload.get("user_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("User ID is required"))?;

        let current_password = payload.get("current_password")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Current password is required"))?;

        let new_password = payload.get("new_password")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("New password is required"))?;

        // Validate new password strength
        validate_password_strength(new_password)?;

        let user_uuid = Uuid::parse_str(user_id)?;

        // Get current user
        let user: (String,) = sqlx::query_as(
            "SELECT password_hash FROM users WHERE id = $1"
        )
        .bind(user_uuid)
        .fetch_optional(&self.database)
        .await?
        .ok_or_else(|| anyhow::anyhow!("User not found"))?;

        // Verify current password
        let parsed_hash = PasswordHash::new(&user.0)
            .map_err(|e| anyhow::anyhow!("Failed to parse password hash: {:?}", e))?;
        self.argon2.verify_password(current_password.as_bytes(), &parsed_hash)
            .map_err(|_| anyhow::anyhow!("Current password is incorrect"))?;

        // Hash new password
        let salt = SaltString::generate(&mut OsRng);
        let new_password_hash = self.argon2
            .hash_password(new_password.as_bytes(), &salt)
            .map_err(|e| anyhow::anyhow!("Failed to hash password: {:?}", e))?
            .to_string();

        // Update password
        sqlx::query(
            "UPDATE users SET password_hash = $1, updated_at = NOW() WHERE id = $2"
        )
        .bind(&new_password_hash)
        .bind(user_uuid)
        .execute(&self.database)
        .await?;

        // Revoke all refresh tokens to force re-authentication
        sqlx::query(
            "UPDATE refresh_tokens SET is_revoked = true, revoked_at = NOW() WHERE user_id = $1"
        )
        .bind(user_uuid)
        .execute(&self.database)
        .await?;

        info!("Password changed successfully for user: {}", user_id);

        Ok(serde_json::json!({
            "message": "Password changed successfully"
        }))
    }

    pub async fn reset_password(&self, payload: serde_json::Value) -> Result<serde_json::Value> {
        let email = payload.get("email")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Email is required"))?;

        // Check if user exists
        let user = sqlx::query_scalar::<_, i32>(
            "SELECT id FROM users WHERE email = $1"
        )
        .bind(email)
        .fetch_optional(&self.database)
        .await?;

        // Always return success to prevent email enumeration
        if user.is_none() {
            return Ok(serde_json::json!({
                "message": "If the email exists, a password reset link has been sent"
            }));
        }

        let user_id = user.unwrap();

        // Generate password reset token
        let reset_token = Uuid::new_v4().to_string();
        let expires_at = Utc::now() + Duration::hours(1); // 1 hour expiry

        sqlx::query(
            "INSERT INTO password_reset_tokens (token, user_id, expires_at)
             VALUES ($1, $2, $3)"
        )
        .bind(&reset_token)
        .bind(user_id)
        .bind(expires_at)
        .execute(&self.database)
        .await?;

        // In production, send email with reset link
        info!("Password reset token generated for user: {}", user_id);

        Ok(serde_json::json!({
            "message": "If the email exists, a password reset link has been sent",
            "reset_token": reset_token // Remove in production
        }))
    }

    pub async fn confirm_password_reset(&self, payload: serde_json::Value) -> Result<serde_json::Value> {
        let token = payload.get("token")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Reset token is required"))?;

        let new_password = payload.get("new_password")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("New password is required"))?;

        // Validate password strength
        validate_password_strength(new_password)?;

        // Validate reset token
        let reset_token: (i32, chrono::DateTime<Utc>, bool) = sqlx::query_as(
            "SELECT user_id, expires_at, used
             FROM password_reset_tokens
             WHERE token = $1 AND expires_at > NOW() AND used = false"
        )
        .bind(token)
        .fetch_optional(&self.database)
        .await?
        .ok_or_else(|| anyhow::anyhow!("Invalid or expired reset token"))?;

        // Hash new password
        let salt = SaltString::generate(&mut OsRng);
        let password_hash = self.argon2
            .hash_password(new_password.as_bytes(), &salt)
            .map_err(|e| anyhow::anyhow!("Failed to hash password: {:?}", e))?
            .to_string();

        // Update password and mark token as used
        let mut tx = self.database.begin().await?;

        sqlx::query(
            "UPDATE users SET password_hash = $1, updated_at = NOW() WHERE id = $2"
        )
        .bind(&password_hash)
        .bind(reset_token.0)
        .execute(&mut *tx)
        .await?;

        sqlx::query(
            "UPDATE password_reset_tokens SET used = true, used_at = NOW() WHERE token = $1"
        )
        .bind(token)
        .execute(&mut *tx)
        .await?;

        // Revoke all refresh tokens
        sqlx::query(
            "UPDATE refresh_tokens SET is_revoked = true, revoked_at = NOW() WHERE user_id = $1"
        )
        .bind(reset_token.0)
        .execute(&mut *tx)
        .await?;

        tx.commit().await?;

        info!("Password reset successfully for user: {}", reset_token.0);

        Ok(serde_json::json!({
            "message": "Password reset successfully"
        }))
    }

    async fn get_user_roles_and_permissions(&self, user_id: &Uuid) -> Result<(Vec<String>, Vec<String>)> {
        // Fetch user roles from database
        let roles = sqlx::query_scalar::<_, String>(
            "SELECT DISTINCT r.name
            FROM roles r
            JOIN user_roles ur ON r.id = ur.role_id
            WHERE ur.user_id = $1
            AND ur.is_active = true
            AND (ur.expires_at IS NULL OR ur.expires_at > NOW())"
        )
        .bind(user_id)
        .fetch_all(&self.database)
        .await?;

        // Fetch user permissions from database
        let permissions = sqlx::query_scalar::<_, String>(
            "SELECT DISTINCT p.name
            FROM permissions p
            JOIN role_permissions rp ON p.id = rp.permission_id
            JOIN user_roles ur ON rp.role_id = ur.role_id
            WHERE ur.user_id = $1
            AND ur.is_active = true
            AND (ur.expires_at IS NULL OR ur.expires_at > NOW())"
        )
        .bind(user_id)
        .fetch_all(&self.database)
        .await?;

        // If no roles found, return default user role
        if roles.is_empty() {
            return Ok((vec!["user".to_string()], vec!["read:own".to_string()]));
        }

        Ok((roles, permissions))
    }

    fn generate_access_token(
        &self,
        user: &User,
        roles: &[String],
        permissions: &[String],
        session_id: &str,
    ) -> Result<String> {
        let now = Utc::now();
        let exp = (now + self.token_expiry).timestamp() as usize;
        let jti = Uuid::new_v4().to_string();

        let claims = Claims {
            sub: user.id.to_string(),
            email: user.email.clone(),
            username: user.username.clone(),
            roles: roles.to_vec(),
            permissions: permissions.to_vec(),
            exp,
            iat: now.timestamp() as usize,
            jti,
            session_id: session_id.to_string(),
        };

        let header = Header::new(Algorithm::HS256);
        let encoding_key = EncodingKey::from_secret(self.jwt_secret.as_bytes());

        encode(&header, &claims, &encoding_key)
            .context("Failed to generate access token")
    }

    async fn generate_refresh_token(&self, user_id: &Uuid, long_lived: bool) -> Result<RefreshToken> {
        let token = Uuid::new_v4().to_string();
        let expires_at = if long_lived {
            Utc::now() + Duration::days(30) // 30 days for "remember me"
        } else {
            Utc::now() + self.refresh_token_expiry
        };

        sqlx::query(
            "INSERT INTO refresh_tokens (token, user_id, expires_at)
             VALUES ($1, $2, $3)"
        )
        .bind(&token)
        .bind(user_id)
        .bind(expires_at)
        .execute(&self.database)
        .await?;

        Ok(RefreshToken {
            token,
            user_id: *user_id,
            expires_at,
            created_at: Utc::now(),
            is_revoked: false,
        })
    }

    async fn store_user_session(&self, user_id: &Uuid, session_id: &str, token: &str) -> Result<()> {
        sqlx::query(
            "INSERT INTO user_sessions (user_id, session_token, jwt_token, expires_at)
             VALUES ($1, $2, $3, $4)"
        )
        .bind(user_id)
        .bind(session_id)
        .bind(token)
        .bind(Utc::now() + self.token_expiry)
        .execute(&self.database)
        .await?;

        // Also store in Redis for fast lookups (both session ID and JWT token)
        let mut conn = self.redis.clone();
        redis::cmd("SETEX")
            .arg(format!("session:{}", session_id))
            .arg(self.token_expiry.num_seconds())
            .arg(serde_json::json!({
                "user_id": user_id.to_string(),
                "token": token
            }).to_string())
            .query_async::<()>(&mut conn)
            .await?;

        Ok(())
    }

    fn decode_token(&self, token: &str) -> Result<Claims> {
        let decoding_key = DecodingKey::from_secret(self.jwt_secret.as_bytes());
        let validation = Validation::new(Algorithm::HS256);

        decode::<Claims>(token, &decoding_key, &validation)
            .map(|token_data| token_data.claims)
            .context("Failed to decode token")
    }

    pub async fn decode_and_validate_token(&self, token: &str) -> Result<Claims> {
        let claims = self.decode_token(token)?;

        // Check if token is blacklisted
        let mut conn = self.redis.clone();
        let blacklisted: Option<String> = conn.get(format!("blacklist:{}", claims.jti)).await?;

        if blacklisted.is_some() {
            return Err(anyhow::anyhow!("Token has been revoked"));
        }

        // Check if session is still active
        let session_active: Option<String> = conn.get(format!("session:{}", claims.session_id)).await?;

        if session_active.is_none() {
            return Err(anyhow::anyhow!("Session has expired"));
        }

        Ok(claims)
    }

    async fn generate_verification_token(&self, user_id: &Uuid) -> Result<String> {
        let token = Uuid::new_v4().to_string();

        // Store verification token (in production, use a proper verification system)
        let mut conn = self.redis.clone();
        redis::cmd("SETEX")
            .arg(format!("verify:{}", token))
            .arg(86400) // 24 hours
            .arg(user_id.to_string())
            .query_async::<()>(&mut conn)
            .await?;

        Ok(token)
    }

    pub fn encrypt_sensitive_data(&self, data: &str) -> Result<String> {
        use aes_gcm::{
            aead::{Aead, KeyInit},
            Aes256Gcm, Nonce,
        };
        use base64::{Engine as _, engine::general_purpose};

        // Derive a 32-byte key from encryption_key
        let key_bytes = self.derive_encryption_key()?;
        let cipher = Aes256Gcm::new_from_slice(&key_bytes)
            .context("Failed to create cipher")?;

        // Generate random nonce
        let nonce_bytes = rand::random::<[u8; 12]>();
        let nonce = Nonce::from_slice(&nonce_bytes);

        // Encrypt data
        let ciphertext = cipher
            .encrypt(nonce, data.as_bytes())
            .map_err(|e| anyhow::anyhow!("Encryption failed: {}", e))?;

        // Combine nonce + ciphertext and encode as base64
        let mut combined = nonce_bytes.to_vec();
        combined.extend_from_slice(&ciphertext);

        Ok(general_purpose::STANDARD.encode(combined))
    }

    pub fn decrypt_sensitive_data(&self, encrypted_data: &str) -> Result<String> {
        use aes_gcm::{
            aead::{Aead, KeyInit},
            Aes256Gcm, Nonce,
        };
        use base64::{Engine as _, engine::general_purpose};

        // Decode from base64
        let combined = general_purpose::STANDARD
            .decode(encrypted_data)
            .context("Failed to decode base64")?;

        if combined.len() < 12 {
            return Err(anyhow::anyhow!("Invalid encrypted data"));
        }

        // Split nonce and ciphertext
        let (nonce_bytes, ciphertext) = combined.split_at(12);
        let nonce = Nonce::from_slice(nonce_bytes);

        // Derive key and create cipher
        let key_bytes = self.derive_encryption_key()?;
        let cipher = Aes256Gcm::new_from_slice(&key_bytes)
            .context("Failed to create cipher")?;

        // Decrypt
        let plaintext = cipher
            .decrypt(nonce, ciphertext)
            .map_err(|e| anyhow::anyhow!("Decryption failed: {}", e))?;

        String::from_utf8(plaintext).context("Failed to convert decrypted data to UTF-8")
    }

    fn derive_encryption_key(&self) -> Result<[u8; 32]> {
        use sha2::{Sha256, Digest};

        // Use SHA-256 to derive a 32-byte key from encryption_key
        let mut hasher = Sha256::new();
        hasher.update(self.encryption_key.as_bytes());
        let result = hasher.finalize();

        let mut key = [0u8; 32];
        key.copy_from_slice(&result[..]);

        Ok(key)
    }
}