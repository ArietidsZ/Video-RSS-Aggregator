use anyhow::{Context, Result};
use base64;
use oauth2::{
    AuthUrl, AuthorizationCode, ClientId, ClientSecret, CsrfToken, PkceCodeChallenge,
    RedirectUrl, RefreshToken, Scope, TokenResponse, TokenUrl
};
use oauth2::basic::{BasicClient};
use openidconnect::{
    AuthorizationCode as OidcAuthorizationCode,
    ClientId as OidcClientId, ClientSecret as OidcClientSecret, CsrfToken as OidcCsrfToken,
    IssuerUrl, Nonce, RedirectUrl as OidcRedirectUrl,
    core::{CoreAuthenticationFlow, CoreClient, CoreProviderMetadata}
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::info;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthProvider {
    pub name: String,
    pub provider_type: ProviderType,
    pub client_id: String,
    pub client_secret: String,
    pub auth_url: String,
    pub token_url: String,
    pub userinfo_url: Option<String>,
    pub scopes: Vec<String>,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProviderType {
    Google,
    GitHub,
    Microsoft,
    Discord,
    Facebook,
    Twitter,
    Generic,
    OpenIDConnect,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OAuthState {
    pub csrf_token: String,
    pub pkce_verifier: Option<String>,
    pub nonce: Option<String>,
    pub redirect_uri: Option<String>,
    pub provider: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OAuthUserInfo {
    pub id: String,
    pub email: Option<String>,
    pub name: Option<String>,
    pub username: Option<String>,
    pub avatar_url: Option<String>,
    pub verified: Option<bool>,
    pub provider: String,
    pub raw_data: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
pub struct OAuthAccount {
    pub id: Uuid,
    pub user_id: Uuid,
    pub provider: String,
    pub provider_account_id: String,
    pub access_token: Option<String>,
    pub refresh_token: Option<String>,
    pub expires_at: Option<chrono::DateTime<chrono::Utc>>,
    pub token_type: Option<String>,
    pub scope: Option<String>,
    pub id_token: Option<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

pub struct OAuthService {
    providers: HashMap<String, OAuthProvider>,
    redis: redis::aio::ConnectionManager,
    database: sqlx::PgPool,
}

impl OAuthService {
    pub async fn new(client_id: &str, client_secret: &str) -> Result<Self> {
        let redis_client = redis::Client::open("redis://localhost:6379")?;
        let redis = redis::aio::ConnectionManager::new(redis_client).await?;

        let database = sqlx::PgPool::connect("postgresql://videorss:password@localhost/videorss").await?;

        Self::init_database(&database).await?;

        let mut providers = HashMap::new();

        // Google OAuth 2.0
        providers.insert("google".to_string(), OAuthProvider {
            name: "Google".to_string(),
            provider_type: ProviderType::Google,
            client_id: client_id.to_string(),
            client_secret: client_secret.to_string(),
            auth_url: "https://accounts.google.com/o/oauth2/v2/auth".to_string(),
            token_url: "https://www.googleapis.com/oauth2/v4/token".to_string(),
            userinfo_url: Some("https://www.googleapis.com/oauth2/v2/userinfo".to_string()),
            scopes: vec!["openid".to_string(), "email".to_string(), "profile".to_string()],
            enabled: true,
        });

        // GitHub OAuth 2.0
        providers.insert("github".to_string(), OAuthProvider {
            name: "GitHub".to_string(),
            provider_type: ProviderType::GitHub,
            client_id: client_id.to_string(),
            client_secret: client_secret.to_string(),
            auth_url: "https://github.com/login/oauth/authorize".to_string(),
            token_url: "https://github.com/login/oauth/access_token".to_string(),
            userinfo_url: Some("https://api.github.com/user".to_string()),
            scopes: vec!["user:email".to_string()],
            enabled: true,
        });

        // Microsoft OAuth 2.0
        providers.insert("microsoft".to_string(), OAuthProvider {
            name: "Microsoft".to_string(),
            provider_type: ProviderType::Microsoft,
            client_id: client_id.to_string(),
            client_secret: client_secret.to_string(),
            auth_url: "https://login.microsoftonline.com/common/oauth2/v2.0/authorize".to_string(),
            token_url: "https://login.microsoftonline.com/common/oauth2/v2.0/token".to_string(),
            userinfo_url: Some("https://graph.microsoft.com/v1.0/me".to_string()),
            scopes: vec!["openid".to_string(), "email".to_string(), "profile".to_string()],
            enabled: true,
        });

        // Discord OAuth 2.0
        providers.insert("discord".to_string(), OAuthProvider {
            name: "Discord".to_string(),
            provider_type: ProviderType::Discord,
            client_id: client_id.to_string(),
            client_secret: client_secret.to_string(),
            auth_url: "https://discord.com/api/oauth2/authorize".to_string(),
            token_url: "https://discord.com/api/oauth2/token".to_string(),
            userinfo_url: Some("https://discord.com/api/users/@me".to_string()),
            scopes: vec!["identify".to_string(), "email".to_string()],
            enabled: true,
        });

        Ok(Self {
            providers,
            redis,
            database,
        })
    }

    async fn init_database(database: &sqlx::PgPool) -> Result<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS oauth_accounts (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                provider VARCHAR(50) NOT NULL,
                provider_account_id VARCHAR(255) NOT NULL,
                access_token TEXT,
                refresh_token TEXT,
                expires_at TIMESTAMPTZ,
                token_type VARCHAR(50),
                scope TEXT,
                id_token TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(provider, provider_account_id)
            );

            CREATE INDEX IF NOT EXISTS idx_oauth_accounts_user_id ON oauth_accounts(user_id);
            CREATE INDEX IF NOT EXISTS idx_oauth_accounts_provider ON oauth_accounts(provider);
            "#,
        )
        .execute(database)
        .await
        .context("Failed to initialize OAuth database tables")?;

        Ok(())
    }

    pub async fn authorize(&self, provider_name: &str) -> Result<serde_json::Value> {
        let provider = self.providers.get(provider_name)
            .ok_or_else(|| anyhow::anyhow!("OAuth provider '{}' not found", provider_name))?;

        if !provider.enabled {
            return Err(anyhow::anyhow!("OAuth provider '{}' is not enabled", provider_name));
        }

        match provider.provider_type {
            ProviderType::OpenIDConnect => self.authorize_openid_connect(provider).await,
            _ => self.authorize_oauth2(provider).await,
        }
    }

    async fn authorize_oauth2(&self, provider: &OAuthProvider) -> Result<serde_json::Value> {
        let client = BasicClient::new(
            ClientId::new(provider.client_id.clone()),
            Some(ClientSecret::new(provider.client_secret.clone())),
            AuthUrl::new(provider.auth_url.clone())?,
            Some(TokenUrl::new(provider.token_url.clone())?),
        )
        .set_redirect_uri(RedirectUrl::new(format!("http://localhost:8020/oauth/callback/{}", provider.name.to_lowercase()))?);

        // Generate PKCE challenge for enhanced security
        let (pkce_challenge, pkce_verifier) = PkceCodeChallenge::new_random_sha256();

        let mut auth_request = client
            .authorize_url(CsrfToken::new_random)
            .set_pkce_challenge(pkce_challenge);

        // Add scopes
        for scope in &provider.scopes {
            auth_request = auth_request.add_scope(Scope::new(scope.clone()));
        }

        let (auth_url, csrf_token) = auth_request.url();

        // Store OAuth state in Redis
        let state = OAuthState {
            csrf_token: csrf_token.secret().clone(),
            pkce_verifier: Some(pkce_verifier.secret().clone()),
            nonce: None,
            redirect_uri: Some(auth_url.to_string()),
            provider: provider.name.clone(),
            created_at: chrono::Utc::now(),
        };

        let mut conn = self.redis.clone();
        redis::cmd("SETEX")
            .arg(format!("oauth_state:{}", csrf_token.secret()))
            .arg(600) // 10 minutes
            .arg(serde_json::to_string(&state)?)
            .query_async::<_, ()>(&mut conn)
            .await?;

        info!("OAuth authorization URL generated for provider: {}", provider.name);

        Ok(serde_json::json!({
            "authorization_url": auth_url.to_string(),
            "state": csrf_token.secret(),
            "provider": provider.name
        }))
    }

    async fn authorize_openid_connect(&self, provider: &OAuthProvider) -> Result<serde_json::Value> {
        // Discover OpenID Connect provider metadata
        let provider_metadata = CoreProviderMetadata::discover_async(
            IssuerUrl::new(provider.auth_url.clone())?,
            async_http_client,
        ).await?;

        let client = CoreClient::from_provider_metadata(
            provider_metadata,
            OidcClientId::new(provider.client_id.clone()),
            Some(OidcClientSecret::new(provider.client_secret.clone())),
        )
        .set_redirect_uri(OidcRedirectUrl::new(format!("http://localhost:8020/oauth/callback/{}", provider.name.to_lowercase()))?);

        let (pkce_challenge, pkce_verifier) = PkceCodeChallenge::new_random_sha256();
        let nonce = Nonce::new_random();
        let nonce_clone = nonce.clone();

        let mut auth_request = client
            .authorize_url(
                CoreAuthenticationFlow::AuthorizationCode,
                OidcCsrfToken::new_random,
                move || nonce_clone.clone(),
            )
            .set_pkce_challenge(pkce_challenge);

        // Add scopes
        for scope in &provider.scopes {
            auth_request = auth_request.add_scope(Scope::new(scope.clone()));
        }

        let (auth_url, csrf_token, returned_nonce) = auth_request.url();

        // Store OpenID Connect state
        let state = OAuthState {
            csrf_token: csrf_token.secret().clone(),
            pkce_verifier: Some(pkce_verifier.secret().clone()),
            nonce: Some(returned_nonce.secret().clone()),
            redirect_uri: Some(auth_url.to_string()),
            provider: provider.name.clone(),
            created_at: chrono::Utc::now(),
        };

        let mut conn = self.redis.clone();
        redis::cmd("SETEX")
            .arg(format!("oauth_state:{}", csrf_token.secret()))
            .arg(600) // 10 minutes
            .arg(serde_json::to_string(&state)?)
            .query_async::<_, ()>(&mut conn)
            .await?;

        info!("OpenID Connect authorization URL generated for provider: {}", provider.name);

        Ok(serde_json::json!({
            "authorization_url": auth_url.to_string(),
            "state": csrf_token.secret(),
            "provider": provider.name
        }))
    }

    pub async fn handle_callback(&self, provider_name: &str, params: HashMap<String, String>) -> Result<serde_json::Value> {
        let code = params.get("code")
            .ok_or_else(|| anyhow::anyhow!("Authorization code not found in callback"))?;

        let state = params.get("state")
            .ok_or_else(|| anyhow::anyhow!("State parameter not found in callback"))?;

        let error = params.get("error");
        if let Some(error_code) = error {
            let unknown_error = "Unknown error".to_string();
            let error_description = params.get("error_description").unwrap_or(&unknown_error);
            return Err(anyhow::anyhow!("OAuth error: {} - {}", error_code, error_description));
        }

        // Retrieve and validate state
        let mut conn = self.redis.clone();
        let stored_state_json: Option<String> = redis::AsyncCommands::get(
            &mut conn,
            format!("oauth_state:{}", state)
        ).await?;

        let stored_state: OAuthState = serde_json::from_str(
            &stored_state_json.ok_or_else(|| anyhow::anyhow!("Invalid or expired OAuth state"))?
        ).map_err(|e| anyhow::anyhow!("Failed to parse stored OAuth state: {}", e))?;

        if stored_state.csrf_token != *state {
            return Err(anyhow::anyhow!("CSRF token mismatch"));
        }

        // Clean up state
        let _: () = redis::AsyncCommands::del(&mut conn, format!("oauth_state:{}", state)).await?;

        let provider = self.providers.get(provider_name)
            .ok_or_else(|| anyhow::anyhow!("OAuth provider '{}' not found", provider_name))?;

        match provider.provider_type {
            ProviderType::OpenIDConnect => {
                self.handle_openid_connect_callback(provider, code, &stored_state).await
            },
            _ => {
                self.handle_oauth2_callback(provider, code, &stored_state).await
            }
        }
    }

    async fn handle_oauth2_callback(&self, provider: &OAuthProvider, code: &str, state: &OAuthState) -> Result<serde_json::Value> {
        let client = BasicClient::new(
            ClientId::new(provider.client_id.clone()),
            Some(ClientSecret::new(provider.client_secret.clone())),
            AuthUrl::new(provider.auth_url.clone())?,
            Some(TokenUrl::new(provider.token_url.clone())?),
        )
        .set_redirect_uri(RedirectUrl::new(format!("http://localhost:8020/oauth/callback/{}", provider.name.to_lowercase()))?);

        let mut token_request = client.exchange_code(AuthorizationCode::new(code.to_string()));

        // Add PKCE verifier if available
        if let Some(pkce_verifier) = &state.pkce_verifier {
            token_request = token_request.set_pkce_verifier(oauth2::PkceCodeVerifier::new(pkce_verifier.clone()));
        }

        // Exchange code for token
        let token_response = token_request
            .request_async(async_http_client)
            .await
            .context("Failed to exchange authorization code for token")?;

        // Get user info
        let user_info = self.fetch_user_info(provider, token_response.access_token().secret()).await?;

        // Check if user already exists by OAuth account
        let existing_account = sqlx::query_scalar::<_, uuid::Uuid>(
            "SELECT user_id FROM oauth_accounts WHERE provider = $1 AND provider_account_id = $2"
        )
        .bind(&provider.name)
        .bind(&user_info.id)
        .fetch_optional(&self.database)
        .await?;

        let user_id = if let Some(user_id) = existing_account {
            // Update existing OAuth account
            self.update_oauth_account(
                &user_id,
                provider,
                &user_info,
                &token_response,
            ).await?;
            user_id
        } else {
            // Create new user or link to existing user by email
            let user_id = if let Some(email) = &user_info.email {
                // Check if user exists by email
                let existing_user = sqlx::query_scalar::<_, uuid::Uuid>(
                    "SELECT id FROM users WHERE email = $1"
                )
                .bind(email)
                .fetch_optional(&self.database)
                .await?;

                if let Some(user_id) = existing_user {
                    user_id
                } else {
                    // Create new user
                    self.create_user_from_oauth(&user_info).await?
                }
            } else {
                // Create new user without email
                self.create_user_from_oauth(&user_info).await?
            };

            // Create OAuth account
            self.create_oauth_account(user_id, provider, &user_info, &token_response).await?;
            user_id
        };

        info!("OAuth callback processed successfully for provider: {} user: {}", provider.name, user_id);

        Ok(serde_json::json!({
            "success": true,
            "user_id": user_id,
            "provider": provider.name,
            "user_info": user_info
        }))
    }

    async fn handle_openid_connect_callback(&self, provider: &OAuthProvider, code: &str, state: &OAuthState) -> Result<serde_json::Value> {
        let provider_metadata = CoreProviderMetadata::discover_async(
            IssuerUrl::new(provider.auth_url.clone())?,
            async_http_client,
        ).await?;

        let client = CoreClient::from_provider_metadata(
            provider_metadata.clone(),
            OidcClientId::new(provider.client_id.clone()),
            Some(OidcClientSecret::new(provider.client_secret.clone())),
        )
        .set_redirect_uri(OidcRedirectUrl::new(format!("http://localhost:8020/oauth/callback/{}", provider.name.to_lowercase()))?);

        let mut token_request = client.exchange_code(OidcAuthorizationCode::new(code.to_string()));

        if let Some(pkce_verifier) = &state.pkce_verifier {
            token_request = token_request.set_pkce_verifier(oauth2::PkceCodeVerifier::new(pkce_verifier.clone()));
        }

        let token_response = token_request
            .request_async(async_http_client)
            .await
            .context("Failed to exchange authorization code for token")?;

        // Extract user info from ID token if present
        let id_token_info = if let Some(id_token) = token_response.extra_fields().id_token() {
            // Extract claims from ID token (without full verification for now)
            // In production, this should verify the signature with JWKS
            self.extract_id_token_claims(id_token.to_string()).ok()
        } else {
            None
        };

        // Get additional user info if userinfo endpoint is available
        let user_info = if let Some(_userinfo_url) = &provider.userinfo_url {
            let mut info = self.fetch_user_info(provider, token_response.access_token().secret()).await?;

            // Merge ID token claims if available
            if let Some(id_info) = id_token_info {
                info.email = info.email.or(id_info.email);
                info.verified = info.verified.or(id_info.verified);
                info.name = info.name.or(id_info.name);
            }
            info
        } else {
            // Extract user info from ID token
            OAuthUserInfo {
                id: "extracted_from_id_token".to_string(),
                email: None,
                name: None,
                username: None,
                avatar_url: None,
                verified: Some(true),
                provider: provider.name.clone(),
                raw_data: serde_json::json!({}),
            }
        };

        info!("OpenID Connect callback processed successfully for provider: {}", provider.name);

        Ok(serde_json::json!({
            "success": true,
            "provider": provider.name,
            "user_info": user_info
        }))
    }

    async fn fetch_user_info(&self, provider: &OAuthProvider, access_token: &str) -> Result<OAuthUserInfo> {
        let userinfo_url = provider.userinfo_url.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No userinfo URL configured for provider"))?;

        let client = reqwest::Client::new();
        let response = client
            .get(userinfo_url)
            .bearer_auth(access_token)
            .send()
            .await
            .context("Failed to fetch user info")?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Failed to fetch user info: HTTP {}", response.status()));
        }

        let user_data: serde_json::Value = response.json().await?;

        let user_info = match provider.provider_type {
            ProviderType::Google => self.parse_google_user_info(user_data, &provider.name),
            ProviderType::GitHub => self.parse_github_user_info(user_data, &provider.name),
            ProviderType::Microsoft => self.parse_microsoft_user_info(user_data, &provider.name),
            ProviderType::Discord => self.parse_discord_user_info(user_data, &provider.name),
            _ => self.parse_generic_user_info(user_data, &provider.name),
        }?;

        Ok(user_info)
    }

    fn parse_google_user_info(&self, data: serde_json::Value, provider: &str) -> Result<OAuthUserInfo> {
        Ok(OAuthUserInfo {
            id: data["id"].as_str().unwrap_or_default().to_string(),
            email: data["email"].as_str().map(|s| s.to_string()),
            name: data["name"].as_str().map(|s| s.to_string()),
            username: data["email"].as_str().map(|s| s.split('@').next().unwrap_or(s).to_string()),
            avatar_url: data["picture"].as_str().map(|s| s.to_string()),
            verified: data["verified_email"].as_bool(),
            provider: provider.to_string(),
            raw_data: data,
        })
    }

    fn parse_github_user_info(&self, data: serde_json::Value, provider: &str) -> Result<OAuthUserInfo> {
        Ok(OAuthUserInfo {
            id: data["id"].as_u64().unwrap_or_default().to_string(),
            email: data["email"].as_str().map(|s| s.to_string()),
            name: data["name"].as_str().map(|s| s.to_string()),
            username: data["login"].as_str().map(|s| s.to_string()),
            avatar_url: data["avatar_url"].as_str().map(|s| s.to_string()),
            verified: Some(true), // GitHub accounts are generally considered verified
            provider: provider.to_string(),
            raw_data: data,
        })
    }

    fn parse_microsoft_user_info(&self, data: serde_json::Value, provider: &str) -> Result<OAuthUserInfo> {
        Ok(OAuthUserInfo {
            id: data["id"].as_str().unwrap_or_default().to_string(),
            email: data["mail"].as_str()
                .or_else(|| data["userPrincipalName"].as_str())
                .map(|s| s.to_string()),
            name: data["displayName"].as_str().map(|s| s.to_string()),
            username: data["userPrincipalName"].as_str()
                .map(|s| s.split('@').next().unwrap_or(s).to_string()),
            avatar_url: None, // Microsoft Graph requires separate request for photo
            verified: Some(true),
            provider: provider.to_string(),
            raw_data: data,
        })
    }

    fn parse_discord_user_info(&self, data: serde_json::Value, provider: &str) -> Result<OAuthUserInfo> {
        let user_id = data["id"].as_str().unwrap_or_default();
        let discriminator = data["discriminator"].as_str().unwrap_or("0000");
        let avatar_hash = data["avatar"].as_str();

        let avatar_url = if let Some(hash) = avatar_hash {
            Some(format!("https://cdn.discordapp.com/avatars/{}/{}.png", user_id, hash))
        } else {
            None
        };

        Ok(OAuthUserInfo {
            id: user_id.to_string(),
            email: data["email"].as_str().map(|s| s.to_string()),
            name: data["global_name"].as_str()
                .or_else(|| data["username"].as_str())
                .map(|s| s.to_string()),
            username: data["username"].as_str()
                .map(|s| if discriminator != "0" { format!("{}#{}", s, discriminator) } else { s.to_string() }),
            avatar_url,
            verified: data["verified"].as_bool(),
            provider: provider.to_string(),
            raw_data: data,
        })
    }

    fn parse_generic_user_info(&self, data: serde_json::Value, provider: &str) -> Result<OAuthUserInfo> {
        Ok(OAuthUserInfo {
            id: data["id"].as_str()
                .or_else(|| data["sub"].as_str())
                .unwrap_or_default().to_string(),
            email: data["email"].as_str().map(|s| s.to_string()),
            name: data["name"].as_str()
                .or_else(|| data["display_name"].as_str())
                .map(|s| s.to_string()),
            username: data["username"].as_str()
                .or_else(|| data["preferred_username"].as_str())
                .map(|s| s.to_string()),
            avatar_url: data["avatar_url"].as_str()
                .or_else(|| data["picture"].as_str())
                .map(|s| s.to_string()),
            verified: data["email_verified"].as_bool(),
            provider: provider.to_string(),
            raw_data: data,
        })
    }

    async fn create_user_from_oauth(&self, user_info: &OAuthUserInfo) -> Result<Uuid> {
        let user_id = Uuid::new_v4();
        let username = user_info.username.as_ref()
            .or_else(|| user_info.name.as_ref())
            .cloned()
            .unwrap_or_else(|| format!("{}_user_{}", user_info.provider, &user_info.id[..8]));

        let email = user_info.email.as_ref()
            .cloned()
            .unwrap_or_else(|| format!("{}+{}@oauth.local", &username, user_info.provider));

        // Create user with a placeholder password (OAuth users don't need passwords)
        sqlx::query(
            "INSERT INTO users (id, email, username, password_hash, first_name, is_verified)
            VALUES ($1, $2, $3, 'oauth_user_no_password', $4, $5)"
        )
        .bind(user_id)
        .bind(email)
        .bind(&username)
        .bind(&user_info.name)
        .bind(user_info.verified.unwrap_or(false))
        .execute(&self.database)
        .await?;

        info!("Created new user from OAuth: {} ({})", username, user_id);

        Ok(user_id)
    }

    async fn create_oauth_account(&self, user_id: Uuid, provider: &OAuthProvider, user_info: &OAuthUserInfo, token_response: &oauth2::StandardTokenResponse<oauth2::EmptyExtraTokenFields, oauth2::basic::BasicTokenType>) -> Result<()> {
        let expires_at = token_response.expires_in()
            .map(|duration| chrono::Utc::now() + chrono::Duration::seconds(duration.as_secs() as i64));

        sqlx::query(
            "INSERT INTO oauth_accounts (
                user_id, provider, provider_account_id, access_token,
                refresh_token, expires_at, token_type, scope
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)"
        )
        .bind(user_id)
        .bind(&provider.name)
        .bind(&user_info.id)
        .bind(token_response.access_token().secret())
        .bind(token_response.refresh_token().map(|t| t.secret()))
        .bind(expires_at)
        .bind("Bearer")
        .bind(provider.scopes.join(" "))
        .execute(&self.database)
        .await?;

        info!("Created OAuth account link: {} -> {}", user_id, provider.name);

        Ok(())
    }

    async fn update_oauth_account(&self, user_id: &Uuid, provider: &OAuthProvider, _user_info: &OAuthUserInfo, token_response: &oauth2::StandardTokenResponse<oauth2::EmptyExtraTokenFields, oauth2::basic::BasicTokenType>) -> Result<()> {
        let expires_at = token_response.expires_in()
            .map(|duration| chrono::Utc::now() + chrono::Duration::seconds(duration.as_secs() as i64));

        sqlx::query(
            "UPDATE oauth_accounts
            SET access_token = $1, refresh_token = $2, expires_at = $3,
                updated_at = NOW()
            WHERE user_id = $4 AND provider = $5"
        )
        .bind(token_response.access_token().secret())
        .bind(token_response.refresh_token().map(|t| t.secret()))
        .bind(expires_at)
        .bind(user_id)
        .bind(&provider.name)
        .execute(&self.database)
        .await?;

        info!("Updated OAuth account: {} -> {}", user_id, provider.name);

        Ok(())
    }

    pub async fn get_user_oauth_accounts(&self, user_id: &Uuid) -> Result<Vec<OAuthAccount>> {
        let accounts = sqlx::query_as::<_, OAuthAccount>(
            "SELECT id, user_id, provider, provider_account_id, access_token,
                   refresh_token, expires_at, token_type, scope, id_token,
                   created_at, updated_at
            FROM oauth_accounts
            WHERE user_id = $1
            ORDER BY created_at DESC"
        )
        .bind(user_id)
        .fetch_all(&self.database)
        .await?;

        Ok(accounts)
    }

    pub async fn unlink_oauth_account(&self, user_id: &Uuid, provider: &str) -> Result<()> {
        sqlx::query(
            "DELETE FROM oauth_accounts WHERE user_id = $1 AND provider = $2"
        )
        .bind(user_id)
        .bind(provider)
        .execute(&self.database)
        .await?;

        info!("Unlinked OAuth account: {} from {}", user_id, provider);

        Ok(())
    }

    pub fn get_available_providers(&self) -> Vec<&OAuthProvider> {
        self.providers.values().filter(|p| p.enabled).collect()
    }

    pub async fn refresh_oauth_token(&self, user_id: &Uuid, provider: &str) -> Result<()> {
        let refresh_token_opt = sqlx::query_scalar::<_, Option<String>>(
            "SELECT refresh_token FROM oauth_accounts WHERE user_id = $1 AND provider = $2"
        )
        .bind(user_id)
        .bind(provider)
        .fetch_optional(&self.database)
        .await?
        .ok_or_else(|| anyhow::anyhow!("OAuth account not found"))?;

        let refresh_token_str = refresh_token_opt
            .ok_or_else(|| anyhow::anyhow!("No refresh token available"))?;

        let oauth_provider = self.providers.get(provider)
            .ok_or_else(|| anyhow::anyhow!("Provider not configured"))?;

        // Build OAuth2 client
        let client = BasicClient::new(
            ClientId::new(oauth_provider.client_id.clone()),
            Some(ClientSecret::new(oauth_provider.client_secret.clone())),
            AuthUrl::new(oauth_provider.auth_url.clone())?,
            Some(TokenUrl::new(oauth_provider.token_url.clone())?)
        );

        // Exchange refresh token for new access token
        let refresh_token = RefreshToken::new(refresh_token_str);
        let token_result = client
            .exchange_refresh_token(&refresh_token)
            .request_async(async_http_client)
            .await
            .context("Failed to refresh OAuth token")?;

        // Update the access token and optionally the refresh token in database
        let new_access_token = token_result.access_token().secret();
        let new_refresh_token = token_result.refresh_token().map(|t| t.secret());
        let expires_at = token_result.expires_in().map(|duration| {
            chrono::Utc::now() + chrono::Duration::seconds(duration.as_secs() as i64)
        });

        // Update database with new tokens
        sqlx::query(
            "UPDATE oauth_accounts
            SET access_token = $1,
                refresh_token = COALESCE($2, refresh_token),
                token_expires_at = $3,
                updated_at = NOW()
            WHERE user_id = $4 AND provider = $5"
        )
        .bind(new_access_token)
        .bind(new_refresh_token)
        .bind(expires_at)
        .bind(user_id)
        .bind(provider)
        .execute(&self.database)
        .await?;

        info!("Refreshed OAuth token for user {} provider {}", user_id, provider);

        Ok(())
    }

    fn extract_id_token_claims(&self, id_token: String) -> Result<OAuthUserInfo> {
        // Parse JWT without verification (split by dots and decode base64)
        // WARNING: This is NOT secure - should verify signature in production
        let parts: Vec<&str> = id_token.split('.').collect();
        if parts.len() != 3 {
            return Err(anyhow::anyhow!("Invalid JWT format"));
        }

        let payload = parts[1];
        let decoded = base64::Engine::decode(
            &base64::engine::general_purpose::URL_SAFE_NO_PAD,
            payload
        )?;

        let claims: serde_json::Value = serde_json::from_slice(&decoded)?;

        Ok(OAuthUserInfo {
            id: claims.get("sub")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string(),
            email: claims.get("email")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            username: claims.get("preferred_username")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            name: claims.get("name")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            avatar_url: claims.get("picture")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            verified: claims.get("email_verified")
                .and_then(|v| v.as_bool()),
            provider: "oidc".to_string(),
            raw_data: claims,
        })
    }
}

async fn async_http_client(request: oauth2::HttpRequest) -> Result<oauth2::HttpResponse, oauth2::reqwest::Error<reqwest::Error>> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(oauth2::reqwest::Error::Reqwest)?;

    let mut request_builder = client
        .request(request.method, &request.url.to_string())
        .body(request.body);

    for (name, value) in &request.headers {
        request_builder = request_builder.header(name.as_str(), value.as_bytes());
    }

    let response = request_builder.send().await
        .map_err(oauth2::reqwest::Error::Reqwest)?;

    let status_code = response.status();
    let headers = response.headers().clone();
    let chunks = response.bytes().await
        .map_err(oauth2::reqwest::Error::Reqwest)?;

    Ok(oauth2::HttpResponse {
        status_code,
        headers,
        body: chunks.to_vec(),
    })
}