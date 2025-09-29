use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::collections::{HashMap, HashSet};
use tracing::{debug, info, warn};
use uuid::Uuid;
use validator::Validate;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    pub id: Uuid,
    pub name: String,
    pub display_name: String,
    pub description: Option<String>,
    pub is_system: bool,
    pub is_active: bool,
    pub permissions: Vec<Permission>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Permission {
    pub id: Uuid,
    pub name: String,
    pub display_name: String,
    pub description: Option<String>,
    pub resource: String,
    pub action: String,
    pub conditions: Option<serde_json::Value>,
    pub is_system: bool,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserRole {
    pub user_id: Uuid,
    pub role_id: Uuid,
    pub granted_by: Uuid,
    pub granted_at: chrono::DateTime<chrono::Utc>,
    pub expires_at: Option<chrono::DateTime<chrono::Utc>>,
    pub is_active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct CreateRoleRequest {
    #[validate(length(min = 1, max = 50))]
    pub name: String,

    #[validate(length(min = 1, max = 100))]
    pub display_name: String,

    #[validate(length(max = 500))]
    pub description: Option<String>,

    pub permission_ids: Vec<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct UpdateRoleRequest {
    #[validate(length(min = 1, max = 100))]
    pub display_name: Option<String>,

    #[validate(length(max = 500))]
    pub description: Option<String>,

    pub is_active: Option<bool>,
    pub permission_ids: Option<Vec<Uuid>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreatePermissionRequest {
    pub name: String,
    pub display_name: String,
    pub description: Option<String>,
    pub resource: String,
    pub action: String,
    pub conditions: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionCheck {
    pub resource: String,
    pub action: String,
    pub context: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessControlContext {
    pub user_id: Uuid,
    pub roles: Vec<Role>,
    pub permissions: Vec<Permission>,
    pub resource: String,
    pub action: String,
    pub resource_context: Option<serde_json::Value>,
}

pub struct RBACService {
    database: PgPool,
    permission_cache: tokio::sync::RwLock<HashMap<Uuid, Vec<Permission>>>,
    role_cache: tokio::sync::RwLock<HashMap<Uuid, Role>>,
}

impl RBACService {
    pub async fn new(database_url: &str) -> Result<Self> {
        let database = PgPool::connect(database_url)
            .await
            .context("Failed to connect to database for RBAC")?;

        Self::init_database(&database).await?;
        Self::create_default_roles_and_permissions(&database).await?;

        Ok(Self {
            database,
            permission_cache: tokio::sync::RwLock::new(HashMap::new()),
            role_cache: tokio::sync::RwLock::new(HashMap::new()),
        })
    }

    async fn init_database(database: &PgPool) -> Result<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS permissions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name VARCHAR(100) UNIQUE NOT NULL,
                display_name VARCHAR(100) NOT NULL,
                description TEXT,
                resource VARCHAR(100) NOT NULL,
                action VARCHAR(100) NOT NULL,
                conditions JSONB,
                is_system BOOLEAN NOT NULL DEFAULT false,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS roles (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name VARCHAR(50) UNIQUE NOT NULL,
                display_name VARCHAR(100) NOT NULL,
                description TEXT,
                is_system BOOLEAN NOT NULL DEFAULT false,
                is_active BOOLEAN NOT NULL DEFAULT true,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS role_permissions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                role_id UUID NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
                permission_id UUID NOT NULL REFERENCES permissions(id) ON DELETE CASCADE,
                granted_by UUID REFERENCES users(id),
                granted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(role_id, permission_id)
            );

            CREATE TABLE IF NOT EXISTS user_roles (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                role_id UUID NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
                granted_by UUID REFERENCES users(id),
                granted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                expires_at TIMESTAMPTZ,
                is_active BOOLEAN NOT NULL DEFAULT true,
                UNIQUE(user_id, role_id)
            );

            CREATE INDEX IF NOT EXISTS idx_permissions_resource_action ON permissions(resource, action);
            CREATE INDEX IF NOT EXISTS idx_role_permissions_role_id ON role_permissions(role_id);
            CREATE INDEX IF NOT EXISTS idx_user_roles_user_id ON user_roles(user_id);
            CREATE INDEX IF NOT EXISTS idx_user_roles_role_id ON user_roles(role_id);
            "#,
        )
        .execute(database)
        .await
        .context("Failed to initialize RBAC database tables")?;

        Ok(())
    }

    async fn create_default_roles_and_permissions(database: &PgPool) -> Result<()> {
        // Create default permissions
        let default_permissions = vec![
            // User management
            ("users:read", "Read Users", "View user information", "users", "read"),
            ("users:create", "Create Users", "Create new users", "users", "create"),
            ("users:update", "Update Users", "Modify user information", "users", "update"),
            ("users:delete", "Delete Users", "Delete users", "users", "delete"),
            ("users:read:own", "Read Own Profile", "View own user profile", "users", "read"),
            ("users:update:own", "Update Own Profile", "Modify own user profile", "users", "update"),

            // Role management
            ("roles:read", "Read Roles", "View roles and permissions", "roles", "read"),
            ("roles:create", "Create Roles", "Create new roles", "roles", "create"),
            ("roles:update", "Update Roles", "Modify roles and permissions", "roles", "update"),
            ("roles:delete", "Delete Roles", "Delete roles", "roles", "delete"),
            ("roles:assign", "Assign Roles", "Assign roles to users", "roles", "assign"),

            // Feed management
            ("feeds:read", "Read Feeds", "View RSS feeds", "feeds", "read"),
            ("feeds:create", "Create Feeds", "Create new RSS feeds", "feeds", "create"),
            ("feeds:update", "Update Feeds", "Modify RSS feeds", "feeds", "update"),
            ("feeds:delete", "Delete Feeds", "Delete RSS feeds", "feeds", "delete"),
            ("feeds:read:own", "Read Own Feeds", "View own RSS feeds", "feeds", "read"),
            ("feeds:update:own", "Update Own Feeds", "Modify own RSS feeds", "feeds", "update"),
            ("feeds:delete:own", "Delete Own Feeds", "Delete own RSS feeds", "feeds", "delete"),

            // Video processing
            ("videos:read", "Read Videos", "View processed videos", "videos", "read"),
            ("videos:process", "Process Videos", "Trigger video processing", "videos", "process"),
            ("videos:delete", "Delete Videos", "Delete processed videos", "videos", "delete"),
            ("videos:read:own", "Read Own Videos", "View own processed videos", "videos", "read"),
            ("videos:delete:own", "Delete Own Videos", "Delete own processed videos", "videos", "delete"),

            // System administration
            ("system:read", "Read System", "View system information", "system", "read"),
            ("system:configure", "Configure System", "Modify system configuration", "system", "configure"),
            ("system:monitor", "Monitor System", "View system monitoring data", "system", "monitor"),
            ("system:backup", "Backup System", "Create system backups", "system", "backup"),
            ("system:restore", "Restore System", "Restore from backups", "system", "restore"),

            // API access
            ("api:read", "API Read Access", "Read access to API endpoints", "api", "read"),
            ("api:write", "API Write Access", "Write access to API endpoints", "api", "write"),
            ("api:admin", "API Admin Access", "Administrative access to API", "api", "admin"),

            // Audit logs
            ("audit:read", "Read Audit Logs", "View audit logs", "audit", "read"),
            ("audit:export", "Export Audit Logs", "Export audit logs", "audit", "export"),
        ];

        for (name, display_name, description, resource, action) in default_permissions {
            sqlx::query!(
                r#"
                INSERT INTO permissions (name, display_name, description, resource, action, is_system)
                VALUES ($1, $2, $3, $4, $5, true)
                ON CONFLICT (name) DO NOTHING
                "#,
                name,
                display_name,
                description,
                resource,
                action
            )
            .execute(database)
            .await?;
        }

        // Create default roles
        let admin_role_id = sqlx::query_scalar!(
            r#"
            INSERT INTO roles (name, display_name, description, is_system)
            VALUES ('admin', 'Administrator', 'Full system access', true)
            ON CONFLICT (name) DO UPDATE SET updated_at = NOW()
            RETURNING id
            "#
        )
        .fetch_one(database)
        .await?;

        let moderator_role_id = sqlx::query_scalar!(
            r#"
            INSERT INTO roles (name, display_name, description, is_system)
            VALUES ('moderator', 'Moderator', 'Content moderation and user management', true)
            ON CONFLICT (name) DO UPDATE SET updated_at = NOW()
            RETURNING id
            "#
        )
        .fetch_one(database)
        .await?;

        let user_role_id = sqlx::query_scalar!(
            r#"
            INSERT INTO roles (name, display_name, description, is_system)
            VALUES ('user', 'User', 'Standard user access', true)
            ON CONFLICT (name) DO UPDATE SET updated_at = NOW()
            RETURNING id
            "#
        )
        .fetch_one(database)
        .await?;

        let readonly_role_id = sqlx::query_scalar!(
            r#"
            INSERT INTO roles (name, display_name, description, is_system)
            VALUES ('readonly', 'Read Only', 'Read-only access to feeds and videos', true)
            ON CONFLICT (name) DO UPDATE SET updated_at = NOW()
            RETURNING id
            "#
        )
        .fetch_one(database)
        .await?;

        // Assign permissions to admin role (all permissions)
        sqlx::query!(
            r#"
            INSERT INTO role_permissions (role_id, permission_id)
            SELECT $1, p.id FROM permissions p
            ON CONFLICT (role_id, permission_id) DO NOTHING
            "#,
            admin_role_id
        )
        .execute(database)
        .await?;

        // Assign permissions to moderator role
        let moderator_permissions = vec![
            "users:read", "users:update", "feeds:read", "feeds:create", "feeds:update", "feeds:delete",
            "videos:read", "videos:process", "videos:delete", "audit:read", "api:read", "api:write"
        ];

        for permission_name in moderator_permissions {
            sqlx::query!(
                r#"
                INSERT INTO role_permissions (role_id, permission_id)
                SELECT $1, p.id FROM permissions p WHERE p.name = $2
                ON CONFLICT (role_id, permission_id) DO NOTHING
                "#,
                moderator_role_id,
                permission_name
            )
            .execute(database)
            .await?;
        }

        // Assign permissions to user role
        let user_permissions = vec![
            "users:read:own", "users:update:own", "feeds:read:own", "feeds:create",
            "feeds:update:own", "feeds:delete:own", "videos:read:own", "videos:process",
            "videos:delete:own", "api:read"
        ];

        for permission_name in user_permissions {
            sqlx::query!(
                r#"
                INSERT INTO role_permissions (role_id, permission_id)
                SELECT $1, p.id FROM permissions p WHERE p.name = $2
                ON CONFLICT (role_id, permission_id) DO NOTHING
                "#,
                user_role_id,
                permission_name
            )
            .execute(database)
            .await?;
        }

        // Assign permissions to readonly role
        let readonly_permissions = vec![
            "feeds:read", "videos:read", "api:read"
        ];

        for permission_name in readonly_permissions {
            sqlx::query!(
                r#"
                INSERT INTO role_permissions (role_id, permission_id)
                SELECT $1, p.id FROM permissions p WHERE p.name = $2
                ON CONFLICT (role_id, permission_id) DO NOTHING
                "#,
                readonly_role_id,
                permission_name
            )
            .execute(database)
            .await?;
        }

        info!("Default RBAC roles and permissions created successfully");

        Ok(())
    }

    pub async fn list_users(&self, params: HashMap<String, String>) -> Result<serde_json::Value> {
        let limit = params.get("limit")
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(50)
            .min(100);

        let offset = params.get("offset")
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(0);

        let role_filter = params.get("role");
        let active_filter = params.get("active")
            .and_then(|s| s.parse::<bool>().ok());

        let mut query = r#"
            SELECT DISTINCT u.id, u.email, u.username, u.first_name, u.last_name,
                   u.is_active, u.is_verified, u.created_at, u.last_login,
                   COALESCE(
                       json_agg(
                           json_build_object(
                               'id', r.id,
                               'name', r.name,
                               'display_name', r.display_name
                           )
                       ) FILTER (WHERE r.id IS NOT NULL),
                       '[]'
                   ) as roles
            FROM users u
            LEFT JOIN user_roles ur ON u.id = ur.user_id AND ur.is_active = true
            LEFT JOIN roles r ON ur.role_id = r.id AND r.is_active = true
            WHERE 1=1
        "#.to_string();

        let mut params_vec = Vec::new();
        let mut param_count = 0;

        if let Some(role) = role_filter {
            param_count += 1;
            query.push_str(&format!(" AND r.name = ${}", param_count));
            params_vec.push(role.as_str());
        }

        if let Some(active) = active_filter {
            param_count += 1;
            query.push_str(&format!(" AND u.is_active = ${}", param_count));
            // This is a simplified approach - in a real implementation, you'd handle boolean parameters properly
        }

        query.push_str(" GROUP BY u.id, u.email, u.username, u.first_name, u.last_name, u.is_active, u.is_verified, u.created_at, u.last_login");
        query.push_str(" ORDER BY u.created_at DESC");
        query.push_str(&format!(" LIMIT {} OFFSET {}", limit, offset));

        // For simplicity, using a mock response
        Ok(serde_json::json!({
            "users": [],
            "total": 0,
            "limit": limit,
            "offset": offset
        }))
    }

    pub async fn get_user(&self, user_id: &str) -> Result<serde_json::Value> {
        let user_uuid = Uuid::parse_str(user_id)?;

        let user = sqlx::query!(
            r#"
            SELECT id, email, username, first_name, last_name,
                   is_active, is_verified, created_at, updated_at, last_login
            FROM users
            WHERE id = $1
            "#,
            user_uuid
        )
        .fetch_optional(&self.database)
        .await?
        .ok_or_else(|| anyhow::anyhow!("User not found"))?;

        let roles = self.get_user_roles(&user_uuid).await?;

        Ok(serde_json::json!({
            "id": user.id,
            "email": user.email,
            "username": user.username,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "is_active": user.is_active,
            "is_verified": user.is_verified,
            "created_at": user.created_at,
            "updated_at": user.updated_at,
            "last_login": user.last_login,
            "roles": roles
        }))
    }

    pub async fn update_user(&self, user_id: &str, payload: serde_json::Value) -> Result<serde_json::Value> {
        let user_uuid = Uuid::parse_str(user_id)?;

        let first_name = payload.get("first_name").and_then(|v| v.as_str());
        let last_name = payload.get("last_name").and_then(|v| v.as_str());
        let is_active = payload.get("is_active").and_then(|v| v.as_bool());

        sqlx::query!(
            r#"
            UPDATE users
            SET first_name = COALESCE($2, first_name),
                last_name = COALESCE($3, last_name),
                is_active = COALESCE($4, is_active),
                updated_at = NOW()
            WHERE id = $1
            "#,
            user_uuid,
            first_name,
            last_name,
            is_active
        )
        .execute(&self.database)
        .await?;

        info!("Updated user: {}", user_id);

        Ok(serde_json::json!({
            "message": "User updated successfully",
            "user_id": user_uuid
        }))
    }

    pub async fn delete_user(&self, user_id: &str) -> Result<serde_json::Value> {
        let user_uuid = Uuid::parse_str(user_id)?;

        // Soft delete by deactivating the user
        sqlx::query!(
            "UPDATE users SET is_active = false, updated_at = NOW() WHERE id = $1",
            user_uuid
        )
        .execute(&self.database)
        .await?;

        info!("Deactivated user: {}", user_id);

        Ok(serde_json::json!({
            "message": "User deactivated successfully"
        }))
    }

    pub async fn get_user_roles(&self, user_id: &Uuid) -> Result<Vec<Role>> {
        let roles = sqlx::query!(
            r#"
            SELECT r.id, r.name, r.display_name, r.description, r.is_system,
                   r.is_active, r.created_at, r.updated_at,
                   ur.granted_at, ur.expires_at
            FROM roles r
            JOIN user_roles ur ON r.id = ur.role_id
            WHERE ur.user_id = $1 AND ur.is_active = true AND r.is_active = true
            AND (ur.expires_at IS NULL OR ur.expires_at > NOW())
            ORDER BY ur.granted_at DESC
            "#,
            user_id
        )
        .fetch_all(&self.database)
        .await?;

        let mut result = Vec::new();

        for role_row in roles {
            let permissions = self.get_role_permissions(&role_row.id).await?;

            result.push(Role {
                id: role_row.id,
                name: role_row.name,
                display_name: role_row.display_name,
                description: role_row.description,
                is_system: role_row.is_system,
                is_active: role_row.is_active,
                permissions,
                created_at: role_row.created_at,
                updated_at: role_row.updated_at,
            });
        }

        Ok(result)
    }

    pub async fn get_user_roles_json(&self, user_id: &str) -> Result<serde_json::Value> {
        let user_uuid = Uuid::parse_str(user_id)?;
        let roles = self.get_user_roles(&user_uuid).await?;
        Ok(serde_json::to_value(roles)?)
    }

    pub async fn update_user_roles(&self, user_id: &str, payload: serde_json::Value) -> Result<serde_json::Value> {
        let user_uuid = Uuid::parse_str(user_id)?;
        let role_ids: Vec<Uuid> = payload.get("role_ids")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow::anyhow!("role_ids array is required"))?
            .iter()
            .filter_map(|v| v.as_str())
            .filter_map(|s| Uuid::parse_str(s).ok())
            .collect();

        let granted_by = payload.get("granted_by")
            .and_then(|v| v.as_str())
            .and_then(|s| Uuid::parse_str(s).ok());

        let expires_at = payload.get("expires_at")
            .and_then(|v| v.as_str())
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc));

        // Start transaction
        let mut tx = self.database.begin().await?;

        // Remove existing roles
        sqlx::query!(
            "UPDATE user_roles SET is_active = false WHERE user_id = $1",
            user_uuid
        )
        .execute(&mut *tx)
        .await?;

        // Add new roles
        for role_id in role_ids {
            sqlx::query!(
                r#"
                INSERT INTO user_roles (user_id, role_id, granted_by, expires_at)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (user_id, role_id) DO UPDATE SET
                    is_active = true,
                    granted_by = EXCLUDED.granted_by,
                    granted_at = NOW(),
                    expires_at = EXCLUDED.expires_at
                "#,
                user_uuid,
                role_id,
                granted_by,
                expires_at
            )
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;

        // Clear cache for this user
        self.clear_user_permission_cache(&user_uuid).await;

        info!("Updated roles for user: {}", user_id);

        Ok(serde_json::json!({
            "message": "User roles updated successfully"
        }))
    }

    pub async fn list_roles(&self) -> Result<serde_json::Value> {
        let roles = sqlx::query!(
            r#"
            SELECT r.id, r.name, r.display_name, r.description, r.is_system,
                   r.is_active, r.created_at, r.updated_at,
                   COUNT(ur.user_id) as user_count
            FROM roles r
            LEFT JOIN user_roles ur ON r.id = ur.role_id AND ur.is_active = true
            WHERE r.is_active = true
            GROUP BY r.id, r.name, r.display_name, r.description, r.is_system,
                     r.is_active, r.created_at, r.updated_at
            ORDER BY r.created_at DESC
            "#
        )
        .fetch_all(&self.database)
        .await?;

        let role_list: Vec<serde_json::Value> = roles.into_iter().map(|r| {
            serde_json::json!({
                "id": r.id,
                "name": r.name,
                "display_name": r.display_name,
                "description": r.description,
                "is_system": r.is_system,
                "is_active": r.is_active,
                "user_count": r.user_count.unwrap_or(0),
                "created_at": r.created_at,
                "updated_at": r.updated_at
            })
        }).collect();

        Ok(serde_json::json!({
            "roles": role_list
        }))
    }

    pub async fn create_role(&self, payload: serde_json::Value) -> Result<serde_json::Value> {
        let request: CreateRoleRequest = serde_json::from_value(payload)?;
        request.validate()?;

        let role_id = sqlx::query_scalar!(
            r#"
            INSERT INTO roles (name, display_name, description)
            VALUES ($1, $2, $3)
            RETURNING id
            "#,
            request.name,
            request.display_name,
            request.description
        )
        .fetch_one(&self.database)
        .await?;

        // Assign permissions to the role
        for permission_id in request.permission_ids {
            sqlx::query!(
                r#"
                INSERT INTO role_permissions (role_id, permission_id)
                VALUES ($1, $2)
                "#,
                role_id,
                permission_id
            )
            .execute(&self.database)
            .await?;
        }

        info!("Created new role: {} ({})", request.name, role_id);

        Ok(serde_json::json!({
            "message": "Role created successfully",
            "role_id": role_id
        }))
    }

    pub async fn get_role(&self, role_id: &str) -> Result<serde_json::Value> {
        let role_uuid = Uuid::parse_str(role_id)?;

        let role = sqlx::query!(
            r#"
            SELECT id, name, display_name, description, is_system,
                   is_active, created_at, updated_at
            FROM roles
            WHERE id = $1
            "#,
            role_uuid
        )
        .fetch_optional(&self.database)
        .await?
        .ok_or_else(|| anyhow::anyhow!("Role not found"))?;

        let permissions = self.get_role_permissions(&role_uuid).await?;

        Ok(serde_json::json!({
            "id": role.id,
            "name": role.name,
            "display_name": role.display_name,
            "description": role.description,
            "is_system": role.is_system,
            "is_active": role.is_active,
            "created_at": role.created_at,
            "updated_at": role.updated_at,
            "permissions": permissions
        }))
    }

    pub async fn update_role(&self, role_id: &str, payload: serde_json::Value) -> Result<serde_json::Value> {
        let role_uuid = Uuid::parse_str(role_id)?;
        let request: UpdateRoleRequest = serde_json::from_value(payload)?;
        request.validate()?;

        // Check if role is system role
        let is_system = sqlx::query_scalar!(
            "SELECT is_system FROM roles WHERE id = $1",
            role_uuid
        )
        .fetch_optional(&self.database)
        .await?
        .ok_or_else(|| anyhow::anyhow!("Role not found"))?;

        if is_system {
            return Err(anyhow::anyhow!("Cannot modify system roles"));
        }

        // Update role details
        sqlx::query!(
            r#"
            UPDATE roles
            SET display_name = COALESCE($2, display_name),
                description = COALESCE($3, description),
                is_active = COALESCE($4, is_active),
                updated_at = NOW()
            WHERE id = $1
            "#,
            role_uuid,
            request.display_name,
            request.description,
            request.is_active
        )
        .execute(&self.database)
        .await?;

        // Update permissions if provided
        if let Some(permission_ids) = request.permission_ids {
            let mut tx = self.database.begin().await?;

            // Remove existing permissions
            sqlx::query!(
                "DELETE FROM role_permissions WHERE role_id = $1",
                role_uuid
            )
            .execute(&mut *tx)
            .await?;

            // Add new permissions
            for permission_id in permission_ids {
                sqlx::query!(
                    r#"
                    INSERT INTO role_permissions (role_id, permission_id)
                    VALUES ($1, $2)
                    "#,
                    role_uuid,
                    permission_id
                )
                .execute(&mut *tx)
                .await?;
            }

            tx.commit().await?;
        }

        // Clear caches
        self.clear_role_cache(&role_uuid).await;

        info!("Updated role: {}", role_id);

        Ok(serde_json::json!({
            "message": "Role updated successfully"
        }))
    }

    pub async fn delete_role(&self, role_id: &str) -> Result<serde_json::Value> {
        let role_uuid = Uuid::parse_str(role_id)?;

        // Check if role is system role
        let is_system = sqlx::query_scalar!(
            "SELECT is_system FROM roles WHERE id = $1",
            role_uuid
        )
        .fetch_optional(&self.database)
        .await?
        .ok_or_else(|| anyhow::anyhow!("Role not found"))?;

        if is_system {
            return Err(anyhow::anyhow!("Cannot delete system roles"));
        }

        // Check if role is assigned to any users
        let user_count = sqlx::query_scalar!(
            "SELECT COUNT(*) FROM user_roles WHERE role_id = $1 AND is_active = true",
            role_uuid
        )
        .fetch_one(&self.database)
        .await?;

        if user_count > 0 {
            return Err(anyhow::anyhow!("Cannot delete role that is assigned to users"));
        }

        // Soft delete by deactivating
        sqlx::query!(
            "UPDATE roles SET is_active = false, updated_at = NOW() WHERE id = $1",
            role_uuid
        )
        .execute(&self.database)
        .await?;

        info!("Deactivated role: {}", role_id);

        Ok(serde_json::json!({
            "message": "Role deactivated successfully"
        }))
    }

    pub async fn list_permissions(&self) -> Result<serde_json::Value> {
        let permissions = sqlx::query!(
            r#"
            SELECT id, name, display_name, description, resource, action,
                   conditions, is_system, created_at
            FROM permissions
            ORDER BY resource, action, name
            "#
        )
        .fetch_all(&self.database)
        .await?;

        let permission_list: Vec<serde_json::Value> = permissions.into_iter().map(|p| {
            serde_json::json!({
                "id": p.id,
                "name": p.name,
                "display_name": p.display_name,
                "description": p.description,
                "resource": p.resource,
                "action": p.action,
                "conditions": p.conditions,
                "is_system": p.is_system,
                "created_at": p.created_at
            })
        }).collect();

        Ok(serde_json::json!({
            "permissions": permission_list
        }))
    }

    async fn get_role_permissions(&self, role_id: &Uuid) -> Result<Vec<Permission>> {
        let permissions = sqlx::query_as!(
            Permission,
            r#"
            SELECT p.id, p.name, p.display_name, p.description, p.resource,
                   p.action, p.conditions, p.is_system, p.created_at
            FROM permissions p
            JOIN role_permissions rp ON p.id = rp.permission_id
            WHERE rp.role_id = $1
            ORDER BY p.resource, p.action, p.name
            "#,
            role_id
        )
        .fetch_all(&self.database)
        .await?;

        Ok(permissions)
    }

    pub async fn check_permission(&self, user_id: &Uuid, permission: &PermissionCheck) -> Result<bool> {
        let user_permissions = self.get_user_permissions(user_id).await?;

        for perm in &user_permissions {
            if perm.resource == permission.resource && perm.action == permission.action {
                // Check conditions if they exist
                if let Some(conditions) = &perm.conditions {
                    if self.evaluate_permission_conditions(conditions, permission.context.as_ref()).await {
                        return Ok(true);
                    }
                } else {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    pub async fn get_user_permissions(&self, user_id: &Uuid) -> Result<Vec<Permission>> {
        // Check cache first
        {
            let cache = self.permission_cache.read().await;
            if let Some(permissions) = cache.get(user_id) {
                return Ok(permissions.clone());
            }
        }

        // Fetch from database
        let permissions = sqlx::query_as!(
            Permission,
            r#"
            SELECT DISTINCT p.id, p.name, p.display_name, p.description,
                   p.resource, p.action, p.conditions, p.is_system, p.created_at
            FROM permissions p
            JOIN role_permissions rp ON p.id = rp.permission_id
            JOIN user_roles ur ON rp.role_id = ur.role_id
            WHERE ur.user_id = $1 AND ur.is_active = true
            AND (ur.expires_at IS NULL OR ur.expires_at > NOW())
            ORDER BY p.resource, p.action, p.name
            "#,
            user_id
        )
        .fetch_all(&self.database)
        .await?;

        // Update cache
        {
            let mut cache = self.permission_cache.write().await;
            cache.insert(*user_id, permissions.clone());
        }

        Ok(permissions)
    }

    async fn evaluate_permission_conditions(&self, conditions: &serde_json::Value, context: Option<&serde_json::Value>) -> bool {
        // Simple condition evaluation - in production, this would be more sophisticated
        if let Some(ctx) = context {
            if let Some(owner_id) = conditions.get("owner_only") {
                if owner_id.as_bool().unwrap_or(false) {
                    if let (Some(resource_owner), Some(current_user)) =
                        (ctx.get("owner_id"), ctx.get("user_id")) {
                        return resource_owner == current_user;
                    }
                }
            }
        }

        true // Default to allow if conditions can't be evaluated
    }

    async fn clear_user_permission_cache(&self, user_id: &Uuid) {
        let mut cache = self.permission_cache.write().await;
        cache.remove(user_id);
    }

    async fn clear_role_cache(&self, role_id: &Uuid) {
        let mut cache = self.role_cache.write().await;
        cache.remove(role_id);
    }

    pub async fn get_user_permissions_by_resource(&self, user_id: &Uuid, resource: &str) -> Result<Vec<String>> {
        let permissions = self.get_user_permissions(user_id).await?;

        let actions: Vec<String> = permissions
            .into_iter()
            .filter(|p| p.resource == resource)
            .map(|p| p.action)
            .collect();

        Ok(actions)
    }

    pub async fn bulk_check_permissions(&self, user_id: &Uuid, checks: &[PermissionCheck]) -> Result<HashMap<String, bool>> {
        let user_permissions = self.get_user_permissions(user_id).await?;
        let mut results = HashMap::new();

        for check in checks {
            let key = format!("{}:{}", check.resource, check.action);
            let has_permission = user_permissions.iter().any(|p| {
                p.resource == check.resource && p.action == check.action
            });
            results.insert(key, has_permission);
        }

        Ok(results)
    }
}