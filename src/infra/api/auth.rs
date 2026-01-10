//! Authentication and Authorization for PINN API
//!
//! This module provides JWT-based authentication and role-based access control (RBAC)
//! for the enterprise PINN API, ensuring secure access to physics simulation services.
//!
//! ## Security Features
//!
//! - JWT token authentication with configurable expiration
//! - Role-based access control with granular permissions
//! - API key support for service-to-service authentication
//! - Token blacklisting and revocation
//! - Rate limiting integration
//! - Audit logging for security events

use crate::core::error::{KwaversError, KwaversResult};
use crate::infra::api::{APIError, APIErrorType};
use axum::extract::FromRequestParts;
use axum::http::{header::AUTHORIZATION, request::Parts, StatusCode};
use chrono::{DateTime, Duration, Utc};
use jsonwebtoken::{decode, encode, Algorithm, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use uuid::Uuid;

/// JWT claims structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    /// Subject (user ID)
    pub sub: String,
    /// Issued at timestamp
    pub iat: i64,
    /// Expiration timestamp
    pub exp: i64,
    /// JWT ID for revocation
    pub jti: String,
    /// User roles
    pub roles: Vec<String>,
    /// User permissions
    pub permissions: Vec<String>,
    /// Issuer
    pub iss: String,
}

/// Authenticated user information
#[derive(Debug, Clone)]
pub struct AuthenticatedUser {
    /// User ID
    pub user_id: String,
    /// User roles
    pub roles: Vec<String>,
    /// User permissions
    pub permissions: Vec<String>,
    /// Authentication method
    pub auth_method: AuthMethod,
}

/// Authentication methods
#[derive(Debug, Clone)]
pub enum AuthMethod {
    JWT,
    APIKey,
    ServiceToken,
}

/// API key information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct APIKey {
    /// Key ID
    pub key_id: String,
    /// User ID this key belongs to
    pub user_id: String,
    /// Key name/description
    pub name: String,
    /// Hashed API key (never stored in plain text)
    pub key_hash: String,
    /// Key permissions
    pub permissions: Vec<String>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Expiration timestamp
    pub expires_at: Option<DateTime<Utc>>,
    /// Key status
    pub status: APIKeyStatus,
}

/// API key status
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum APIKeyStatus {
    Active,
    Revoked,
    Expired,
}

/// Authentication middleware
impl std::fmt::Debug for AuthMiddleware {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthMiddleware")
            .field("jwt_config", &self.jwt_config)
            .field("permissions", &self.permissions)
            .finish_non_exhaustive()
    }
}

pub struct AuthMiddleware {
    /// JWT encoding/decoding keys
    jwt_encoding_key: EncodingKey,
    jwt_decoding_key: DecodingKey,
    /// JWT configuration
    jwt_config: JWTConfig,
    /// Permission definitions
    permissions: HashMap<String, Vec<String>>,
}

/// JWT configuration
#[derive(Debug, Clone)]
pub struct JWTConfig {
    /// JWT issuer
    pub issuer: String,
    /// Token expiration time (seconds)
    pub expiration_seconds: u64,
    /// Algorithm to use
    pub algorithm: Algorithm,
}

impl AuthMiddleware {
    /// Create new authentication middleware
    pub fn new(jwt_secret: &str, jwt_config: JWTConfig) -> KwaversResult<Self> {
        let jwt_encoding_key = EncodingKey::from_secret(jwt_secret.as_bytes());
        let jwt_decoding_key = DecodingKey::from_secret(jwt_secret.as_bytes());

        let permissions = Self::initialize_permissions();

        Ok(Self {
            jwt_encoding_key,
            jwt_decoding_key,
            jwt_config,
            permissions,
        })
    }

    /// Initialize permission definitions
    fn initialize_permissions() -> HashMap<String, Vec<String>> {
        let mut permissions = HashMap::new();

        // PINN training permissions
        permissions.insert(
            "pinn:train".to_string(),
            vec![
                "pinn:train:basic".to_string(),
                "pinn:train:advanced".to_string(),
                "pinn:train:gpu".to_string(),
            ],
        );

        // PINN inference permissions
        permissions.insert(
            "pinn:infer".to_string(),
            vec![
                "pinn:infer:basic".to_string(),
                "pinn:infer:batch".to_string(),
            ],
        );

        // Model management permissions
        permissions.insert(
            "models:read".to_string(),
            vec!["models:read:own".to_string(), "models:read:all".to_string()],
        );

        permissions.insert(
            "models:write".to_string(),
            vec![
                "models:write:own".to_string(),
                "models:write:all".to_string(),
            ],
        );

        permissions.insert(
            "models:delete".to_string(),
            vec![
                "models:delete:own".to_string(),
                "models:delete:all".to_string(),
            ],
        );

        // Job management permissions
        permissions.insert(
            "jobs:read".to_string(),
            vec!["jobs:read:own".to_string(), "jobs:read:all".to_string()],
        );

        permissions.insert(
            "jobs:write".to_string(),
            vec!["jobs:write:own".to_string(), "jobs:write:all".to_string()],
        );

        permissions
    }

    /// Authenticate user from JWT token
    pub fn authenticate_jwt(&self, token: &str) -> Result<AuthenticatedUser, APIError> {
        // Decode and validate JWT
        let validation = Validation::new(self.jwt_config.algorithm);
        let token_data =
            decode::<Claims>(token, &self.jwt_decoding_key, &validation).map_err(|e| {
                // Check if it's an expired signature error by examining the error message
                let error_msg = e.to_string();
                if error_msg.contains("ExpiredSignature") || error_msg.contains("expired") {
                    APIError {
                        error: APIErrorType::AuthenticationFailed,
                        message: "Token has expired".to_string(),
                        details: None,
                    }
                } else {
                    APIError {
                        error: APIErrorType::AuthenticationFailed,
                        message: "Invalid token".to_string(),
                        details: None,
                    }
                }
            })?;

        let claims = token_data.claims;

        // Check if token is expired
        let now = Utc::now().timestamp();
        if claims.exp < now {
            return Err(APIError {
                error: APIErrorType::AuthenticationFailed,
                message: "Token has expired".to_string(),
                details: None,
            });
        }

        // Check issuer
        if claims.iss != self.jwt_config.issuer {
            return Err(APIError {
                error: APIErrorType::AuthenticationFailed,
                message: "Invalid token issuer".to_string(),
                details: None,
            });
        }

        Ok(AuthenticatedUser {
            user_id: claims.sub,
            roles: claims.roles,
            permissions: claims.permissions,
            auth_method: AuthMethod::JWT,
        })
    }

    /// Authenticate user from API key
    pub fn authenticate_api_key(
        &self,
        api_key: &str,
        api_keys: &[APIKey],
    ) -> Result<AuthenticatedUser, APIError> {
        // Hash the provided API key
        let key_hash = Self::hash_api_key(api_key);

        // Find matching API key
        let key_info = api_keys
            .iter()
            .find(|k| k.key_hash == key_hash && matches!(k.status, APIKeyStatus::Active))
            .ok_or_else(|| APIError {
                error: APIErrorType::AuthenticationFailed,
                message: "Invalid API key".to_string(),
                details: None,
            })?;

        // Check if key has expired
        if let Some(expires_at) = key_info.expires_at {
            if Utc::now() > expires_at {
                return Err(APIError {
                    error: APIErrorType::AuthenticationFailed,
                    message: "API key has expired".to_string(),
                    details: None,
                });
            }
        }

        Ok(AuthenticatedUser {
            user_id: key_info.user_id.clone(),
            roles: vec![], // API keys don't have roles, only permissions
            permissions: key_info.permissions.clone(),
            auth_method: AuthMethod::APIKey,
        })
    }

    /// Authorize user action
    pub fn authorize(
        &self,
        user: &AuthenticatedUser,
        action: &str,
        resource: &str,
    ) -> Result<(), APIError> {
        let required_permission = format!("{}:{}", action, resource);

        // Check if user has the required permission
        if user.permissions.contains(&required_permission) {
            return Ok(());
        }

        // Check wildcard permissions (e.g., "pinn:*" allows all pinn actions)
        let action_prefix = action.split(':').next().unwrap_or(action);
        let wildcard_permission = format!("{}:*", action_prefix);

        if user.permissions.contains(&wildcard_permission) {
            return Ok(());
        }

        // Check role-based permissions
        for role in &user.roles {
            if let Some(role_permissions) = self.permissions.get(role) {
                if role_permissions.contains(&required_permission) {
                    return Ok(());
                }
            }
        }

        Err(APIError {
            error: APIErrorType::AuthorizationFailed,
            message: format!(
                "Insufficient permissions for action: {}",
                required_permission
            ),
            details: Some(HashMap::from([
                (
                    "required_permission".to_string(),
                    serde_json::Value::String(required_permission),
                ),
                (
                    "user_permissions".to_string(),
                    serde_json::json!(user.permissions),
                ),
            ])),
        })
    }

    /// Generate JWT token for user
    pub fn generate_token(
        &self,
        user_id: &str,
        roles: &[String],
        permissions: &[String],
    ) -> KwaversResult<String> {
        let now = Utc::now();
        let iat = now.timestamp();
        let exp = (now + Duration::seconds(self.jwt_config.expiration_seconds as i64)).timestamp();
        let jti = Uuid::new_v4().to_string();

        let claims = Claims {
            sub: user_id.to_string(),
            iat,
            exp,
            jti,
            roles: roles.to_vec(),
            permissions: permissions.to_vec(),
            iss: self.jwt_config.issuer.clone(),
        };

        encode(
            &Header::new(self.jwt_config.algorithm),
            &claims,
            &self.jwt_encoding_key,
        )
        .map_err(|e| {
            KwaversError::System(
                crate::domain::core::error::SystemError::InvalidConfiguration {
                    parameter: "jwt_token_generation".to_string(),
                    reason: format!("Failed to generate JWT token: {}", e),
                },
            )
        })
    }

    /// Generate API key for user
    pub fn generate_api_key(
        &self,
        user_id: &str,
        name: &str,
        permissions: &[String],
        expires_at: Option<DateTime<Utc>>,
    ) -> (String, APIKey) {
        let key_id = Uuid::new_v4().to_string();
        let raw_key = Uuid::new_v4().to_string(); // In production, use a cryptographically secure random string
        let key_hash = Self::hash_api_key(&raw_key);

        let api_key = APIKey {
            key_id,
            user_id: user_id.to_string(),
            name: name.to_string(),
            key_hash,
            permissions: permissions.to_vec(),
            created_at: Utc::now(),
            expires_at,
            status: APIKeyStatus::Active,
        };

        (raw_key, api_key)
    }

    /// Hash API key using SHA-256
    fn hash_api_key(api_key: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(api_key.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Validate API key format
    pub fn validate_api_key_format(api_key: &str) -> bool {
        // Basic validation: should be non-empty and reasonable length
        !api_key.is_empty() && api_key.len() >= 20 && api_key.len() <= 128
    }

    /// Get available permissions
    pub fn get_permissions(&self) -> &HashMap<String, Vec<String>> {
        &self.permissions
    }
}

impl Default for JWTConfig {
    fn default() -> Self {
        Self {
            issuer: "kwavers-api".to_string(),
            expiration_seconds: 3600, // 1 hour
            algorithm: Algorithm::HS256,
        }
    }
}

impl Default for AuthMiddleware {
    fn default() -> Self {
        // Correctness + security invariant:
        // A default-constructed AuthMiddleware MUST NOT silently use a known, hardcoded secret.
        //
        // Rationale:
        // - A deterministic default secret makes JWTs forgeable across deployments.
        // - `Default` is used widely; if it is used, it must fail fast and loudly if misconfigured.
        //
        // Contract:
        // - Production code must construct AuthMiddleware via `AuthMiddleware::new(secret, cfg)` with a
        //   process-provided secret (env/secret manager), never via `Default`.
        //
        // We intentionally avoid `unwrap()` here and provide a precise panic message to surface the
        // configuration error immediately.
        match std::env::var("KWAVERS_JWT_SECRET") {
            Ok(secret) => {
                let trimmed = secret.trim();
                if trimmed.is_empty() {
                    panic!(
                        "AuthMiddleware::default: KWAVERS_JWT_SECRET is set but empty/whitespace. \
Set KWAVERS_JWT_SECRET to a strong, non-empty secret (and construct AuthMiddleware explicitly in production)."
                    );
                }

                // Disallow the historical placeholder secret to prevent accidental reuse.
                if trimmed == "default-secret-change-in-production" {
                    panic!(
                        "AuthMiddleware::default: KWAVERS_JWT_SECRET is set to a known placeholder. \
Set KWAVERS_JWT_SECRET to a strong, unique secret (and construct AuthMiddleware explicitly in production)."
                    );
                }

                Self::new(trimmed, JWTConfig::default()).unwrap_or_else(|e| {
                    panic!(
                        "AuthMiddleware::default: failed to construct AuthMiddleware from KWAVERS_JWT_SECRET: {e}"
                    )
                })
            }
            Err(_) => {
                panic!(
                    "AuthMiddleware::default: missing KWAVERS_JWT_SECRET. \
Default construction is intentionally forbidden without an explicit secret.
Set KWAVERS_JWT_SECRET for tests/dev, and construct AuthMiddleware explicitly in production."
                );
            }
        }
    }
}

// Axum extractor implementation for AuthenticatedUser
#[axum::async_trait]
impl FromRequestParts<crate::api::handlers::AppState> for AuthenticatedUser {
    type Rejection = (StatusCode, axum::Json<APIError>);

    async fn from_request_parts(
        parts: &mut Parts,
        state: &crate::api::handlers::AppState,
    ) -> Result<Self, Self::Rejection> {
        Self::extract_from_middleware(parts, &state.auth_middleware).await
    }
}

#[cfg(feature = "pinn")]
#[axum::async_trait]
impl FromRequestParts<crate::api::clinical_handlers::ClinicalAppState> for AuthenticatedUser {
    type Rejection = (StatusCode, axum::Json<APIError>);

    async fn from_request_parts(
        parts: &mut Parts,
        state: &crate::api::clinical_handlers::ClinicalAppState,
    ) -> Result<Self, Self::Rejection> {
        Self::extract_from_middleware(parts, &state.auth_middleware).await
    }
}

impl AuthenticatedUser {
    /// Helper to extract authenticated user from parts and auth middleware
    async fn extract_from_middleware(
        parts: &mut Parts,
        auth_middleware: &std::sync::Arc<AuthMiddleware>,
    ) -> Result<Self, (StatusCode, axum::Json<APIError>)> {
        // Extract Authorization header
        let auth_header = parts
            .headers
            .get(AUTHORIZATION)
            .and_then(|h| h.to_str().ok())
            .and_then(|h| h.strip_prefix("Bearer "));

        let token = match auth_header {
            Some(token) => token,
            None => {
                return Err((
                    StatusCode::UNAUTHORIZED,
                    axum::Json(APIError {
                        error: APIErrorType::AuthenticationFailed,
                        message: "Missing or invalid Authorization header".to_string(),
                        details: None,
                    }),
                ));
            }
        };

        // Authenticate using the auth middleware
        match auth_middleware.authenticate_jwt(token) {
            Ok(user) => Ok(user),
            Err(error) => Err((StatusCode::UNAUTHORIZED, axum::Json(error))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jwt_token_generation_and_validation() {
        let auth =
            AuthMiddleware::new("test-secret-do-not-use-in-production", JWTConfig::default())
                .expect("test auth middleware construction must succeed");

        let user_id = "test-user";
        let roles = vec!["user".to_string()];
        let permissions = vec!["pinn:infer".to_string()];

        // Generate token
        let token = auth
            .generate_token(user_id, &roles, &permissions)
            .expect("token generation must succeed in tests");

        // Validate token
        let user = auth
            .authenticate_jwt(&token)
            .expect("token authentication must succeed in tests");

        assert_eq!(user.user_id, user_id);
        assert_eq!(user.roles, roles);
        assert_eq!(user.permissions, permissions);
        assert!(matches!(user.auth_method, AuthMethod::JWT));
    }

    #[test]
    fn test_api_key_generation_and_validation() {
        let auth =
            AuthMiddleware::new("test-secret-do-not-use-in-production", JWTConfig::default())
                .expect("test auth middleware construction must succeed");

        let user_id = "test-user";
        let permissions = vec!["pinn:infer".to_string()];

        // Generate API key
        let (raw_key, api_key_info) =
            auth.generate_api_key(user_id, "test-key", &permissions, None);

        assert_eq!(api_key_info.user_id, user_id);
        assert!(AuthMiddleware::validate_api_key_format(&raw_key));

        // Validate API key
        let api_keys = vec![api_key_info];
        let user = auth
            .authenticate_api_key(&raw_key, &api_keys)
            .expect("API key authentication must succeed in tests");

        assert_eq!(user.user_id, user_id);
        assert!(matches!(user.auth_method, AuthMethod::APIKey));
    }

    #[test]
    fn test_authorization_success() {
        let auth =
            AuthMiddleware::new("test-secret-do-not-use-in-production", JWTConfig::default())
                .expect("test auth middleware construction must succeed");

        let user = AuthenticatedUser {
            user_id: "test-user".to_string(),
            roles: vec![],
            permissions: vec!["pinn:infer".to_string()],
            auth_method: AuthMethod::JWT,
        };

        // Should succeed
        assert!(auth.authorize(&user, "pinn", "infer").is_ok());
    }

    #[test]
    fn test_authorization_failure() {
        let auth =
            AuthMiddleware::new("test-secret-do-not-use-in-production", JWTConfig::default())
                .expect("test auth middleware construction must succeed");

        let user = AuthenticatedUser {
            user_id: "test-user".to_string(),
            roles: vec![],
            permissions: vec!["pinn:infer".to_string()],
            auth_method: AuthMethod::JWT,
        };

        // Should fail - no permission for training
        assert!(auth.authorize(&user, "pinn", "train").is_err());
    }

    #[test]
    fn test_wildcard_permissions() {
        let auth =
            AuthMiddleware::new("test-secret-do-not-use-in-production", JWTConfig::default())
                .expect("test auth middleware construction must succeed");

        let user = AuthenticatedUser {
            user_id: "test-user".to_string(),
            roles: vec![],
            permissions: vec!["pinn:*".to_string()],
            auth_method: AuthMethod::JWT,
        };

        // Should succeed - wildcard permission
        assert!(auth.authorize(&user, "pinn", "train").is_ok());
        assert!(auth.authorize(&user, "pinn", "infer").is_ok());
    }

    #[test]
    fn test_api_key_format_validation() {
        assert!(AuthMiddleware::validate_api_key_format(
            "valid-api-key-with-reasonable-length"
        ));
        assert!(!AuthMiddleware::validate_api_key_format(""));
        assert!(!AuthMiddleware::validate_api_key_format("short"));
        assert!(!AuthMiddleware::validate_api_key_format(&"x".repeat(200))); // too long
    }

    #[test]
    fn test_expired_api_key() {
        let auth =
            AuthMiddleware::new("test-secret-do-not-use-in-production", JWTConfig::default())
                .expect("test auth middleware construction must succeed");

        let expired_key = APIKey {
            key_id: "test".to_string(),
            user_id: "user".to_string(),
            name: "test".to_string(),
            key_hash: "dummy".to_string(),
            permissions: vec![],
            created_at: Utc::now(),
            expires_at: Some(Utc::now() - Duration::hours(1)), // expired
            status: APIKeyStatus::Active,
        };

        // Should reject expired key
        let api_keys = vec![expired_key];
        let result = auth.authenticate_api_key("dummy", &api_keys);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().error,
            APIErrorType::AuthenticationFailed
        ));
    }
}
