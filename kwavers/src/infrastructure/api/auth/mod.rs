//! Authentication and Authorization for PINN API
//!
//! Provides JWT-based authentication and role-based access control (RBAC)
//! for the enterprise PINN API.
//!
//! ## Security Features
//! - JWT token authentication with configurable expiration
//! - Role-based access control with granular permissions
//! - API key support for service-to-service authentication
//! - Token blacklisting and revocation

use crate::core::error::KwaversResult;
use chrono::{DateTime, Utc};
use jsonwebtoken::{Algorithm, DecodingKey, EncodingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

mod authenticate;
mod extractor;
#[cfg(test)]
mod tests;

/// JWT claims structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,
    pub iat: i64,
    pub exp: i64,
    pub jti: String,
    pub roles: Vec<String>,
    pub permissions: Vec<String>,
    pub iss: String,
}

/// Authenticated user information
#[derive(Debug, Clone)]
pub struct AuthenticatedUser {
    pub user_id: String,
    pub roles: Vec<String>,
    pub permissions: Vec<String>,
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
    pub key_id: String,
    pub user_id: String,
    pub name: String,
    pub key_hash: String,
    pub permissions: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
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

/// JWT configuration
#[derive(Debug, Clone)]
pub struct JWTConfig {
    pub issuer: String,
    pub expiration_seconds: u64,
    pub algorithm: Algorithm,
}

impl Default for JWTConfig {
    fn default() -> Self {
        Self {
            issuer: "kwavers-api".to_string(),
            expiration_seconds: 3600,
            algorithm: Algorithm::HS256,
        }
    }
}

pub struct AuthMiddleware {
    pub(super) jwt_encoding_key: EncodingKey,
    pub(super) jwt_decoding_key: DecodingKey,
    pub(super) jwt_config: JWTConfig,
    pub(super) permissions: HashMap<String, Vec<String>>,
}

impl std::fmt::Debug for AuthMiddleware {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthMiddleware")
            .field("jwt_config", &self.jwt_config)
            .field("permissions", &self.permissions)
            .finish_non_exhaustive()
    }
}

impl AuthMiddleware {
    /// Create new authentication middleware
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
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

    fn initialize_permissions() -> HashMap<String, Vec<String>> {
        let mut permissions = HashMap::new();

        permissions.insert(
            "pinn:train".to_string(),
            vec![
                "pinn:train:basic".to_string(),
                "pinn:train:advanced".to_string(),
                "pinn:train:gpu".to_string(),
            ],
        );

        permissions.insert(
            "pinn:infer".to_string(),
            vec![
                "pinn:infer:basic".to_string(),
                "pinn:infer:batch".to_string(),
            ],
        );

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

    pub(super) fn hash_api_key(api_key: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(api_key.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Validate API key format
    pub fn validate_api_key_format(api_key: &str) -> bool {
        !api_key.is_empty() && api_key.len() >= 20 && api_key.len() <= 128
    }

    /// Get available permissions
    pub fn get_permissions(&self) -> &HashMap<String, Vec<String>> {
        &self.permissions
    }
}

impl Default for AuthMiddleware {
    fn default() -> Self {
        // A default-constructed AuthMiddleware MUST NOT use a hardcoded secret.
        // Production code must use AuthMiddleware::new(secret, cfg).
        match std::env::var("KWAVERS_JWT_SECRET") {
            Ok(secret) => {
                let trimmed = secret.trim();
                if trimmed.is_empty() {
                    panic!(
                        "AuthMiddleware::default: KWAVERS_JWT_SECRET is set but empty/whitespace. \
Set KWAVERS_JWT_SECRET to a strong, non-empty secret."
                    );
                }
                if trimmed == "default-secret-change-in-production" {
                    panic!(
                        "AuthMiddleware::default: KWAVERS_JWT_SECRET is set to a known placeholder. \
Set KWAVERS_JWT_SECRET to a strong, unique secret."
                    );
                }
                Self::new(trimmed, JWTConfig::default()).unwrap_or_else(|e| {
                    panic!(
                        "AuthMiddleware::default: failed to construct from KWAVERS_JWT_SECRET: {e}"
                    )
                })
            }
            Err(_) => {
                panic!(
                    "AuthMiddleware::default: missing KWAVERS_JWT_SECRET. \
Default construction is intentionally forbidden without an explicit secret."
                );
            }
        }
    }
}
