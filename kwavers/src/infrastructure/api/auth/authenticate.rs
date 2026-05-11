//! JWT and API key authentication and authorization methods.

use super::{
    APIKey, APIKeyStatus, AuthMethod, AuthMiddleware, AuthenticatedUser, Claims, JWTConfig,
};
use crate::core::error::{KwaversError, KwaversResult};
use crate::infrastructure::api::{APIError, APIErrorType};
use chrono::{DateTime, Duration, Utc};
use jsonwebtoken::{decode, encode, Header, Validation};
use std::collections::HashMap;
use uuid::Uuid;

impl AuthMiddleware {
    /// Authenticate user from JWT token
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn authenticate_jwt(&self, token: &str) -> Result<AuthenticatedUser, APIError> {
        let validation = Validation::new(self.jwt_config.algorithm);
        let token_data =
            decode::<Claims>(token, &self.jwt_decoding_key, &validation).map_err(|e| {
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

        let now = Utc::now().timestamp();
        if claims.exp < now {
            return Err(APIError {
                error: APIErrorType::AuthenticationFailed,
                message: "Token has expired".to_string(),
                details: None,
            });
        }

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
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn authenticate_api_key(
        &self,
        api_key: &str,
        api_keys: &[APIKey],
    ) -> Result<AuthenticatedUser, APIError> {
        let key_hash = Self::hash_api_key(api_key);

        let key_info = api_keys
            .iter()
            .find(|k| k.key_hash == key_hash && matches!(k.status, APIKeyStatus::Active))
            .ok_or_else(|| APIError {
                error: APIErrorType::AuthenticationFailed,
                message: "Invalid API key".to_string(),
                details: None,
            })?;

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
            roles: vec![],
            permissions: key_info.permissions.clone(),
            auth_method: AuthMethod::APIKey,
        })
    }

    /// Authorize user action
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn authorize(
        &self,
        user: &AuthenticatedUser,
        action: &str,
        resource: &str,
    ) -> Result<(), APIError> {
        let required_permission = format!("{}:{}", action, resource);

        if user.permissions.contains(&required_permission) {
            return Ok(());
        }

        let action_prefix = action.split(':').next().unwrap_or(action);
        let wildcard_permission = format!("{}:*", action_prefix);

        if user.permissions.contains(&wildcard_permission) {
            return Ok(());
        }

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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
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
            KwaversError::System(crate::core::error::SystemError::InvalidConfiguration {
                parameter: "jwt_token_generation".to_string(),
                reason: format!("Failed to generate JWT token: {}", e),
            })
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
        let raw_key = Uuid::new_v4().to_string();
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
}
